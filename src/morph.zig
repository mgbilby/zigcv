// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/morph.dispatch.cpp, modules/imgproc/src/morph.simd.hpp.
// Licence of the sources: BSD-3-Clause; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.erode`, `cv2.dilate` and `cv2.morphologyEx` with OpenCV's structuring
//! elements (`getStructuringElement`: `MORPH_RECT`, `MORPH_ELLIPSE`) and its
//! default border (pixels outside the image do not take part).
//!
//! A kernel is described by the half-width of each of its rows (every row of a
//! rectangle or of OpenCV's discrete ellipse is one contiguous run). The
//! operation is separated into horizontal running extremes, one per distinct
//! half-width, combined by vertical running extremes over the kernel rows of
//! that width; windows wider than 17 use logarithmic doubling, narrower ones a
//! direct pass. Every pass is a vector loop over row bands.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

pub const Shape = enum { rect, ellipse };
pub const Op = enum { erode, dilate };

/// A structuring element: `hw[i]` is the half-width of kernel row `i`
/// (`-r..=r` from the anchor), rows `hw.len`, odd.
pub const Kernel = struct {
    hw: []usize,

    pub fn radius(self: Kernel) usize {
        return self.hw.len / 2;
    }

    pub fn deinit(self: Kernel, a: Allocator) void {
        a.free(self.hw);
    }

    /// `cv2.getStructuringElement(shape, (kw, kh))` for odd sizes.
    pub fn init(a: Allocator, shape: Shape, kw: usize, kh: usize) Allocator.Error!Kernel {
        std.debug.assert(kw % 2 == 1 and kh % 2 == 1);
        const hw = try a.alloc(usize, kh);
        const c = kw / 2;
        const r = kh / 2;
        switch (shape) {
            .rect => @memset(hw, c),
            .ellipse => {
                const rf: f64 = @floatFromInt(r);
                const inv_r2: f64 = if (r > 0) 1.0 / (rf * rf) else 0.0;
                for (hw, 0..) |*v, i| {
                    const dy = @as(f64, @floatFromInt(i)) - rf;
                    // dx = saturate_cast<int>(c * sqrt((r*r - dy*dy) * inv_r2)), i.e. rounded.
                    const dx: usize = @intCast(util.cvRound(@as(f64, @floatFromInt(c)) * @sqrt(@max((rf * rf - dy * dy) * inv_r2, 0.0))));
                    v.* = @min(dx, c);
                }
            },
        }
        return .{ .hw = hw };
    }

    /// `cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))` / `np.ones((k, k))`.
    pub fn square(a: Allocator, k: usize) Allocator.Error!Kernel {
        return init(a, .rect, k | 1, k | 1);
    }

    /// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))`.
    pub fn ellipse(a: Allocator, k: usize) Allocator.Error!Kernel {
        return init(a, .ellipse, k | 1, k | 1);
    }
};

const lanes = 32;
const V = @Vector(lanes, u8);

inline fn ext(comptime max: bool, p: V, q: V) V {
    return if (max) @max(p, q) else @min(p, q);
}

inline fn extS(comptime max: bool, p: u8, q: u8) u8 {
    return if (max) @max(p, q) else @min(p, q);
}

fn neutral(comptime max: bool) u8 {
    return if (max) 0 else 255;
}

/// Elementwise extreme of two equally long slices into `dst` (any may alias `dst`).
fn extSlices(comptime max: bool, p: []const u8, q: []const u8, dst: []u8) void {
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        dst[i..][0..lanes].* = ext(max, p[i..][0..lanes].*, q[i..][0..lanes].*);
    }
    while (i < dst.len) : (i += 1) dst[i] = extS(max, p[i], q[i]);
}

/// `dst = extreme of rows[0..n]` with the accumulator kept in registers across
/// the rows (`MorphColumnVec`): each 64-byte block loads every row once and
/// stores once. `null` rows are outside the image and ignored.
fn colExtreme(comptime max: bool, rows: []const ?[*]const u8, w: usize, dst: []u8) void {
    const n = rows.len;
    var x: usize = 0;
    while (x + 2 * lanes <= w) : (x += 2 * lanes) {
        var a0: V = @splat(neutral(max));
        var a1: V = @splat(neutral(max));
        for (rows[0..n]) |row| {
            const r = row orelse continue;
            a0 = ext(max, a0, r[x..][0..lanes].*);
            a1 = ext(max, a1, r[x + lanes ..][0..lanes].*);
        }
        dst[x..][0..lanes].* = a0;
        dst[x + lanes ..][0..lanes].* = a1;
    }
    while (x + lanes <= w) : (x += lanes) {
        var a0: V = @splat(neutral(max));
        for (rows[0..n]) |row| {
            const r = row orelse continue;
            a0 = ext(max, a0, r[x..][0..lanes].*);
        }
        dst[x..][0..lanes].* = a0;
    }
    while (x < w) : (x += 1) {
        var a: u8 = neutral(max);
        for (rows[0..n]) |row| {
            const r = row orelse continue;
            a = extS(max, a, r[x]);
        }
        dst[x] = a;
    }
}

/// Two output rows of a uniform (rectangular) kernel at once: the extreme of
/// the shared middle rows `rows[1..n]` is taken once, then combined with
/// `rows[0]` for the first output and `rows[n]` for the second
/// (`MorphColumnVec`'s two-row loop).
fn colExtremePair(comptime max: bool, rows: []const ?[*]const u8, w: usize, dst0: []u8, dst1: []u8) void {
    const n = rows.len - 1;
    var x: usize = 0;
    while (x + 2 * lanes <= w) : (x += 2 * lanes) {
        var a0: V = @splat(neutral(max));
        var a1: V = @splat(neutral(max));
        for (rows[1..n]) |row| {
            const r = row orelse continue;
            a0 = ext(max, a0, r[x..][0..lanes].*);
            a1 = ext(max, a1, r[x + lanes ..][0..lanes].*);
        }
        const first = rows[0];
        const last = rows[n];
        dst0[x..][0..lanes].* = if (first) |r| ext(max, a0, r[x..][0..lanes].*) else a0;
        dst0[x + lanes ..][0..lanes].* = if (first) |r| ext(max, a1, r[x + lanes ..][0..lanes].*) else a1;
        dst1[x..][0..lanes].* = if (last) |r| ext(max, a0, r[x..][0..lanes].*) else a0;
        dst1[x + lanes ..][0..lanes].* = if (last) |r| ext(max, a1, r[x + lanes ..][0..lanes].*) else a1;
    }
    while (x < w) : (x += 1) {
        var a: u8 = neutral(max);
        for (rows[1..n]) |row| {
            const r = row orelse continue;
            a = extS(max, a, r[x]);
        }
        dst0[x] = if (rows[0]) |r| extS(max, a, r[x]) else a;
        dst1[x] = if (rows[n]) |r| extS(max, a, r[x]) else a;
    }
}

// -- horizontal pass -----------------------------------------------------------------

/// Horizontal running extreme of one row with half-width `hw` into `out`;
/// `s0` / `s1` are scratch rows of at least `w + 2 hw + lanes` bytes.
fn rowExtreme(comptime max: bool, row: []const u8, w: usize, hw: usize, out: []u8, s0: []u8, s1: []u8) void {
    const n = 2 * hw + 1;
    const pad_len = w + 2 * hw;
    // Padded copy: window of output x is p[x .. x + n).
    @memset(s0[0..hw], neutral(max));
    @memcpy(s0[hw .. hw + w], row);
    @memset(s0[hw + w .. s0.len], neutral(max));
    if (n <= 17) {
        var x: usize = 0;
        while (x + lanes <= w) : (x += lanes) {
            var acc: V = s0[x..][0..lanes].*;
            for (1..n) |d| acc = ext(max, acc, s0[x + d ..][0..lanes].*);
            out[x..][0..lanes].* = acc;
        }
        while (x < w) : (x += 1) {
            var acc = s0[x];
            for (1..n) |d| acc = extS(max, acc, s0[x + d]);
            out[x] = acc;
        }
    } else {
        // Doubling: cur[i] = extreme of p[i .. i + len).
        var cur = s0;
        var nxt = s1;
        var len: usize = 1;
        while (2 * len <= n) : (len *= 2) {
            extSlices(max, cur[0 .. pad_len - len], cur[len..pad_len], nxt[0 .. pad_len - len]);
            @memset(nxt[pad_len - len .. nxt.len], neutral(max));
            std.mem.swap([]u8, &cur, &nxt);
        }
        extSlices(max, cur[0..w], cur[n - len .. n - len + w], out);
    }
}

const RowJob = struct {
    src: []const u8,
    w: usize,
    hw: usize,
    dst: []u8,
    scratch: []u8, // bands * 2 * (w + 2 hw + lanes)
    scratch_len: usize,

    fn run(self: *const RowJob, comptime max: bool, band: usize, y0: usize, y1: usize) void {
        const w = self.w;
        const s0 = self.scratch[band * 2 * self.scratch_len ..][0..self.scratch_len];
        const s1 = self.scratch[band * 2 * self.scratch_len + self.scratch_len ..][0..self.scratch_len];
        for (y0..y1) |y| rowExtreme(max, self.src[y * w .. (y + 1) * w], w, self.hw, self.dst[y * w .. (y + 1) * w], s0, s1);
    }

    fn runMax(self: *const RowJob, band: usize, y0: usize, y1: usize) void {
        self.run(true, band, y0, y1);
    }
    fn runMin(self: *const RowJob, band: usize, y0: usize, y1: usize) void {
        self.run(false, band, y0, y1);
    }
};

/// Small kernels: one banded pass. Per chunk of output rows the horizontal
/// extremes of the needed source rows are built for every distinct non-zero
/// half-width (incrementally, narrow to wide) in a per-band buffer, then each
/// output row is the vertical extreme over the kernel rows of its plane
/// (kernel rows of half-width 0 read the source rows themselves).
const FusedJob = struct {
    src: []const u8,
    w: usize,
    h: usize,
    kernel: Kernel,
    widths: []const usize,
    kind: []const usize,
    out: []u8,
    chunk: usize,
    buf: []u8,
    buf_per_band: usize,
    scratch: []u8,
    scratch_len: usize,

    fn run(self: *const FusedJob, comptime max: bool, band: usize, y0: usize, y1: usize) void {
        const w = self.w;
        const kh = self.kernel.hw.len;
        const r = kh / 2;
        const nk = self.widths.len;
        const zero_kind = self.widths[0] == 0; // kind 0 is the source itself
        const first_plane: usize = if (zero_kind) 1 else 0;
        const rows_cap = self.chunk + 2 * r;
        const plane = rows_cap * w;
        const buf = self.buf[band * self.buf_per_band ..][0..self.buf_per_band];
        const s0 = self.scratch[band * 2 * self.scratch_len ..][0..self.scratch_len];
        const s1 = self.scratch[band * 2 * self.scratch_len + self.scratch_len ..][0..self.scratch_len];
        const uniform = nk == 1 and !zero_kind;
        var rows: [max_kernel_rows + 1]?[*]const u8 = undefined;
        var yc = y0;
        while (yc < y1) : (yc += self.chunk) {
            const ye = @min(yc + self.chunk, y1);
            const nrows = ye - yc + 2 * r;
            for (0..nrows) |j| {
                const sy = @as(isize, @intCast(yc + j)) - @as(isize, @intCast(r));
                if (sy < 0 or sy >= @as(isize, @intCast(self.h))) continue;
                var prev: []const u8 = self.src[@as(usize, @intCast(sy)) * w ..][0..w];
                var prev_hw: usize = 0;
                for (first_plane..nk) |k| {
                    const dst = buf[(k - first_plane) * plane + j * w ..][0..w];
                    rowExtreme(max, prev, w, self.widths[k] - prev_hw, dst, s0, s1);
                    prev = dst;
                    prev_hw = self.widths[k];
                }
            }
            // Row `i` of the kernel for output row `y`: plane row `y - yc + i`
            // (source row `y - r + i`), null outside the image.
            var y = yc;
            while (y < ye) {
                const j0 = y - yc;
                const pair = uniform and y + 1 < ye;
                const n = if (pair) kh + 1 else kh;
                for (0..n) |i| {
                    const sy = @as(isize, @intCast(y + i)) - @as(isize, @intCast(r));
                    if (sy < 0 or sy >= @as(isize, @intCast(self.h))) {
                        rows[i] = null;
                    } else if (self.kind[@min(i, kh - 1)] == 0 and zero_kind) {
                        rows[i] = self.src[@as(usize, @intCast(sy)) * w ..].ptr;
                    } else {
                        rows[i] = buf[(self.kind[@min(i, kh - 1)] - first_plane) * plane + (j0 + i) * w ..].ptr;
                    }
                }
                if (pair) {
                    colExtremePair(max, rows[0..n], w, self.out[y * w .. (y + 1) * w], self.out[(y + 1) * w .. (y + 2) * w]);
                    y += 2;
                } else {
                    colExtreme(max, rows[0..n], w, self.out[y * w .. (y + 1) * w]);
                    y += 1;
                }
            }
        }
    }

    fn runMax(self: *const FusedJob, band: usize, y0: usize, y1: usize) void {
        self.run(true, band, y0, y1);
    }
    fn runMin(self: *const FusedJob, band: usize, y0: usize, y1: usize) void {
        self.run(false, band, y0, y1);
    }
};

// -- vertical pass -------------------------------------------------------------------

/// `dst[y] = extreme of rows y + o .. y + o + m - 1` of `src` (outside rows ignored):
/// direct for small windows, doubling through `tmp_a` / `tmp_b` otherwise.
const ColJob = struct {
    src: []const u8,
    w: usize,
    h: usize,
    o: isize,
    m: usize,
    dst: []u8,

    fn rowOr(self: *const ColJob, buf: []const u8, y: isize) ?[]const u8 {
        if (y < 0 or y >= @as(isize, @intCast(self.h))) return null;
        const yu: usize = @intCast(y);
        return buf[yu * self.w .. (yu + 1) * self.w];
    }

    fn direct(self: *const ColJob, comptime max: bool, band: usize, y0: usize, y1: usize) void {
        _ = band;
        var rows: [max_kernel_rows]?[*]const u8 = undefined;
        for (y0..y1) |y| {
            for (0..self.m) |j| {
                const row = self.rowOr(self.src, @as(isize, @intCast(y)) + self.o + @as(isize, @intCast(j)));
                rows[j] = if (row) |r| r.ptr else null;
            }
            colExtreme(max, rows[0..self.m], self.w, self.dst[y * self.w .. (y + 1) * self.w]);
        }
    }

    fn directMax(self: *const ColJob, band: usize, y0: usize, y1: usize) void {
        self.direct(true, band, y0, y1);
    }
    fn directMin(self: *const ColJob, band: usize, y0: usize, y1: usize) void {
        self.direct(false, band, y0, y1);
    }
};

/// One doubling step over rows. Buffer row `b` stands for the image row
/// `b - off`, so windows that start above the image exist too:
/// `dst[b] = ext(src[b + o], src[b + o + shift])`, rows outside `src` ignored.
const StepJob = struct {
    src: []const u8,
    src_rows: usize,
    w: usize,
    o: isize,
    shift: usize,
    dst: []u8,

    fn step(self: *const StepJob, comptime max: bool, band: usize, y0: usize, y1: usize) void {
        _ = band;
        const hi: isize = @intCast(self.src_rows);
        for (y0..y1) |y| {
            const out = self.dst[y * self.w .. (y + 1) * self.w];
            const ya = @as(isize, @intCast(y)) + self.o;
            const yb = ya + @as(isize, @intCast(self.shift));
            const a_ok = ya >= 0 and ya < hi;
            const b_ok = yb >= 0 and yb < hi;
            if (a_ok and b_ok) {
                extSlices(max, self.src[@as(usize, @intCast(ya)) * self.w ..][0..self.w], self.src[@as(usize, @intCast(yb)) * self.w ..][0..self.w], out);
            } else if (a_ok) {
                @memcpy(out, self.src[@as(usize, @intCast(ya)) * self.w ..][0..self.w]);
            } else if (b_ok) {
                @memcpy(out, self.src[@as(usize, @intCast(yb)) * self.w ..][0..self.w]);
            } else @memset(out, neutral(max));
        }
    }

    fn stepMax(self: *const StepJob, band: usize, y0: usize, y1: usize) void {
        self.step(true, band, y0, y1);
    }
    fn stepMin(self: *const StepJob, band: usize, y0: usize, y1: usize) void {
        self.step(false, band, y0, y1);
    }
};

fn runStep(job: StepJob, rows: usize, max: bool) void {
    if (max) par.forBandsWork(rows, 16, job.w, &job, StepJob.stepMax) else par.forBandsWork(rows, 16, job.w, &job, StepJob.stepMin);
}

const Scratch = struct {
    a: []u8, // w * h: row-pass results
    b: []u8, // w * h
    c: []u8, // w * h: one vertical band result
    e: []u8, // w * (h + kh): doubling buffers
    f: []u8,
    row: []u8,
    row_len: usize,
};

fn rowPass(src: []const u8, w: usize, h: usize, hw: usize, max: bool, dst: []u8, sc: Scratch) void {
    const job = RowJob{ .src = src, .w = w, .hw = hw, .dst = dst, .scratch = sc.row, .scratch_len = sc.row_len };
    if (max) par.forBandsWork(h, 16, w, &job, RowJob.runMax) else par.forBandsWork(h, 16, w, &job, RowJob.runMin);
}

/// `dst[y] = extreme of src rows y + o .. y + o + m - 1`; direct for small
/// windows, doubling through `sc.e` / `sc.f` (which hold `h + m` rows) otherwise.
fn colPass(src: []const u8, w: usize, h: usize, o: isize, m: usize, max: bool, dst: []u8, sc: Scratch) void {
    if (m <= 17) {
        const job = ColJob{ .src = src, .w = w, .h = h, .o = o, .m = m, .dst = dst };
        if (max) par.forBandsWork(h, 16, w * m, &job, ColJob.directMax) else par.forBandsWork(h, 16, w * m, &job, ColJob.directMin);
        return;
    }
    // Buffer row b is image row b - m: windows may start up to m rows above the image.
    const off: isize = @intCast(m);
    const rows = h + m;
    const tmp = [2][]u8{ sc.e, sc.f };
    // First step from the image itself (shift 1 -> windows of 2 rows).
    runStep(.{ .src = src, .src_rows = h, .w = w, .o = -off, .shift = 1, .dst = tmp[0] }, rows, max);
    var cur: []const u8 = tmp[0];
    var which: usize = 1;
    var len: usize = 2;
    while (2 * len <= m) : (len *= 2) {
        runStep(.{ .src = cur, .src_rows = rows, .w = w, .o = 0, .shift = len, .dst = tmp[which] }, rows, max);
        cur = tmp[which];
        which = 1 - which;
    }
    runStep(.{ .src = cur, .src_rows = rows, .w = w, .o = o + off, .shift = m - len, .dst = dst }, h, max);
}

const max_widths = 512;

/// One erosion or dilation of a single-channel image.
fn applyOnce(src: []const u8, w: usize, h: usize, kernel: Kernel, max: bool, out: []u8, sc: Scratch) void {
    const r = kernel.radius();
    const kh = kernel.hw.len;
    // Distinct half-widths, ascending.
    var widths: [max_widths]usize = undefined;
    var nw: usize = 0;
    for (kernel.hw) |v| {
        var seen = false;
        for (widths[0..nw]) |x| if (x == v) {
            seen = true;
        };
        if (!seen) {
            std.debug.assert(nw < max_widths);
            widths[nw] = v;
            nw += 1;
        }
    }
    std.mem.sort(usize, widths[0..nw], {}, std.sort.asc(usize));
    // `cur` holds the horizontal running extreme of the current width, built
    // incrementally in `sc.a` / `sc.b`; the other of the two is free.
    var cur: []const u8 = src;
    var free_buf: []u8 = sc.a;
    var held_buf: []u8 = sc.b;
    var prev: usize = 0;
    var first = true;
    for (widths[0..nw]) |hw| {
        if (hw > prev) {
            rowPass(cur, w, h, hw - prev, max, free_buf, sc);
            cur = free_buf;
            std.mem.swap([]u8, &free_buf, &held_buf);
            prev = hw;
        }
        // Bands of kernel rows with exactly this half-width.
        var i: usize = 0;
        while (i < kh) {
            if (kernel.hw[i] != hw) {
                i += 1;
                continue;
            }
            var j = i;
            while (j < kh and kernel.hw[j] == hw) j += 1;
            const o = @as(isize, @intCast(i)) - @as(isize, @intCast(r));
            if (first and nw == 1 and j - i == kh) {
                colPass(cur, w, h, o, j - i, max, out, sc);
            } else {
                colPass(cur, w, h, o, j - i, max, sc.c, sc);
                if (first) @memcpy(out, sc.c) else if (max) extSlices(true, out, sc.c, out) else extSlices(false, out, sc.c, out);
            }
            first = false;
            i = j;
        }
    }
}

const Widths = struct {
    all: [max_widths]usize = undefined,
    n: usize = 0,
    kind: [max_widths]usize = undefined,

    fn init(kernel: Kernel) Widths {
        var wd = Widths{};
        for (kernel.hw) |v| {
            var seen = false;
            for (wd.all[0..wd.n]) |x| if (x == v) {
                seen = true;
            };
            if (!seen) {
                std.debug.assert(wd.n < max_widths);
                wd.all[wd.n] = v;
                wd.n += 1;
            }
        }
        std.mem.sort(usize, wd.all[0..wd.n], {}, std.sort.asc(usize));
        for (kernel.hw, 0..) |v, i| wd.kind[i] = std.mem.indexOfScalar(usize, wd.all[0..wd.n], v).?;
        return wd;
    }
};

/// Kernel rows the fused path handles (its column pass keeps that many row pointers).
const max_kernel_rows = 64;

/// Per-band buffer budget of the fused path.
const fused_budget = 16 << 20;

fn fusedOnce(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, max: bool, out: []u8, wd: *const Widths, hw_max: usize) Allocator.Error!bool {
    const kh = kernel.hw.len;
    const r = kh / 2;
    if (kh > max_kernel_rows) return false;
    // Long chunks: the rows re-done at a chunk boundary (2 r of them) stay a small share.
    const chunk = @max(64, 8 * r);
    const rows_cap = chunk + 2 * r;
    const planes = wd.n - @as(usize, if (wd.all[0] == 0) 1 else 0);
    const per_band = @max(planes, 1) * rows_cap * w;
    if (per_band > fused_budget) return false;
    const bands = par.bandCountWork(h, chunk, w * kh);
    const buf = try a.alloc(u8, bands * per_band);
    defer a.free(buf);
    const scratch_len = w + 2 * hw_max + lanes;
    const scratch = try a.alloc(u8, bands * 2 * scratch_len);
    defer a.free(scratch);
    const job = FusedJob{ .src = src, .w = w, .h = h, .kernel = kernel, .widths = wd.all[0..wd.n], .kind = wd.kind[0..kh], .out = out, .chunk = chunk, .buf = buf, .buf_per_band = per_band, .scratch = scratch, .scratch_len = scratch_len };
    if (max) par.forBandsWork(h, chunk, w * kh, &job, FusedJob.runMax) else par.forBandsWork(h, chunk, w * kh, &job, FusedJob.runMin);
    return true;
}

/// `cv2.erode` / `cv2.dilate` of a single-channel 8-bit image with `iterations`.
pub fn morph(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, op: Op, iterations: usize) Allocator.Error![]u8 {
    std.debug.assert(src.len == w * h and kernel.hw.len % 2 == 1);
    const out = try a.alloc(u8, w * h);
    errdefer a.free(out);
    if (iterations == 0) {
        @memcpy(out, src);
        return out;
    }
    var hw_max: usize = 0;
    for (kernel.hw) |v| hw_max = @max(hw_max, v);
    const max = op == .dilate;
    const wd = Widths.init(kernel);
    var input: []const u8 = src;
    var target: []u8 = out;
    var spare: ?[]u8 = null;
    defer if (spare) |sp| a.free(sp);
    var sc: ?Scratch = null;
    defer if (sc) |s| {
        a.free(s.a);
        a.free(s.b);
        a.free(s.c);
        a.free(s.e);
        a.free(s.f);
        a.free(s.row);
    };
    for (0..iterations) |it| {
        if (!try fusedOnce(a, input, w, h, kernel, max, target, &wd, hw_max)) {
            if (sc == null) {
                const kh = kernel.hw.len;
                const bands = par.bandCountWork(h, 16, w);
                const row_len = w + 2 * hw_max + lanes;
                sc = Scratch{
                    .a = try a.alloc(u8, w * h),
                    .b = try a.alloc(u8, w * h),
                    .c = try a.alloc(u8, w * h),
                    .e = try a.alloc(u8, w * (h + kh)),
                    .f = try a.alloc(u8, w * (h + kh)),
                    .row = try a.alloc(u8, bands * 2 * row_len),
                    .row_len = row_len,
                };
            }
            applyOnce(input, w, h, kernel, max, target, sc.?);
        }
        input = target;
        if (it + 1 < iterations) {
            if (spare == null) spare = try a.alloc(u8, w * h);
            target = if (target.ptr == out.ptr) spare.? else out;
        }
    }
    if (input.ptr != out.ptr) @memcpy(out, input);
    return out;
}

pub fn erode(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, iterations: usize) Allocator.Error![]u8 {
    return morph(a, src, w, h, kernel, .erode, iterations);
}

pub fn dilate(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, iterations: usize) Allocator.Error![]u8 {
    return morph(a, src, w, h, kernel, .dilate, iterations);
}

/// `cv2.morphologyEx(src, MORPH_OPEN, kernel, iterations)`: erode `iterations` times, then dilate as often.
pub fn open(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, iterations: usize) Allocator.Error![]u8 {
    const e = try erode(a, src, w, h, kernel, iterations);
    defer a.free(e);
    return dilate(a, e, w, h, kernel, iterations);
}

/// `cv2.morphologyEx(src, MORPH_CLOSE, kernel, iterations)`.
pub fn close(a: Allocator, src: []const u8, w: usize, h: usize, kernel: Kernel, iterations: usize) Allocator.Error![]u8 {
    const d = try dilate(a, src, w, h, kernel, iterations);
    defer a.free(d);
    return erode(a, d, w, h, kernel, iterations);
}

// -- tests ---------------------------------------------------------------------------

test "ellipse matches opencv shapes" {
    const a = std.testing.allocator;
    const e5 = try Kernel.ellipse(a, 5);
    defer e5.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2, 2, 0 }, e5.hw);
    const e9 = try Kernel.ellipse(a, 9);
    defer e9.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 3, 4, 4, 4, 3, 3, 0 }, e9.hw);
    const e15 = try Kernel.ellipse(a, 15);
    defer e15.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{ 0, 4, 5, 6, 6, 7, 7, 7, 7, 7, 6, 6, 5, 4, 0 }, e15.hw);
    const e27 = try Kernel.ellipse(a, 27);
    defer e27.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 7, 8, 9, 10, 11, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 11, 10, 9, 8, 7, 5, 0 }, e27.hw);
    const s3 = try Kernel.square(a, 3);
    defer s3.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, s3.hw);
    const wide = try Kernel.init(a, .rect, 7, 1);
    defer wide.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{3}, wide.hw);
}

fn brute(a: Allocator, src: []const u8, w: usize, h: usize, k: Kernel, max: bool) ![]u8 {
    const r: isize = @intCast(k.radius());
    const out = try a.alloc(u8, w * h);
    for (0..h) |yu| for (0..w) |xu| {
        var acc: ?u8 = null;
        var dy: isize = -r;
        while (dy <= r) : (dy += 1) {
            const hw: isize = @intCast(k.hw[@intCast(dy + r)]);
            var dx: isize = -hw;
            while (dx <= hw) : (dx += 1) {
                const sx = @as(isize, @intCast(xu)) + dx;
                const sy = @as(isize, @intCast(yu)) + dy;
                if (sx < 0 or sy < 0 or sx >= @as(isize, @intCast(w)) or sy >= @as(isize, @intCast(h))) continue;
                const v = src[@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))];
                acc = if (acc) |p| (if (max) @max(p, v) else @min(p, v)) else v;
            }
        }
        out[yu * w + xu] = acc orelse (if (max) 0 else 255);
    };
    return out;
}

test "erode and dilate match the brute force extreme" {
    const a = std.testing.allocator;
    const w = 83;
    const h = 47;
    var seed: u32 = 777;
    var gray: [w * h]u8 = undefined;
    var mask: [w * h]u8 = undefined;
    for (&gray, &mask) |*g, *m| {
        seed = seed *% 1103515245 +% 12345;
        g.* = @truncate(seed >> 16);
        m.* = if ((seed >> 16) & 3 == 0) 255 else 0;
    }
    const kernels = [_]Kernel{ try Kernel.square(a, 3), try Kernel.ellipse(a, 5), try Kernel.ellipse(a, 11), try Kernel.ellipse(a, 45), try Kernel.square(a, 15), try Kernel.square(a, 41), try Kernel.init(a, .rect, 21, 1), try Kernel.init(a, .rect, 1, 9), try Kernel.init(a, .ellipse, 15, 7) };
    defer for (kernels) |k| k.deinit(a);
    for (kernels) |k| {
        for ([_][]const u8{ &gray, &mask }) |src| {
            for ([_]bool{ true, false }) |max| {
                const got = try morph(a, src, w, h, k, if (max) .dilate else .erode, 1);
                defer a.free(got);
                const want = try brute(a, src, w, h, k, max);
                defer a.free(want);
                try std.testing.expectEqualSlices(u8, want, got);
            }
        }
    }
    // The whole-image pass path (large kernels) agrees with the fused one.
    for (kernels) |k| {
        var hw_max: usize = 0;
        for (k.hw) |v| hw_max = @max(hw_max, v);
        const wd = Widths.init(k);
        const fused = try a.alloc(u8, w * h);
        defer a.free(fused);
        try std.testing.expect(try fusedOnce(a, &gray, w, h, k, true, fused, &wd, hw_max));
        const kh = k.hw.len;
        const row_len = w + 2 * hw_max + lanes;
        const sc = Scratch{ .a = try a.alloc(u8, w * h), .b = try a.alloc(u8, w * h), .c = try a.alloc(u8, w * h), .e = try a.alloc(u8, w * (h + kh)), .f = try a.alloc(u8, w * (h + kh)), .row = try a.alloc(u8, par.bandCount(h, 16) * 2 * row_len), .row_len = row_len };
        defer {
            a.free(sc.a);
            a.free(sc.b);
            a.free(sc.c);
            a.free(sc.e);
            a.free(sc.f);
            a.free(sc.row);
        }
        const passes = try a.alloc(u8, w * h);
        defer a.free(passes);
        applyOnce(&gray, w, h, k, true, passes, sc);
        try std.testing.expectEqualSlices(u8, fused, passes);
    }
    // Two iterations are two applications; open is erode then dilate.
    const k5 = kernels[1];
    const twice = try dilate(a, &gray, w, h, k5, 2);
    defer a.free(twice);
    const once = try dilate(a, &gray, w, h, k5, 1);
    defer a.free(once);
    const again = try dilate(a, once, w, h, k5, 1);
    defer a.free(again);
    try std.testing.expectEqualSlices(u8, again, twice);
    const opened = try open(a, &mask, w, h, k5, 1);
    defer a.free(opened);
    const er = try erode(a, &mask, w, h, k5, 1);
    defer a.free(er);
    const od = try dilate(a, er, w, h, k5, 1);
    defer a.free(od);
    try std.testing.expectEqualSlices(u8, od, opened);
    const closed = try close(a, &mask, w, h, kernels[0], 1);
    defer a.free(closed);
    for (closed, mask) |c, m| try std.testing.expect(c >= m);
}
