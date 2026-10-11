// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/smooth.dispatch.cpp, modules/imgproc/src/smooth.simd.hpp, modules/imgproc/src/filter.simd.hpp, modules/imgproc/src/fixedpoint.inl.hpp.
// Licence of the sources: Apache-2.0; BSD-3-Clause; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.GaussianBlur` with OpenCV's exact numerics.
//!
//! 8-bit images take OpenCV's bit-exact fixed-point path: the kernel is
//! `getGaussianKernelBitExact` quantised to 8 fractional bits by error
//! diffusion (`getGaussianKernelFixedPoint_ED`), the horizontal pass sums
//! `tap * pixel` in 16 bits, the vertical pass in 32 bits and rounds once
//! (`(sum + 2^15) >> 16`). Float images take OpenCV's `sepFilter2D` engine:
//! sequential fused multiply-adds along the row, centre tap then symmetric
//! pairs down the column, as the AVX2/FMA kernels compute it.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

pub const Border = util.Border;

/// Kernel sizes and sigmas as `cv2.GaussianBlur(src, (kx, ky), sigma_x, sigma_y)`
/// takes them: a size of 0 is derived from the sigma, a sigma of 0 from the size.
pub const Params = struct {
    kx: usize = 0,
    ky: usize = 0,
    sigma_x: f64 = 0,
    sigma_y: f64 = 0,
    border: Border = .reflect101,
};

/// `cv::getGaussianKernelBitExact`: the normalised taps, in double.
pub fn kernelBitExact(a: Allocator, n: usize, sigma: f64) Allocator.Error![]f64 {
    const out = try a.alloc(f64, n);
    if (sigma <= 0) {
        const table: ?[]const f64 = switch (n) {
            1 => &.{1.0},
            3 => &.{ 0.25, 0.5, 0.25 },
            5 => &.{ 0.0625, 0.25, 0.375, 0.25, 0.0625 },
            7 => &.{ 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 },
            9 => &.{ 4.0 / 256.0, 13.0 / 256.0, 30.0 / 256.0, 51.0 / 256.0, 60.0 / 256.0, 51.0 / 256.0, 30.0 / 256.0, 13.0 / 256.0, 4.0 / 256.0 },
            else => null,
        };
        if (table) |t| {
            @memcpy(out, t);
            return out;
        }
    }
    const sigma_x: f64 = if (sigma > 0) sigma else @mulAdd(f64, @floatFromInt(n), 0.15, 0.35);
    const scale2x = -0.125 / (sigma_x * sigma_x);
    const n2 = (n - 1) / 2;
    var sum: f64 = 0;
    var x: i64 = 1 - @as(i64, @intCast(n));
    for (0..n2) |i| {
        const t = @exp(@as(f64, @floatFromInt(x * x)) * scale2x);
        out[i] = t;
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if (n % 2 == 0) sum += 1.0;
    const mul1 = 1.0 / sum;
    for (0..n2) |i| {
        const t = out[i] * mul1;
        out[i] = t;
        out[n - 1 - i] = t;
    }
    out[n2] = 1.0 * mul1;
    if (n % 2 == 0) out[n2 + 1] = out[n2];
    return out;
}

/// `cv::getGaussianKernelFixedPoint_ED` with 8 fractional bits: taps summing
/// to exactly 256, the outer ones rounded with error diffusion.
pub fn fixedTaps(a: Allocator, n: usize, sigma: f64) Allocator.Error![]u16 {
    const k = try kernelBitExact(a, n, sigma);
    defer a.free(k);
    const out = try a.alloc(u16, n);
    const n2 = n / 2;
    var err: f64 = 0;
    var sum: i64 = 0;
    for (0..n2) |i| {
        const adj = k[i] * 256.0 + err;
        const v0 = util.cvRound(adj);
        err = adj - @as(f64, @floatFromInt(v0));
        out[i] = @intCast(v0);
        out[n - 1 - i] = @intCast(v0);
        sum += v0;
    }
    out[n2] = @intCast(256 - 2 * sum);
    return out;
}

/// Float taps as `getGaussianKernel(n, sigma, CV_32F)` returns them.
pub fn floatTaps(a: Allocator, n: usize, sigma: f64) Allocator.Error![]f32 {
    const k = try kernelBitExact(a, n, sigma);
    defer a.free(k);
    const out = try a.alloc(f32, n);
    for (out, k) |*o, v| o.* = @floatCast(v);
    return out;
}

/// Kernel size OpenCV derives from a sigma: `cvRound(sigma * (3 or 4) * 2 + 1) | 1`.
pub fn ksizeFromSigma(sigma: f64, depth8: bool) usize {
    const f: f64 = if (depth8) 3.0 else 4.0;
    return @as(usize, @intCast(util.cvRound(sigma * f * 2.0 + 1.0))) | 1;
}

const Resolved = struct { kx: usize, ky: usize, sx: f64, sy: f64, same: bool };

fn resolve(p: Params, w: usize, h: usize, depth8: bool) Resolved {
    var sy = p.sigma_y;
    if (sy <= 0) sy = p.sigma_x;
    var kx = p.kx;
    var ky = p.ky;
    if (kx == 0 and p.sigma_x > 0) kx = ksizeFromSigma(p.sigma_x, depth8);
    if (ky == 0 and sy > 0) ky = ksizeFromSigma(sy, depth8);
    if (w == 1) kx = 1;
    if (h == 1) ky = 1;
    std.debug.assert(kx % 2 == 1 and ky % 2 == 1);
    const sx = @max(p.sigma_x, 0.0);
    sy = @max(sy, 0.0);
    return .{ .kx = kx, .ky = ky, .sx = sx, .sy = sy, .same = kx == ky and @abs(sx - sy) < std.math.floatEps(f64) };
}

// -- 8-bit fixed-point engine ----------------------------------------------------

const lanes = 16;
const V16 = @Vector(lanes, u16);
const V32 = @Vector(lanes, u32);
const V8 = @Vector(lanes, u8);

/// Source pixel of every column of a border-extended row: `tab[j]` is the
/// pixel read for extended column `j`, `j - r` in `[-r, w + r)`, as
/// `FilterEngine` tabulates its border once per filter (`borderTab`).
fn borderTable(a: Allocator, w: usize, r: usize, border: Border) Allocator.Error![]usize {
    const tab = try a.alloc(usize, w + 2 * r);
    for (tab, 0..) |*t, j| t.* = util.borderIndex(@as(isize, @intCast(j)) - @as(isize, @intCast(r)), w, border);
    return tab;
}

/// One row of `w` pixels with `r` border pixels on each side (`tab` from
/// `borderTable`) into `ext`, `(w + 2 r) * ch` samples: the row itself in the
/// middle, the border pixels gathered around it.
fn extendRow(comptime T: type, src: []const T, w: usize, ch: usize, tab: []const usize, ext: []T) void {
    const r = (tab.len - w) / 2;
    const rc = r * ch;
    @memcpy(ext[rc .. rc + w * ch], src[0 .. w * ch]);
    for (0..r) |j| {
        @memcpy(ext[j * ch .. (j + 1) * ch], src[tab[j] * ch .. (tab[j] + 1) * ch]);
        const k = r + w + j;
        @memcpy(ext[k * ch .. (k + 1) * ch], src[tab[k] * ch .. (tab[k] + 1) * ch]);
    }
}

/// Horizontal pass of one row: `dst[s] = sum(taps[i] * ext[s + i * ch])` in 16
/// bits, where `ext` is the row with its border filled in (`extendRow`), so
/// every sample takes the vector loop, kernels wider than the image included.
fn hrowU8(ext: []const u8, wc: usize, ch: usize, taps: []const u16, dst: []u16) void {
    const n = taps.len;
    const r = n / 2;
    if (n == 1) {
        for (dst[0..wc], ext[0..wc]) |*d, s| d.* = @as(u16, s) * taps[0];
        return;
    }
    const rc = r * ch;
    var s: usize = 0;
    while (s + lanes <= wc) : (s += lanes) {
        var acc: V16 = @as(V16, @intCast(@as(V8, ext[s + rc ..][0..lanes].*))) *% @as(V16, @splat(taps[r]));
        for (0..r) |i| {
            const av: V16 = @intCast(@as(V8, ext[s + i * ch ..][0..lanes].*));
            const bv: V16 = @intCast(@as(V8, ext[s + (n - 1 - i) * ch ..][0..lanes].*));
            acc +%= (av +% bv) *% @as(V16, @splat(taps[i]));
        }
        dst[s..][0..lanes].* = acc;
    }
    while (s < wc) : (s += 1) {
        var acc: u32 = 0;
        for (taps, 0..) |t, i| acc += @as(u32, t) * ext[s + i * ch];
        dst[s] = @truncate(acc);
    }
}

/// Vertical pass: `buf` holds `n` consecutive row results of `wc` samples
/// starting at row `row0`; writes one output row.
fn vrowU8(buf: []const u16, wc: usize, row0: usize, taps: []const u16, dst: []u8) void {
    const n = taps.len;
    const r = n / 2;
    const half: V32 = @splat(1 << 15);
    const max: V32 = @splat(255);
    const shift: @Vector(lanes, u5) = @splat(16);
    var s: usize = 0;
    while (s + lanes <= wc) : (s += lanes) {
        var acc: V32 = @as(V32, @intCast(@as(V16, buf[(row0 + r) * wc + s ..][0..lanes].*))) * @as(V32, @splat(taps[r]));
        for (0..r) |j| {
            const av: V32 = @intCast(@as(V16, buf[(row0 + j) * wc + s ..][0..lanes].*));
            const bv: V32 = @intCast(@as(V16, buf[(row0 + n - 1 - j) * wc + s ..][0..lanes].*));
            acc += (av + bv) * @as(V32, @splat(taps[j]));
        }
        acc = @min((acc + half) >> shift, max);
        dst[s..][0..lanes].* = @as(V8, @intCast(acc));
    }
    while (s < wc) : (s += 1) {
        var acc: u32 = 0;
        for (taps, 0..) |t, j| acc += @as(u32, t) * buf[(row0 + j) * wc + s];
        dst[s] = @intCast(@min((acc + (1 << 15)) >> 16, 255));
    }
}

fn chunkRows(ry: usize) usize {
    return @max(32, 4 * ry);
}

const JobU8 = struct {
    src: []const u8,
    w: usize,
    h: usize,
    ch: usize,
    tx: []const u16,
    ty: []const u16,
    tab: []const usize,
    border: Border,
    out: []u8,
    chunk: usize,
    buf_rows: usize,
    bufs: []u16,
    exts: []u8,

    fn run(self: *const JobU8, band: usize, y0: usize, y1: usize) void {
        const wc = self.w * self.ch;
        const ry = self.ty.len / 2;
        const buf = self.bufs[band * self.buf_rows * wc ..][0 .. self.buf_rows * wc];
        const ext_len = self.tab.len * self.ch;
        const ext = self.exts[band * ext_len ..][0..ext_len];
        var yc = y0;
        while (yc < y1) : (yc += self.chunk) {
            const ye = @min(yc + self.chunk, y1);
            const rows = ye - yc + 2 * ry;
            for (0..rows) |k| {
                const sy = util.borderIndex(@as(isize, @intCast(yc + k)) - @as(isize, @intCast(ry)), self.h, self.border);
                extendRow(u8, self.src[sy * wc .. (sy + 1) * wc], self.w, self.ch, self.tab, ext);
                hrowU8(ext, wc, self.ch, self.tx, buf[k * wc .. (k + 1) * wc]);
            }
            for (yc..ye) |y| vrowU8(buf, wc, y - yc, self.ty, self.out[y * wc .. (y + 1) * wc]);
        }
    }
};

/// `cv2.GaussianBlur` on an 8-bit image with `ch` interleaved channels (bit-exact).
pub fn blurU8(a: Allocator, src: []const u8, w: usize, h: usize, ch: usize, p: Params) Allocator.Error![]u8 {
    const res = resolve(p, w, h, true);
    const out = try a.alloc(u8, w * h * ch);
    errdefer a.free(out);
    if (res.kx == 1 and res.ky == 1) {
        @memcpy(out, src[0 .. w * h * ch]);
        return out;
    }
    const tx = try fixedTaps(a, res.kx, res.sx);
    defer a.free(tx);
    const ty = if (res.same) tx else try fixedTaps(a, res.ky, res.sy);
    defer if (!res.same) a.free(ty);
    const ry = ty.len / 2;
    const chunk = chunkRows(ry);
    const bands = par.bandCountWork(h, chunk, w * ch * (tx.len + ty.len));
    const buf_rows = chunk + 2 * ry;
    const bufs = try a.alloc(u16, bands * buf_rows * w * ch);
    defer a.free(bufs);
    const tab = try borderTable(a, w, tx.len / 2, p.border);
    defer a.free(tab);
    const exts = try a.alloc(u8, bands * tab.len * ch);
    defer a.free(exts);
    const job = JobU8{ .src = src, .w = w, .h = h, .ch = ch, .tx = tx, .ty = ty, .tab = tab, .border = p.border, .out = out, .chunk = chunk, .buf_rows = buf_rows, .bufs = bufs, .exts = exts };
    par.forBandsWork(h, chunk, w * ch * (tx.len + ty.len), &job, JobU8.run);
    return out;
}

// -- float engine -----------------------------------------------------------------
//
// OpenCV's `sepFilter2D` float path (filter.simd.hpp, dispatched for AVX2): the
// vector kernels fuse multiply-adds, the scalar tail of each row (the samples
// past the last full 8-lane block) does not. The row filter is the symmetric
// `SymmRowSmallFilter` for 3 and 5 taps, sequential taps otherwise; the column
// filter is `SymmColumnSmallFilter` for 3 taps, centre then symmetric pairs
// otherwise.

const flanes = 8;
const VF = @Vector(flanes, f32);

inline fn loadF(comptime In: type, src: []const In, i: usize) VF {
    if (In == f32) return src[i..][0..flanes].*;
    return @floatFromInt(@as(@Vector(flanes, u8), src[i..][0..flanes].*));
}

inline fn scalarF(comptime In: type, v: In) f32 {
    return if (In == f32) v else @floatFromInt(v);
}

inline fn mulAddOr(comptime T: type, comptime fused: bool, a: T, b: T, c: T) T {
    return if (fused) @mulAdd(T, a, b, c) else a * b + c;
}

/// The row filter on one output sample given its `n` input taps `v[0..n]`
/// (left to right) and the kernel.
inline fn rowFormula(comptime T: type, comptime fused: bool, v: []const T, taps: []const f32, comptime splat: fn (f32) T) T {
    const n = taps.len;
    if (n == 3) {
        return mulAddOr(T, fused, v[1], splat(taps[1]), (v[0] + v[2]) * splat(taps[0]));
    } else if (n == 5) {
        const inner = mulAddOr(T, fused, v[2], splat(taps[2]), (v[1] + v[3]) * splat(taps[1]));
        return mulAddOr(T, fused, v[0] + v[4], splat(taps[0]), inner);
    } else {
        var acc = v[0] * splat(taps[0]);
        for (1..n) |i| acc = mulAddOr(T, fused, v[i], splat(taps[i]), acc);
        return acc;
    }
}

/// The column filter: `v[r]` is the centre row's sample, `v[r - k]`, `v[r + k]` the pairs.
inline fn colFormula(comptime T: type, comptime fused: bool, v: []const T, taps: []const f32, comptime splat: fn (f32) T) T {
    const n = taps.len;
    const r = n / 2;
    if (n == 3) {
        return if (fused) @mulAdd(T, v[0] + v[2], splat(taps[0]), v[1] * splat(taps[1])) else (v[0] + v[2]) * splat(taps[0]) + v[1] * splat(taps[1]);
    }
    var acc = v[r] * splat(taps[r]);
    for (1..r + 1) |k| acc = mulAddOr(T, fused, v[r - k] + v[r + k], splat(taps[r + k]), acc);
    return acc;
}

fn splatF(x: f32) f32 {
    return x;
}

fn splatV(x: f32) VF {
    return @splat(x);
}

const max_taps = 1024;

/// Row pass of one row: `ext` is the row with its border filled in
/// (`extendRow`); samples `[0, W - W % 8)` as the vector kernels compute them
/// (8 lanes, fused multiply-adds), the rest as the scalar tail.
fn hrowF32(comptime In: type, ext: []const In, wc: usize, ch: usize, taps: []const f32, dst: []f32) void {
    const n = taps.len;
    std.debug.assert(n <= max_taps);
    if (n == 1) {
        for (dst[0..wc], ext[0..wc]) |*d, s| d.* = scalarF(In, s) * taps[0];
        return;
    }
    const vec_end = wc - wc % flanes;
    var vbuf: [max_taps]VF = undefined;
    var s: usize = 0;
    while (s < vec_end) : (s += flanes) {
        for (0..n) |i| vbuf[i] = loadF(In, ext, s + i * ch);
        dst[s..][0..flanes].* = rowFormula(VF, true, vbuf[0..n], taps, splatV);
    }
    var sbuf: [max_taps]f32 = undefined;
    while (s < wc) : (s += 1) {
        for (0..n) |i| sbuf[i] = scalarF(In, ext[s + i * ch]);
        dst[s] = rowFormula(f32, false, sbuf[0..n], taps, splatF);
    }
}

/// Column pass: `buf` holds `n` consecutive row results of `wc` samples
/// starting at row `row0`; writes one output row.
fn vrowF32(comptime Out: type, buf: []const f32, wc: usize, row0: usize, taps: []const f32, dst: []Out) void {
    const n = taps.len;
    const vec_end = wc - wc % flanes;
    var vbuf: [max_taps]VF = undefined;
    var s: usize = 0;
    while (s < vec_end) : (s += flanes) {
        for (0..n) |j| vbuf[j] = buf[(row0 + j) * wc + s ..][0..flanes].*;
        const acc = colFormula(VF, true, vbuf[0..n], taps, splatV);
        if (Out == f32) {
            dst[s..][0..flanes].* = acc;
        } else {
            const rounded = util.roundEvenVec(flanes, acc);
            const clamped = @min(@max(rounded, @as(VF, @splat(0))), @as(VF, @splat(255)));
            dst[s..][0..flanes].* = @as(@Vector(flanes, u8), @intFromFloat(clamped));
        }
    }
    var sbuf: [max_taps]f32 = undefined;
    while (s < wc) : (s += 1) {
        for (0..n) |j| sbuf[j] = buf[(row0 + j) * wc + s];
        const acc = colFormula(f32, false, sbuf[0..n], taps, splatF);
        dst[s] = if (Out == f32) acc else util.saturateU8F(acc);
    }
}

fn JobF32(comptime In: type, comptime Out: type) type {
    return struct {
        src: []const In,
        w: usize,
        h: usize,
        ch: usize,
        tx: []const f32,
        ty: []const f32,
        tab: []const usize,
        border: Border,
        out: []Out,
        chunk: usize,
        buf_rows: usize,
        bufs: []f32,
        exts: []In,

        fn run(self: *const @This(), band: usize, y0: usize, y1: usize) void {
            const wc = self.w * self.ch;
            const ry = self.ty.len / 2;
            const buf = self.bufs[band * self.buf_rows * wc ..][0 .. self.buf_rows * wc];
            const ext_len = self.tab.len * self.ch;
            const ext = self.exts[band * ext_len ..][0..ext_len];
            var yc = y0;
            while (yc < y1) : (yc += self.chunk) {
                const ye = @min(yc + self.chunk, y1);
                const rows = ye - yc + 2 * ry;
                for (0..rows) |k| {
                    const sy = util.borderIndex(@as(isize, @intCast(yc + k)) - @as(isize, @intCast(ry)), self.h, self.border);
                    extendRow(In, self.src[sy * wc .. (sy + 1) * wc], self.w, self.ch, self.tab, ext);
                    hrowF32(In, ext, wc, self.ch, self.tx, buf[k * wc .. (k + 1) * wc]);
                }
                for (yc..ye) |y| vrowF32(Out, buf, wc, y - yc, self.ty, self.out[y * wc .. (y + 1) * wc]);
            }
        }
    };
}

fn blurFloatEngine(comptime In: type, comptime Out: type, a: Allocator, src: []const In, w: usize, h: usize, ch: usize, p: Params) Allocator.Error![]Out {
    const res = resolve(p, w, h, false);
    const out = try a.alloc(Out, w * h * ch);
    errdefer a.free(out);
    const tx = try floatTaps(a, res.kx, res.sx);
    defer a.free(tx);
    const ty = if (res.same) tx else try floatTaps(a, res.ky, res.sy);
    defer if (!res.same) a.free(ty);
    const ry = ty.len / 2;
    const chunk = chunkRows(ry);
    const bands = par.bandCountWork(h, chunk, w * ch * (tx.len + ty.len));
    const buf_rows = chunk + 2 * ry;
    const bufs = try a.alloc(f32, bands * buf_rows * w * ch);
    defer a.free(bufs);
    const tab = try borderTable(a, w, tx.len / 2, p.border);
    defer a.free(tab);
    const exts = try a.alloc(In, bands * tab.len * ch);
    defer a.free(exts);
    const Job = JobF32(In, Out);
    const job = Job{ .src = src, .w = w, .h = h, .ch = ch, .tx = tx, .ty = ty, .tab = tab, .border = p.border, .out = out, .chunk = chunk, .buf_rows = buf_rows, .bufs = bufs, .exts = exts };
    par.forBandsWork(h, chunk, w * ch * (tx.len + ty.len), &job, Job.run);
    return out;
}

/// `cv2.GaussianBlur` on a float32 image (OpenCV's float filter engine).
pub fn blurF32(a: Allocator, src: []const f32, w: usize, h: usize, ch: usize, p: Params) Allocator.Error![]f32 {
    return blurFloatEngine(f32, f32, a, src, w, h, ch, p);
}

/// The float-engine blur of an 8-bit image converted back to 8 bits, as
/// `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C)` computes its mean
/// (`GaussianBlur` on the float copy, then `convertTo(CV_8U)`).
pub fn blurU8ViaFloat(a: Allocator, src: []const u8, w: usize, h: usize, ch: usize, p: Params) Allocator.Error![]u8 {
    return blurFloatEngine(u8, u8, a, src, w, h, ch, p);
}

// -- tests -------------------------------------------------------------------------

test "bit-exact kernels" {
    const a = std.testing.allocator;
    const k5 = try fixedTaps(a, 5, 0);
    defer a.free(k5);
    try std.testing.expectEqualSlices(u16, &.{ 16, 64, 96, 64, 16 }, k5);
    const k9 = try fixedTaps(a, 9, 0);
    defer a.free(k9);
    try std.testing.expectEqualSlices(u16, &.{ 4, 13, 30, 51, 60, 51, 30, 13, 4 }, k9);
    const k61 = try fixedTaps(a, 61, 0);
    defer a.free(k61);
    var sum: u32 = 0;
    for (k61) |t| sum += t;
    try std.testing.expectEqual(@as(u32, 256), sum);
    try std.testing.expectEqual(k61[0], k61[60]);
    const f = try floatTaps(a, 31, 0);
    defer a.free(f);
    var fs: f64 = 0;
    for (f) |t| fs += t;
    try std.testing.expect(@abs(fs - 1.0) < 1e-6);
    try std.testing.expectEqual(@as(usize, 13), ksizeFromSigma(2.0, true));
    try std.testing.expectEqual(@as(usize, 17), ksizeFromSigma(2.0, false));
}

fn bruteU8(a: Allocator, src: []const u8, w: usize, h: usize, ch: usize, tx: []const u16, ty: []const u16, border: Border) ![]u8 {
    const rx = tx.len / 2;
    const ry = ty.len / 2;
    const tmp = try a.alloc(u32, w * h * ch);
    defer a.free(tmp);
    for (0..h) |y| for (0..w) |x| for (0..ch) |c| {
        var acc: u32 = 0;
        for (tx, 0..) |t, i| acc += t * src[(y * w + util.borderIndex(@as(isize, @intCast(x + i)) - @as(isize, @intCast(rx)), w, border)) * ch + c];
        tmp[(y * w + x) * ch + c] = acc;
    };
    const out = try a.alloc(u8, w * h * ch);
    for (0..h) |y| for (0..w) |x| for (0..ch) |c| {
        var acc: u32 = 0;
        for (ty, 0..) |t, j| acc += t * tmp[(util.borderIndex(@as(isize, @intCast(y + j)) - @as(isize, @intCast(ry)), h, border) * w + x) * ch + c];
        out[(y * w + x) * ch + c] = @intCast(@min((acc + (1 << 15)) >> 16, 255));
    };
    return out;
}

test "fixed-point blur matches the brute force sum" {
    const a = std.testing.allocator;
    var seed: u32 = 99;
    const w = 53;
    const h = 41;
    for ([_]usize{ 1, 3 }) |ch| {
        const src = try a.alloc(u8, w * h * ch);
        defer a.free(src);
        for (src) |*v| {
            seed = seed *% 1103515245 +% 12345;
            v.* = @truncate(seed >> 16);
        }
        for ([_]Params{ .{ .kx = 5, .ky = 5 }, .{ .kx = 3, .ky = 9 }, .{ .kx = 31, .ky = 31, .border = .replicate }, .{ .sigma_x = 2.0 }, .{ .kx = 61, .ky = 61 } }) |p| {
            const got = try blurU8(a, src, w, h, ch, p);
            defer a.free(got);
            const res = resolve(p, w, h, true);
            const tx = try fixedTaps(a, res.kx, res.sx);
            defer a.free(tx);
            const ty = try fixedTaps(a, res.ky, res.sy);
            defer a.free(ty);
            const want = try bruteU8(a, src, w, h, ch, tx, ty, p.border);
            defer a.free(want);
            try std.testing.expectEqualSlices(u8, want, got);
        }
    }
}

/// The float engine one sample at a time: every tap through `borderIndex`,
/// the fused formulas for the samples of the vector region and the plain
/// ones for the tail, as the kernels define them.
fn bruteF32(a: Allocator, src: []const f32, w: usize, h: usize, ch: usize, tx: []const f32, ty: []const f32, border: Border) ![]f32 {
    const rx = tx.len / 2;
    const ry = ty.len / 2;
    const wc = w * ch;
    const vec_end = wc - wc % flanes;
    const tmp = try a.alloc(f32, wc * h);
    defer a.free(tmp);
    var sbuf: [max_taps]f32 = undefined;
    for (0..h) |y| for (0..wc) |s| {
        for (0..tx.len) |i| sbuf[i] = src[(y * w + util.borderIndex(@as(isize, @intCast(s / ch + i)) - @as(isize, @intCast(rx)), w, border)) * ch + s % ch];
        tmp[y * wc + s] = if (s < vec_end) rowFormula(f32, true, sbuf[0..tx.len], tx, splatF) else rowFormula(f32, false, sbuf[0..tx.len], tx, splatF);
    };
    const out = try a.alloc(f32, wc * h);
    for (0..h) |y| for (0..wc) |s| {
        for (0..ty.len) |j| sbuf[j] = tmp[util.borderIndex(@as(isize, @intCast(y + j)) - @as(isize, @intCast(ry)), h, border) * wc + s];
        out[y * wc + s] = if (s < vec_end) colFormula(f32, true, sbuf[0..ty.len], ty, splatF) else colFormula(f32, false, sbuf[0..ty.len], ty, splatF);
    };
    return out;
}

test "kernels wider than the image match the per-sample formulas" {
    const a = std.testing.allocator;
    var seed: u32 = 7;
    const w = 23;
    const h = 17;
    for ([_]usize{ 1, 3 }) |ch| {
        const src = try a.alloc(u8, w * h * ch);
        defer a.free(src);
        const srcf = try a.alloc(f32, w * h * ch);
        defer a.free(srcf);
        for (src, srcf) |*v, *f| {
            seed = seed *% 1103515245 +% 12345;
            v.* = @truncate(seed >> 16);
            f.* = @floatFromInt(v.*);
        }
        for ([_]Params{ .{ .kx = 61, .ky = 61 }, .{ .sigma_x = 9.0, .border = .replicate }, .{ .kx = 5, .ky = 61 }, .{ .kx = 61, .ky = 3 } }) |p| {
            const got = try blurU8(a, src, w, h, ch, p);
            defer a.free(got);
            const res = resolve(p, w, h, true);
            const tx = try fixedTaps(a, res.kx, res.sx);
            defer a.free(tx);
            const ty = try fixedTaps(a, res.ky, res.sy);
            defer a.free(ty);
            const want = try bruteU8(a, src, w, h, ch, tx, ty, p.border);
            defer a.free(want);
            try std.testing.expectEqualSlices(u8, want, got);
            const gotf = try blurF32(a, srcf, w, h, ch, p);
            defer a.free(gotf);
            const resf = resolve(p, w, h, false);
            const fx = try floatTaps(a, resf.kx, resf.sx);
            defer a.free(fx);
            const fy = try floatTaps(a, resf.ky, resf.sy);
            defer a.free(fy);
            const wantf = try bruteF32(a, srcf, w, h, ch, fx, fy, p.border);
            defer a.free(wantf);
            try std.testing.expectEqualSlices(f32, wantf, gotf);
        }
    }
}

test "float blur keeps a flat image and rounds like adaptiveThreshold's mean" {
    const a = std.testing.allocator;
    const flat = try a.alloc(f32, 40 * 30 * 3);
    defer a.free(flat);
    @memset(flat, 7.0);
    const b = try blurF32(a, flat, 40, 30, 3, .{ .sigma_x = 1.5 });
    defer a.free(b);
    for (b) |v| try std.testing.expect(@abs(v - 7.0) < 1e-4);
    var step: [20]u8 = undefined;
    for (&step, 0..) |*v, i| v.* = if (i < 10) 0 else 200;
    const rep = try blurU8ViaFloat(a, &step, 20, 1, 1, .{ .kx = 7, .ky = 7, .border = .replicate });
    defer a.free(rep);
    try std.testing.expectEqual(@as(u8, 0), rep[0]);
    try std.testing.expectEqual(@as(u8, 200), rep[19]);
    try std.testing.expect(rep[9] > 0 and rep[9] < 200);
    const fixed = try blurU8(a, &step, 20, 1, 1, .{ .kx = 7, .ky = 7, .border = .replicate });
    defer a.free(fixed);
    try std.testing.expectEqual(@as(u8, 200), fixed[19]);
}
