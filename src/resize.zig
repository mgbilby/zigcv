// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/resize.cpp.
// Licence of the sources: BSD-3-Clause; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.resize` for 8-bit images, bit-exact with OpenCV: `INTER_NEAREST`,
//! `INTER_LINEAR` and `INTER_CUBIC` in OpenCV's 11-bit fixed point (with the
//! same border clamping and the "2x INTER_LINEAR is INTER_AREA" rule),
//! `INTER_AREA` with integer (block sum) and fractional (float coverage
//! tables) scales.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

pub const Interp = enum { nearest, linear, cubic, area };

const coef_bits = 11;
const coef_scale = 1 << coef_bits;

/// Size of `cv2.resize(img, None, fx=scale, fy=scale)`: `round(w * scale)`, at least 1.
pub fn scaledSize(w: usize, h: usize, fx: f64, fy: f64) [2]usize {
    return .{
        @max(@as(usize, @intCast(util.cvRound(@as(f64, @floatFromInt(w)) * fx))), 1),
        @max(@as(usize, @intCast(util.cvRound(@as(f64, @floatFromInt(h)) * fy))), 1),
    };
}

/// `cv2.resize(src, (dw, dh), interpolation)` on `ch` interleaved 8-bit channels.
pub fn resize(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, interp_in: Interp) Allocator.Error![]u8 {
    std.debug.assert(sw > 0 and sh > 0 and dw > 0 and dh > 0);
    if (dw == sw and dh == sh) return a.dupe(u8, src[0 .. sw * sh * ch]);
    var interp = interp_in;
    if (interp == .nearest) return nearest(a, src, sw, sh, ch, dw, dh);
    const scale_x = @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(dw));
    const scale_y = @as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(dh));
    const iscale_x = util.cvRound(scale_x);
    const iscale_y = util.cvRound(scale_y);
    const eps = std.math.floatEps(f64);
    const area_fast = @abs(scale_x - @as(f64, @floatFromInt(iscale_x))) < eps and @abs(scale_y - @as(f64, @floatFromInt(iscale_y))) < eps;
    if (interp == .linear and area_fast and iscale_x == 2 and iscale_y == 2) interp = .area;
    if (interp == .area and scale_x >= 1 and scale_y >= 1) {
        if (area_fast) return areaFast(a, src, sw, sh, ch, dw, dh, @intCast(iscale_x), @intCast(iscale_y));
        return areaFrac(a, src, sw, sh, ch, dw, dh);
    }
    return generic(a, src, sw, sh, ch, dw, dh, interp);
}

// -- nearest -----------------------------------------------------------------

const NearestJob = struct {
    src: []const u8,
    sw: usize,
    sh: usize,
    ch: usize,
    dw: usize,
    dh: usize,
    xofs: []const usize,
    out: []u8,

    fn run(self: *const NearestJob, band: usize, y0: usize, y1: usize) void {
        _ = band;
        const ify = 1.0 / (@as(f64, @floatFromInt(self.dh)) / @as(f64, @floatFromInt(self.sh)));
        const ch = self.ch;
        for (y0..y1) |y| {
            const sy = @min(@as(usize, @intCast(util.cvFloor(@as(f64, @floatFromInt(y)) * ify))), self.sh - 1);
            const srow = self.src[sy * self.sw * ch .. (sy + 1) * self.sw * ch];
            const drow = self.out[y * self.dw * ch .. (y + 1) * self.dw * ch];
            if (ch == 1) {
                for (drow, self.xofs) |*d, o| d.* = srow[o];
            } else if (ch == 3) {
                for (self.xofs, 0..) |o, x| drow[x * 3 ..][0..3].* = srow[o..][0..3].*;
            } else {
                for (self.xofs, 0..) |o, x| @memcpy(drow[x * ch .. (x + 1) * ch], srow[o .. o + ch]);
            }
        }
    }
};

fn nearest(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) Allocator.Error![]u8 {
    const out = try a.alloc(u8, dw * dh * ch);
    errdefer a.free(out);
    const ifx = 1.0 / (@as(f64, @floatFromInt(dw)) / @as(f64, @floatFromInt(sw)));
    const xofs = try a.alloc(usize, dw);
    defer a.free(xofs);
    for (xofs, 0..) |*o, x| o.* = @min(@as(usize, @intCast(util.cvFloor(@as(f64, @floatFromInt(x)) * ifx))), sw - 1) * ch;
    const job = NearestJob{ .src = src, .sw = sw, .sh = sh, .ch = ch, .dw = dw, .dh = dh, .xofs = xofs, .out = out };
    par.forBandsWork(dh, 32, dw * ch, &job, NearestJob.run);
    return out;
}

// -- area, integer scale --------------------------------------------------------

/// Sum of groups of `isx` samples `ch` apart, 16 output pixels at a time: for
/// channel `c` the lanes `p * isx * ch + i * ch + c`.
fn groupMask(comptime isx: usize, comptime ch: usize, comptime i: usize, comptime c: usize) @Vector(16, i32) {
    var m: [16]i32 = undefined;
    for (&m, 0..) |*v, pp| v.* = @intCast(pp * isx * ch + i * ch + c);
    return m;
}

fn interleaveMask(comptime ch: usize) @Vector(16 * ch, i32) {
    // Lanes of the concatenation [c0 (16), c1 (16), c2 (16)] in pixel order.
    var m: [16 * ch]i32 = undefined;
    for (&m, 0..) |*v, i| v.* = @intCast((i % ch) * 16 + i / ch);
    return m;
}

const AreaFastJob = struct {
    src: []const u8,
    sw: usize,
    ch: usize,
    dw: usize,
    isx: usize,
    isy: usize,
    out: []u8,
    rows: []u16, // bands * sw * ch

    fn run(self: *const AreaFastJob, band: usize, y0: usize, y1: usize) void {
        switch (self.ch) {
            1 => switch (self.isx) {
                2 => self.runT(2, 1, band, y0, y1),
                4 => self.runT(4, 1, band, y0, y1),
                else => self.runT(0, 1, band, y0, y1),
            },
            3 => switch (self.isx) {
                2 => self.runT(2, 3, band, y0, y1),
                4 => self.runT(4, 3, band, y0, y1),
                else => self.runT(0, 3, band, y0, y1),
            },
            else => self.runT(0, 0, band, y0, y1),
        }
    }

    /// `isx_c` / `ch_c` are the comptime specialisations (0: generic scalar).
    fn runT(self: *const AreaFastJob, comptime isx_c: usize, comptime ch_c: usize, band: usize, y0: usize, y1: usize) void {
        const wc = self.sw * self.ch;
        const acc = self.rows[band * wc .. (band + 1) * wc];
        const area = self.isx * self.isy;
        const scale: f32 = 1.0 / @as(f32, @floatFromInt(area));
        const two_by_two = self.isx == 2 and self.isy == 2 and (self.ch == 1 or self.ch == 3 or self.ch == 4);
        const dwc = self.dw * self.ch;
        for (y0..y1) |dy| {
            // Sum the isy source rows into 16-bit samples.
            const base = dy * self.isy * wc;
            @memset(acc, 0);
            for (0..self.isy) |k| {
                const row = self.src[base + k * wc ..][0..wc];
                var i: usize = 0;
                while (i + 16 <= wc) : (i += 16) {
                    const v: @Vector(16, u16) = @intCast(@as(@Vector(16, u8), row[i..][0..16].*));
                    acc[i..][0..16].* = @as(@Vector(16, u16), acc[i..][0..16].*) + v;
                }
                while (i < wc) : (i += 1) acc[i] += row[i];
            }
            const drow = self.out[dy * dwc .. (dy + 1) * dwc];
            var dx: usize = 0;
            if (isx_c != 0 and ch_c != 0) {
                const VS = @Vector(16, u16);
                const n_in = 16 * isx_c * ch_c;
                while (dx + 16 <= self.dw) : (dx += 16) {
                    const inp: @Vector(n_in, u16) = acc[dx * isx_c * ch_c ..][0..n_in].*;
                    var sums: [ch_c]VS = undefined;
                    inline for (0..ch_c) |c| {
                        var sum: VS = @shuffle(u16, inp, undefined, groupMask(isx_c, ch_c, 0, c));
                        inline for (1..isx_c) |i| sum += @shuffle(u16, inp, undefined, groupMask(isx_c, ch_c, i, c));
                        sums[c] = sum;
                    }
                    var bytes: [ch_c]@Vector(16, u8) = undefined;
                    inline for (0..ch_c) |c| {
                        if (two_by_two) {
                            bytes[c] = @intCast((sums[c] + @as(VS, @splat(2))) >> @splat(2));
                        } else {
                            const f: @Vector(16, f32) = @floatFromInt(sums[c]);
                            const r = @min(@max(util.roundEvenVec(16, f * @as(@Vector(16, f32), @splat(scale))), @as(@Vector(16, f32), @splat(0))), @as(@Vector(16, f32), @splat(255)));
                            bytes[c] = @intFromFloat(r);
                        }
                    }
                    if (ch_c == 1) {
                        drow[dx..][0..16].* = bytes[0];
                    } else {
                        const rg: @Vector(32, u8) = std.simd.join(bytes[0], bytes[1]);
                        const rgb: @Vector(48, u8) = std.simd.join(rg, bytes[2]);
                        drow[dx * 3 ..][0..48].* = @shuffle(u8, rgb, undefined, interleaveMask(3));
                    }
                }
            }
            while (dx < self.dw) : (dx += 1) {
                for (0..self.ch) |c| {
                    var sum: u32 = 0;
                    for (0..self.isx) |i| sum += acc[(dx * self.isx + i) * self.ch + c];
                    drow[dx * self.ch + c] = if (two_by_two) @intCast((sum + 2) >> 2) else util.saturateU8F(@as(f32, @floatFromInt(sum)) * scale);
                }
            }
        }
    }
};

fn areaFast(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, isx: usize, isy: usize) Allocator.Error![]u8 {
    _ = sh;
    std.debug.assert(isy <= 256);
    const out = try a.alloc(u8, dw * dh * ch);
    errdefer a.free(out);
    const bands = par.bandCountWork(dh, 16, sw * ch * isy);
    const rows = try a.alloc(u16, bands * sw * ch);
    defer a.free(rows);
    const job = AreaFastJob{ .src = src, .sw = sw, .ch = ch, .dw = dw, .isx = isx, .isy = isy, .out = out, .rows = rows };
    par.forBandsWork(dh, 16, sw * ch * isy, &job, AreaFastJob.run);
    return out;
}

// -- area, fractional scale -----------------------------------------------------

const DecimateAlpha = struct { di: usize, si: usize, alpha: f32 };

/// `computeResizeAreaTab`: per destination index the source indices it covers
/// and their float weights, in OpenCV's order.
fn areaTab(a: Allocator, ssize: usize, dsize: usize) Allocator.Error![]DecimateAlpha {
    var tab: std.ArrayList(DecimateAlpha) = .empty;
    errdefer tab.deinit(a);
    const scale = @as(f64, @floatFromInt(ssize)) / @as(f64, @floatFromInt(dsize));
    for (0..dsize) |dx| {
        const fsx1 = @as(f64, @floatFromInt(dx)) * scale;
        const fsx2 = fsx1 + scale;
        const cell = @min(scale, @as(f64, @floatFromInt(ssize)) - fsx1);
        var sx1 = util.cvCeil(fsx1);
        var sx2 = util.cvFloor(fsx2);
        sx2 = @min(sx2, @as(i64, @intCast(ssize)) - 1);
        sx1 = @min(sx1, sx2);
        if (@as(f64, @floatFromInt(sx1)) - fsx1 > 1e-3) {
            try tab.append(a, .{ .di = dx, .si = @intCast(sx1 - 1), .alpha = @floatCast((@as(f64, @floatFromInt(sx1)) - fsx1) / cell) });
        }
        var sx = sx1;
        while (sx < sx2) : (sx += 1) {
            try tab.append(a, .{ .di = dx, .si = @intCast(sx), .alpha = @floatCast(1.0 / cell) });
        }
        if (fsx2 - @as(f64, @floatFromInt(sx2)) > 1e-3) {
            try tab.append(a, .{ .di = dx, .si = @intCast(sx2), .alpha = @floatCast(@min(@min(fsx2 - @as(f64, @floatFromInt(sx2)), 1.0), cell) / cell) });
        }
    }
    return tab.toOwnedSlice(a);
}

const AreaFracJob = struct {
    src: []const u8,
    sw: usize,
    ch: usize,
    dw: usize,
    xtab: []const DecimateAlpha,
    xstart: []const usize, // first xtab entry of each destination column, plus the end
    ytab: []const DecimateAlpha,
    ystart: []const usize, // first ytab entry of each destination row, plus the end
    out: []u8,
    bufs: []f32,

    fn run(self: *const AreaFracJob, band: usize, y0: usize, y1: usize) void {
        switch (self.ch) {
            1 => self.runT(1, band, y0, y1),
            3 => self.runT(3, band, y0, y1),
            else => self.runT(0, band, y0, y1),
        }
    }

    fn runT(self: *const AreaFracJob, comptime ch_c: usize, band: usize, y0: usize, y1: usize) void {
        const ch = if (ch_c != 0) ch_c else self.ch;
        const dwc = self.dw * ch;
        const buf = self.bufs[band * 2 * dwc ..][0..dwc];
        const sum = self.bufs[band * 2 * dwc + dwc ..][0..dwc];
        for (y0..y1) |dy| {
            @memset(sum, 0);
            for (self.ytab[self.ystart[dy]..self.ystart[dy + 1]]) |yt| {
                const srow = self.src[yt.si * self.sw * ch ..];
                // Horizontal pass: each destination sample accumulates its taps in
                // table order (the same float additions as OpenCV's scatter loop).
                for (0..self.dw) |dx| {
                    var accs: [4]f32 = .{ 0, 0, 0, 0 };
                    for (self.xtab[self.xstart[dx]..self.xstart[dx + 1]]) |xt| {
                        const s0 = xt.si * ch;
                        inline for (0..4) |c| {
                            if (c < ch) accs[c] += @as(f32, @floatFromInt(srow[s0 + c])) * xt.alpha;
                        }
                    }
                    inline for (0..4) |c| {
                        if (c < ch) buf[dx * ch + c] = accs[c];
                    }
                }
                const beta: @Vector(8, f32) = @splat(yt.alpha);
                var i: usize = 0;
                while (i + 8 <= dwc) : (i += 8) {
                    sum[i..][0..8].* = @as(@Vector(8, f32), sum[i..][0..8].*) + beta * @as(@Vector(8, f32), buf[i..][0..8].*);
                }
                while (i < dwc) : (i += 1) sum[i] += yt.alpha * buf[i];
            }
            const drow = self.out[dy * dwc .. (dy + 1) * dwc];
            var i: usize = 0;
            while (i + 8 <= dwc) : (i += 8) {
                const v: @Vector(8, f32) = sum[i..][0..8].*;
                const r = @min(@max(util.roundEvenVec(8, v), @as(@Vector(8, f32), @splat(0))), @as(@Vector(8, f32), @splat(255)));
                drow[i..][0..8].* = @as(@Vector(8, u8), @intFromFloat(r));
            }
            while (i < dwc) : (i += 1) drow[i] = util.saturateU8F(sum[i]);
        }
    }
};

fn areaFrac(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) Allocator.Error![]u8 {
    const xtab = try areaTab(a, sw, dw);
    defer a.free(xtab);
    const ytab = try areaTab(a, sh, dh);
    defer a.free(ytab);
    const ystart = try a.alloc(usize, dh + 1);
    defer a.free(ystart);
    {
        var k: usize = 0;
        for (0..dh) |dy| {
            ystart[dy] = k;
            while (k < ytab.len and ytab[k].di == dy) k += 1;
        }
        ystart[dh] = ytab.len;
    }
    const xstart = try a.alloc(usize, dw + 1);
    defer a.free(xstart);
    {
        var k: usize = 0;
        for (0..dw) |dx| {
            xstart[dx] = k;
            while (k < xtab.len and xtab[k].di == dx) k += 1;
        }
        xstart[dw] = xtab.len;
    }
    const out = try a.alloc(u8, dw * dh * ch);
    errdefer a.free(out);
    const bands = par.bandCountWork(dh, 8, (sw * sh / @max(dh, 1) + dw) * ch);
    const bufs = try a.alloc(f32, bands * 2 * dw * ch);
    defer a.free(bufs);
    const job = AreaFracJob{ .src = src, .sw = sw, .ch = ch, .dw = dw, .xtab = xtab, .xstart = xstart, .ytab = ytab, .ystart = ystart, .out = out, .bufs = bufs };
    par.forBandsWork(dh, 8, (sw * sh / @max(dh, 1) + dw) * ch, &job, AreaFracJob.run);
    return out;
}

// -- linear / cubic, fixed point ------------------------------------------------

/// OpenCV's bicubic coefficients (A = -0.75) for a fraction `x` in [0, 1), in float.
pub fn cubicCoeffs(x: f32) [4]f32 {
    const A: f32 = -0.75;
    const c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    const c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    const c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    return .{ c0, c1, c2, 1.0 - c0 - c1 - c2 };
}

fn toShort(v: f32) i32 {
    return std.math.clamp(util.cvRoundF(v * coef_scale), -32768, 32767);
}

const Axis = struct {
    /// Byte offset of the first tap in a source row (`xofs` of `resize()`),
    /// `sx ch` for linear, `(sx - 1) ch` for cubic; on the y axis the row
    /// index of the first tap, which may fall outside and is clamped when read.
    ofs: []i64,
    /// `ksize` coefficients per destination index, tap-major: `alpha[k * dsize + d]`.
    alpha: []i32,
    xmin: usize,
    xmax: usize,

    fn deinit(self: Axis, a: Allocator) void {
        a.free(self.ofs);
        a.free(self.alpha);
    }
};

/// Coefficient tables as `resize()` builds them: along x the source index is
/// clamped and the fraction zeroed at the borders (linear only); along y the
/// index may fall outside and the rows are clamped when read.
fn axisTable(a: Allocator, ssize: usize, dsize: usize, interp: Interp, area_mode: bool, clamp_border: bool, step: usize) Allocator.Error!Axis {
    const ksize: usize = if (interp == .cubic) 4 else 2;
    const ksize2 = ksize / 2;
    const ofs = try a.alloc(i64, dsize);
    errdefer a.free(ofs);
    const alpha = try a.alloc(i32, dsize * ksize);
    const scale = @as(f64, @floatFromInt(ssize)) / @as(f64, @floatFromInt(dsize));
    const inv_scale = @as(f64, @floatFromInt(dsize)) / @as(f64, @floatFromInt(ssize));
    var xmin: usize = 0;
    var xmax: usize = dsize;
    for (0..dsize) |dx| {
        var fx: f32 = undefined;
        var sx: i64 = undefined;
        if (!area_mode) {
            fx = @floatCast((@as(f64, @floatFromInt(dx)) + 0.5) * scale - 0.5);
            sx = @intFromFloat(@floor(fx));
            fx -= @floatFromInt(sx);
        } else {
            sx = util.cvFloor(@as(f64, @floatFromInt(dx)) * scale);
            fx = @floatCast(@as(f64, @floatFromInt(dx + 1)) - @as(f64, @floatFromInt(sx + 1)) * inv_scale);
            fx = if (fx <= 0) 0 else fx - @floor(fx);
        }
        if (sx < @as(i64, @intCast(ksize2)) - 1) {
            xmin = dx + 1;
            if (sx < 0 and interp != .cubic and clamp_border) {
                fx = 0;
                sx = 0;
            }
        }
        if (sx + @as(i64, @intCast(ksize2)) >= @as(i64, @intCast(ssize))) {
            xmax = @min(xmax, dx);
            if (sx >= @as(i64, @intCast(ssize)) - 1 and interp != .cubic and clamp_border) {
                fx = 0;
                sx = @as(i64, @intCast(ssize)) - 1;
            }
        }
        ofs[dx] = (sx - @as(i64, @intCast(ksize2)) + 1) * @as(i64, @intCast(step));
        if (interp == .cubic) {
            const c = cubicCoeffs(fx);
            for (0..4) |k| alpha[k * dsize + dx] = toShort(c[k]);
        } else {
            alpha[dx] = toShort(1.0 - fx);
            alpha[dsize + dx] = toShort(fx);
        }
    }
    return .{ .ofs = ofs, .alpha = alpha, .xmin = xmin, .xmax = xmax };
}

const hlanes = 16;
const HV = @Vector(hlanes, i32);
const HU = @Vector(hlanes, u32);

/// Lanes of `[c0 (16), c1 (16), ...]` in sample order.
fn hInterleaveMask(comptime ch: usize) @Vector(hlanes * ch, i32) {
    var m: [hlanes * ch]i32 = undefined;
    for (&m, 0..) |*v, i| v.* = @intCast((i % ch) * hlanes + i / ch);
    return m;
}

/// Horizontal pass of one source row into `dst` (`dw ch` samples, 11-bit
/// fixed point): `HResizeLinear` / `HResizeCubic` with the per-pixel gathers of
/// `HResizeLinearVec_8u32s` applied to both: 16 destination pixels at a time,
/// the `ksize ch` bytes of each pixel's taps fetched as 32-bit words, split
/// into tap planes in registers and multiplied by the tap-major coefficient
/// vectors. Pixels whose taps leave the row (the `xmin` / `xmax` borders)
/// take the scalar formulas.
fn hresize(comptime ch: usize, comptime ksize: usize, srow: []const u8, sw: usize, dw: usize, xs: Axis, dst: []i32) void {
    @setEvalBranchQuota(100_000);
    const swc = sw * ch;
    const need = ksize * ch;
    const nwords = (need + 3) / 4;
    const lo = if (ksize == 4) xs.xmin else 0;
    const hi = xs.xmax;
    var dx: usize = lo;
    while (dx + hlanes <= hi) : (dx += hlanes) {
        var words: [nwords]HU = undefined;
        inline for (0..nwords) |s| {
            const n = comptime @min(4, need - s * 4);
            var v: HU = undefined;
            inline for (0..hlanes) |i| v[i] = util.readUpTo4(n, srow.ptr + @as(usize, @intCast(xs.ofs[dx + i])) + s * 4);
            words[s] = v;
        }
        var alpha: [ksize]HV = undefined;
        inline for (0..ksize) |k| alpha[k] = xs.alpha[k * dw + dx ..][0..hlanes].*;
        var planes: [ch]HV = undefined;
        inline for (0..ch) |c| {
            var acc: HV = @splat(0);
            inline for (0..ksize) |k| {
                const byte = k * ch + c;
                const sh: @Vector(hlanes, u5) = @splat(@intCast((byte % 4) * 8));
                const tap: HV = @intCast((words[byte / 4] >> sh) & @as(HU, @splat(0xFF)));
                acc += tap * alpha[k];
            }
            planes[c] = acc;
        }
        if (ch == 1) {
            dst[dx..][0..hlanes].* = planes[0];
        } else if (ch == 3) {
            const v: @Vector(3 * hlanes, i32) = std.simd.join(std.simd.join(planes[0], planes[1]), planes[2]);
            dst[dx * 3 ..][0 .. 3 * hlanes].* = @shuffle(i32, v, undefined, hInterleaveMask(3));
        } else {
            const v: @Vector(4 * hlanes, i32) = std.simd.join(std.simd.join(planes[0], planes[1]), std.simd.join(planes[2], planes[3]));
            dst[dx * 4 ..][0 .. 4 * hlanes].* = @shuffle(i32, v, undefined, hInterleaveMask(4));
        }
    }
    // The scalar formulas: the interior pixels the vector loop left and both borders.
    var x: usize = 0;
    while (x < dw) : (x += 1) {
        if (x == lo) x = dx;
        if (x >= dw) break;
        const sx0 = xs.ofs[x];
        if (ksize == 2) {
            const sx: usize = @intCast(sx0);
            const a0 = xs.alpha[x];
            const a1 = xs.alpha[dw + x];
            inline for (0..ch) |c| {
                const s0: i32 = srow[sx + c];
                dst[x * ch + c] = if (x < hi) s0 * a0 + @as(i32, srow[sx + ch + c]) * a1 else s0 * coef_scale;
            }
        } else {
            inline for (0..ch) |c| {
                var acc: i32 = 0;
                inline for (0..4) |j| {
                    var sxj = sx0 + @as(i64, j * ch + c);
                    while (sxj < 0) sxj += ch;
                    while (sxj >= @as(i64, @intCast(swc))) sxj -= ch;
                    acc += @as(i32, srow[@intCast(sxj)]) * xs.alpha[j * dw + x];
                }
                dst[x * ch + c] = acc;
            }
        }
    }
}

/// `VResizeLinear<uchar, int, short>` with `VResizeLinearVec_32s8u`: 16-bit
/// high products of the 4-bit-shifted row sums, `(h0 + h1 + 2) >> 2`.
fn vresizeLinear(s0: []const i32, s1: []const i32, b0: i32, b1: i32, dst: []u8) void {
    const n = dst.len;
    const V16 = @Vector(16, i16);
    const V32 = @Vector(16, i32);
    const vb0: V32 = @splat(b0);
    const vb1: V32 = @splat(b1);
    const four: @Vector(16, u5) = @splat(4);
    const sixteen: @Vector(16, u5) = @splat(16);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const p0: V16 = @truncate(@as(V32, s0[i..][0..16].*) >> four);
        const p1: V16 = @truncate(@as(V32, s1[i..][0..16].*) >> four);
        const h0: V16 = @truncate((@as(V32, p0) * vb0) >> sixteen);
        const h1: V16 = @truncate((@as(V32, p1) * vb1) >> sixteen);
        const sum = h0 + h1 + @as(V16, @splat(2));
        const r = @min(@max(sum >> @as(@Vector(16, u4), @splat(2)), @as(V16, @splat(0))), @as(V16, @splat(255)));
        dst[i..][0..16].* = @as(@Vector(16, u8), @intCast(r));
    }
    while (i < n) : (i += 1) {
        dst[i] = @intCast(((((b0 * (s0[i] >> 4)) >> 16) + ((b1 * (s1[i] >> 4)) >> 16)) + 2) >> 2);
    }
}

/// `VResizeCubic<uchar, int, short>` with `VResizeCubicVec_32s8u` (the
/// baseline SSE kernel, no FMA): float taps, innermost product first, in
/// 8-sample blocks; the samples past the last block use the integer cast.
fn vresizeCubic(rows: [4][]const i32, beta: []const i32, dst: []u8) void {
    const n = dst.len;
    const scale: f32 = 1.0 / @as(f32, @floatFromInt(coef_scale * coef_scale));
    const vec_end = n - n % 8;
    var i: usize = 0;
    inline for ([_]usize{ 16, 8 }) |lanes| {
        const V = @Vector(lanes, f32);
        const b0: V = @splat(@as(f32, @floatFromInt(beta[0])) * scale);
        const b1: V = @splat(@as(f32, @floatFromInt(beta[1])) * scale);
        const b2: V = @splat(@as(f32, @floatFromInt(beta[2])) * scale);
        const b3: V = @splat(@as(f32, @floatFromInt(beta[3])) * scale);
        while (i + lanes <= vec_end) : (i += lanes) {
            const s0: V = @floatFromInt(@as(@Vector(lanes, i32), rows[0][i..][0..lanes].*));
            const s1: V = @floatFromInt(@as(@Vector(lanes, i32), rows[1][i..][0..lanes].*));
            const s2: V = @floatFromInt(@as(@Vector(lanes, i32), rows[2][i..][0..lanes].*));
            const s3: V = @floatFromInt(@as(@Vector(lanes, i32), rows[3][i..][0..lanes].*));
            const v = s0 * b0 + (s1 * b1 + (s2 * b2 + s3 * b3));
            const r = @min(@max(util.roundEvenVec(lanes, v), @as(V, @splat(0))), @as(V, @splat(255)));
            dst[i..][0..lanes].* = @as(@Vector(lanes, u8), @intFromFloat(r));
        }
    }
    while (i < n) : (i += 1) {
        const sum: i64 = @as(i64, rows[0][i]) * beta[0] + @as(i64, rows[1][i]) * beta[1] + @as(i64, rows[2][i]) * beta[2] + @as(i64, rows[3][i]) * beta[3];
        dst[i] = util.saturateU8((sum + (1 << 21)) >> 22);
    }
}

/// `resizeGeneric_Invoker`: per destination row the `ksize` source rows it
/// needs, horizontally resized once each and kept while the next rows reuse
/// them (`prev_sy`), then the vertical pass.
const GenericJob = struct {
    src: []const u8,
    sw: usize,
    sh: usize,
    ch: usize,
    dw: usize,
    xs: Axis,
    ys: Axis,
    ksize: usize,
    out: []u8,
    rows: []i32, // bands * ksize rows of dw * ch
    prev_sy: []i64, // bands * ksize

    fn run(self: *const GenericJob, band: usize, y0: usize, y1: usize) void {
        switch (self.ch) {
            1 => if (self.ksize == 2) self.runT(1, 2, band, y0, y1) else self.runT(1, 4, band, y0, y1),
            3 => if (self.ksize == 2) self.runT(3, 2, band, y0, y1) else self.runT(3, 4, band, y0, y1),
            else => if (self.ksize == 2) self.runT(4, 2, band, y0, y1) else self.runT(4, 4, band, y0, y1),
        }
    }

    fn runT(self: *const GenericJob, comptime ch: usize, comptime ksize: usize, band: usize, y0: usize, y1: usize) void {
        const dwc = self.dw * ch;
        const prev = self.prev_sy[band * ksize .. (band + 1) * ksize];
        @memset(prev, -1);
        var rows: [ksize][]i32 = undefined;
        for (0..ksize) |k| rows[k] = self.rows[(band * ksize + k) * dwc ..][0..dwc];
        var srows: [ksize]usize = undefined;
        for (y0..y1) |dy| {
            const sy_first = self.ys.ofs[dy];
            var k0: usize = ksize;
            var k1: usize = 0;
            for (0..ksize) |k| {
                const sy: i64 = std.math.clamp(sy_first + @as(i64, @intCast(k)), 0, @as(i64, @intCast(self.sh)) - 1);
                k1 = @max(k1, k);
                while (k1 < ksize) : (k1 += 1) {
                    if (sy == prev[k1]) {
                        // Already resized for an earlier row: reuse it.
                        if (k1 > k) @memcpy(rows[k], rows[k1]);
                        break;
                    }
                }
                if (k1 == ksize) k0 = @min(k0, k);
                srows[k] = @intCast(sy);
                prev[k] = sy;
            }
            for (k0..ksize) |k| {
                hresize(ch, ksize, self.src[srows[k] * self.sw * ch ..][0 .. self.sw * ch], self.sw, self.dw, self.xs, rows[k]);
            }
            const drow = self.out[dy * dwc .. (dy + 1) * dwc];
            if (ksize == 2) {
                vresizeLinear(rows[0], rows[1], self.ys.alpha[dy], self.ys.alpha[self.ys.ofs.len + dy], drow);
            } else {
                const n = self.ys.ofs.len;
                const beta = [4]i32{ self.ys.alpha[dy], self.ys.alpha[n + dy], self.ys.alpha[2 * n + dy], self.ys.alpha[3 * n + dy] };
                vresizeCubic(.{ rows[0], rows[1], rows[2], rows[3] }, &beta, drow);
            }
        }
    }
};

fn generic(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, interp: Interp) Allocator.Error![]u8 {
    const area_mode = interp == .area;
    const kind: Interp = if (interp == .cubic) .cubic else .linear;
    const ksize: usize = if (kind == .cubic) 4 else 2;
    const xs = try axisTable(a, sw, dw, kind, area_mode, true, ch);
    defer xs.deinit(a);
    const ys = try axisTable(a, sh, dh, kind, area_mode, false, 1);
    defer ys.deinit(a);
    const out = try a.alloc(u8, dw * dh * ch);
    errdefer a.free(out);
    const bands = par.bandCountWork(dh, 8, dw * ch * ksize * 4);
    const rows = try a.alloc(i32, bands * ksize * dw * ch);
    defer a.free(rows);
    const prev = try a.alloc(i64, bands * ksize);
    defer a.free(prev);
    const job = GenericJob{ .src = src, .sw = sw, .sh = sh, .ch = ch, .dw = dw, .xs = xs, .ys = ys, .ksize = ksize, .out = out, .rows = rows, .prev_sy = prev };
    par.forBandsWork(dh, 8, dw * ch * ksize * 4, &job, GenericJob.run);
    return out;
}
// -- tests -------------------------------------------------------------------------

test "sizes and flat images" {
    const a = std.testing.allocator;
    try std.testing.expectEqual([2]usize{ 1224, 918 }, scaledSize(4896, 3672, 0.25, 0.25));
    try std.testing.expectEqual([2]usize{ 1000, 750 }, scaledSize(4896, 3672, 1000.0 / 4896.0, 1000.0 / 4896.0));
    const flat = try a.alloc(u8, 64 * 48 * 3);
    defer a.free(flat);
    @memset(flat, 200);
    for ([_]Interp{ .nearest, .linear, .cubic, .area }) |i| {
        for ([_][2]usize{ .{ 16, 12 }, .{ 13, 9 }, .{ 100, 70 }, .{ 128, 96 }, .{ 32, 24 } }) |s| {
            const out = try resize(a, flat, 64, 48, 3, s[0], s[1], i);
            defer a.free(out);
            try std.testing.expectEqual(s[0] * s[1] * 3, out.len);
            for (out) |v| try std.testing.expectEqual(@as(u8, 200), v);
        }
    }
}

test "known values" {
    const a = std.testing.allocator;
    // 2x2 area average rounds half up: (0 + 1 + 2 + 3 + 2) >> 2 = 2.
    const q = [_]u8{ 0, 1, 2, 3 };
    const half = try resize(a, &q, 2, 2, 1, 1, 1, .area);
    defer a.free(half);
    try std.testing.expectEqual(@as(u8, 2), half[0]);
    // INTER_LINEAR 2x downscale is the area average too.
    const lin = try resize(a, &q, 2, 2, 1, 1, 1, .linear);
    defer a.free(lin);
    try std.testing.expectEqual(@as(u8, 2), lin[0]);
    // 4x4 of 10 and one 42: sum 192, * 1/16 = 12.
    var b: [16]u8 = @splat(10);
    b[5] = 42;
    const quarter = try resize(a, &b, 4, 4, 1, 1, 1, .area);
    defer a.free(quarter);
    try std.testing.expectEqual(@as(u8, 12), quarter[0]);
    // Values from cv2.resize on the same inputs.
    const ramp = [_]u8{ 0, 100, 200, 255 };
    const up = try resize(a, &ramp, 4, 1, 1, 16, 1, .linear);
    defer a.free(up);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 13, 38, 63, 88, 113, 138, 163, 188, 207, 221, 234, 248, 255, 255 }, up);
    const nn = try resize(a, &ramp, 4, 1, 1, 8, 1, .nearest);
    defer a.free(nn);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 100, 100, 200, 200, 255, 255 }, nn);
    const cubic = try resize(a, &ramp, 4, 1, 1, 8, 1, .cubic);
    defer a.free(cubic);
    try std.testing.expectEqualSlices(u8, &.{ 0, 19, 67, 131, 175, 223, 246, 255 }, cubic);
    const two = [_]u8{ 0, 100, 200, 255, 50, 60, 70, 80 };
    const lin2 = try resize(a, &two, 4, 2, 1, 8, 6, .linear);
    defer a.free(lin2);
    try std.testing.expectEqualSlices(u8, &.{ 0, 25, 75, 125, 175, 214, 241, 255, 0, 25, 75, 125, 175, 214, 241, 255, 17, 34, 69, 104, 139, 166, 187, 196, 33, 43, 63, 83, 103, 120, 132, 138, 50, 53, 58, 63, 68, 73, 78, 80, 50, 52, 57, 62, 67, 72, 77, 80 }, lin2);
    const cub2 = try resize(a, &two, 4, 2, 1, 8, 6, .cubic);
    defer a.free(cub2);
    try std.testing.expectEqualSlices(u8, &.{ 0, 15, 68, 139, 187, 240, 255, 255, 0, 19, 67, 131, 175, 223, 246, 255, 8, 29, 64, 110, 141, 176, 193, 204, 30, 42, 60, 84, 101, 120, 131, 138, 49, 52, 57, 63, 67, 73, 78, 81, 56, 56, 56, 55, 55, 57, 59, 61 }, cub2);
    const grad = try a.alloc(u8, 64 * 48);
    defer a.free(grad);
    for (grad, 0..) |*v, i| v.* = @truncate(i);
    const frac = try resize(a, grad, 64, 48, 1, 13, 9, .area);
    defer a.free(frac);
    try std.testing.expectEqualSlices(u8, &.{ 78, 83, 88, 93, 98, 103, 108, 112, 117, 122, 127, 132, 137, 98, 103, 108, 113, 118, 123, 128, 132, 137, 142, 147, 152, 157 }, frac[0..26]);
    const c = cubicCoeffs(0.5);
    try std.testing.expect(@abs(c[0] + 0.09375) < 1e-6 and @abs(c[1] - 0.59375) < 1e-6);
}
