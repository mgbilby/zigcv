// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/imgwarp.cpp, modules/imgproc/src/warp_kernels.simd.hpp, modules/imgproc/src/warp_common.vector.hpp, modules/imgproc/src/warp_common.scalar.hpp.
// Licence of the sources: Apache-2.0; BSD-3-Clause; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.warpPerspective`, `cv2.warpAffine` and `cv2.remap` for 8-bit images,
//! with the numerics of OpenCV's table-free warp kernels
//! (`warp_kernels.simd.hpp`, `warp_common.*.hpp`, `imgwarp.cpp`):
//!
//! - INTER_LINEAR / INTER_NEAREST: source coordinates in float32 (fused
//!   multiply-add and a true division in the 16-pixel vector region of a row,
//!   the compiler-contracted scalar expression in the tail), floor / round,
//!   bilinear interpolation `p0 + f (p1 - p0)` with fused multiply-adds;
//! - INTER_CUBIC (`genericWarp` + `bicubicVec`): coordinates in double from the
//!   float32 matrix, `bicubicWeights` and the fused row / column chains;
//! - BORDER_REPLICATE clamps every tap, BORDER_CONSTANT substitutes the value.
//!
//! The kernels run as OpenCV's vector invokers do: 16 destination pixels at a
//! time, their taps gathered into channel-major planes (`pixbuf`) and
//! interpolated in 16-lane vectors, the results interleaved on store.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

pub const Interp = enum { nearest, linear, cubic };
pub const BorderMode = enum { replicate, constant };

/// 3x3 homography, row major, mapping source to destination (as `cv2.warpPerspective` takes it).
pub const Matrix3 = [3][3]f64;

/// Options shared by the warps.
pub const Options = struct {
    interp: Interp = .linear,
    border: BorderMode = .replicate,
    /// Border value per channel (`borderValue`), used with `.constant`.
    value: [4]u8 = .{ 0, 0, 0, 0 },
    /// `WARP_INVERSE_MAP`: `m` already maps destination to source.
    inverse_map: bool = false,
};

/// `cv::invert` of a 3x3 double matrix (the cofactor formula OpenCV uses for n <= 3).
pub fn invert3(m: Matrix3) Matrix3 {
    const a = m[0][0];
    const b = m[0][1];
    const c = m[0][2];
    const d = m[1][0];
    const e = m[1][1];
    const f = m[1][2];
    const g = m[2][0];
    const h = m[2][1];
    const i = m[2][2];
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det == 0.0) return .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    const inv = 1.0 / det;
    return .{
        .{ (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv },
        .{ (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv },
        .{ (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv },
    };
}

// -- sampling ---------------------------------------------------------------------------

/// The vector kernels handle 16 pixels per step.
const B = 16;
const VF = @Vector(B, f32);
const VI = @Vector(B, i32);
const VU = @Vector(B, u32);
const VB = @Vector(B, u8);

const Src = struct {
    data: []const u8,
    w: usize,
    h: usize,
    ch: usize,
    border: BorderMode,
    value: [4]u8,

    /// Byte offset of pixel (`x`, `y`) when it is inside the image, else the
    /// border rule: `null` for the constant value, the clamped pixel for replicate.
    inline fn tap(self: Src, x: i32, y: i32) ?usize {
        const wi: i32 = @intCast(self.w);
        const hi: i32 = @intCast(self.h);
        if (x >= 0 and y >= 0 and x < wi and y < hi) {
            return (@as(usize, @intCast(y)) * self.w + @as(usize, @intCast(x))) * self.ch;
        }
        if (self.border == .constant) return null;
        const cx: usize = @intCast(std.math.clamp(x, 0, wi - 1));
        const cy: usize = @intCast(std.math.clamp(y, 0, hi - 1));
        return (cy * self.w + cx) * self.ch;
    }
};

fn splatF(x: f32) VF {
    return @splat(x);
}

fn splatI(x: i32) VI {
    return @splat(x);
}

/// Lanes of `[plane 0 (16), plane 1 (16), ...]` in pixel-interleaved order.
fn interleaveMask(comptime ch: usize) @Vector(B * ch, i32) {
    var m: [B * ch]i32 = undefined;
    for (&m, 0..) |*v, i| v.* = @intCast((i % ch) * B + i / ch);
    return m;
}

/// Store `ch` planes of 16 bytes as 16 interleaved pixels (`v_store_interleave`).
inline fn storeInterleaved(comptime ch: usize, planes: [ch]VB, out: *[B * ch]u8) void {
    if (ch == 1) {
        out.* = planes[0];
    } else if (ch == 3) {
        const rg: @Vector(2 * B, u8) = std.simd.join(planes[0], planes[1]);
        const rgb: @Vector(3 * B, u8) = std.simd.join(rg, planes[2]);
        out.* = @shuffle(u8, rgb, undefined, interleaveMask(3));
    } else {
        const rg: @Vector(2 * B, u8) = std.simd.join(planes[0], planes[1]);
        const ba: @Vector(2 * B, u8) = std.simd.join(planes[2], planes[3]);
        const rgba: @Vector(4 * B, u8) = std.simd.join(rg, ba);
        out.* = @shuffle(u8, rgba, undefined, interleaveMask(4));
    }
}

/// `v_round` then `v_pack_u`: round half to even and clamp to 8 bits.
inline fn packU8(v: VF) VB {
    const r = @min(@max(util.roundEvenVec(B, v), splatF(0)), splatF(255));
    return @intFromFloat(r);
}

/// Fill 16 output pixels with the border value.
inline fn fillValue(comptime ch: usize, src: Src, out: *[B * ch]u8) void {
    for (0..B) |i| {
        inline for (0..ch) |c| out[i * ch + c] = src.value[c];
    }
}

// -- gathers ---------------------------------------------------------------------

fn wordCount(comptime taps: usize, comptime ch: usize) usize {
    return (taps * ch + 3) / 4;
}

/// The `taps ch` consecutive bytes of 16 pixels (at `addr[i] + ofs`) as
/// 32-bit words, one vector per word position: the gather of OpenCV's
/// `FETCH_INLIERS` done per dword, so the tap planes are byte extractions
/// instead of a byte-by-byte scatter.
inline fn gatherWords(comptime taps: usize, comptime ch: usize, data: [*]const u8, addr: *const [B]usize, ofs: usize) [wordCount(taps, ch)]VU {
    @setEvalBranchQuota(100_000);
    const need = taps * ch;
    var w: [wordCount(taps, ch)]VU = undefined;
    inline for (0..comptime wordCount(taps, ch)) |s| {
        const n = comptime @min(4, need - s * 4);
        var v: VU = undefined;
        inline for (0..B) |i| v[i] = util.readUpTo4(n, data + addr[i] + ofs + s * 4);
        w[s] = v;
    }
    return w;
}

/// Float plane of tap `t`, channel `c` from the gathered words.
inline fn planeOf(comptime ch: usize, w: anytype, comptime t: usize, comptime c: usize) VF {
    const b = t * ch + c;
    const sh: @Vector(B, u5) = @splat(@intCast((b % 4) * 8));
    return @floatFromInt((w[b / 4] >> sh) & @as(VU, @splat(0xFF)));
}

/// Addresses of the top-left tap of 16 pixels at integer positions.
inline fn addresses(comptime ch: usize, src: Src, ix: VI, iy: VI) [B]usize {
    const xs: [B]i32 = ix;
    const ys: [B]i32 = iy;
    var addr: [B]usize = undefined;
    for (0..B) |i| addr[i] = (@as(usize, @intCast(ys[i])) * src.w + @as(usize, @intCast(xs[i]))) * ch;
    return addr;
}

/// The 2 x 2 taps of 16 pixels at integer positions (`ix`, `iy`) as float
/// planes `f[channel][tap]` (taps 00, 01, 10, 11), the
/// `CV_WARP_VECTOR_SHUFFLE_ALLWITHIN` / `NOTALLWITHIN` gathers; outside pixels
/// follow the border rule. Returns false when, with a constant border, every
/// pixel lies entirely outside (the caller stores the value).
inline fn gatherLinear(comptime ch: usize, src: Src, ix: VI, iy: VI, f: *[ch][4]VF) bool {
    const wi: i32 = @intCast(src.w);
    const hi: i32 = @intCast(src.h);
    const inside = @reduce(.And, (ix >= splatI(0)) & (iy >= splatI(0)) & (ix < splatI(wi - 1)) & (iy < splatI(hi - 1)));
    if (inside) {
        const addr = addresses(ch, src, ix, iy);
        const w0 = gatherWords(2, ch, src.data.ptr, &addr, 0);
        const w1 = gatherWords(2, ch, src.data.ptr, &addr, src.w * ch);
        inline for (0..ch) |c| f[c] = .{ planeOf(ch, w0, 0, c), planeOf(ch, w0, 1, c), planeOf(ch, w1, 0, c), planeOf(ch, w1, 1, c) };
        return true;
    }
    if (src.border == .constant) {
        const partly = (ix >= splatI(-1)) & (iy >= splatI(-1)) & (ix < splatI(wi)) & (iy < splatI(hi));
        if (!@reduce(.Or, partly)) return false;
    }
    const xs: [B]i32 = ix;
    const ys: [B]i32 = iy;
    var pixbuf: [ch][4][B]u8 = undefined;
    for (0..B) |i| {
        inline for (0..4) |t| {
            if (src.tap(xs[i] + @as(i32, t & 1), ys[i] + @as(i32, t >> 1))) |o| {
                inline for (0..ch) |c| pixbuf[c][t][i] = src.data[o + c];
            } else {
                inline for (0..ch) |c| pixbuf[c][t][i] = src.value[c];
            }
        }
    }
    inline for (0..ch) |c| {
        inline for (0..4) |t| f[c][t] = @floatFromInt(@as(VB, pixbuf[c][t]));
    }
    return true;
}

/// Row `r` of the 4 x 4 taps of 16 pixels whose top-left tap is (`ix0`, `iy0`),
/// as float planes `f[channel][column]` (`FETCH_INLIERS` / `bicubicFetchPixels`);
/// `addr` holds the top-left addresses when every pixel's taps are inside.
inline fn gatherCubicRow(comptime ch: usize, src: Src, inside: bool, addr: *const [B]usize, ix0: VI, iy0: VI, comptime r: usize, f: *[ch][4]VF) void {
    if (inside) {
        const w = gatherWords(4, ch, src.data.ptr, addr, r * src.w * ch);
        inline for (0..ch) |c| f[c] = .{ planeOf(ch, w, 0, c), planeOf(ch, w, 1, c), planeOf(ch, w, 2, c), planeOf(ch, w, 3, c) };
        return;
    }
    const xs: [B]i32 = ix0;
    const ys: [B]i32 = iy0;
    var pixbuf: [ch][4][B]u8 = undefined;
    for (0..B) |i| {
        inline for (0..4) |k| {
            if (src.tap(xs[i] + @as(i32, k), ys[i] + @as(i32, r))) |o| {
                inline for (0..ch) |c| pixbuf[c][k][i] = src.data[o + c];
            } else {
                inline for (0..ch) |c| pixbuf[c][k][i] = src.value[c];
            }
        }
    }
    inline for (0..ch) |c| {
        inline for (0..4) |k| f[c][k] = @floatFromInt(@as(VB, pixbuf[c][k]));
    }
}

/// Keep float coordinates convertible to i32 (OpenCV's `cvtps` yields
/// INT_MIN for anything else, which also lands outside every image).
inline fn clampCoord(v: VF) VF {
    return @min(@max(v, splatF(-1.0e9)), splatF(1.0e9));
}

// -- block kernels ----------------------------------------------------------------

/// `warpAffineLinearInvoker` / `warpPerspectiveLinearInvoker` / `remapLinearInvoker`
/// on 16 pixels: floor, `p0 + alpha (p1 - p0)` with fused multiply-adds, round.
fn linearBlock(comptime ch: usize, src: Src, xs: *const [B]f32, ys: *const [B]f32, out: *[B * ch]u8) void {
    const sx = clampCoord(xs.*);
    const sy = clampCoord(ys.*);
    const fx = @floor(sx);
    const fy = @floor(sy);
    const ix: VI = @intFromFloat(fx);
    const iy: VI = @intFromFloat(fy);
    const alpha = sx - fx;
    const beta = sy - fy;
    var f: [ch][4]VF = undefined;
    if (!gatherLinear(ch, src, ix, iy, &f)) return fillValue(ch, src, out);
    var planes: [ch]VB = undefined;
    inline for (0..ch) |c| {
        var f00 = f[c][0];
        const f01 = f[c][1];
        var f10 = f[c][2];
        const f11 = f[c][3];
        f00 = @mulAdd(VF, alpha, f01 - f00, f00);
        f10 = @mulAdd(VF, alpha, f11 - f10, f10);
        f00 = @mulAdd(VF, beta, f10 - f00, f00);
        planes[c] = packU8(f00);
    }
    storeInterleaved(ch, planes, out);
}

/// `v_round` of the coordinates, then the pixel or the border rule.
fn nearestBlock(comptime ch: usize, src: Src, xs: *const [B]f32, ys: *const [B]f32, out: *[B * ch]u8) void {
    const ix: VI = @intFromFloat(util.roundEvenVec(B, clampCoord(xs.*)));
    const iy: VI = @intFromFloat(util.roundEvenVec(B, clampCoord(ys.*)));
    const xi: [B]i32 = ix;
    const yi: [B]i32 = iy;
    for (0..B) |i| {
        if (src.tap(xi[i], yi[i])) |o| {
            inline for (0..ch) |c| out[i * ch + c] = src.data[o + c];
        } else {
            inline for (0..ch) |c| out[i * ch + c] = src.value[c];
        }
    }
}

/// `bicubicWeights` as the vector kernel computes them (A = -0.75).
inline fn cubicWeightsV(alpha: VF) [4]VF {
    const A: f32 = -0.75;
    const a2 = alpha * alpha;
    const b = splatF(1.0) - alpha;
    const b2 = b * b;
    const w0 = splatF(A) * (alpha * b2);
    const w3 = splatF(A) * (a2 * b);
    const w1 = @mulAdd(VF, a2, @mulAdd(VF, splatF(A + 2.0), alpha, splatF(-(A + 3.0))), splatF(1.0));
    const w2 = ((splatF(1.0) - w0) - w1) - w3;
    return .{ w0, w1, w2, w3 };
}

/// `bicubicVec` (`genericWarp`): clamped coordinates, 4 x 4 taps, fused row
/// sums (`BICUBIC_UPDATE_ACC_VEC`) then fused row weights, row by row.
fn cubicBlock(comptime ch: usize, src: Src, xs: *const [B]f32, ys: *const [B]f32, out: *[B * ch]u8) void {
    const min_size = 16;
    const bigw: f32 = @floatFromInt(@max(src.w, min_size));
    const bigh: f32 = @floatFromInt(@max(src.h, min_size));
    const vx0 = @min(@max(@as(VF, xs.*), splatF(-bigw)), splatF(bigw * 2));
    const vy0 = @min(@max(@as(VF, ys.*), splatF(-bigh)), splatF(bigh * 2));
    const fx0 = @floor(vx0);
    const fy0 = @floor(vy0);
    const alpha = vx0 - fx0;
    const beta = vy0 - fy0;
    const ix0 = @as(VI, @intFromFloat(fx0)) - splatI(1);
    const iy0 = @as(VI, @intFromFloat(fy0)) - splatI(1);
    const wi: i32 = @intCast(src.w);
    const hi: i32 = @intCast(src.h);
    const inside = @reduce(.And, (ix0 >= splatI(0)) & (iy0 >= splatI(0)) & (ix0 < splatI(wi - 3)) & (iy0 < splatI(hi - 3)));
    if (!inside and src.border == .constant) {
        const partly = (ix0 >= splatI(-3)) & (iy0 >= splatI(-3)) & (ix0 < splatI(wi)) & (iy0 < splatI(hi));
        if (!@reduce(.Or, partly)) return fillValue(ch, src, out);
    }
    const addr = if (inside) addresses(ch, src, ix0, iy0) else undefined;
    const wx = cubicWeightsV(alpha);
    const wy = cubicWeightsV(beta);
    var acc: [ch]VF = undefined;
    inline for (0..ch) |c| acc[c] = @splat(0);
    inline for (0..4) |r| {
        var f: [ch][4]VF = undefined;
        gatherCubicRow(ch, src, inside, &addr, ix0, iy0, r, &f);
        inline for (0..ch) |c| {
            var sumwx = @mulAdd(VF, f[c][1], wx[1], f[c][0] * wx[0]);
            sumwx = @mulAdd(VF, f[c][2], wx[2], sumwx);
            sumwx = @mulAdd(VF, f[c][3], wx[3], sumwx);
            acc[c] = @mulAdd(VF, sumwx, wy[r], acc[c]);
        }
    }
    var planes: [ch]VB = undefined;
    inline for (0..ch) |c| planes[c] = packU8(acc[c]);
    storeInterleaved(ch, planes, out);
}

// -- coordinates ---------------------------------------------------------------------

const Job = struct {
    src: Src,
    dw: usize,
    out: []u8,
    interp: Interp,
    /// Destination -> source matrix (inverse of the forward one), in double.
    inv: Matrix3,
    affine: bool,
    map_x: ?[]const f32 = null,
    map_y: ?[]const f32 = null,
    xs: []f32, // bands * dw scratch
    ys: []f32,

    fn run(self: *const Job, band: usize, y0: usize, y1: usize) void {
        switch (self.src.ch) {
            1 => self.runCh(1, band, y0, y1),
            3 => self.runCh(3, band, y0, y1),
            else => self.runCh(4, band, y0, y1),
        }
    }

    fn runCh(self: *const Job, comptime ch: usize, band: usize, y0: usize, y1: usize) void {
        const dw = self.dw;
        for (y0..y1) |y| {
            const drow = self.out[y * dw * ch .. (y + 1) * dw * ch];
            var xs = self.xs[band * dw .. (band + 1) * dw];
            var ys = self.ys[band * dw .. (band + 1) * dw];
            if (self.map_x) |mx| {
                xs = @constCast(mx[y * dw .. (y + 1) * dw]);
                ys = @constCast(self.map_y.?[y * dw .. (y + 1) * dw]);
            } else if (self.interp == .cubic) {
                self.cubicCoords(y, xs, ys);
            } else {
                self.linearCoords(y, xs, ys);
            }
            var x: usize = 0;
            while (x + B <= dw) : (x += B) {
                self.block(ch, xs[x..][0..B], ys[x..][0..B], drow[x * ch ..][0 .. B * ch]);
            }
            if (x < dw) {
                // A short last block repeats its final pixel: every lane is
                // computed on its own, so the padding changes nothing.
                const n = dw - x;
                var bx: [B]f32 = undefined;
                var by: [B]f32 = undefined;
                @memcpy(bx[0..n], xs[x..]);
                @memcpy(by[0..n], ys[x..]);
                @memset(bx[n..], xs[dw - 1]);
                @memset(by[n..], ys[dw - 1]);
                var blk: [B * ch]u8 = undefined;
                self.block(ch, &bx, &by, &blk);
                @memcpy(drow[x * ch ..], blk[0 .. n * ch]);
            }
        }
    }

    inline fn block(self: *const Job, comptime ch: usize, xs: *const [B]f32, ys: *const [B]f32, out: *[B * ch]u8) void {
        switch (self.interp) {
            .linear => linearBlock(ch, self.src, xs, ys, out),
            .cubic => cubicBlock(ch, self.src, xs, ys, out),
            .nearest => nearestBlock(ch, self.src, xs, ys, out),
        }
    }

    /// Coordinates of the linear / nearest kernels: float32 matrix, fused
    /// multiply-add and division per vector lane, the compiler-contracted
    /// scalar expression in the tail of the row.
    fn linearCoords(self: *const Job, y: usize, xs: []f32, ys: []f32) void {
        var m: [9]f32 = undefined;
        for (0..9) |i| m[i] = @floatCast(self.inv[i / 3][i % 3]);
        const yf: f32 = @floatFromInt(y);
        const m_x = yf * m[1] + m[2];
        const m_y = yf * m[4] + m[5];
        const m_w = yf * m[7] + m[8];
        const vec_end = self.dw - self.dw % B;
        for (0..vec_end) |x| {
            const xf: f32 = @floatFromInt(x);
            var sx = @mulAdd(f32, m[0], xf, m_x);
            var sy = @mulAdd(f32, m[3], xf, m_y);
            if (!self.affine) {
                const sw = @mulAdd(f32, m[6], xf, m_w);
                sx /= sw;
                sy /= sw;
            }
            xs[x] = sx;
            ys[x] = sy;
        }
        // The scalar tail: `x*M[0] + y*M[1] + M[2]`, which the compiler contracts
        // to a fused multiply-add of the first product onto the second.
        for (vec_end..self.dw) |x| {
            const xf: f32 = @floatFromInt(x);
            var sx = @mulAdd(f32, xf, m[0], yf * m[1]) + m[2];
            var sy = @mulAdd(f32, xf, m[3], yf * m[4]) + m[5];
            if (!self.affine) {
                const w = @mulAdd(f32, xf, m[6], yf * m[7]) + m[8];
                sx /= w;
                sy /= w;
            }
            xs[x] = sx;
            ys[x] = sy;
        }
    }

    /// Coordinates of `genericWarp` (cubic): the float32 matrix evaluated in double per pixel.
    fn cubicCoords(self: *const Job, y: usize, xs: []f32, ys: []f32) void {
        var m: [9]f32 = undefined;
        for (0..9) |i| m[i] = @floatCast(self.inv[i / 3][i % 3]);
        const yf: f32 = @floatFromInt(y);
        const m_x: f64 = yf * m[1] + m[2];
        const m_y: f64 = yf * m[4] + m[5];
        const m_z: f64 = if (self.affine) 1.0 else yf * m[7] + m[8];
        const m_xx: f64 = m[0];
        const m_yx: f64 = m[3];
        const m_zx: f64 = if (self.affine) 0.0 else m[6];
        for (0..self.dw) |x| {
            const xf: f64 = @floatFromInt(x);
            if (self.affine) {
                xs[x] = @floatCast(@as(f32, @floatCast(m_x)) + @as(f32, @floatCast(m_xx)) * @as(f32, @floatCast(xf)));
                ys[x] = @floatCast(@as(f32, @floatCast(m_y)) + @as(f32, @floatCast(m_yx)) * @as(f32, @floatCast(xf)));
            } else {
                const invz = 1.0 / (m_z + m_zx * xf);
                xs[x] = @floatCast((m_x + m_xx * xf) * invz);
                ys[x] = @floatCast((m_y + m_yx * xf) * invz);
            }
        }
    }
};

fn runJob(a: Allocator, src: Src, dw: usize, dh: usize, interp: Interp, inv: Matrix3, affine: bool, map_x: ?[]const f32, map_y: ?[]const f32) Allocator.Error![]u8 {
    const out = try a.alloc(u8, dw * dh * src.ch);
    errdefer a.free(out);
    const bands = par.bandCountWork(dh, 4, dw * src.ch * 16);
    const xs = try a.alloc(f32, bands * dw);
    defer a.free(xs);
    const ys = try a.alloc(f32, bands * dw);
    defer a.free(ys);
    const job = Job{ .src = src, .dw = dw, .out = out, .interp = interp, .inv = inv, .affine = affine, .map_x = map_x, .map_y = map_y, .xs = xs, .ys = ys };
    par.forBandsWork(dh, 4, dw * src.ch * 16, &job, Job.run);
    return out;
}

/// `cv2.warpPerspective(src, m, (dw, dh), flags=interp, borderMode, borderValue)`.
pub fn warpPerspective(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, m: Matrix3, dw: usize, dh: usize, opts: Options) Allocator.Error![]u8 {
    std.debug.assert(ch == 1 or ch == 3 or ch == 4);
    const inv = if (opts.inverse_map) m else invert3(m);
    const s = Src{ .data = src, .w = sw, .h = sh, .ch = ch, .border = opts.border, .value = opts.value };
    return runJob(a, s, dw, dh, opts.interp, inv, false, null, null);
}

/// `cv2.warpAffine(src, m, (dw, dh), ...)` with a 2x3 matrix.
pub fn warpAffine(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, m: [2][3]f64, dw: usize, dh: usize, opts: Options) Allocator.Error![]u8 {
    std.debug.assert(ch == 1 or ch == 3 or ch == 4);
    var inv: Matrix3 = .{ .{ m[0][0], m[0][1], m[0][2] }, .{ m[1][0], m[1][1], m[1][2] }, .{ 0, 0, 1 } };
    if (!opts.inverse_map) {
        // cv::invertAffineTransform: 2x2 inverse in double, then the translation.
        const d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        const id = if (d != 0) 1.0 / d else 0.0;
        const a11 = m[1][1] * id;
        const a22 = m[0][0] * id;
        const a12 = -m[0][1] * id;
        const a21 = -m[1][0] * id;
        const b1 = -a11 * m[0][2] - a12 * m[1][2];
        const b2 = -a21 * m[0][2] - a22 * m[1][2];
        inv = .{ .{ a11, a12, b1 }, .{ a21, a22, b2 }, .{ 0, 0, 1 } };
    }
    const s = Src{ .data = src, .w = sw, .h = sh, .ch = ch, .border = opts.border, .value = opts.value };
    return runJob(a, s, dw, dh, opts.interp, inv, true, null, null);
}

/// `cv2.remap(src, map_x, map_y, interp, borderMode, borderValue)` with planar
/// float32 maps of `dw * dh` source positions (INTER_LINEAR, INTER_CUBIC or INTER_NEAREST).
pub fn remap(a: Allocator, src: []const u8, sw: usize, sh: usize, ch: usize, map_x: []const f32, map_y: []const f32, dw: usize, dh: usize, opts: Options) Allocator.Error![]u8 {
    std.debug.assert(map_x.len == dw * dh and map_y.len == dw * dh);
    std.debug.assert(ch == 1 or ch == 3 or ch == 4);
    const s = Src{ .data = src, .w = sw, .h = sh, .ch = ch, .border = opts.border, .value = opts.value };
    const ident: Matrix3 = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } };
    return runJob(a, s, dw, dh, opts.interp, ident, false, map_x, map_y);
}

// -- tests -------------------------------------------------------------------------

test "identity warps copy the image" {
    const a = std.testing.allocator;
    const w = 37;
    const h = 21;
    var img: [w * h * 3]u8 = undefined;
    for (&img, 0..) |*v, i| v.* = @truncate(i * 7 + i / 5);
    const ident: Matrix3 = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } };
    for ([_]Interp{ .nearest, .linear, .cubic }) |interp| {
        const out = try warpPerspective(a, &img, w, h, 3, ident, w, h, .{ .interp = interp });
        defer a.free(out);
        try std.testing.expectEqualSlices(u8, &img, out);
        const aff = try warpAffine(a, &img, w, h, 3, .{ .{ 1, 0, 0 }, .{ 0, 1, 0 } }, w, h, .{ .interp = interp });
        defer a.free(aff);
        try std.testing.expectEqualSlices(u8, &img, aff);
    }
    var mx: [w * h]f32 = undefined;
    var my: [w * h]f32 = undefined;
    for (0..w * h) |i| {
        mx[i] = @floatFromInt(i % w);
        my[i] = @floatFromInt(i / w);
    }
    const same = try remap(a, &img, w, h, 3, &mx, &my, w, h, .{ .border = .constant });
    defer a.free(same);
    try std.testing.expectEqualSlices(u8, &img, same);
    // Half a pixel to the right interpolates; far outside is the border value.
    for (&mx) |*v| v.* += 0.5;
    const shifted = try remap(a, &img, w, h, 3, &mx, &my, w, h, .{ .border = .constant, .value = .{ 7, 7, 7, 0 } });
    defer a.free(shifted);
    const i = (w + 4) * 3;
    const expect: u32 = (@as(u32, img[i]) + @as(u32, img[i + 3]) + 1) / 2;
    try std.testing.expect(@abs(@as(i32, shifted[i]) - @as(i32, @intCast(expect))) <= 1);
    @memset(&mx, 500.0);
    const outside = try remap(a, &img, w, h, 3, &mx, &my, w, h, .{ .border = .constant, .value = .{ 7, 8, 9, 0 } });
    defer a.free(outside);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9 }, outside[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9 }, outside[outside.len - 3 ..]);
}

test "translation by whole pixels is exact in every mode" {
    const a = std.testing.allocator;
    const w = 40;
    const h = 30;
    var img: [w * h]u8 = undefined;
    for (&img, 0..) |*v, i| v.* = @truncate(i * 3);
    const shift: Matrix3 = .{ .{ 1, 0, 5 }, .{ 0, 1, 3 }, .{ 0, 0, 1 } };
    for ([_]Interp{ .nearest, .linear, .cubic }) |interp| {
        const out = try warpPerspective(a, &img, w, h, 1, shift, w, h, .{ .interp = interp, .border = .constant, .value = .{ 9, 0, 0, 0 } });
        defer a.free(out);
        try std.testing.expectEqual(img[0], out[3 * w + 5]);
        try std.testing.expectEqual(img[10 * w + 10], out[13 * w + 15]);
        try std.testing.expectEqual(@as(u8, 9), out[0]);
        const aff = try warpAffine(a, &img, w, h, 1, .{ .{ 1, 0, 5 }, .{ 0, 1, 3 } }, w, h, .{ .interp = interp, .border = .replicate });
        defer a.free(aff);
        try std.testing.expectEqual(img[0], aff[3 * w + 5]);
        try std.testing.expectEqual(img[0], aff[0]);
        try std.testing.expectEqual(img[10 * w + 10], aff[13 * w + 15]);
    }
    const inv = invert3(shift);
    try std.testing.expectEqual(@as(f64, -5), inv[0][2]);
    const wts = cubicWeightsV(@splat(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), wts[1][0]);
    try std.testing.expectEqual(@as(f32, 0.0), wts[0][0]);
}

test "a short last block and a one-pixel image" {
    const a = std.testing.allocator;
    // 19 columns: one full block and a 3-pixel tail.
    const w = 19;
    const h = 5;
    var img: [w * h]u8 = undefined;
    for (&img, 0..) |*v, i| v.* = @truncate(i * 11 + 3);
    const m: Matrix3 = .{ .{ 1, 0, 0.5 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } };
    const out = try warpPerspective(a, &img, w, h, 1, m, w, h, .{ .interp = .linear, .border = .constant, .value = .{ 1, 0, 0, 0 } });
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, w * h), out.len);
    // Column 0 is half source column -1 (the border value) and half column 0.
    try std.testing.expectEqual(@as(u8, (1 + img[0]) / 2), out[0]);
    const tiny = [_]u8{200};
    const grown = try warpPerspective(a, &tiny, 1, 1, 1, .{ .{ 3, 0, 0 }, .{ 0, 3, 0 }, .{ 0, 0, 1 } }, 3, 3, .{ .interp = .cubic, .border = .replicate });
    defer a.free(grown);
    for (grown) |v| try std.testing.expectEqual(@as(u8, 200), v);
}
