// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/color_rgb.simd.hpp, modules/imgproc/src/color_hsv.simd.hpp.
// Licence of the sources: Apache-2.0; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.cvtColor` conversions with OpenCV's integer rounding. The buffers
//! here are RGB-ordered (the app decodes JPEGs to RGB), so `rgbToGray` on RGB
//! data equals `cv2.cvtColor(bgr, COLOR_BGR2GRAY)` on the same image in BGR.
//!
//! Both conversions run as the vector loops of `RGB2Gray<uchar>` and
//! `RGB2HSV_b` do: a block of pixels is split into channel planes, converted
//! in integer vector lanes (the HSV division tables looked up per lane) and
//! packed back; the last pixels of a row take the same arithmetic one at a time.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

// OpenCV (color_rgb.simd.hpp, RGB2Gray<uchar>): R 0.299, G 0.587, B 0.114 in
// 15-bit fixed point (RY15, GY15, BY15), rounded once.
const yuv_shift = 15;
const r_coef: u32 = 9798;
const g_coef: u32 = 19235;
const b_coef: u32 = 3735;

pub inline fn grayPixel(r: u8, g: u8, b: u8) u8 {
    return @intCast((r_coef * @as(u32, r) + g_coef * @as(u32, g) + b_coef * @as(u32, b) + (1 << (yuv_shift - 1))) >> yuv_shift);
}

/// Pixels per vector block.
const B = 16;
const VU = @Vector(B, u32);
const VI = @Vector(B, i32);
const VB = @Vector(B, u8);

/// The three channel planes of 16 RGB pixels (`v_load_deinterleave`), as 32-bit lanes.
inline fn loadPlanes(px: *const [3 * B]u8) [3]VU {
    const v: @Vector(3 * B, u8) = px.*;
    comptime var mr: [B]i32 = undefined;
    comptime var mg: [B]i32 = undefined;
    comptime var mb: [B]i32 = undefined;
    inline for (0..B) |i| {
        mr[i] = @intCast(i * 3);
        mg[i] = @intCast(i * 3 + 1);
        mb[i] = @intCast(i * 3 + 2);
    }
    return .{
        @intCast(@shuffle(u8, v, undefined, mr)),
        @intCast(@shuffle(u8, v, undefined, mg)),
        @intCast(@shuffle(u8, v, undefined, mb)),
    };
}

/// Interleave three planes of 16 bytes into 48 bytes (`v_store_interleave`).
inline fn storePlanes(planes: [3]VB, out: *[3 * B]u8) void {
    const rg: @Vector(2 * B, u8) = std.simd.join(planes[0], planes[1]);
    const rgb: @Vector(3 * B, u8) = std.simd.join(rg, planes[2]);
    comptime var m: [3 * B]i32 = undefined;
    inline for (&m, 0..) |*v, i| v.* = @intCast((i % 3) * B + i / 3);
    out.* = @shuffle(u8, rgb, undefined, m);
}

const GrayJob = struct {
    src: []const u8,
    w: usize,
    out: []u8,

    fn run(self: *const GrayJob, band: usize, y0: usize, y1: usize) void {
        _ = band;
        const s = self.src[y0 * self.w * 3 .. y1 * self.w * 3];
        const d = self.out[y0 * self.w .. y1 * self.w];
        for (d, 0..) |*o, i| {
            const r: u32 = s[i * 3];
            const g: u32 = s[i * 3 + 1];
            const b: u32 = s[i * 3 + 2];
            o.* = @intCast((r * r_coef + g * g_coef + b * b_coef + (1 << (yuv_shift - 1))) >> yuv_shift);
        }
    }
};

/// RGB -> grey (`cv2.COLOR_BGR2GRAY` on the BGR view of the same pixels).
pub fn rgbToGray(a: Allocator, src: []const u8, w: usize, h: usize) Allocator.Error![]u8 {
    std.debug.assert(src.len >= w * h * 3);
    const out = try a.alloc(u8, w * h);
    const job = GrayJob{ .src = src[0 .. w * h * 3], .w = w, .out = out };
    par.forBandsWork(h, 64, w * 3, &job, GrayJob.run);
    return out;
}

// -- HSV ----------------------------------------------------------------------------

const hsv_shift = 12;

/// `sdiv_table[v] = saturate_cast<int>((255 << 12) / (1. * v))`, 0 for v = 0,
/// and `hdiv_table180[d] = saturate_cast<int>((180 << 12) / (6. * d))`.
const sdiv_table = blk: {
    var t: [256]i32 = undefined;
    t[0] = 0;
    for (1..256) |v| t[v] = @intCast(@divTrunc(2 * (@as(i64, 255) << hsv_shift) + v, 2 * @as(i64, v)));
    break :blk t;
};
const hdiv_table = blk: {
    var t: [256]i32 = undefined;
    t[0] = 0;
    for (1..256) |d| t[d] = @intCast(@divTrunc(2 * (@as(i64, 180) << hsv_shift) + 6 * @as(i64, d), 2 * 6 * @as(i64, d)));
    break :blk t;
};

fn sdiv(v: i32) i64 {
    return sdiv_table[@intCast(v)];
}

fn hdiv(d: i32) i64 {
    return hdiv_table[@intCast(d)];
}

/// OpenCV `COLOR_BGR2HSV` on 8-bit pixels: hue 0..180, saturation and value 0..255.
pub fn hsvPixel(r8: u8, g8: u8, b8: u8) [3]u8 {
    const r: i32 = r8;
    const g: i32 = g8;
    const b: i32 = b8;
    const v = @max(r, @max(g, b));
    const vmin = @min(r, @min(g, b));
    const diff = v - vmin;
    if (diff == 0) return .{ 0, 0, @intCast(v) };
    const s = (@as(i64, diff) * sdiv(v) + (1 << (hsv_shift - 1))) >> hsv_shift;
    const numer: i64 = if (v == r) g - b else if (v == g) b - r + 2 * diff else r - g + 4 * diff;
    var h = (numer * hdiv(diff) + (1 << (hsv_shift - 1))) >> hsv_shift;
    if (h < 0) h += 180;
    return .{ @intCast(std.math.clamp(h, 0, 180)), @intCast(std.math.clamp(s, 0, 255)), @intCast(v) };
}

/// `table[idx]` for 16 lanes (`vx_lut`).
inline fn lut(table: *const [256]i32, idx: VU) VI {
    @setEvalBranchQuota(10_000);
    const ix: [B]u32 = idx;
    var out: VI = undefined;
    inline for (0..B) |i| out[i] = table[ix[i]];
    return out;
}

/// The vector loop of `RGB2HSV_b` on 16 pixels: max / min / diff per lane,
/// `s = (diff sdiv[v] + 2^11) >> 12`, the hue numerator chosen by the
/// `v == r` / `v == g` masks, `h = (num hdiv[diff] + 2^11) >> 12`, plus 180 when negative.
inline fn hsvBlock(px: [3]VU, out: *[3 * B]u8) void {
    const r: VI = @intCast(px[0]);
    const g: VI = @intCast(px[1]);
    const b: VI = @intCast(px[2]);
    const v = @max(r, @max(g, b));
    const vmin = @min(r, @min(g, b));
    const diff = v - vmin;
    const half: VI = @splat(1 << (hsv_shift - 1));
    const shift: @Vector(B, u5) = @splat(hsv_shift);
    const s = (diff * lut(&sdiv_table, @intCast(v)) + half) >> shift;
    const two: VI = @splat(2);
    const four: VI = @splat(4);
    const num = @select(i32, v == r, g - b, @select(i32, v == g, b - r + two * diff, r - g + four * diff));
    var h = (num * lut(&hdiv_table, @intCast(diff)) + half) >> shift;
    h = @select(i32, h < @as(VI, @splat(0)), h + @as(VI, @splat(180)), h);
    const hi: VB = @intCast(@min(@max(h, @as(VI, @splat(0))), @as(VI, @splat(255))));
    const si: VB = @intCast(@min(@max(s, @as(VI, @splat(0))), @as(VI, @splat(255))));
    const vi: VB = @intCast(v);
    storePlanes(.{ hi, si, vi }, out);
}

const HsvJob = struct {
    src: []const u8,
    w: usize,
    out: []u8,

    fn run(self: *const HsvJob, band: usize, y0: usize, y1: usize) void {
        _ = band;
        const s = self.src[y0 * self.w * 3 .. y1 * self.w * 3];
        const d = self.out[y0 * self.w * 3 .. y1 * self.w * 3];
        const n = (y1 - y0) * self.w;
        var i: usize = 0;
        while (i + B <= n) : (i += B) hsvBlock(loadPlanes(s[i * 3 ..][0 .. 3 * B]), d[i * 3 ..][0 .. 3 * B]);
        while (i < n) : (i += 1) d[i * 3 ..][0..3].* = hsvPixel(s[i * 3], s[i * 3 + 1], s[i * 3 + 2]);
    }
};

/// RGB -> HSV (`cv2.COLOR_BGR2HSV` on the BGR view of the same pixels).
pub fn rgbToHsv(a: Allocator, src: []const u8, w: usize, h: usize) Allocator.Error![]u8 {
    std.debug.assert(src.len >= w * h * 3);
    const out = try a.alloc(u8, w * h * 3);
    const job = HsvJob{ .src = src[0 .. w * h * 3], .w = w, .out = out };
    par.forBandsWork(h, 64, w * 3, &job, HsvJob.run);
    return out;
}

test "grey and hsv pixels" {
    try std.testing.expectEqual(@as(u8, 255), grayPixel(255, 255, 255));
    try std.testing.expectEqual(@as(u8, 76), grayPixel(255, 0, 0));
    try std.testing.expectEqual(@as(u8, 150), grayPixel(0, 255, 0));
    try std.testing.expectEqual(@as(u8, 29), grayPixel(0, 0, 255));
    try std.testing.expectEqual([3]u8{ 0, 255, 255 }, hsvPixel(255, 0, 0));
    try std.testing.expectEqual([3]u8{ 60, 255, 255 }, hsvPixel(0, 255, 0));
    try std.testing.expectEqual([3]u8{ 120, 255, 255 }, hsvPixel(0, 0, 255));
    try std.testing.expectEqual([3]u8{ 0, 0, 77 }, hsvPixel(77, 77, 77));
    try std.testing.expectEqual([3]u8{ 30, 255, 255 }, hsvPixel(255, 255, 0));
    const a = std.testing.allocator;
    var px: [17 * 3]u8 = undefined;
    for (&px, 0..) |*v, i| v.* = @intCast((i * 37) % 256);
    const g = try rgbToGray(a, &px, 17, 1);
    defer a.free(g);
    for (g, 0..) |v, i| try std.testing.expectEqual(grayPixel(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]), v);
    const hsv = try rgbToHsv(a, &px, 17, 1);
    defer a.free(hsv);
    try std.testing.expectEqual(hsvPixel(px[3], px[4], px[5]), hsv[3..6].*);
}

test "vector blocks agree with the pixel formulas" {
    const a = std.testing.allocator;
    const w = 83;
    const h = 7;
    const px = try a.alloc(u8, w * h * 3);
    defer a.free(px);
    var seed: u32 = 5;
    for (px) |*v| {
        seed = seed *% 1103515245 +% 12345;
        v.* = @truncate(seed >> 16);
    }
    // Flat and primary pixels exercise diff == 0 and every hue branch.
    px[0..9].* = .{ 50, 50, 50, 255, 0, 0, 0, 200, 0 };
    px[9..18].* = .{ 0, 0, 7, 0, 255, 255, 255, 0, 255 };
    const g = try rgbToGray(a, px, w, h);
    defer a.free(g);
    const hsv = try rgbToHsv(a, px, w, h);
    defer a.free(hsv);
    for (0..w * h) |i| {
        try std.testing.expectEqual(grayPixel(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]), g[i]);
        try std.testing.expectEqual(hsvPixel(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]), hsv[i * 3 ..][0..3].*);
    }
}
