// Derived from OpenCV (https://github.com/opencv/opencv): modules/imgproc/src/thresh.cpp.
// Licence of the sources: BSD-3-Clause; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! `cv2.threshold` (fixed and Otsu) and `cv2.adaptiveThreshold` with OpenCV's
//! exact decisions: Otsu's search in double as `getThreshVal_Otsu_8u`, the
//! adaptive Gaussian mean through the float filter engine.
const std = @import("std");
const Allocator = std.mem.Allocator;
const gauss = @import("gauss.zig");
const util = @import("util.zig");
const par = @import("par.zig");

pub const Type = enum { binary, binary_inv, trunc, tozero, tozero_inv };

/// Otsu's threshold of an 8-bit image (`cv2.threshold(..., THRESH_OTSU)[0]`).
const HistJob = struct {
    src: []const u8,
    hists: *[par.max_threads][256]u32,

    fn run(self: *const HistJob, band: usize, start: usize, end: usize) void {
        var h0: [256]u32 = @splat(0);
        var h1: [256]u32 = @splat(0);
        var h2: [256]u32 = @splat(0);
        var h3: [256]u32 = @splat(0);
        var i = start;
        while (i + 4 <= end) : (i += 4) {
            h0[self.src[i]] += 1;
            h1[self.src[i + 1]] += 1;
            h2[self.src[i + 2]] += 1;
            h3[self.src[i + 3]] += 1;
        }
        while (i < end) : (i += 1) h0[self.src[i]] += 1;
        for (&self.hists[band], h0, h1, h2, h3) |*d, a, b, c, e| d.* = a + b + c + e;
    }
};

/// Histogram of an 8-bit image (threaded, four interleaved counters per band).
pub fn histogram(src: []const u8) [256]u32 {
    var hists: [par.max_threads][256]u32 = undefined;
    for (&hists) |*h| @memset(h, 0);
    const job = HistJob{ .src = src, .hists = &hists };
    par.forBands(src.len, 1 << 18, &job, HistJob.run);
    var hist: [256]u32 = @splat(0);
    for (hists) |h| for (&hist, h) |*d, v| {
        d.* += v;
    };
    return hist;
}

pub fn otsu(src: []const u8) u8 {
    const hist = histogram(src);
    const scale = 1.0 / @as(f64, @floatFromInt(src.len));
    var mu: f64 = 0;
    for (hist, 0..) |n, i| mu += @as(f64, @floatFromInt(i)) * @as(f64, @floatFromInt(n));
    mu *= scale;
    var mu1: f64 = 0;
    var q1: f64 = 0;
    var max_sigma: f64 = 0;
    var max_val: u8 = 0;
    const eps = std.math.floatEps(f32);
    for (hist, 0..) |n, i| {
        const p_i = @as(f64, @floatFromInt(n)) * scale;
        mu1 *= q1;
        q1 += p_i;
        const q2 = 1.0 - q1;
        if (@min(q1, q2) < eps or @max(q1, q2) > 1.0 - eps) continue;
        mu1 = (mu1 + @as(f64, @floatFromInt(i)) * p_i) / q1;
        const mu2 = (mu - q1 * mu1) / q2;
        const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_val = @intCast(i);
        }
    }
    return max_val;
}

/// `cv2.threshold(src, thresh, maxval, type)[1]`.
pub fn threshold(a: Allocator, src: []const u8, thresh: u8, maxval: u8, kind: Type) Allocator.Error![]u8 {
    const out = try a.alloc(u8, src.len);
    switch (kind) {
        .binary => for (out, src) |*o, v| {
            o.* = if (v > thresh) maxval else 0;
        },
        .binary_inv => for (out, src) |*o, v| {
            o.* = if (v > thresh) 0 else maxval;
        },
        .trunc => for (out, src) |*o, v| {
            o.* = @min(v, thresh);
        },
        .tozero => for (out, src) |*o, v| {
            o.* = if (v > thresh) v else 0;
        },
        .tozero_inv => for (out, src) |*o, v| {
            o.* = if (v > thresh) 0 else v;
        },
    }
    return out;
}

/// `cv2.threshold(src, 0, maxval, type | THRESH_OTSU)`: the threshold and the image.
pub fn thresholdOtsu(a: Allocator, src: []const u8, maxval: u8, kind: Type) Allocator.Error!struct { thresh: u8, img: []u8 } {
    const t = otsu(src);
    return .{ .thresh = t, .img = try threshold(a, src, t, maxval, kind) };
}

/// `cv2.adaptiveThreshold(src, maxval, ADAPTIVE_THRESH_GAUSSIAN_C, type, block, delta)`
/// for `THRESH_BINARY` / `THRESH_BINARY_INV`.
pub fn adaptiveGaussian(a: Allocator, src: []const u8, w: usize, h: usize, maxval: u8, inverse: bool, block: usize, delta: f64) Allocator.Error![]u8 {
    const mean = try gauss.blurU8ViaFloat(a, src, w, h, 1, .{ .kx = block, .ky = block, .border = .replicate });
    defer a.free(mean);
    const idelta: i32 = @intCast(if (inverse) util.cvFloor(delta) else util.cvCeil(delta));
    const out = try a.alloc(u8, w * h);
    for (out, src, mean) |*o, s, m| {
        const d = @as(i32, s) - @as(i32, m);
        o.* = if (inverse) (if (d <= -idelta) maxval else 0) else (if (d > -idelta) maxval else 0);
    }
    return out;
}

test "otsu splits two modes" {
    var img: [1000]u8 = undefined;
    for (&img, 0..) |*v, i| v.* = if (i % 3 == 0) @intCast(40 + i % 7) else @intCast(200 + i % 11);
    const t = otsu(&img);
    try std.testing.expect(t >= 46 and t < 200);
    const a = std.testing.allocator;
    const r = try thresholdOtsu(a, &img, 255, .binary);
    defer a.free(r.img);
    try std.testing.expectEqual(t, r.thresh);
    try std.testing.expectEqual(@as(u8, 0), r.img[0]);
    try std.testing.expectEqual(@as(u8, 255), r.img[1]);
    const tr = try threshold(a, &img, 100, 255, .trunc);
    defer a.free(tr);
    try std.testing.expectEqual(@as(u8, 100), tr[1]);
    const flat: [64]u8 = @splat(9);
    try std.testing.expectEqual(@as(u8, 0), otsu(&flat));
}

test "adaptive threshold marks dark text on bright paper" {
    const a = std.testing.allocator;
    const w = 40;
    const h = 20;
    var img: [w * h]u8 = @splat(230);
    for (5..15) |y| for (18..22) |x| {
        img[y * w + x] = 20;
    };
    const ink = try adaptiveGaussian(a, &img, w, h, 255, true, 11, 15);
    defer a.free(ink);
    try std.testing.expectEqual(@as(u8, 255), ink[10 * w + 20]);
    try std.testing.expectEqual(@as(u8, 0), ink[2 * w + 2]);
    const bright = try adaptiveGaussian(a, &img, w, h, 255, false, 11, 15);
    defer a.free(bright);
    try std.testing.expectEqual(@as(u8, 0), bright[10 * w + 20]);
    try std.testing.expectEqual(@as(u8, 255), bright[2 * w + 2]);
}
