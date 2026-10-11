// Derived from OpenCV (https://github.com/opencv/opencv): modules/core/include/opencv2/core/fast_math.hpp (cvRound), modules/core/src/copy.cpp (borderInterpolate).
// Licence of the sources: Apache-2.0; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! Rounding and border helpers shared by the zigcv operations, with OpenCV's
//! conventions: `cvRound` rounds half to even, `BORDER_REFLECT_101` is the
//! default border, `BORDER_REPLICATE` repeats the edge pixel.
const std = @import("std");

/// How pixels outside the image are read.
pub const Border = enum { reflect101, replicate };

/// Reflect-101 index (`-1 -> 1`, `n -> n - 2`).
pub fn reflect101(i_in: isize, n_in: usize) usize {
    const n: isize = @intCast(n_in);
    if (n == 1) return 0;
    var i = i_in;
    while (i < 0 or i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return @intCast(i);
}

pub fn borderIndex(i: isize, n: usize, border: Border) usize {
    return switch (border) {
        .reflect101 => reflect101(i, n),
        .replicate => @intCast(std.math.clamp(i, 0, @as(isize, @intCast(n)) - 1)),
    };
}

/// Round half to even (`cvRound` on a double, Python's `round`).
pub fn roundEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2.0) != 0.0) return r - std.math.sign(x);
    return r;
}

pub fn cvRound(x: f64) i64 {
    return @intFromFloat(roundEven(x));
}

pub fn cvFloor(x: f64) i64 {
    return @intFromFloat(@floor(x));
}

pub fn cvCeil(x: f64) i64 {
    return @intFromFloat(@ceil(x));
}

/// `cvRound` on a float, as OpenCV's SIMD `v_round` does it (nearest even).
pub fn cvRoundF(x: f32) i32 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2.0) != 0.0) return @intFromFloat(r - std.math.sign(x));
    return @intFromFloat(r);
}

/// Round a vector of floats half to even; `|x| < 2^22` (pixel sums), using the
/// 1.5 * 2^23 trick, which is exact in strict float mode.
pub inline fn roundEvenVec(comptime n: comptime_int, x: @Vector(n, f32)) @Vector(n, f32) {
    const magic: @Vector(n, f32) = @splat(12582912.0);
    return (x + magic) - magic;
}

/// Little-endian word of the `n` (1..4) bytes at `p`: the per-pixel gather
/// of the vector kernels, which fetch each pixel's taps as one word and split
/// the bytes out in registers.
pub inline fn readUpTo4(comptime n: usize, p: [*]const u8) u32 {
    return switch (n) {
        4 => std.mem.readInt(u32, p[0..4], .little),
        3 => @as(u32, std.mem.readInt(u16, p[0..2], .little)) | (@as(u32, p[2]) << 16),
        2 => std.mem.readInt(u16, p[0..2], .little),
        1 => p[0],
        else => unreachable,
    };
}

pub fn saturateU8(v: i64) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// `saturate_cast<uchar>(float)`: round half to even, then clamp.
pub fn saturateU8F(x: f32) u8 {
    return @intCast(std.math.clamp(cvRoundF(x), 0, 255));
}

test "rounding" {
    try std.testing.expectEqual(@as(i64, 2), cvRound(2.5));
    try std.testing.expectEqual(@as(i64, 4), cvRound(3.5));
    try std.testing.expectEqual(@as(i64, -2), cvRound(-2.5));
    try std.testing.expectEqual(@as(i32, 0), cvRoundF(0.5));
    try std.testing.expectEqual(@as(i32, 2), cvRoundF(1.5));
    try std.testing.expectEqual(@as(i32, 3), cvRoundF(2.7));
    const v = roundEvenVec(4, .{ 0.5, 1.5, 2.5, 254.49 });
    try std.testing.expectEqual(@Vector(4, f32){ 0, 2, 2, 254 }, v);
    try std.testing.expectEqual(@as(u8, 255), saturateU8F(300.0));
    try std.testing.expectEqual(@as(u8, 0), saturateU8F(-3.0));
}

test "borders" {
    try std.testing.expectEqual(@as(usize, 1), reflect101(-1, 10));
    try std.testing.expectEqual(@as(usize, 8), reflect101(10, 10));
    try std.testing.expectEqual(@as(usize, 0), borderIndex(-3, 10, .replicate));
    try std.testing.expectEqual(@as(usize, 9), borderIndex(12, 10, .replicate));
    try std.testing.expectEqual(@as(usize, 0), reflect101(5, 1));
}
