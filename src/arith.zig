// Derived from OpenCV (https://github.com/opencv/opencv): modules/core/src/arithm.simd.hpp.
// Licence of the sources: Apache-2.0; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! Per-pixel arithmetic with OpenCV's conventions (`arithm.simd.hpp`): 8-bit
//! operands are converted to float32, scaled, divided and rounded half to
//! even; a zero divisor gives 0.
const std = @import("std");
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const par = @import("par.zig");

const DivJob = struct {
    num: []const u8,
    den: []const u8,
    scale: f32,
    out: []u8,

    fn run(self: *const DivJob, band: usize, start: usize, end: usize) void {
        _ = band;
        const V = @Vector(8, f32);
        const vscale: V = @splat(self.scale);
        const zero: V = @splat(0);
        const top: V = @splat(255);
        var i = start;
        while (i + 8 <= end) : (i += 8) {
            const a: V = @floatFromInt(@as(@Vector(8, u8), self.num[i..][0..8].*));
            const b: V = @floatFromInt(@as(@Vector(8, u8), self.den[i..][0..8].*));
            const q = (a * vscale) / b;
            const r = @min(@max(util.roundEvenVec(8, q), zero), top);
            const picked = @select(f32, b == zero, zero, r);
            self.out[i..][0..8].* = @as(@Vector(8, u8), @intFromFloat(picked));
        }
        while (i < end) : (i += 1) {
            const b: f32 = @floatFromInt(self.den[i]);
            self.out[i] = if (b == 0) 0 else util.saturateU8F(@as(f32, @floatFromInt(self.num[i])) * self.scale / b);
        }
    }
};

/// `cv2.divide(num, den, scale=scale)` on 8-bit data of any channel count.
pub fn divide(a: Allocator, num: []const u8, den: []const u8, scale: f64) Allocator.Error![]u8 {
    std.debug.assert(num.len == den.len);
    const out = try a.alloc(u8, num.len);
    const job = DivJob{ .num = num, .den = den, .scale = @floatCast(scale), .out = out };
    par.forBands(num.len, 1 << 16, &job, DivJob.run);
    return out;
}

test "divide rounds like OpenCV and zero gives zero" {
    const a = std.testing.allocator;
    const num = [_]u8{ 200, 255, 1, 128, 3, 100, 7, 9, 10, 11 };
    const den = [_]u8{ 200, 0, 255, 255, 2, 101, 7, 8, 9, 13 };
    const q = try divide(a, &num, &den, 255);
    defer a.free(q);
    // 1 * 255 / 255 = 1; 3 * 255 / 2 = 382.5 -> saturates; 10 * 255 / 9 = 283.3 -> 255; 11 * 255 / 13 = 215.77 -> 216.
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 1, 128, 255, 252, 255, 255, 255, 216 }, q);
    const half = try divide(a, &num, &den, 0.5);
    defer a.free(half);
    try std.testing.expectEqual(@as(u8, 0), half[0]); // 200 * 0.5 / 200 = 0.5 -> rounds to even 0
    try std.testing.expectEqual(@as(u8, 0), half[3]); // 128 * 0.5 / 255 = 0.2509 -> 0
}
