//! Pixel structs for typed views of interleaved 8-bit buffers.
//!
//! `Rgb24`, `Bgr24` and `Grayscale8` follow `src/color.zig` of zigimg
//! (https://github.com/zigimg/zigimg, MIT License, Copyright (c) 2019-2021
//! zigimg developers): the same field names and `extern` layout, so a
//! `[]const Rgb24` view of zigcv data is a zigimg `Rgb24` slice. zigimg's
//! colour conversions are not used: its grey weights are Rec. 709 floats,
//! while the operations here need OpenCV's integer 0.299/0.587/0.114.
const std = @import("std");

pub const Rgb24 = extern struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn from(r: u8, g: u8, b: u8) Rgb24 {
        return .{ .r = r, .g = g, .b = b };
    }
};

pub const Bgr24 = extern struct {
    b: u8,
    g: u8,
    r: u8,

    pub fn from(r: u8, g: u8, b: u8) Bgr24 {
        return .{ .r = r, .g = g, .b = b };
    }
};

pub const Grayscale8 = extern struct {
    value: u8,
};

/// View interleaved RGB bytes as pixels.
pub fn rgbView(data: []const u8) []const Rgb24 {
    return std.mem.bytesAsSlice(Rgb24, data[0 .. data.len - data.len % 3]);
}

pub fn rgbViewMut(data: []u8) []Rgb24 {
    return std.mem.bytesAsSlice(Rgb24, data[0 .. data.len - data.len % 3]);
}

test "views share the bytes" {
    var data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const px = rgbView(&data);
    try std.testing.expectEqual(@as(usize, 2), px.len);
    try std.testing.expectEqual(Rgb24.from(4, 5, 6), px[1]);
    rgbViewMut(&data)[0].g = 9;
    try std.testing.expectEqual(@as(u8, 9), data[1]);
    try std.testing.expectEqual(@as(usize, 3), @sizeOf(Bgr24));
}
