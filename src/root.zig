//! zigcv: OpenCV-equivalent image processing in pure Zig for 8-bit grey and
//! interleaved RGB buffers. Every operation follows the exact numerics of the
//! matching `cv2` call (see README.md for the equivalence and accuracy table).
//!
//! Buffers are plain slices with a width, height and channel count; nothing
//! here owns pixels beyond the result it returns through the caller's allocator.
const std = @import("std");

pub const version = "0.18.0";

pub const util = @import("util.zig");
pub const par = @import("par.zig");
pub const pixel = @import("pixel.zig");
pub const gauss = @import("gauss.zig");
pub const resize = @import("resize.zig");
pub const color = @import("color.zig");
pub const threshold = @import("threshold.zig");
pub const arith = @import("arith.zig");
pub const warp = @import("warp.zig");
pub const morph = @import("morph.zig");
pub const tiff = @import("tiff.zig");
pub const png = @import("png.zig");

pub const Border = util.Border;
pub const Interp = resize.Interp;

pub const gaussianBlur = gauss.blurU8;
pub const gaussianBlurF32 = gauss.blurF32;
pub const rgbToGray = color.rgbToGray;
pub const rgbToHsv = color.rgbToHsv;
pub const otsu = threshold.otsu;
pub const adaptiveThresholdGaussian = threshold.adaptiveGaussian;
pub const divide = arith.divide;
pub const warpPerspective = warp.warpPerspective;
pub const warpAffine = warp.warpAffine;
pub const remap = warp.remap;
pub const erode = morph.erode;
pub const dilate = morph.dilate;
pub const morphOpen = morph.open;
pub const morphClose = morph.close;

test {
    std.testing.refAllDecls(@This());
}
