# zigcv

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgbilby/zigcv/blob/main/LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.17%2B-f7a41d)](https://ziglang.org/)
[![accuracy](https://img.shields.io/badge/vs%20cv2-bit--exact-brightgreen.svg)](BENCHMARKS.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573956.svg)](https://doi.org/10.5281/zenodo.23573956)

OpenCV-equivalent image processing in pure Zig, bit-exact with `cv2`: the kernels are ported from OpenCV's own sources with their fixed-point and float arithmetic intact, so a pipeline moved from `cv2` to zigcv produces the same bytes. TIFF and PNG readers and writers included, no C libraries.

## Features

| Area | Operations |
|---|---|
| Filters | `GaussianBlur` on 8-bit and float32, both border modes, any kernel size or sigma |
| Geometry | `resize` (INTER_AREA, LINEAR, CUBIC, NEAREST); `warpPerspective`, `warpAffine`, `remap` (linear, cubic, nearest; replicate or constant borders) |
| Colour | BGR to grey and BGR to HSV with OpenCV's integer coefficients and tables |
| Thresholds | `threshold` in five modes, Otsu, `adaptiveThreshold` with the Gaussian mean |
| Morphology | erode, dilate, open, close; rectangular or elliptical elements, iterations |
| Arithmetic | `divide` with a scale |
| TIFF | Group 4 (CCITT T.6) writer byte-identical to libtiff's; LZW and Deflate writers; reader for none, G4, LZW, Deflate, PackBits |
| PNG | writer for 1-bit bilevel, 8-bit grey, grey+alpha, RGB, RGBA with adaptive filters and chunked Deflate on all cores; reader for the same |
| Parallelism | row bands on `std.Thread`, one per CPU, sized by work; identical bytes single-threaded (`ZIGCV_THREADS=1`) |

## Status

- Version 0.18.0, numbered with the Zig release it targets; the API may change between minor versions
- Every 8-bit operation is bit-exact with OpenCV 5.0; float32 blur within 1 ulp ([BENCHMARKS.md](BENCHMARKS.md))
- OpenCV 5 only: the 4.x releases warp and remap with other arithmetic, so their results differ by a few levels
- Images are plain slices, 8-bit grey or interleaved RGB with width, height and channel count; results come back through the caller's allocator
- No dependencies: the manifest's dependency list is empty and every file imports only the standard library

## Installation

```console
zig fetch --save git+https://github.com/mgbilby/zigcv
```

```zig
const zigcv = b.dependency("zigcv", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("zigcv", zigcv.module("zigcv"));
```

Requires Zig 0.17 or newer. Inside a larger repository the module can also be created from `src/root.zig` with `b.createModule`.

## Quick start

```zig
const cv = @import("zigcv");
// rgb: []const u8 of w * h * 3 interleaved bytes, as cv2 holds a BGR image
const gray = try cv.rgbToGray(a, rgb, w, h);                                  // cvtColor(BGR2GRAY)
const blurred = try cv.gaussianBlur(a, gray, w, h, 1, .{ .kx = 5, .ky = 5 });   // GaussianBlur((5, 5), 0)
const small = try cv.resize.resize(a, blurred, w, h, 1, w / 4, h / 4, .area);   // resize(INTER_AREA)
const bw = try cv.threshold.thresholdOtsu(a, blurred, 255, .binary);           // threshold(BINARY | OTSU)
const tif = try cv.tiff.encodeG4(a, bw.img, w, h, 300, 0);                     // 1-bit CCITT G4 TIFF at 300 dpi
const png = try cv.png.encode(a, bw.img, w, h, 1, .{ .bit_depth = 1, .filter = .none, .dpi = 300 });
```

Every function takes the allocator first, then the source slice with its width, height and channel count, and returns a newly allocated result; free each with the same allocator.

## API tour

| Task | Call |
|---|---|
| Blur | `cv.gaussianBlur(a, src, w, h, ch, .{ .kx = 5, .ky = 5, .sigma_x = 0, .border = .reflect101 })`; `cv.gaussianBlurF32` for float32 |
| Resize | `cv.resize.resize(a, src, sw, sh, ch, dw, dh, .area)`; `.nearest`, `.linear`, `.cubic` |
| Grey, HSV | `cv.rgbToGray(a, rgb, w, h)`, `cv.rgbToHsv(a, rgb, w, h)` |
| Threshold, Otsu | `cv.threshold.threshold(a, gray, 127, 255, .binary)`; `cv.threshold.thresholdOtsu(a, gray, 255, .binary)` gives `.thresh` and `.img` |
| Adaptive threshold | `cv.adaptiveThresholdGaussian(a, gray, w, h, 255, false, 31, 15)` |
| Morphology | `const k = try cv.morph.Kernel.ellipse(a, 5);` then `cv.erode`, `cv.dilate`, `cv.morphOpen`, `cv.morphClose(a, src, w, h, k, iterations)` |
| Divide | `cv.divide(a, num, den, 255.0)` |
| Warps | `cv.warpPerspective(a, src, sw, sh, ch, m3x3, dw, dh, .{ .interp = .cubic, .border = .replicate })`; `cv.warpAffine` with a 2x3 matrix; `cv.remap(a, src, sw, sh, ch, map_x, map_y, dw, dh, .{})` |
| TIFF | `cv.tiff.encodeG4(a, bw, w, h, dpi, rows_per_strip)`; `cv.tiff.encode8(a, data, w, h, ch, .{ .dpi = 300, .compression = .lzw })`; `cv.tiff.decode(a, bytes)` |
| PNG | `cv.png.encode(a, data, w, h, ch, .{ .dpi = 300 })`; `.{ .bit_depth = 1, .filter = .none }` for a B&W page; `cv.png.decode(a, bytes)` |
| Decoding limits | both decoders check every size product, refuse images above `max_pixels` (2^28) with `error.TooLarge` and keep runtime safety on in ReleaseFast |
| Threads | `cv.par.thread_override = 1;` or `ZIGCV_THREADS` in the host program |

`zig build docs` renders the per-function documentation.

## Operations and their OpenCV sources

| zigcv | OpenCV call | OpenCV source | Numerics |
|---|---|---|---|
| `gauss.blurU8` | `GaussianBlur` on 8-bit | `smooth.dispatch.cpp`, `smooth.simd.hpp`, `fixedpoint.inl.hpp` | `getGaussianKernelBitExact` taps, error-diffused to 8 fractional bits; 16-bit row sums, 32-bit column sums, `(s + 2^15) >> 16`; each row is filtered from a border-extended copy (`FilterEngine`'s row buffer and border table), so the border columns and kernels wider than the image take the vector loop |
| `gauss.blurF32` | `GaussianBlur` on float32 | `filter.simd.hpp` (`RowVec_32f`, `SymmRowSmallVec_32f`, `SymmColumnVec_32f`, `SymmColumnSmallVec_32f`) | float taps; fused multiply-add in the 8-lane vector region, plain multiply-add in each row's scalar tail; rows border-extended as above |
| `gauss.ksizeFromSigma` | `createGaussianKernels` | `smooth.dispatch.cpp` | `cvRound(sigma * (3 or 4) * 2 + 1) \| 1` |
| `resize.resize` `.area` | `resize(INTER_AREA)` | `resize.cpp` (`resizeAreaFast_`, `ResizeAreaFastVec`, `resizeArea_`, `computeResizeAreaTab`) | integer scale: block sum, `(s + 2) >> 2` for 2x2, `cvRound(s * 1/area)` else; fractional: float coverage tables in OpenCV's order |
| `resize.resize` `.linear` | `resize(INTER_LINEAR)` | `resize.cpp` (`resizeGeneric_Invoker`, `HResizeLinear` with the per-pixel gathers of `HResizeLinearVec_8u32s`, `VResizeLinearVec_32s8u`) | 11-bit fixed point, `((b0 (S0 >> 4)) >> 16 + (b1 (S1 >> 4)) >> 16 + 2) >> 2` as 16-bit high products; x clamped at the borders, y rows clamped and reused between output rows; 2x downscale routed to `INTER_AREA` as OpenCV does |
| `resize.resize` `.cubic` | `resize(INTER_CUBIC)` | `resize.cpp` (`resizeGeneric_Invoker`, `HResizeCubic` with per-pixel word gathers, `VResizeCubicVec_32s8u`) | A = -0.75 in 11-bit fixed point; vertical pass in float32 without FMA (the baseline SSE kernel, 16 lanes), integer cast on the scalar tail |
| `resize.resize` `.nearest` | `resize(INTER_NEAREST)` | `resize.cpp` (`resizeNN`) | `floor(x * (1 / inv_scale))` |
| `color.rgbToGray` | `cvtColor(BGR2GRAY)` | `color_rgb.simd.hpp` (`RGB2Gray<uchar>`) | 15-bit coefficients 9798 / 19235 / 3735, `(s + 2^14) >> 15`; the per-pixel loop compiles to the vector loop's interleaved loads and 16-bit dot products |
| `color.rgbToHsv` | `cvtColor(BGR2HSV)` | `color_hsv.simd.hpp` (`RGB2HSV_b`, its vector loop) | 12-bit `sdiv` / `hdiv` tables looked up per lane, hue 0..180 chosen by the `v == r` / `v == g` lane masks, 16 pixels per step |
| `threshold.otsu` | `threshold(THRESH_OTSU)` | `thresh.cpp` (`getThreshVal_Otsu_8u`) | the same double-precision scan and `FLT_EPSILON` guards |
| `threshold.threshold` | `threshold` | `thresh.cpp` | binary, inverse, trunc, tozero |
| `threshold.adaptiveGaussian` | `adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C)` | `thresh.cpp` | mean = float blur of the float copy with `BORDER_REPLICATE`, rounded to 8 bits; `idelta` = ceil / floor of delta |
| `arith.divide` | `divide(a, b, scale)` | `arithm.simd.hpp` (`div8u`) | float32 `a * scale / b`, round half to even, 0 where `b == 0` |
| `morph.erode` / `dilate` / `open` / `close` | `erode`, `dilate`, `morphologyEx` with `getStructuringElement` rect / ellipse, `iterations` | `morph.dispatch.cpp`, `morph.simd.hpp` (`MorphRowVec`, `MorphColumnVec`) | min / max over the element, pixels outside the image ignored; row extremes per distinct element row width, column extremes with the accumulator kept in registers (two rows per pass for rectangles); one banded pass for small kernels, log-step passes for large ones |
| `warp.warpPerspective` `.linear` / `.nearest` | `warpPerspective(INTER_LINEAR / NEAREST)` | `warp_kernels.simd.hpp` (`warpPerspectiveLinearInvoker`, `warpPerspectiveNearestInvoker`), `warp_common.*.hpp` | float32 matrix, fused multiply-add and true division per 16-pixel block, the compiler-contracted scalar expression in the tail; the taps of 16 pixels gathered into channel planes (`pixbuf`), `p0 + f (p1 - p0)` fused in 16 lanes; round half to even |
| `warp.warpPerspective` `.cubic` | `warpPerspective(INTER_CUBIC)` | `imgwarp.cpp` (`genericWarp`), `warp_kernels.simd.hpp` (`bicubicVec`, `bicubicCoeffs`, `FETCH_INLIERS`) | float32 matrix evaluated in double per pixel; `bicubicWeights` and the fused row / column chains of the vector kernel over 16-pixel channel planes |
| `warp.warpAffine` | `warpAffine` | same kernels | `invertAffineTransform`, then as above without the division |
| `warp.remap` | `remap` with float maps | `warp_kernels.simd.hpp` (`remapLinearInvoker`, `remapNearestInvoker`) | map rows read in place, interpolation as the warps |
| `tiff.encodeG4` / `decode` | `cv2.imwrite` / `imread` TIFF, Pillow `group4`, `tiff_lzw`, `tiff_adobe_deflate`, `packbits` | ITU-T T.4 / T.6, TIFF 6.0 | Group 4 codestream byte-identical to libtiff's; LZW with early change; Deflate through `std.compress.flate`; horizontal predictor and FillOrder 2 on read |
| `png.encode` / `decode` | `cv2.imwrite` / `imread` PNG, Pillow `PNG` | PNG (ISO/IEC 15948), RFC 1950 / 1951 | filters None, Sub, Up, Average, Paeth as specified, adaptive choice by the minimum sum of absolute differences on every eighth row of a tall image; one zlib stream from independent Deflate chunks of 128 KB of rows, each ended on a byte boundary, Adler-32 and CRC-32 combined; `huffman` is a dynamic Huffman block per chunk whose only matches are runs of a repeated byte (length-limited code lengths from the chunk's histogram, the block's size known before it is written, stored blocks when smaller); 1-bit packing by vector compare; the same bytes at any thread count |

## Build and test

```console
zig build test                   # unit tests of every operation
zig build docs                   # API documentation in zig-out/docs
zig fmt --check src build.zig
```

## Works with

| Need | Package | Notes |
|---|---|---|
| JPEG, BMP, GIF, QOI, TGA and other formats | [zigimg](https://github.com/zigimg/zigimg) (MIT) or [zignal](https://github.com/arrufat/zignal) (MIT) | pure Zig; decode to an 8-bit grey or RGB slice and hand it to zigcv. zigimg reads TIFF but does not write it; zigcv does both |
| JPEG at libjpeg-turbo speed | the system's `libjpeg` through `@cImport` | the usual choice for photos of 18 MP and up |
| Feature detection, Hough lines, drawing, matrices | [zignal](https://github.com/arrufat/zignal) | generic float formulations; not OpenCV-exact, which is why zigcv does not reuse them for its kernels |
| Runtime SIMD dispatch | [oma](https://github.com/ATTron/oma) (MIT) | one binary that picks AVX2 or AVX-512 kernels at start; a candidate for a later release |
| Benchmarks | [zBench](https://github.com/hendriknielaender/zBench) (MIT) | a harness for timing the kernels; the figures in BENCHMARKS.md were taken with a hand-written one |

The pixel struct layouts in `pixel.zig` follow zigimg's, so a zigimg `rgb24` or `grayscale8` buffer can be viewed as zigcv input without copying.

## Scope and non-goals

- Reproducing OpenCV's results, not improving on them: a different rounding is a bug here even when it is closer to the ideal
- 8-bit grey and interleaved RGB, plus float32 where OpenCV's path is float; no 16-bit or planar images
- No codecs besides TIFF and PNG, no I/O beyond byte slices, no GUI, no C API

## Contributing

- Each new operation names its OpenCV source file in the table above and ships with a test against `cv2` output on real and synthetic images
- Keep the vector and scalar paths of a kernel separate when OpenCV's round differently; tests must pass with `ZIGCV_THREADS=1` and with the default thread count
- Run `zig fmt` and `zig build test` before opening a pull request

## License

| File | Covers |
|---|---|
| [LICENSE](LICENSE) | zigcv's own code, MIT |
| [LICENSE-APACHE](LICENSE-APACHE) | the Apache License 2.0 under which OpenCV is distributed, included because the kernels are derivative works of OpenCV sources |
| [NOTICE](NOTICE) | which zigcv file derives from which OpenCV file and under which licence (Apache-2.0, or the legacy 3-clause BSD header some OpenCV files carry, reproduced there), OpenCV's copyright notices, the zigimg, zignal and zlib notices |

Every ported source file states its OpenCV origin and that it was modified. Binary distributions ship NOTICE and LICENSE-APACHE alongside LICENSE.
