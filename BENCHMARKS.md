# zigcv benchmarks and accuracy

All figures from one machine: Intel Core i5, 8 threads, OpenCV 5.0 without IPP, `zig build -Doptimize=ReleaseFast`, an 18 MP scanned page (4896x3672) or its quarter where the row says so. Each row is the better of two runs, each kernel called the same way on both sides; time the calls in your own harness to reproduce them.

## Kernels against OpenCV

zigcv is faster on 18 of 23 kernels; the five above 1.0 are 1 to 4 ms operations where starting threads outweighs the work.

| Kernel | OpenCV ms | zigcv ms | zigcv / OpenCV |
|---|---|---|---|
| cvtColor BGR2GRAY | 8.0 | 5.1 | 0.64 |
| cvtColor BGR2HSV | 24.4 | 8.8 | 0.36 |
| GaussianBlur 5x5 gray | 9.3 | 4.7 | 0.51 |
| GaussianBlur 5x5 BGR | 30.9 | 23.2 | 0.75 |
| GaussianBlur 61x61 gray quarter | 7.2 | 3.1 | 0.43 |
| GaussianBlur 307x307 gray quarter | 236.5 | 41.6 | 0.18 |
| resize INTER_AREA x0.25 gray | 3.4 | 1.3 | 0.38 |
| resize INTER_AREA x0.25 BGR | 9.4 | 3.6 | 0.38 |
| resize INTER_AREA 1000 px BGR | 17.8 | 11.7 | 0.66 |
| resize INTER_LINEAR x4 gray | 2.8 | 3.3 | 1.18 |
| resize INTER_NEAREST x4 mask | 2.7 | 3.0 | 1.11 |
| resize INTER_CUBIC x0.5 gray | 2.3 | 3.7 | 1.61 |
| threshold OTSU gray | 16.9 | 5.7 | 0.34 |
| adaptiveThreshold GAUSSIAN 31 15 gray | 100.0 | 73.6 | 0.74 |
| morphologyEx CLOSE ellipse 5 gray quarter | 0.3 | 1.4 | 4.67 |
| morphologyEx CLOSE ellipse 307 gray quarter | 854.7 | 187.6 | 0.22 |
| erode rect 15x15 mask | 4.6 | 4.1 | 0.89 |
| dilate rect 9x9 mask | 3.9 | 4.1 | 1.05 |
| warpPerspective INTER_CUBIC BGR | 125.2 | 43.9 | 0.35 |
| warpPerspective INTER_LINEAR BGR | 41.2 | 29.6 | 0.72 |
| remap INTER_LINEAR BGR | 41.0 | 30.7 | 0.75 |
| divide scale=255 gray | 6.8 | 4.9 | 0.72 |
| lighting estimate (quarter close + blur + up) | 15.4 | 10.0 | 0.65 |

## File writers

zigcv's PNG writer is the fastest of the three on every page kind, mostly through chunked Deflate on all cores; single-threaded it is level with the Rust png crate on a B&W page and behind it on grey and colour pages.

| Writer, 18 MP page | Pillow ms | Rust png / own G4 ms | zigcv ms | zigcv single-threaded ms |
|---|---|---|---|---|
| PNG, B&W (1-bit) | 56.9 | 6.8 | 2.4 | 3.0 |
| PNG, grey | 167.5 | 19.9 | 7.6 | 22.2 |
| PNG, colour | 461.6 | 58.4 | 19.9 | 80.6 |
| TIFF, B&W Group 4 | 82.4 | 12.6 | 11.6 | 11.6 |

## Accuracy against OpenCV 5.0

Every 8-bit operation matches `cv2` bit for bit on an 18 MP page, a 1200x900 crop and synthetic images; the float32 blur is within 1 ulp; the file formats decode pixel-exact in Pillow, OpenCV and poppler.

| Operation | Images | Result |
|---|---|---|
| GaussianBlur 8-bit (3x3 .. 307x307, sigma-sized, (3,9), REFLECT_101 and REPLICATE, grey and RGB) | 18 MP page, its quarter, 1200x900 crop, synthetic | bit-exact |
| GaussianBlur float32 (5x5, 31x31, sigma 1.5, (3,9); grey and 3-channel) | crop, synthetic | bit-exact except at most 1 ulp on rare samples of a row's scalar tail |
| resize INTER_AREA (x0.25, x0.5, 1000 px, x1/3 x1/5, one axis up) | 18 MP, crop, synthetic, grey and RGB | bit-exact |
| resize INTER_LINEAR (x4, x0.5, 0.61 x 0.37, 1.5 x 1.67) | 18 MP quarter, crop, synthetic | bit-exact |
| resize INTER_CUBIC (x0.5, x2, 1.4 x 0.75) | crop, synthetic | bit-exact (the pip wheel's IPP cubic upscale differs by 1 on a few pixels) |
| resize INTER_NEAREST (x4, 10/13 x 7/9) | crop, synthetic | bit-exact |
| BGR2GRAY, BGR2HSV | 18 MP, crop, synthetic | bit-exact |
| threshold, Otsu threshold value and image | 18 MP, crop, synthetic | bit-exact |
| adaptiveThreshold GAUSSIAN_C (31/15, 11/7, 31/15.5, 5/2; BINARY and INV) | crop, synthetic | bit-exact |
| divide scale=255 | crop | bit-exact |
| erode / dilate / open / close (rect 3, 5 x2, 9, 15; ellipse 5, 7 x2, 9, 27, 101) | crop grey and mask, synthetic | bit-exact |
| warpPerspective INTER_LINEAR and INTER_CUBIC, BORDER_REPLICATE and CONSTANT | crop, synthetic | bit-exact |
| remap INTER_LINEAR float maps, BORDER_CONSTANT (RGB and grey) | crop, synthetic | bit-exact |
| TIFF Group 4 | crop as a B&W page | strips byte-identical to libtiff's (Pillow) at Pillow's rows per strip; Pillow and poppler (`pdfimages`) decode it pixel-exact; Pillow's multi-strip file reads back pixel-exact |
| TIFF 8-bit grey / RGB, none / LZW / Deflate | crop | Pillow reads zigcv's files pixel-exact with the DPI; zigcv reads Pillow's LZW, Adobe Deflate and PackBits and OpenCV's LZW-with-predictor pixel-exact |
| PNG 1-bit B&W, 8-bit grey, 8-bit RGB | the page of a synthetic 18 MP spread as B&W, grey and colour, synthetic images | Pillow and OpenCV decode zigcv's files pixel-exact to the TIFF of the same page, with the DPI; every filter, compression and chunk count reads back exactly in the unit tests |

## Method

- OpenCV is called from Python with `OPENCV_IPP=disabled`, the behaviour of the distribution packages; Intel's IPP replaces a few kernels (cubic upscaling) with non-bit-exact ones
- Threaded rows use one thread per CPU on both sides; `ZIGCV_THREADS=1` gives the single-thread zigcv figures
- The accuracy rows come from a 105-case suite that runs each operation through zigcv and through `cv2` on the same inputs and compares the output arrays
