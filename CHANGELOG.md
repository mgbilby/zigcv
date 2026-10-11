# Changelog

All notable changes to zigcv are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- PNG writer (`png.zig`): 8-bit grey, grey+alpha, RGB and RGBA, bilevel pages packed to 1 bit per pixel, the five filters and the adaptive choice as vector kernels, zlib level 1 / 6 / 9 or stored blocks, 128 KB row chunks on every CPU with combined Adler-32 and CRC-32, pHYs for the DPI; the output is the same at any thread count
- `png.Compression.huffman`: zigcv's own Deflate coder, a dynamic Huffman block per chunk with repeated-byte runs as its only matches, the block's size known before it is written and a stored block when that is smaller; faster than a Huffman-only zlib pass on filtered rows and far smaller on flat ones
- `png.adler32Update`: a vector Adler-32 (32-byte lanes, folded every 4 KB)
- PNG reader for non-interlaced 1/2/4/8-bit greyscale, grey+alpha, RGB and RGBA files with the DPI from pHYs
- `png.crc32Update` and `png.crc32Combine`: slicing-by-8 CRC-32 and the combination of two CRCs

### Changed

- warpPerspective, warpAffine and remap run as OpenCV's vector invokers do: 16 destination pixels per step, each pixel's taps fetched as 32-bit words and split into channel planes in registers, interpolated in 16-lane float vectors; remap reads its map rows in place
- resize INTER_LINEAR and INTER_CUBIC follow `resizeGeneric_Invoker` (horizontally resized rows kept while the next rows reuse them) with per-pixel word gathers in both horizontal kernels, the 16-bit high-product vertical linear kernel and a 16-lane vertical cubic kernel
- BGR2GRAY is the plain per-pixel loop, compiled to `RGB2Gray`'s interleaved loads and 16-bit dot products; BGR2HSV is the `RGB2HSV_b` vector block with per-lane table lookups
- erode, dilate, open and close keep the column accumulator in registers across the kernel rows (`MorphColumnVec`), produce two rows per pass for rectangular kernels, read the source directly for kernel rows of half-width zero and work on longer row chunks
- The Gaussian blur's row pass filters a border-extended copy of each row, the border pixels tabulated once per blur as `FilterEngine` does, so the border columns and kernels wider than the image run in the vector loop instead of one tap at a time (the quarter-size background blur of a small region was all border before); a unit test holds both engines to the per-sample formulas
- Results are unchanged: every operation stays bit-exact with OpenCV 5.0
- Row bands are sized by work as well as rows (`par.forBandsWork`, `par.bandCountWork`, `par.min_band_work`): an image under 256 KiB of work runs on the calling thread and larger ones get a band per 256 KiB at most, so the morphology passes of a large element over a small image no longer spawn hundreds of threads; results are unchanged

## [0.18.0]

### Added

- GaussianBlur on 8-bit and float32 images, bit-exact with OpenCV's `getGaussianKernelBitExact` path
- resize with INTER_AREA, INTER_LINEAR, INTER_CUBIC and INTER_NEAREST
- warpPerspective, warpAffine and remap with linear, cubic and nearest sampling
- BGR to grey and BGR to HSV conversion
- threshold in five modes, Otsu's threshold, adaptiveThreshold with the Gaussian mean
- erode, dilate, open and close with rectangular and elliptical elements
- divide with scale
- TIFF writer (Group 4, LZW, Deflate, none) and reader (plus PackBits, predictor, FillOrder 2)
- Row-band threading with single-threaded override
