// Derived from OpenCV (https://github.com/opencv/opencv): modules/core/src/parallel.cpp (the row-band scheme only).
// Licence of the sources: Apache-2.0; see NOTICE and LICENSE-APACHE.
// Modified: translated to Zig, restructured around slices and the caller's
// allocator, OpenCV's dispatch, OpenCL, IPP and HAL layers removed.

//! Row-band parallelism: an operation is split into horizontal bands, one
//! `std.Thread` per band, like OpenCV's `parallel_for_` over rows.
const std = @import("std");

pub const max_threads = 16;

/// Set to use that many CPUs instead of the CPU count (a host reads its
/// `ZIGCV_THREADS` and sets this; 1 compares single-threaded timings).
pub var thread_override: usize = 0;

pub fn threadCount() usize {
    if (thread_override != 0) return @min(thread_override, max_threads);
    return std.math.clamp(std.Thread.getCpuCount() catch 1, 1, max_threads);
}

/// Call `func(ctx, band, y0, y1)` for bands of `[0, n)`, at most one band per
/// CPU and none shorter than `min_band` rows (except the last). Every band but
/// the last runs on its own thread, the last on the caller; when a thread
/// cannot be spawned its band runs inline.
pub fn forBands(n: usize, min_band: usize, ctx: anytype, comptime func: anytype) void {
    const bands = @min(threadCount(), @max(n / @max(min_band, 1), 1));
    if (bands <= 1) {
        func(ctx, 0, 0, n);
        return;
    }
    const per = (n + bands - 1) / bands;
    var handles: [max_threads]?std.Thread = @splat(null);
    var y: usize = 0;
    var k: usize = 0;
    while (y < n and k < max_threads) : (y += per) {
        const y1 = @min(y + per, n);
        if (y1 == n) {
            // The last band runs on the calling thread while the others work.
            func(ctx, k, y, y1);
        } else {
            handles[k] = std.Thread.spawn(.{}, func, .{ ctx, k, y, y1 }) catch blk: {
                func(ctx, k, y, y1);
                break :blk null;
            };
        }
        k += 1;
    }
    for (handles[0..k]) |h| if (h) |t| t.join();
}

/// How many bands `forBands(n, min_band, ...)` will use (to size per-band scratch).
pub fn bandCount(n: usize, min_band: usize) usize {
    return @min(threadCount(), @max(n / @max(min_band, 1), 1));
}

/// Bytes of work a band must hold before it is worth a thread of its own:
/// spawning and joining one costs about as much as this many byte operations,
/// so small images run inline and medium ones get fewer bands (like
/// OpenCV's `parallel_for_`, which sizes its stripes by the work too).
pub const min_band_work = 1 << 18;

/// The `min_band` for `forBands` that also holds `min_band_work` bytes of
/// work when every row costs `work_per_row` bytes.
pub fn minBandFor(min_band: usize, work_per_row: usize) usize {
    const rows_for_work = (min_band_work + @max(work_per_row, 1) - 1) / @max(work_per_row, 1);
    return @max(min_band, rows_for_work);
}

/// `forBands` with bands sized by work as well as rows: `work_per_row` is the
/// bytes each row costs (its output bytes, or more for heavy kernels). An
/// image under `min_band_work` bytes runs on the calling thread. The result
/// is the same as `forBands` with any band count, bands being independent.
pub fn forBandsWork(n: usize, min_band: usize, work_per_row: usize, ctx: anytype, comptime func: anytype) void {
    forBands(n, minBandFor(min_band, work_per_row), ctx, func);
}

/// How many bands `forBandsWork(n, min_band, work_per_row, ...)` will use.
pub fn bandCountWork(n: usize, min_band: usize, work_per_row: usize) usize {
    return bandCount(n, minBandFor(min_band, work_per_row));
}

test "bands cover the range once" {
    const Ctx = struct {
        hits: []u8,
        fn run(self: *@This(), band: usize, y0: usize, y1: usize) void {
            _ = band;
            for (self.hits[y0..y1]) |*h| h.* += 1;
        }
    };
    var hits: [1000]u8 = @splat(0);
    var ctx = Ctx{ .hits = &hits };
    forBands(hits.len, 8, &ctx, Ctx.run);
    for (hits) |h| try std.testing.expectEqual(@as(u8, 1), h);
    try std.testing.expect(bandCount(1000, 8) >= 1);
    try std.testing.expectEqual(@as(usize, 1), bandCount(10, 100));
}

test "work-sized bands run small images inline and cover large ones once" {
    const Ctx = struct {
        hits: []u8,
        bands_seen: [max_threads]bool = @splat(false),
        fn run(self: *@This(), band: usize, y0: usize, y1: usize) void {
            self.bands_seen[band] = true;
            for (self.hits[y0..y1]) |*h| h.* += 1;
        }
    };
    // 300 rows of 300 bytes: 90 KB, under min_band_work, so one band on this thread.
    try std.testing.expectEqual(@as(usize, 1), bandCountWork(300, 16, 300));
    var hits: [300]u8 = @splat(0);
    var ctx = Ctx{ .hits = &hits };
    forBandsWork(hits.len, 16, 300, &ctx, Ctx.run);
    for (hits) |h| try std.testing.expectEqual(@as(u8, 1), h);
    try std.testing.expect(ctx.bands_seen[0]);
    for (ctx.bands_seen[1..]) |b| try std.testing.expect(!b);
    // Each band holds at least min_band_work bytes: 1000 rows of 1000 bytes make at most 3 bands.
    try std.testing.expect(bandCountWork(1000, 8, 1000) <= 3);
    try std.testing.expect(bandCountWork(1000, 8, 1000) >= 1);
    try std.testing.expectEqual(@as(usize, 263), minBandFor(8, 1000));
    try std.testing.expectEqual(@as(usize, 300), minBandFor(300, 1000));
    // Wide rows keep the row minimum and the thread count decides.
    try std.testing.expectEqual(bandCount(4000, 8), bandCountWork(4000, 8, 1 << 16));
    var big: [4000]u8 = @splat(0);
    var bctx = Ctx{ .hits = &big };
    forBandsWork(big.len, 8, 1 << 16, &bctx, Ctx.run);
    for (big) |h| try std.testing.expectEqual(@as(u8, 1), h);
}
