//! PNG without libpng: a writer for 8-bit grey, grey+alpha, RGB and RGBA
//! images and for bilevel pages packed to one bit per pixel, and a reader
//! for the non-interlaced files such pipelines meet.
//!
//! The writer is built for speed on scanned pages:
//! - rows are filtered and deflated in chunks of about 128 KB, the chunks
//!   spread over the CPUs in bands (`par.zig`); every chunk is its own
//!   Deflate stream ended on a byte boundary, so the chunks concatenate into
//!   one zlib member whose Adler-32 is combined from the chunks' checksums
//!   (the scheme of zignal's PNG writer) and the bytes do not depend on the
//!   thread count;
//! - the five PNG filters, the bit packing and the Adler-32 are `@Vector`
//!   kernels; the adaptive choice scores a sampled row by the usual sum of
//!   absolute differences and reuses the choice until the next sample;
//! - `huffman` is zigcv's own Deflate coder: one dynamic Huffman block per
//!   chunk, literals plus runs of a repeated byte as distance-1 matches and
//!   no other match search, the block's size known before it is written and
//!   a stored block when that is smaller; `fast`, `default` and `best` are
//!   `std.compress.flate` at level 1, 6 and 9; `none` writes stored blocks;
//! - the pixels are read in place; per-band scratch comes from the page
//!   allocator (thread-safe) and only the result from the caller's allocator.
//!
//! The reader handles bit depths 1, 2, 4 and 8, colour types grey,
//! grey+alpha, RGB and RGBA, every filter, and the pHYs chunk for the DPI;
//! palettes, 16-bit samples and Adam7 interlacing are reported as unsupported.
const std = @import("std");
const Allocator = std.mem.Allocator;
const flate = std.compress.flate;
const par = @import("par.zig");

pub const Error = error{ NotPng, Malformed, Unsupported, TooLarge } || Allocator.Error;

/// Largest image `decode` accepts (width times height); larger ones are `error.TooLarge`.
pub const max_pixels: usize = 1 << 28;

/// A size product of values read from a file; overflow is `error.TooLarge`.
fn mulSize(x: usize, y: usize) Error!usize {
    return std.math.mul(usize, x, y) catch return error.TooLarge;
}

pub const signature = "\x89PNG\r\n\x1a\n";

pub fn isPng(bytes: []const u8) bool {
    return bytes.len >= 8 and std.mem.eql(u8, bytes[0..8], signature);
}

/// The five row filters of the PNG specification (filter method 0).
pub const Filter = enum(u8) { none = 0, sub = 1, up = 2, average = 3, paeth = 4 };

/// How rows are filtered before Deflate.
pub const FilterChoice = enum {
    /// `adaptive` for 8-bit samples, `none` for packed bilevel rows.
    auto,
    none,
    sub,
    up,
    average,
    paeth,
    /// The filter with the smallest sum of absolute differences, scored on
    /// every row of a short image and on every eighth row of a tall one.
    adaptive,
};

/// The Deflate side of the IDAT stream.
pub const Compression = enum {
    /// Stored blocks: no compression, the fastest write.
    none,
    /// zigcv's coder: Huffman coding plus runs of one repeated byte as
    /// distance-1 matches, no other match search. A fraction of `fast`'s
    /// time on filtered photographs; flat areas still shrink to almost nothing.
    huffman,
    /// zlib level 1.
    fast,
    /// zlib level 6.
    default,
    /// zlib level 9.
    best,
};

pub const Options = struct {
    /// 8 writes the samples as they are; 1 packs a single-channel image to one
    /// bit per pixel (values below 128 are black, the rest white).
    bit_depth: u8 = 8,
    filter: FilterChoice = .auto,
    compression: Compression = .fast,
    /// Pixels per inch written as a pHYs chunk; 0 writes none.
    dpi: u32 = 0,
};

// -- row kernels ------------------------------------------------------------------

const N = 32;
const V = @Vector(N, u8);

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p = @as(i16, a) + @as(i16, b) - @as(i16, c);
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

fn paethVec(l: V, u: V, ul: V) V {
    const I = @Vector(N, i16);
    const a: I = @intCast(l);
    const b: I = @intCast(u);
    const c: I = @intCast(ul);
    const p = a + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    return @select(u8, pa <= pb, @select(u8, pa <= pc, l, ul), @select(u8, pb <= pc, u, ul));
}

/// One filtered row: stores it in `out` or, with `score`, only sums the
/// magnitudes of the filtered bytes read as signed (the adaptive heuristic).
/// `prev` is the row above (zeros for the first row); `bpp` the filter distance.
fn kernel(comptime f: Filter, comptime score: bool, out: []u8, cur: []const u8, prev: []const u8, bpp: usize) u64 {
    const n = cur.len;
    const start: usize = if (f == .sub or f == .average or f == .paeth) @min(bpp, n) else 0;
    var total: u64 = 0;
    // The first pixel has no left neighbour: left and upper-left are zero.
    for (0..start) |i| {
        const v: u8 = switch (f) {
            .none, .sub => cur[i],
            .up => cur[i] -% prev[i],
            .average => cur[i] -% (prev[i] >> 1),
            .paeth => cur[i] -% prev[i],
        };
        if (score) total += @abs(@as(i8, @bitCast(v))) else out[i] = v;
    }
    var acc: @Vector(N, u32) = @splat(0);
    var i = start;
    while (i + N <= n) : (i += N) {
        const c: V = cur[i..][0..N].*;
        const v: V = switch (f) {
            .none => c,
            .sub => c -% @as(V, cur[i - bpp ..][0..N].*),
            .up => c -% @as(V, prev[i..][0..N].*),
            .average => blk: {
                const l: V = cur[i - bpp ..][0..N].*;
                const u: V = prev[i..][0..N].*;
                break :blk c -% ((l & u) +% ((l ^ u) >> @splat(1)));
            },
            .paeth => c -% paethVec(cur[i - bpp ..][0..N].*, prev[i..][0..N].*, prev[i - bpp ..][0..N].*),
        };
        if (score) {
            acc += @as(@Vector(N, u32), @intCast(@abs(@as(@Vector(N, i8), @bitCast(v)))));
        } else {
            out[i..][0..N].* = v;
        }
    }
    while (i < n) : (i += 1) {
        const v: u8 = switch (f) {
            .none => cur[i],
            .sub => cur[i] -% cur[i - bpp],
            .up => cur[i] -% prev[i],
            .average => cur[i] -% @as(u8, @intCast((@as(u16, cur[i - bpp]) + prev[i]) >> 1)),
            .paeth => cur[i] -% paeth(cur[i - bpp], prev[i], prev[i - bpp]),
        };
        if (score) total += @abs(@as(i8, @bitCast(v))) else out[i] = v;
    }
    return total + @reduce(.Add, acc);
}

fn filterRow(f: Filter, out: []u8, cur: []const u8, prev: []const u8, bpp: usize) void {
    switch (f) {
        inline else => |ft| _ = kernel(ft, false, out, cur, prev, bpp),
    }
}

fn scoreRow(f: Filter, cur: []const u8, prev: []const u8, bpp: usize) u64 {
    return switch (f) {
        inline else => |ft| kernel(ft, true, &.{}, cur, prev, bpp),
    };
}

/// The filter with the smallest sum of absolute differences; ties go to the
/// earlier filter in the specification's order.
fn bestFilter(cur: []const u8, prev: []const u8, bpp: usize) Filter {
    var best: Filter = .none;
    var best_score: u64 = std.math.maxInt(u64);
    inline for (.{ .none, .sub, .up, .average, .paeth }) |f| {
        const s = scoreRow(f, cur, prev, bpp);
        if (s < best_score) {
            best_score = s;
            best = f;
        }
    }
    return best;
}

/// Pack grey pixels to bits, most significant first: 1 where the value is at
/// least 128 (white), 0 below (black); the last byte's spare bits are zero.
pub fn packRow(row: []const u8, out: []u8) void {
    const P = @Vector(64, u8);
    const thr: P = @splat(128);
    const zero: P = @splat(0);
    // Pixel 8k + i gets the bit 0x80 >> i; the eight bytes of a pixel group
    // are folded by OR into their lowest byte (little-endian lanes).
    const weights: P = comptime blk: {
        var wts: [64]u8 = undefined;
        for (&wts, 0..) |*wt, i| wt.* = @as(u8, 0x80) >> @intCast(i % 8);
        break :blk wts;
    };
    var x: usize = 0;
    var o: usize = 0;
    while (x + 64 <= row.len) : ({
        x += 64;
        o += 8;
    }) {
        const v: P = row[x..][0..64].*;
        var lanes: @Vector(8, u64) = @bitCast(@select(u8, v >= thr, weights, zero));
        lanes |= lanes >> @splat(32);
        lanes |= lanes >> @splat(16);
        lanes |= lanes >> @splat(8);
        out[o..][0..8].* = @as(@Vector(8, u8), @truncate(lanes));
    }
    while (x < row.len) : (o += 1) {
        var byte: u8 = 0;
        var bit: u8 = 0x80;
        while (bit != 0 and x < row.len) : ({
            bit >>= 1;
            x += 1;
        }) {
            if (row[x] >= 128) byte |= bit;
        }
        out[o] = byte;
    }
}

// -- checksums -------------------------------------------------------------------

/// Adler-32 of `adler`'s data followed by `bytes` (`adler` is a running value,
/// 1 for none): 32-byte blocks summed in vector lanes, the lane sums folded
/// into the two running sums once per 4 KB group.
pub fn adler32Update(adler: u32, bytes: []const u8) u32 {
    const base: u32 = 65521;
    const U = @Vector(N, u32);
    const iota: U = comptime blk: {
        var t: [N]u32 = undefined;
        for (&t, 0..) |*x, k| x.* = @intCast(k);
        break :blk t;
    };
    var s1: u32 = adler & 0xffff;
    var s2: u32 = adler >> 16;
    var i: usize = 0;
    while (i + N <= bytes.len) {
        const n = @min((bytes.len - i) / N * N, 128 * N);
        var v1: U = @splat(0);
        var v2: U = @splat(0);
        var j: usize = 0;
        while (j < n) : (j += N) {
            const b: U = @intCast(@as(V, bytes[i + j ..][0..N].*));
            v1 += b;
            v2 += v1;
        }
        // Over n bytes b_j: s2 += n * s1 + sum((n - j) * b_j), and with
        // j = N * m + lane that weight is N * (blocks - m) - lane.
        const weighted: u64 = @as(u64, N) * @reduce(.Add, v2) - @reduce(.Add, v1 * iota);
        s2 = @intCast((@as(u64, s2) + @as(u64, s1) * n + weighted) % base);
        s1 = @intCast((@as(u64, s1) + @reduce(.Add, v1)) % base);
        i += n;
    }
    while (i < bytes.len) : (i += 1) {
        s1 = (s1 + bytes[i]) % base;
        s2 = (s2 + s1) % base;
    }
    return s1 | (s2 << 16);
}

/// Adler-32 of `a ++ b` from the two checksums and `b`'s length (the
/// arithmetic of zlib's `adler32_combine`).
fn adlerCombine(a: u32, b: u32, b_len: usize) u32 {
    const base: u64 = 65521;
    const rem: u64 = b_len % base;
    const s1a: u64 = a & 0xffff;
    const s2a: u64 = a >> 16;
    const s1b: u64 = b & 0xffff;
    const s2b: u64 = b >> 16;
    const s1 = (s1a + s1b + base - 1) % base;
    const s2 = (s2a + s2b + rem * (s1a + base - 1)) % base;
    return @intCast(s1 | (s2 << 16));
}

/// Slicing-by-8 tables of CRC-32/ISO-HDLC (the PNG chunk CRC).
const crc_tables: [8][256]u32 = blk: {
    @setEvalBranchQuota(100000);
    var t: [8][256]u32 = undefined;
    for (0..256) |i| {
        var c: u32 = @intCast(i);
        for (0..8) |_| c = if (c & 1 != 0) (c >> 1) ^ 0xedb88320 else c >> 1;
        t[0][i] = c;
    }
    for (1..8) |k| {
        for (0..256) |i| t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xff];
    }
    break :blk t;
};

/// CRC-32 of `crc`'s data followed by `bytes` (`crc` is a finished value, 0 for none).
pub fn crc32Update(crc: u32, bytes: []const u8) u32 {
    var c = ~crc;
    var i: usize = 0;
    while (i + 8 <= bytes.len) : (i += 8) {
        c ^= std.mem.readInt(u32, bytes[i..][0..4], .little);
        const next = std.mem.readInt(u32, bytes[i + 4 ..][0..4], .little);
        c = crc_tables[7][c & 0xff] ^ crc_tables[6][(c >> 8) & 0xff] ^ crc_tables[5][(c >> 16) & 0xff] ^ crc_tables[4][c >> 24] ^
            crc_tables[3][next & 0xff] ^ crc_tables[2][(next >> 8) & 0xff] ^ crc_tables[1][(next >> 16) & 0xff] ^ crc_tables[0][next >> 24];
    }
    while (i < bytes.len) : (i += 1) c = crc_tables[0][(c ^ bytes[i]) & 0xff] ^ (c >> 8);
    return ~c;
}

fn gf2Times(mat: *const [32]u32, vec_in: u32) u32 {
    var vec = vec_in;
    var sum: u32 = 0;
    var i: usize = 0;
    while (vec != 0) : (i += 1) {
        if (vec & 1 != 0) sum ^= mat[i];
        vec >>= 1;
    }
    return sum;
}

fn gf2Square(square: *[32]u32, mat: *const [32]u32) void {
    for (0..32) |n| square[n] = gf2Times(mat, mat[n]);
}

/// CRC-32 of `a ++ b` from the two CRCs and `b`'s length: `crc_a` is carried
/// over `len_b` zero bytes by GF(2) matrix powers (the method of zlib's
/// `crc32_combine`).
pub fn crc32Combine(crc_a: u32, crc_b: u32, len_b: usize) u32 {
    if (len_b == 0) return crc_a;
    var crc = crc_a;
    var len = len_b;
    var even: [32]u32 = undefined;
    var odd: [32]u32 = undefined;
    // The operator for one zero bit.
    odd[0] = 0xedb88320;
    var row: u32 = 1;
    for (1..32) |n| {
        odd[n] = row;
        row <<= 1;
    }
    gf2Square(&even, &odd); // two zero bits
    gf2Square(&odd, &even); // four zero bits
    while (true) {
        gf2Square(&even, &odd);
        if (len & 1 != 0) crc = gf2Times(&even, crc);
        len >>= 1;
        if (len == 0) break;
        gf2Square(&odd, &even);
        if (len & 1 != 0) crc = gf2Times(&odd, crc);
        len >>= 1;
        if (len == 0) break;
    }
    return crc ^ crc_b;
}

// -- Deflate: Huffman codes and repeated-byte runs ----------------------------------

/// Deflate's length codes: symbol, extra bits and base length, indexed by the
/// match length (3..258).
const LengthCode = struct { sym: u16, extra: u5, base: u16 };

const length_codes: [259]LengthCode = blk: {
    const bases = [_]u16{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
    const extras = [_]u5{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
    var t: [259]LengthCode = undefined;
    for (bases, extras, 0..) |base, extra, k| {
        const top: usize = if (k == 27) 257 else @min(258, @as(usize, base) + (@as(usize, 1) << extra) - 1);
        var len: usize = base;
        while (len <= top) : (len += 1) t[len] = .{ .sym = @intCast(257 + k), .extra = extra, .base = base };
    }
    break :blk t;
};

const max_lit_codes = 286;
const eob = 256;
const max_bits = 15;
const max_cl_bits = 7;
/// The order in which the code-length code lengths are written.
const cl_order = [19]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

/// Code lengths of a Huffman code over `freq` (0 for an unused symbol), none
/// longer than `max_len`: a two-queue Huffman tree, its depths pushed back
/// under the limit by moving codes between neighbouring lengths while the
/// Kraft sum stays at exactly one (the code stays complete), the shortest
/// codes given to the most frequent symbols.
fn huffmanLengths(freq: []const u32, max_len: u8, lens: []u8) void {
    const cap = 2 * max_lit_codes;
    var order: [max_lit_codes]u16 = undefined;
    var used: usize = 0;
    for (freq, 0..) |f, s| {
        lens[s] = 0;
        if (f != 0) {
            order[used] = @intCast(s);
            used += 1;
        }
    }
    if (used == 0) return;
    if (used == 1) {
        lens[order[0]] = 1;
        return;
    }
    const Ctx = struct {
        freq: []const u32,
        fn less(self: @This(), a: u16, b: u16) bool {
            return self.freq[a] < self.freq[b];
        }
    };
    std.mem.sort(u16, order[0..used], Ctx{ .freq = freq }, Ctx.less);
    // Leaves 0..used in weight order, internal nodes appended as they are made.
    var weight: [cap]u32 = undefined;
    var parent: [cap]u16 = undefined;
    for (order[0..used], 0..) |s, k| weight[k] = freq[s];
    var leaf: usize = 0;
    var node: usize = used;
    var next: usize = used;
    while (next < 2 * used - 1) : (next += 1) {
        var pick: [2]usize = undefined;
        for (&pick) |*p| {
            if (leaf < used and (node >= next or weight[leaf] <= weight[node])) {
                p.* = leaf;
                leaf += 1;
            } else {
                p.* = node;
                node += 1;
            }
        }
        weight[next] = weight[pick[0]] + weight[pick[1]];
        parent[pick[0]] = @intCast(next);
        parent[pick[1]] = @intCast(next);
    }
    var depth: [cap]u16 = undefined;
    depth[next - 1] = 0;
    var k = next - 1;
    while (k > 0) {
        k -= 1;
        depth[k] = depth[parent[k]] + 1;
    }
    var bl_count: [cap]u32 = @splat(0);
    for (0..used) |l| bl_count[depth[l]] += 1;
    var overflow = false;
    for (max_len + 1..cap) |l| {
        if (bl_count[l] != 0) {
            bl_count[max_len] += bl_count[l];
            bl_count[l] = 0;
            overflow = true;
        }
    }
    if (overflow) {
        var total: u64 = 0;
        for (1..@as(usize, max_len) + 1) |l| total += @as(u64, bl_count[l]) << @intCast(max_len - l);
        while (total > @as(u64, 1) << @intCast(max_len)) : (total -= 1) {
            // One code leaves the longest length; a shorter code splits in two to keep the count.
            bl_count[max_len] -= 1;
            var l: usize = max_len - 1;
            while (bl_count[l] == 0) l -= 1;
            bl_count[l] -= 1;
            bl_count[l + 1] += 2;
        }
    }
    var at = used;
    for (1..@as(usize, max_len) + 1) |l| {
        for (0..bl_count[l]) |_| {
            at -= 1;
            lens[order[at]] = @intCast(l);
        }
    }
}

/// Canonical Deflate codes from the lengths, bit-reversed for LSB-first output.
fn canonicalCodes(lens: []const u8, codes: []u16) void {
    var bl_count: [max_bits + 1]u16 = @splat(0);
    for (lens) |l| bl_count[l] += 1;
    bl_count[0] = 0;
    var next_code: [max_bits + 1]u16 = undefined;
    var code: u16 = 0;
    for (1..max_bits + 1) |bits| {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (lens, 0..) |l, s| {
        if (l == 0) continue;
        codes[s] = @bitReverse(next_code[l]) >> @intCast(16 - l);
        next_code[l] += 1;
    }
}

const BitWriter = struct {
    out: []u8,
    pos: usize = 0,
    acc: u64 = 0,
    nbits: u6 = 0,

    /// Append `n` (at most 32) bits, least significant first: the whole
    /// accumulator is stored every time and the position moves on by the
    /// complete bytes, so there is no branch; `out` needs 8 bytes of slack.
    inline fn put(self: *BitWriter, bits: u32, n: u6) void {
        self.acc |= @as(u64, bits) << self.nbits;
        self.nbits += n;
        std.mem.writeInt(u64, self.out[self.pos..][0..8], self.acc, .little);
        const bytes = self.nbits >> 3;
        self.pos += bytes;
        self.acc >>= @intCast(@as(u7, bytes) * 8);
        self.nbits &= 7;
    }

    /// Three codes at once (at most 16 bits each), one store for the three.
    inline fn put3(self: *BitWriter, b0: u32, n0: u6, b1: u32, n1: u6, b2: u32, n2: u6) void {
        self.acc |= @as(u64, b0) << self.nbits;
        self.nbits += n0;
        self.acc |= @as(u64, b1) << self.nbits;
        self.nbits += n1;
        self.acc |= @as(u64, b2) << self.nbits;
        self.nbits += n2;
        std.mem.writeInt(u64, self.out[self.pos..][0..8], self.acc, .little);
        const bytes = self.nbits >> 3;
        self.pos += bytes;
        self.acc >>= @intCast(@as(u7, bytes) * 8);
        self.nbits &= 7;
    }

    /// Pad to a byte boundary and write out every buffered bit.
    fn alignToByte(self: *BitWriter) void {
        while (self.nbits > 0) {
            self.out[self.pos] = @truncate(self.acc);
            self.pos += 1;
            self.acc >>= 8;
            self.nbits -|= 8;
        }
    }
};

/// The first index at or after `from` (and after 0) where the byte before
/// repeats at least three times, or `buf.len`; 32 positions compared at a
/// time. Shorter repeats stay literals: on noisy rows a one- or two-byte
/// repeat comes every few bytes and a match would cost more than it saves.
fn nextRun(buf: []const u8, from: usize) usize {
    var j = @max(from, 1);
    while (j + 2 + N <= buf.len) : (j += N) {
        const p: V = buf[j - 1 ..][0..N].*;
        const eq = @select(bool, p == @as(V, buf[j..][0..N].*), p == @as(V, buf[j + 1 ..][0..N].*), @as(@Vector(N, bool), @splat(false)));
        const eq3 = @select(bool, eq, p == @as(V, buf[j + 2 ..][0..N].*), @as(@Vector(N, bool), @splat(false)));
        const mask: u32 = @bitCast(eq3);
        if (mask != 0) return j + @ctz(mask);
    }
    while (j + 2 < buf.len) : (j += 1) {
        if (buf[j] == buf[j - 1] and buf[j + 1] == buf[j - 1] and buf[j + 2] == buf[j - 1]) return j;
    }
    return buf.len;
}

/// Walk a chunk as Deflate tokens: `ctx.literals(bytes)` for a stretch of
/// bytes with no repeat of three or more, `ctx.run(len)` for `len` more
/// copies of the byte before (a match of that length at distance 1).
fn tokenize(buf: []const u8, ctx: anytype) void {
    var i: usize = 0;
    while (i < buf.len) {
        const j = nextRun(buf, i);
        ctx.literals(buf[i..j]);
        i = j;
        if (i >= buf.len) break;
        const prev = buf[i - 1];
        var r: usize = 1;
        const pv: V = @splat(prev);
        var ended = false;
        while (i + r + N <= buf.len) {
            const eq = @as(V, buf[i + r ..][0..N].*) == pv;
            const mask: u32 = @bitCast(eq);
            if (mask == std.math.maxInt(u32)) {
                r += N;
                continue;
            }
            r += @ctz(~mask);
            ended = true;
            break;
        }
        if (!ended) while (i + r < buf.len and buf[i + r] == prev) {
            r += 1;
        };
        if (r < 3) {
            ctx.literals(buf[i..][0..r]);
        } else {
            var rest = r;
            while (rest > 0) {
                var l: usize = @min(rest, 258);
                // Leave at least three for the next match rather than one or two literals.
                if (rest > 258 and rest - 258 < 3) l = rest - 3;
                ctx.run(l);
                rest -= l;
            }
        }
        i += r;
    }
}

const Counter = struct {
    /// Four interleaved histograms: a stretch of one symbol then does not
    /// wait on the store of the count before it.
    lits: [4][max_lit_codes]u32 = @splat(@splat(0)),
    lit: [max_lit_codes]u32 = undefined,
    matches: u64 = 0,
    extra_bits: u64 = 0,

    inline fn literals(self: *Counter, bytes: []const u8) void {
        var k: usize = 0;
        while (k + 4 <= bytes.len) : (k += 4) {
            self.lits[0][bytes[k]] += 1;
            self.lits[1][bytes[k + 1]] += 1;
            self.lits[2][bytes[k + 2]] += 1;
            self.lits[3][bytes[k + 3]] += 1;
        }
        for (bytes[k..], 0..) |b, m| self.lits[m][b] += 1;
    }
    inline fn run(self: *Counter, len: usize) void {
        const lc = length_codes[len];
        self.lits[0][lc.sym] += 1;
        self.extra_bits += lc.extra;
        self.matches += 1;
    }
    fn total(self: *Counter) void {
        for (&self.lit, 0..) |*t, s| t.* = self.lits[0][s] + self.lits[1][s] + self.lits[2][s] + self.lits[3][s];
    }
};

const Emitter = struct {
    bw: *BitWriter,
    codes: *const [max_lit_codes]u16,
    lens: *const [max_lit_codes]u8,

    inline fn literals(self: Emitter, bytes: []const u8) void {
        var k: usize = 0;
        // Three codes of at most 15 bits fit the accumulator with its spare bits.
        while (k + 3 <= bytes.len) : (k += 3) {
            self.bw.put3(self.codes[bytes[k]], @intCast(self.lens[bytes[k]]), self.codes[bytes[k + 1]], @intCast(self.lens[bytes[k + 1]]), self.codes[bytes[k + 2]], @intCast(self.lens[bytes[k + 2]]));
        }
        for (bytes[k..]) |b| self.bw.put(self.codes[b], @intCast(self.lens[b]));
    }
    inline fn run(self: Emitter, len: usize) void {
        const lc = length_codes[len];
        self.bw.put(self.codes[lc.sym], @intCast(self.lens[lc.sym]));
        if (lc.extra != 0) self.bw.put(@intCast(len - lc.base), lc.extra);
        // Distance 1 is distance symbol 0, a one-bit code.
        self.bw.put(0, 1);
    }
};

/// A dynamic block's header: the code lengths run-length coded as the format
/// prescribes, the code-length code over them, and the header's bit count.
const Header = struct {
    const Item = struct { sym: u8, extra: u8, value: u8 };
    items: [max_lit_codes + 2]Item = undefined,
    n_items: usize = 0,
    hlit: usize,
    cl_lens: [19]u8 = @splat(0),
    cl_codes: [19]u16 = @splat(0),
    hclen: usize = 19,
    bits: u64 = 0,

    fn push(self: *Header, sym: u8, extra: u8, value: u8) void {
        self.items[self.n_items] = .{ .sym = sym, .extra = extra, .value = value };
        self.n_items += 1;
    }

    fn init(lit_lens: *const [max_lit_codes]u8) Header {
        var hlit: usize = max_lit_codes;
        while (hlit > 257 and lit_lens[hlit - 1] == 0) hlit -= 1;
        var h = Header{ .hlit = hlit };
        // The two distance codes of length 1, after the literal lengths.
        var seq: [max_lit_codes + 2]u8 = undefined;
        @memcpy(seq[0..hlit], lit_lens[0..hlit]);
        seq[hlit] = 1;
        seq[hlit + 1] = 1;
        const n = hlit + 2;
        var i: usize = 0;
        while (i < n) {
            const v = seq[i];
            var run: usize = 1;
            while (i + run < n and seq[i + run] == v) run += 1;
            var rest = run;
            if (v == 0) {
                while (rest >= 11) {
                    const r = @min(rest, 138);
                    h.push(18, 7, @intCast(r - 11));
                    rest -= r;
                }
                if (rest >= 3) {
                    h.push(17, 3, @intCast(rest - 3));
                    rest = 0;
                }
            } else {
                h.push(v, 0, 0);
                rest -= 1;
                while (rest >= 3) {
                    const r = @min(rest, 6);
                    h.push(16, 2, @intCast(r - 3));
                    rest -= r;
                }
            }
            while (rest > 0) : (rest -= 1) h.push(v, 0, 0);
            i += run;
        }
        var cl_freq: [19]u32 = @splat(0);
        for (h.items[0..h.n_items]) |it| cl_freq[it.sym] += 1;
        huffmanLengths(&cl_freq, max_cl_bits, &h.cl_lens);
        // The code-length code must be complete: a lone symbol gets a partner.
        var cl_used: usize = 0;
        for (h.cl_lens) |l| cl_used += @intFromBool(l != 0);
        if (cl_used == 1) h.cl_lens[if (h.cl_lens[0] != 0) 1 else 0] = 1;
        canonicalCodes(&h.cl_lens, &h.cl_codes);
        while (h.hclen > 4 and h.cl_lens[cl_order[h.hclen - 1]] == 0) h.hclen -= 1;
        h.bits = 3 + 5 + 5 + 4 + 3 * h.hclen;
        for (h.items[0..h.n_items]) |it| h.bits += h.cl_lens[it.sym] + it.extra;
        return h;
    }

    fn write(self: *const Header, bw: *BitWriter, final: bool) void {
        bw.put(@intFromBool(final), 1);
        bw.put(2, 2);
        bw.put(@intCast(self.hlit - 257), 5);
        bw.put(1, 5);
        bw.put(@intCast(self.hclen - 4), 4);
        for (cl_order[0..self.hclen]) |s| bw.put(self.cl_lens[s], 3);
        for (self.items[0..self.n_items]) |it| {
            bw.put(self.cl_codes[it.sym], @intCast(self.cl_lens[it.sym]));
            if (it.extra != 0) bw.put(it.value, @intCast(it.extra));
        }
    }
};

/// Bytes of `len` bytes as stored blocks of at most 65535.
fn storedSize(len: usize) usize {
    return len + 5 * @max((len + 65534) / 65535, 1);
}

/// `buf` as stored blocks (the last one final when `final`).
fn writeStored(out: []u8, buf: []const u8, final: bool) usize {
    var pos: usize = 0;
    var off: usize = 0;
    while (true) {
        const n: usize = @min(buf.len - off, 65535);
        const last = off + n == buf.len;
        out[pos] = @intFromBool(final and last);
        std.mem.writeInt(u16, out[pos + 1 ..][0..2], @intCast(n), .little);
        std.mem.writeInt(u16, out[pos + 3 ..][0..2], @intCast(~n & 0xffff), .little);
        @memcpy(out[pos + 5 ..][0..n], buf[off..][0..n]);
        pos += 5 + n;
        off += n;
        if (last) return pos;
    }
}

/// `buf` as one Deflate block appended to `out`: a dynamic Huffman block of
/// literals and distance-1 runs, or stored blocks when those are smaller. A
/// block that is not `final` is followed by an empty stored block, so the
/// stream ends on a byte boundary and the next chunk's stream can follow.
fn deflateChunk(out: *std.Io.Writer, buf: []const u8, final: bool) !void {
    var count = Counter{};
    tokenize(buf, &count);
    count.total();
    count.lit[eob] = 1;
    var lens: [max_lit_codes]u8 = undefined;
    huffmanLengths(&count.lit, max_bits, &lens);
    const header = Header.init(&lens);
    var bits: u64 = header.bits + count.extra_bits + count.matches;
    for (count.lit, lens) |c, l| bits += @as(u64, c) * l;
    if (!final) bits += 3;
    const size = (bits + 7) / 8 + @as(usize, if (final) 0 else 4);
    const stored = storedSize(buf.len);
    if (size >= stored) {
        const dst = try out.writableSliceGreedy(stored);
        out.advance(writeStored(dst, buf, final));
        return;
    }
    var codes: [max_lit_codes]u16 = @splat(0);
    canonicalCodes(&lens, &codes);
    var bw = BitWriter{ .out = try out.writableSliceGreedy(size + 8) };
    header.write(&bw, final);
    tokenize(buf, Emitter{ .bw = &bw, .codes = &codes, .lens = &lens });
    bw.put(codes[eob], @intCast(lens[eob]));
    if (!final) {
        bw.put(0, 3);
        bw.alignToByte();
        bw.out[bw.pos..][0..4].* = .{ 0, 0, 0xff, 0xff };
        bw.pos += 4;
    } else {
        bw.alignToByte();
    }
    std.debug.assert(bw.pos == size);
    out.advance(bw.pos);
}

// -- encoder ---------------------------------------------------------------------

/// Filtered bytes per chunk at least: a page splits into chunks of this many
/// rows, each its own Deflate stream, so the chunks spread over the CPUs and
/// the output does not depend on the thread count; a small image is one chunk.
const chunk_bytes = 128 * 1024;

const Band = struct {
    out: std.Io.Writer.Allocating,
    /// Adler-32 and length of the filtered rows, CRC-32 of the Deflate output.
    adler: u32 = 1,
    len: usize = 0,
    crc: u32 = 0,
    failed: bool = false,
};

fn level(c: Compression) flate.Compress.Options {
    return switch (c) {
        .fast => .level_1,
        .default => .level_6,
        .best => .level_9,
        .none, .huffman => unreachable,
    };
}

const Job = struct {
    data: []const u8,
    w: usize,
    h: usize,
    /// Bytes of one filtered row without its filter byte.
    stride: usize,
    /// Filter distance in bytes.
    bpp: usize,
    packed_bits: bool,
    filter: FilterChoice,
    compression: Compression,
    sample: usize,
    rows_per_chunk: usize,
    chunks: usize,
    bands: []Band,

    fn run(self: *const Job, k: usize, c0: usize, c1: usize) void {
        self.runBand(k, c0, c1) catch {
            self.bands[k].failed = true;
        };
    }

    fn runBand(self: *const Job, k: usize, c0: usize, c1: usize) !void {
        const pa = std.heap.page_allocator;
        const band = &self.bands[k];
        const stride = self.stride;
        const scratch = try pa.alloc(u8, self.rows_per_chunk * (stride + 1) + 3 * stride);
        defer pa.free(scratch);
        const buf_all = scratch[0 .. self.rows_per_chunk * (stride + 1)];
        var pack_a = scratch[buf_all.len..][0..stride];
        var pack_b = scratch[buf_all.len + stride ..][0..stride];
        const zero = scratch[buf_all.len + 2 * stride ..][0..stride];
        @memset(zero, 0);
        const window: []u8 = switch (self.compression) {
            .fast, .default, .best => try pa.alloc(u8, flate.max_window_len),
            .none, .huffman => &.{},
        };
        defer if (window.len > 0) pa.free(window);
        const y_end = @min(self.h, c1 * self.rows_per_chunk);
        const bytes = (y_end - c0 * self.rows_per_chunk) * (stride + 1);
        // std's compressor wants a buffer of its own in the writer even for a tiny image.
        try band.out.ensureUnusedCapacity(@max(switch (self.compression) {
            .none => storedSize(bytes),
            else => bytes / 3,
        }, 4096));
        for (c0..c1) |chunk| {
            const y0 = chunk * self.rows_per_chunk;
            const y1 = @min(self.h, y0 + self.rows_per_chunk);
            const buf = buf_all[0 .. (y1 - y0) * (stride + 1)];
            // The row above the chunk, for the filters of its first row.
            var prev: []const u8 = zero;
            if (y0 > 0) {
                if (self.packed_bits) {
                    packRow(self.data[(y0 - 1) * self.w ..][0..self.w], pack_b);
                    prev = pack_b;
                } else {
                    prev = self.data[(y0 - 1) * stride ..][0..stride];
                }
            }
            var choice: Filter = .none;
            for (y0..y1) |y| {
                const cur: []const u8 = if (self.packed_bits) blk: {
                    packRow(self.data[y * self.w ..][0..self.w], pack_a);
                    break :blk pack_a;
                } else self.data[y * stride ..][0..stride];
                const f: Filter = switch (self.filter) {
                    .none => .none,
                    .sub => .sub,
                    .up => .up,
                    .average => .average,
                    .paeth => .paeth,
                    .adaptive => blk: {
                        if (y == y0 or y % self.sample == 0) choice = bestFilter(cur, prev, self.bpp);
                        break :blk choice;
                    },
                    .auto => unreachable,
                };
                const line = buf[(y - y0) * (stride + 1) ..][0 .. stride + 1];
                line[0] = @backingInt(f);
                if (f == .none) @memcpy(line[1..], cur) else filterRow(f, line[1..], cur, prev, self.bpp);
                prev = cur;
                if (self.packed_bits) std.mem.swap([]u8, &pack_a, &pack_b);
            }
            band.adler = adler32Update(band.adler, buf);
            band.len += buf.len;
            const last = chunk + 1 == self.chunks;
            switch (self.compression) {
                .none => {
                    const dst = try band.out.writer.writableSliceGreedy(storedSize(buf.len));
                    band.out.writer.advance(writeStored(dst, buf, last));
                },
                .huffman => try deflateChunk(&band.out.writer, buf, last),
                .fast, .default, .best => {
                    var c = try flate.Compress.init(&band.out.writer, window, .raw, level(self.compression));
                    try c.writer.writeAll(buf);
                    if (last) try c.finish() else try c.writer.flush();
                },
            }
        }
        band.crc = crc32Update(0, band.out.written());
    }
};

fn colorType(ch: usize) u8 {
    return switch (ch) {
        1 => 0,
        2 => 4,
        3 => 2,
        else => 6,
    };
}

const ChunkWriter = struct {
    out: []u8,
    pos: usize = 0,

    fn begin(self: *ChunkWriter, kind: *const [4]u8, len: usize) u32 {
        std.mem.writeInt(u32, self.out[self.pos..][0..4], @intCast(len), .big);
        @memcpy(self.out[self.pos + 4 ..][0..4], kind);
        self.pos += 8;
        return crc32Update(0, kind);
    }

    fn put(self: *ChunkWriter, crc: *u32, bytes: []const u8) void {
        @memcpy(self.out[self.pos..][0..bytes.len], bytes);
        crc.* = crc32Update(crc.*, bytes);
        self.pos += bytes.len;
    }

    /// Append bytes whose CRC is known already.
    fn putHashed(self: *ChunkWriter, crc: *u32, bytes: []const u8, bytes_crc: u32) void {
        @memcpy(self.out[self.pos..][0..bytes.len], bytes);
        crc.* = crc32Combine(crc.*, bytes_crc, bytes.len);
        self.pos += bytes.len;
    }

    fn end(self: *ChunkWriter, crc: u32) void {
        std.mem.writeInt(u32, self.out[self.pos..][0..4], crc, .big);
        self.pos += 4;
    }

    fn chunk(self: *ChunkWriter, kind: *const [4]u8, data: []const u8) void {
        var crc = self.begin(kind, data.len);
        self.put(&crc, data);
        self.end(crc);
    }
};

/// Encode `w` x `h` pixels of `ch` interleaved 8-bit channels (1 grey, 2
/// grey+alpha, 3 RGB, 4 RGBA) as a PNG. With `opts.bit_depth == 1` the image
/// must be single-channel and is written as 1-bit greyscale.
pub fn encode(a: Allocator, data: []const u8, w: usize, h: usize, ch: usize, opts: Options) Allocator.Error![]u8 {
    std.debug.assert(w > 0 and h > 0 and ch >= 1 and ch <= 4 and data.len == w * h * ch);
    std.debug.assert(opts.bit_depth == 8 or (opts.bit_depth == 1 and ch == 1));
    const packed_bits = opts.bit_depth == 1;
    const stride = if (packed_bits) (w + 7) / 8 else w * ch;
    const filter: FilterChoice = switch (opts.filter) {
        .auto => if (packed_bits) FilterChoice.none else .adaptive,
        else => |f| f,
    };
    const rows_per_chunk = @max(chunk_bytes / (stride + 1), 1);
    const chunks = (h + rows_per_chunk - 1) / rows_per_chunk;
    const n_bands = par.bandCount(chunks, 1);
    const pa = std.heap.page_allocator;
    const bands = try pa.alloc(Band, n_bands);
    defer pa.free(bands);
    for (bands) |*b| b.* = .{ .out = .init(pa) };
    defer for (bands) |*b| b.out.deinit();
    const job = Job{
        .data = data,
        .w = w,
        .h = h,
        .stride = stride,
        .bpp = if (packed_bits) 1 else ch,
        .packed_bits = packed_bits,
        .filter = filter,
        .compression = opts.compression,
        .sample = if (h > 512) 8 else 1,
        .rows_per_chunk = rows_per_chunk,
        .chunks = chunks,
        .bands = bands,
    };
    par.forBands(chunks, 1, &job, Job.run);

    var zlen: usize = 0;
    var adler: u32 = 1;
    for (bands) |*b| {
        if (b.failed) return error.OutOfMemory;
        zlen += b.out.written().len;
        adler = adlerCombine(adler, b.adler, b.len);
    }
    const zlib_header = flate.Container.zlib.header();
    const idat_len = zlib_header.len + zlen + 4;
    const total = signature.len + (12 + 13) + (if (opts.dpi != 0) @as(usize, 12 + 9) else 0) + (12 + idat_len) + 12;
    const out = try a.alloc(u8, total);
    errdefer a.free(out);
    var cw = ChunkWriter{ .out = out };
    @memcpy(out[0..signature.len], signature);
    cw.pos = signature.len;

    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(w), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(h), .big);
    ihdr[8] = opts.bit_depth;
    ihdr[9] = colorType(ch);
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    cw.chunk("IHDR", &ihdr);
    if (opts.dpi != 0) {
        var phys: [9]u8 = undefined;
        const ppm: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(opts.dpi)) / 0.0254));
        std.mem.writeInt(u32, phys[0..4], ppm, .big);
        std.mem.writeInt(u32, phys[4..8], ppm, .big);
        phys[8] = 1;
        cw.chunk("pHYs", &phys);
    }
    var crc = cw.begin("IDAT", idat_len);
    cw.put(&crc, zlib_header);
    for (bands) |*b| cw.putHashed(&crc, b.out.written(), b.crc);
    var adler_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler_bytes, adler, .big);
    cw.put(&crc, &adler_bytes);
    cw.end(crc);
    cw.chunk("IEND", "");
    std.debug.assert(cw.pos == total);
    return out;
}

// -- decoder ---------------------------------------------------------------------

pub const Decoded = struct {
    w: usize,
    h: usize,
    /// 1 grey, 2 grey+alpha, 3 RGB, 4 RGBA; sub-byte grey is expanded to 0..255.
    ch: usize,
    data: []u8,
    /// From the pHYs chunk, 0 when absent or not in metres.
    dpi: u32,

    pub fn deinit(self: Decoded, a: Allocator) void {
        a.free(self.data);
    }
};

/// Decode a non-interlaced PNG of bit depth 1, 2, 4 or 8 without a palette.
/// Images above `max_pixels` are `error.TooLarge`.
pub fn decode(a: Allocator, bytes: []const u8) Error!Decoded {
    @setRuntimeSafety(true); // untrusted input: bounds and overflow checks even in ReleaseFast
    if (!isPng(bytes)) return error.NotPng;
    var pos: usize = 8;
    var w: usize = 0;
    var h: usize = 0;
    var depth: u8 = 0;
    var ctype: u8 = 0;
    var dpi: u32 = 0;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    while (pos + 8 <= bytes.len) {
        const len = std.mem.readInt(u32, bytes[pos..][0..4], .big);
        const kind = bytes[pos + 4 .. pos + 8];
        const start = pos + 8;
        if (start + len + 4 > bytes.len) return error.Malformed;
        const body = bytes[start .. start + len];
        if (std.mem.eql(u8, kind, "IHDR")) {
            if (len != 13) return error.Malformed;
            w = std.mem.readInt(u32, body[0..4], .big);
            h = std.mem.readInt(u32, body[4..8], .big);
            depth = body[8];
            ctype = body[9];
            if (body[12] != 0) return error.Unsupported;
        } else if (std.mem.eql(u8, kind, "pHYs")) {
            if (len == 9 and body[8] == 1) {
                const ppm = std.mem.readInt(u32, body[0..4], .big);
                dpi = @intFromFloat(@round(@as(f64, @floatFromInt(ppm)) * 0.0254));
            }
        } else if (std.mem.eql(u8, kind, "IDAT")) {
            try idat.appendSlice(a, body);
        } else if (std.mem.eql(u8, kind, "IEND")) {
            break;
        }
        pos = start + len + 4;
    }
    if (w == 0 or h == 0) return error.Malformed;
    const pixels = try mulSize(w, h);
    if (pixels > max_pixels) return error.TooLarge;
    const ch: usize = switch (ctype) {
        0 => 1,
        2 => 3,
        3 => return error.Unsupported,
        4 => 2,
        6 => 4,
        else => return error.Malformed,
    };
    if (depth != 8 and !(ctype == 0 and (depth == 1 or depth == 2 or depth == 4))) return error.Unsupported;
    const bits_per_px = @as(usize, depth) * ch;
    const stride = (try mulSize(w, bits_per_px) + 7) / 8;
    const line_len = stride + 1;
    const bpp = @max(bits_per_px / 8, 1);

    var reader = std.Io.Reader.fixed(idat.items);
    const window = try a.alloc(u8, flate.max_window_len);
    defer a.free(window);
    var dec = flate.Decompress.init(&reader, .zlib, window);
    const raw = dec.reader.allocRemaining(a, .unlimited) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Malformed,
    };
    defer a.free(raw);
    if (raw.len < try mulSize(line_len, h)) return error.Malformed;

    const row_len = try mulSize(w, ch);
    const out = try a.alloc(u8, try mulSize(row_len, h));
    errdefer a.free(out);
    const prev = try a.alloc(u8, stride);
    defer a.free(prev);
    @memset(prev, 0);
    const cur = try a.alloc(u8, stride);
    defer a.free(cur);
    for (0..h) |y| {
        const line = raw[y * line_len .. (y + 1) * line_len];
        const filter = line[0];
        if (filter > 4) return error.Malformed;
        @memcpy(cur, line[1..]);
        for (cur, 0..) |*v, i| {
            const left: u8 = if (i >= bpp) cur[i - bpp] else 0;
            const up = prev[i];
            const ul: u8 = if (i >= bpp) prev[i - bpp] else 0;
            v.* +%= switch (filter) {
                0 => 0,
                1 => left,
                2 => up,
                3 => @intCast((@as(u16, left) + up) / 2),
                else => paeth(left, up, ul),
            };
        }
        const o = out[y * row_len .. (y + 1) * row_len];
        if (depth == 8) {
            @memcpy(o, cur);
        } else {
            const per_byte = 8 / @as(usize, depth);
            const mask: u8 = @intCast((@as(u16, 1) << @intCast(depth)) - 1);
            for (0..w) |x| {
                const shift: u3 = @intCast(8 - depth - (x % per_byte) * depth);
                const v = (cur[x / per_byte] >> shift) & mask;
                o[x] = @intCast(@as(u16, v) * 255 / mask);
            }
        }
        @memcpy(prev, cur);
    }
    return .{ .w = w, .h = h, .ch = ch, .data = out, .dpi = dpi };
}

// -- tests -----------------------------------------------------------------------

fn testImage(a: Allocator, w: usize, h: usize, ch: usize) ![]u8 {
    const img = try a.alloc(u8, w * h * ch);
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    for (0..h) |y| for (0..w) |x| for (0..ch) |c| {
        // Smooth shading with a little noise: every filter gets a chance.
        const base: usize = (x * 3 + y * 5 + c * 40) % 256;
        img[(y * w + x) * ch + c] = @intCast((base + r.uintLessThan(usize, 9)) % 256);
    };
    return img;
}

fn expectRoundtrip(a: Allocator, w: usize, h: usize, ch: usize, opts: Options) !void {
    const img = try testImage(a, w, h, ch);
    defer a.free(img);
    const png = try encode(a, img, w, h, ch, opts);
    defer a.free(png);
    try std.testing.expect(isPng(png));
    const d = try decode(a, png);
    defer d.deinit(a);
    try std.testing.expectEqual(w, d.w);
    try std.testing.expectEqual(h, d.h);
    try std.testing.expectEqual(ch, d.ch);
    try std.testing.expectEqualSlices(u8, img, d.data);
    try std.testing.expectEqual(opts.dpi, d.dpi);
}

test "png roundtrips every filter and compression" {
    const a = std.testing.allocator;
    inline for (.{ .none, .sub, .up, .average, .paeth, .adaptive, .auto }) |f| {
        inline for (.{ .none, .huffman, .fast, .best }) |c| {
            try expectRoundtrip(a, 37, 11, 1, .{ .filter = f, .compression = c });
            try expectRoundtrip(a, 37, 11, 3, .{ .filter = f, .compression = c });
        }
    }
    try expectRoundtrip(a, 70, 5, 2, .{});
    try expectRoundtrip(a, 70, 5, 4, .{ .dpi = 300 });
    try expectRoundtrip(a, 1, 1, 1, .{});
    try expectRoundtrip(a, 1, 1, 1, .{ .compression = .huffman });
    try expectRoundtrip(a, 1, 1, 3, .{ .compression = .none });
}

test "png bilevel pages pack to one bit per pixel" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 7, 8, 9, 63, 64, 65, 200 }) |w| {
        const h: usize = 9;
        const img = try a.alloc(u8, w * h);
        defer a.free(img);
        for (0..h) |y| for (0..w) |x| {
            img[y * w + x] = if ((x * 7 + y * 3) % 5 < 2 or x == w - 1) 0 else 255;
        };
        inline for ([_]FilterChoice{ .auto, .up, .adaptive }) |f| {
            const png = try encode(a, img, w, h, 1, .{ .bit_depth = 1, .filter = f });
            defer a.free(png);
            try std.testing.expectEqual(@as(u8, 1), png[24]); // IHDR bit depth
            try std.testing.expectEqual(@as(u8, 0), png[25]); // greyscale
            const d = try decode(a, png);
            defer d.deinit(a);
            try std.testing.expectEqualSlices(u8, img, d.data);
        }
    }
    // Grey levels threshold at 128.
    const g = [_]u8{ 0, 127, 128, 255, 1, 254, 100, 200, 50 };
    var packed_row: [2]u8 = undefined;
    packRow(&g, &packed_row);
    try std.testing.expectEqual([2]u8{ 0b0011_0101, 0b0000_0000 }, packed_row);
}

test "png chunks concatenate into one zlib stream" {
    const a = std.testing.allocator;
    const saved = par.thread_override;
    defer par.thread_override = saved;
    par.thread_override = 3;
    const w: usize = 1024;
    const h: usize = 600;
    try std.testing.expect(h / (chunk_bytes / (w + 1)) >= 2);
    try expectRoundtrip(a, w, h, 1, .{});
    try expectRoundtrip(a, w, h, 1, .{ .compression = .none });
    try expectRoundtrip(a, w, h, 1, .{ .compression = .huffman, .filter = .paeth });
    try expectRoundtrip(a, w, h, 1, .{ .compression = .huffman, .filter = .adaptive });
    // The IDAT's zlib stream, checksum included, inflates on its own.
    const img = try testImage(a, w, h, 1);
    defer a.free(img);
    inline for ([_]Compression{ .fast, .huffman }) |c| {
        const png = try encode(a, img, w, h, 1, .{ .compression = c });
        defer a.free(png);
        const idat_len = std.mem.readInt(u32, png[33..37], .big);
        try std.testing.expectEqualStrings("IDAT", png[37..41]);
        var reader = std.Io.Reader.fixed(png[41 .. 41 + idat_len]);
        const window = try a.alloc(u8, flate.max_window_len);
        defer a.free(window);
        var dec = flate.Decompress.init(&reader, .zlib, window);
        const raw = try dec.reader.allocRemaining(a, .unlimited);
        defer a.free(raw);
        try std.testing.expectEqual((w + 1) * h, raw.len);
        // The chunking does not depend on the thread count: single-threaded output is byte-identical.
        par.thread_override = 1;
        const png1 = try encode(a, img, w, h, 1, .{ .compression = c });
        defer a.free(png1);
        try std.testing.expectEqualSlices(u8, png, png1);
        par.thread_override = 3;
    }
}

test "png huffman coder: runs shrink, noise is stored, every block decodes" {
    const a = std.testing.allocator;
    const w: usize = 3000;
    const h: usize = 100;
    // A flat image: after the Up filter every row but the first is zeros.
    const flat = try a.alloc(u8, w * h);
    defer a.free(flat);
    @memset(flat, 200);
    const flat_png = try encode(a, flat, w, h, 1, .{ .compression = .huffman, .filter = .up });
    defer a.free(flat_png);
    // 300 KB of rows become about 1200 runs of 258 zero bytes, a few bits each.
    try std.testing.expect(flat_png.len < 1200);
    const d = try decode(a, flat_png);
    defer d.deinit(a);
    try std.testing.expectEqualSlices(u8, flat, d.data);
    // Random bytes cannot be coded below their size: stored blocks, two chunks of them.
    const noise = try a.alloc(u8, w * h);
    defer a.free(noise);
    var rng = std.Random.DefaultPrng.init(3);
    rng.random().bytes(noise);
    const noise_png = try encode(a, noise, w, h, 1, .{ .compression = .huffman, .filter = .none });
    defer a.free(noise_png);
    try std.testing.expect(noise_png.len < w * h + (w * h) / 100);
    const dn = try decode(a, noise_png);
    defer dn.deinit(a);
    try std.testing.expectEqualSlices(u8, noise, dn.data);
    // Runs of every length around the match limits, and runs across row ends.
    const img = try a.alloc(u8, w * h);
    defer a.free(img);
    var x: usize = 0;
    var v: u8 = 0;
    var run: usize = 1;
    while (x < img.len) {
        const n = @min(run, img.len - x);
        @memset(img[x..][0..n], v);
        x += n;
        v +%= 37;
        run = (run * 7 + 3) % 600;
    }
    inline for ([_]FilterChoice{ .none, .sub, .adaptive }) |f| {
        const png = try encode(a, img, w, h, 1, .{ .compression = .huffman, .filter = f });
        defer a.free(png);
        const back = try decode(a, png);
        defer back.deinit(a);
        try std.testing.expectEqualSlices(u8, img, back.data);
    }
}

test "png huffman lengths stay within the limit and complete" {
    // Fibonacci frequencies force a tree deeper than 15.
    var freq: [40]u32 = undefined;
    var fa: u32 = 1;
    var fb: u32 = 1;
    for (&freq) |*f| {
        f.* = fa;
        const t = fa + fb;
        fa = fb;
        fb = t;
    }
    var lens: [40]u8 = undefined;
    huffmanLengths(&freq, max_bits, &lens);
    var kraft: u64 = 0;
    var max: u8 = 0;
    for (lens) |l| {
        try std.testing.expect(l >= 1 and l <= max_bits);
        kraft += @as(u64, 1) << @intCast(max_bits - l);
        max = @max(max, l);
    }
    try std.testing.expectEqual(@as(u64, 1) << max_bits, kraft);
    try std.testing.expectEqual(@as(u8, max_bits), max);
    try std.testing.expect(lens[39] < lens[0]);
    // Few symbols, one of them unused.
    const small = [_]u32{ 5, 0, 1, 1 };
    var sl: [4]u8 = undefined;
    huffmanLengths(&small, max_bits, &sl);
    try std.testing.expectEqualSlices(u8, &.{ 1, 0, 2, 2 }, &sl);
    var codes: [4]u16 = @splat(0);
    canonicalCodes(&sl, &codes);
    // Canonical: 0, 10, 11; stored bit-reversed.
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, 1, 3 }, &codes);
}

test "png structure and the pHYs chunk" {
    const a = std.testing.allocator;
    const img = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const png = try encode(a, &img, 2, 1, 3, .{ .dpi = 300, .compression = .none });
    defer a.free(png);
    try std.testing.expectEqualSlices(u8, signature, png[0..8]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 13, 'I', 'H', 'D', 'R', 0, 0, 0, 2, 0, 0, 0, 1, 8, 2, 0, 0, 0 }, png[8..29]);
    try std.testing.expectEqualStrings("pHYs", png[37..41]);
    try std.testing.expectEqual(@as(u32, 11811), std.mem.readInt(u32, png[41..45], .big));
    try std.testing.expectEqual(@as(u8, 1), png[49]);
    try std.testing.expectEqualStrings("IEND", png[png.len - 8 .. png.len - 4]);
    // Every chunk's CRC checks.
    var pos: usize = 8;
    while (pos < png.len) {
        const len = std.mem.readInt(u32, png[pos..][0..4], .big);
        const crc = std.hash.Crc32.hash(png[pos + 4 .. pos + 8 + len]);
        try std.testing.expectEqual(crc, std.mem.readInt(u32, png[pos + 8 + len ..][0..4], .big));
        pos += 12 + len;
    }
    try std.testing.expectError(error.NotPng, decode(a, "nope"));
    try std.testing.expectError(error.Malformed, decode(a, signature ++ "\x00\x00\x00\x0dIHDR"));
}

test "png filters match the specification" {
    const prev = [_]u8{ 10, 20, 30, 40, 50, 60 };
    const cur = [_]u8{ 12, 25, 31, 45, 55, 58 };
    var out: [6]u8 = undefined;
    filterRow(.sub, &out, &cur, &prev, 3);
    try std.testing.expectEqualSlices(u8, &.{ 12, 25, 31, 45 - 12, 55 - 25, 58 - 31 }, &out);
    filterRow(.up, &out, &cur, &prev, 3);
    try std.testing.expectEqualSlices(u8, &.{ 2, 5, 1, 5, 5, 0xfe }, &out);
    filterRow(.average, &out, &cur, &prev, 3);
    try std.testing.expectEqualSlices(u8, &.{ 12 - 5, 25 - 10, 31 - 15, 45 - (12 + 40) / 2, 55 - (25 + 50) / 2, 58 - (31 + 60) / 2 }, &out);
    filterRow(.paeth, &out, &cur, &prev, 3);
    try std.testing.expectEqualSlices(u8, &.{ 2, 5, 1, 45 -% paeth(12, 40, 10), 55 -% paeth(25, 50, 20), 58 -% paeth(31, 60, 30) }, &out);
    // The vector path agrees with the scalar tail on a long row.
    var long_prev: [100]u8 = undefined;
    var long_cur: [100]u8 = undefined;
    for (0..100) |i| {
        long_prev[i] = @intCast((i * 37) % 256);
        long_cur[i] = @intCast((i * 91 + 13) % 256);
    }
    inline for ([_]Filter{ .sub, .up, .average, .paeth }) |f| {
        var vec: [100]u8 = undefined;
        filterRow(f, &vec, &long_cur, &long_prev, 1);
        for (1..100) |i| {
            const expected: u8 = switch (f) {
                .sub => long_cur[i] -% long_cur[i - 1],
                .up => long_cur[i] -% long_prev[i],
                .average => long_cur[i] -% @as(u8, @intCast((@as(u16, long_cur[i - 1]) + long_prev[i]) / 2)),
                .paeth => long_cur[i] -% paeth(long_cur[i - 1], long_prev[i], long_prev[i - 1]),
                .none => unreachable,
            };
            try std.testing.expectEqual(expected, vec[i]);
        }
        var sum: u64 = 0;
        for (vec) |v| sum += @abs(@as(i8, @bitCast(v)));
        try std.testing.expectEqual(sum, scoreRow(f, &long_cur, &long_prev, 1));
    }
}

test "png checksums agree with std and combine" {
    try std.testing.expectEqual(std.hash.Adler32.hash("hello world"), adlerCombine(std.hash.Adler32.hash("hello "), std.hash.Adler32.hash("world"), 5));
    var buf: [10000]u8 = undefined;
    for (&buf, 0..) |*b, i| b.* = @intCast((i * 131 + 7) % 256);
    for ([_]usize{ 0, 1, 7, 8, 9, 31, 32, 33, 100, 1000, 4095, 4096, 4097, 8200, 10000 }) |n| {
        try std.testing.expectEqual(std.hash.Crc32.hash(buf[0..n]), crc32Update(0, buf[0..n]));
        try std.testing.expectEqual(std.hash.Adler32.hash(buf[0..n]), adler32Update(1, buf[0..n]));
        for ([_]usize{ 0, 1, 5, 64, 4100 }) |split| {
            if (split > n) continue;
            const whole = crc32Update(0, buf[0..n]);
            try std.testing.expectEqual(whole, crc32Update(crc32Update(0, buf[0..split]), buf[split..n]));
            try std.testing.expectEqual(whole, crc32Combine(crc32Update(0, buf[0..split]), crc32Update(0, buf[split..n]), n - split));
            try std.testing.expectEqual(std.hash.Adler32.hash(buf[0..n]), adler32Update(adler32Update(1, buf[0..split]), buf[split..n]));
        }
    }
    // Bytes of 255 stress the sums.
    @memset(&buf, 255);
    try std.testing.expectEqual(std.hash.Adler32.hash(&buf), adler32Update(1, &buf));
    try std.testing.expectEqual(@as(u32, 0xcbf43926), crc32Update(0, "123456789"));
}
