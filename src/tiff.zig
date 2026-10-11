//! TIFF without libtiff: a writer and a reader for the baseline files a
//! document scanner produces.
//!
//! - 1-bit CCITT Group 4 (T.6) bilevel pages, any number of strips, with an
//!   encoder whose codestream matches libtiff's bit for bit (vertical, pass
//!   and horizontal modes as T.6 prescribes, EOFB per strip);
//! - 8-bit grey and RGB, uncompressed, LZW (TIFF variant, early change) or
//!   Deflate (zlib through `std.compress.flate`), horizontal predictor on read;
//! - PackBits on read;
//! - baseline tags plus XResolution / YResolution / ResolutionUnit for the DPI,
//!   little-endian, strips of `rows_per_strip` rows, IFD after the data.
//!
//! The Group 4 code tables follow ITU-T T.4 / T.6 (the modified Huffman and
//! modified modified READ codes as the Recommendations print them).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{ NotTiff, Malformed, Unsupported, TooLarge } || Allocator.Error;

/// Largest image `decode` accepts (width times height); larger ones are `error.TooLarge`.
pub const max_pixels: usize = 1 << 28;

/// A size product of values read from a file; overflow is `error.TooLarge`.
fn mulSize(x: usize, y: usize) Error!usize {
    return std.math.mul(usize, x, y) catch return error.TooLarge;
}

// -- modified Huffman run-length tables (T.4) ---------------------------------------

const white_term = [_][]const u8{
    "00110101", "000111",   "0111",     "1000",     "1011",     "1100",     "1110",     "1111",     "10011",    "10100",    "00111",    "01000",    "001000",
    "000011",   "110100",   "110101",   "101010",   "101011",   "0100111",  "0001100",  "0001000",  "0010111",  "0000011",  "0000100",  "0101000",  "0101011",
    "0010011",  "0100100",  "0011000",  "00000010", "00000011", "00011010", "00011011", "00010010", "00010011", "00010100", "00010101", "00010110", "00010111",
    "00101000", "00101001", "00101010", "00101011", "00101100", "00101101", "00000100", "00000101", "00001010", "00001011", "01010010", "01010011", "01010100",
    "01010101", "00100100", "00100101", "01011000", "01011001", "01011010", "01011011", "01001010", "01001011", "00110010", "00110011", "00110100",
};

/// Make-up codes for white runs 64, 128, ..., 1728.
const white_makeup = [_][]const u8{
    "11011",     "10010",     "010111",    "0110111",   "00110110",  "00110111",  "01100100",  "01100101",  "01101000",  "01100111",
    "011001100", "011001101", "011010010", "011010011", "011010100", "011010101", "011010110", "011010111", "011011000", "011011001",
    "011011010", "011011011", "010011000", "010011001", "010011010", "011000",    "010011011",
};

const black_term = [_][]const u8{
    "0000110111",   "010",          "11",           "10",           "011",          "0011",         "0010",         "00011",        "000101",       "000100",       "0000100",      "0000101",      "0000111",
    "00000100",     "00000111",     "000011000",    "0000010111",   "0000011000",   "0000001000",   "00001100111",  "00001101000",  "00001101100",  "00000110111",  "00000101000",  "00000010111",  "00000011000",
    "000011001010", "000011001011", "000011001100", "000011001101", "000001101000", "000001101001", "000001101010", "000001101011", "000011010010", "000011010011", "000011010100", "000011010101", "000011010110",
    "000011010111", "000001101100", "000001101101", "000011011010", "000011011011", "000001010100", "000001010101", "000001010110", "000001010111", "000001100100", "000001100101", "000001010010", "000001010011",
    "000000100100", "000000110111", "000000111000", "000000100111", "000000101000", "000001011000", "000001011001", "000000101011", "000000101100", "000001011010", "000001100110", "000001100111",
};

/// Make-up codes for black runs 64, 128, ..., 1728.
const black_makeup = [_][]const u8{
    "0000001111",    "000011001000",  "000011001001",  "000001011011",  "000000110011",  "000000110100",  "000000110101",
    "0000001101100", "0000001101101", "0000001001010", "0000001001011", "0000001001100", "0000001001101", "0000001110010",
    "0000001110011", "0000001110100", "0000001110101", "0000001110110", "0000001110111", "0000001010010", "0000001010011",
    "0000001010100", "0000001010101", "0000001011010", "0000001011011", "0000001100100", "0000001100101",
};

/// Extended make-up codes shared by both colours: runs 1792, 1856, ..., 2560.
const ext_makeup = [_][]const u8{
    "00000001000",  "00000001100",  "00000001101",  "000000010010", "000000010011", "000000010100", "000000010101",
    "000000010110", "000000010111", "000000011100", "000000011101", "000000011110", "000000011111",
};

const Code = struct { bits: u16, len: u8 };

fn parseCode(s: []const u8) Code {
    var bits: u16 = 0;
    for (s) |c| bits = (bits << 1) | @as(u16, if (c == '1') 1 else 0);
    return .{ .bits = bits, .len = @intCast(s.len) };
}

fn parseCodes(comptime n: usize, comptime strings: [n][]const u8) [n]Code {
    @setEvalBranchQuota(20000);
    var out: [n]Code = undefined;
    for (strings, 0..) |s, i| out[i] = parseCode(s);
    return out;
}

const term_codes = [2][64]Code{ parseCodes(64, white_term), parseCodes(64, black_term) };
const makeup_codes = [2][27]Code{ parseCodes(27, white_makeup), parseCodes(27, black_makeup) };
const ext_codes = parseCodes(13, ext_makeup);

// -- bit writer ---------------------------------------------------------------------

const BitWriter = struct {
    out: std.ArrayList(u8) = .empty,
    acc: u32 = 0,
    n: u5 = 0,

    fn put(self: *BitWriter, a: Allocator, bits: u32, len: u5) Allocator.Error!void {
        self.acc = (self.acc << len) | bits;
        self.n += len;
        while (self.n >= 8) {
            try self.out.append(a, @truncate(self.acc >> (self.n - 8)));
            self.n -= 8;
            self.acc &= (@as(u32, 1) << self.n) - 1;
        }
    }

    fn code(self: *BitWriter, a: Allocator, c: Code) Allocator.Error!void {
        try self.put(a, c.bits, @intCast(c.len));
    }

    fn run(self: *BitWriter, a: Allocator, len_in: u32, black: bool) Allocator.Error!void {
        var len = len_in;
        const colour: usize = @intFromBool(black);
        while (len >= 2560) : (len -= 2560) try self.code(a, ext_codes[12]);
        if (len >= 1792) {
            try self.code(a, ext_codes[(len - 1792) / 64]);
            len %= 64;
        } else if (len >= 64) {
            try self.code(a, makeup_codes[colour][len / 64 - 1]);
            len %= 64;
        }
        try self.code(a, term_codes[colour][len]);
    }

    fn finish(self: *BitWriter, a: Allocator) Allocator.Error![]u8 {
        if (self.n > 0) try self.out.append(a, @truncate(self.acc << (8 - self.n)));
        return self.out.toOwnedSlice(a);
    }
};

/// Positions where the colour changes along a row (the pixel before the row
/// is white), followed by two `width` sentinels.
fn changes(row: []const u8, out: *std.ArrayList(u32), a: Allocator) Allocator.Error!void {
    out.clearRetainingCapacity();
    var black = false;
    var x: usize = 0;
    const V = @Vector(32, u8);
    const thr: V = @splat(128);
    while (x < row.len) {
        // Skip stretches without a change quickly.
        if (x + 32 <= row.len) {
            const v: V = row[x..][0..32].*;
            const is_black = v < thr;
            const want: @Vector(32, bool) = @splat(black);
            if (@reduce(.And, is_black == want)) {
                x += 32;
                continue;
            }
        }
        const b = row[x] < 128;
        if (b != black) {
            try out.append(a, @intCast(x));
            black = b;
        }
        x += 1;
    }
    const w: u32 = @intCast(row.len);
    try out.append(a, w);
    try out.append(a, w);
}

inline fn at(v: []const u32, i: usize, width: u32) u32 {
    return if (i < v.len) v[i] else width;
}

/// T.6 (CCITT Group 4) codestream of `h` rows of `w` 8-bit pixels (values
/// below 128 are black), EOFB appended, as libtiff's `Fax4Encode` writes it.
pub fn encodeG4Strip(a: Allocator, gray: []const u8, w: usize, h: usize) Allocator.Error![]u8 {
    const width: u32 = @intCast(w);
    var bw = BitWriter{};
    errdefer bw.out.deinit(a);
    var reference: std.ArrayList(u32) = .empty;
    defer reference.deinit(a);
    try reference.appendSlice(a, &.{ width, width });
    var coding: std.ArrayList(u32) = .empty;
    defer coding.deinit(a);
    for (0..h) |y| {
        try changes(gray[y * w .. (y + 1) * w], &coding, a);
        var a0: i64 = -1;
        var black = false; // colour of a0
        var ci: usize = 0; // first coding change > a0
        var ri: usize = 0; // first reference change > a0
        while (true) {
            while (ci < coding.items.len and coding.items[ci] <= a0) ci += 1;
            while (ri < reference.items.len and reference.items[ri] <= a0) ri += 1;
            const a1 = at(coding.items, ci, width);
            const a2 = at(coding.items, ci + 1, width);
            // b1: first reference change > a0 of the colour opposite to a0's;
            // even indices are changes to black, odd ones changes to white.
            const bi = if ((ri & 1 == 0) != black) ri else ri + 1;
            const b1 = at(reference.items, bi, width);
            const b2 = at(reference.items, bi + 1, width);
            if (b2 < a1) {
                try bw.put(a, 0b0001, 4); // pass mode
                a0 = b2;
            } else {
                const d = @as(i64, a1) - @as(i64, b1);
                if (d >= -3 and d <= 3) {
                    switch (d) {
                        0 => try bw.put(a, 0b1, 1),
                        1 => try bw.put(a, 0b011, 3),
                        2 => try bw.put(a, 0b000011, 6),
                        3 => try bw.put(a, 0b0000011, 7),
                        -1 => try bw.put(a, 0b010, 3),
                        -2 => try bw.put(a, 0b000010, 6),
                        else => try bw.put(a, 0b0000010, 7),
                    }
                    a0 = a1;
                    black = !black;
                } else {
                    try bw.put(a, 0b001, 3); // horizontal mode
                    const start: u32 = @intCast(@max(a0, 0));
                    try bw.run(a, a1 - start, black);
                    try bw.run(a, a2 - a1, !black);
                    a0 = a2;
                }
            }
            if (a0 >= width) break;
        }
        std.mem.swap(std.ArrayList(u32), &reference, &coding);
    }
    try bw.put(a, 0b000000000001, 12); // EOFB
    try bw.put(a, 0b000000000001, 12);
    return bw.finish(a);
}

// -- T.6 decoder -----------------------------------------------------------------------

const DecodeEntry = struct { run: u32, makeup: bool };

/// Run-length codes of one colour, indexed by (length, bits) in a flat table of 2^14 entries.
const DecodeTable = struct {
    entries: [1 << 14]?DecodeEntry,
    max_len: u8,

    fn key(len: u8, bits: u16) usize {
        return (@as(usize, len) << 9) | bits; // len <= 13 < 32, bits < 2^13
    }
};

fn buildDecodeTable(comptime colour: usize) DecodeTable {
    @setEvalBranchQuota(100000);
    var t = DecodeTable{ .entries = @splat(null), .max_len = 0 };
    for (term_codes[colour], 0..) |c, i| {
        t.entries[DecodeTable.key(c.len, c.bits)] = .{ .run = @intCast(i), .makeup = false };
        t.max_len = @max(t.max_len, c.len);
    }
    for (makeup_codes[colour], 0..) |c, i| {
        t.entries[DecodeTable.key(c.len, c.bits)] = .{ .run = @intCast((i + 1) * 64), .makeup = true };
        t.max_len = @max(t.max_len, c.len);
    }
    for (ext_codes, 0..) |c, i| {
        t.entries[DecodeTable.key(c.len, c.bits)] = .{ .run = @intCast(1792 + 64 * i), .makeup = true };
        t.max_len = @max(t.max_len, c.len);
    }
    return t;
}

const decode_tables = [2]DecodeTable{ buildDecodeTable(0), buildDecodeTable(1) };

const BitReader = struct {
    d: []const u8,
    pos: usize = 0,

    fn bit(self: *BitReader) u32 {
        const b: u32 = if (self.pos / 8 < self.d.len) (self.d[self.pos / 8] >> @intCast(7 - self.pos % 8)) & 1 else 0;
        self.pos += 1;
        return b;
    }

    fn peek(self: *const BitReader, n: usize) u32 {
        var v: u32 = 0;
        for (0..n) |i| {
            const p = self.pos + i;
            const b: u32 = if (p / 8 < self.d.len) (self.d[p / 8] >> @intCast(7 - p % 8)) & 1 else 0;
            v = (v << 1) | b;
        }
        return v;
    }

    fn exhausted(self: *const BitReader) bool {
        return self.pos >= self.d.len * 8;
    }

    /// A complete run (make-up codes followed by a terminating code).
    fn run(self: *BitReader, black: bool) ?u32 {
        const table = &decode_tables[@intFromBool(black)];
        var total: u32 = 0;
        while (true) {
            var bits: u16 = 0;
            var len: u8 = 0;
            var found: ?DecodeEntry = null;
            while (len < table.max_len) {
                bits = (bits << 1) | @as(u16, @intCast(self.bit()));
                len += 1;
                if (table.entries[DecodeTable.key(len, bits)]) |e| {
                    found = e;
                    break;
                }
            }
            const e = found orelse return null;
            total +|= e.run; // saturates: a run longer than any row is clamped by the caller
            if (!e.makeup) return total;
        }
    }
};

/// Decode a T.6 codestream of `w` x `h` pixels into 8-bit pixels (0 black,
/// 255 white). Decoding stops quietly at EOFB or when the data runs out (the
/// remaining rows stay white); malformed data is an error.
pub fn decodeG4Strip(a: Allocator, data: []const u8, w: usize, h: usize) Error![]u8 {
    @setRuntimeSafety(true); // untrusted input: bounds and overflow checks even in ReleaseFast
    var r = BitReader{ .d = data };
    const out = try a.alloc(u8, try mulSize(w, h));
    errdefer a.free(out);
    @memset(out, 255);
    const width: u32 = @intCast(w);
    var reference: std.ArrayList(u32) = .empty;
    defer reference.deinit(a);
    try reference.appendSlice(a, &.{ width, width });
    var coding: std.ArrayList(u32) = .empty;
    defer coding.deinit(a);
    for (0..h) |y| {
        if (r.exhausted() or r.peek(24) == 0x001001) break; // EOFB
        var a0: i64 = -1;
        var black = false;
        const row = out[y * w .. (y + 1) * w];
        var ri: usize = 0;
        while (true) {
            while (ri < reference.items.len and reference.items[ri] <= a0) ri += 1;
            const bi = if ((ri & 1 == 0) != black) ri else ri + 1;
            const b1 = at(reference.items, bi, width);
            const b2 = at(reference.items, bi + 1, width);
            if (r.peek(4) == 0b0001) {
                r.pos += 4; // pass mode
                if (black) paint(row, a0, b2);
                a0 = b2;
            } else if (r.peek(3) == 0b001) {
                r.pos += 3; // horizontal mode
                const r1 = r.run(black) orelse return error.Malformed;
                const r2 = r.run(!black) orelse return error.Malformed;
                const a1 = @min(@as(u32, @intCast(@max(a0, 0))) +| r1, width);
                const a2 = @min(a1 +| r2, width);
                if (black) paint(row, a0, a1) else paint(row, a1, a2);
                a0 = a2;
            } else {
                const d: i64 = if (r.bit() == 1) 0 else if (r.bit() == 1) (if (r.bit() == 1) @as(i64, 1) else -1) else blk: {
                    _ = r.bit();
                    _ = r.bit();
                    if (r.bit() == 1) {
                        break :blk if (r.bit() == 1) @as(i64, 2) else -2;
                    } else if (r.bit() == 1) {
                        break :blk if (r.bit() == 1) @as(i64, 3) else -3;
                    }
                    return error.Malformed; // EOL / garbage inside a row
                };
                const a1: u32 = @intCast(std.math.clamp(@as(i64, b1) + d, 0, @as(i64, width)));
                if (black) paint(row, a0, a1);
                a0 = a1;
                black = !black;
            }
            if (a0 >= width) break;
        }
        try changes(row, &coding, a);
        std.mem.swap(std.ArrayList(u32), &reference, &coding);
    }
    return out;
}

fn paint(row: []u8, from: i64, to: u32) void {
    const start: usize = @intCast(@max(from, 0));
    const end: usize = @min(@as(usize, to), row.len);
    if (end > start) @memset(row[start..end], 0);
}

// -- LZW (TIFF variant) -------------------------------------------------------------------

const lzw_clear = 256;
const lzw_eoi = 257;

/// TIFF LZW: MSB-first codes of 9..12 bits, "early change" width switching,
/// a ClearCode first and whenever the table reaches 4094 entries.
pub fn lzwEncode(a: Allocator, src: []const u8) Allocator.Error![]u8 {
    var bw = BitWriter{};
    errdefer bw.out.deinit(a);
    // Table as a hash from (prefix code, byte) to code.
    var table = std.AutoHashMap(u32, u16).init(a);
    defer table.deinit();
    var next_code: u16 = 258;
    var width: u5 = 9;
    try bw.put(a, lzw_clear, 9);
    if (src.len == 0) {
        try bw.put(a, lzw_eoi, 9);
        return bw.finish(a);
    }
    var prefix: u16 = src[0];
    for (src[1..]) |k| {
        const key = (@as(u32, prefix) << 8) | k;
        if (table.get(key)) |c| {
            prefix = c;
            continue;
        }
        try bw.put(a, prefix, width);
        try table.put(key, next_code);
        next_code += 1;
        if (next_code == 512 or next_code == 1024 or next_code == 2048) width += 1;
        if (next_code == 4094) {
            try bw.put(a, lzw_clear, width);
            table.clearRetainingCapacity();
            next_code = 258;
            width = 9;
        }
        prefix = k;
    }
    try bw.put(a, prefix, width);
    try bw.put(a, lzw_eoi, width);
    return bw.finish(a);
}

/// Decode TIFF LZW data into `expected` bytes (shorter output is padded with zeros).
pub fn lzwDecode(a: Allocator, data: []const u8, expected: usize) Error![]u8 {
    @setRuntimeSafety(true);
    const out = try a.alloc(u8, expected);
    errdefer a.free(out);
    @memset(out, 0);
    var pos: usize = 0;
    // Table entries as (prefix code, last byte, length); strings are rebuilt backwards.
    const LzwEntry = struct { prefix: u16, last: u8, len: u32 };
    var table = try a.alloc(LzwEntry, 4096);
    defer a.free(table);
    for (0..256) |i| table[i] = .{ .prefix = 0xffff, .last = @intCast(i), .len = 1 };
    var next_code: usize = 258;
    var width: u5 = 9;
    var r = BitReader{ .d = data };
    var prev: ?u16 = null;
    const total_bits = data.len * 8;
    while (r.pos + width <= total_bits) {
        const code: u16 = @intCast(r.peek(width));
        r.pos += width;
        if (code == lzw_clear) {
            next_code = 258;
            width = 9;
            prev = null;
            continue;
        }
        if (code == lzw_eoi) break;
        var emit_code = code;
        var first: u8 = undefined;
        if (code < next_code) {
            // Known string: emit it; its first byte extends the previous string.
            first = firstByte(table, code);
        } else if (code == next_code and prev != null) {
            // The KwKwK case: previous string plus its own first byte.
            first = firstByte(table, prev.?);
            table[next_code] = .{ .prefix = prev.?, .last = first, .len = table[prev.?].len + 1 };
            emit_code = @intCast(next_code);
        } else return error.Malformed;
        if (prev) |p| {
            if (code < next_code) {
                table[next_code] = .{ .prefix = p, .last = first, .len = table[p].len + 1 };
            }
            next_code += 1;
            if (next_code + 1 >= (@as(usize, 1) << width) and width < 12) width += 1;
        }
        // Write the string of emit_code backwards.
        const len = table[emit_code].len;
        if (pos + len > expected) return error.Malformed;
        var c = emit_code;
        var i: usize = pos + len;
        while (true) {
            i -= 1;
            out[i] = table[c].last;
            if (table[c].prefix == 0xffff) break;
            c = table[c].prefix;
        }
        pos += len;
        prev = emit_code;
        if (next_code >= 4096) return error.Malformed;
    }
    return out;
}

fn firstByte(table: anytype, code: u16) u8 {
    var c = code;
    while (table[c].prefix != 0xffff) c = table[c].prefix;
    return table[c].last;
}

// -- PackBits ---------------------------------------------------------------------------

fn packBitsDecode(data: []const u8, out: []u8) Error!void {
    @setRuntimeSafety(true);
    var i: usize = 0;
    var o: usize = 0;
    while (i < data.len and o < out.len) {
        const n: i8 = @bitCast(data[i]);
        i += 1;
        if (n >= 0) {
            const len: usize = @as(usize, @intCast(n)) + 1;
            if (i + len > data.len or o + len > out.len) return error.Malformed;
            @memcpy(out[o .. o + len], data[i .. i + len]);
            i += len;
            o += len;
        } else if (n != -128) {
            const len: usize = @as(usize, @intCast(-@as(i16, n))) + 1;
            if (i >= data.len or o + len > out.len) return error.Malformed;
            @memset(out[o .. o + len], data[i]);
            i += 1;
            o += len;
        }
    }
}

// -- Deflate ------------------------------------------------------------------------------

fn deflateEncode(a: Allocator, src: []const u8) Allocator.Error![]u8 {
    var zout = try std.Io.Writer.Allocating.initCapacity(a, src.len / 2 + 64);
    defer zout.deinit();
    const window = try a.alloc(u8, std.compress.flate.max_window_len);
    defer a.free(window);
    var comp = std.compress.flate.Compress.init(&zout.writer, window, .zlib, .level_6) catch return error.OutOfMemory;
    comp.writer.writeAll(src) catch return error.OutOfMemory;
    comp.finish() catch return error.OutOfMemory;
    return a.dupe(u8, zout.written());
}

fn deflateDecode(a: Allocator, data: []const u8, expected: usize) Error![]u8 {
    @setRuntimeSafety(true);
    var reader = std.Io.Reader.fixed(data);
    const window = try a.alloc(u8, std.compress.flate.max_window_len);
    defer a.free(window);
    var dec = std.compress.flate.Decompress.init(&reader, .zlib, window);
    const raw = dec.reader.allocRemaining(a, .unlimited) catch return error.Malformed;
    if (raw.len < expected) {
        a.free(raw);
        return error.Malformed;
    }
    return raw;
}

// -- container: writer ----------------------------------------------------------------------

pub const Compression = enum { none, g4, lzw, deflate };

/// How an 8-bit image is written.
pub const Options = struct {
    dpi: u32 = 0,
    /// Rows per strip; 0 means one strip for the whole image.
    rows_per_strip: usize = 0,
    compression: Compression = .none,
};

const Entry = struct { tag: u16, ty: u16, count: u32, data: []const u8 };

fn le16(v: u16) [2]u8 {
    return @bitCast(std.mem.nativeToLittle(u16, v));
}

fn le32(v: u32) [4]u8 {
    return @bitCast(std.mem.nativeToLittle(u32, v));
}

const Ifd = struct {
    a: Allocator,
    entries: std.ArrayList(Entry) = .empty,
    arena: std.heap.ArenaAllocator,

    fn init(a: Allocator) Ifd {
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a) };
    }

    fn deinit(self: *Ifd) void {
        self.entries.deinit(self.a);
        self.arena.deinit();
    }

    fn add(self: *Ifd, tag: u16, ty: u16, count: u32, data: []const u8) Allocator.Error!void {
        try self.entries.append(self.a, .{ .tag = tag, .ty = ty, .count = count, .data = try self.arena.allocator().dupe(u8, data) });
    }

    fn short(self: *Ifd, tag: u16, v: u16) Allocator.Error!void {
        try self.add(tag, 3, 1, &le16(v));
    }

    fn long(self: *Ifd, tag: u16, v: u32) Allocator.Error!void {
        try self.add(tag, 4, 1, &le32(v));
    }

    fn rational(self: *Ifd, tag: u16, num: u32, den: u32) Allocator.Error!void {
        try self.add(tag, 5, 1, &(le32(num) ++ le32(den)));
    }

    fn longs(self: *Ifd, tag: u16, vs: []const u32) Allocator.Error!void {
        const bytes = try self.arena.allocator().alloc(u8, vs.len * 4);
        for (vs, 0..) |v, i| bytes[i * 4 ..][0..4].* = le32(v);
        try self.entries.append(self.a, .{ .tag = tag, .ty = 4, .count = @intCast(vs.len), .data = bytes });
    }

    fn shorts(self: *Ifd, tag: u16, vs: []const u16) Allocator.Error!void {
        const bytes = try self.arena.allocator().alloc(u8, vs.len * 2);
        for (vs, 0..) |v, i| bytes[i * 2 ..][0..2].* = le16(v);
        try self.entries.append(self.a, .{ .tag = tag, .ty = 3, .count = @intCast(vs.len), .data = bytes });
    }
};

fn entryLess(_: void, p: Entry, q: Entry) bool {
    return p.tag < q.tag;
}

/// Little-endian TIFF: header, the strips from offset 8, then the IFD and its
/// out-of-line values.
fn writeTiff(a: Allocator, ifd: *Ifd, strips: []const []const u8) Allocator.Error![]u8 {
    var total: usize = 0;
    for (strips) |s| total += s.len + s.len % 2;
    const ifd_offset: u32 = @intCast(8 + total);
    std.mem.sort(Entry, ifd.entries.items, {}, entryLess);
    const n: u32 = @intCast(ifd.entries.items.len);
    var out = try std.ArrayList(u8).initCapacity(a, 8 + total + 2 + 12 * n + 4 + 64);
    errdefer out.deinit(a);
    try out.appendSlice(a, "II");
    try out.appendSlice(a, &le16(42));
    try out.appendSlice(a, &le32(ifd_offset));
    for (strips) |s| {
        try out.appendSlice(a, s);
        if (s.len % 2 == 1) try out.append(a, 0);
    }
    var extra: std.ArrayList(u8) = .empty;
    defer extra.deinit(a);
    const extra_base = ifd_offset + 2 + 12 * n + 4;
    try out.appendSlice(a, &le16(@intCast(n)));
    for (ifd.entries.items) |e| {
        try out.appendSlice(a, &le16(e.tag));
        try out.appendSlice(a, &le16(e.ty));
        try out.appendSlice(a, &le32(e.count));
        if (e.data.len <= 4) {
            var v: [4]u8 = .{ 0, 0, 0, 0 };
            @memcpy(v[0..e.data.len], e.data);
            try out.appendSlice(a, &v);
        } else {
            try out.appendSlice(a, &le32(@intCast(extra_base + extra.items.len)));
            try extra.appendSlice(a, e.data);
            if (extra.items.len % 2 == 1) try extra.append(a, 0);
        }
    }
    try out.appendSlice(a, &le32(0)); // no next IFD
    try out.appendSlice(a, extra.items);
    return out.toOwnedSlice(a);
}

fn stripOffsets(a: Allocator, strips: []const []const u8) Allocator.Error!struct { offsets: []u32, counts: []u32 } {
    const offsets = try a.alloc(u32, strips.len);
    errdefer a.free(offsets);
    const counts = try a.alloc(u32, strips.len);
    var off: u32 = 8;
    for (strips, 0..) |s, i| {
        offsets[i] = off;
        counts[i] = @intCast(s.len);
        off += @intCast(s.len + s.len % 2);
    }
    return .{ .offsets = offsets, .counts = counts };
}

fn baseIfd(ifd: *Ifd, w: usize, h: usize, dpi: u32, rows_per_strip: usize, strips: []const []const u8) Allocator.Error!void {
    const a = ifd.a;
    try ifd.long(256, @intCast(w));
    try ifd.long(257, @intCast(h));
    const so = try stripOffsets(a, strips);
    defer a.free(so.offsets);
    defer a.free(so.counts);
    try ifd.longs(273, so.offsets);
    try ifd.long(278, @intCast(rows_per_strip));
    try ifd.longs(279, so.counts);
    if (dpi > 0) {
        try ifd.rational(282, dpi, 1);
        try ifd.rational(283, dpi, 1);
        try ifd.short(296, 2);
    }
    try ifd.short(284, 1);
}

/// 1-bit CCITT Group 4 TIFF of a bilevel page given as 8-bit pixels (values
/// below 128 are black): PhotometricInterpretation 0 (white is zero),
/// FillOrder 1, strips of `rows_per_strip` rows (0: one strip).
pub fn encodeG4(a: Allocator, gray: []const u8, w: usize, h: usize, dpi: u32, rows_per_strip: usize) Allocator.Error![]u8 {
    std.debug.assert(gray.len == w * h and w > 0 and h > 0);
    const rps = if (rows_per_strip == 0 or rows_per_strip > h) h else rows_per_strip;
    const n = (h + rps - 1) / rps;
    const strips = try a.alloc([]u8, n);
    defer a.free(strips);
    var made: usize = 0;
    defer for (strips[0..made]) |s| a.free(s);
    for (0..n) |i| {
        const y0 = i * rps;
        const rows = @min(rps, h - y0);
        strips[i] = try encodeG4Strip(a, gray[y0 * w .. (y0 + rows) * w], w, rows);
        made += 1;
    }
    var ifd = Ifd.init(a);
    defer ifd.deinit();
    try baseIfd(&ifd, w, h, dpi, rps, strips);
    try ifd.short(258, 1);
    try ifd.short(259, 4);
    try ifd.short(262, 0);
    try ifd.short(266, 1);
    try ifd.short(277, 1);
    try ifd.long(293, 0); // T6Options
    return writeTiff(a, &ifd, strips);
}

/// 8-bit grey (`ch` = 1) or RGB (`ch` = 3) TIFF, uncompressed, LZW or Deflate.
pub fn encode8(a: Allocator, data: []const u8, w: usize, h: usize, ch: usize, opts: Options) Allocator.Error![]u8 {
    std.debug.assert(data.len == w * h * ch and (ch == 1 or ch == 3) and w > 0 and h > 0);
    const rps = if (opts.rows_per_strip == 0 or opts.rows_per_strip > h) h else opts.rows_per_strip;
    const n = (h + rps - 1) / rps;
    const strips = try a.alloc([]const u8, n);
    defer a.free(strips);
    var made: usize = 0;
    defer if (opts.compression != .none) for (strips[0..made]) |s| a.free(s);
    for (0..n) |i| {
        const y0 = i * rps;
        const rows = @min(rps, h - y0);
        const raw = data[y0 * w * ch .. (y0 + rows) * w * ch];
        strips[i] = switch (opts.compression) {
            .none => raw,
            .lzw => try lzwEncode(a, raw),
            .deflate => try deflateEncode(a, raw),
            .g4 => unreachable,
        };
        made += 1;
    }
    var ifd = Ifd.init(a);
    defer ifd.deinit();
    try baseIfd(&ifd, w, h, opts.dpi, rps, strips);
    const eights = [_]u16{ 8, 8, 8 };
    try ifd.shorts(258, eights[0..ch]);
    try ifd.short(259, switch (opts.compression) {
        .none => 1,
        .lzw => 5,
        .deflate => 8,
        .g4 => unreachable,
    });
    try ifd.short(262, if (ch == 3) 2 else 1);
    try ifd.short(277, @intCast(ch));
    return writeTiff(a, &ifd, strips);
}

// -- container: reader -------------------------------------------------------------------

pub fn isTiff(bytes: []const u8) bool {
    return bytes.len >= 8 and (std.mem.eql(u8, bytes[0..4], "II*\x00") or std.mem.eql(u8, bytes[0..4], "MM\x00*"));
}

/// A decoded image: 8-bit samples (a bilevel page comes back as 0 / 255 grey).
pub const Decoded = struct {
    w: usize,
    h: usize,
    ch: usize,
    data: []u8,
    /// XResolution in dots per inch when the file says so.
    dpi: ?u32,
    bits: u16,
    compression: u16,
    photometric: u16,
    rows_per_strip: usize,
    strips: usize,

    pub fn deinit(self: Decoded, a: Allocator) void {
        a.free(self.data);
    }
};

const Parser = struct {
    d: []const u8,
    le: bool,

    fn u16At(self: Parser, o: usize) Error!u16 {
        if (o + 2 > self.d.len) return error.Malformed;
        const v = std.mem.readInt(u16, self.d[o..][0..2], .little);
        return if (self.le) v else @byteSwap(v);
    }

    fn u32At(self: Parser, o: usize) Error!u32 {
        if (o + 4 > self.d.len) return error.Malformed;
        const v = std.mem.readInt(u32, self.d[o..][0..4], .little);
        return if (self.le) v else @byteSwap(v);
    }

    /// The values of entry `e` as u32 (BYTE, SHORT, LONG or the numerator of a RATIONAL).
    fn values(self: Parser, a: Allocator, e: usize) Error![]u32 {
        @setRuntimeSafety(true);
        const ty = try self.u16At(e + 2);
        const count: usize = try self.u32At(e + 4);
        const size: usize = switch (ty) {
            1, 2, 6, 7 => 1,
            3, 8 => 2,
            4, 9 => 4,
            5, 10 => 8,
            else => return error.Unsupported,
        };
        const base: usize = if (size * count <= 4) e + 8 else try self.u32At(e + 8);
        const out = try a.alloc(u32, count);
        errdefer a.free(out);
        for (out, 0..) |*v, i| {
            v.* = switch (size) {
                1 => blk: {
                    if (base + i >= self.d.len) return error.Malformed;
                    break :blk self.d[base + i];
                },
                2 => try self.u16At(base + 2 * i),
                4 => try self.u32At(base + 4 * i),
                else => blk: {
                    const num = try self.u32At(base + 8 * i);
                    const den = try self.u32At(base + 8 * i + 4);
                    break :blk if (den == 0) num else num / den;
                },
            };
        }
        return out;
    }
};

/// Decode the first image of a baseline TIFF (Group 4 or uncompressed 1-bit;
/// 8-bit grey / RGB uncompressed, LZW, Deflate or PackBits, with the
/// horizontal predictor; either byte order; FillOrder 1 or 2). Images above
/// `max_pixels` are `error.TooLarge`.
pub fn decode(a: Allocator, bytes: []const u8) Error!Decoded {
    @setRuntimeSafety(true); // untrusted input: bounds and overflow checks even in ReleaseFast
    if (!isTiff(bytes)) return error.NotTiff;
    const p = Parser{ .d = bytes, .le = bytes[0] == 'I' };
    const ifd: usize = try p.u32At(4);
    const n: usize = try p.u16At(ifd);
    var width: u32 = 0;
    var height: u32 = 0;
    var bits: u32 = 1;
    var compression: u32 = 1;
    var photometric: u32 = 1;
    var spp: u32 = 1;
    var rows_per_strip: u32 = std.math.maxInt(u32);
    var fill_order: u32 = 1;
    var predictor: u32 = 1;
    var planar: u32 = 1;
    var dpi: ?u32 = null;
    var unit: u32 = 2;
    var offsets: []u32 = &.{};
    defer a.free(offsets);
    var counts: []u32 = &.{};
    defer a.free(counts);
    for (0..n) |k| {
        const e = ifd + 2 + 12 * k;
        const tag = try p.u16At(e);
        switch (tag) {
            256, 257, 258, 259, 262, 277, 278, 266, 317, 284, 282, 296 => {
                const vs = try p.values(a, e);
                defer a.free(vs);
                if (vs.len == 0) continue;
                switch (tag) {
                    256 => width = vs[0],
                    257 => height = vs[0],
                    258 => bits = vs[0],
                    259 => compression = vs[0],
                    262 => photometric = vs[0],
                    277 => spp = vs[0],
                    278 => rows_per_strip = vs[0],
                    266 => fill_order = vs[0],
                    317 => predictor = vs[0],
                    284 => planar = vs[0],
                    282 => dpi = vs[0],
                    296 => unit = vs[0],
                    else => {},
                }
            },
            273 => {
                a.free(offsets);
                offsets = try p.values(a, e);
            },
            279 => {
                a.free(counts);
                counts = try p.values(a, e);
            },
            else => {},
        }
    }
    if (width == 0 or height == 0 or offsets.len == 0 or offsets.len != counts.len) return error.Malformed;
    if (planar != 1) return error.Unsupported;
    if (unit == 3 and dpi != null) dpi = std.math.lossyCast(u32, @round(@as(f64, @floatFromInt(dpi.?)) * 2.54));
    const w: usize = width;
    const h: usize = height;
    const pixels = try mulSize(w, h);
    if (pixels > max_pixels) return error.TooLarge;
    const rps: usize = @max(@min(rows_per_strip, height), 1);
    // 32-bit tag values in the file; every value decoded below fits in 16 bits.
    var result = Decoded{ .w = w, .h = h, .ch = spp, .data = &.{}, .dpi = dpi, .bits = @truncate(bits), .compression = @truncate(compression), .photometric = @truncate(photometric), .rows_per_strip = rps, .strips = offsets.len };
    if (bits == 1 and spp == 1) {
        const out = try a.alloc(u8, pixels);
        errdefer a.free(out);
        @memset(out, 255);
        const row_bytes = (w + 7) / 8;
        for (offsets, counts, 0..) |off, cnt, i| {
            const y0 = try mulSize(i, rps);
            if (y0 >= h) break;
            const rows = @min(rps, h - y0);
            if (@as(usize, off) + cnt > bytes.len) return error.Malformed;
            var strip = bytes[off .. off + cnt];
            var owned: ?[]u8 = null;
            defer if (owned) |o| a.free(o);
            if (fill_order == 2) {
                const rev = try a.alloc(u8, strip.len);
                for (rev, strip) |*r, s| r.* = @bitReverse(s);
                owned = rev;
                strip = rev;
            }
            switch (compression) {
                4 => {
                    const px = try decodeG4Strip(a, strip, w, rows);
                    defer a.free(px);
                    @memcpy(out[y0 * w .. (y0 + rows) * w], px);
                },
                1, 32773 => {
                    var raw = strip;
                    var unpacked: ?[]u8 = null;
                    defer if (unpacked) |u| a.free(u);
                    const packed_len = try mulSize(row_bytes, rows);
                    if (compression == 32773) {
                        const u = try a.alloc(u8, packed_len);
                        try packBitsDecode(strip, u);
                        unpacked = u;
                        raw = u;
                    }
                    if (raw.len < packed_len) return error.Malformed;
                    for (0..rows) |yy| for (0..w) |x| {
                        const bit = (raw[yy * row_bytes + x / 8] >> @intCast(7 - x % 8)) & 1;
                        // In fax files 1 is black (white is zero); MinIsBlack files store 1 for white.
                        out[(y0 + yy) * w + x] = if ((bit == 1) == (photometric == 0)) 0 else 255;
                    };
                },
                else => return error.Unsupported,
            }
            if (compression == 4 and photometric == 1) {
                // MinIsBlack: libtiff coded 0 bits as "white", so the fax colours are swapped.
                for (out[y0 * w .. (y0 + rows) * w]) |*v| v.* = 255 - v.*;
            }
        }
        result.data = out;
        return result;
    }
    if (bits != 8 or (spp != 1 and spp != 3 and spp != 4)) return error.Unsupported;
    const stride = try mulSize(w, spp);
    const out = try a.alloc(u8, try mulSize(stride, h));
    errdefer a.free(out);
    for (offsets, counts, 0..) |off, cnt, i| {
        const y0 = try mulSize(i, rps);
        if (y0 >= h) break;
        const rows = @min(rps, h - y0);
        if (@as(usize, off) + cnt > bytes.len) return error.Malformed;
        const strip = bytes[off .. off + cnt];
        const dst = out[y0 * stride .. (y0 + rows) * stride];
        switch (compression) {
            1 => {
                if (strip.len < dst.len) return error.Malformed;
                @memcpy(dst, strip[0..dst.len]);
            },
            5 => {
                const raw = try lzwDecode(a, strip, dst.len);
                defer a.free(raw);
                @memcpy(dst, raw[0..dst.len]);
            },
            8, 32946 => {
                const raw = try deflateDecode(a, strip, dst.len);
                defer a.free(raw);
                @memcpy(dst, raw[0..dst.len]);
            },
            32773 => try packBitsDecode(strip, dst),
            else => return error.Unsupported,
        }
        if (predictor == 2) {
            for (0..rows) |yy| {
                const row = dst[yy * stride .. (yy + 1) * stride];
                for (spp..stride) |x| row[x] +%= row[x - spp];
            }
        }
    }
    if (spp == 1 and photometric == 0) for (out) |*v| {
        v.* = 255 - v.*;
    };
    result.data = out;
    return result;
}

// -- tests -------------------------------------------------------------------------------

fn roundtripG4(a: Allocator, img: []const u8, w: usize, h: usize) !void {
    const data = try encodeG4Strip(a, img, w, h);
    defer a.free(data);
    const back = try decodeG4Strip(a, data, w, h);
    defer a.free(back);
    try std.testing.expectEqualSlices(u8, img, back);
}

fn pattern(a: Allocator, w: usize, h: usize, comptime f: fn (usize, usize) bool) ![]u8 {
    const img = try a.alloc(u8, w * h);
    for (0..h) |y| for (0..w) |x| {
        img[y * w + x] = if (f(x, y)) 0 else 255;
    };
    return img;
}

test "g4 roundtrips patterns" {
    const a = std.testing.allocator;
    const text = try pattern(a, 300, 60, struct {
        fn f(x: usize, y: usize) bool {
            return (y % 10 < 3 and (x / 7) % 3 == 0) or x == 150 or (x > 250 and y > 40) or (x == 17 and y == 5);
        }
    }.f);
    defer a.free(text);
    try roundtripG4(a, text, 300, 60);
    const white = try pattern(a, 5000, 3, struct {
        fn f(_: usize, _: usize) bool {
            return false;
        }
    }.f);
    defer a.free(white);
    try roundtripG4(a, white, 5000, 3);
    const black = try pattern(a, 5000, 3, struct {
        fn f(_: usize, _: usize) bool {
            return true;
        }
    }.f);
    defer a.free(black);
    try roundtripG4(a, black, 5000, 3);
    const runs = try pattern(a, 5500, 90, struct {
        fn f(x: usize, y: usize) bool {
            return x < y * 30 + 1;
        }
    }.f);
    defer a.free(runs);
    try roundtripG4(a, runs, 5500, 90);
    const mod = try pattern(a, 600, 70, struct {
        fn f(x: usize, y: usize) bool {
            return (x + y * 3) % (y + 2) == 0;
        }
    }.f);
    defer a.free(mod);
    try roundtripG4(a, mod, 600, 70);
}

test "tiff container roundtrips g4, grey and rgb" {
    const a = std.testing.allocator;
    const img = try pattern(a, 123, 45, struct {
        fn f(x: usize, y: usize) bool {
            return (x / 5 + y / 3) % 2 == 0;
        }
    }.f);
    defer a.free(img);
    for ([_]usize{ 0, 7 }) |rps| {
        const t = try encodeG4(a, img, 123, 45, 300, rps);
        defer a.free(t);
        try std.testing.expect(isTiff(t));
        const d = try decode(a, t);
        defer d.deinit(a);
        try std.testing.expectEqual(@as(usize, 123), d.w);
        try std.testing.expectEqual(@as(usize, 1), d.ch);
        try std.testing.expectEqual(@as(?u32, 300), d.dpi);
        try std.testing.expectEqual(@as(usize, if (rps == 0) 1 else 7), d.strips);
        try std.testing.expectEqualSlices(u8, img, d.data);
    }
    const gray = try a.alloc(u8, 50 * 20);
    defer a.free(gray);
    for (gray, 0..) |*v, i| v.* = @truncate(i * 5 + i / 50);
    for ([_]Compression{ .none, .lzw, .deflate }) |comp| {
        const t = try encode8(a, gray, 50, 20, 1, .{ .dpi = 72, .compression = comp, .rows_per_strip = 6 });
        defer a.free(t);
        const d = try decode(a, t);
        defer d.deinit(a);
        try std.testing.expectEqualSlices(u8, gray, d.data);
        try std.testing.expectEqual(@as(usize, 4), d.strips);
    }
    const rgb = try a.alloc(u8, 30 * 10 * 3);
    defer a.free(rgb);
    for (rgb, 0..) |*v, i| v.* = @truncate(i * 7);
    const t = try encode8(a, rgb, 30, 10, 3, .{ .compression = .lzw });
    defer a.free(t);
    const d = try decode(a, t);
    defer d.deinit(a);
    try std.testing.expectEqual(@as(usize, 3), d.ch);
    try std.testing.expectEqual(@as(?u32, null), d.dpi);
    try std.testing.expectEqualSlices(u8, rgb, d.data);
    try std.testing.expectError(error.NotTiff, decode(a, "not a tiff"));
}

test "tiff structure" {
    const a = std.testing.allocator;
    const img = try pattern(a, 64, 8, struct {
        fn f(x: usize, _: usize) bool {
            return x < 10;
        }
    }.f);
    defer a.free(img);
    const t = try encodeG4(a, img, 64, 8, 300, 0);
    defer a.free(t);
    try std.testing.expectEqualSlices(u8, "II\x2a\x00", t[0..4]);
    const ifd = std.mem.readInt(u32, t[4..8], .little);
    const n = std.mem.readInt(u16, t[ifd..][0..2], .little);
    try std.testing.expectEqual(@as(u16, 15), n);
    var tags: [15]u16 = undefined;
    for (&tags, 0..) |*tag, i| tag.* = std.mem.readInt(u16, t[ifd + 2 + 12 * i ..][0..2], .little);
    try std.testing.expectEqualSlices(u16, &.{ 256, 257, 258, 259, 262, 266, 273, 277, 278, 279, 282, 283, 284, 293, 296 }, &tags);
    try std.testing.expect(t.len < 240);
    const plain = try encode8(a, img, 64, 8, 1, .{ .dpi = 300 });
    defer a.free(plain);
    try std.testing.expectEqual(@as(usize, 8 + 64 * 8 + 2 + 12 * 13 + 4 + 16), plain.len);
}

test "lzw roundtrips and the width switches" {
    const a = std.testing.allocator;
    const src = try a.alloc(u8, 20000);
    defer a.free(src);
    var seed: u32 = 5;
    for (src, 0..) |*v, i| {
        seed = seed *% 1103515245 +% 12345;
        v.* = if (i % 3000 < 1500) @truncate(seed >> 24) else @truncate(i / 7);
    }
    const z = try lzwEncode(a, src);
    defer a.free(z);
    try std.testing.expect(z.len > 4094 / 8);
    const back = try lzwDecode(a, z, src.len);
    defer a.free(back);
    try std.testing.expectEqualSlices(u8, src, back);
    const empty = try lzwEncode(a, "");
    defer a.free(empty);
    try std.testing.expectEqualSlices(u8, &.{ 0x80, 0x40, 0x40 }, empty);
}
