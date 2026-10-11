const std = @import("std");

// zigcv on its own: `zig build test` runs the unit tests of every operation.
// Other projects import `src/root.zig` as the module `zigcv` (a path
// dependency, or `b.createModule` on this file).
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("zigcv", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tests = b.addTest(.{ .root_module = mod });
    const test_step = b.step("test", "Run the zigcv unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // `zig build docs`: the API documentation from the doc comments, in zig-out/docs.
    const docs_obj = b.addObject(.{ .name = "zigcv", .root_module = mod });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Build the API documentation");
    docs_step.dependOn(&install_docs.step);
}
