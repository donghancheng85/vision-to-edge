"""Repository rule that wraps the system-installed OpenCV.

Discovers include subdirectories via pkg-config (opencv4) or falls back
to the standard /usr/include/opencv4 path.

Usage in MODULE.bazel:
    opencv_system_repo = use_repo_rule("//tools:opencv_system.bzl", "opencv_system_repo")
    opencv_system_repo(name = "opencv")
"""

def _opencv_system_repo_impl(rctx):
    # Try pkg-config first for accurate include/lib paths.
    # `pkg-config --variable=includedir opencv4` already returns the full
    # opencv4 include dir (e.g. /usr/include/opencv4), not a generic prefix.
    result = rctx.execute(["pkg-config", "--variable=includedir", "opencv4"])
    if result.return_code == 0:
        include_dir = result.stdout.strip()
    else:
        include_dir = "/usr/include/opencv4"

    result_lib = rctx.execute(["pkg-config", "--variable=libdir", "opencv4"])
    lib_dir = result_lib.stdout.strip() if result_lib.return_code == 0 else "/usr/lib/x86_64-linux-gnu"

    rctx.file("BUILD.bazel", content = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "opencv",
    hdrs = glob(["include/**/*.hpp", "include/**/*.h"]),
    includes = ["include"],
    linkopts = [
        "-Wl,-rpath,{lib_dir}",
        "-L{lib_dir}",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_highgui",
        "-lopencv_videoio",
    ],
    visibility = ["//visibility:public"],
)
""".format(include_dir = include_dir, lib_dir = lib_dir))

    # Copy (not symlink) so Bazel glob can find all files without follow_symlinks.
    copy_result = rctx.execute(["cp", "-rL", include_dir, "include"])
    if copy_result.return_code != 0:
        fail("opencv_system_repo: failed to copy '{}' -> include: {}".format(
            include_dir, copy_result.stderr))

opencv_system_repo = repository_rule(
    implementation = _opencv_system_repo_impl,
    configure = True,
    local = True,
    doc = "Exposes system-installed OpenCV 4 to Bazel builds.",
)
