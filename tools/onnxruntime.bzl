"""Repository rule that downloads the ONNX Runtime C++ SDK from GitHub releases.

The downloaded archive provides:
  include/onnxruntime_cxx_api.h   — header-only C++ wrapper
  include/onnxruntime_c_api.h     — C API (required by the C++ wrapper)
  lib/libonnxruntime.so           — shared library

Usage in MODULE.bazel:
    onnxruntime_repo = use_repo_rule("//tools:onnxruntime.bzl", "onnxruntime_repo")
    onnxruntime_repo(
        name    = "onnxruntime",
        version = "1.20.1",
        arch    = "x86_64",   # or "aarch64" for Orin NX
        gpu     = False,
    )

To switch to the CUDA-enabled build, set gpu=True and ensure CUDA is present.
The binary will then use CUDAExecutionProvider by default.
"""

def _onnxruntime_repo_impl(rctx):
    version = rctx.attr.version
    arch    = rctx.attr.arch     # "x86_64" or "aarch64"
    gpu     = rctx.attr.gpu

    # Map to the GitHub release asset naming convention
    arch_tag = "x64"       if arch == "x86_64" else "aarch64"
    gpu_tag  = "-gpu"      if gpu               else ""
    filename = "onnxruntime-linux-{arch}{gpu}-{ver}.tgz".format(
        arch = arch_tag, gpu = gpu_tag, ver = version)
    strip    = "onnxruntime-linux-{arch}{gpu}-{ver}".format(
        arch = arch_tag, gpu = gpu_tag, ver = version)

    url = "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/{fn}".format(
        ver = version, fn = filename)

    rctx.report_progress("Downloading ONNX Runtime C++ SDK v{} ...".format(version))
    rctx.download_and_extract(url = url, stripPrefix = strip)

    rctx.file("BUILD.bazel", content = """\
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

# cc_import provides the .so but cannot add -I flags.
# cc_library wraps it and sets includes so #include <onnxruntime_cxx_api.h> works.
cc_import(
    name = "_ort_so",
    shared_library = "lib/libonnxruntime.so",
)

cc_library(
    name = "onnxruntime",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    deps = [":_ort_so"],
    visibility = ["//visibility:public"],
)
""")

onnxruntime_repo = repository_rule(
    implementation = _onnxruntime_repo_impl,
    attrs = {
        "version": attr.string(mandatory = True,
                               doc = "ORT version, e.g. '1.20.1'"),
        "arch":    attr.string(default = "x86_64",
                               doc = "'x86_64' or 'aarch64'"),
        "gpu":     attr.bool(default = False,
                             doc = "Download the CUDA-enabled build."),
    },
    doc = "Downloads the ONNX Runtime C++ SDK from GitHub releases.",
)
