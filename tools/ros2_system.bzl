"""Repository rule that wraps a system-installed ROS 2 distribution.

Usage in MODULE.bazel:
    ros2_system_repo = use_repo_rule("//tools:ros2_system.bzl", "ros2_system_repo")
    ros2_system_repo(
        name  = "ros2_system",
        distro = "jazzy",
        prefix = "/opt/ros/jazzy",
    )

Then depend on it with:
    deps = ["@ros2_system//:rclcpp"]
"""

def _ros2_system_repo_impl(rctx):
    prefix = rctx.attr.prefix

    # Each ROS 2 package installs headers under its own subdirectory:
    #   /opt/ros/jazzy/include/<pkg>/<pkg>/*.hpp
    # Adding each package dir as an include root lets the compiler find
    # headers via their canonical path, e.g. #include <rclcpp/rclcpp.hpp>
    result = rctx.execute([
        "find",
        prefix + "/include",
        "-maxdepth", "1",
        "-mindepth", "1",
        "-type", "d",
        "-printf", "%f\n",
    ])
    if result.return_code != 0:
        fail("Failed to list ROS 2 include dirs under {}/include: {}".format(
            prefix,
            result.stderr,
        ))

    include_subdirs = [d for d in result.stdout.strip().split("\n") if d]
    includes = ["include/" + d for d in include_subdirs]

    rctx.file("BUILD.bazel", content = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

# Wraps the entire ROS 2 system installation as a single cc_library.
# Individual targets (yolo_detector, etc.) depend on this via
#   deps = ["@ros2_system//:rclcpp"]
cc_library(
    name = "rclcpp",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    includes = {includes},
    linkopts = [
        "-Wl,-rpath,{prefix}/lib",
        "-L{prefix}/lib",
        "-lrclcpp",
    ],
    visibility = ["//visibility:public"],
)
""".format(includes = repr(includes), prefix = prefix))

    # Symlink the full include tree into the repository root
    rctx.symlink(prefix + "/include", "include")

ros2_system_repo = repository_rule(
    implementation = _ros2_system_repo_impl,
    attrs = {
        "distro": attr.string(
            default = "jazzy",
            doc = "ROS 2 distribution name (informational).",
        ),
        "prefix": attr.string(
            mandatory = True,
            doc = "Absolute path to the ROS 2 installation root (e.g. /opt/ros/jazzy).",
        ),
    },
    configure = True,
    local = True,
    doc = "Exposes a system-installed ROS 2 distribution to Bazel builds.",
)
