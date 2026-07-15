# arm64 deployment image (NVIDIA Orin NX)
# Base: JetPack 6.x (Ubuntu 22.04) — TensorRT bundled by NVIDIA
# ROS 2: Humble (the distro for Ubuntu 22.04)
# Usage:
#   docker buildx build --platform linux/arm64 \
#     -t vision-edge:deploy-arm64 -f docker/deploy-arm64/Dockerfile .

# r36.x = JetPack 6.x, Ubuntu 22.04, CUDA 12.x, TensorRT 10.x
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.20.1

# ── ROS 2 Humble (Ubuntu 22.04 / jammy) ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release ca-certificates build-essential \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       ros-humble-rclcpp \
    && rm -rf /var/lib/apt/lists/*

# ── Bazelisk (auto-downloads Bazel version pinned in .bazelversion) ───────────
RUN curl -fsSL \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 \
    -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# ── ONNX Runtime C++ arm64 ────────────────────────────────────────────────────
# Used for models not yet converted to TensorRT engines.
# TensorRT itself is pre-installed by the JetPack base image.
RUN curl -fsSL \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz \
    | tar -xz -C /usr/local --strip-components=1

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . .

# Source ROS 2 automatically in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# Pre-warm the Bazel server so the first build is faster
RUN bazel version > /dev/null 2>&1 || true

CMD ["/bin/bash"]
