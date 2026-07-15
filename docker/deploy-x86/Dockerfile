# x86_64 deployment image
# ROS 2 Jazzy + ONNX Runtime + Bazel
# Usage:
#   docker build -t vision-edge:deploy-x86 -f docker/deploy-x86/Dockerfile .
#   docker run --rm -it vision-edge:deploy-x86

FROM ros:jazzy

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.20.1

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rclcpp \
    curl \
    ca-certificates \
    build-essential \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ── Bazelisk (auto-downloads Bazel version pinned in .bazelversion) ───────────
RUN curl -fsSL \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# ── ONNX Runtime C++ (CPU build; sufficient for x86 dev/test) ────────────────
RUN curl -fsSL \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    | tar -xz -C /usr/local --strip-components=1

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . .

# Source ROS 2 automatically in interactive shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# Pre-warm the Bazel server so the first build is faster
RUN bazel version > /dev/null 2>&1 || true

CMD ["/bin/bash"]
