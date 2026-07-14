// Minimal ROS 2 verification: include the rclcpp version header and print the
// version constants.  No node creation — avoids runtime rmw plugin dependency.
#include <rclcpp/version.h>
#include <cstdio>

int main()
{
    std::printf(
        "ROS 2 rclcpp version : %d.%d.%d\n",
        RCLCPP_VERSION_MAJOR,
        RCLCPP_VERSION_MINOR,
        RCLCPP_VERSION_PATCH);
    std::printf("Bazel bzlmod + system ROS 2 Jazzy : OK\n");
    return 0;
}
