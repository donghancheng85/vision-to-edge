#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vision {

struct Detection;  // forward declaration

/// Preprocess a BGR image for YOLOv11 inference.
///
/// Steps:
///   1. Resize to target_size × target_size (bilinear, keep aspect via padding
///      is NOT applied here — simple resize matches the Ultralytics ONNX export)
///   2. BGR → RGB
///   3. Normalise to [0, 1]  (divide by 255)
///   4. HWC → CHW
///   5. Wrap in a batch dimension: [1, 3, H, W]
///
/// @param orig_h  If non-null, receives the original image height.
/// @param orig_w  If non-null, receives the original image width.
/// @return        Flat float32 buffer ready to feed into OnnxSession::Run().
std::vector<float> PreprocessYolo(const cv::Mat& image_bgr,
                                    int            target_size,
                                    int*           orig_h = nullptr,
                                    int*           orig_w = nullptr);

/// Draw detection bounding boxes and labels on a copy of image_bgr.
/// Returns the annotated image (original is not modified).
cv::Mat DrawDetections(const cv::Mat&              image_bgr,
                        const std::vector<Detection>& dets);

}  // namespace vision
