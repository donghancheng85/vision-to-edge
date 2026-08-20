#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vision {

/// One detected object.
struct Detection {
  float       x1, y1, x2, y2;  ///< Box corners in original image pixel coords
  float       score;            ///< Confidence in [0, 1]
  int         class_id;
  std::string class_name;
};

class OnnxSession;

/// YOLOv11 object detector.
///
/// Wraps pre/post-processing around an OnnxSession.
/// Expects Ultralytics ONNX export format:
///   output0 shape: [batch, 84, 8400]
///     rows  0-3:  cx, cy, w, h   (pixel coords in model input space)
///     rows  4-83: class scores   (sigmoid already applied by exporter)
///
/// No CUDA kernel code is required: the GPU is accessed entirely through
/// ONNX Runtime’s CUDAExecutionProvider.
class YoloDetector {
 public:
  /// @param model_path      Path to the .onnx file.
  /// @param class_names     Ordered list of class name strings (e.g. COCO-80).
  /// @param conf_threshold  Minimum score to keep a candidate detection.
  /// @param iou_threshold   NMS overlap threshold.
  /// @param use_cuda        Request CUDA execution provider (falls back to CPU).
  YoloDetector(const std::string&              model_path,
                const std::vector<std::string>& class_names,
                float conf_threshold = 0.25f,
                float iou_threshold  = 0.45f,
                bool  use_cuda       = true);
  ~YoloDetector();

  YoloDetector(const YoloDetector&)            = delete;
  YoloDetector& operator=(const YoloDetector&) = delete;

  /// Run detection on a BGR image (as returned by cv::imread).
  std::vector<Detection> Detect(const cv::Mat& image_bgr);

  bool UsingCuda() const;

 private:
  std::unique_ptr<OnnxSession>  session_;
  std::vector<std::string>      class_names_;
  float                         conf_threshold_;
  float                         iou_threshold_;
  int                           model_size_{640};

  std::vector<Detection> Decode(const std::vector<float>& output,
                                  int orig_h, int orig_w) const;

  static std::vector<Detection> Nms(std::vector<Detection> dets,
                                     float iou_threshold);
  static float IoU(const Detection& a, const Detection& b);
};

}  // namespace vision
