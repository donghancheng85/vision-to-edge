#include "yolo/yolo_detector.hpp"
#include "common/onnx_infer.hpp"
#include "common/image_utils.hpp"

#include <algorithm>
#include <cmath>

namespace vision {

// ── Constructor / destructor ────────────────────────────────────────────────────
YoloDetector::YoloDetector(const std::string&              model_path,
                            const std::vector<std::string>& class_names,
                            float conf_threshold,
                            float iou_threshold,
                            bool  use_cuda)
    : session_(std::make_unique<OnnxSession>(model_path, use_cuda)),
      class_names_(class_names),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold) {}

YoloDetector::~YoloDetector() = default;

bool YoloDetector::UsingCuda() const { return session_->UsingCuda(); }

// ── Detect ─────────────────────────────────────────────────────────────────────
std::vector<Detection> YoloDetector::Detect(const cv::Mat& image_bgr) {
  int orig_h, orig_w;
  std::vector<float> blob = PreprocessYolo(
      image_bgr, model_size_, &orig_h, &orig_w);

  std::vector<int64_t> shape = {1, 3, model_size_, model_size_};
  std::vector<float>   output = session_->Run(blob, shape);

  return Nms(Decode(output, orig_h, orig_w), iou_threshold_);
}

// ── Decode [1, 84, 8400] → Detection list ─────────────────────────────────────────
std::vector<Detection> YoloDetector::Decode(
    const std::vector<float>& output,
    int orig_h, int orig_w) const {

  const auto& shape = session_->OutputShape();  // {1, 84, 8400}
  if (shape.size() < 3) return {};

  const int64_t rows    = shape[1];   // 84  (4 box coords + 80 classes)
  const int64_t anchors = shape[2];   // 8400
  const int     num_cls = static_cast<int>(rows) - 4;

  // Scale factors: model 640-space → original image pixels
  const float sx = static_cast<float>(orig_w) / model_size_;
  const float sy = static_cast<float>(orig_h) / model_size_;

  // output layout: [rows][anchors] (row-major)
  // output[r * anchors + a] = channel r, anchor a

  std::vector<Detection> dets;
  dets.reserve(64);

  for (int64_t a = 0; a < anchors; ++a) {
    // Find the highest-scoring class
    float best_score = 0.0f;
    int   best_cls   = 0;
    for (int c = 0; c < num_cls; ++c) {
      float s = output[(4 + c) * anchors + a];
      if (s > best_score) { best_score = s; best_cls = c; }
    }
    if (best_score < conf_threshold_) continue;

    // Decode bounding box (cx, cy, w, h) → (x1, y1, x2, y2)
    const float cx = output[0 * anchors + a];
    const float cy = output[1 * anchors + a];
    const float bw = output[2 * anchors + a];
    const float bh = output[3 * anchors + a];

    Detection d;
    d.x1         = std::max(0.0f, (cx - bw * 0.5f) * sx);
    d.y1         = std::max(0.0f, (cy - bh * 0.5f) * sy);
    d.x2         = std::min(static_cast<float>(orig_w), (cx + bw * 0.5f) * sx);
    d.y2         = std::min(static_cast<float>(orig_h), (cy + bh * 0.5f) * sy);
    d.score      = best_score;
    d.class_id   = best_cls;
    d.class_name = (best_cls < static_cast<int>(class_names_.size()))
                     ? class_names_[best_cls]
                     : std::to_string(best_cls);
    dets.push_back(d);
  }
  return dets;
}

// ── Greedy per-class NMS ────────────────────────────────────────────────────────────
float YoloDetector::IoU(const Detection& a, const Detection& b) {
  const float ix1 = std::max(a.x1, b.x1);
  const float iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2);
  const float iy2 = std::min(a.y2, b.y2);

  const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  if (inter == 0.0f) return 0.0f;

  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Detection> YoloDetector::Nms(
    std::vector<Detection> dets, float iou_threshold) {

  // Sort by score descending
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) {
              return a.score > b.score;
            });

  std::vector<bool>      suppressed(dets.size(), false);
  std::vector<Detection> result;
  result.reserve(dets.size());

  for (std::size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    result.push_back(dets[i]);
    for (std::size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) continue;
      // Only suppress same-class overlaps (per-class NMS)
      if (dets[i].class_id != dets[j].class_id) continue;
      if (IoU(dets[i], dets[j]) > iou_threshold) suppressed[j] = true;
    }
  }
  return result;
}

}  // namespace vision
