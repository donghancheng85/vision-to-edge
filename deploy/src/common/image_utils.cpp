#include "common/image_utils.hpp"
#include "yolo/yolo_detector.hpp"

#include <algorithm>
#include <array>

#include <opencv2/imgproc.hpp>

namespace vision {

// 20-colour BGR palette (one per class-id modulo 20)
static constexpr std::array<std::array<int, 3>, 20> kPalette = {{
  {{ 56, 56, 255}}, {{ 56,157,255}}, {{ 56,212,255}}, {{255,212, 56}},
  {{ 56,255, 56}}, {{ 56,255,157}}, {{255,157, 56}}, {{255, 56, 56}},
  {{157, 56,255}}, {{255, 56,212}}, {{  0,165,255}}, {{127,255,  0}},
  {{255,  0,127}}, {{  0,127,255}}, {{  0,255,127}}, {{255,127,  0}},
  {{  0,200,200}}, {{200,  0,200}}, {{200,200,  0}}, {{128,128,128}},
}};

// ── Preprocessing ─────────────────────────────────────────────────────────────
std::vector<float> PreprocessYolo(const cv::Mat& image_bgr,
                                    int            target_size,
                                    int*           orig_h,
                                    int*           orig_w) {
  if (orig_h) *orig_h = image_bgr.rows;
  if (orig_w) *orig_w = image_bgr.cols;

  cv::Mat resized;
  cv::resize(image_bgr, resized, {target_size, target_size},
             0, 0, cv::INTER_LINEAR);

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // HWC uint8 → CHW float32 in [0, 1]
  std::vector<float> blob(3 * target_size * target_size);
  const int area = target_size * target_size;

  for (int r = 0; r < target_size; ++r) {
    for (int c = 0; c < target_size; ++c) {
      const auto& px = rgb.at<cv::Vec3b>(r, c);
      blob[0 * area + r * target_size + c] = px[0] / 255.0f;  // R
      blob[1 * area + r * target_size + c] = px[1] / 255.0f;  // G
      blob[2 * area + r * target_size + c] = px[2] / 255.0f;  // B
    }
  }
  return blob;
}

// ── Drawing ───────────────────────────────────────────────────────────────────
cv::Mat DrawDetections(const cv::Mat&              image_bgr,
                        const std::vector<Detection>& dets) {
  cv::Mat out = image_bgr.clone();

  for (const auto& d : dets) {
    const auto& bgr = kPalette[d.class_id % kPalette.size()];
    cv::Scalar colour(bgr[0], bgr[1], bgr[2]);

    cv::Point tl(static_cast<int>(d.x1), static_cast<int>(d.y1));
    cv::Point br(static_cast<int>(d.x2), static_cast<int>(d.y2));
    cv::rectangle(out, tl, br, colour, 2);

    // Label background + text
    const std::string label =
        d.class_name + " " + std::to_string(static_cast<int>(d.score * 100)) + "%";
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                   0.5, 1, &baseline);
    cv::rectangle(out, {tl.x, tl.y - ts.height - 4},
                       {tl.x + ts.width, tl.y}, colour, -1);
    cv::putText(out, label, {tl.x, tl.y - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1,
                cv::LINE_AA);
  }
  return out;
}

}  // namespace vision
