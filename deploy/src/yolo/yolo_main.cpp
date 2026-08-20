/// YOLOv11 ONNX Runtime inference — CLI binary.
///
/// Runs detection on a single image, a directory of images, a video file,
/// or a webcam.  Saves annotated results to the output directory.
///
/// Build:
///   bazel build //deploy/src/yolo:yolo_main
///
/// Run (after installing the model):
///   ./bazel-bin/deploy/src/yolo/yolo_main \
///       --model  artifacts/models/yolo11n.onnx \
///       --source path/to/image.jpg
///   ./bazel-bin/deploy/src/yolo/yolo_main --source 0   # webcam

#include "yolo/yolo_detector.hpp"
#include "common/image_utils.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

// ── COCO-80 class names ────────────────────────────────────────────────────────────
static const std::vector<std::string> kCoco80 = {
  "person","bicycle","car","motorcycle","airplane","bus","train",
  "truck","boat","traffic light","fire hydrant","stop sign",
  "parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
  "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
  "baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon",
  "bowl","banana","apple","sandwich","orange","broccoli","carrot",
  "hot dog","pizza","donut","cake","chair","couch","potted plant",
  "bed","dining table","toilet","tv","laptop","mouse","remote",
  "keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear",
  "hair drier","toothbrush",
};

// ── Helpers ────────────────────────────────────────────────────────────────────
static const std::set<std::string> kImageExts{
  ".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif",
};

static bool IsImageFile(const fs::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return kImageExts.count(ext) > 0;
}

static void PrintUsage(const char* prog) {
  std::cout
    << "Usage: " << prog << " [options]\n"
    << "  --model   <path>   Path to .onnx file (required)\n"
    << "  --source  <path|N> Image file, image directory, video file,\n"
    << "                     or webcam index (0, 1, ...)\n"
    << "  --output  <dir>    Directory to save annotated output (default: output/)\n"
    << "  --conf    <float>  Confidence threshold (default: 0.25)\n"
    << "  --iou     <float>  NMS IoU threshold (default: 0.45)\n"
    << "  --no-cuda          Disable CUDA EP (use CPU only)\n"
    << "  --show             Display results in a window\n";
}

static void RunOnImage(vision::YoloDetector& det,
                       const fs::path&      img_path,
                       const fs::path&      out_dir,
                       bool                 show) {
  cv::Mat img = cv::imread(img_path.string());
  if (img.empty()) {
    std::cerr << "[skip] Cannot read: " << img_path << "\n";
    return;
  }

  auto t0 = std::chrono::steady_clock::now();
  auto dets = det.Detect(img);
  auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

  cv::Mat annotated = vision::DrawDetections(img, dets);
  fs::path out_path = out_dir / img_path.filename();
  cv::imwrite(out_path.string(), annotated);

  std::cout << img_path.filename().string()
            << "  " << dets.size() << " det(s)  " << ms << " ms\n";
  for (const auto& d : dets) {
    std::printf("  %-22s score=%.2f  [%4.0f,%4.0f,%4.0f,%4.0f]\n",
                d.class_name.c_str(), d.score,
                d.x1, d.y1, d.x2, d.y2);
  }

  if (show) {
    cv::imshow("YOLOv11", annotated);
    cv::waitKey(0);
  }
}

static void RunOnVideo(vision::YoloDetector& det,
                       const std::string&   source,
                       const fs::path&      out_dir,
                       bool                 show) {
  bool      is_cam = (source.size() == 1 && std::isdigit(source[0])) ||
                     (source.size() == 2 && std::isdigit(source[0]));
  cv::VideoCapture cap;
  if (is_cam) {
    cap.open(std::stoi(source));
  } else {
    cap.open(source);
  }
  if (!cap.isOpened()) {
    std::cerr << "[error] Cannot open video source: " << source << "\n";
    return;
  }

  const int w   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int h   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS) > 0
                      ? cap.get(cv::CAP_PROP_FPS) : 30.0;

  cv::VideoWriter writer;
  if (!is_cam) {
    fs::path out_vid = out_dir / (fs::path(source).stem().string() + "_out.mp4");
    writer.open(out_vid.string(),
                cv::VideoWriter::fourcc('m','p','4','v'),
                fps, {w, h});
  }

  int   frames  = 0;
  double fps_ema = 0.0;
  auto  t_prev  = std::chrono::steady_clock::now();
  cv::Mat frame;

  while (cap.read(frame)) {
    auto dets     = det.Detect(frame);
    auto annotated = vision::DrawDetections(frame, dets);

    auto t_now = std::chrono::steady_clock::now();
    double dt  = std::chrono::duration<double>(t_now - t_prev).count();
    t_prev     = t_now;
    fps_ema    = 0.9 * fps_ema + 0.1 * (1.0 / dt);

    // FPS overlay
    cv::putText(annotated,
                "FPS: " + std::to_string(static_cast<int>(fps_ema)),
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

    if (writer.isOpened()) writer.write(annotated);
    if (show) {
      cv::imshow("YOLOv11", annotated);
      if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }
    ++frames;
  }
  std::cout << "Processed " << frames << " frames  avg "
            << static_cast<int>(fps_ema) << " FPS\n";
}

// ── main ───────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  std::string model_path;
  std::string source;
  std::string out_dir = "output";
  float conf    = 0.25f;
  float iou     = 0.45f;
  bool  use_cuda = true;
  bool  show     = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if      (a == "--model"   && i+1 < argc) model_path = argv[++i];
    else if (a == "--source"  && i+1 < argc) source     = argv[++i];
    else if (a == "--output"  && i+1 < argc) out_dir    = argv[++i];
    else if (a == "--conf"    && i+1 < argc) conf       = std::stof(argv[++i]);
    else if (a == "--iou"     && i+1 < argc) iou        = std::stof(argv[++i]);
    else if (a == "--no-cuda")               use_cuda   = false;
    else if (a == "--show")                  show       = true;
    else if (a == "--help" || a == "-h") { PrintUsage(argv[0]); return 0; }
  }

  if (model_path.empty() || source.empty()) {
    PrintUsage(argv[0]); return 1;
  }

  // Create output directory
  fs::create_directories(out_dir);

  // Build detector
  vision::YoloDetector detector(model_path, kCoco80, conf, iou, use_cuda);
  std::cout << "Provider: " << (detector.UsingCuda() ? "CUDA" : "CPU") << "\n"
            << "Conf: " << conf << "  IoU: " << iou << "\n\n";

  fs::path src(source);

  // Directory of images
  if (fs::is_directory(src)) {
    for (const auto& entry : fs::directory_iterator(src)) {
      if (entry.is_regular_file() && IsImageFile(entry.path()))
        RunOnImage(detector, entry.path(), out_dir, show);
    }
    return 0;
  }

  // Single image file
  if (fs::is_regular_file(src) && IsImageFile(src)) {
    RunOnImage(detector, src, out_dir, show);
    return 0;
  }

  // Video / webcam
  RunOnVideo(detector, source, out_dir, show);
  return 0;
}
