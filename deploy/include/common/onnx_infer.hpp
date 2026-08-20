#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vision {

/// Thin RAII wrapper around an ONNX Runtime inference session.
///
/// Tries CUDAExecutionProvider first; falls back to CPU when CUDA is
/// unavailable or the GPU architecture is not supported by this ORT build.
/// The fallback is silent — check the log for the active provider.
///
/// Usage:
///   OnnxSession session("model.onnx", /*use_cuda=*/true);
///   auto output = session.Run(input_f32, {1, 3, 640, 640});
class OnnxSession {
 public:
  /// @param model_path  Path to the .onnx file.
  /// @param use_cuda    Request the CUDA execution provider.
  /// @param device_id   GPU index (ignored when use_cuda=false).
  explicit OnnxSession(const std::string& model_path,
                       bool use_cuda  = true,
                       int  device_id = 0);
  ~OnnxSession();

  OnnxSession(const OnnxSession&)            = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  /// Run a single forward pass.
  /// @param input_data   Flat float32 buffer; must match input_shape.
  /// @param input_shape  Tensor dimensions, e.g. {1, 3, 640, 640}.
  /// @return             Flat float32 output buffer.
  std::vector<float> Run(const std::vector<float>&   input_data,
                          const std::vector<int64_t>& input_shape);

  /// Shape of the output tensor from the most recent Run() call.
  const std::vector<int64_t>& OutputShape() const { return output_shape_; }

  std::string InputName()  const;
  std::string OutputName() const;
  bool        UsingCuda()  const { return using_cuda_; }

 private:
  struct Impl;
  std::unique_ptr<Impl>  impl_;
  std::vector<int64_t>   output_shape_;
  bool                   using_cuda_{false};
};

}  // namespace vision
