#include "common/onnx_infer.hpp"

#include <iostream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace vision {

// ── Singleton ORT environment (one per process) ──────────────────────────────
static Ort::Env& GetOrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vision-edge");
  return env;
}

// ── Pimpl ────────────────────────────────────────────────────────────────────
struct OnnxSession::Impl {
  Ort::Session                  session{nullptr};
  Ort::AllocatorWithDefaultOptions allocator;
};

// ── Constructor ──────────────────────────────────────────────────────────────
OnnxSession::OnnxSession(const std::string& model_path,
                         bool use_cuda,
                         int  device_id)
    : impl_(std::make_unique<Impl>()) {

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Try CUDA EP — falls back to CPU if anything goes wrong
  if (use_cuda) {
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = device_id;
    try {
      opts.AppendExecutionProvider_CUDA(cuda_opts);
      using_cuda_ = true;
    } catch (const Ort::Exception& e) {
      std::cerr << "[OnnxSession] CUDA EP unavailable (" << e.what()
                << "), falling back to CPU.\n";
    }
  }

  impl_->session = Ort::Session(GetOrtEnv(), model_path.c_str(), opts);

  std::cout << "[OnnxSession] Loaded: " << model_path
            << "  provider: " << (using_cuda_ ? "CUDA" : "CPU") << "\n";
}

OnnxSession::~OnnxSession() = default;

// ── Inference ─────────────────────────────────────────────────────────────────
std::vector<float> OnnxSession::Run(
    const std::vector<float>&   input_data,
    const std::vector<int64_t>& input_shape) {

  auto mem_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info,
      const_cast<float*>(input_data.data()), input_data.size(),
      input_shape.data(), input_shape.size());

  auto inp_name = impl_->session.GetInputNameAllocated(0, impl_->allocator);
  auto out_name = impl_->session.GetOutputNameAllocated(0, impl_->allocator);
  const char* input_names[]  = {inp_name.get()};
  const char* output_names[] = {out_name.get()};

  auto outputs = impl_->session.Run(
      Ort::RunOptions{nullptr},
      input_names,  &input_tensor, 1,
      output_names, 1);

  output_shape_ = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

  int64_t total = 1;
  for (auto d : output_shape_) total *= d;

  const float* data = outputs[0].GetTensorData<float>();
  return std::vector<float>(data, data + total);
}

std::string OnnxSession::InputName() const {
  return impl_->session.GetInputNameAllocated(0, impl_->allocator).get();
}

std::string OnnxSession::OutputName() const {
  return impl_->session.GetOutputNameAllocated(0, impl_->allocator).get();
}

}  // namespace vision
