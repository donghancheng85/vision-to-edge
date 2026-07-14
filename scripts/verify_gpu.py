"""Quick sanity-check: PyTorch version and GPU accessibility."""
import sys
import torch

print(f"PyTorch : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected.", file=sys.stderr)
    sys.exit(1)

print(f"CUDA runtime : {torch.version.cuda}")
print(f"cuDNN version : {torch.backends.cudnn.version()}")
print(f"GPU count : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GB  "
          f"SM {p.major}.{p.minor}")

# Smoke-test: allocate a tiny tensor on GPU
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
assert x.sum().item() == 6.0
print("\nGPU tensor ops : OK")
