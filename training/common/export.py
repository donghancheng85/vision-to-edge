"""PyTorch → ONNX export utility.

Typical usage:
    from training.common.export import export_onnx
    export_onnx(model, input_size=640, output_path="artifacts/models/yolo.onnx")

The exported ONNX file can then be:
  1. Validated locally with onnxruntime (CPU)
  2. Converted to TensorRT engine on the Orin NX via trtexec
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    input_size: int = 640,
    output_path: Path | str = "model.onnx",
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export *model* to ONNX and validate the resulting graph.

    Args:
        model:         Model to export.  Must be in eval() mode and on the
                       target device before calling.
        input_size:    Square input resolution (height == width, e.g. 640).
        output_path:   Destination .onnx file.  Parent dirs are created.
        opset:         ONNX opset version.  17 is recommended for
                       TensorRT 10+ compatibility.
        dynamic_batch: Whether to expose the batch dimension as dynamic.
                       Required for variable-batch-size inference in C++.

    Returns:
        Absolute path to the saved ONNX file.

    Raises:
        onnx.checker.ValidationError: if the exported graph fails validation.
    """
    import onnx

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "images":  {0: "batch_size"},
            "output0": {0: "batch_size"},
        }

    log.info("Exporting to ONNX (opset=%d): %s", opset, output_path)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=dynamic_axes,
    )

    # Validate — raises if the graph is malformed
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1e6
    log.info("ONNX export complete: %.1f MB, graph validated.", size_mb)
    return output_path
