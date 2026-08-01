"""Inspect YOLOv11: show input/output tensors and print model summary.

Runs both training mode (raw feature maps) and inference mode (decoded
boxes + scores), explaining what every dimension means.

Usage
─────
    uv run python scripts/inspect_model.py
    uv run python scripts/inspect_model.py --size s --batch 2
"""
from __future__ import annotations

from pathlib import Path

import torch
import typer
from torchinfo import summary

from training.common.coco_names import COCO_NAMES
from training.models.yolo.model import YOLOv11, _SCALES

app = typer.Typer(no_args_is_help=False)

DIV  = "─" * 68
DIV2 = "═" * 68


@app.command()
def main(
    size:  str = typer.Option("n", "--size", "-s",
                              help=f"Model size: {list(_SCALES.keys())}"),
    batch: int = typer.Option(1,   "--batch", "-b", help="Batch size"),
    imgsz: int = typer.Option(640, "--imgsz",       help="Input image size (square)"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu",
                               "--device"),
) -> None:
    """Show YOLOv11 input/output tensors and model summary."""

    dev = torch.device(device)
    typer.echo(f"\n{DIV2}")
    typer.echo(f"  YOLOv11-{size.upper()}   device={dev}   batch={batch}   imgsz={imgsz}")
    typer.echo(DIV2)

    # ── Build model ───────────────────────────────────────────────────────────
    model = YOLOv11(model_size=size, nc=80).to(dev)
    n_params  = sum(p.numel() for p in model.parameters())
    n_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"\n  Total params   : {n_params/1e6:.2f} M")
    typer.echo(f"  Trainable      : {n_trained/1e6:.2f} M")

    # ── Input ─────────────────────────────────────────────────────────────────
    typer.echo(f"\n{DIV}")
    typer.echo("  INPUT")
    typer.echo(DIV)
    x = torch.rand(batch, 3, imgsz, imgsz, device=dev)
    typer.echo(f"  Tensor shape   : {list(x.shape)}")
    typer.echo(f"  dtype          : {x.dtype}")
    typer.echo(f"  value range    : [{x.min():.3f}, {x.max():.3f}]")
    typer.echo(f"""
  Meaning of each dimension:
    dim 0 = batch size          ({batch} image{'s' if batch > 1 else ''})
    dim 1 = colour channels     (3 → RGB, values in [0, 1])
    dim 2 = image height        ({imgsz} pixels)
    dim 3 = image width         ({imgsz} pixels)
""")

    # ── Training mode output ──────────────────────────────────────────────────
    typer.echo(DIV)
    typer.echo("  TRAINING OUTPUT  (model.train()  → raw feature maps, no decoding)")
    typer.echo(DIV)
    model.train()
    with torch.no_grad():
        train_out = model(x)

    reg_max = model.detect.reg_max   # 16
    nc      = model.detect.nc        # 80
    strides = [8, 16, 32]
    grid_sizes = [imgsz // s for s in strides]  # [80, 40, 20] for 640

    typer.echo(f"\n  Returns a list of {len(train_out)} tensors, one per detection scale:\n")
    for i, (feat, stride, g) in enumerate(zip(train_out, strides, grid_sizes)):
        typer.echo(f"  scale {i}  stride={stride}x  grid={g}×{g}")
        typer.echo(f"    shape  : {list(feat.shape)}")
        typer.echo(f"    layout : [batch={batch}, channels={feat.shape[1]}, h={g}, w={g}]")
        typer.echo(f"             channels = 4×reg_max + nc = 4×{reg_max} + {nc} = {feat.shape[1]}")
        typer.echo(f"    anchor points in this scale : {g*g}")
        typer.echo()

    total_anchors = sum(g*g for g in grid_sizes)
    typer.echo(f"  Total anchor points across all scales: "
               f"{' + '.join(f'{g}²={g*g}' for g in grid_sizes)} = {total_anchors}")
    typer.echo()
    typer.echo(f"  How the {feat.shape[1]} channels are used during loss computation:")
    typer.echo(f"    feat[:, :4×{reg_max}, :, :]  = regression distribution")
    typer.echo(f"      4 coords × {reg_max} DFL bins each → DFL decodes to (left, top, right, bottom) offsets")
    typer.echo(f"    feat[:, {4*reg_max}:, :, :]  = classification logits")
    typer.echo(f"      {nc} values, one per COCO class, passed through BCE loss")

    # ── Inference mode output ─────────────────────────────────────────────────
    typer.echo(f"\n{DIV}")
    typer.echo("  INFERENCE OUTPUT  (model.eval()  → decoded boxes + scores)")
    typer.echo(DIV)
    model.eval()
    with torch.inference_mode():
        inf_out = model(x)

    B, N, C = inf_out.shape
    typer.echo(f"\n  shape : {list(inf_out.shape)}")
    typer.echo(f"  layout: [batch={B},  anchors={N},  4 + nc = 4 + {nc} = {C}]")
    typer.echo(f"""
  Each of the {N} rows is one candidate detection:
    inf_out[b, i, 0:4]  = bounding box in xyxy pixel coordinates
                          x1,y1 = top-left corner
                          x2,y2 = bottom-right corner
                          range: [0, {imgsz}]

    inf_out[b, i, 4:{4+nc}] = class confidence scores (after sigmoid)
                          {nc} values, one per COCO class
                          range: [0.0, 1.0]
""")

    # Show top-5 detections from image 0
    scores, cls_ids = inf_out[0, :, 4:].max(dim=-1)   # [N]
    top5_idx = scores.topk(5).indices

    typer.echo(f"  Top-5 detections in image 0 (randomly initialised weights → scores ≈ 0.01):")
    typer.echo(f"  {'rank':>4}  {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}  "
               f"{'score':>7}  class")
    typer.echo(f"  {'─'*4}  {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*7}  {'─'*20}")
    for rank, idx in enumerate(top5_idx, 1):
        x1, y1, x2, y2 = inf_out[0, idx, :4]
        score           = scores[idx].item()
        cls_id          = cls_ids[idx].item()
        cls_name        = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
        typer.echo(f"  {rank:>4}  {x1:6.1f} {y1:6.1f} {x2:6.1f} {y2:6.1f}  "
                   f"{score:7.4f}  [{cls_id:2d}] {cls_name}")

    typer.echo(f"""
  Note: scores are low (≈0.01) because the model uses random weights.
  After training on COCO128 for ~100 epochs, person/car detections
  should reach scores > 0.50 on seen images.

  Post-processing needed before using for detection:
    1. Filter by score threshold (e.g. score > 0.25)
    2. Apply Non-Maximum Suppression (NMS) per class
    3. torchvision.ops.batched_nms() handles both steps
""")

    # ── torchinfo summary ─────────────────────────────────────────────────────
    typer.echo(DIV)
    typer.echo("  MODEL SUMMARY  (torchinfo)")
    typer.echo(DIV)
    typer.echo()
    summary(
        model,
        input_size=(batch, 3, imgsz, imgsz),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=18,
        row_settings=["var_names"],
        depth=3,
        device=dev,
        verbose=1,
    )


if __name__ == "__main__":
    app()
