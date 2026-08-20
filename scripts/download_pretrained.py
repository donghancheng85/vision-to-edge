"""Download a pre-trained YOLOv11 model and export it to ONNX.

Uses the Ultralytics package to fetch official pre-trained weights and
produces a standard ONNX file ready for ONNX Runtime or TensorRT.

Output format (Ultralytics ONNX export):
    input  "images"  : [batch, 3, 640, 640]  float32  values in [0, 1]
    output "output0" : [batch, 84, 8400]     float32
        rows  0-3  : cx, cy, w, h  (pixel coords in the 640×640 input space)
        rows  4-83 : class scores  (sigmoid already applied by Ultralytics)

Usage
─────
    uv run python scripts/download_pretrained.py            # default: yolo11n
    uv run python scripts/download_pretrained.py --size s
    uv run python scripts/download_pretrained.py --size n --dest artifacts/models
"""
from __future__ import annotations

import shutil
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=False)


@app.command()
def main(
    size: str  = typer.Option("n",  "--size",  "-s",
                              help="Model variant: n / s / m / l / x"),
    dest: Path = typer.Option(Path("artifacts/models"), "--dest", "-d",
                              help="Directory to save the exported ONNX file."),
    imgsz: int = typer.Option(640,  "--imgsz",
                              help="Export input resolution (square)."),
    opset: int = typer.Option(17,   "--opset",
                              help="ONNX opset (17 = TensorRT 10+ compatible)."),
    half:  bool = typer.Option(False, "--half",
                               help="Export in fp16 (smaller, same accuracy on GPU)."),
) -> None:
    """Download pre-trained YOLOv11 weights and export to ONNX."""
    from ultralytics import YOLO

    valid = ("n", "s", "m", "l", "x")
    if size not in valid:
        typer.echo(f"[error] --size must be one of {valid}", err=True)
        raise typer.Exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    onnx_path = dest / f"yolo11{size}.onnx"

    if onnx_path.exists():
        typer.echo(f"[skip] {onnx_path} already exists. Delete it to re-export.")
        raise typer.Exit()

    # Ultralytics downloads the .pt weights automatically on first use
    typer.echo(f"Loading yolo11{size}.pt (auto-downloads on first call) …")
    model = YOLO(f"yolo11{size}.pt")

    typer.echo(f"Exporting to ONNX  opset={opset}  imgsz={imgsz}  half={half} …")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        half=half,
        dynamic=True,     # expose batch as a dynamic axis
        simplify=True,    # run onnx-simplifier for a cleaner graph
    )

    # Ultralytics saves next to the .pt file; move to our preferred location
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        shutil.move(str(exported_path), onnx_path)

    size_mb = onnx_path.stat().st_size / 1e6
    typer.echo(f"\n[done] {onnx_path.resolve()}  ({size_mb:.1f} MB)")
    typer.echo(f"\nNext step — build the C++ binary and run inference:")
    typer.echo(f"  bazel build //deploy/src/yolo:yolo_main")
    typer.echo(f"  ./bazel-bin/deploy/src/yolo/yolo_main \\")
    typer.echo(f"      --model {onnx_path} \\")
    typer.echo(f"      --source artifacts/data/coco128/images/val")


if __name__ == "__main__":
    app()
