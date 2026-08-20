"""Run YOLOv11 inference on an image, video file, or webcam (Python / ONNX Runtime).

Usage
─────
    # Single image
    uv run python scripts/infer.py artifacts/models/yolo11n.onnx --source image.jpg

    # Directory of images
    uv run python scripts/infer.py artifacts/models/yolo11n.onnx --source images/

    # Video file
    uv run python scripts/infer.py artifacts/models/yolo11n.onnx --source video.mp4

    # Webcam  (press q to quit)
    uv run python scripts/infer.py artifacts/models/yolo11n.onnx --source 0
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import typer

from training.common.infer import OnnxInferencer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    model:  Path  = typer.Argument(..., help="Path to .onnx model file."),
    source: str   = typer.Option(..., "--source", "-s",
                                 help="Image path, image dir, video file, or webcam index."),
    conf:   float = typer.Option(0.25, "--conf",  help="Confidence threshold."),
    iou:    float = typer.Option(0.45, "--iou",   help="NMS IoU threshold."),
    output: Path  = typer.Option(Path("artifacts/infer_out"), "--output", "-o"),
    show:   bool  = typer.Option(False, "--show",
                                 help="Display results in a window (requires display)."),
    names_yaml: Path | None = typer.Option(None, "--names-yaml"),
) -> None:
    """Run YOLOv11 ONNX inference (Python front-end)."""
    from training.common.coco_names import get_names

    class_names = get_names(names_yaml)
    inferencer  = OnnxInferencer(model, conf_threshold=conf,
                                 iou_threshold=iou, class_names=class_names)
    output.mkdir(parents=True, exist_ok=True)

    src = source.strip()
    src_path = Path(src)

    # Directory of images
    if src_path.is_dir():
        images = [p for p in sorted(src_path.iterdir())
                  if p.suffix.lower() in IMAGE_EXTS]
        for img_path in images:
            _run_image(inferencer, img_path, output, show)
        typer.echo(f"[done] {len(images)} images → {output}/")
        return

    # Single image
    if src_path.is_file() and src_path.suffix.lower() in IMAGE_EXTS:
        _run_image(inferencer, src_path, output, show)
        return

    # Video / webcam
    _run_video(inferencer, src, output, show)


def _run_image(inferencer, img_path, output, show):
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("Cannot read %s — skipping.", img_path)
        return

    t0   = time.perf_counter()
    dets = inferencer.run(img)
    ms   = (time.perf_counter() - t0) * 1000

    annotated = inferencer.draw(img, dets)
    cv2.imwrite(str(output / img_path.name), annotated)

    log.info("%-30s  %2d det(s)  %5.1f ms", img_path.name, len(dets), ms)
    for d in dets:
        log.info("  %-20s score=%.2f  box=%s",
                 d.class_name, d.score, d.box.astype(int).tolist())

    if show:
        cv2.imshow("YOLOv11", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_video(inferencer, source, output, show):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        log.error("Cannot open: %s", source); raise typer.Exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_cam = source.isdigit()

    writer = None
    if not is_cam:
        out_p  = output / (Path(source).stem + "_out.mp4")
        writer = cv2.VideoWriter(str(out_p),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    frames, fps_ema = 0, 0.0
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        dets      = inferencer.run(frame)
        annotated = inferencer.draw(frame, dets)
        t_now     = time.perf_counter()
        fps_ema   = 0.9 * fps_ema + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev    = t_now
        cv2.putText(annotated, f"FPS: {fps_ema:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if writer: writer.write(annotated)
        if show:
            cv2.imshow("YOLOv11", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"): break
        frames += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    log.info("Processed %d frames  avg %.1f FPS", frames, fps_ema)


if __name__ == "__main__":
    app()
