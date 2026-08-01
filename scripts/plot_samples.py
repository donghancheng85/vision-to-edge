"""Visualise ground-truth bounding boxes from a YOLO-txt dataset.

Draws a grid of images with coloured bounding boxes and class-name labels.
Useful before training to confirm the dataset loaded correctly and the
label format is what the model expects.

Usage
─────
    # Show 16 random train images (saved to artifacts/sample_grid.png):
    uv run python scripts/plot_samples.py --data-path artifacts/data/coco128

    # Show 9 val images, custom output:
    uv run python scripts/plot_samples.py \\
        --data-path artifacts/data/coco128 --split val --n 9 \\
        --output artifacts/val_samples.png
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(no_args_is_help=False)

# 20-colour palette (BGR in cv2, converted to RGB for matplotlib)
_PALETTE = [
    (255,  56,  56), (255, 157,  56), (255, 212,  56), (255, 255,  56),
    ( 56, 255,  56), ( 56, 255, 157), ( 56, 255, 212), ( 56, 212, 255),
    ( 56, 157, 255), ( 56,  56, 255), (157,  56, 255), (212,  56, 255),
    (255,  56, 212), (255,  56, 157), (255, 165,   0), (  0, 255, 127),
    (127,   0, 255), (255,   0, 127), (  0, 127, 255), (127, 255,   0),
]


@app.command()
def main(
    data_path: Path = typer.Option(
        Path("artifacts/data/coco128"), "--data-path", "-d",
        help="Dataset root (must contain images/<split>/ and labels/<split>/)."),
    split: str   = typer.Option("train", "--split", "-s",
                                help="'train' or 'val'"),
    n: int       = typer.Option(16, "--n",
                                help="Number of images to show."),
    output: Path = typer.Option(
        Path("artifacts/sample_grid.png"), "--output", "-o",
        help="Where to save the output PNG."),
    names_yaml: Path | None = typer.Option(
        None, "--names-yaml",
        help="Dataset YAML with a 'names' key.  Defaults to COCO-80."),
    seed: int    = typer.Option(42, "--seed", help="Random seed for sampling."),
) -> None:
    """Plot a grid of labelled dataset images."""
    from training.common.coco_names import get_names

    class_names = get_names(names_yaml)
    img_dir = data_path / "images" / split
    lbl_dir = data_path / "labels" / split

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        typer.echo(f"[error] No images found in {img_dir}", err=True)
        raise typer.Exit(1)

    random.seed(seed)
    chosen = random.sample(img_files, min(n, len(img_files)))

    # ── Grid layout ───────────────────────────────────────────────────────────
    cols  = int(np.ceil(np.sqrt(len(chosen))))
    rows  = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.patch.set_facecolor("#1e1e1e")

    axes_flat = np.array(axes).flatten() if rows * cols > 1 else [axes]

    for ax, img_path in zip(axes_flat, chosen):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.axis("off")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        boxes    = _read_labels(lbl_path)

        _draw_boxes(ax, img_rgb, boxes, w, h, class_names)
        ax.set_title(img_path.name, color="white", fontsize=7, pad=2)
        ax.axis("off")

    # Hide unused axes
    for ax in axes_flat[len(chosen):]:
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    typer.echo(f"[ok] Saved {len(chosen)}-image grid → {output.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO-txt label file → list of (cls, cx, cy, w, h) normalised."""
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            result.append((cls, cx, cy, bw, bh))
    return result


def _draw_boxes(
    ax: "plt.Axes",
    img: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
    class_names: list[str],
) -> None:
    """Draw image and overlay bounding boxes with class labels."""
    ax.imshow(img)

    for cls, cx, cy, bw, bh in boxes:
        # Convert normalised cxcywh → pixel xyxy
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        pw = bw * img_w
        ph = bh * img_h

        color_rgb = tuple(c / 255 for c in _PALETTE[cls % len(_PALETTE)])
        name      = class_names[cls] if cls < len(class_names) else str(cls)

        # Bounding box rectangle
        rect = mpatches.FancyBboxPatch(
            (x1, y1), pw, ph,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color_rgb,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label background + text
        ax.text(
            x1, y1 - 2,
            name,
            color="white",
            fontsize=6,
            fontweight="bold",
            va="bottom",
            bbox=dict(facecolor=color_rgb, edgecolor="none", pad=1, alpha=0.85),
        )


if __name__ == "__main__":
    app()
