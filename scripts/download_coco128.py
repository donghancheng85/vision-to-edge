"""Download and organise COCO128 for YOLO training.

COCO128 is a 128-image subset of COCO train2017 widely used as an
"educational" dataset for YOLO development.

Download size  : ~7 MB
Output size    : ~7 MB
Classes        : 80 (full COCO set)
Images         : 102 train / 26 val  (80/20 deterministic split)
Label format   : YOLO-txt  (cls cx cy w h, normalised)

Usage
─────
    uv run python scripts/download_coco128.py
    uv run python scripts/download_coco128.py --dest /custom/path
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=False)

URL  = "https://ultralytics.com/assets/coco128.zip"
DEST_DEFAULT = Path("artifacts/data/coco128")


@app.command()
def main(
    dest: Path = typer.Option(DEST_DEFAULT, "--dest", "-d",
                              help="Output directory for the organised dataset."),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Re-download even if dest already exists."),
) -> None:
    """Download COCO128 and split into train/val for YOLO training."""
    if dest.exists() and not force:
        typer.echo(f"[skip] {dest} already exists. Use --force to re-download.")
        raise typer.Exit()

    # ── Download ──────────────────────────────────────────────────────────────
    tmp_zip = Path("artifacts/data/_coco128.zip")
    tmp_zip.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading {URL} …")
    urllib.request.urlretrieve(URL, tmp_zip, _progress_hook)
    typer.echo()  # newline after progress bar

    # ── Extract ───────────────────────────────────────────────────────────────
    raw_dir = Path("artifacts/data/coco128_raw")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    typer.echo(f"Extracting to {raw_dir} …")
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(raw_dir)
    tmp_zip.unlink()

    # COCO128 extracts to: raw_dir/coco128/images/train2017/ and labels/train2017/
    src_imgs   = raw_dir / "coco128" / "images"   / "train2017"
    src_labels = raw_dir / "coco128" / "labels"   / "train2017"

    if not src_imgs.exists():
        typer.echo(f"[error] Expected {src_imgs} after extraction.", err=True)
        raise typer.Exit(1)

    # ── Split 80/20 (deterministic: sorted filenames) ─────────────────────────
    all_imgs = sorted(src_imgs.glob("*.jpg")) + sorted(src_imgs.glob("*.png"))
    n_train  = int(len(all_imgs) * 0.8)
    train_imgs, val_imgs = all_imgs[:n_train], all_imgs[n_train:]

    typer.echo(f"Splitting: {len(train_imgs)} train / {len(val_imgs)} val")

    dest.mkdir(parents=True, exist_ok=True)
    _copy_split(train_imgs, src_labels, dest, "train")
    _copy_split(val_imgs,   src_labels, dest, "val")

    # ── Cleanup raw ───────────────────────────────────────────────────────────
    shutil.rmtree(raw_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(dest)
    typer.echo(f"\n[done] Dataset ready at: {dest.resolve()}")
    typer.echo(f"       Train: uv run python -m training.models.yolo.train {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _copy_split(img_paths: list[Path], src_labels: Path, dest: Path, split: str) -> None:
    img_out = dest / "images" / split
    lbl_out = dest / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in img_paths:
        shutil.copy(img, img_out / img.name)
        lbl = src_labels / img.with_suffix(".txt").name
        if lbl.exists():
            shutil.copy(lbl, lbl_out / lbl.name)


def _print_summary(dest: Path) -> None:
    from training.common.coco_names import COCO_NAMES

    typer.echo("\n── Dataset summary ──────────────────────────────")
    for split in ("train", "val"):
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        imgs  = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls  = list(lbl_dir.glob("*.txt"))

        cls_counter: Counter = Counter()
        box_count = 0
        for lbl in lbls:
            lines = lbl.read_text().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_counter[int(parts[0])] += 1
                    box_count += 1

        top5 = cls_counter.most_common(5)
        top5_str = "  ".join(
            f"{COCO_NAMES[c] if c < len(COCO_NAMES) else c}({n})"
            for c, n in top5
        )
        typer.echo(f"  {split:5s}  images={len(imgs):4d}  boxes={box_count:5d}  "
                   f"classes={len(cls_counter):2d}  top5: {top5_str}")
    typer.echo("─────────────────────────────────────────────────")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    typer.echo(f"\r  [{bar}] {pct:3d}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB",
               nl=False)


if __name__ == "__main__":
    app()
