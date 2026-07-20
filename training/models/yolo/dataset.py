"""YOLO dataset: wraps the common adapters and adds mosaic + geometric augmentations.

Why a separate dataset?
  Common BaseDetectionDataset handles loading and colour jitter.
  YOLO needs mosaic (combines 4 images) and horizontal flip with bbox adjustment,
  which require operating on multiple samples — impossible in a generic base.

Mosaic augmentation
───────────────────
Four images are placed in the quadrants of a 2s×2s canvas around a random
centre point (xc, yc) ∈ [s/2, 3s/2].  The output is cropped to the final
s×s around that centre.  Bounding boxes are translated and clipped accordingly.
Boxes that become too small after clipping are removed.

Mosaic is disabled for the last ``close_mosaic`` epochs (see BaseTrainer).
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from training.common.base_trainer import TrainingConfig
from training.common.dataset import (
    BaseDetectionDataset,
    DatasetAdapter,
    build_augmentations,
)


class YOLODataset(BaseDetectionDataset):
    """YOLO-specific dataset with mosaic and horizontal flip.

    Args:
        root:     Dataset root (passed to DatasetAdapter).
        split:    ``"train"`` or ``"val"``.
        cfg:      TrainingConfig for format, input_size, augmentation params.
        is_train: Whether to apply augmentation (mosaic, flip, colour jitter).
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        cfg: TrainingConfig,
        is_train: bool = True,
    ) -> None:
        self.cfg      = cfg
        self.is_train = is_train
        self.s        = cfg.input_size          # convenience alias
        self._mosaic  = is_train and cfg.mosaic > 0

        colour_aug = build_augmentations(
            is_train=is_train,
            hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        )
        self._base = DatasetAdapter.build(
            data_path=Path(root),
            split=split,
            dataset_format=cfg.dataset_format,
            input_size=cfg.input_size,
            transforms=colour_aug,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mosaic(self, enable: bool) -> None:
        """Called by BaseTrainer to disable mosaic in the last N epochs."""
        self._mosaic = enable and self.is_train and self.cfg.mosaic > 0

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self._mosaic and random.random() < self.cfg.mosaic:
            img, targets = self._load_mosaic(idx)
        else:
            img, targets = self._base[idx]

        # Horizontal flip (with bbox x-coord adjustment)
        if self.is_train and random.random() < self.cfg.fliplr:
            img, targets = self._flip_lr(img, targets)

        return img, targets

    # ── Mosaic ────────────────────────────────────────────────────────────────

    def _load_mosaic(self, idx: int) -> tuple[Tensor, Tensor]:
        """Combine 4 images into a 2s×2s canvas, then crop to s×s."""
        s = self.s
        indices = [idx] + random.choices(range(len(self)), k=3)

        # Random centre of the 2×2 grid split (biased toward centre)
        xc = random.randint(s // 2, 3 * s // 2)
        yc = random.randint(s // 2, 3 * s // 2)

        canvas  = torch.zeros(3, 2 * s, 2 * s)
        all_tgts: list[Tensor] = []

        # Quadrant placement: TL, TR, BL, BR
        for i, src_idx in enumerate(indices):
            img, tgts = self._base[src_idx]

            if i == 0:   # top-left: bottom-right corner is at (xc, yc)
                x1, y1 = max(xc - s, 0), max(yc - s, 0)
                x2, y2 = xc, yc
                sx1 = s - (x2 - x1)
                sy1 = s - (y2 - y1)
            elif i == 1: # top-right: bottom-left corner is at (xc, yc)
                x1, y1 = xc, max(yc - s, 0)
                x2, y2 = min(xc + s, 2 * s), yc
                sx1 = 0
                sy1 = s - (y2 - y1)
            elif i == 2: # bottom-left: top-right corner is at (xc, yc)
                x1, y1 = max(xc - s, 0), yc
                x2, y2 = xc, min(yc + s, 2 * s)
                sx1 = s - (x2 - x1)
                sy1 = 0
            else:        # bottom-right: top-left corner is at (xc, yc)
                x1, y1 = xc, yc
                x2, y2 = min(xc + s, 2 * s), min(yc + s, 2 * s)
                sx1 = 0
                sy1 = 0

            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            canvas[:, y1:y2, x1:x2] = img[:, sy1:sy2, sx1:sx2]

            if tgts.shape[0] > 0:
                t = tgts.clone()
                # Shift bbox centres from [0,1] in their sub-image to
                # absolute pixel coords in the full 2s×2s canvas.
                t[:, 1] = tgts[:, 1] * s - sx1 + x1   # cx_px in canvas
                t[:, 2] = tgts[:, 2] * s - sy1 + y1   # cy_px in canvas
                t[:, 3] = tgts[:, 3] * s               # w_px
                t[:, 4] = tgts[:, 4] * s               # h_px
                all_tgts.append(t)

        # Crop s×s starting at (s//2, s//2) so the quadrant seam is centred
        cx0, cy0 = s // 2, s // 2
        out = canvas[:, cy0: cy0 + s, cx0: cx0 + s].clone()

        if not all_tgts:
            return out, torch.zeros(0, 5)

        t = torch.cat(all_tgts)

        # Translate to cropped space and normalise
        t[:, 1] = (t[:, 1] - cx0) / s
        t[:, 2] = (t[:, 2] - cy0) / s
        t[:, 3] = t[:, 3] / s
        t[:, 4] = t[:, 4] / s

        # Convert to xyxy for clipping, then back to cxcywh
        x1 = (t[:, 1] - t[:, 3] / 2).clamp(0, 1)
        y1 = (t[:, 2] - t[:, 4] / 2).clamp(0, 1)
        x2 = (t[:, 1] + t[:, 3] / 2).clamp(0, 1)
        y2 = (t[:, 2] + t[:, 4] / 2).clamp(0, 1)
        nw, nh = x2 - x1, y2 - y1

        valid = (nw > 0.01) & (nh > 0.01)
        cls  = t[:, 0][valid]
        cx_n = ((x1 + x2) / 2)[valid]
        cy_n = ((y1 + y2) / 2)[valid]
        return out, torch.stack([cls, cx_n, cy_n, nw[valid], nh[valid]], dim=1)

    # ── Horizontal flip ───────────────────────────────────────────────────────

    @staticmethod
    def _flip_lr(img: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        img = torch.flip(img, dims=[2])         # flip W dimension
        if targets.shape[0] > 0:
            t = targets.clone()
            t[:, 1] = 1.0 - targets[:, 1]      # cx → 1 - cx
            return img, t
        return img, targets


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_yolo_dataloader(
    cfg: TrainingConfig,
    split: str,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for the YOLO dataset.

    Args:
        cfg:     Full training config.
        split:   ``"train"`` or ``"val"``.
        shuffle: Override shuffle; default is True for train, False for val.
    """
    is_train = split == "train"
    dataset  = YOLODataset(cfg.data_path, split, cfg, is_train=is_train)
    _shuffle  = is_train if shuffle is None else shuffle

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=_shuffle,
        num_workers=cfg.num_workers,
        collate_fn=YOLODataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,          # avoid tiny last batch during training
    )
