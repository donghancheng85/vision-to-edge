"""Base dataset classes and a format-agnostic DatasetAdapter.

Output contract (enforced by BaseDetectionDataset)
───────────────────────────────────────────────────
    __getitem__ returns:
        img     : Tensor[3, H, W]  float32, values in [0, 1]
        targets : Tensor[N, 5]    float32  — [cls, cx, cy, w, h]  (normalized)

    collate_fn (pass to DataLoader) prepends the per-image batch index:
        targets : Tensor[N, 6]    float32  — [img_idx, cls, cx, cy, w, h]

Supported formats
─────────────────
    "coco"     → COCODetectionDataset
                 root/images/{split}/*.jpg
                 root/annotations/{split}.json

    "yolo-txt" → YOLOTxtDataset
                 root/images/{split}/*.jpg
                 root/labels/{split}/*.txt   (one file per image)
                 each line: cls cx cy w h    (all values normalized to [0,1])

Add new formats by subclassing BaseDetectionDataset and registering the class
in DatasetAdapter._REGISTRY.
"""
from __future__ import annotations

import abc
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseDetectionDataset(Dataset, abc.ABC):
    """Interface shared by all detection datasets."""

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]: ...

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Stack images and prepend the batch index to each target row.

        This is the function you pass as collate_fn= to DataLoader.
        After collation targets have shape [N_total, 6]:
            col 0 = image index within the batch  (filled here)
            col 1 = class id
            col 2..5 = cx, cy, w, h  (normalized)
        """
        imgs, targets = zip(*batch)
        labelled: list[Tensor] = []
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                idx_col = torch.full((t.shape[0], 1), float(i))
                labelled.append(torch.cat([idx_col, t], dim=1))
        stacked = torch.stack(imgs)
        combined = torch.cat(labelled) if labelled else torch.zeros(0, 6)
        return stacked, combined


# ─────────────────────────────────────────────────────────────────────────────
# COCO JSON format
# ─────────────────────────────────────────────────────────────────────────────

class COCODetectionDataset(BaseDetectionDataset):
    """Loads a COCO-format dataset.

    Expected directory layout::

        root/
          images/train/  *.jpg | *.png
          images/val/    *.jpg | *.png
          annotations/
            train.json
            val.json

    COCO bounding boxes (in the JSON) are stored as [x, y, width, height] in
    absolute pixel coordinates.  We convert them to normalized [cx, cy, w, h]
    on load.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        input_size: int = 640,
        transforms: T.Compose | None = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.input_size = input_size
        self.transforms = transforms

        ann_file = self.root / "annotations" / f"{split}.json"
        self.coco = COCO(str(ann_file))
        # Only keep images that have at least one non-crowd annotation
        all_ids = list(self.coco.imgs.keys())
        self.img_ids = [
            iid for iid in all_ids
            if self.coco.getAnnIds(imgIds=iid, iscrowd=False)
        ]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_id = self.img_ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        img_path = self.root / "images" / info["file_name"]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        rows: list[list[float]] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]           # COCO: top-left pixel coords
            cx = (x + w / 2.0) / w0             # normalize to [0, 1]
            cy = (y + h / 2.0) / h0
            nw, nh = w / w0, h / h0
            rows.append([float(ann["category_id"]), cx, cy, nw, nh])

        targets_t = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 5)

        if self.transforms:
            img_t = self.transforms(img_t)

        return img_t, targets_t


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-txt format
# ─────────────────────────────────────────────────────────────────────────────

class YOLOTxtDataset(BaseDetectionDataset):
    """Loads a dataset in Ultralytics YOLO-txt format.

    Expected directory layout::

        root/
          images/train/  *.jpg | *.png
          images/val/    *.jpg | *.png
          labels/train/  *.txt
          labels/val/    *.txt

    Each label file has one detection per line::

        <class_id>  <cx>  <cy>  <w>  <h>

    All values are already normalized to [0, 1].
    Images without a corresponding label file are treated as background
    (zero targets).
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        input_size: int = 640,
        transforms: T.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.input_size = input_size
        self.transforms = transforms
        self.label_dir = self.root / "labels" / split

        img_dir = self.root / "images" / split
        self.img_files: list[Path] = sorted(
            list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        )
        if not self.img_files:
            raise FileNotFoundError(f"No images found under {img_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label_path = self.label_dir / img_path.with_suffix(".txt").name
        if label_path.exists():
            data = np.loadtxt(str(label_path), ndmin=2)   # shape [N, 5]
            targets_t = torch.tensor(data, dtype=torch.float32)
        else:
            targets_t = torch.zeros(0, 5)

        if self.transforms:
            img_t = self.transforms(img_t)

        return img_t, targets_t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset adapter / factory
# ─────────────────────────────────────────────────────────────────────────────

class DatasetAdapter:
    """Return the correct dataset class for a given format string.

    Register custom formats::

        DatasetAdapter._REGISTRY["my-format"] = MyDataset
    """

    _REGISTRY: dict[str, type[BaseDetectionDataset]] = {
        "coco":     COCODetectionDataset,
        "yolo-txt": YOLOTxtDataset,
    }

    @classmethod
    def build(
        cls,
        data_path: Path,
        split: str,
        dataset_format: str,
        input_size: int = 640,
        transforms: T.Compose | None = None,
    ) -> BaseDetectionDataset:
        if dataset_format not in cls._REGISTRY:
            raise ValueError(
                f"Unknown dataset format {dataset_format!r}. "
                f"Registered formats: {list(cls._REGISTRY)}"
            )
        return cls._REGISTRY[dataset_format](
            root=data_path,
            split=split,
            input_size=input_size,
            transforms=transforms,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Colour-space augmentations (image-only, no bbox adjustment needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentations(
    is_train: bool,
    *,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
) -> T.Compose | None:
    """Return a torchvision v2 colour-jitter pipeline for training.

    Why only colour-space augmentations here?
    Geometric augmentations (flip, mosaic, scale) require simultaneous
    transformation of bounding-box coordinates, so they are handled inside
    the per-model dataset (e.g. models/yolo/dataset.py).

    Returns None for validation (no augmentation).
    """
    if not is_train:
        return None

    ops: list[T.Transform] = []
    if any(x > 0 for x in (hsv_h, hsv_s, hsv_v)):
        ops.append(T.ColorJitter(
            brightness=hsv_v,
            contrast=hsv_v * 0.5,
            saturation=hsv_s,
            hue=min(hsv_h, 0.5),   # hue must be in [0, 0.5]
        ))
    return T.Compose(ops) if ops else None
