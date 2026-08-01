"""COCO 80-class names and a helper to read custom class names from a YAML file.

Used by:
  • scripts/plot_samples.py     — box labels in visualisations
  • training/models/yolo/       — logging during training
  • deploy/                     — class-name look-up in C++ (via generated header)
"""
from __future__ import annotations

from pathlib import Path

# ── COCO 80 class names (ordered by category ID) ─────────────────────────────
COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_names(yaml_path: Path | str | None = None) -> list[str]:
    """Return class names from a dataset YAML, falling back to COCO defaults.

    The YAML is expected to contain a ``names`` key with either a list or
    a dict of ``{id: name}`` pairs (both formats used in the wild):

        names: [cat, dog, bird]          # list form
        names: {0: cat, 1: dog, 2: bird} # dict form

    Args:
        yaml_path: Path to a dataset YAML file.  Pass ``None`` to get
                   the COCO-80 names without reading any file.
    """
    if yaml_path is None:
        return COCO_NAMES

    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("names", COCO_NAMES)
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw)]
    return list(raw)
