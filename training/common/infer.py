"""ONNX Runtime inferencer for YOLOv11 (Ultralytics ONNX export format).

Handles pre-processing, inference, NMS post-processing, and result drawing.
Tries CUDAExecutionProvider first; falls back to CPU transparently.

ONNX output format (Ultralytics export):
    output0: [batch, 84, 8400]
        rows 0-3  : cx, cy, w, h  (pixel coords for the model input resolution)
        rows 4-83 : class scores  (sigmoid already applied by Ultralytics)
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# 20-colour palette (BGR — OpenCV convention)
_PALETTE = [
    ( 56,  56, 255), ( 56, 157, 255), ( 56, 212, 255), (255, 212,  56),
    ( 56, 255,  56), ( 56, 255, 157), (255, 157,  56), (255,  56,  56),
    (157,  56, 255), (255,  56, 212), (  0, 165, 255), (127, 255,   0),
    (255,   0, 127), (  0, 127, 255), (  0, 255, 127), (255, 127,   0),
    (  0, 200, 200), (200,   0, 200), (200, 200,   0), (128, 128, 128),
]


@dataclasses.dataclass
class Detection:
    """One detected object."""
    box:        np.ndarray   # [x1, y1, x2, y2] in original image pixel coords
    score:      float        # confidence in [0, 1]
    class_id:   int
    class_name: str


class OnnxInferencer:
    """Run YOLOv11 inference via ONNX Runtime.

    Args:
        model_path:      Path to the .onnx file (Ultralytics export format).
        conf_threshold:  Minimum confidence score to keep a detection.
        iou_threshold:   NMS IoU threshold.
        class_names:     Optional list of class name strings.
                         Defaults to COCO-80 names.
    """

    def __init__(
        self,
        model_path: Path | str,
        conf_threshold: float = 0.25,
        iou_threshold: float  = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        if class_names is None:
            from training.common.coco_names import COCO_NAMES
            self.class_names = COCO_NAMES
        else:
            self.class_names = class_names

        # ── ONNX Runtime session ──────────────────────────────────────────────
        available = ort.get_available_providers()   # list of strings
        providers  = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                      if "CUDAExecutionProvider" in available
                      else ["CPUExecutionProvider"])

        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])

        meta          = self.session.get_inputs()[0]
        self.inp_name = meta.name
        self.img_size = meta.shape[2] if isinstance(meta.shape[2], int) else 640

        active = self.session.get_providers()[0].replace("ExecutionProvider", "")
        log.info("Loaded %s  |  provider: %s  |  input: %s",
                 Path(model_path).name, active, meta.shape)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR image (uint8, as returned by cv2.imread)."""
        blob, orig_h, orig_w = self._preprocess(image_bgr)
        raw = self.session.run(None, {self.inp_name: blob})[0]   # [1, 84, 8400]
        return self._postprocess(raw, orig_h, orig_w)

    def draw(self, image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detections on a copy of *image_bgr* and return it."""
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box.astype(int)
            color = _PALETTE[det.class_id % len(_PALETTE)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    def _preprocess(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        orig_h, orig_w = image_bgr.shape[:2]
        img = cv2.resize(image_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]   # [1, 3, H, W]
        return np.ascontiguousarray(img), orig_h, orig_w

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> list[Detection]:
        # raw: [1, 84, 8400]  rows 0-3 = box, rows 4-83 = class scores
        preds        = raw[0].T                          # [8400, 84]
        boxes_xywh   = preds[:, :4]
        scores       = preds[:, 4:]

        class_scores = scores.max(axis=1)
        class_ids    = scores.argmax(axis=1)

        keep = class_scores >= self.conf_threshold
        if not keep.any():
            return []

        boxes_xywh   = boxes_xywh[keep]
        class_scores = class_scores[keep]
        class_ids    = class_ids[keep]

        # cx,cy,w,h → x1,y1,x2,y2 (still in model input space)
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Scale to original image size
        sx, sy = orig_w / self.img_size, orig_h / self.img_size
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy

        # Per-class NMS via torchvision
        from torchvision.ops import batched_nms
        import torch
        keep_idx = batched_nms(
            torch.tensor(boxes_xyxy, dtype=torch.float32),
            torch.tensor(class_scores, dtype=torch.float32),
            torch.tensor(class_ids, dtype=torch.int64),
            self.iou_threshold,
        ).numpy()

        results = []
        for i in keep_idx:
            cid  = int(class_ids[i])
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            results.append(Detection(
                box        = boxes_xyxy[i],
                score      = float(class_scores[i]),
                class_id   = cid,
                class_name = name,
            ))
        return results
