"""Detection metrics backed by pycocotools (COCO-style mAP).

Expected input format
---------------------
Each call to update() receives per-image lists:
    preds:   list of np.ndarray shape (M, 6) -- [x1, y1, x2, y2, score, class_id]  (pixel coords)
    targets: list of np.ndarray shape (N, 5) -- [class_id, x1, y1, x2, y2]         (pixel coords)

All coordinates are absolute pixels in the resized (input_size × input_size) space.

Results
-------
compute() returns:
    val/mAP50-95  --  COCO AP averaged over IoU thresholds 0.50:0.05:0.95
    val/mAP50     --  AP at IoU = 0.50
    val/mAP75     --  AP at IoU = 0.75
    val/AR        --  Average recall at maxDets=100
"""
from __future__ import annotations

import contextlib
import io

import numpy as np


class DetectionMetrics:
    """Accumulate predictions across a validation epoch, then compute mAP."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._preds: list[dict] = []
        self._gts: list[dict] = []
        self._image_id = 0
        self._ann_id = 0

    # ── Accumulation ─────────────────────────────────────────────────────────

    def update(
        self,
        preds: list[np.ndarray],
        targets: list[np.ndarray],
    ) -> None:
        """Add predictions and ground truths for one batch."""
        for pred, gt in zip(preds, targets):
            img_id = self._image_id
            self._image_id += 1

            # Ground truth
            for row in gt:
                cls_id, x1, y1, x2, y2 = row
                w, h = float(x2 - x1), float(y2 - y1)
                self._gts.append({
                    "id": self._ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), w, h],  # COCO: [x, y, w, h]
                    "area": w * h,
                    "iscrowd": 0,
                })
                self._ann_id += 1

            # Predictions
            for row in pred:
                x1, y1, x2, y2, score, cls_id = row
                w, h = float(x2 - x1), float(y2 - y1)
                self._preds.append({
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                })

    # ── Evaluation ───────────────────────────────────────────────────────────

    def compute(self) -> dict[str, float]:
        """Run COCOeval and return summary metrics."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        images = [{"id": i} for i in range(self._image_id)]
        categories = [{"id": c, "name": str(c)} for c in range(self.num_classes)]

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images,
            "annotations": self._gts,
            "categories": categories,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()

        empty = {"val/mAP50-95": 0.0, "val/mAP50": 0.0, "val/mAP75": 0.0, "val/AR": 0.0}
        if not self._preds:
            return empty

        coco_dt = coco_gt.loadRes(self._preds)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        s = evaluator.stats  # 12-element array (see pycocotools docs)
        return {
            "val/mAP50-95": float(s[0]),  # AP @ IoU=0.50:0.95
            "val/mAP50":    float(s[1]),  # AP @ IoU=0.50
            "val/mAP75":    float(s[2]),  # AP @ IoU=0.75
            "val/AR":       float(s[8]),  # AR @ maxDets=100
        }

    def reset(self) -> None:
        self._preds.clear()
        self._gts.clear()
        self._image_id = 0
        self._ann_id = 0
