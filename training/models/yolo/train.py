"""YOLOv11 trainer + typer CLI entry point.

Usage examples
──────────────
    # Full training with defaults from configs/yolo.yaml:
    uv run python -m training.models.yolo.train /path/to/dataset

    # Override specific hyperparameters:
    uv run python -m training.models.yolo.train /data \\
        --model-size m --epochs 100 --batch-size 32 --lr0 0.005

    # Show all available options:
    uv run python -m training.models.yolo.train --help
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from training.common.base_trainer import BaseTrainer, TrainingConfig
from training.common.metrics import DetectionMetrics
from training.models.yolo.dataset import build_yolo_dataloader
from training.models.yolo.loss import YOLOv11Loss
from training.models.yolo.model import YOLOv11

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer implementation
# ─────────────────────────────────────────────────────────────────────────────

class YOLOTrainer(BaseTrainer):
    """Concrete trainer for YOLOv11.

    Implements the four abstract methods required by BaseTrainer:
        build_model()        — creates a YOLOv11 of the configured size
        build_dataloaders()  — wraps YOLODataset for train + val splits
        compute_loss()       — calls YOLOv11Loss
        compute_metrics()    — runs NMS then computes COCO mAP
    """

    PRIMARY_METRIC = "val/mAP50-95"

    # ── Abstract implementations ──────────────────────────────────────────────

    def build_model(self) -> YOLOv11:
        model = YOLOv11(model_size=self.cfg.model_size, nc=self.cfg.num_classes)
        log.info(
            "YOLOv11-%s  |  classes=%d  |  params=%.1fM",
            self.cfg.model_size,
            self.cfg.num_classes,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )
        if self.cfg.pretrained:
            state = torch.load(self.cfg.pretrained, map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("model", state), strict=False)
            log.info("Loaded pretrained weights from %s", self.cfg.pretrained)
        return model

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return (
            build_yolo_dataloader(self.cfg, "train"),
            build_yolo_dataloader(self.cfg, "val"),
        )

    def compute_loss(
        self, predictions: list[Tensor], targets: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        # Lazily create the loss function so the model is already on device
        if not hasattr(self, "_loss_fn"):
            self._loss_fn = YOLOv11Loss(
                self.model,
                box_gain=self.cfg.box_gain,
                cls_gain=self.cfg.cls_gain,
                dfl_gain=self.cfg.dfl_gain,
            )
        return self._loss_fn(predictions, targets)

    def compute_metrics(
        self, all_preds: list[Tensor], all_targets: list[Tensor]
    ) -> dict[str, float]:
        """Run NMS on decoded predictions and compute COCO mAP.

        The model is already in eval mode when this is called, so
        ``all_preds`` contains decoded [B, N, 4+nc] tensors.
        """
        from torchvision.ops import batched_nms

        metrics = DetectionMetrics(self.cfg.num_classes)
        s = float(self.cfg.input_size)

        for preds_batch, targets_batch in zip(all_preds, all_targets):
            # preds_batch: [B, N, 4+nc]  — xyxy pixel coords + sigmoid scores
            # targets_batch: [K, 6]      — [img_idx, cls, cx, cy, w, h] normalised
            B = preds_batch.shape[0]

            batch_np_preds:   list[np.ndarray] = []
            batch_np_targets: list[np.ndarray] = []

            for b in range(B):
                # ── Predictions ──
                p      = preds_batch[b]              # [N, 4+nc]
                boxes  = p[:, :4]                    # xyxy
                scores, cls_ids = p[:, 4:].max(dim=1)

                mask = scores > 0.25                 # confidence threshold
                if mask.any():
                    keep = batched_nms(boxes[mask], scores[mask], cls_ids[mask], iou_threshold=0.45)
                    det  = torch.cat([
                        boxes[mask][keep],
                        scores[mask][keep, None],
                        cls_ids[mask][keep, None].float(),
                    ], dim=1).cpu().numpy()          # [K, 6]: x1,y1,x2,y2,score,cls
                else:
                    det = np.zeros((0, 6))
                batch_np_preds.append(det)

                # ── Ground truth ──
                gt_rows = targets_batch[targets_batch[:, 0] == b]  # [M, 6]
                if gt_rows.shape[0] > 0:
                    cx  = gt_rows[:, 2] * s
                    cy  = gt_rows[:, 3] * s
                    w   = gt_rows[:, 4] * s
                    h   = gt_rows[:, 5] * s
                    gt_np = torch.stack(
                        [gt_rows[:, 1], cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1
                    ).cpu().numpy()                  # [M, 5]: cls, x1,y1,x2,y2
                else:
                    gt_np = np.zeros((0, 5))
                batch_np_targets.append(gt_np)

            metrics.update(batch_np_preds, batch_np_targets)

        return metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Typer CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def train(
    # Positional: dataset root
    data_path: Path = typer.Argument(..., help="Root directory of the dataset."),

    # Config file — all defaults live in the YAML
    config: Path = typer.Option(
        Path("training/configs/yolo.yaml"),
        "--config", "-c",
        help="Base hyperparameter YAML (values can be overridden below).",
    ),

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset_format: Optional[str] = typer.Option(None, help="'coco' or 'yolo-txt'"),
    input_size:     Optional[int] = typer.Option(None, help="Square input resolution."),
    num_classes:    Optional[int] = typer.Option(None, help="Number of detection classes."),
    num_workers:    Optional[int] = typer.Option(None, help="DataLoader worker count."),

    # ── Model ─────────────────────────────────────────────────────────────────
    model_size:  Optional[str]  = typer.Option(None, help="n/s/m/l/x"),
    pretrained:  Optional[Path] = typer.Option(None, help="Pretrained checkpoint path."),

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer:    Optional[str]   = typer.Option(None, help="'SGD' or 'AdamW'"),
    epochs:       Optional[int]   = typer.Option(None, help="Training epochs."),
    batch_size:   Optional[int]   = typer.Option(None, help="Batch size."),
    lr0:          Optional[float] = typer.Option(None, help="Initial learning rate."),
    lrf:          Optional[float] = typer.Option(None, help="Final LR = lr0 × lrf."),
    weight_decay: Optional[float] = typer.Option(None, help="L2 weight decay."),
    warmup_epochs: Optional[float] = typer.Option(None, help="Warmup epochs."),

    # ── Augmentation ──────────────────────────────────────────────────────────
    mosaic:  Optional[float] = typer.Option(None, help="Mosaic probability."),
    fliplr:  Optional[float] = typer.Option(None, help="Horizontal flip probability."),

    # ── Loss ──────────────────────────────────────────────────────────────────
    box_gain: Optional[float] = typer.Option(None, help="Box regression loss weight."),
    cls_gain: Optional[float] = typer.Option(None, help="Classification loss weight."),
    dfl_gain: Optional[float] = typer.Option(None, help="DFL loss weight."),

    # ── Output & logging ──────────────────────────────────────────────────────
    output_dir:    Optional[Path] = typer.Option(None, help="Where to save checkpoints."),
    project:       Optional[str]  = typer.Option(None, help="W&B project name."),
    run_name:      Optional[str]  = typer.Option(None, help="W&B run name."),
    resume:        Optional[Path] = typer.Option(None, help="Checkpoint to resume from."),
    seed:          Optional[int]  = typer.Option(None, help="Random seed."),
    mixed_precision: Optional[bool] = typer.Option(None, help="Enable AMP fp16."),
) -> None:
    """Train a YOLOv11 model.

    Loads defaults from CONFIG, then applies any explicitly-provided options.
    Only non-None values override the config, so you can override just what
    you need without specifying everything.
    """
    cfg = TrainingConfig.from_yaml(config)
    cfg = cfg.override(
        data_path=data_path,
        dataset_format=dataset_format,
        input_size=input_size,
        num_classes=num_classes,
        num_workers=num_workers,
        model_size=model_size,
        pretrained=pretrained,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        mosaic=mosaic,
        fliplr=fliplr,
        box_gain=box_gain,
        cls_gain=cls_gain,
        dfl_gain=dfl_gain,
        output_dir=output_dir,
        project=project,
        run_name=run_name,
        resume=resume,
        seed=seed,
        mixed_precision=mixed_precision,
    )

    log.info("Config: %s", cfg)
    trainer = YOLOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    app()
