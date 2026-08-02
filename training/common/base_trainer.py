"""Base trainer: TrainingConfig dataclass + abstract BaseTrainer class.

All model-specific trainers (YOLOTrainer, DETRTrainer, BEVFormerTrainer) inherit
from BaseTrainer and implement exactly four abstract methods:

    build_model()        → nn.Module
    build_dataloaders()  → (train_loader, val_loader)
    compute_loss()       → (loss_tensor, {component_name: float})
    compute_metrics()    → {metric_name: float}
"""
from __future__ import annotations

import abc
import dataclasses
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB = False


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrainingConfig:
    """All training hyperparameters in a single dataclass.

    Load defaults from a YAML file, then override individual fields via
    the typer CLI:

        cfg = TrainingConfig.from_yaml("training/configs/yolo.yaml")
        cfg = cfg.override(epochs=50, lr0=1e-3)   # CLI overrides
    """

    # ── Data ─────────────────────────────────────────────────────────────────
    data_path: Path = dataclasses.field(default_factory=lambda: Path("data"))
    dataset_format: str = "yolo-txt"   # "coco" | "yolo-txt"
    input_size: int = 640
    num_classes: int = 80
    num_workers: int = 8

    # ── Model ────────────────────────────────────────────────────────────────
    model_size: str = "s"              # n | s | m | l | x
    pretrained: Path | None = None

    # ── Optimizer & LR schedule ──────────────────────────────────────────────
    optimizer: str = "AdamW"           # "SGD" | "AdamW"
    lr0: float = 0.01                  # initial learning rate
    lrf: float = 0.01                  # final lr = lr0 × lrf (cosine floor)
    momentum: float = 0.937            # SGD momentum / Adam β₁
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0         # can be fractional
    warmup_bias_lr: float = 0.1        # bias LR at start of warmup

    # ── Training loop ────────────────────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 16
    mixed_precision: bool = True       # AMP fp16
    gradient_clip_val: float = 10.0
    seed: int = 0
    close_mosaic: int = 10             # disable mosaic in last N epochs

    # ── Augmentation (geometric augmentations handled per-model dataset) ──────
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # ── Loss weights ─────────────────────────────────────────────────────────
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5

    # ── Output & logging ─────────────────────────────────────────────────────
    output_dir: Path = dataclasses.field(default_factory=lambda: Path("artifacts/runs"))
    project: str = "vision-to-edge"
    run_name: str | None = None
    save_period: int = -1              # -1 = save last + best only
    resume: Path | None = None

    # Fields that hold filesystem paths (for YAML → dataclass conversion)
    _PATH_FIELDS: dataclasses.ClassVar[frozenset[str]] = frozenset(
        {"data_path", "pretrained", "output_dir", "resume"}
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingConfig:
        """Load config from a YAML file.  Unknown keys are silently ignored."""
        import yaml
        with open(path) as f:
            data: dict = yaml.safe_load(f) or {}
        for key in cls._PATH_FIELDS:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def override(self, **kwargs: Any) -> TrainingConfig:
        """Return a copy with non-None kwargs applied.

        Designed for typer CLI usage where unset Optional args are None:
            cfg = cfg.override(epochs=epochs, lr0=lr0)
        """
        changes = {k: v for k, v in kwargs.items() if v is not None}
        return dataclasses.replace(self, **changes)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base trainer
# ─────────────────────────────────────────────────────────────────────────────

class BaseTrainer(abc.ABC):
    """Reusable training loop shared by all model families.

    Subclasses implement:
        build_model()        → nn.Module
        build_dataloaders()  → (DataLoader, DataLoader)
        compute_loss()       → (Tensor, dict[str, float])
        compute_metrics()    → dict[str, float]

    The metric named by PRIMARY_METRIC (default "val/mAP50-95") drives both
    best-checkpoint selection and early stopping.
    """

    PRIMARY_METRIC: str = "val/mAP50-95"

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = self._make_run_dir()
        # GradScaler manages loss scaling for AMP; no-op when cuda unavailable
        self.scaler = GradScaler("cuda", enabled=cfg.mixed_precision and torch.cuda.is_available())
        self.wandb_run = None
        self._best_metric: float = 0.0
        self._global_step: int = 0

        # Reproducibility
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        log.info("Device      : %s", self.device)
        log.info("Run dir     : %s", self.run_dir)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate and return the model (not yet moved to device)."""

    @abc.abstractmethod
    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader)."""

    @abc.abstractmethod
    def compute_loss(
        self,
        predictions: Any,
        targets: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss.

        Returns:
            loss:       Differentiable scalar tensor (for .backward()).
            components: Named sub-losses as plain floats for logging,
                        e.g. {"box": 2.1, "cls": 0.8, "dfl": 0.5}.
        """

    @abc.abstractmethod
    def compute_metrics(
        self,
        predictions: list[Any],
        targets: list[Any],
    ) -> dict[str, float]:
        """Evaluate a full validation epoch.

        Returns a dict that must include self.PRIMARY_METRIC.
        """

    # ── Main training entry point ─────────────────────────────────────────────

    def train(self) -> None:
        import warnings
        from training.common.callbacks import EarlyStopping
        # SequentialLR calls step() on sub-schedulers during __init__, which
        # triggers a spurious "step before optimizer" warning.  It is harmless.
        warnings.filterwarnings("ignore", message="Detected call of", category=UserWarning)

        self._init_wandb()

        model = self.build_model().to(self.device)
        train_loader, val_loader = self.build_dataloaders()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, steps_per_epoch=len(train_loader))
        early_stop = EarlyStopping(patience=50)

        # Optionally resume from a previous checkpoint
        start_epoch = 0
        if self.cfg.resume:
            start_epoch = self._load_checkpoint(self.cfg.resume, model, optimizer, scheduler)

        # Store as instance attributes so subclass hooks can access them
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        log.info("Training for %d epochs (start=%d)", self.cfg.epochs, start_epoch)

        for epoch in range(start_epoch, self.cfg.epochs):
            # Tell YOLO dataset to disable mosaic near the end of training
            if hasattr(train_loader.dataset, "set_mosaic"):
                enable_mosaic = epoch < (self.cfg.epochs - self.cfg.close_mosaic)
                train_loader.dataset.set_mosaic(enable_mosaic)

            train_metrics = self._train_epoch(epoch, train_loader)
            val_metrics = self._validate(epoch, val_loader)

            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            primary = val_metrics.get(self.PRIMARY_METRIC, 0.0)
            is_best = primary > self._best_metric
            if is_best:
                self._best_metric = primary

            self._save_checkpoint(epoch, model, optimizer, scheduler, all_metrics, is_best)
            self._log(all_metrics, step=self._global_step)

            log.info(
                "[%d/%d]  %s",
                epoch + 1, self.cfg.epochs,
                "  ".join(f"{k}={v:.4f}" for k, v in all_metrics.items() if k != "epoch"),
            )

            if early_stop(primary):
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

        if self.wandb_run:
            self.wandb_run.finish()

    # ── Per-epoch methods ─────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        component_sums: dict[str, float] = {}

        for imgs, targets in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass under AMP
            with autocast("cuda", enabled=self.cfg.mixed_precision and torch.cuda.is_available()):
                preds = self.model(imgs)
                loss, components = self.compute_loss(preds, targets)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()  # step-based (one step per batch)

            total_loss += loss.item()
            for k, v in components.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
            self._global_step += 1

        n = len(loader)
        metrics: dict[str, float] = {"train/loss": total_loss / n}
        metrics.update({f"train/{k}": v / n for k, v in component_sums.items()})
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    @torch.inference_mode()  # stricter than no_grad: skips version counters, ~10% faster
    def _validate(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds: list[Any] = []
        all_targets: list[Any] = []

        for imgs, targets in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.cfg.mixed_precision and torch.cuda.is_available()):
                preds = self.model(imgs)
            all_preds.append(preds)
            all_targets.append(targets)

        return self.compute_metrics(all_preds, all_targets)

    # ── Checkpoint management ─────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: dict[str, float],
        is_best: bool,
    ) -> None:
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "cfg": dataclasses.asdict(self.cfg),
        }
        last_path = self.run_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            shutil.copy(last_path, self.run_dir / "best.pt")
            log.info("New best (%.4f) → best.pt", self._best_metric)

        if self.cfg.save_period > 0 and (epoch + 1) % self.cfg.save_period == 0:
            torch.save(state, self.run_dir / f"epoch{epoch + 1:04d}.pt")

        # Upload ONLY the best checkpoint as a W&B artifact.
        # Uploading last.pt every epoch wastes bandwidth (300 × model_size MB).
        if self.wandb_run and is_best:
            best_path = self.run_dir / "best.pt"
            artifact = _wandb.Artifact(
                name=f"model-{self.wandb_run.id}",
                type="model",
                metadata=metrics,
            )
            artifact.add_file(str(best_path))
            self.wandb_run.log_artifact(artifact)

    def _load_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> int:
        """Load checkpoint; returns the epoch to resume from."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self._best_metric = state["metrics"].get(self.PRIMARY_METRIC, 0.0)
        start = state["epoch"] + 1
        log.info("Resumed from %s (epoch %d → %d)", path, state["epoch"], start)
        return start

    # ── Optimizer & scheduler ─────────────────────────────────────────────────

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build optimizer with YOLOv11-style parameter groups.

        Why three groups?
            BN weights and biases should NOT have weight decay applied
            (weight decay on them harms normalization and can destabilize training).
            Only the main weight matrices get L2 regularization.
        """
        g_bn:     list[nn.Parameter] = []  # BatchNorm/GroupNorm weights — no decay
        g_bias:   list[nn.Parameter] = []  # All biases — no decay
        g_weight: list[nn.Parameter] = []  # Everything else — decay applied

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                g_bn.append(module.weight)
            elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                g_weight.append(module.weight)
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                g_bias.append(module.bias)

        if self.cfg.optimizer == "SGD":
            opt: torch.optim.Optimizer = SGD(
                g_bn, lr=self.cfg.lr0, momentum=self.cfg.momentum, nesterov=True
            )
        else:  # AdamW
            opt = AdamW(g_bn, lr=self.cfg.lr0, betas=(self.cfg.momentum, 0.999), weight_decay=0.0)

        opt.add_param_group({"params": g_bias,   "weight_decay": 0.0})
        opt.add_param_group({"params": g_weight, "weight_decay": self.cfg.weight_decay})

        log.info(
            "Optimizer %s | g_bn=%d g_bias=%d g_weight=%d",
            self.cfg.optimizer, len(g_bn), len(g_bias), len(g_weight),
        )
        return opt

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, steps_per_epoch: int
    ) -> SequentialLR:
        """Step-based LR schedule: linear warmup → cosine annealing.

        Why step-based (not epoch-based)?
            Finer-grained LR updates produce smoother loss curves,
            especially during the warmup phase.
        """
        import warnings
        warmup_steps = max(1, round(self.cfg.warmup_epochs * steps_per_epoch))
        total_steps  = self.cfg.epochs * steps_per_epoch
        cosine_steps = max(1, total_steps - warmup_steps)

        # All three scheduler constructors call step() during __init__, which
        # triggers a harmless PyTorch warning about step order. Wrap all three.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=self.cfg.lr0 * self.cfg.lrf)
            sched  = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return sched

    # ── W&B helpers ───────────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        if not _WANDB:
            log.warning("wandb not installed; run `uv add wandb` or set WANDB_MODE=offline.")
            return
        self.wandb_run = _wandb.init(
            project=self.cfg.project,
            name=self.cfg.run_name,
            config=dataclasses.asdict(self.cfg),
            dir=str(self.run_dir),
            resume="allow",
        )
        log.info("W&B run: %s", self.wandb_run.url)

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_run_dir(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.cfg.run_name or ts
        run_dir = Path(self.cfg.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
