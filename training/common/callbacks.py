"""Training callbacks."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when the primary validation metric stops improving.

    Args:
        patience:   Epochs to wait for improvement before stopping.
        min_delta:  Minimum improvement to count as progress.

    Usage:
        early_stop = EarlyStopping(patience=50)
        if early_stop(val_map):   # returns True when triggered
            break
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = 0.0
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Update state; return True if training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            log.debug(
                "EarlyStopping: no improvement for %d/%d epochs (best=%.4f).",
                self.counter, self.patience, self.best,
            )
            if self.counter >= self.patience:
                log.info(
                    "EarlyStopping triggered after %d epochs without improvement.",
                    self.patience,
                )
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        self.best = 0.0
        self.counter = 0
        self.should_stop = False
