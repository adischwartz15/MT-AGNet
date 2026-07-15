"""Lightweight training callbacks: early stopping."""

from __future__ import annotations


class EarlyStopping:
    """Stops training when a monitored metric hasn't improved for ``patience`` epochs."""

    def __init__(self, patience: int = 8, mode: str = "min", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with the latest metric value; returns True if training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop
