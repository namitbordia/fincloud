"""Configuration for statistical volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ForecastConfig:
    """Reproducible model, target, and chronological split settings."""

    horizon: int = 20
    trading_days_per_year: int = 252
    ewma_lambda: float = 0.94
    train_end: str = "2022-12-31"
    validation_end: str = "2023-12-31"
    min_garch_observations: int = 252
    qlike_variance_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if self.trading_days_per_year < 1:
            raise ValueError("trading_days_per_year must be positive.")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be strictly between zero and one.")
        if self.min_garch_observations < 20:
            raise ValueError("min_garch_observations must be at least 20.")
        if self.qlike_variance_floor <= 0.0:
            raise ValueError("qlike_variance_floor must be positive.")
        train_end = date.fromisoformat(self.train_end)
        validation_end = date.fromisoformat(self.validation_end)
        if train_end >= validation_end:
            raise ValueError("train_end must be earlier than validation_end.")

