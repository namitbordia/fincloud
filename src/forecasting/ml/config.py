"""Configuration for machine-learning volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


DEFAULT_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": 0.8,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    "ridge": {"alpha": 1.0},
}


@dataclass(frozen=True)
class MLForecastConfig:
    horizon: int = 20
    trading_days_per_year: int = 252
    train_end: str = "2022-12-31"
    validation_end: str = "2023-12-31"
    random_seed: int = 42
    refit_frequency: str = "monthly"
    variance_floor: float = 1e-12
    model_parameters: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            name: dict(parameters)
            for name, parameters in DEFAULT_MODEL_PARAMETERS.items()
        }
    )

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2.")
        if self.trading_days_per_year < 1:
            raise ValueError("trading_days_per_year must be positive.")
        if date.fromisoformat(self.train_end) >= date.fromisoformat(
            self.validation_end
        ):
            raise ValueError("train_end must be earlier than validation_end.")
        if self.refit_frequency not in {"monthly", "quarterly"}:
            raise ValueError("refit_frequency must be monthly or quarterly.")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive.")
        unknown = set(self.model_parameters) - set(DEFAULT_MODEL_PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown ML model parameters: {sorted(unknown)}")
