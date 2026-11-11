"""Portfolio optimization and backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    estimation_windows: tuple[int, ...] = (60, 120, 252)
    rebalance_frequencies: tuple[str, ...] = ("monthly", "quarterly")
    maximum_asset_weights: tuple[float, ...] = (0.6,)
    transaction_cost_bps: tuple[float, ...] = (0.0, 10.0)
    cvar_confidences: tuple[float, ...] = (0.95,)
    minimum_asset_weight: float = 0.0
    maximum_holdings: int | None = None
    annual_risk_free_rate: float = 0.0
    rebalance_start: str = "2023-01-01"
    trading_days_per_year: int = 252
    solver_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not self.estimation_windows or any(window < 2 for window in self.estimation_windows):
            raise ValueError("estimation_windows must contain values of at least two.")
        if len(set(self.estimation_windows)) != len(self.estimation_windows):
            raise ValueError("estimation_windows must be unique.")
        if not self.rebalance_frequencies or any(
            frequency not in {"monthly", "quarterly"}
            for frequency in self.rebalance_frequencies
        ):
            raise ValueError("rebalance_frequencies must use monthly or quarterly.")
        if any(not 0.0 < value <= 1.0 for value in self.maximum_asset_weights):
            raise ValueError("maximum asset weights must be in (0, 1].")
        if any(value < 0.0 for value in self.transaction_cost_bps):
            raise ValueError("transaction costs cannot be negative.")
        if any(not 0.0 < value < 1.0 for value in self.cvar_confidences):
            raise ValueError("CVaR confidences must be between zero and one.")
        if not 0.0 <= self.minimum_asset_weight <= 1.0:
            raise ValueError("minimum_asset_weight must be in [0, 1].")
        if self.maximum_holdings is not None and self.maximum_holdings < 1:
            raise ValueError("maximum_holdings must be positive when supplied.")
        if self.annual_risk_free_rate <= -1.0:
            raise ValueError("annual_risk_free_rate must be greater than -1.")
        if self.trading_days_per_year < 1 or self.solver_tolerance <= 0.0:
            raise ValueError("Invalid trading-day count or solver tolerance.")

