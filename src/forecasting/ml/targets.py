"""Supervised training targets matching Phase 5 realized-volatility semantics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.forecasting.targets import forward_realized_volatility


def build_ml_targets(
    frame: pd.DataFrame,
    *,
    horizon: int = 20,
    trading_days_per_year: int = 252,
    train_end: str = "2022-12-31",
    validation_end: str = "2023-12-31",
) -> pd.DataFrame:
    """Label each origin using the next returns, bounded within its split."""
    required = {"date", "symbol", "daily_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Target input is missing columns: {missing}")
    train_cutoff = pd.Timestamp(date.fromisoformat(train_end))
    validation_cutoff = pd.Timestamp(date.fromisoformat(validation_end))
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end.")
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    for symbol, group in ordered.groupby("symbol", sort=True):
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"], errors="coerce"))
        returns = pd.to_numeric(group["daily_return"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for index, origin in enumerate(dates):
            future_dates = dates[index + 1 : index + 1 + horizon]
            future_returns = returns[index + 1 : index + 1 + horizon]
            split = (
                "train"
                if origin <= train_cutoff
                else "validation"
                if origin <= validation_cutoff
                else "test"
            )
            target_end = (
                future_dates[-1].strftime("%Y-%m-%d")
                if len(future_dates)
                else None
            )
            actual: float | None = None
            status = "available"
            if len(future_returns) != horizon:
                status = "insufficient_future_returns"
            elif split == "train" and future_dates[-1] > train_cutoff:
                status = "crosses_split_boundary"
            elif split == "validation" and future_dates[-1] > validation_cutoff:
                status = "crosses_split_boundary"
            elif not np.isfinite(future_returns).all() or (
                future_returns < -1.0
            ).any():
                status = "missing_or_invalid_future_return"
            else:
                actual = forward_realized_volatility(
                    future_returns,
                    horizon=horizon,
                    trading_days_per_year=trading_days_per_year,
                )
            rows.append(
                {
                    "date": origin,
                    "symbol": str(symbol),
                    "split": split,
                    "target_end_date": target_end,
                    "actual_realized_volatility": actual,
                    "target_status": status,
                    "target_observations": int(len(future_returns)),
                }
            )
    return pd.DataFrame(rows)

