"""Forward realized-volatility targets and chronological split assignment."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def assign_split(
    origin: pd.Timestamp,
    *,
    train_end: str,
    validation_end: str,
) -> str:
    """Assign a forecast origin to a fixed chronological date split."""
    train_cutoff = pd.Timestamp(date.fromisoformat(train_end))
    validation_cutoff = pd.Timestamp(date.fromisoformat(validation_end))
    if origin <= train_cutoff:
        return "train"
    if origin <= validation_cutoff:
        return "validation"
    return "test"


def forward_realized_volatility(
    future_returns: np.ndarray | pd.Series,
    *,
    horizon: int = 20,
    trading_days_per_year: int = 252,
) -> float:
    """Annualized zero-mean realized volatility over exactly the next horizon."""
    values = np.asarray(future_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("future_returns must be one-dimensional.")
    if horizon < 1 or trading_days_per_year < 1:
        raise ValueError("horizon and trading_days_per_year must be positive.")
    if len(values) != horizon:
        raise ValueError(f"Exactly {horizon} future returns are required.")
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise ValueError("Future returns must be finite and at least -1.")
    realized_variance = (
        trading_days_per_year / horizon * float(np.sum(values**2))
    )
    return float(np.sqrt(realized_variance))


def build_forward_targets(
    frame: pd.DataFrame,
    *,
    horizon: int = 20,
    trading_days_per_year: int = 252,
    train_end: str = "2022-12-31",
    validation_end: str = "2023-12-31",
) -> pd.DataFrame:
    """Build targets from the next observed ticker rows, never current/future features."""
    required = {"date", "symbol", "daily_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Target input is missing required columns: {missing}")
    if horizon < 1 or trading_days_per_year < 1:
        raise ValueError("horizon and trading_days_per_year must be positive.")
    train_cutoff = pd.Timestamp(date.fromisoformat(train_end))
    validation_cutoff = pd.Timestamp(date.fromisoformat(validation_end))
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end.")

    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    for symbol, group in ordered.groupby("symbol", sort=True):
        dates = pd.DatetimeIndex(group["date"])
        returns = pd.to_numeric(group["daily_return"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for position, origin in enumerate(dates):
            split = assign_split(
                origin, train_end=train_end, validation_end=validation_end
            )
            if split == "train":
                continue
            future_start = position + 1
            future_stop = min(future_start + horizon, len(group))
            future_dates = dates[future_start:future_stop]
            future = returns[future_start:future_stop]
            target_end = (
                future_dates[-1].strftime("%Y-%m-%d") if len(future_dates) else None
            )
            if len(future) < horizon:
                status = "insufficient_future_returns"
                actual: float | None = None
            elif split == "validation" and future_dates[-1] > validation_cutoff:
                status = "crosses_split_boundary"
                actual = None
            elif not np.isfinite(future).all() or (future < -1.0).any():
                status = "missing_or_invalid_future_return"
                actual = None
            else:
                status = "available"
                actual = forward_realized_volatility(
                    future,
                    horizon=horizon,
                    trading_days_per_year=trading_days_per_year,
                )
            rows.append(
                {
                    "date": origin.strftime("%Y-%m-%d"),
                    "symbol": str(symbol),
                    "split": split,
                    "target_end_date": target_end,
                    "actual_realized_volatility": actual,
                    "target_status": status,
                    "target_observations": int(len(future)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "split",
            "target_end_date",
            "actual_realized_volatility",
            "target_status",
            "target_observations",
        ],
    )

