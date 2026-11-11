"""Return-series calculation and input validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnValidation:
    """Validated return observations and explicit exclusion records."""

    returns: pd.Series
    exclusions: tuple[dict[str, Any], ...]
    duplicate_count: int


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from a positive price series without filling gaps."""
    numeric = pd.to_numeric(prices, errors="coerce").astype("float64")
    valid = numeric.notna() & np.isfinite(numeric.to_numpy()) & (numeric > 0)
    safe_prices = numeric.where(valid)
    result = safe_prices.pct_change(fill_method=None)
    return result.where(valid & valid.shift(1, fill_value=False))


def validate_return_frame(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
    return_column: str = "daily_return",
) -> ReturnValidation:
    """Validate daily returns; invalid, missing, and duplicate rows are reported."""
    required = {date_column, symbol_column, return_column}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Return data is missing required columns: {missing_columns}")

    data = frame[[date_column, symbol_column, return_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    data[date_column] = data[date_column].dt.tz_convert(None).dt.normalize()
    data[symbol_column] = data[symbol_column].astype("string").str.strip().str.upper()
    data[return_column] = pd.to_numeric(data[return_column], errors="coerce")

    exclusions: list[dict[str, Any]] = []

    def exclude(position: int, reason: str) -> None:
        row = data.iloc[position]
        exclusions.append(
            {
                "row_number": int(position + 2),
                "date": (
                    row[date_column].strftime("%Y-%m-%d")
                    if pd.notna(row[date_column])
                    else None
                ),
                "symbol": str(row[symbol_column])
                if pd.notna(row[symbol_column])
                else None,
                "reason": reason,
            }
        )

    invalid_date = data[date_column].isna()
    invalid_symbol = data[symbol_column].isna() | data[symbol_column].eq("")
    for position in np.flatnonzero(invalid_date.to_numpy()):
        exclude(int(position), "invalid_date")
    for position in np.flatnonzero(invalid_symbol.to_numpy()):
        exclude(int(position), "invalid_symbol")

    keyed = data.loc[~invalid_date & ~invalid_symbol]
    duplicate_mask = keyed.duplicated([symbol_column, date_column], keep=False)
    duplicate_positions = keyed.index[duplicate_mask]
    for position in duplicate_positions:
        exclude(int(position), "duplicate_symbol_date")
    duplicate_count = int(duplicate_mask.sum())

    numeric = data[return_column].to_numpy(dtype=float, na_value=np.nan)
    for position, value in enumerate(numeric):
        if pd.isna(value):
            exclude(position, "missing_return")
        elif not np.isfinite(value):
            exclude(position, "non_finite_return")
        elif value < -1.0:
            exclude(position, "return_below_minus_one")

    excluded_positions = {entry["row_number"] - 2 for entry in exclusions}
    valid_positions = [
        position
        for position in range(len(data))
        if position not in excluded_positions
    ]
    valid = data.iloc[valid_positions].sort_values(
        [symbol_column, date_column], kind="stable"
    )
    series = pd.Series(
        valid[return_column].to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays(
            [valid[symbol_column].astype(str), valid[date_column]],
            names=[symbol_column, date_column],
        ),
        name=return_column,
        dtype="float64",
    )
    return ReturnValidation(series, tuple(exclusions), duplicate_count)


def risk_free_daily_rate(annual_rate: float, trading_days: int = 252) -> float:
    """Convert an effective annual risk-free rate to an effective daily rate."""
    if not np.isfinite(annual_rate) or annual_rate <= -1.0:
        raise ValueError("The annual risk-free rate must be finite and greater than -1.")
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")
    return float((1.0 + annual_rate) ** (1.0 / trading_days) - 1.0)
