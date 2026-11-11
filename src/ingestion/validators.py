"""Validation checks for downloaded daily stock data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_COLUMNS = ("adj_close",)
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


def validate_stock_data(
    data: pd.DataFrame, symbol: str | None = None
) -> dict[str, Any]:
    """Return validation results without modifying or dropping any records."""
    issues: list[dict[str, Any]] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    checks: dict[str, Any] = {
        "required_columns": {
            "passed": not missing_columns,
            "missing": missing_columns,
        }
    }

    if missing_columns:
        issues.append(
            {
                "code": "missing_required_columns",
                "columns": missing_columns,
            }
        )

    if data.empty:
        issues.append({"code": "no_data", "count": 0})

    available_columns = [
        column
        for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
        if column in data
    ]
    missing_values = {
        column: int(data[column].isna().sum()) for column in available_columns
    }
    missing_value_count = sum(missing_values.values())
    checks["missing_values"] = {
        "passed": missing_value_count == 0,
        "counts_by_column": missing_values,
        "count": missing_value_count,
    }
    if missing_value_count:
        issues.append(
            {
                "code": "missing_values",
                "counts_by_column": {
                    column: count
                    for column, count in missing_values.items()
                    if count
                },
            }
        )

    numeric_columns = [
        column
        for column in (*PRICE_COLUMNS, "volume")
        if column in data
    ]
    non_numeric_values: dict[str, int] = {}
    numeric_data: dict[str, pd.Series] = {}
    for column in numeric_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        numeric_data[column] = numeric
        invalid = data[column].notna() & numeric.isna()
        if invalid.any():
            non_numeric_values[column] = int(invalid.sum())
    checks["numeric_values"] = {
        "passed": not non_numeric_values,
        "invalid_counts_by_column": non_numeric_values,
    }
    if non_numeric_values:
        issues.append(
            {
                "code": "non_numeric_values",
                "counts_by_column": non_numeric_values,
            }
        )

    price_columns = [column for column in PRICE_COLUMNS if column in numeric_data]
    non_positive_prices: dict[str, int] = {}
    for column in price_columns:
        numeric = numeric_data[column]
        invalid = numeric.notna() & (
            (numeric <= 0) | ~np.isfinite(numeric.to_numpy(dtype=float))
        )
        if invalid.any():
            non_positive_prices[column] = int(invalid.sum())
    checks["positive_prices"] = {
        "passed": not non_positive_prices,
        "invalid_counts_by_column": non_positive_prices,
    }
    if non_positive_prices:
        issues.append(
            {
                "code": "non_positive_prices",
                "counts_by_column": non_positive_prices,
            }
        )

    ohlc_columns = {"open", "high", "low", "close"}
    if ohlc_columns.issubset(numeric_data):
        open_prices = numeric_data["open"]
        high_prices = numeric_data["high"]
        low_prices = numeric_data["low"]
        close_prices = numeric_data["close"]
        complete = pd.concat(
            [open_prices, high_prices, low_prices, close_prices], axis=1
        ).notna().all(axis=1)
        invalid_ohlc = complete & (
            (high_prices < np.maximum(open_prices, np.maximum(low_prices, close_prices)))
            | (low_prices > np.minimum(open_prices, np.minimum(high_prices, close_prices)))
        )
        invalid_ohlc_count = int(invalid_ohlc.sum())
    else:
        invalid_ohlc_count = 0
    checks["ohlc_relationships"] = {
        "passed": invalid_ohlc_count == 0,
        "invalid_rows": invalid_ohlc_count,
    }
    if invalid_ohlc_count:
        issues.append(
            {"code": "invalid_ohlc_relationships", "count": invalid_ohlc_count}
        )

    if "volume" in numeric_data:
        volume = numeric_data["volume"]
        negative_volume = volume.notna() & (volume < 0)
        negative_volume_count = int(negative_volume.sum())
    else:
        negative_volume_count = 0
    checks["non_negative_volume"] = {
        "passed": negative_volume_count == 0,
        "invalid_rows": negative_volume_count,
    }
    if negative_volume_count:
        issues.append(
            {"code": "negative_volume", "count": negative_volume_count}
        )

    if "date" in data:
        dates = pd.to_datetime(data["date"], errors="coerce")
        duplicate_rows = data.assign(_parsed_date=dates).duplicated(
            subset=["symbol", "_parsed_date"]
            if "symbol" in data
            else ["_parsed_date"],
            keep=False,
        )
        duplicate_count = int(duplicate_rows.sum())
        sort_groups = (
            data.assign(_parsed_date=dates).groupby("symbol", sort=False)
            if "symbol" in data
            else [(symbol or "", data.assign(_parsed_date=dates))]
        )
        unsorted_groups = [
            str(group_symbol)
            for group_symbol, group in sort_groups
            if not group["_parsed_date"].dropna().is_monotonic_increasing
        ]
    else:
        duplicate_count = 0
        unsorted_groups = []
    checks["duplicate_dates"] = {
        "passed": duplicate_count == 0,
        "duplicate_rows": duplicate_count,
    }
    if duplicate_count:
        issues.append({"code": "duplicate_dates", "duplicate_rows": duplicate_count})
    checks["date_sorting"] = {
        "passed": not unsorted_groups,
        "unsorted_symbols": unsorted_groups,
    }
    if unsorted_groups:
        issues.append({"code": "dates_not_sorted", "symbols": unsorted_groups})

    return {
        "symbol": symbol,
        "row_count": int(len(data)),
        "valid": not issues,
        "checks": checks,
        "issues": issues,
    }
