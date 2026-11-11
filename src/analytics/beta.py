"""Date-aligned asset beta against an explicitly supplied benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_benchmark_returns(path: str) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Load a benchmark CSV containing date and return/daily_return columns."""
    frame = pd.read_csv(path)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    date_column = columns.get("date")
    return_column = columns.get("daily_return")
    if return_column is None:
        return_column = columns.get("return")
    if date_column is None or return_column is None:
        raise ValueError(
            "Benchmark CSV must include 'date' and either 'daily_return' or 'return'."
        )
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce", utc=True),
            "return": pd.to_numeric(frame[return_column], errors="coerce"),
        }
    )
    data["date"] = data["date"].dt.tz_convert(None).dt.normalize()
    exclusions: list[dict[str, Any]] = []
    duplicate = data["date"].notna() & data.duplicated("date", keep=False)
    for position, row in data.iterrows():
        reason = None
        if pd.isna(row["date"]):
            reason = "invalid_date"
        elif duplicate.loc[position]:
            reason = "duplicate_benchmark_date"
        elif pd.isna(row["return"]):
            reason = "missing_benchmark_return"
        elif not np.isfinite(row["return"]):
            reason = "non_finite_benchmark_return"
        elif row["return"] < -1.0:
            reason = "benchmark_return_below_minus_one"
        if reason:
            exclusions.append(
                {
                    "row_number": int(position + 2),
                    "date": row["date"].strftime("%Y-%m-%d")
                    if pd.notna(row["date"])
                    else None,
                    "reason": reason,
                }
            )
    valid = data.loc[
        data["date"].notna()
        & ~duplicate
        & data["return"].notna()
        & np.isfinite(data["return"])
        & (data["return"] >= -1.0)
    ].sort_values("date", kind="stable")
    return valid.set_index("date")["return"].rename("benchmark_return"), exclusions


def calculate_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    minimum_observations: int = 2,
) -> dict[str, Any]:
    """Calculate sample covariance(asset, benchmark) / sample variance(benchmark)."""
    if benchmark_returns is None:
        return {
            "beta": float("nan"),
            "observation_count": 0,
            "available": False,
            "unavailable_reason": "No benchmark was supplied.",
        }
    aligned = pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    )
    numeric = aligned.apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    values = numeric.loc[complete].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < minimum_observations:
        return {
            "beta": float("nan"),
            "observation_count": int(len(values)),
            "available": False,
            "unavailable_reason": (
                f"At least {minimum_observations} aligned observations are required."
            ),
        }
    benchmark_variance = float(np.var(values[:, 1], ddof=1))
    if benchmark_variance == 0.0:
        return {
            "beta": float("nan"),
            "observation_count": int(len(values)),
            "available": False,
            "unavailable_reason": "Benchmark variance is zero.",
        }
    covariance = float(np.cov(values[:, 0], values[:, 1], ddof=1)[0, 1])
    return {
        "beta": covariance / benchmark_variance,
        "observation_count": int(len(values)),
        "available": True,
        "unavailable_reason": None,
    }
