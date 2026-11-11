"""Portfolio weight validation and constant-weight rebalanced returns."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def equal_weights(symbols: Sequence[str]) -> dict[str, float]:
    """Return constant equal weights for a non-empty, unique symbol list."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be a non-empty list of unique names.")
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def validate_weights(
    weights: Mapping[str, float],
    symbols: Sequence[str],
    *,
    long_only: bool = True,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Validate exact asset coverage and a unit-sum portfolio weight vector."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be a non-empty list of unique names.")
    if set(weights) != set(symbols):
        missing = sorted(set(symbols) - set(weights))
        unknown = sorted(set(weights) - set(symbols))
        raise ValueError(
            f"Weights must cover exactly the selected symbols; missing={missing}, "
            f"unknown={unknown}."
        )
    normalized = {symbol: float(weights[symbol]) for symbol in symbols}
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError("All portfolio weights must be finite.")
    if long_only and any(value < 0 for value in normalized.values()):
        raise ValueError("Negative weights are not permitted in long-only mode.")
    if not np.isclose(sum(normalized.values()), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Portfolio weights must sum to 1 within tolerance {tolerance:g}."
        )
    return normalized


def calculate_portfolio_returns(
    aligned_returns: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Calculate daily returns for a constant-weight, daily rebalanced portfolio.

    A portfolio return is missing if any weighted asset return is missing for
    that date. Missing returns are never filled with zero.
    """
    missing_columns = sorted(set(weights) - set(aligned_returns.columns))
    if missing_columns:
        raise ValueError(f"Return matrix is missing weighted symbols: {missing_columns}")
    selected = aligned_returns.loc[:, list(weights)]
    complete = selected.notna().all(axis=1)
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    valid_values = np.isfinite(numeric.to_numpy(dtype=float)) & (
        numeric.to_numpy(dtype=float) >= -1.0
    )
    complete &= pd.Series(valid_values.all(axis=1), index=selected.index)
    result = pd.Series(np.nan, index=selected.index, name="portfolio_return")
    if complete.any():
        weight_vector = np.array([weights[symbol] for symbol in selected.columns])
        result.loc[complete] = numeric.loc[complete].to_numpy() @ weight_vector
    return result
