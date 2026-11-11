"""Aligned daily-return correlation analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def correlation_analysis(
    aligned_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute pairwise-complete Pearson correlations without return filling."""
    numeric = aligned_returns.apply(pd.to_numeric, errors="coerce")
    finite = numeric.replace([float("inf"), float("-inf")], float("nan"))
    matrix = finite.corr(method="pearson", min_periods=2)
    counts = {
        left: {
            right: int((finite[left].notna() & finite[right].notna()).sum())
            for right in finite.columns
        }
        for left in finite.columns
    }
    return matrix, {
        "method": "Pearson correlation, pairwise complete observations",
        "missing_values_filled": False,
        "observation_counts": counts,
    }
