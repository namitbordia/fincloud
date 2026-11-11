"""Historical tail risk and drawdown calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def historical_var_cvar(
    returns: pd.Series | np.ndarray,
    confidence: float,
    *,
    minimum_observations: int = 2,
) -> dict[str, Any]:
    """Calculate historical VaR/CVaR as positive losses (negative tail returns)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    values = np.asarray(returns, dtype=float)
    finite = values[np.isfinite(values)]
    invalid = values.size - finite.size
    if finite.size < minimum_observations:
        return {
            "var": float("nan"),
            "cvar": float("nan"),
            "observation_count": int(finite.size),
            "excluded_non_finite_count": int(invalid),
            "available": False,
            "unavailable_reason": (
                f"At least {minimum_observations} finite observations are required."
            ),
        }
    losses = -finite
    value_at_risk = float(np.quantile(losses, confidence, method="linear"))
    tail_losses = losses[losses >= value_at_risk]
    expected_shortfall = float(tail_losses.mean())
    return {
        "var": value_at_risk,
        "cvar": expected_shortfall,
        "observation_count": int(finite.size),
        "excluded_non_finite_count": int(invalid),
        "available": True,
        "unavailable_reason": None,
    }


def drawdown_analysis(
    returns: pd.Series | np.ndarray,
    *,
    initial_value: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """Return wealth, running peak, and percentage drawdown from daily returns."""
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be finite and greater than zero.")
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        raise ValueError("At least one return observation is required.")
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise ValueError("Returns must be finite and no lower than -1.")

    wealth = initial_value * np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(np.concatenate(([initial_value], wealth)))[1:]
    drawdown = wealth / running_peak - 1.0
    result = pd.DataFrame(
        {
            "wealth_index": wealth,
            "running_peak": running_peak,
            "drawdown": drawdown,
        }
    )
    return result, float(drawdown.min())
