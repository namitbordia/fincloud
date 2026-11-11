"""Stock and portfolio performance metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.returns import risk_free_daily_rate
from src.analytics.risk_metrics import drawdown_analysis, historical_var_cvar


def calculate_performance_metrics(
    returns: pd.Series | np.ndarray,
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    initial_value: float = 1.0,
) -> dict[str, Any]:
    """Calculate common performance metrics from simple daily returns.

    CAGR uses actual elapsed calendar days divided by 365.25. Sharpe and
    Sortino use the effective daily rate derived from the effective annual
    risk-free rate and the configured trading-day count.
    """
    values = np.asarray(returns, dtype=float)
    parsed_dates = pd.to_datetime(dates, errors="coerce", utc=True)
    if len(values) != len(parsed_dates):
        raise ValueError("returns and dates must have equal lengths.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be finite and greater than zero.")
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    valid = np.isfinite(values) & (values >= -1.0) & parsed_dates.notna()
    valid_returns = values[valid]
    valid_dates = pd.DatetimeIndex(parsed_dates[valid]).tz_convert(None)
    count = int(valid_returns.size)
    daily_rf = risk_free_daily_rate(annual_risk_free_rate, trading_days)
    metrics: dict[str, Any] = {
        "observation_count": count,
        "annual_risk_free_rate": float(annual_risk_free_rate),
        "daily_risk_free_rate": daily_rf,
        "total_return": float("nan"),
        "cagr": float("nan"),
        "annualized_volatility": float("nan"),
        "sharpe_ratio": float("nan"),
        "sortino_ratio": float("nan"),
        "maximum_drawdown": float("nan"),
        "cagr_day_count": (
            int((valid_dates.max() - valid_dates.min()).days)
            if count >= 2
            else 0
        ),
        "cagr_day_count_convention": "actual elapsed calendar days / 365.25",
    }
    if count == 0:
        metrics["metric_availability"] = {
            "total_return": False,
            "cagr": False,
            "annualized_volatility": False,
            "sharpe_ratio": False,
            "sortino_ratio": False,
            "maximum_drawdown": False,
        }
        return metrics

    wealth = initial_value * np.prod(1.0 + valid_returns)
    metrics["total_return"] = float(wealth / initial_value - 1.0)
    elapsed_days = metrics["cagr_day_count"]
    if count >= 2 and elapsed_days > 0 and wealth > 0:
        metrics["cagr"] = float(
            (wealth / initial_value) ** (365.25 / elapsed_days) - 1.0
        )

    if count >= 2:
        daily_volatility = float(np.std(valid_returns, ddof=1))
        metrics["annualized_volatility"] = float(
            daily_volatility * np.sqrt(trading_days)
        )
        excess_returns = valid_returns - daily_rf
        if daily_volatility > 0:
            metrics["sharpe_ratio"] = float(
                np.mean(excess_returns)
                * np.sqrt(trading_days)
                / daily_volatility
            )
        downside_deviation = float(
            np.sqrt(np.mean(np.minimum(excess_returns, 0.0) ** 2))
        )
        if downside_deviation > 0:
            metrics["sortino_ratio"] = float(
                np.mean(excess_returns)
                * np.sqrt(trading_days)
                / downside_deviation
            )
        _, maximum_drawdown = drawdown_analysis(
            valid_returns, initial_value=initial_value
        )
        metrics["maximum_drawdown"] = maximum_drawdown

    metrics["metric_availability"] = {
        "total_return": True,
        "cagr": bool(
            count >= 2 and elapsed_days > 0 and wealth > 0
        ),
        "annualized_volatility": count >= 2,
        "sharpe_ratio": bool(
            count >= 2 and np.std(valid_returns, ddof=1) > 0
        ),
        "sortino_ratio": bool(
            count >= 2
            and np.sqrt(
                np.mean(
                    np.minimum(valid_returns - daily_rf, 0.0) ** 2
                )
            )
            > 0
        ),
        "maximum_drawdown": count >= 2,
    }

    for confidence in (0.95, 0.99):
        risk = historical_var_cvar(valid_returns, confidence)
        label = int(confidence * 100)
        metrics[f"var_{label}"] = risk["var"]
        metrics[f"cvar_{label}"] = risk["cvar"]
        metrics[f"var_cvar_{label}_available"] = risk["available"]
    return metrics
