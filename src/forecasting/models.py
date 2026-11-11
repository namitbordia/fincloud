"""Historical, EWMA, and GARCH volatility forecast implementations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from arch import arch_model

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ModelForecast:
    """One model's forecast and explicit fit/forecast status."""

    model: str
    volatility: float | None
    status: str
    history_observations: int
    convergence_flag: int | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None


def _valid_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional sequence.")
    return values


def historical_rolling_forecast(
    returns: Sequence[float] | np.ndarray,
    *,
    window: int = 20,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ModelForecast:
    """Use trailing sample standard deviation as a persistence forecast."""
    values = _valid_returns(returns)
    if window < 2:
        raise ValueError("window must be at least 2.")
    if trading_days_per_year < 1:
        raise ValueError("trading_days_per_year must be positive.")
    if len(values) < window:
        return ModelForecast(
            "historical_rolling",
            None,
            "insufficient_history",
            len(values),
        )
    trailing = values[-window:]
    if not np.isfinite(trailing).all() or (trailing < -1.0).any():
        return ModelForecast(
            "historical_rolling", None, "invalid_history", len(values)
        )
    volatility = float(np.std(trailing, ddof=1) * np.sqrt(trading_days_per_year))
    return ModelForecast("historical_rolling", volatility, "ok", len(values))


def ewma_forecast(
    returns: Sequence[float] | np.ndarray,
    *,
    decay: float = 0.94,
    initialization_window: int = 20,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ModelForecast:
    """Filter variance with EWMA and hold the current variance over the horizon."""
    values = _valid_returns(returns)
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between zero and one.")
    if initialization_window < 2:
        raise ValueError("initialization_window must be at least 2.")
    if trading_days_per_year < 1:
        raise ValueError("trading_days_per_year must be positive.")
    if len(values) < initialization_window:
        return ModelForecast("ewma", None, "insufficient_history", len(values))
    if not np.isfinite(values).all() or (values < -1.0).any():
        return ModelForecast("ewma", None, "invalid_history", len(values))

    variance = float(np.var(values[:initialization_window], ddof=1))
    for observation in values[initialization_window:]:
        variance = decay * variance + (1.0 - decay) * float(observation**2)
    volatility = float(np.sqrt(max(variance, 0.0) * trading_days_per_year))
    return ModelForecast("ewma", volatility, "ok", len(values))


def garch_forecast(
    returns: Sequence[float] | np.ndarray,
    *,
    horizon: int = 20,
    min_observations: int = 252,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ModelForecast:
    """Fit normal-innovation GARCH(1,1) to history and forecast its variance."""
    values = _valid_returns(returns)
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if min_observations < 1:
        raise ValueError("min_observations must be positive.")
    if trading_days_per_year < 1:
        raise ValueError("trading_days_per_year must be positive.")
    if len(values) < min_observations:
        return ModelForecast("garch_1_1", None, "insufficient_history", len(values))
    if not np.isfinite(values).all() or (values < -1.0).any():
        return ModelForecast("garch_1_1", None, "invalid_history", len(values))

    warning_records: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            model = arch_model(
                values * 100.0,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            result = model.fit(disp="off")
            captured = tuple(str(item.message) for item in warning_records)
            convergence_flag = int(result.convergence_flag)
            if convergence_flag != 0:
                message = (
                    f"GARCH optimizer did not converge "
                    f"(convergence_flag={convergence_flag})."
                )
                return ModelForecast(
                    "garch_1_1",
                    None,
                    "fit_failed",
                    len(values),
                    convergence_flag,
                    captured,
                    message,
                )
            forecast = result.forecast(horizon=horizon, reindex=False)
            variances_percent_squared = forecast.variance.iloc[-1].to_numpy(
                dtype=np.float64
            )
        if (
            len(variances_percent_squared) != horizon
            or not np.isfinite(variances_percent_squared).all()
            or (variances_percent_squared < 0.0).any()
        ):
            raise ValueError("GARCH returned invalid conditional variance forecasts.")
        mean_daily_variance = float(np.mean(variances_percent_squared)) / 10000.0
        volatility = float(
            np.sqrt(mean_daily_variance * trading_days_per_year)
        )
        return ModelForecast(
            "garch_1_1",
            volatility,
            "ok",
            len(values),
            convergence_flag,
            tuple(str(item.message) for item in warning_records),
        )
    except Exception as error:
        return ModelForecast(
            "garch_1_1",
            None,
            "fit_failed",
            len(values),
            None,
            tuple(str(item.message) for item in warning_records),
            f"{type(error).__name__}: {error}",
        )
