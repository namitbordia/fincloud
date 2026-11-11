"""Volatility forecast evaluation metrics and coverage summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def forecast_metrics(
    actual_volatility: np.ndarray | pd.Series,
    forecast_volatility: np.ndarray | pd.Series,
    *,
    variance_floor: float = 1e-12,
) -> dict[str, Any]:
    """Calculate MAE, RMSE, and variance-scale QLIKE for paired forecasts."""
    actual = np.asarray(actual_volatility, dtype=np.float64)
    forecast = np.asarray(forecast_volatility, dtype=np.float64)
    if actual.ndim != 1 or forecast.ndim != 1 or actual.shape != forecast.shape:
        raise ValueError("actual and forecast must be equal-length one-dimensional arrays.")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive.")
    paired = np.isfinite(actual) & np.isfinite(forecast)
    if (actual[paired] < 0.0).any() or (forecast[paired] < 0.0).any():
        raise ValueError("Volatility values must be non-negative.")
    if not paired.any():
        return {"mae": None, "rmse": None, "qlike": None, "valid_forecasts": 0}
    actual_paired = actual[paired]
    forecast_paired = forecast[paired]
    errors = forecast_paired - actual_paired
    actual_variance = np.maximum(actual_paired**2, variance_floor)
    forecast_variance = np.maximum(forecast_paired**2, variance_floor)
    variance_ratio = actual_variance / forecast_variance
    qlike = variance_ratio - np.log(variance_ratio) - 1.0
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qlike": float(np.mean(qlike)),
        "valid_forecasts": int(paired.sum()),
    }


def evaluate_forecasts(forecasts: pd.DataFrame, *, variance_floor: float) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Return per-model/split aggregate and per-stock evaluation tables."""
    metric_columns = [
        "model",
        "split",
        "eligible_origins",
        "valid_forecasts",
        "forecast_coverage",
        "failed_fit_count",
        "unavailable_forecast_count",
        "mae",
        "rmse",
        "qlike",
    ]
    stock_columns = [*metric_columns, "symbol"]
    model_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    if forecasts.empty:
        return pd.DataFrame(columns=metric_columns), pd.DataFrame(
            columns=stock_columns
        )
    for (model, split), group in forecasts.groupby(
        ["model", "split"], sort=True, dropna=False
    ):
        model_rows.append(
            _metric_row(group, str(model), str(split), variance_floor)
        )
    for model, group in forecasts.groupby("model", sort=True, dropna=False):
        model_rows.append(_metric_row(group, str(model), "all", variance_floor))
    for (symbol, model, split), group in forecasts.groupby(
        ["symbol", "model", "split"], sort=True, dropna=False
    ):
        stock_rows.append(
            _metric_row(group, str(model), str(split), variance_floor, str(symbol))
        )
    for (symbol, model), group in forecasts.groupby(
        ["symbol", "model"], sort=True, dropna=False
    ):
        stock_rows.append(
            _metric_row(group, str(model), "all", variance_floor, str(symbol))
        )
    return pd.DataFrame(model_rows, columns=metric_columns), pd.DataFrame(
        stock_rows, columns=stock_columns
    )


def _metric_row(
    group: pd.DataFrame,
    model: str,
    split: str,
    variance_floor: float,
    symbol: str | None = None,
) -> dict[str, Any]:
    eligible = group["target_status"].eq("available")
    valid = (
        eligible
        & group["forecast_status"].eq("ok")
        & group["actual_realized_volatility"].notna()
        & group["forecast_volatility"].notna()
    )
    metrics = forecast_metrics(
        group.loc[valid, "actual_realized_volatility"],
        group.loc[valid, "forecast_volatility"],
        variance_floor=variance_floor,
    )
    row: dict[str, Any] = {
        "model": model,
        "split": split,
        "eligible_origins": int(eligible.sum()),
        "valid_forecasts": metrics["valid_forecasts"],
        "forecast_coverage": (
            float(metrics["valid_forecasts"] / eligible.sum())
            if eligible.sum()
            else None
        ),
        "failed_fit_count": int(group["forecast_status"].eq("fit_failed").sum()),
        "unavailable_forecast_count": int(
            (~group["forecast_status"].eq("ok")).sum()
        ),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "qlike": metrics["qlike"],
    }
    if symbol is not None:
        row["symbol"] = symbol
    return row
