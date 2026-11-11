"""Leakage-safe historical feature generation for volatility forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

FEATURE_WINDOWS = (5, 10, 20, 60)
RETURN_LAGS = (1, 2, 5, 10, 20, 60)
FEATURE_COLUMNS = (
    "daily_return",
    *(f"return_lag_{lag}" for lag in RETURN_LAGS),
    *(f"volatility_{window}d" for window in FEATURE_WINDOWS),
    *(f"mean_return_{window}d" for window in FEATURE_WINDOWS),
    *(f"adjusted_close_sma_distance_{window}d" for window in (5, 20, 60)),
    "rsi_14",
    "volume_change",
    "log_dollar_turnover",
    "mean_log_dollar_turnover_20d",
)


def build_features(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    date_column: str = "date",
    return_column: str = "daily_return",
) -> pd.DataFrame:
    """Create past-only rolling/lag features, retaining missing warm-up values."""
    required = {symbol_column, date_column, return_column, "adj_close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature input is missing required columns: {missing}")
    data = frame.copy(deep=True)
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[symbol_column] = data[symbol_column].astype("string").str.strip().str.upper()
    data[return_column] = pd.to_numeric(data[return_column], errors="coerce")
    data["adj_close"] = pd.to_numeric(data["adj_close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    if data[date_column].isna().any() or data[symbol_column].isna().any():
        raise ValueError("Feature input contains invalid dates or symbols.")
    if data.duplicated([symbol_column, date_column]).any():
        raise ValueError("Feature input contains duplicate symbol/date rows.")
    data = data.sort_values([symbol_column, date_column], kind="stable")
    feature_frames: list[pd.DataFrame] = []
    for symbol, group in data.groupby(symbol_column, sort=True):
        group = group.copy()
        returns = group[return_column].astype("float64")
        features = pd.DataFrame(index=group.index)
        features["daily_return"] = returns
        for lag in RETURN_LAGS:
            features[f"return_lag_{lag}"] = returns.shift(lag)
        for window in FEATURE_WINDOWS:
            features[f"volatility_{window}d"] = returns.rolling(
                window, min_periods=window
            ).std(ddof=1)
            features[f"mean_return_{window}d"] = returns.rolling(
                window, min_periods=window
            ).mean()
        adjusted_close = group["adj_close"].where(
            np.isfinite(group["adj_close"]) & (group["adj_close"] > 0)
        )
        for window in (5, 20, 60):
            moving_average = adjusted_close.rolling(
                window, min_periods=window
            ).mean()
            features[f"adjusted_close_sma_distance_{window}d"] = (
                adjusted_close / moving_average - 1.0
            )

        changes = returns
        gains = changes.clip(lower=0.0)
        losses = -changes.clip(upper=0.0)
        average_gain = gains.rolling(14, min_periods=14).mean()
        average_loss = losses.rolling(14, min_periods=14).mean()
        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        only_gains = (average_loss == 0.0) & (average_gain > 0.0)
        no_movement = (average_loss == 0.0) & (average_gain == 0.0)
        rsi.loc[only_gains] = 100.0
        rsi.loc[no_movement] = 50.0
        features["rsi_14"] = rsi

        volume = group["volume"].where(
            np.isfinite(group["volume"]) & (group["volume"] >= 0)
        )
        features["volume_change"] = volume.pct_change(fill_method=None)
        turnover = group["adj_close"] * volume
        valid_turnover = np.isfinite(turnover) & (turnover > 0)
        log_turnover = pd.Series(np.nan, index=group.index, dtype="float64")
        log_turnover.loc[valid_turnover] = np.log(turnover.loc[valid_turnover])
        features["log_dollar_turnover"] = log_turnover
        features["mean_log_dollar_turnover_20d"] = log_turnover.rolling(
            20, min_periods=20
        ).mean()
        features[symbol_column] = str(symbol)
        features[date_column] = group[date_column]
        feature_frames.append(features)

    result = pd.concat(feature_frames).sort_values(
        [symbol_column, date_column], kind="stable"
    )
    result.index = data.index
    result.attrs["feature_columns"] = list(FEATURE_COLUMNS)
    result.attrs["feature_information_cutoff"] = "At or before each row's date."
    return result.loc[
        :, [symbol_column, date_column, *FEATURE_COLUMNS]
    ].reset_index(drop=True)


def validate_feature_order(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    """Require the canonical feature names and order for saved model use."""
    names = tuple(feature_names)
    if names != tuple(FEATURE_COLUMNS):
        raise ValueError("Feature columns do not match the canonical feature order.")
    return names

