"""Inference with verified S3 model artifacts and statistical baselines."""

from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np

from backend.app.errors import ForecastUnavailableError, RequestValidationError
from backend.app.schemas.requests import parse_forecast_request
from backend.app.schemas.responses import json_safe
from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository
from src.forecasting.ml.features import FEATURE_COLUMNS, build_features
from src.forecasting.models import ewma_forecast, garch_forecast, historical_rolling_forecast

FORECAST_HORIZON = 20
TRADING_DAYS_PER_YEAR = 252
TARGET_DEFINITION = (
    "sqrt((trading_days_per_year / horizon) * "
    "sum(next horizon daily_return squared))"
)


def run_forecast(
    body: dict[str, Any],
    repository: S3Repository,
) -> dict[str, Any]:
    symbol, model_name = parse_forecast_request(body)
    frame, source = repository.load_market_data()
    selected = frame.loc[frame["symbol"] == symbol].sort_values(
        "date", kind="stable"
    )
    if selected.empty:
        raise RequestValidationError(f"Unsupported symbol: {symbol}.")
    feature_frame = build_features(selected)
    latest = feature_frame.iloc[-1]
    values = latest.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ForecastUnavailableError(
            "The latest observation does not have a complete finite feature vector."
        )
    period = {
        "start_date": selected["date"].min().date().isoformat(),
        "end_date": selected["date"].max().date().isoformat(),
        "forecast_origin": selected["date"].max().date().isoformat(),
    }
    history = _contiguous_finite_history(selected["daily_return"].to_numpy(dtype=np.float64))
    if model_name in {"ridge", "random_forest"}:
        prediction, artifact = _trained_prediction(repository, symbol, values, model_name)
        model_version = artifact["sha256"]
        model_metadata = {
            "artifact_key": artifact["key"],
            "sha256": artifact["sha256"],
            "fit_date": artifact["fit_date"],
            "training_start_date": artifact["training_start_date"],
            "training_end_date": artifact["training_end_date"],
            "training_observations": artifact["training_observations"],
            "feature_count": len(FEATURE_COLUMNS),
            "features_verified": True,
            "target_definition": TARGET_DEFINITION,
        }
        model_verified = True
    else:
        forecast_fn = {
            "historical_rolling": historical_rolling_forecast,
            "ewma": ewma_forecast,
            "garch_1_1": garch_forecast,
        }[model_name]
        forecast = forecast_fn(history)
        if forecast.status != "ok" or forecast.volatility is None:
            raise ForecastUnavailableError(
            f"{model_name} forecast is unavailable: {forecast.status}."
            )
        prediction = float(forecast.volatility)
        model_version = f"src.forecasting.models.{model_name}"
        model_metadata = {
            "artifact_key": None,
            "sha256": None,
            "fit_date": None,
            "training_start_date": None,
            "training_end_date": None,
            "training_observations": forecast.history_observations,
            "feature_count": None,
            "features_verified": None,
            "target_definition": TARGET_DEFINITION,
        }
        model_verified = False
    result = {
        "symbol": symbol,
        "model": model_name,
        "forecasted_annualized_volatility": prediction,
        "volatility_unit": "annualized decimal",
        "forecast_horizon_sessions": FORECAST_HORIZON,
        "input_data_period": period,
        "model_version": model_version,
        "model_metadata": model_metadata,
        "new_forecast_actuals_available": False,
        "evaluation_metrics": None,
        "data_source": source,
    }
    result["cloud_execution"] = cloud_execution(
        repository,
        repository.settings,
        data_verified=True,
        model_verified=model_verified,
    )
    result["result_key"] = repository.save_result("forecasts", result)
    result["result_persisted"] = True
    return json_safe(result)


def _trained_prediction(
    repository: S3Repository,
    symbol: str,
    values: np.ndarray,
    model_name: str,
) -> tuple[float, dict[str, Any]]:
    model_bytes, registered, digest = repository.load_model(symbol, model_name)
    if registered.get("filename") != f"{symbol}_{model_name}.joblib":
        raise ForecastUnavailableError(
            f"S3 model metadata does not reference the expected {model_name} artifact."
        )
    if registered.get("feature_names") != list(FEATURE_COLUMNS):
        raise ForecastUnavailableError(
            f"S3 {model_name} feature names do not match the runtime feature order."
        )
    try:
        model = joblib.load(io.BytesIO(model_bytes))
        predicted = float(model.predict(values.reshape(1, -1))[0])
    except (ValueError, TypeError, AttributeError, ImportError) as error:
        raise ForecastUnavailableError(
            f"{model_name} artifact could not run inference ({type(error).__name__})."
        ) from error
    if not np.isfinite(predicted) or predicted < 0:
        raise ForecastUnavailableError(
            f"{model_name} model produced a non-finite or negative volatility."
        )
    metadata = {
        **registered,
        "sha256": digest,
        "key": repository.settings.model_prefix + registered["filename"],
    }
    return predicted, metadata


def _contiguous_finite_history(values: np.ndarray) -> np.ndarray:
    invalid = np.flatnonzero(~np.isfinite(values) | (values < -1.0))
    start = int(invalid[-1] + 1) if len(invalid) else 0
    return values[start:]
