"""Validation helpers for JSON API request bodies."""

from __future__ import annotations

from datetime import date
from typing import Any

from backend.app.errors import RequestValidationError

SUPPORTED_OPTIMIZERS = {"minimum_variance", "maximum_sharpe", "cvar_minimizing"}
SUPPORTED_FORECASTS = {"historical_rolling", "ewma", "garch_1_1", "ridge", "random_forest"}


def parse_object(value: Any, *, name: str = "request") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestValidationError(f"{name} must be a JSON object.")
    return value


def parse_symbols(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RequestValidationError("symbols must be a non-empty array.")
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in value):
        raise RequestValidationError("Each symbol must be a non-empty string.")
    symbols = [symbol.strip().upper() for symbol in value]
    if len(set(symbols)) != len(symbols):
        raise RequestValidationError("symbols must be unique.")
    return symbols


def parse_dates(body: dict[str, Any]) -> tuple[date | None, date | None]:
    start_value = body.get("start_date")
    end_value = body.get("end_date")
    if start_value is None and end_value is None:
        return None, None
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        raise RequestValidationError(
            "start_date and end_date must both be ISO dates (YYYY-MM-DD)."
        )
    try:
        start = date.fromisoformat(start_value)
        end = date.fromisoformat(end_value)
    except ValueError as error:
        raise RequestValidationError(
            "start_date and end_date must be valid ISO dates (YYYY-MM-DD)."
        ) from error
    if start >= end:
        raise RequestValidationError("start_date must be earlier than end_date.")
    return start, end


def parse_weights(body: dict[str, Any], symbols: list[str]) -> dict[str, float]:
    raw = body.get("weights")
    if not isinstance(raw, dict):
        raise RequestValidationError("weights must be an object keyed by symbol.")
    if set(raw) != set(symbols):
        raise RequestValidationError(
            "weights must contain exactly the selected symbols."
        )
    try:
        weights = {symbol: float(raw[symbol]) for symbol in symbols}
    except (TypeError, ValueError) as error:
        raise RequestValidationError("Every weight must be numeric.") from error
    if any(not _finite(weight) for weight in weights.values()):
        raise RequestValidationError("Every weight must be finite.")
    if any(weight < 0 for weight in weights.values()):
        raise RequestValidationError("Negative portfolio weights are not supported.")
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise RequestValidationError("Portfolio weights must sum to 1 (100%).")
    return weights


def parse_forecast_request(body: dict[str, Any]) -> tuple[str, str]:
    symbol = body.get("symbol")
    model = body.get("model", "ridge")
    if not isinstance(symbol, str) or not symbol.strip():
        raise RequestValidationError("symbol must be a non-empty string.")
    if not isinstance(model, str) or model.lower() not in SUPPORTED_FORECASTS:
        raise RequestValidationError(
            "model must be one of: " + ", ".join(sorted(SUPPORTED_FORECASTS)) + "."
        )
    return symbol.strip().upper(), model.lower()


def parse_optimization_request(
    body: dict[str, Any],
) -> tuple[list[str], list[str], float, float, float]:
    symbols = parse_symbols(body.get("symbols"))
    strategy = body.get("strategy", "minimum_variance")
    if strategy == "compare":
        strategies = ["minimum_variance", "maximum_sharpe", "cvar_minimizing"]
    elif isinstance(strategy, str) and strategy in SUPPORTED_OPTIMIZERS:
        strategies = [strategy]
    else:
        raise RequestValidationError(
            "strategy must be minimum_variance, maximum_sharpe, cvar_minimizing, or compare."
        )
    try:
        minimum_weight = float(body.get("minimum_weight", 0.0))
        maximum_weight = float(body.get("maximum_weight", 1.0))
        risk_free_rate = float(body.get("annual_risk_free_rate", 0.0))
    except (TypeError, ValueError) as error:
        raise RequestValidationError(
            "Weight bounds and annual_risk_free_rate must be numeric."
        ) from error
    if not all(_finite(value) for value in (minimum_weight, maximum_weight, risk_free_rate)):
        raise RequestValidationError("Weight bounds and risk-free rate must be finite.")
    if minimum_weight < 0 or maximum_weight <= 0 or minimum_weight > maximum_weight:
        raise RequestValidationError("Invalid asset weight bounds.")
    if risk_free_rate <= -1:
        raise RequestValidationError("annual_risk_free_rate must be greater than -1.")
    if len(symbols) * minimum_weight > 1 + 1e-8:
        raise RequestValidationError("Minimum weight bounds cannot sum to 100%.")
    if len(symbols) * maximum_weight < 1 - 1e-8:
        raise RequestValidationError("Maximum weight bounds cannot sum to 100%.")
    return symbols, strategies, minimum_weight, maximum_weight, risk_free_rate


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
