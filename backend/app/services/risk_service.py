"""S3-backed portfolio risk analysis using existing FinCloud calculations."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from backend.app.errors import RequestValidationError
from backend.app.schemas.requests import parse_dates, parse_symbols, parse_weights
from backend.app.schemas.responses import json_safe
from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository
from src.analytics.correlation import correlation_analysis
from src.analytics.performance import calculate_performance_metrics
from src.analytics.portfolio import calculate_portfolio_returns


def run_risk(
    body: dict[str, Any],
    repository: S3Repository,
    *,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    symbols = parse_symbols(body.get("symbols"))
    weights = parse_weights(body, symbols)
    start, end = parse_dates(body)
    frame, source = repository.load_market_data()
    _require_symbols(frame, symbols)
    selected = _period(frame, symbols, start, end)
    returns = selected.pivot(index="date", columns="symbol", values="daily_return")
    returns = returns.reindex(columns=symbols)
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    valid = portfolio_returns.notna()
    if int(valid.sum()) < 2:
        raise RequestValidationError(
            "The selected period has fewer than two complete portfolio return observations."
        )
    metrics = calculate_performance_metrics(
        portfolio_returns[valid].to_numpy(),
        portfolio_returns.index[valid],
        annual_risk_free_rate=annual_risk_free_rate,
    )
    matrix, correlation_metadata = correlation_analysis(returns)
    result: dict[str, Any] = {
        "symbols": symbols,
        "weights": weights,
        "period": {
            "start_date": selected["date"].min().date().isoformat(),
            "end_date": selected["date"].max().date().isoformat(),
            "requested_start_date": start.isoformat() if start else None,
            "requested_end_date": end.isoformat() if end else None,
        },
        "metrics": metrics,
        "correlation": {
            "symbols": list(matrix.columns),
            "values": matrix.to_numpy().tolist(),
            "observation_counts": correlation_metadata["observation_counts"],
            "method": correlation_metadata["method"],
            "missing_values_filled": False,
        },
        "daily_returns": [
            {"date": dt.date().isoformat(), "return": float(value)}
            for dt, value in portfolio_returns[valid].items()
        ],
        "data_source": source,
    }
    result["cloud_execution"] = cloud_execution(
        repository, repository.settings, data_verified=True
    )
    result["result_key"] = repository.save_result("risk", result)
    result["result_persisted"] = True
    return json_safe(result)


def _period(
    frame: pd.DataFrame,
    symbols: list[str],
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    selected = frame.loc[frame["symbol"].isin(symbols)].copy()
    available = set(str(symbol) for symbol in selected["symbol"].unique())
    missing = sorted(set(symbols) - available)
    if missing:
        raise RequestValidationError(
            f"Unsupported symbols: {missing}. Check GET /stocks for available symbols."
        )
    if start is not None:
        selected = selected.loc[selected["date"].dt.date >= start]
    if end is not None:
        selected = selected.loc[selected["date"].dt.date <= end]
    if selected.empty:
        raise RequestValidationError("No market data exists for the selected period.")
    return selected


def _require_symbols(frame: pd.DataFrame, symbols: list[str]) -> None:
    available = set(str(symbol) for symbol in frame["symbol"].unique())
    missing = sorted(set(symbols) - available)
    if missing:
        raise RequestValidationError(
            f"Unsupported symbols: {missing}. Check GET /stocks for available symbols."
        )
