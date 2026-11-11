"""Interactive optimization using FinCloud's existing constrained solvers."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.errors import (
    OptimizationUnavailableError,
    RequestValidationError,
)
from backend.app.schemas.requests import parse_dates, parse_optimization_request
from backend.app.schemas.responses import json_safe
from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository
from src.analytics.performance import calculate_performance_metrics
from src.optimization.solvers import OptimizationError, optimize_weights

TRADING_DAYS_PER_YEAR = 252
MINIMUM_HISTORY = 60


def run_optimization(
    body: dict[str, Any],
    repository: S3Repository,
) -> dict[str, Any]:
    symbols, strategies, minimum_weight, maximum_weight, risk_free_rate = (
        parse_optimization_request(body)
    )
    start, end = parse_dates(body)
    frame, source = repository.load_market_data()
    available = set(str(symbol) for symbol in frame["symbol"].unique())
    missing = sorted(set(symbols) - available)
    if missing:
        raise RequestValidationError(f"Unsupported symbols: {missing}.")
    selected = frame.loc[frame["symbol"].isin(symbols)].copy()
    if start is not None:
        selected = selected.loc[selected["date"].dt.date >= start]
    if end is not None:
        selected = selected.loc[selected["date"].dt.date <= end]
    matrix = selected.pivot(
        index="date", columns="symbol", values="daily_return"
    ).reindex(columns=symbols)
    complete = matrix.dropna()
    values = complete.to_numpy(dtype=np.float64)
    if len(values) < MINIMUM_HISTORY:
        raise RequestValidationError(
            f"At least {MINIMUM_HISTORY} complete daily observations are required "
            "for optimization."
        )
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise RequestValidationError(
            "Optimization history contains invalid daily returns."
        )
    expected = values.mean(axis=0) * TRADING_DAYS_PER_YEAR
    covariance = np.cov(values, rowvar=False, ddof=1) * TRADING_DAYS_PER_YEAR
    if len(symbols) == 1:
        covariance = np.asarray([[float(covariance)]])
    results = []
    for strategy in strategies:
        try:
            solved = optimize_weights(
                strategy,
                expected,
                covariance,
                historical_returns=values,
                annual_risk_free_rate=risk_free_rate,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
            )
        except (OptimizationError, ValueError, np.linalg.LinAlgError) as error:
            raise OptimizationUnavailableError(
                f"{strategy} optimization failed: {error}"
            ) from error
        weights = {
            symbol: float(solved.weights[index])
            for index, symbol in enumerate(symbols)
        }
        daily = values @ solved.weights
        volatility = float(np.sqrt(max(float(solved.weights @ covariance @ solved.weights), 0.0)))
        expected_return = float(solved.weights @ expected)
        sharpe = (
            (expected_return - risk_free_rate) / volatility
            if volatility > 0
            else None
        )
        performance = calculate_performance_metrics(
            daily,
            complete.index,
            annual_risk_free_rate=risk_free_rate,
        )
        results.append(
            {
                "strategy": strategy,
                "weights": weights,
                "expected_annual_return": expected_return,
                "annualized_volatility": volatility,
                "sharpe_ratio": sharpe,
                "maximum_drawdown": performance["maximum_drawdown"],
                "observations": len(values),
                "solver": solved.solver,
                "objective_value": solved.objective_value,
                "diagnostics": solved.diagnostics,
            }
        )
    result = {
        "symbols": symbols,
        "period": {
            "start_date": complete.index.min().date().isoformat(),
            "end_date": complete.index.max().date().isoformat(),
            "requested_start_date": start.isoformat() if start else None,
            "requested_end_date": end.isoformat() if end else None,
        },
        "annual_risk_free_rate": risk_free_rate,
        "strategies": results,
        "comparison": len(results) > 1,
        "data_source": source,
        "estimation_method": (
            "Annualized arithmetic mean and sample covariance of complete "
            "daily Silver returns; no missing observations were filled."
        ),
    }
    result["cloud_execution"] = cloud_execution(
        repository, repository.settings, data_verified=True
    )
    result["result_key"] = repository.save_result("optimization", result)
    result["result_persisted"] = True
    return json_safe(result)
