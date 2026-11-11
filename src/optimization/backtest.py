"""Walk-forward portfolio optimization and out-of-sample backtest engine."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.config import OptimizationConfig
from src.optimization.solvers import OptimizationError, optimize_weights

STRATEGIES = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "cvar_minimizing",
    "minimum_variance_forecast_aware",
)


def scenario_grid(config: OptimizationConfig) -> list[dict[str, Any]]:
    """Expand explicit sensitivity settings into reproducible scenario configs."""
    scenarios: list[dict[str, Any]] = []
    for window, frequency, cap, cost, confidence in itertools.product(
        config.estimation_windows,
        config.rebalance_frequencies,
        config.maximum_asset_weights,
        config.transaction_cost_bps,
        config.cvar_confidences,
    ):
        scenario_id = (
            f"window{window}_{frequency}_cap{cap:g}_cost{cost:g}_"
            f"cvar{confidence:g}"
        )
        scenarios.append(
            {
                "configuration_id": scenario_id,
                "estimation_window": int(window),
                "rebalance_frequency": frequency,
                "maximum_asset_weight": float(cap),
                "transaction_cost_bps": float(cost),
                "cvar_confidence": float(confidence),
            }
        )
    return scenarios


def _period_key(date: pd.Timestamp, frequency: str) -> str:
    if frequency == "monthly":
        return date.strftime("%Y-%m")
    return f"{date.year}-Q{date.quarter}"


def _rebalance_dates(
    dates: pd.DatetimeIndex,
    *,
    frequency: str,
    start_date: str,
) -> list[pd.Timestamp]:
    eligible = dates[dates >= pd.Timestamp(start_date)]
    if len(eligible) == 0:
        return []
    series = pd.Series(eligible, index=eligible)
    first_per_period = series.groupby(
        [_period_key(date, frequency) for date in eligible], sort=True
    ).min()
    return list(pd.DatetimeIndex(first_per_period.to_numpy()).sort_values())


def _forecast_covariance(
    forecast_frame: pd.DataFrame,
    *,
    model: str,
    rebalance_date: pd.Timestamp,
    symbols: tuple[str, ...],
    estimation_returns: pd.DataFrame,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Combine valid forecast volatilities with past-only sample correlations."""
    before = forecast_frame.loc[
        (forecast_frame["model"] == model)
        & (forecast_frame["date"] < rebalance_date)
    ]
    if before.empty:
        return None, None, "No prior-origin volatility forecasts are available."
    origin = before["date"].max()
    selected = before.loc[before["date"] == origin]
    if selected["symbol"].duplicated().any():
        return None, origin.strftime("%Y-%m-%d"), "Duplicate forecasts at selected origin."
    selected = selected.set_index("symbol").reindex(symbols)
    if selected["forecast_status"].isna().any() or not selected[
        "forecast_status"
    ].eq("ok").all():
        return (
            None,
            origin.strftime("%Y-%m-%d"),
            "One or more assets have no successful prior-origin volatility forecast.",
        )
    volatility = pd.to_numeric(
        selected["forecast_volatility"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(volatility).all() or (volatility <= 0.0).any():
        return (
            None,
            origin.strftime("%Y-%m-%d"),
            "Forecast volatilities must be finite and strictly positive.",
        )
    correlation = estimation_returns.corr().to_numpy(dtype=np.float64)
    if not np.isfinite(correlation).all():
        return (
            None,
            origin.strftime("%Y-%m-%d"),
            "Historical correlation matrix is not fully defined.",
        )
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues.min() < -1e-8:
        return (
            None,
            origin.strftime("%Y-%m-%d"),
            "Historical correlation matrix is not positive semidefinite.",
        )
    covariance = correlation * np.outer(volatility, volatility)
    return covariance, origin.strftime("%Y-%m-%d"), None


def _split(date: pd.Timestamp) -> str:
    if date.year == 2023:
        return "validation"
    if date.year >= 2024:
        return "test"
    return "training"


def walk_forward_backtest(
    returns_frame: pd.DataFrame,
    *,
    config: OptimizationConfig = OptimizationConfig(),
    forecast_frame: pd.DataFrame | None = None,
    forecast_model: str = "random_forest",
) -> dict[str, Any]:
    """Optimize at period starts using prior data and apply to later returns."""
    required = {"date", "symbol", "daily_return"}
    missing = sorted(required - set(returns_frame.columns))
    if missing:
        raise ValueError(f"Backtest input is missing columns: {missing}")
    data = returns_frame.loc[:, ["date", "symbol", "daily_return"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["symbol"] = data["symbol"].astype("string").str.strip().str.upper()
    data["daily_return"] = pd.to_numeric(data["daily_return"], errors="coerce")
    if data["date"].isna().any() or data["symbol"].isna().any():
        raise ValueError("Backtest input has invalid date or symbol keys.")
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("Backtest input has duplicate date/symbol records.")
    symbols = tuple(sorted(str(item) for item in data["symbol"].unique()))
    matrix = data.pivot(index="date", columns="symbol", values="daily_return").reindex(
        columns=symbols
    )
    matrix = matrix.sort_index()
    if (matrix.notna() & (matrix < -1.0)).any().any():
        raise ValueError("Backtest returns cannot be less than -100%.")
    complete_matrix = matrix.dropna(how="any")
    if complete_matrix.empty:
        raise ValueError("No complete asset-return rows are available.")
    if forecast_frame is not None:
        forecast_frame = forecast_frame.copy()
        forecast_frame["date"] = pd.to_datetime(
            forecast_frame["date"], errors="coerce"
        )
        forecast_frame["symbol"] = (
            forecast_frame["symbol"].astype("string").str.strip().str.upper()
        )
        if forecast_frame["date"].isna().any():
            raise ValueError("Forecast input contains invalid dates.")

    scenarios = scenario_grid(config)
    daily_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rf_annual = config.annual_risk_free_rate

    for scenario in scenarios:
        frequency = str(scenario["rebalance_frequency"])
        window = int(scenario["estimation_window"])
        dates = complete_matrix.index
        decisions = _rebalance_dates(
            dates,
            frequency=frequency,
            start_date=config.rebalance_start,
        )
        previous_weights: dict[str, np.ndarray | None] = {
            strategy: None for strategy in STRATEGIES
        }
        for decision_index, decision in enumerate(decisions):
            next_decision = (
                decisions[decision_index + 1]
                if decision_index + 1 < len(decisions)
                else pd.Timestamp.max
            )
            estimation = complete_matrix.loc[
                complete_matrix.index < decision
            ].tail(window)
            if len(estimation) < window:
                diagnostics.append(
                    {
                        "configuration_id": scenario["configuration_id"],
                        "rebalance_date": decision.strftime("%Y-%m-%d"),
                        "strategy": "all",
                        "status": "insufficient_estimation_history",
                        "available_observations": int(len(estimation)),
                        "required_observations": window,
                    }
                )
                continue
            estimation_array = estimation.to_numpy(dtype=np.float64)
            expected = estimation.mean(axis=0).to_numpy(dtype=np.float64) * config.trading_days_per_year
            covariance = estimation.cov().to_numpy(dtype=np.float64) * config.trading_days_per_year
            base_results: dict[str, tuple[np.ndarray | None, dict[str, Any]]] = {}
            equal = np.full(len(symbols), 1.0 / len(symbols), dtype=np.float64)
            equal_feasible = (
                config.minimum_asset_weight - config.solver_tolerance
                <= float(equal[0])
                <= float(scenario["maximum_asset_weight"])
                + config.solver_tolerance
                and (
                    config.maximum_holdings is None
                    or config.maximum_holdings >= len(symbols)
                )
            )
            base_results["equal_weight"] = (
                equal if equal_feasible else None,
                {
                    "status": "ok" if equal_feasible else "constraints_infeasible",
                    "solver": "deterministic equal weight",
                    **(
                        {}
                        if equal_feasible
                        else {
                            "error": (
                                "Equal-weight baseline violates the configured "
                                "weight bounds or maximum-holdings constraint."
                            )
                        }
                    ),
                },
            )
            if not equal_feasible:
                diagnostics.append(
                    {
                        "configuration_id": scenario["configuration_id"],
                        "rebalance_date": decision.strftime("%Y-%m-%d"),
                        "strategy": "equal_weight",
                        "status": "constraints_infeasible",
                        "error": base_results["equal_weight"][1]["error"],
                    }
                )
            for strategy, method in (
                ("minimum_variance", "minimum_variance"),
                ("maximum_sharpe", "maximum_sharpe"),
                ("cvar_minimizing", "cvar_minimizing"),
            ):
                try:
                    optimized = optimize_weights(
                        method,
                        expected,
                        covariance,
                        historical_returns=estimation_array,
                        annual_risk_free_rate=rf_annual,
                        cvar_confidence=float(scenario["cvar_confidence"]),
                        minimum_weight=config.minimum_asset_weight,
                        maximum_weight=float(scenario["maximum_asset_weight"]),
                        maximum_holdings=config.maximum_holdings,
                        tolerance=config.solver_tolerance,
                    )
                    base_results[strategy] = (
                        optimized.weights,
                        {
                            "status": "ok",
                            "solver": optimized.solver,
                            "objective_value": optimized.objective_value,
                            **optimized.diagnostics,
                        },
                    )
                except OptimizationError as error:
                    base_results[strategy] = (None, {"status": "optimization_failed", "error": str(error)})
                    diagnostics.append(
                        {
                            "configuration_id": scenario["configuration_id"],
                            "rebalance_date": decision.strftime("%Y-%m-%d"),
                            "strategy": strategy,
                            "status": "optimization_failed",
                            "error": str(error),
                        }
                    )

            forecast_covariance = None
            forecast_origin = None
            forecast_error = "Forecast input was not provided."
            if forecast_frame is not None:
                forecast_covariance, forecast_origin, forecast_error = _forecast_covariance(
                    forecast_frame,
                    model=forecast_model,
                    rebalance_date=decision,
                    symbols=symbols,
                    estimation_returns=estimation,
                )
            if forecast_covariance is None:
                base_results["minimum_variance_forecast_aware"] = (
                    None,
                    {
                        "status": "forecast_unavailable",
                        "error": forecast_error,
                        "forecast_origin": forecast_origin,
                    },
                )
                diagnostics.append(
                    {
                        "configuration_id": scenario["configuration_id"],
                        "rebalance_date": decision.strftime("%Y-%m-%d"),
                        "strategy": "minimum_variance_forecast_aware",
                        "status": "forecast_unavailable",
                        "forecast_origin": forecast_origin,
                        "error": forecast_error,
                    }
                )
            else:
                try:
                    aware = optimize_weights(
                        "minimum_variance",
                        expected,
                        forecast_covariance,
                        historical_returns=estimation_array,
                        annual_risk_free_rate=rf_annual,
                        minimum_weight=config.minimum_asset_weight,
                        maximum_weight=float(scenario["maximum_asset_weight"]),
                        maximum_holdings=config.maximum_holdings,
                        tolerance=config.solver_tolerance,
                    )
                    base_results["minimum_variance_forecast_aware"] = (
                        aware.weights,
                        {
                            "status": "ok",
                            "solver": aware.solver,
                            "objective_value": aware.objective_value,
                            "forecast_origin": forecast_origin,
                            "risk_estimation_method": (
                                "Forecast annualized volatilities times "
                                "trailing historical correlation matrix."
                            ),
                            **aware.diagnostics,
                        },
                    )
                except OptimizationError as error:
                    base_results["minimum_variance_forecast_aware"] = (
                        None,
                        {
                            "status": "optimization_failed",
                            "error": str(error),
                            "forecast_origin": forecast_origin,
                        },
                    )
                    diagnostics.append(
                        {
                            "configuration_id": scenario["configuration_id"],
                            "rebalance_date": decision.strftime("%Y-%m-%d"),
                            "strategy": "minimum_variance_forecast_aware",
                            "status": "optimization_failed",
                            "error": str(error),
                        }
                    )

            for strategy, (weights, result) in base_results.items():
                status = str(result["status"])
                turnover: float | None = None
                cost_fraction: float | None = None
                if weights is not None:
                    prior = previous_weights[strategy]
                    turnover = (
                        float(np.abs(weights).sum())
                        if prior is None
                        else float(0.5 * np.abs(weights - prior).sum())
                    )
                    cost_fraction = (
                        turnover * float(scenario["transaction_cost_bps"]) / 10000.0
                    )
                    previous_weights[strategy] = weights.copy()
                for position, symbol in enumerate(symbols):
                    weight_rows.append(
                        {
                            "configuration_id": scenario["configuration_id"],
                            **scenario,
                            "rebalance_date": decision.strftime("%Y-%m-%d"),
                            "strategy": strategy,
                            "symbol": symbol,
                            "estimation_start_date": estimation.index.min().strftime("%Y-%m-%d"),
                            "estimation_end_date": estimation.index.max().strftime("%Y-%m-%d"),
                            "estimation_observations": len(estimation),
                            "weight": (
                                float(weights[position])
                                if weights is not None
                                else None
                            ),
                            "optimization_status": status,
                            "solver": result.get("solver"),
                            "objective_value": result.get("objective_value"),
                            "forecast_origin": result.get("forecast_origin"),
                            "risk_estimation_method": (
                                "forecast_volatility_historical_correlation"
                                if strategy.endswith("forecast_aware")
                                and status == "ok"
                                else "historical_window"
                            ),
                        }
                    )
                turnover_rows.append(
                    {
                        "configuration_id": scenario["configuration_id"],
                        **scenario,
                        "rebalance_date": decision.strftime("%Y-%m-%d"),
                        "strategy": strategy,
                        "estimation_start_date": estimation.index.min().strftime("%Y-%m-%d"),
                        "estimation_end_date": estimation.index.max().strftime("%Y-%m-%d"),
                        "estimation_observations": len(estimation),
                        "optimization_status": status,
                        "turnover": turnover,
                        "transaction_cost_bps": float(
                            scenario["transaction_cost_bps"]
                        ),
                        "transaction_cost_fraction": cost_fraction,
                        "forecast_origin": result.get("forecast_origin"),
                    }
                )
                if weights is None:
                    continue
                holding = complete_matrix.loc[
                    (complete_matrix.index >= decision)
                    & (complete_matrix.index < next_decision)
                ]
                for holding_date, row in holding.iterrows():
                    gross_return = float(row.to_numpy(dtype=np.float64) @ weights)
                    first_holding_day = holding_date == decision
                    cost = cost_fraction if first_holding_day else 0.0
                    net_return = (
                        (1.0 + gross_return) * (1.0 - cost) - 1.0
                        if cost is not None
                        else gross_return
                    )
                    daily_rows.append(
                        {
                            "configuration_id": scenario["configuration_id"],
                            **scenario,
                            "date": holding_date.strftime("%Y-%m-%d"),
                            "split": _split(holding_date),
                            "strategy": strategy,
                            "risk_estimation_method": (
                                "forecast_volatility_historical_correlation"
                                if strategy.endswith("forecast_aware")
                                else "historical_window"
                            ),
                            "gross_return": gross_return,
                            "transaction_cost_fraction": float(cost or 0.0),
                            "net_return": float(net_return),
                            "rebalance_date": decision.strftime("%Y-%m-%d"),
                            "turnover": turnover if first_holding_day else 0.0,
                            "transaction_cost_bps": float(
                                scenario["transaction_cost_bps"]
                            ),
                        }
                    )
    daily_returns = pd.DataFrame(daily_rows)
    if not daily_returns.empty:
        daily_returns = daily_returns.sort_values(
            ["configuration_id", "date", "strategy"], kind="stable"
        ).reset_index(drop=True)
    return {
        "daily_returns": daily_returns,
        "weights": pd.DataFrame(weight_rows),
        "turnover": pd.DataFrame(turnover_rows),
        "diagnostics": diagnostics,
        "scenarios": scenarios,
        "symbols": list(symbols),
    }
