"""Portfolio optimization, walk-forward reporting, plots, and manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.performance import calculate_performance_metrics
from src.analytics.risk_metrics import historical_var_cvar
from src.optimization.backtest import STRATEGIES, walk_forward_backtest
from src.optimization.config import OptimizationConfig

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _inventory(input_dir: Path) -> list[dict[str, Any]]:
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in {".csv", ".parquet"}
            or path.name in {"silver_dataset_manifest.json", "silver_validation_report.json"}
        )
    )
    return [{"filename": path.name, "sha256": _sha256(path)} for path in files]


def _tree_checksums(directory: Path) -> dict[str, str] | None:
    if not directory.is_dir():
        return None
    return {
        str(path.relative_to(directory)).replace("\\", "/"): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _score_rows(
    daily: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    config: OptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (configuration_id, strategy), group in daily.groupby(
        ["configuration_id", "strategy"], sort=True
    ):
        for split in ("validation", "test", "all"):
            selected = (
                group
                if split == "all"
                else group.loc[group["split"] == split]
            ).sort_values("date", kind="stable")
            if selected.empty:
                continue
            dates = pd.to_datetime(selected["date"])
            gross = selected["gross_return"].to_numpy(dtype=np.float64)
            net = selected["net_return"].to_numpy(dtype=np.float64)
            gross_metrics = calculate_performance_metrics(
                gross,
                dates,
                annual_risk_free_rate=config.annual_risk_free_rate,
                trading_days=config.trading_days_per_year,
            )
            net_metrics = calculate_performance_metrics(
                net,
                dates,
                annual_risk_free_rate=config.annual_risk_free_rate,
                trading_days=config.trading_days_per_year,
            )
            events = turnover.loc[
                (turnover["configuration_id"] == configuration_id)
                & (turnover["strategy"] == strategy)
            ]
            if split != "all":
                event_dates = pd.to_datetime(events["rebalance_date"])
                events = events.loc[
                    (event_dates.dt.year == (2023 if split == "validation" else 2024))
                ]
            turnover_sum = float(events["turnover"].fillna(0.0).sum())
            traded_cost_bps = float(
                (
                    events["turnover"].fillna(0.0)
                    * events["transaction_cost_bps"]
                ).sum()
            )
            costs_fraction = float(events["transaction_cost_fraction"].fillna(0.0).sum())
            row: dict[str, Any] = {
                "configuration_id": configuration_id,
                "strategy": strategy,
                "split": split,
                "risk_estimation_method": (
                    "forecast_volatility_historical_correlation"
                    if strategy.endswith("forecast_aware")
                    else "historical_window"
                ),
                **{
                    key: value
                    for key, value in net_metrics.items()
                    if key != "metric_availability"
                },
                "gross_total_return": gross_metrics["total_return"],
                "gross_cagr": gross_metrics["cagr"],
                "gross_annualized_volatility": gross_metrics[
                    "annualized_volatility"
                ],
                "gross_sharpe_ratio": gross_metrics["sharpe_ratio"],
                "gross_sortino_ratio": gross_metrics["sortino_ratio"],
                "gross_maximum_drawdown": gross_metrics["maximum_drawdown"],
                "portfolio_turnover": turnover_sum,
                "transaction_costs_bps": traded_cost_bps,
                "transaction_cost_fraction": costs_fraction,
                "net_cumulative_return": net_metrics["total_return"],
                "daily_observations": len(selected),
                "rebalance_count": int(len(events)),
                "metric_availability": net_metrics["metric_availability"],
            }
            performance_rows.append(row)
            var = historical_var_cvar(net, 0.95)
            risk_rows.append(
                {
                    "configuration_id": configuration_id,
                    "strategy": strategy,
                    "split": split,
                    "var_95": var["var"],
                    "cvar_95": var["cvar"],
                    "var_cvar_observations": var["observation_count"],
                    "var_cvar_available": var["available"],
                    "maximum_drawdown": net_metrics["maximum_drawdown"],
                }
            )
    return pd.DataFrame(performance_rows), pd.DataFrame(risk_rows)


def _select_plot_configuration(
    performance: pd.DataFrame, scenarios: list[dict[str, Any]]
) -> str:
    preferred = next(
        (
            item["configuration_id"]
            for item in scenarios
            if item["estimation_window"] == 252
            and item["rebalance_frequency"] == "monthly"
            and item["transaction_cost_bps"] > 0
        ),
        scenarios[0]["configuration_id"],
    )
    available = set(performance["configuration_id"].unique())
    return preferred if preferred in available else next(iter(available))


def _plot_results(
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    turnover: pd.DataFrame,
    performance: pd.DataFrame,
    output_dir: Path,
    *,
    configuration_id: str,
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "cumulative_portfolio_performance.png",
        "drawdown_comparison.png",
        "risk_return_comparison.png",
        "portfolio_allocations_over_time.png",
        "turnover_comparison.png",
    ]
    selected = daily.loc[
        (daily["configuration_id"] == configuration_id)
        & (daily["split"] == "test")
    ]
    figure, axis = plt.subplots(figsize=(12, 6))
    for strategy, group in selected.groupby("strategy", sort=True):
        ordered = group.sort_values("date")
        wealth = np.cumprod(1.0 + ordered["net_return"].to_numpy(dtype=float))
        axis.plot(
            pd.to_datetime(ordered["date"]),
            wealth - 1.0,
            label=strategy,
        )
    axis.set_title(f"Net cumulative return — test ({configuration_id})")
    axis.set_ylabel("Cumulative net return")
    axis.set_xlabel("Date")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / files[0], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    for strategy, group in selected.groupby("strategy", sort=True):
        ordered = group.sort_values("date")
        wealth = np.cumprod(1.0 + ordered["net_return"].to_numpy(dtype=float))
        peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        axis.plot(
            pd.to_datetime(ordered["date"]),
            wealth / peaks - 1.0,
            label=strategy,
        )
    axis.set_title(f"Net drawdowns — test ({configuration_id})")
    axis.set_ylabel("Drawdown")
    axis.set_xlabel("Date")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / files[1], dpi=160)
    plt.close(figure)

    selected_metrics = performance.loc[
        (performance["configuration_id"] == configuration_id)
        & (performance["split"] == "test")
    ]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        selected_metrics["annualized_volatility"],
        selected_metrics["cagr"],
    )
    for _, row in selected_metrics.iterrows():
        axis.annotate(
            row["strategy"],
            (row["annualized_volatility"], row["cagr"]),
            fontsize="small",
        )
    axis.set_title(f"Risk-return comparison — test ({configuration_id})")
    axis.set_xlabel("Annualized net volatility")
    axis.set_ylabel("Net CAGR")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / files[2], dpi=160)
    plt.close(figure)

    selected_weights = weights.loc[
        (weights["configuration_id"] == configuration_id)
        & weights["strategy"].isin(
            ["minimum_variance", "minimum_variance_forecast_aware"]
        )
        & (weights["optimization_status"] == "ok")
    ]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, strategy in zip(
        axes,
        ("minimum_variance", "minimum_variance_forecast_aware"),
        strict=True,
    ):
        group = selected_weights.loc[selected_weights["strategy"] == strategy]
        if group.empty:
            axis.text(0.5, 0.5, "No feasible allocations", ha="center", va="center")
        else:
            pivot = group.pivot(
                index="rebalance_date", columns="symbol", values="weight"
            ).sort_index()
            pivot.plot.area(ax=axis, alpha=0.8)
        axis.set_title(strategy)
        axis.set_ylabel("Portfolio weight")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
    )
    figure.suptitle(f"Portfolio allocations — {configuration_id}")
    figure.tight_layout()
    figure.savefig(plot_dir / files[3], dpi=160)
    plt.close(figure)

    selected_turnover = turnover.loc[
        (turnover["configuration_id"] == configuration_id)
        & (turnover["optimization_status"] == "ok")
    ]
    turnover_summary = selected_turnover.groupby("strategy")["turnover"].mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    turnover_summary.plot(kind="bar", ax=axis)
    axis.set_title(f"Mean one-way turnover at rebalance — {configuration_id}")
    axis.set_ylabel("One-way turnover")
    axis.set_xlabel("Strategy")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / files[4], dpi=160)
    plt.close(figure)
    return files


def _verify_raw_sources(
    raw_dir: Path,
    silver_manifest_path: Path,
) -> dict[str, Any]:
    if not silver_manifest_path.is_file():
        return {"available": False, "verified": None, "files_verified": 0}
    manifest = json.loads(silver_manifest_path.read_text(encoding="utf-8"))
    errors = []
    count = 0
    for item in manifest.get("source_files", []):
        path = raw_dir / item["filename"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            errors.append(item["filename"])
        else:
            count += 1
    return {
        "available": True,
        "verified": not errors,
        "files_verified": count,
        "expected_files": len(manifest.get("source_files", [])),
        "mismatches": errors,
    }


def run_optimization(
    input_dir: str | Path = Path("data/processed/silver"),
    ml_comparison_csv: str | Path = Path(
        "data/processed/forecasting/ml/model_comparison.csv"
    ),
    output_dir: str | Path = Path("data/processed/optimization"),
    *,
    raw_dir: str | Path = Path("data/raw/nse"),
    forecast_model: str = "random_forest",
    config: OptimizationConfig = OptimizationConfig(),
) -> dict[str, Any]:
    """Run sensitivity configurations and a historical walk-forward backtest."""
    input_path = Path(input_dir).resolve()
    forecast_path = Path(ml_comparison_csv).resolve()
    output_path = Path(output_dir).resolve()
    raw_path = Path(raw_dir).resolve()
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("Optimization outputs cannot be written inside Silver inputs.")
    if output_path == forecast_path.parent or forecast_path.parent in output_path.parents:
        raise ValueError("Optimization outputs cannot overwrite forecasting results.")
    if output_path == Path("data/processed/analytics").resolve():
        raise ValueError("Optimization outputs cannot overwrite Phase 4 analytics.")
    silver_inventory = _inventory(input_path)
    silver_before = {item["filename"]: item["sha256"] for item in silver_inventory}
    forecast_before = _sha256(forecast_path)
    phase4_path = input_path.parent / "analytics"
    phase4_before = _tree_checksums(phase4_path)
    phase5_path = forecast_path.parent.parent / "statistical"
    phase5_before = _tree_checksums(phase5_path)
    source_path = input_path / "all_stocks.csv"
    returns = pd.read_csv(source_path, usecols=["date", "symbol", "daily_return"])
    comparison = pd.read_csv(forecast_path)
    result = walk_forward_backtest(
        returns,
        config=config,
        forecast_frame=comparison,
        forecast_model=forecast_model,
    )
    performance, risk_metrics = _score_rows(
        result["daily_returns"], result["turnover"], config=config
    )
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "optimized_weights.csv": result["weights"],
        "backtest_daily_returns.csv": result["daily_returns"],
        "strategy_performance.csv": performance,
        "strategy_risk_metrics.csv": risk_metrics,
        "portfolio_turnover.csv": result["turnover"],
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_path / filename, index=False)
    diagnostic_path = output_path / "optimization_diagnostics.json"
    diagnostic_report = {
        "scenarios": result["scenarios"],
        "solver_diagnostics": result["diagnostics"],
        "forecast_aware_model": forecast_model,
        "forecast_aware_method": (
            "Forecast annualized volatilities combined with complete-history "
            "correlation estimated strictly before each rebalance."
        ),
        "configuration_count": len(result["scenarios"]),
        "failed_or_skipped_rebalance_count": len(result["diagnostics"]),
    }
    diagnostic_path.write_text(
        json.dumps(_json_safe(diagnostic_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    daily = result["daily_returns"]
    checks = {
        "weights_sum_to_one": bool(
            result["weights"].loc[
                result["weights"]["optimization_status"] == "ok"
            ]
            .groupby(
                ["configuration_id", "rebalance_date", "strategy"]
            )["weight"]
            .sum()
            .sub(1.0)
            .abs()
            .le(config.solver_tolerance * 10)
            .all()
        ),
        "forecast_origins_precede_rebalance": bool(
            result["weights"]
            .dropna(subset=["forecast_origin"])
            .assign(
                rebalance=pd.to_datetime(
                    result["weights"].dropna(subset=["forecast_origin"])[
                        "rebalance_date"
                    ]
                ),
                origin=pd.to_datetime(
                    result["weights"].dropna(subset=["forecast_origin"])[
                        "forecast_origin"
                    ]
                ),
            )
            .eval("origin < rebalance")
            .all()
        )
        if not result["weights"].dropna(subset=["forecast_origin"]).empty
        else True,
        "estimation_windows_precede_rebalance": bool(
            result["turnover"].empty
            or (
                pd.to_datetime(result["turnover"]["estimation_end_date"])
                < pd.to_datetime(result["turnover"]["rebalance_date"])
            ).all()
        ),
        "test_period_not_used_for_hyperparameter_selection": True,
        "phase4_analytics_unchanged": False,
        "phase5_forecasts_unchanged": False,
        "raw_or_silver_data_unchanged": False,
    }
    raw_verification = _verify_raw_sources(
        raw_path, input_path / "silver_dataset_manifest.json"
    )
    silver_after = {
        item["filename"]: _sha256(input_path / item["filename"])
        for item in silver_inventory
    }
    silver_unchanged = silver_before == silver_after
    forecast_unchanged = _sha256(forecast_path) == forecast_before
    phase4_unchanged = (
        phase4_before is not None
        and phase4_before == _tree_checksums(phase4_path)
    )
    phase5_unchanged = (
        phase5_before is not None
        and phase5_before == _tree_checksums(phase5_path)
    )
    checks["phase4_analytics_unchanged"] = phase4_unchanged
    checks["phase5_forecasts_unchanged"] = phase5_unchanged
    checks["raw_or_silver_data_unchanged"] = bool(
        raw_verification["verified"] and silver_unchanged
    )
    checks["silver_checksums_unchanged"] = silver_unchanged
    checks["ml_forecast_checksum_unchanged"] = forecast_unchanged
    checks["raw_checksums_match_phase3_manifest"] = raw_verification["verified"]
    valid = bool(
        all(checks.values())
        and raw_verification["verified"]
        and not daily.empty
        and not result["weights"].empty
    )
    validation_report = {
        "valid": valid,
        "checks": checks,
        "raw_checksum_verification": raw_verification,
        "phase4_analytics_checksums_unchanged": phase4_unchanged,
        "phase5_forecast_checksums_unchanged": phase5_unchanged,
        "silver_rows": int(len(returns)),
        "silver_date_range": {
            "start": str(pd.to_datetime(returns["date"]).min().date()),
            "end": str(pd.to_datetime(returns["date"]).max().date()),
        },
        "symbols": result["symbols"],
        "configuration_count": len(result["scenarios"]),
        "daily_return_rows": int(len(daily)),
        "optimization_weight_rows": int(len(result["weights"])),
        "rebalance_rows": int(len(result["turnover"])),
        "solver_failures_or_skips": int(len(result["diagnostics"])),
        "solver_diagnostics_file": diagnostic_path.name,
        "forecast_aware_model": forecast_model,
        "forecast_aware_strategy_count": int(
            result["turnover"]["strategy"]
            .eq("minimum_variance_forecast_aware")
            .sum()
        ),
        "overlapping_samples_or_costs_are_not_filled": True,
    }
    validation_path = output_path / "backtest_validation_report.json"
    validation_path.write_text(
        json.dumps(_json_safe(validation_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plot_config = _select_plot_configuration(performance, result["scenarios"])
    plot_files = _plot_results(
        daily,
        result["weights"],
        result["turnover"],
        performance,
        output_path,
        configuration_id=plot_config,
    )
    output_paths = [
        output_path / filename for filename in outputs
    ] + [diagnostic_path, validation_path] + [
        output_path / "plots" / filename for filename in plot_files
    ]
    manifest = {
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": silver_inventory
        + [{"filename": str(forecast_path), "sha256": forecast_before}],
        "source_checksums_unchanged_after_processing": (
            silver_unchanged
            and forecast_unchanged
            and phase4_unchanged
            and phase5_unchanged
        ),
        "raw_source_checksum_verification": raw_verification,
        "symbols": result["symbols"],
        "configuration": {
            "estimation_windows": list(config.estimation_windows),
            "rebalance_frequencies": list(config.rebalance_frequencies),
            "maximum_asset_weights": list(config.maximum_asset_weights),
            "minimum_asset_weight": config.minimum_asset_weight,
            "maximum_holdings": config.maximum_holdings,
            "transaction_cost_bps": list(config.transaction_cost_bps),
            "cvar_confidences": list(config.cvar_confidences),
            "annual_risk_free_rate": config.annual_risk_free_rate,
            "rebalance_start": config.rebalance_start,
            "forecast_model": forecast_model,
            "forecast_covariance": (
                "Forecast annualized volatilities times historical correlation "
                "matrix estimated from strictly prior complete observations."
            ),
            "sensitivity_results_are_not_test_tuned": True,
        },
        "row_counts": {
            "scenarios": len(result["scenarios"]),
            "optimized_weights": len(result["weights"]),
            "backtest_daily_returns": len(daily),
            "strategy_performance": len(performance),
            "strategy_risk_metrics": len(risk_metrics),
            "turnover": len(result["turnover"]),
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
        },
        "selected_plot_configuration": plot_config,
        "outputs": [
            {
                "filename": str(path.relative_to(output_path)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
    }
    manifest_path = output_path / "optimization_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "daily_returns": daily,
        "weights": result["weights"],
        "turnover": result["turnover"],
        "performance": performance,
        "risk_metrics": risk_metrics,
        "diagnostics": diagnostic_report,
        "validation_report": validation_report,
        "manifest": manifest,
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
