"""Statistical volatility forecasting pipeline, reports, plots, and CLI."""

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

from src.forecasting.config import ForecastConfig
from src.forecasting.evaluation import evaluate_forecasts
from src.forecasting.models import (
    ModelForecast,
    ewma_forecast,
    garch_forecast,
    historical_rolling_forecast,
)
from src.forecasting.targets import build_forward_targets

LOGGER = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
FORECAST_COLUMNS = [
    "date",
    "symbol",
    "split",
    "target_end_date",
    "target_status",
    "target_observations",
    "actual_realized_volatility",
    "model",
    "forecast_volatility",
    "forecast_status",
    "history_observations",
    "garch_convergence_flag",
    "fit_warnings",
    "forecast_error",
]


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


def _input_inventory(input_dir: Path) -> list[dict[str, Any]]:
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in {".csv", ".parquet"}
            or path.name
            in {"silver_dataset_manifest.json", "silver_validation_report.json"}
        )
    )
    if not files:
        raise FileNotFoundError(f"No Silver inputs found in {input_dir}.")
    return [
        {
            "filename": path.name,
            "sha256": _sha256(path),
            "used_for_calculation": path.name == "all_stocks.csv",
        }
        for path in files
    ]


def _read_silver(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = input_dir / "all_stocks.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Combined Silver CSV not found: {source_path}")
    frame = pd.read_csv(source_path)
    required = {"date", "symbol", "daily_return", "adj_close"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Silver input is missing required columns: {missing_columns}")

    original_rows = len(frame)
    frame = frame.copy(deep=True)
    source_row_numbers = np.arange(2, len(frame) + 2, dtype=np.int64)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["date"] = frame["date"].dt.tz_convert(None).dt.normalize()
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    frame["daily_return"] = pd.to_numeric(frame["daily_return"], errors="coerce")

    invalid_date = frame["date"].isna()
    invalid_symbol = frame["symbol"].isna() | frame["symbol"].eq("")
    invalid_keys = invalid_date | invalid_symbol
    duplicate_mask = frame.loc[~invalid_keys].duplicated(
        ["symbol", "date"], keep=False
    )
    duplicate_indices = frame.loc[~invalid_keys].index[duplicate_mask]
    duplicate_mask_all = pd.Series(False, index=frame.index)
    duplicate_mask_all.loc[duplicate_indices] = True
    exclusions: list[dict[str, Any]] = []
    for position in np.flatnonzero(invalid_keys.to_numpy()):
        exclusions.append(
            {
                "source_row_number": int(source_row_numbers[position]),
                "date": (
                    frame.iloc[position]["date"].strftime("%Y-%m-%d")
                    if pd.notna(frame.iloc[position]["date"])
                    else None
                ),
                "symbol": (
                    str(frame.iloc[position]["symbol"])
                    if pd.notna(frame.iloc[position]["symbol"])
                    else None
                ),
                "reason": (
                    "invalid_date"
                    if invalid_date.iloc[position]
                    else "missing_or_invalid_symbol"
                ),
            }
        )
    for position in duplicate_indices:
        row = frame.loc[position]
        exclusions.append(
            {
                "source_row_number": int(source_row_numbers[position]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "symbol": str(row["symbol"]),
                "reason": "duplicate_symbol_date",
            }
        )
    excluded = invalid_keys | duplicate_mask_all
    processable = frame.loc[~excluded].copy()
    processable = processable.sort_values(
        ["symbol", "date"], kind="stable"
    ).reset_index(drop=True)

    returns = frame["daily_return"].to_numpy(dtype=np.float64, na_value=np.nan)
    nonfinite_returns = frame["daily_return"].notna().to_numpy() & ~np.isfinite(returns)
    below_minus_one = np.isfinite(returns) & (returns < -1.0)
    missing_returns = frame["daily_return"].isna()
    first_date_by_symbol = (
        processable.dropna(subset=["date", "symbol"])
        .groupby("symbol")["date"]
        .min()
        .to_dict()
    )
    missing_return_exclusions: list[dict[str, Any]] = []
    for position in np.flatnonzero(missing_returns.to_numpy()):
        row = frame.iloc[position]
        reason = "missing_return"
        first_date = first_date_by_symbol.get(str(row["symbol"]))
        if pd.notna(row["date"]) and first_date == row["date"]:
            reason = "first_observation_without_prior_return"
        missing_return_exclusions.append(
            {
                "source_row_number": int(source_row_numbers[position]),
                "date": (
                    row["date"].strftime("%Y-%m-%d")
                    if pd.notna(row["date"])
                    else None
                ),
                "symbol": str(row["symbol"]) if pd.notna(row["symbol"]) else None,
                "reason": reason,
            }
        )
    invalid_return_exclusions = [
        {
            "source_row_number": int(source_row_numbers[position]),
            "date": (
                frame.iloc[position]["date"].strftime("%Y-%m-%d")
                if pd.notna(frame.iloc[position]["date"])
                else None
            ),
            "symbol": str(frame.iloc[position]["symbol"]),
            "reason": (
                "non_finite_return"
                if nonfinite_returns[position]
                else "return_below_minus_one"
            ),
        }
        for position in np.flatnonzero(nonfinite_returns | below_minus_one)
    ]

    provenance: bool | None = None
    silver_report_path = input_dir / "silver_validation_report.json"
    if silver_report_path.is_file():
        silver_report = json.loads(silver_report_path.read_text(encoding="utf-8"))
        method = silver_report.get("methodology", {})
        calculations = silver_report.get("checks", {}).get("calculations", {})
        provenance = (
            method.get("adjusted_close_source_column") == "adj_close"
            and calculations.get("daily_returns_mathematically_correct") is True
        )
    missing_by_symbol = {
        str(symbol): int(group["daily_return"].isna().sum())
        for symbol, group in frame.groupby("symbol", sort=True, dropna=False)
    }
    report = {
        "input_rows": int(original_rows),
        "processable_rows": int(len(processable)),
        "symbols_found": sorted(
            str(item) for item in processable["symbol"].dropna().unique()
        ),
        "date_range": {
            "start": (
                processable["date"].min().strftime("%Y-%m-%d")
                if not processable.empty
                else None
            ),
            "end": (
                processable["date"].max().strftime("%Y-%m-%d")
                if not processable.empty
                else None
            ),
        },
        "duplicate_symbol_date_rows": int(duplicate_mask_all.sum()),
        "invalid_date_rows": int(invalid_date.sum()),
        "invalid_symbol_rows": int(invalid_symbol.sum()),
        "missing_returns_by_symbol": missing_by_symbol,
        "non_finite_return_rows": int(nonfinite_returns.sum()),
        "returns_below_minus_one_rows": int(below_minus_one.sum()),
        "adjusted_close_return_provenance_verified": provenance,
        "documented_input_exclusions": exclusions
        + missing_return_exclusions
        + invalid_return_exclusions,
    }
    return processable, report


def _trailing_valid_history(returns: np.ndarray, position: int) -> np.ndarray:
    """Return only the contiguous finite history ending at this origin."""
    history = returns[: position + 1]
    valid = np.isfinite(history) & (history >= -1.0)
    invalid_positions = np.flatnonzero(~valid)
    start = int(invalid_positions[-1] + 1) if len(invalid_positions) else 0
    return history[start:]


def _model_forecasts(
    history: np.ndarray, config: ForecastConfig
) -> tuple[ModelForecast, ModelForecast, ModelForecast]:
    return (
        historical_rolling_forecast(
            history,
            window=config.horizon,
            trading_days_per_year=config.trading_days_per_year,
        ),
        ewma_forecast(
            history,
            decay=config.ewma_lambda,
            initialization_window=config.horizon,
            trading_days_per_year=config.trading_days_per_year,
        ),
        garch_forecast(
            history,
            horizon=config.horizon,
            min_observations=config.min_garch_observations,
            trading_days_per_year=config.trading_days_per_year,
        ),
    )


def _forecast_series(
    data: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: ForecastConfig,
) -> pd.DataFrame:
    target_lookup = {
        (row.symbol, row.date): row
        for row in targets.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for symbol, group in data.groupby("symbol", sort=True):
        group = group.sort_values("date", kind="stable").reset_index(drop=True)
        returns = group["daily_return"].to_numpy(dtype=np.float64, na_value=np.nan)
        for position, row in group.iterrows():
            origin = row["date"]
            if origin <= pd.Timestamp(config.train_end):
                continue
            target = target_lookup.get((symbol, origin.strftime("%Y-%m-%d")))
            if target is None:
                raise RuntimeError(
                    f"No target record exists for {symbol} at {origin:%Y-%m-%d}."
                )
            history = _trailing_valid_history(returns, position)
            model_results = _model_forecasts(history, config)
            for result in model_results:
                if result.status == "fit_failed":
                    LOGGER.warning(
                        "%s fit failed for %s on %s: %s",
                        result.model,
                        symbol,
                        origin.strftime("%Y-%m-%d"),
                        result.error,
                    )
                rows.append(
                    {
                        "date": origin.strftime("%Y-%m-%d"),
                        "symbol": str(symbol),
                        "split": target.split,
                        "target_end_date": target.target_end_date,
                        "target_status": target.target_status,
                        "target_observations": target.target_observations,
                        "actual_realized_volatility": target.actual_realized_volatility,
                        "model": result.model,
                        "forecast_volatility": result.volatility,
                        "forecast_status": result.status,
                        "history_observations": result.history_observations,
                        "garch_convergence_flag": result.convergence_flag,
                        "fit_warnings": " | ".join(result.warnings) or None,
                        "forecast_error": result.error,
                    }
                )
    forecasts = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    if not forecasts.empty:
        forecasts = forecasts.sort_values(
            ["symbol", "date", "model"], kind="stable"
        ).reset_index(drop=True)
    return forecasts


def _raw_source_verification(
    raw_dir: Path | None,
    silver_manifest_path: Path,
) -> dict[str, Any]:
    if raw_dir is None or not silver_manifest_path.is_file():
        return {
            "available": False,
            "verified": None,
            "verified_files": 0,
            "mismatches": [],
        }
    manifest = json.loads(silver_manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("source_files", [])
    mismatches: list[dict[str, str]] = []
    verified = 0
    for entry in entries:
        path = raw_dir / entry["filename"]
        if not path.is_file():
            mismatches.append(
                {"filename": entry["filename"], "reason": "missing_source_file"}
            )
        elif _sha256(path) != entry["sha256"]:
            mismatches.append(
                {"filename": entry["filename"], "reason": "checksum_mismatch"}
            )
        else:
            verified += 1
    return {
        "available": True,
        "verified": not mismatches and verified == len(entries),
        "verified_files": verified,
        "expected_files": len(entries),
        "mismatches": mismatches,
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _save_plots(
    forecasts: pd.DataFrame,
    model_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    plot_dir: Path,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "actual_vs_forecast_test.png",
        "model_comparison_test.png",
        "per_stock_performance_test.png",
        "forecast_errors_over_time_test.png",
    ]
    test = forecasts.loc[forecasts["split"] == "test"].copy()
    symbols = sorted(str(item) for item in forecasts["symbol"].unique())
    models = ("historical_rolling", "ewma", "garch_1_1")
    colors = {
        "historical_rolling": "#1f77b4",
        "ewma": "#ff7f0e",
        "garch_1_1": "#2ca02c",
    }

    figure, axes = plt.subplots(
        max(len(symbols), 1), 1, figsize=(12, max(3.0 * len(symbols), 4.0)), sharex=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, symbol in zip(axes_array, symbols, strict=False):
        symbol_rows = test.loc[test["symbol"] == symbol]
        origin_targets = symbol_rows.loc[
            symbol_rows["model"] == models[0], ["date", "target_status"]
        ]
        target_unavailable = int(
            origin_targets["target_status"].ne("available").sum()
        )
        axis.text(
            0.99,
            0.98,
            f"Target unavailable: {target_unavailable}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize="small",
        )
        actual = symbol_rows.loc[
            symbol_rows["model"] == models[0],
            ["date", "actual_realized_volatility"],
        ]
        dates = pd.to_datetime(actual["date"], errors="coerce")
        axis.plot(
            dates,
            actual["actual_realized_volatility"],
            color="black",
            linewidth=1.5,
            label="Actual forward realized volatility",
        )
        for model in models:
            series = symbol_rows.loc[
                symbol_rows["model"] == model,
                ["date", "forecast_volatility", "forecast_status"],
            ]
            model_dates = pd.to_datetime(series["date"], errors="coerce")
            failure_count = int(series["forecast_status"].ne("ok").sum())
            axis.plot(
                model_dates,
                series["forecast_volatility"],
                color=colors[model],
                linewidth=1.0,
                label=f"{model} (unavailable: {failure_count})",
            )
        axis.set_title(symbol)
        axis.set_ylabel("Annualized volatility")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", fontsize="small", ncol=2)
    if not symbols:
        axes_array[0].text(0.5, 0.5, "No test forecasts", ha="center", va="center")
    axes_array[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes_array[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes_array[-1].xaxis.get_major_locator())
    )
    figure.suptitle("Actual and forecast volatility — test period")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[0], dpi=160, metadata={"Software": "FinCloud"})
    plt.close(figure)

    comparison = model_metrics.loc[
        model_metrics["split"] == "test"
    ].set_index("model")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(
        axes, ("mae", "rmse", "qlike"), ("MAE", "RMSE", "QLIKE"), strict=True
    ):
        values = (
            comparison.reindex(models)[metric]
            if not comparison.empty
            else pd.Series(index=models, dtype=float)
        )
        axis.bar(models, values, color=[colors[model] for model in models])
        axis.set_title(title)
        axis.set_ylabel("Loss (lower is better)")
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Statistical model comparison — test period")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[1], dpi=160, metadata={"Software": "FinCloud"})
    plt.close(figure)

    stock_test = stock_metrics.loc[stock_metrics["split"] == "test"]
    pivot = (
        stock_test.pivot(index="symbol", columns="model", values="rmse")
        .reindex(index=symbols, columns=models)
        if not stock_test.empty
        else pd.DataFrame(index=symbols, columns=models, dtype=float)
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    if not pivot.empty:
        pivot.plot(kind="bar", ax=axis, color=[colors[model] for model in models])
    else:
        axis.text(0.5, 0.5, "No test metrics", ha="center", va="center")
    axis.set_title("Per-stock forecast performance — test period")
    axis.set_ylabel("RMSE (annualized volatility)")
    axis.set_xlabel("Stock")
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[2], dpi=160, metadata={"Software": "FinCloud"})
    plt.close(figure)

    figure, axes = plt.subplots(
        max(len(symbols), 1), 1, figsize=(12, max(3.0 * len(symbols), 4.0)), sharex=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, symbol in zip(axes_array, symbols, strict=False):
        symbol_rows = test.loc[test["symbol"] == symbol]
        for model in models:
            series = symbol_rows.loc[symbol_rows["model"] == model].copy()
            error = (
                series["forecast_volatility"]
                - series["actual_realized_volatility"]
            )
            unavailable_count = int(
                series["forecast_status"].ne("ok").sum()
                + series["target_status"].ne("available").sum()
            )
            axis.plot(
                pd.to_datetime(series["date"], errors="coerce"),
                error,
                color=colors[model],
                linewidth=0.9,
                label=f"{model} (unavailable: {unavailable_count})",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(symbol)
        axis.set_ylabel("Forecast error")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", fontsize="small")
    if not symbols:
        axes_array[0].text(0.5, 0.5, "No test forecasts", ha="center", va="center")
    axes_array[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes_array[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes_array[-1].xaxis.get_major_locator())
    )
    figure.suptitle("Forecast errors — test period (forecast minus actual)")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[3], dpi=160, metadata={"Software": "FinCloud"})
    plt.close(figure)
    return outputs


def run_forecasting(
    input_dir: str | Path = Path("data/processed/silver"),
    output_dir: str | Path = Path("data/processed/forecasting/statistical"),
    *,
    raw_dir: str | Path | None = Path("data/raw/nse"),
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    config: ForecastConfig = ForecastConfig(),
) -> dict[str, Any]:
    """Run reproducible statistical volatility forecasts without changing inputs."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    raw_path = Path(raw_dir).resolve() if raw_dir is not None else None
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("Forecast outputs must not be written inside Silver inputs.")
    if raw_path is not None and (
        output_path == raw_path or raw_path in output_path.parents
    ):
        raise ValueError("Forecast outputs must not be written inside raw inputs.")
    analytics_path = Path("data/processed/analytics").resolve()
    if output_path == analytics_path or analytics_path in output_path.parents:
        raise ValueError("Forecast outputs must not be written inside Phase 4 outputs.")
    selected_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not selected_symbols or len(set(selected_symbols)) != len(selected_symbols):
        raise ValueError("symbols must contain unique stock names.")

    input_files = _input_inventory(input_path)
    checksums_before = {item["filename"]: item["sha256"] for item in input_files}
    data, input_report = _read_silver(input_path)
    present_symbols = set(data["symbol"].dropna().astype(str))
    absent_symbols = sorted(set(selected_symbols) - present_symbols)
    unexpected_symbols = sorted(present_symbols - set(selected_symbols))
    input_report["expected_symbols"] = list(selected_symbols)
    input_report["absent_symbols"] = absent_symbols
    input_report["unexpected_symbols"] = unexpected_symbols
    quality_valid = (
        not absent_symbols
        and not unexpected_symbols
        and input_report["duplicate_symbol_date_rows"] == 0
        and input_report["invalid_date_rows"] == 0
        and input_report["invalid_symbol_rows"] == 0
        and input_report["non_finite_return_rows"] == 0
        and input_report["returns_below_minus_one_rows"] == 0
        and input_report["adjusted_close_return_provenance_verified"] is not False
    )
    data = data.loc[data["symbol"].isin(selected_symbols)].copy()

    targets = build_forward_targets(
        data,
        horizon=config.horizon,
        trading_days_per_year=config.trading_days_per_year,
        train_end=config.train_end,
        validation_end=config.validation_end,
    )
    forecasts = _forecast_series(data, targets, config=config)
    model_metrics, stock_metrics = evaluate_forecasts(
        forecasts, variance_floor=config.qlike_variance_floor
    )
    if not forecasts.empty:
        forecasts = forecasts.loc[:, FORECAST_COLUMNS]

    output_path.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_path / "statistical_forecasts.csv", index=False)
    model_metrics.to_csv(output_path / "statistical_model_metrics.csv", index=False)
    stock_metrics.to_csv(output_path / "statistical_stock_metrics.csv", index=False)

    target_status_counts = {
        str(key): int(value)
        for key, value in forecasts["target_status"].value_counts(dropna=False).items()
    } if not forecasts.empty else {}
    target_status_origin_counts = (
        {
            str(key): int(value)
            for key, value in (
                forecasts[["symbol", "date", "target_status"]]
                .drop_duplicates()
                .groupby("target_status", dropna=False)
                .size()
                .items()
            )
        }
        if not forecasts.empty
        else {}
    )
    forecast_status_counts = {
        str(key): int(value)
        for key, value in forecasts["forecast_status"].value_counts(dropna=False).items()
    } if not forecasts.empty else {}
    fit_failures = (
        forecasts.loc[
            forecasts["forecast_status"] == "fit_failed",
            [
                "date",
                "symbol",
                "model",
                "garch_convergence_flag",
                "fit_warnings",
                "forecast_error",
            ],
        ].to_dict(orient="records")
        if not forecasts.empty
        else []
    )
    fit_warning_count = (
        int(forecasts["fit_warnings"].notna().sum())
        if not forecasts.empty
        else 0
    )
    result_keys_unique = (
        not forecasts.duplicated(["symbol", "date", "model"]).any()
        if not forecasts.empty
        else True
    )
    input_report.update(
        {
            "valid": bool(quality_valid and result_keys_unique),
            "checks": {
                "all_configured_symbols_present": not absent_symbols
                and not unexpected_symbols,
                "no_duplicate_input_symbol_dates": input_report[
                    "duplicate_symbol_date_rows"
                ]
                == 0,
                "unique_forecast_symbol_date_model": result_keys_unique,
                "forecast_origins_are_chronological": bool(
                    forecasts.empty
                    or forecasts.groupby("symbol", sort=False)["date"]
                    .apply(lambda dates: dates.is_monotonic_increasing)
                    .all()
                ),
                "features_cut_off_at_forecast_origin": True,
                "targets_use_only_next_horizon_rows": True,
                "missing_values_are_not_filled": True,
                "source_adjusted_close_provenance": input_report[
                    "adjusted_close_return_provenance_verified"
                ],
            },
            "forecast_rows": int(len(forecasts)),
            "target_status_counts": target_status_counts,
            "target_status_origin_counts": target_status_origin_counts,
            "forecast_status_counts": forecast_status_counts,
            "failed_fit_count": int(len(fit_failures)),
            "fit_failures": fit_failures,
            "fit_warning_count": fit_warning_count,
            "date_splits": {
                "training_origin_start": input_report["date_range"]["start"],
                "training_origin_end": config.train_end,
                "validation_origin_start_exclusive": config.train_end,
                "validation_origin_end": config.validation_end,
                "test_origin_start_exclusive": config.validation_end,
                "test_origin_end": input_report["date_range"]["end"],
                "boundary_crossing_targets_excluded": True,
            },
            "excluded_input_records": input_report["documented_input_exclusions"],
        }
    )
    report_path = output_path / "statistical_validation_report.json"
    report_path.write_text(
        json.dumps(_json_safe(input_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    plot_files = _save_plots(
        forecasts, model_metrics, stock_metrics, output_path / "plots"
    )
    raw_verification = _raw_source_verification(
        raw_path, input_path / "silver_dataset_manifest.json"
    )
    input_checksums_after = {
        item["filename"]: _sha256(input_path / item["filename"])
        for item in input_files
    }
    silver_unchanged = checksums_before == input_checksums_after
    if not silver_unchanged:
        raise RuntimeError("Silver input checksums changed during forecasting.")
    if raw_verification["available"] and not raw_verification["verified"]:
        LOGGER.error("Raw input checksum verification failed: %s", raw_verification)
    input_report["silver_source_checksums_unchanged"] = silver_unchanged
    input_report["raw_source_checksum_verification"] = raw_verification
    input_report["checks"]["raw_source_checksums_match_phase3_manifest"] = (
        raw_verification["verified"]
    )
    if raw_verification["available"]:
        input_report["valid"] = bool(
            input_report["valid"] and raw_verification["verified"]
        )
    report_path.write_text(
        json.dumps(_json_safe(input_report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    output_files = [
        output_path / "statistical_forecasts.csv",
        output_path / "statistical_model_metrics.csv",
        output_path / "statistical_stock_metrics.csv",
        report_path,
        *(output_path / "plots" / name for name in plot_files),
    ]
    manifest = {
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(input_path),
        "source_files": input_files,
        "source_checksums_unchanged_after_processing": silver_unchanged,
        "raw_source_checksum_verification": raw_verification,
        "stock_symbols": list(selected_symbols),
        "date_range": input_report["date_range"],
        "row_counts": {
            "silver_input_rows": input_report["input_rows"],
            "processable_rows": input_report["processable_rows"],
            "forecast_comparison_rows": len(forecasts),
            "valid_actual_targets": int(
                forecasts.loc[
                    forecasts["target_status"] == "available",
                    ["symbol", "date"],
                ].drop_duplicates().shape[0]
            )
            if not forecasts.empty
            else 0,
        },
        "model_configuration": {
            "target": "sqrt((252 / horizon) * sum(next_horizon_daily_return ** 2))",
            "horizon_trading_sessions": config.horizon,
            "annualization_trading_days": config.trading_days_per_year,
            "training_origin_end": config.train_end,
            "validation_origin_end": config.validation_end,
            "ewma_lambda": config.ewma_lambda,
            "ewma_initialization": (
                "Sample variance (ddof=1) of first horizon contiguous returns; "
                "then lambda * previous_variance + (1-lambda) * current_return^2."
            ),
            "historical_baseline": (
                "Annualized sample standard deviation of the trailing horizon "
                "returns, used as a persistence forecast, not a forward estimate."
            ),
            "garch": {
                "specification": "Constant mean, normal innovations, GARCH(1,1)",
                "variance_equation": (
                    "h(t)=omega+alpha*epsilon(t-1)^2+beta*h(t-1)"
                ),
                "minimum_history_observations": config.min_garch_observations,
                "fit_window": "Expanding contiguous history through each forecast origin",
                "horizon": config.horizon,
                "return_scale_for_estimation": 100.0,
            },
            "qlike_variance_floor": config.qlike_variance_floor,
            "missing_and_invalid_returns": (
                "Never filled; break the contiguous history and invalidate "
                "forward targets containing the return."
            ),
            "split_boundary_target_rule": (
                f"Validation targets crossing {config.validation_end} are excluded; "
                "test targets must have the complete horizon inside available test data."
            ),
            "random_seed": None,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "packages": {
                package: _version(package)
                for package in ("numpy", "pandas", "arch", "matplotlib")
            },
        },
        "output_files": [
            {
                "filename": str(path.relative_to(output_path)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in output_files
        ],
    }
    manifest_path = output_path / "statistical_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "forecasts": forecasts,
        "model_metrics": model_metrics,
        "stock_metrics": stock_metrics,
        "report": input_report,
        "manifest": manifest,
    }
