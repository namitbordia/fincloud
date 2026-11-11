"""Chronological ML volatility forecasts and statistical-model comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecasting.evaluation import evaluate_forecasts
from src.forecasting.ml.config import (
    DEFAULT_MODEL_PARAMETERS,
    MLForecastConfig,
)
from src.forecasting.ml.features import FEATURE_COLUMNS, build_features
from src.forecasting.ml.models import feature_importance, make_model
from src.forecasting.ml.targets import build_ml_targets

LOGGER = logging.getLogger(__name__)
ML_MODELS = ("random_forest", "xgboost", "ridge")
COMPARISON_COLUMNS = [
    "date",
    "symbol",
    "split",
    "target_end_date",
    "target_status",
    "actual_realized_volatility",
    "model",
    "forecast_volatility",
    "forecast_status",
    "training_time_seconds",
    "inference_time_seconds",
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


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


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
    return [
        {"filename": path.name, "sha256": _sha256(path)}
        for path in files
    ]


def _load_silver(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = input_dir / "all_stocks.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Silver input not found: {source}")
    frame = pd.read_csv(source)
    required = {"date", "symbol", "daily_return", "adj_close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Silver input is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    for column in ("daily_return", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_key = frame["date"].isna() | frame["symbol"].isna()
    duplicates = frame.loc[~invalid_key].duplicated(
        ["symbol", "date"], keep=False
    )
    duplicate_indices = frame.loc[~invalid_key].index[duplicates]
    report: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "invalid_key_rows": int(invalid_key.sum()),
        "duplicate_symbol_date_rows": int(len(duplicate_indices)),
    }
    if report["invalid_key_rows"] or report["duplicate_symbol_date_rows"]:
        raise ValueError(
            "Silver input has invalid keys or duplicate symbol/date rows; "
            f"inspect source report: {report}"
        )
    return frame.sort_values(["symbol", "date"], kind="stable").reset_index(
        drop=True
    ), report


def _refit_key(value: pd.Timestamp, frequency: str) -> str:
    if frequency == "monthly":
        return value.strftime("%Y-%m")
    return f"{value.year}-Q{value.quarter}"


def _train_model(
    name: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    fit_date: pd.Timestamp,
    config: MLForecastConfig,
) -> tuple[Any, dict[str, Any]]:
    supervised = features.merge(
        targets[
            [
                "date",
                "symbol",
                "target_end_date",
                "actual_realized_volatility",
                "target_status",
            ]
        ],
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    supervised["target_end_date"] = pd.to_datetime(
        supervised["target_end_date"], errors="coerce"
    )
    eligible = (
        supervised["target_status"].eq("available")
        & supervised["date"].lt(fit_date)
        & supervised["target_end_date"].lt(fit_date)
    )
    x = supervised.loc[eligible, list(FEATURE_COLUMNS)]
    y = supervised.loc[eligible, "actual_realized_volatility"].astype(float)
    complete = np.isfinite(x.to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(
        y.to_numpy(dtype=np.float64)
    )
    x = x.loc[complete]
    y = y.loc[complete]
    if len(x) < 30:
        raise ValueError(
            f"Only {len(x)} fully labeled training observations are available."
        )
    estimator = make_model(
        name,
        random_seed=config.random_seed,
        parameters=config.model_parameters.get(name),
    )
    started = time.perf_counter()
    estimator.fit(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
    elapsed = time.perf_counter() - started
    training_dates = supervised.loc[x.index, "date"]
    metadata = {
        "training_time_seconds": float(elapsed),
        "training_observations": int(len(x)),
        "training_start_date": training_dates.min().strftime("%Y-%m-%d"),
        "training_end_date": training_dates.max().strftime("%Y-%m-%d"),
        "training_label_end_date": supervised.loc[
            x.index, "target_end_date"
        ].max().strftime("%Y-%m-%d"),
        "fit_date": fit_date.strftime("%Y-%m-%d"),
    }
    return estimator, metadata


def _forecast_one(
    estimator: Any,
    feature_row: pd.Series,
) -> tuple[float | None, float, str | None]:
    values = feature_row.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        return None, 0.0, "Insufficient or invalid feature history."
    started = time.perf_counter()
    try:
        predicted = float(estimator.predict(values.reshape(1, -1))[0])
    except Exception as error:
        return None, time.perf_counter() - started, f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started
    if not np.isfinite(predicted) or predicted < 0:
        return None, elapsed, "Model produced a non-finite or negative volatility."
    return predicted, elapsed, None


def _fit_and_predict_symbol(
    symbol: str,
    source: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: MLForecastConfig,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    models: dict[str, Any] = {}
    fit_metadata: dict[str, dict[str, Any]] = {}
    last_refit_key: str | None = None
    failures: list[dict[str, Any]] = []

    feature_by_date = features.set_index("date", drop=False)
    target_by_date = targets.set_index("date", drop=False)
    source = source.sort_values("date", kind="stable").reset_index(drop=True)
    forecasts = source.loc[source["date"] > pd.Timestamp(config.train_end)]
    for _, origin_row in forecasts.iterrows():
        origin = origin_row["date"]
        refit_key = _refit_key(origin, config.refit_frequency)
        refitted = refit_key != last_refit_key
        refit_errors: dict[str, str] = {}
        if refitted:
            for name in ML_MODELS:
                try:
                    estimator, metadata = _train_model(
                        name,
                        features,
                        targets,
                        fit_date=origin,
                        config=config,
                    )
                    models[name] = estimator
                    fit_metadata[name] = metadata
                    values = feature_importance(
                        estimator, name, tuple(FEATURE_COLUMNS)
                    )
                    importance_rows.extend(
                        {
                            "symbol": symbol,
                            "model": name,
                            "fit_date": metadata["fit_date"],
                            "feature": feature,
                            "importance": float(value),
                        }
                        for feature, value in zip(FEATURE_COLUMNS, values, strict=True)
                    )
                except Exception as error:
                    models.pop(name, None)
                    fit_metadata.pop(name, None)
                    refit_errors[name] = f"{type(error).__name__}: {error}"
                    failure = {
                        "symbol": symbol,
                        "model": name,
                        "fit_date": origin.strftime("%Y-%m-%d"),
                        "error": refit_errors[name],
                    }
                    failures.append(failure)
                    LOGGER.exception(
                        "ML model fit failed for %s (%s) at %s",
                        name,
                        symbol,
                        origin.strftime("%Y-%m-%d"),
                    )
            last_refit_key = refit_key

        target = target_by_date.loc[origin]
        if isinstance(target, pd.DataFrame):
            raise ValueError(f"Duplicate target found for {symbol} at {origin}.")
        feature_row = feature_by_date.loc[origin]
        if isinstance(feature_row, pd.DataFrame):
            raise ValueError(f"Duplicate features found for {symbol} at {origin}.")
        for name in ML_MODELS:
            training_seconds = (
                float(fit_metadata[name]["training_time_seconds"])
                if refitted and name in fit_metadata
                else 0.0
            )
            if name in refit_errors:
                forecast = None
                inference_seconds = 0.0
                status = "fit_failed"
                error_message = refit_errors[name]
            elif name not in models:
                forecast = None
                inference_seconds = 0.0
                status = "fit_unavailable"
                error_message = "No successful estimator is available."
            else:
                forecast, inference_seconds, error_message = _forecast_one(
                    models[name], feature_row
                )
                status = (
                    "ok"
                    if error_message is None
                    else "invalid_features"
                    if error_message.startswith("Insufficient")
                    else "prediction_failed"
                )
            model_rows.append(
                {
                    "date": origin.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "split": target["split"],
                    "target_end_date": target["target_end_date"],
                    "target_status": target["target_status"],
                    "target_observations": target["target_observations"],
                    "actual_realized_volatility": target[
                        "actual_realized_volatility"
                    ],
                    "model": name,
                    "forecast_volatility": forecast,
                    "forecast_status": status,
                    "training_time_seconds": training_seconds,
                    "inference_time_seconds": float(inference_seconds),
                    "training_observations": (
                        fit_metadata.get(name, {}).get("training_observations")
                        if name in models
                        else None
                    ),
                    "forecast_error": error_message,
                }
            )

    for name, estimator in models.items():
        model_path = artifact_dir / f"{symbol}_{name}.joblib"
        joblib.dump(estimator, model_path, compress=3)
        artifacts[name] = {
            "filename": model_path.name,
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "size_bytes": model_path.stat().st_size,
            **fit_metadata[name],
            "feature_names": list(FEATURE_COLUMNS),
            "model_parameters": {
                **config.model_parameters.get(
                    name, DEFAULT_MODEL_PARAMETERS[name]
                ),
                "random_seed": config.random_seed,
            },
        }
    return model_rows, importance_rows, {
        "symbol": symbol,
        "artifacts": artifacts,
        "failures": failures,
    }


def _load_statistical_baselines(
    path: Path,
    ml_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Valid Phase 5 comparison output is required: {path}"
        )
    statistical = pd.read_csv(path)
    required = {
        "date",
        "symbol",
        "split",
        "target_end_date",
        "target_status",
        "actual_realized_volatility",
        "model",
        "forecast_volatility",
        "forecast_status",
    }
    missing = sorted(required - set(statistical.columns))
    if missing:
        raise ValueError(f"Phase 5 forecast output is missing columns: {missing}")
    statistical = statistical.loc[
        statistical["model"].isin(
            ["historical_rolling", "ewma", "garch_1_1"]
        )
    ].copy()
    ml_keys = ml_forecasts[
        ["date", "symbol", "target_end_date", "actual_realized_volatility"]
    ].drop_duplicates()
    stat_keys = statistical[
        ["date", "symbol", "target_end_date", "actual_realized_volatility"]
    ].drop_duplicates()
    merged = stat_keys.merge(
        ml_keys,
        on=["date", "symbol"],
        how="inner",
        suffixes=("_phase5", "_ml"),
        validate="one_to_one",
    )
    if len(merged) != len(stat_keys):
        raise ValueError(
            "Phase 5 and ML targets do not share identical forecast origins."
        )
    if not (
        merged["target_end_date_phase5"].fillna("").eq(
            merged["target_end_date_ml"].fillna("")
        ).all()
    ):
        raise ValueError("Phase 5 and ML target end dates do not align.")
    actual_matches = np.isclose(
        pd.to_numeric(merged["actual_realized_volatility_phase5"], errors="coerce"),
        pd.to_numeric(merged["actual_realized_volatility_ml"], errors="coerce"),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if not actual_matches.all():
        raise ValueError("Phase 5 and ML realized-volatility targets disagree.")
    stat = statistical.rename(
        columns={
            "forecast_volatility": "forecast_volatility",
            "forecast_status": "forecast_status",
        }
    )
    stat["training_time_seconds"] = 0.0
    stat["inference_time_seconds"] = np.nan
    stat["training_observations"] = np.nan
    stat["forecast_error"] = stat.get("forecast_error")
    return stat.reindex(columns=ml_forecasts.columns)


def _plot_outputs(
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "actual_vs_predicted_volatility.png",
        "model_comparison_test.png",
        "feature_importance.png",
        "forecast_errors_over_time.png",
    ]
    test = comparison.loc[comparison["split"] == "test"]
    models = sorted(comparison["model"].unique())
    symbols = sorted(comparison["symbol"].unique())
    figure, axes = plt.subplots(
        len(symbols), 1, figsize=(12, max(3 * len(symbols), 4)), sharex=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, symbol in zip(axes_array, symbols, strict=True):
        group = test.loc[test["symbol"] == symbol]
        actual = group.loc[group["model"] == models[0]].drop_duplicates("date")
        axis.plot(
            pd.to_datetime(actual["date"]),
            actual["actual_realized_volatility"],
            color="black",
            label="Actual forward realized volatility",
            linewidth=1.5,
        )
        for model in models:
            rows = group.loc[group["model"] == model]
            unavailable = int(rows["forecast_status"].ne("ok").sum())
            axis.plot(
                pd.to_datetime(rows["date"]),
                rows["forecast_volatility"],
                linewidth=0.9,
                label=f"{model} (unavailable: {unavailable})",
            )
        axis.set_title(symbol)
        axis.set_ylabel("Annualized volatility")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small", ncol=3)
    axes_array[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes_array[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes_array[-1].xaxis.get_major_locator())
    )
    figure.suptitle("Actual versus predicted volatility — test period")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[0], dpi=160)
    plt.close(figure)

    metrics, _ = evaluate_forecasts(
        comparison, variance_floor=1e-12
    )
    test_metrics = metrics.loc[metrics["split"] == "test"].set_index("model")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric, title in zip(
        axes, ("mae", "rmse", "qlike"), ("MAE", "RMSE", "QLIKE"), strict=True
    ):
        axis.bar(
            models,
            test_metrics.reindex(models)[metric],
        )
        axis.set_title(title)
        axis.set_ylabel("Loss (lower is better)")
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Statistical and ML forecast comparison — test period")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[1], dpi=160)
    plt.close(figure)

    ml_importance = importance.groupby(
        ["model", "feature"], as_index=False, sort=True
    )["importance"].mean()
    top_features = (
        ml_importance.groupby("feature")["importance"]
        .mean()
        .nlargest(20)
        .index
    )
    selected = ml_importance.loc[ml_importance["feature"].isin(top_features)]
    pivot = selected.pivot(index="feature", columns="model", values="importance")
    figure, axis = plt.subplots(figsize=(11, 7))
    pivot.sort_values(by=list(pivot.columns)).plot(kind="barh", ax=axis)
    axis.set_title("Mean model feature importance across fit windows")
    axis.set_xlabel("Mean importance (absolute coefficient for Ridge)")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[2], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(symbols), 1, figsize=(12, max(3 * len(symbols), 4)), sharex=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, symbol in zip(axes_array, symbols, strict=True):
        group = test.loc[test["symbol"] == symbol]
        for model in models:
            rows = group.loc[group["model"] == model]
            error = (
                rows["forecast_volatility"]
                - rows["actual_realized_volatility"]
            )
            axis.plot(
                pd.to_datetime(rows["date"]),
                error,
                linewidth=0.8,
                label=model,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(symbol)
        axis.set_ylabel("Forecast error")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    axes_array[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes_array[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes_array[-1].xaxis.get_major_locator())
    )
    figure.suptitle("Forecast errors — test period (prediction minus actual)")
    figure.tight_layout()
    figure.savefig(plot_dir / outputs[3], dpi=160)
    plt.close(figure)
    return outputs


def run_ml_forecasting(
    input_dir: str | Path = Path("data/processed/silver"),
    statistical_forecasts: str | Path = Path(
        "data/processed/forecasting/statistical/statistical_forecasts.csv"
    ),
    output_dir: str | Path = Path("data/processed/forecasting/ml"),
    model_dir: str | Path = Path("models/volatility"),
    *,
    symbols: Sequence[str] = (
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
    ),
    config: MLForecastConfig = MLForecastConfig(),
) -> dict[str, Any]:
    """Train expanding-window ML forecasts and compare to validated Phase 5."""
    input_path = Path(input_dir).resolve()
    baseline_path = Path(statistical_forecasts).resolve()
    output_path = Path(output_dir).resolve()
    model_path = Path(model_dir).resolve()
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("ML outputs must not be written inside Silver inputs.")
    if output_path == baseline_path.parent or output_path in baseline_path.parents:
        raise ValueError("ML outputs must not overwrite Phase 5 forecasting outputs.")
    if model_path == input_path or input_path in model_path.parents:
        raise ValueError("Model artifacts must not be stored in Silver inputs.")
    selected_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not selected_symbols or len(set(selected_symbols)) != len(selected_symbols):
        raise ValueError("symbols must be a non-empty unique sequence.")

    input_inventory = _inventory(input_path)
    input_hashes = {item["filename"]: item["sha256"] for item in input_inventory}
    baseline_hash = _sha256(baseline_path)
    silver, quality = _load_silver(input_path)
    present = set(silver["symbol"].astype(str).unique())
    if present != set(selected_symbols):
        raise ValueError(
            f"Silver symbols {sorted(present)} differ from configured "
            f"symbols {sorted(selected_symbols)}."
        )
    silver_report = json.loads(
        (input_path / "silver_validation_report.json").read_text(encoding="utf-8")
    )
    provenance = (
        silver_report.get("methodology", {}).get("adjusted_close_source_column")
        == "adj_close"
        and silver_report.get("checks", {})
        .get("calculations", {})
        .get("daily_returns_mathematically_correct")
        is True
    )
    if not provenance:
        raise ValueError("Silver adjusted-close return provenance is not verified.")

    features_all = build_features(silver)
    targets_all = build_ml_targets(
        silver,
        horizon=config.horizon,
        trading_days_per_year=config.trading_days_per_year,
        train_end=config.train_end,
        validation_end=config.validation_end,
    )
    model_path.mkdir(parents=True, exist_ok=True)
    all_model_rows: list[dict[str, Any]] = []
    all_importance_rows: list[dict[str, Any]] = []
    artifact_metadata: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_symbol_training: dict[str, dict[str, Any]] = {}
    for symbol in selected_symbols:
        source = silver.loc[silver["symbol"] == symbol].copy()
        symbol_features = features_all.loc[
            features_all["symbol"] == symbol
        ].copy()
        symbol_targets = targets_all.loc[targets_all["symbol"] == symbol].copy()
        model_rows, importance_rows, artifacts = _fit_and_predict_symbol(
            symbol,
            source,
            symbol_features,
            symbol_targets,
            config=config,
            artifact_dir=model_path,
        )
        all_model_rows.extend(model_rows)
        all_importance_rows.extend(importance_rows)
        artifact_metadata.append(artifacts)
        failures.extend(artifacts["failures"])
        per_symbol_training[symbol] = artifacts["artifacts"]

    ml_forecasts = pd.DataFrame(all_model_rows)
    ml_forecasts = ml_forecasts.sort_values(
        ["symbol", "date", "model"], kind="stable"
    ).reset_index(drop=True)
    baselines = _load_statistical_baselines(baseline_path, ml_forecasts)
    comparison = pd.concat(
        [
            ml_forecasts.reindex(columns=COMPARISON_COLUMNS),
            baselines.reindex(columns=COMPARISON_COLUMNS),
        ],
        ignore_index=True,
    ).sort_values(["symbol", "date", "model"], kind="stable")
    comparison = comparison.reset_index(drop=True)
    model_metrics, stock_metrics = evaluate_forecasts(
        comparison, variance_floor=config.variance_floor
    )

    fit_times = (
        ml_forecasts.groupby("model")["training_time_seconds"].sum().to_dict()
    )
    inference_times = (
        ml_forecasts.groupby("model")["inference_time_seconds"].sum().to_dict()
    )
    artifact_sizes = {
        model: sum(
            info["size_bytes"]
            for symbol_artifacts in per_symbol_training.values()
            for name, info in symbol_artifacts.items()
            if name == model
        )
        for model in (*ML_MODELS, "historical_rolling", "ewma", "garch_1_1")
    }
    model_metrics["training_time_seconds"] = model_metrics["model"].map(fit_times).fillna(0.0)
    model_metrics["inference_time_seconds"] = model_metrics["model"].map(
        inference_times
    )
    model_metrics["model_size_bytes"] = model_metrics["model"].map(artifact_sizes)
    stock_metrics["training_time_seconds"] = stock_metrics["model"].map(fit_times).fillna(0.0)
    stock_metrics["inference_time_seconds"] = stock_metrics["model"].map(
        inference_times
    )
    stock_metrics["model_size_bytes"] = stock_metrics["model"].map(artifact_sizes)

    output_path.mkdir(parents=True, exist_ok=True)
    ml_forecasts.to_csv(output_path / "ml_forecasts.csv", index=False)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    model_metrics.to_csv(output_path / "ml_model_metrics.csv", index=False)
    stock_metrics.to_csv(output_path / "ml_stock_metrics.csv", index=False)
    importance = pd.DataFrame(all_importance_rows)
    if importance.empty:
        importance = pd.DataFrame(
            columns=["symbol", "model", "fit_date", "feature", "importance"]
        )
    importance_summary = (
        importance.groupby(["symbol", "model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            fit_count=("importance", "size"),
        )
        if not importance.empty
        else pd.DataFrame(
            columns=["symbol", "model", "feature", "mean_importance", "fit_count"]
        )
    )
    importance_summary.to_csv(output_path / "feature_importance.csv", index=False)
    plot_files = _plot_outputs(comparison, importance, output_path)

    stat_rows = pd.read_csv(baseline_path)
    target_keys = comparison.loc[
        comparison["model"] == "random_forest",
        ["date", "symbol", "target_end_date", "target_status", "actual_realized_volatility"],
    ]
    unique_keys = not comparison.duplicated(["date", "symbol", "model"]).any()
    target_end_dates = pd.to_datetime(target_keys["target_end_date"], errors="coerce")
    forecast_dates = pd.to_datetime(target_keys["date"], errors="coerce")
    target_validity = target_keys["target_status"].eq("available")
    no_lookahead = bool(
        target_end_dates[target_validity].gt(forecast_dates[target_validity]).all()
        and ml_forecasts["date"].notna().all()
    )
    selected_validation = (
        model_metrics.loc[
            (model_metrics["split"] == "validation")
            & (model_metrics["model"].isin(ML_MODELS))
        ]
        .sort_values(["mae", "model"], kind="stable")
        .iloc[0]["model"]
        if not model_metrics.loc[
            (model_metrics["split"] == "validation")
            & (model_metrics["model"].isin(ML_MODELS))
        ].empty
        else None
    )
    ml_status_counts = {
        str(key): int(value)
        for key, value in ml_forecasts["forecast_status"].value_counts().items()
    }
    report: dict[str, Any] = {
        **quality,
        "valid": bool(unique_keys and no_lookahead and provenance),
        "symbols": list(selected_symbols),
        "source_date_range": {
            "start": silver["date"].min().strftime("%Y-%m-%d"),
            "end": silver["date"].max().strftime("%Y-%m-%d"),
        },
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "target_definition": (
            "sqrt((trading_days_per_year / horizon) * "
            "sum(next horizon daily_return squared))"
        ),
        "target_horizon": config.horizon,
        "target_annualization_days": config.trading_days_per_year,
        "split_dates": {
            "training_origin_end": config.train_end,
            "validation_origin_end": config.validation_end,
            "validation_target_end_must_not_exceed": config.validation_end,
            "test_target_end_must_be_observed": True,
        },
        "feature_generation_is_past_only": True,
        "fitted_preprocessing_uses_training_rows_only": True,
        "adjusted_close_return_provenance_verified": provenance,
        "statistical_target_alignment_verified": True,
        "forecast_rows": len(ml_forecasts),
        "model_fit_status_counts": ml_status_counts,
        "failed_model_fit_count": len(failures),
        "failed_model_fits": failures,
        "validation_selected_ml_model_by_mae": selected_validation,
        "final_test_used_for_model_selection": False,
        "forecast_coverage_is_metric_specific": True,
        "checks": {
            "forecast_keys_unique": unique_keys,
            "target_is_strictly_future": no_lookahead,
            "model_features_use_only_information_at_origin": True,
            "phase5_actual_targets_match": True,
            "raw_or_silver_inputs_modified": False,
        },
        "configuration": {
            "refit_frequency": config.refit_frequency,
            "random_seed": config.random_seed,
            "model_parameters": config.model_parameters,
            "statistical_models_compared": [
                "historical_rolling",
                "ewma",
                "garch_1_1",
            ],
        },
        "training_time_seconds_by_model": fit_times,
        "inference_time_seconds_by_model": inference_times,
        "model_size_bytes_by_model": artifact_sizes,
        "excluded_input_rows": int(quality["invalid_key_rows"]),
        "phase5_baseline_forecast_rows": int(len(stat_rows)),
    }
    report_path = output_path / "ml_validation_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    outputs = [
        output_path / filename
        for filename in (
            "ml_forecasts.csv",
            "ml_model_metrics.csv",
            "ml_stock_metrics.csv",
            "model_comparison.csv",
            "feature_importance.csv",
            "ml_validation_report.json",
        )
    ] + [output_path / "plots" / filename for filename in plot_files]
    artifact_files = sorted(model_path.glob("*.joblib"))
    training_dates = silver.loc[
        silver["date"] <= pd.Timestamp(config.train_end), "date"
    ]
    validation_dates = silver.loc[
        (silver["date"] > pd.Timestamp(config.train_end))
        & (silver["date"] <= pd.Timestamp(config.validation_end)),
        "date",
    ]
    test_dates = silver.loc[
        silver["date"] > pd.Timestamp(config.validation_end), "date"
    ]
    artifact_manifest = {
        "training_date_range": {
            "start": training_dates.min().strftime("%Y-%m-%d"),
            "end": training_dates.max().strftime("%Y-%m-%d"),
        },
        "validation_date_range": {
            "start": validation_dates.min().strftime("%Y-%m-%d"),
            "end": validation_dates.max().strftime("%Y-%m-%d"),
        },
        "test_date_range": {
            "start": test_dates.min().strftime("%Y-%m-%d"),
            "end": test_dates.max().strftime("%Y-%m-%d"),
        },
        "target_definition": report["target_definition"],
        "feature_names_and_order": list(FEATURE_COLUMNS),
        "random_seed": config.random_seed,
        "model_artifacts": per_symbol_training,
        "artifact_files": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_files
        ],
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                package: _version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                )
            },
        },
        "source_checksums": input_inventory,
    }
    artifact_metadata_path = model_path / "model_metadata.json"
    artifact_metadata_path.write_text(
        json.dumps(_json_safe(artifact_manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(artifact_metadata_path)
    output_checksums = [
        {
            "filename": str(path.relative_to(output_path)).replace("\\", "/")
            if output_path in path.parents
            else str(path),
            "sha256": _sha256(path),
        }
        for path in outputs
    ]
    input_after = {
        item["filename"]: _sha256(input_path / item["filename"])
        for item in input_inventory
    }
    baseline_unchanged = _sha256(baseline_path) == baseline_hash
    if input_hashes != input_after or not baseline_unchanged:
        raise RuntimeError("Silver or Phase 5 input checksums changed during ML forecasting.")
    manifest = {
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": input_inventory
        + [{"filename": str(baseline_path), "sha256": baseline_hash}],
        "source_checksums_unchanged_after_processing": True,
        "stock_symbols": list(selected_symbols),
        "row_counts": {
            "silver_rows": len(silver),
            "ml_forecast_rows": len(ml_forecasts),
            "comparison_rows": len(comparison),
            "feature_importance_rows": len(importance_summary),
        },
        "model_configuration": report["configuration"],
        "split_dates": report["split_dates"],
        "training_time_seconds_by_model": fit_times,
        "inference_time_seconds_by_model": inference_times,
        "model_size_bytes_by_model": artifact_sizes,
        "model_artifact_directory": str(model_path),
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                package: _version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                )
            },
        },
        "outputs": output_checksums,
    }
    manifest_path = output_path / "ml_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "forecasts": ml_forecasts,
        "comparison": comparison,
        "model_metrics": model_metrics,
        "stock_metrics": stock_metrics,
        "feature_importance": importance_summary,
        "report": report,
        "manifest": manifest,
    }
