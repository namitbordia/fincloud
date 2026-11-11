"""Local analytics pipeline, audit outputs, and command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analytics.beta import calculate_beta, load_benchmark_returns
from src.analytics.correlation import correlation_analysis
from src.analytics.performance import calculate_performance_metrics
from src.analytics.portfolio import (
    calculate_portfolio_returns,
    equal_weights,
    validate_weights,
)
from src.analytics.returns import validate_return_frame
from src.analytics.risk_metrics import drawdown_analysis, historical_var_cvar

LOGGER = logging.getLogger(__name__)
TRADING_DAYS_PER_YEAR = 252
WEIGHT_TOLERANCE = 1e-8
DEFAULT_SYMBOLS = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
EXPECTED_INPUT_COLUMNS = {"date", "symbol", "daily_return", "adj_close"}


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


def _normalize_input(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(EXPECTED_INPUT_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Silver input is missing required columns: {missing}")
    normalized = data.copy(deep=True)
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    normalized["symbol"] = (
        normalized["symbol"].astype("string").str.strip().str.upper()
    )
    for column in (
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "daily_return",
        "log_return",
        "rolling_volatility_20d",
    ):
        if column in normalized:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            )
    return normalized


def _source_numeric_quality(data: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "daily_return",
            "log_return",
            "rolling_volatility_20d",
        )
        if column in data
    ]
    missing: dict[str, dict[str, int]] = {}
    invalid_numeric: dict[str, dict[str, int]] = {}
    invalid_prices: dict[str, dict[str, int]] = {}
    invalid_volume: dict[str, int] = {}
    invalid_return_bounds: dict[str, int] = {}
    invalid_ohlc: dict[str, int] = {}
    for symbol, group in data.groupby("symbol", sort=False, dropna=False):
        symbol_name = str(symbol)
        missing[symbol_name] = {
            column: int(group[column].isna().sum()) for column in numeric_columns
        }
        invalid_numeric[symbol_name] = {}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            invalid = values.notna() & ~np.isfinite(values.to_numpy(dtype=float))
            if invalid.any():
                invalid_numeric[symbol_name][column] = int(invalid.sum())
        invalid_prices[symbol_name] = {}
        for column in ("open", "high", "low", "close", "adj_close"):
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            invalid = values.notna() & (
                ~np.isfinite(values.to_numpy(dtype=float)) | (values <= 0)
            )
            if invalid.any():
                invalid_prices[symbol_name][column] = int(invalid.sum())
        if "volume" in group:
            volume = pd.to_numeric(group["volume"], errors="coerce")
            invalid_volume[symbol_name] = int(
                (volume.notna() & np.isfinite(volume.to_numpy(dtype=float)) & (volume < 0)).sum()
            )
        else:
            invalid_volume[symbol_name] = 0
        if "daily_return" in group:
            returns = pd.to_numeric(group["daily_return"], errors="coerce")
            invalid_return_bounds[symbol_name] = int(
                (
                    returns.notna()
                    & np.isfinite(returns.to_numpy(dtype=float))
                    & (returns < -1.0)
                ).sum()
            )
        else:
            invalid_return_bounds[symbol_name] = 0
        if set(("open", "high", "low", "close")).issubset(group.columns):
            ohlc = group[["open", "high", "low", "close"]].to_numpy(dtype=float)
            complete = np.isfinite(ohlc).all(axis=1)
            bad = complete & (
                (ohlc[:, 1] < np.maximum(ohlc[:, 0], np.maximum(ohlc[:, 2], ohlc[:, 3])))
                | (ohlc[:, 2] > np.minimum(ohlc[:, 0], np.minimum(ohlc[:, 1], ohlc[:, 3])))
            )
            invalid_ohlc[symbol_name] = int(bad.sum())
        else:
            invalid_ohlc[symbol_name] = 0
    return {
        "missing_values_by_symbol_column": missing,
        "non_finite_values_by_symbol_column": invalid_numeric,
        "non_positive_prices_by_symbol_column": invalid_prices,
        "negative_volume_rows_by_symbol": invalid_volume,
        "returns_below_minus_one_by_symbol": invalid_return_bounds,
        "invalid_ohlc_rows_by_symbol": invalid_ohlc,
        "missing_values_are_not_filled": True,
    }


def _exclusion_counts(exclusions: Sequence[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in exclusions:
        reason = str(item["reason"])
        result[reason] = result.get(reason, 0) + 1
    return dict(sorted(result.items()))


def _input_inventory(input_dir: Path) -> list[dict[str, Any]]:
    relevant = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in {".csv", ".parquet"}
            or path.name in {
                "silver_dataset_manifest.json",
                "silver_validation_report.json",
            }
        )
    )
    if not relevant:
        raise FileNotFoundError(f"No Silver input files found in {input_dir}.")
    return [
        {
            "filename": path.name,
            "sha256": _sha256(path),
            "used_for_calculation": path.name == "all_stocks.csv",
        }
        for path in relevant
    ]


def _validate_silver_provenance(input_dir: Path, row_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "silver_validation_report_available": False,
        "adjusted_close_return_provenance_verified": None,
        "silver_manifest_row_count_matches": None,
    }
    report_path = input_dir / "silver_validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result["silver_validation_report_available"] = True
        method = report.get("methodology", {})
        calculations = report.get("checks", {}).get("calculations", {})
        result["adjusted_close_return_provenance_verified"] = (
            method.get("adjusted_close_source_column") == "adj_close"
            and calculations.get("daily_returns_mathematically_correct") is True
        )

    manifest_path = input_dir / "silver_dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["silver_manifest_row_count_matches"] = (
            manifest.get("combined_row_count") == row_count
        )
    return result


def _performance_row(
    entity: str, metrics: Mapping[str, Any], exclusions: int, beta: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {"entity": entity, **metrics}
    availability = row.pop("metric_availability", {})
    row.update(
        {
            f"{metric}_available": available
            for metric, available in availability.items()
        }
    )
    row["excluded_observations"] = exclusions
    row["beta"] = beta["beta"]
    row["beta_available"] = beta["available"]
    row["beta_observation_count"] = beta["observation_count"]
    return row


def run_analytics(
    input_dir: str | Path = Path("data/processed/silver"),
    output_dir: str | Path = Path("data/processed/analytics"),
    *,
    weights: Mapping[str, float] | None = None,
    annual_risk_free_rate: float = 0.0,
    benchmark_csv: str | Path | None = None,
    initial_portfolio_value: float = 1.0,
    long_only: bool = True,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
) -> dict[str, Any]:
    """Generate stock/portfolio analytics and auditable local output files."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    if input_path == output_path or input_path in output_path.parents:
        raise ValueError("Analytics outputs must not be written inside Silver inputs.")
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1.0:
        raise ValueError("annual_risk_free_rate must be finite and greater than -1.")
    if not np.isfinite(initial_portfolio_value) or initial_portfolio_value <= 0:
        raise ValueError("initial_portfolio_value must be finite and greater than zero.")
    selected_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not selected_symbols or len(set(selected_symbols)) != len(selected_symbols):
        raise ValueError("symbols must contain unique stock names.")
    portfolio_weights = validate_weights(
        weights if weights is not None else equal_weights(selected_symbols),
        selected_symbols,
        long_only=long_only,
        tolerance=WEIGHT_TOLERANCE,
    )

    input_files = _input_inventory(input_path)
    if benchmark_csv is not None:
        benchmark_path = Path(benchmark_csv).resolve()
        if not benchmark_path.is_file():
            raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_path}")
        input_files.append(
            {
                "filename": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
                "used_for_calculation": True,
            }
        )
    checksums_before = {item["filename"]: item["sha256"] for item in input_files}
    source_path = input_path / "all_stocks.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Combined Silver CSV not found: {source_path}")
    normalized = _normalize_input(pd.read_csv(source_path))
    if set(normalized["symbol"].dropna().unique()) != set(selected_symbols):
        raise ValueError(
            "Silver input must contain exactly the configured portfolio symbols. "
            f"Found {sorted(normalized['symbol'].dropna().unique())}."
        )
    silver_provenance = _validate_silver_provenance(
        input_path, len(normalized)
    )

    source_order = normalized[["symbol", "date"]].copy()
    original_ordered = source_order.equals(
        source_order.sort_values(["symbol", "date"], kind="stable").reset_index(
            drop=True
        )
    )
    validation = validate_return_frame(normalized)
    source_quality = _source_numeric_quality(normalized)
    clean = validation.returns.rename("daily_return").reset_index()
    clean["symbol"] = clean["symbol"].astype(str)
    clean["date"] = pd.to_datetime(clean["date"], utc=True).dt.tz_convert(None)
    clean = clean.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    exclusions = list(validation.exclusions)
    first_date_by_symbol = (
        normalized.dropna(subset=["date", "symbol"])
        .groupby("symbol")["date"]
        .min()
        .to_dict()
    )
    unexpected_exclusions = [
        item
        for item in exclusions
        if item["reason"] != "missing_return"
        or item.get("symbol") not in first_date_by_symbol
        or item.get("date")
        != first_date_by_symbol[item["symbol"]].strftime("%Y-%m-%d")
    ]

    observed_dates = pd.DatetimeIndex(
        normalized["date"].dropna().unique()
    ).sort_values()
    return_matrix = (
        clean.pivot(index="date", columns="symbol", values="daily_return")
        .reindex(index=observed_dates, columns=selected_symbols)
        .sort_index()
    )
    if return_matrix.index.has_duplicates:
        raise ValueError("Validated return matrix contains duplicate dates.")
    complete_dates = return_matrix.notna().all(axis=1)
    portfolio_returns = calculate_portfolio_returns(
        return_matrix, portfolio_weights
    )
    complete_dates &= portfolio_returns.notna()
    portfolio_exclusions = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "reason": "missing_or_invalid_asset_return",
            "missing_symbols": [
                symbol
                for symbol in selected_symbols
                if pd.isna(return_matrix.at[date, symbol])
            ],
        }
        for date in return_matrix.index[~complete_dates]
    ]
    valid_portfolio_returns = portfolio_returns.loc[complete_dates]
    valid_portfolio_dates = valid_portfolio_returns.index

    benchmark_exclusions: list[dict[str, Any]] = []
    benchmark_returns: pd.Series | None = None
    benchmark_input = Path(benchmark_csv).resolve() if benchmark_csv else None
    if benchmark_input is not None:
        benchmark_returns, benchmark_exclusions = load_benchmark_returns(
            str(benchmark_input)
        )

    stock_metrics: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    beta_results: dict[str, Any] = {}
    exclusions_by_symbol = {
        symbol: [
            item for item in exclusions if item.get("symbol") == symbol
        ]
        for symbol in selected_symbols
    }
    for symbol in selected_symbols:
        symbol_rows = clean.loc[clean["symbol"] == symbol]
        returns = symbol_rows.set_index("date")["daily_return"]
        metrics = calculate_performance_metrics(
            returns.to_numpy(),
            returns.index,
            annual_risk_free_rate=annual_risk_free_rate,
            trading_days=TRADING_DAYS_PER_YEAR,
            initial_value=initial_portfolio_value,
        )
        beta_result = calculate_beta(returns, benchmark_returns)
        beta_results[symbol] = beta_result
        stock_metrics.append(
            _performance_row(
                symbol,
                metrics,
                len(exclusions_by_symbol[symbol]),
                beta_result,
            )
        )
        risk = {"entity": symbol}
        for confidence in (0.95, 0.99):
            tail = historical_var_cvar(returns.to_numpy(), confidence)
            label = int(confidence * 100)
            risk[f"var_{label}"] = tail["var"]
            risk[f"cvar_{label}"] = tail["cvar"]
            risk[f"var_cvar_{label}_available"] = tail["available"]
        _, maximum_drawdown = drawdown_analysis(
            returns.to_numpy(), initial_value=initial_portfolio_value
        ) if len(returns) else (pd.DataFrame(), float("nan"))
        risk["maximum_drawdown"] = maximum_drawdown
        risk_rows.append(risk)

    portfolio_metrics = calculate_performance_metrics(
        valid_portfolio_returns.to_numpy(),
        valid_portfolio_dates,
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=TRADING_DAYS_PER_YEAR,
        initial_value=initial_portfolio_value,
    )
    portfolio_beta = calculate_beta(valid_portfolio_returns, benchmark_returns)
    daily_output = return_matrix.copy()
    daily_output["portfolio_return"] = portfolio_returns
    cumulative = pd.Series(np.nan, index=return_matrix.index, dtype="float64")
    if len(valid_portfolio_returns):
        cumulative_values = (
            initial_portfolio_value
            * (1.0 + valid_portfolio_returns).cumprod()
            / initial_portfolio_value
            - 1.0
        )
        cumulative.loc[valid_portfolio_dates] = cumulative_values
    daily_output["cumulative_return"] = cumulative
    daily_output = daily_output.reset_index(names="date")
    daily_output["date"] = daily_output["date"].dt.strftime("%Y-%m-%d")

    drawdown_frame, portfolio_maximum_drawdown = drawdown_analysis(
        valid_portfolio_returns.to_numpy(),
        initial_value=initial_portfolio_value,
    ) if len(valid_portfolio_returns) else (
        pd.DataFrame(columns=["wealth_index", "running_peak", "drawdown"]),
        float("nan"),
    )
    drawdown_frame.insert(
        0,
        "date",
        valid_portfolio_dates.strftime("%Y-%m-%d"),
    )
    portfolio_metrics["maximum_drawdown"] = portfolio_maximum_drawdown
    portfolio_metrics["beta"] = portfolio_beta["beta"]
    portfolio_metrics["beta_available"] = portfolio_beta["available"]
    portfolio_metrics["beta_observation_count"] = portfolio_beta["observation_count"]
    portfolio_metrics["weighting_method"] = "constant weights, daily rebalanced"
    portfolio_metrics["weights"] = portfolio_weights
    portfolio_metrics["initial_portfolio_value"] = float(initial_portfolio_value)
    portfolio_metrics["benchmark_beta"] = portfolio_beta
    risk_rows.append(
        {
            "entity": "PORTFOLIO",
            **{
                key: value
                for key, value in portfolio_metrics.items()
                if key.startswith(("var_", "cvar_"))
            },
            "maximum_drawdown": portfolio_maximum_drawdown,
        }
    )

    matrix, correlation_details = correlation_analysis(return_matrix)
    correlation_counts = correlation_details.pop("observation_counts")
    if benchmark_returns is None:
        benchmark_details: dict[str, Any] = {
            "supplied": False,
            "beta_available": False,
            "unavailable_reason": "No benchmark supplied; beta is unavailable.",
            "exclusions": [],
        }
    else:
        benchmark_details = {
            "supplied": True,
            "filename": benchmark_input.name if benchmark_input else None,
            "beta_available": portfolio_beta["available"],
            "portfolio_beta": portfolio_beta,
            "asset_beta": beta_results,
            "exclusions": benchmark_exclusions,
            "valid_observation_count": int(len(benchmark_returns)),
        }

    report: dict[str, Any] = {
        "input_file": source_path.name,
        "input_row_count": int(len(normalized)),
        "selected_symbols": list(selected_symbols),
        "input_ordered_by_symbol_and_date": original_ordered,
        "excluded_observations": {
            "count": len(exclusions),
            "by_reason": _exclusion_counts(exclusions),
            "records": exclusions,
        },
        "excluded_portfolio_dates": {
            "count": len(portfolio_exclusions),
            "by_reason": _exclusion_counts(portfolio_exclusions),
            "records": portfolio_exclusions,
        },
        "benchmark": benchmark_details,
        "silver_provenance": silver_provenance,
        "source_numeric_quality": source_quality,
        "correlation": {
            **correlation_details,
            "pairwise_observation_counts": correlation_counts,
        },
        "checks": {
            "required_columns_present": True,
            "no_non_finite_source_numeric_values": not any(
                count
                for per_symbol in source_quality[
                    "non_finite_values_by_symbol_column"
                ].values()
                for count in per_symbol.values()
            ),
            "no_invalid_source_prices": not any(
                count
                for per_symbol in source_quality[
                    "non_positive_prices_by_symbol_column"
                ].values()
                for count in per_symbol.values()
            ),
            "no_invalid_source_ohlc": not any(
                source_quality["invalid_ohlc_rows_by_symbol"].values()
            ),
            "no_negative_source_volume": not any(
                source_quality["negative_volume_rows_by_symbol"].values()
            ),
            "no_returns_below_minus_one": not any(
                source_quality["returns_below_minus_one_by_symbol"].values()
            ),
            "no_missing_source_market_values": not any(
                count
                for per_symbol, column_counts in source_quality[
                    "missing_values_by_symbol_column"
                ].items()
                for column, count in column_counts.items()
                if column
                in {"open", "high", "low", "close", "adj_close", "volume"}
            ),
            "adjusted_close_return_provenance_verified": (
                silver_provenance["adjusted_close_return_provenance_verified"]
                in (None, True)
            ),
            "silver_manifest_row_count_matches": (
                silver_provenance["silver_manifest_row_count_matches"]
                in (None, True)
            ),
            "all_selected_symbols_present": set(clean["symbol"]) == set(selected_symbols),
            "no_duplicate_dates_after_exclusions": not clean.duplicated(
                ["symbol", "date"]
            ).any(),
            "daily_returns_are_finite_and_at_least_minus_one": bool(
                np.isfinite(clean["daily_return"]).all()
                and (clean["daily_return"] >= -1.0).all()
            ),
            "no_missing_returns_in_analytics_input": bool(clean["daily_return"].notna().all()),
            "portfolio_only_uses_complete_aligned_dates": bool(
                portfolio_returns.loc[complete_dates].notna().all()
            ),
            "correlation_does_not_fill_missing_returns": True,
            "benchmark_beta_unavailable_without_benchmark": (
                benchmark_returns is not None
                or not portfolio_beta["available"]
            ),
        },
        "valid": not unexpected_exclusions
        and all(
            item["date"] == (
                return_matrix.index.min().strftime("%Y-%m-%d")
                if len(return_matrix.index)
                else None
            )
            and len(item["missing_symbols"]) == len(selected_symbols)
            for item in portfolio_exclusions
        )
        and original_ordered
        and not benchmark_exclusions
        and silver_provenance["adjusted_close_return_provenance_verified"]
        in (None, True)
        and silver_provenance["silver_manifest_row_count_matches"] in (None, True)
        and not any(
            count
            for per_symbol in source_quality[
                "non_finite_values_by_symbol_column"
            ].values()
            for count in per_symbol.values()
        )
        and not any(
            count
            for per_symbol in source_quality[
                "non_positive_prices_by_symbol_column"
            ].values()
            for count in per_symbol.values()
        )
        and not any(source_quality["invalid_ohlc_rows_by_symbol"].values())
        and not any(source_quality["negative_volume_rows_by_symbol"].values())
        and not any(source_quality["returns_below_minus_one_by_symbol"].values())
        and not any(
            count
            for column_counts in source_quality[
                "missing_values_by_symbol_column"
            ].values()
            for column, count in column_counts.items()
            if column
            in {"open", "high", "low", "close", "adj_close", "volume"}
        )
        and (
            not benchmark_exclusions
            or benchmark_returns is None
        ),
    }
    output_path.mkdir(parents=True, exist_ok=True)

    stock_metrics_frame = pd.DataFrame(stock_metrics)
    stock_metrics_frame.to_csv(
        output_path / "stock_performance_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    daily_output.to_csv(
        output_path / "portfolio_daily_returns.csv",
        index=False,
        float_format="%.17g",
    )
    portfolio_json = {
        "portfolio": _json_safe(portfolio_metrics),
        "stock_betas": _json_safe(beta_results),
        "metric_conventions": {
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "annual_risk_free_rate": annual_risk_free_rate,
            "daily_risk_free_rate": portfolio_metrics["daily_risk_free_rate"],
            "cagr_day_count": "actual elapsed calendar days / 365.25",
            "portfolio_method": "constant configured weights, rebalanced daily",
            "var_cvar": (
                "Historical simulation of losses=-returns; positive values "
                "represent losses; CVaR is mean loss at or beyond the "
                "interpolated VaR threshold."
            ),
        },
    }
    (output_path / "portfolio_performance_metrics.json").write_text(
        json.dumps(_json_safe(portfolio_json), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    matrix.to_csv(output_path / "correlation_matrix.csv", float_format="%.17g")
    drawdown_frame.to_csv(
        output_path / "portfolio_drawdown.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(risk_rows).to_csv(
        output_path / "risk_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    report_path = output_path / "analytics_validation_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    checksums_after = {
        item["filename"]: _sha256(
            Path(item["filename"])
            if Path(item["filename"]).is_absolute()
            else input_path / item["filename"]
        )
        for item in input_files
    }
    inputs_unchanged = checksums_after == checksums_before
    if not inputs_unchanged:
        raise RuntimeError("Silver input files changed during analytics processing.")
    output_names = [
        "stock_performance_metrics.csv",
        "portfolio_daily_returns.csv",
        "portfolio_performance_metrics.json",
        "correlation_matrix.csv",
        "portfolio_drawdown.csv",
        "risk_metrics.csv",
        "analytics_validation_report.json",
    ]
    manifest = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": input_files,
        "input_checksums_unchanged_after_processing": inputs_unchanged,
        "configuration": {
            "selected_symbols": list(selected_symbols),
            "portfolio_weights": portfolio_weights,
            "weighting_method": "constant weights, daily rebalanced",
            "long_only": long_only,
            "weight_sum_tolerance": WEIGHT_TOLERANCE,
            "annual_risk_free_rate": annual_risk_free_rate,
            "daily_risk_free_rate": portfolio_metrics["daily_risk_free_rate"],
            "risk_free_rate_default_note": (
                "0% is a computational fallback, not an estimate of the "
                "Indian risk-free rate."
            ),
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "initial_portfolio_value": initial_portfolio_value,
            "benchmark_csv": str(benchmark_input) if benchmark_input else None,
            "benchmark_source_rows_excluded": len(benchmark_exclusions),
        },
        "date_range": {
            "start": (
                str(clean["date"].min().date()) if not clean.empty else None
            ),
            "end": (
                str(clean["date"].max().date()) if not clean.empty else None
            ),
        },
        "observation_counts": {
            "input_rows": len(normalized),
            "valid_stock_return_rows": len(clean),
            "excluded_stock_observations": len(exclusions),
            "portfolio_aligned_observations": int(complete_dates.sum()),
            "excluded_portfolio_dates": len(portfolio_exclusions),
        },
        "output_filenames": [*output_names, "analytics_manifest.json"],
        "output_sha256": {
            name: _sha256(output_path / name) for name in output_names
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("pandas", "numpy", "pyarrow")
            },
        },
    }
    (output_path / "analytics_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "report": report,
        "manifest": manifest,
        "stock_metrics": stock_metrics_frame,
        "portfolio_daily_returns": daily_output,
        "correlation_matrix": matrix,
        "risk_metrics": pd.DataFrame(risk_rows),
    }


def _parse_weights(values: Sequence[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    result: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid weight {value!r}; use SYMBOL=WEIGHT (for example TCS=0.2)."
            )
        symbol, raw_weight = value.split("=", 1)
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol or normalized_symbol in result:
            raise ValueError(f"Invalid or duplicate portfolio symbol in {value!r}.")
        try:
            result[normalized_symbol] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric weight in {value!r}.") from exc
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate local stock and portfolio risk analytics."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/silver"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/analytics")
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        metavar="SYMBOL=WEIGHT",
        help="Custom weights; omit for equal weights across the five default stocks.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--initial-value", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args(argv)
    try:
        weights = _parse_weights(args.weights)
        result = run_analytics(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            weights=weights,
            annual_risk_free_rate=args.risk_free_rate,
            benchmark_csv=args.benchmark_csv,
            initial_portfolio_value=args.initial_value,
            long_only=not args.allow_short,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Analytics output: {args.output_dir}")
    print(f"Input validation: {'passed' if result['report']['valid'] else 'issues found'}")
    return 0 if result["report"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
