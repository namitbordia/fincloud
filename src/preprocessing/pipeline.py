"""Build a validated, reproducible Silver layer from local NSE market data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
OHLC_COLUMNS = ("open", "high", "low", "close")
SOURCE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
DATA_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "daily_return",
    "log_return",
    "rolling_volatility_20d",
    "rolling_volatility_warmup",
    "source_row_number",
    "quality_flags",
)
FLAG_EXPLANATIONS = {
    "invalid_date": "The source date could not be parsed.",
    "missing_source_symbol": (
        "The source row had no symbol; the individual source filename was used."
    ),
    "source_symbol_mismatch": (
        "The source symbol did not match the individual stock filename; "
        "the filename symbol is authoritative."
    ),
    "missing_adjusted_close": (
        "The source adjusted-close field is absent or missing; no other price "
        "field is substituted."
    ),
    "missing_value": "A source market-data value is missing.",
    "non_numeric_value": "A source market-data value is not numeric.",
    "invalid_price": "A price is non-finite or not strictly positive.",
    "invalid_ohlc": (
        "High is below an OHLC value or low is above an OHLC value."
    ),
    "negative_volume": "Source volume is negative.",
    "invalid_volume": "Source volume is non-finite.",
    "non_integer_volume": "Source volume is not an integer-valued quantity.",
    "duplicate_ticker_date": (
        "Multiple source rows share a ticker/date; the first source row is "
        "retained and later rows are listed in the exclusions report."
    ),
    "after_missing_session": (
        "At least one session observed in another source ticker is missing "
        "for this ticker before this row."
    ),
}


def normalize_symbol(value: Any) -> str:
    """Normalize common Yahoo/NSE ticker spellings to the canonical symbol."""
    if pd.isna(value):
        return ""
    symbol = str(value).strip().upper()
    if symbol.startswith("NSE:"):
        symbol = symbol[4:]
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _column_alias(value: Any) -> str:
    normalized = "".join(
        character for character in str(value).strip().lower() if character.isalnum()
    )
    aliases = {
        "date": "date",
        "datetime": "date",
        "symbol": "symbol",
        "ticker": "symbol",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "adj_close",
        "adjustedclose": "adj_close",
        "volume": "volume",
    }
    return aliases.get(normalized, "")


def _canonicalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[Any, str] = {}
    seen: set[str] = set()
    for column in data.columns:
        canonical = _column_alias(column)
        if not canonical:
            continue
        if canonical in seen:
            raise ValueError(
                f"Multiple source columns map to the canonical column {canonical!r}."
            )
        renamed[column] = canonical
        seen.add(canonical)
    return data.rename(columns=renamed).loc[:, list(renamed.values())].copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory(raw_dir: Path, symbols: Sequence[str]) -> list[dict[str, Any]]:
    files = [
        raw_dir / f"{symbol}.{extension}"
        for symbol in symbols
        for extension in ("csv", "parquet")
    ]
    files.extend(
        [raw_dir / "all_stocks.csv", raw_dir / "all_stocks.parquet"]
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Phase 2 source files are missing: " + ", ".join(missing)
        )
    return [
        {
            "filename": path.name,
            "sha256": _sha256(path),
            "role": (
                "per_stock_csv_processing_input"
                if path.suffix.lower() == ".csv" and path.stem in symbols
                else "source_integrity_and_parity_check"
            ),
        }
        for path in files
    ]


def _source_frame(path: Path, expected_symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    canonical = _canonicalize_columns(raw)
    required = set(SOURCE_COLUMNS) - {"symbol"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    flags: list[set[str]] = [set() for _ in range(len(canonical))]
    result = pd.DataFrame(index=range(len(canonical)))
    parsed_dates = pd.to_datetime(canonical["date"], errors="coerce", utc=True)
    result["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()
    for position, parsed_date in enumerate(result["date"]):
        if pd.isna(parsed_date):
            flags[position].add("invalid_date")

    if "symbol" not in canonical:
        normalized_symbols = [expected_symbol] * len(canonical)
        for row_flags in flags:
            row_flags.add("missing_source_symbol")
    else:
        normalized_symbols = [
            normalize_symbol(value) for value in canonical["symbol"]
        ]
        for position, source_symbol in enumerate(normalized_symbols):
            if not source_symbol:
                flags[position].add("missing_source_symbol")
                normalized_symbols[position] = expected_symbol
            elif source_symbol != expected_symbol:
                flags[position].add("source_symbol_mismatch")
                normalized_symbols[position] = expected_symbol
    result["symbol"] = normalized_symbols

    source_columns = (*OHLC_COLUMNS, "adj_close", "volume")
    for column in source_columns:
        if column not in canonical:
            result[column] = np.nan
            for row_flags in flags:
                row_flags.add(
                    "missing_adjusted_close"
                    if column == "adj_close"
                    else f"missing_value:{column}"
                )
            continue
        original = canonical[column]
        numeric = pd.to_numeric(original, errors="coerce").astype("float64")
        result[column] = numeric.to_numpy()
        for position, (original_value, numeric_value) in enumerate(
            zip(original, numeric, strict=True)
        ):
            if pd.isna(original_value):
                flags[position].add(
                    "missing_adjusted_close"
                    if column == "adj_close"
                    else f"missing_value:{column}"
                )
            elif pd.isna(numeric_value):
                flags[position].add(f"non_numeric_value:{column}")
        if column in PRICE_COLUMNS:
            values = numeric.to_numpy()
            invalid_price = numeric.notna().to_numpy() & (
                ~np.isfinite(values) | (values <= 0)
            )
            for position in np.flatnonzero(invalid_price):
                flags[int(position)].add(f"invalid_price:{column}")
        if column == "volume":
            volume_values = numeric.to_numpy()
            for position in np.flatnonzero(
                numeric.notna().to_numpy() & ~np.isfinite(volume_values)
            ):
                flags[int(position)].add("invalid_volume")
            for position in np.flatnonzero(
                numeric.notna().to_numpy() & np.isfinite(volume_values)
            ):
                if volume_values[position] < 0:
                    flags[int(position)].add("negative_volume")
                elif not float(volume_values[position]).is_integer():
                    flags[int(position)].add("non_integer_volume")

    complete_ohlc = result[list(OHLC_COLUMNS)].notna().all(axis=1).to_numpy()
    ohlc = result[list(OHLC_COLUMNS)].to_numpy(dtype=float)
    invalid_ohlc = complete_ohlc & (
        (ohlc[:, 1] < np.maximum(ohlc[:, 0], np.maximum(ohlc[:, 2], ohlc[:, 3])))
        | (ohlc[:, 2] > np.minimum(ohlc[:, 0], np.minimum(ohlc[:, 1], ohlc[:, 3])))
    )
    for position in np.flatnonzero(invalid_ohlc):
        flags[int(position)].add("invalid_ohlc")

    result["source_row_number"] = np.arange(2, len(result) + 2, dtype="int64")
    result["_quality_flags"] = flags
    return result


def _frames_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        first, second = left[column].reset_index(drop=True), right[column].reset_index(
            drop=True
        )
        if column == "date":
            first_dates = pd.to_datetime(first, errors="coerce", utc=True)
            second_dates = pd.to_datetime(second, errors="coerce", utc=True)
            if not first_dates.equals(second_dates):
                return False
        elif column in ("symbol", "quality_flags"):
            if not first.astype("string").fillna("").equals(
                second.astype("string").fillna("")
            ):
                return False
        else:
            first_numeric = pd.to_numeric(first, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            second_numeric = pd.to_numeric(second, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            if not np.isclose(
                first_numeric,
                second_numeric,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ).all():
                return False
    return True


def _add_duplicate_flags_and_exclusions(
    frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    for symbol, data in frames.items():
        excluded_indices: list[int] = []
        valid_dates = data["date"].notna()
        duplicates = data.loc[valid_dates].duplicated(
            subset=["symbol", "date"], keep=False
        )
        duplicate_indices = data.loc[valid_dates].index[duplicates]
        for _, group in data.loc[duplicate_indices].groupby(
            ["symbol", "date"], sort=False, dropna=False
        ):
            ordered = group.sort_values("source_row_number", kind="stable")
            winner = int(ordered.iloc[0]["source_row_number"])
            for index in ordered.index:
                frames[symbol].at[index, "_quality_flags"].add(
                    "duplicate_ticker_date"
                )
            for _, excluded in ordered.iloc[1:].iterrows():
                excluded_indices.append(int(excluded.name))
                exclusions.append(
                    {
                        "symbol": symbol,
                        "date": excluded["date"].strftime("%Y-%m-%d"),
                        "source_row_number": int(excluded["source_row_number"]),
                        "retained_source_row_number": winner,
                        "rule": "retain lowest source_row_number for ticker/date",
                        "reason": "duplicate_ticker_date",
                        "source_quality_flags": sorted(
                            excluded["_quality_flags"]
                        ),
                    }
                )
        frames[symbol] = data.drop(index=excluded_indices).copy()
    return exclusions


def _process_stock(
    data: pd.DataFrame,
    session_positions: dict[pd.Timestamp, int],
    rolling_window: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = data.sort_values(
        ["date", "source_row_number"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    flags: list[set[str]] = [set(row) for row in ordered["_quality_flags"]]
    session_ids = ordered["date"].map(session_positions)
    previous_session_ids = session_ids.shift(1)
    after_gap = (
        session_ids.notna()
        & previous_session_ids.notna()
        & ((session_ids - previous_session_ids) > 1)
    )
    for position in np.flatnonzero(after_gap.to_numpy()):
        flags[int(position)].add("after_missing_session")

    adjusted = ordered["adj_close"].to_numpy(dtype=float)
    dates_valid = ordered["date"].notna().to_numpy()
    previous_adjusted = pd.Series(adjusted).shift(1).to_numpy(dtype=float)
    consecutive_sessions = (
        session_ids.notna()
        & previous_session_ids.notna()
        & ((session_ids - previous_session_ids) == 1)
    ).to_numpy()
    valid_pair = (
        dates_valid
        & np.isfinite(adjusted)
        & (adjusted > 0)
        & np.isfinite(previous_adjusted)
        & (previous_adjusted > 0)
        & consecutive_sessions
    )
    daily_return = np.full(len(ordered), np.nan, dtype=float)
    log_return = np.full(len(ordered), np.nan, dtype=float)
    daily_return[valid_pair] = (
        adjusted[valid_pair] / previous_adjusted[valid_pair]
    ) - 1.0
    log_return[valid_pair] = np.log(
        adjusted[valid_pair] / previous_adjusted[valid_pair]
    )

    return_series = pd.Series(daily_return, index=ordered.index, dtype="float64")
    volatility = return_series.rolling(
        window=rolling_window, min_periods=rolling_window
    ).std(ddof=1)
    warmup = volatility.isna().to_numpy()

    processed = ordered.drop(columns="_quality_flags").copy()
    processed["daily_return"] = daily_return
    processed["log_return"] = log_return
    processed["rolling_volatility_20d"] = volatility.to_numpy()
    processed["rolling_volatility_warmup"] = warmup
    processed["quality_flags"] = [
        json.dumps(sorted(row_flags), separators=(",", ":"))
        for row_flags in flags
    ]
    processed = processed.loc[:, DATA_COLUMNS]

    report = {
        "row_count": int(len(processed)),
        "date_min": (
            processed["date"].min().strftime("%Y-%m-%d")
            if processed["date"].notna().any()
            else None
        ),
        "date_max": (
            processed["date"].max().strftime("%Y-%m-%d")
            if processed["date"].notna().any()
            else None
        ),
        "missing_source_values": {
            column: int(ordered[column].isna().sum())
            for column in (*SOURCE_COLUMNS, "adj_close")
            if column in ordered
        },
        "derived_missing_values": {
            "daily_return": int(pd.isna(daily_return).sum()),
            "log_return": int(pd.isna(log_return).sum()),
            "rolling_volatility_20d": int(volatility.isna().sum()),
        },
        "flagged_rows": int(processed["quality_flags"].ne("[]").sum()),
        "quality_flag_counts": _flag_counts(flags),
        "daily_return_count": int(np.isfinite(daily_return).sum()),
        "log_return_count": int(np.isfinite(log_return).sum()),
        "rolling_volatility_count": int(volatility.notna().sum()),
        "rolling_volatility_warmup_rows": int(warmup.sum()),
        "adjusted_close_available": bool(
            ordered["adj_close"].notna().any()
        ),
        "invalid_ohlc_rows": int(
            sum("invalid_ohlc" in row_flags for row_flags in flags)
        ),
        "date_sorted": bool(
            processed["date"].dropna().is_monotonic_increasing
        ),
    }
    return processed, report


def _flag_counts(flags: Sequence[set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row_flags in flags:
        for flag in row_flags:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def _write_dataset(data: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False, date_format="%Y-%m-%d", float_format="%.17g")
    data.to_parquet(parquet_path, index=False, engine="pyarrow")


def _check_calculations(
    data: pd.DataFrame,
    rolling_window: int,
    session_positions: dict[pd.Timestamp, int],
) -> dict[str, Any]:
    expected_return = np.full(len(data), np.nan, dtype=float)
    expected_log_return = np.full(len(data), np.nan, dtype=float)
    expected_volatility = np.full(len(data), np.nan, dtype=float)
    offset = 0
    for _, group in data.groupby("symbol", sort=False):
        count = len(group)
        adjusted = group["adj_close"].to_numpy(dtype=float)
        group_dates = group["date"]
        session_ids = group_dates.map(session_positions).to_numpy(dtype=float)
        pairs = np.zeros(count, dtype=bool)
        if count > 1:
            adjacent_dates = np.diff(session_ids) == 1
            pairs[1:] = (
                adjacent_dates
                & np.isfinite(adjusted[1:])
                & (adjusted[1:] > 0)
                & np.isfinite(adjusted[:-1])
                & (adjusted[:-1] > 0)
            )
        returns = np.full(count, np.nan, dtype=float)
        logs = np.full(count, np.nan, dtype=float)
        previous = np.arange(count)[pairs] - 1
        returns[pairs] = adjusted[pairs] / adjusted[previous] - 1
        logs[pairs] = np.log(adjusted[pairs] / adjusted[previous])
        volatility = pd.Series(returns).rolling(
            rolling_window, min_periods=rolling_window
        ).std(ddof=1).to_numpy()
        expected_return[offset : offset + count] = returns
        expected_log_return[offset : offset + count] = logs
        expected_volatility[offset : offset + count] = volatility
        offset += count

    def equal_values(column: str, expected: np.ndarray) -> bool:
        actual = pd.to_numeric(data[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        return bool(
            np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True).all()
        )

    return {
        "daily_returns_mathematically_correct": equal_values(
            "daily_return", expected_return
        ),
        "log_returns_mathematically_correct": equal_values(
            "log_return", expected_log_return
        ),
        "rolling_volatility_mathematically_correct": equal_values(
            "rolling_volatility_20d", expected_volatility
        ),
        "rolling_window": rolling_window,
        "rolling_standard_deviation_ddof": 1,
    }


def _missing_sessions(
    frames: dict[str, pd.DataFrame], session_dates: Sequence[pd.Timestamp]
) -> dict[str, list[str]]:
    expected = set(session_dates)
    missing: dict[str, list[str]] = {}
    for symbol, frame in frames.items():
        actual = set(frame.loc[frame["date"].notna(), "date"].unique())
        missing[symbol] = [
            value.strftime("%Y-%m-%d") for value in sorted(expected - actual)
        ]
    return missing


def _input_parity(
    raw_dir: Path,
    symbols: Sequence[str],
    individual_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    parity: dict[str, Any] = {}
    for symbol in symbols:
        csv_frame = pd.read_csv(raw_dir / f"{symbol}.csv")
        parquet_frame = pd.read_parquet(raw_dir / f"{symbol}.parquet")
        parity[symbol] = _frames_equivalent(csv_frame, parquet_frame)

    combined_csv = pd.read_csv(raw_dir / "all_stocks.csv")
    combined_parquet = pd.read_parquet(raw_dir / "all_stocks.parquet")
    concatenated = pd.concat(
        [pd.read_csv(raw_dir / f"{symbol}.csv") for symbol in symbols],
        ignore_index=True,
        sort=False,
    )
    parity["all_stocks_csv_vs_parquet"] = _frames_equivalent(
        combined_csv, combined_parquet
    )
    parity["all_stocks_csv_vs_individual_csvs"] = _frames_equivalent(
        combined_csv, concatenated
    )
    return parity


def process_market_data(
    raw_dir: Path | str = Path("data/raw/nse"),
    output_dir: Path | str = Path("data/processed/silver"),
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    rolling_window: int = 20,
) -> dict[str, Any]:
    """Create Silver files and validation/manifest JSON without editing raw data."""
    raw_path = Path(raw_dir).resolve()
    output_path = Path(output_dir).resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise ValueError("Silver output must not be located inside the raw data directory.")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2.")
    normalized_symbols = tuple(normalize_symbol(symbol) for symbol in symbols)
    if not normalized_symbols or len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("Configure a non-empty, unique set of stock symbols.")

    source_files = _source_inventory(raw_path, normalized_symbols)
    checksums_before = {
        entry["filename"]: entry["sha256"] for entry in source_files
    }
    source_frames = {
        symbol: _source_frame(raw_path / f"{symbol}.csv", symbol)
        for symbol in normalized_symbols
    }
    source_row_counts = {
        symbol: int(len(frame)) for symbol, frame in source_frames.items()
    }
    source_quality_flags = {
        symbol: _flag_counts(frame["_quality_flags"])
        for symbol, frame in source_frames.items()
    }
    exclusions = _add_duplicate_flags_and_exclusions(source_frames)

    observed_dates = sorted(
        {
            value
            for frame in source_frames.values()
            for value in frame["date"].dropna().unique()
        }
    )
    session_positions = {
        pd.Timestamp(value): index for index, value in enumerate(observed_dates)
    }
    missing_sessions = _missing_sessions(source_frames, observed_dates)

    stock_outputs: dict[str, pd.DataFrame] = {}
    stock_reports: dict[str, Any] = {}
    output_files: dict[str, dict[str, str]] = {}
    for symbol in normalized_symbols:
        processed, stock_report = _process_stock(
            source_frames[symbol], session_positions, rolling_window
        )
        stock_outputs[symbol] = processed
        stock_report["source_row_count"] = source_row_counts[symbol]
        stock_report["source_quality_flag_counts"] = source_quality_flags[symbol]
        stock_report["missing_observed_sessions"] = missing_sessions[symbol]
        stock_report["missing_observed_session_count"] = len(missing_sessions[symbol])
        stock_reports[symbol] = stock_report
        csv_path = output_path / f"{symbol}.csv"
        parquet_path = output_path / f"{symbol}.parquet"
        _write_dataset(processed, csv_path, parquet_path)
        output_files[symbol] = {
            "csv": csv_path.name,
            "parquet": parquet_path.name,
        }

    combined = pd.concat(stock_outputs.values(), ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["symbol", "date", "source_row_number"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    combined_csv = output_path / "all_stocks.csv"
    combined_parquet = output_path / "all_stocks.parquet"
    _write_dataset(combined, combined_csv, combined_parquet)
    output_files["all_stocks"] = {
        "csv": combined_csv.name,
        "parquet": combined_parquet.name,
    }

    output_parity: dict[str, bool] = {}
    for name in (*normalized_symbols, "all_stocks"):
        csv_data = pd.read_csv(output_path / f"{name}.csv")
        parquet_data = pd.read_parquet(output_path / f"{name}.parquet")
        output_parity[name] = _frames_equivalent(csv_data, parquet_data)

    input_parity = _input_parity(raw_path, normalized_symbols, source_frames)
    calculation_checks = _check_calculations(
        combined, rolling_window, session_positions
    )
    valid_date_rows = combined["date"].notna()
    duplicate_count = int(
        combined.loc[valid_date_rows].duplicated(["symbol", "date"]).sum()
    )
    sorted_ok = bool(
        combined.sort_values(
            ["symbol", "date", "source_row_number"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)[list(combined.columns)].equals(combined)
    )
    dtype_checks = {
        column: bool(
            pd.api.types.is_numeric_dtype(combined[column])
            if column in (*PRICE_COLUMNS, "volume", "daily_return", "log_return", "rolling_volatility_20d")
            else True
        )
        for column in combined.columns
    }
    all_row_flags = [
        row_flags
        for stock_frame in stock_outputs.values()
        for row_flags in stock_frame["quality_flags"]
    ]
    source_missing_counts = {
        symbol: report["missing_source_values"]
        for symbol, report in stock_reports.items()
    }
    quality_checks = {
        "no_missing_source_values": all(
            count == 0
            for counts in source_missing_counts.values()
            for count in counts.values()
        ),
        "no_invalid_price_values": not any(
            flag.startswith("invalid_price:")
            for encoded in all_row_flags
            for flag in json.loads(encoded)
        ),
        "no_invalid_ohlc_relationships": not any(
            "invalid_ohlc" in json.loads(encoded) for encoded in all_row_flags
        ),
        "no_invalid_volume_values": not any(
            flag in {"negative_volume", "invalid_volume", "non_integer_volume"}
            for encoded in all_row_flags
            for flag in json.loads(encoded)
        ),
        "no_missing_observed_sessions": not any(missing_sessions.values()),
    }
    report_timestamp = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "generated_at": report_timestamp,
        "source": "Existing Phase 2 local CSV and Parquet files; no downloads",
        "date_range": {
            "min": (
                combined["date"].min().strftime("%Y-%m-%d")
                if combined["date"].notna().any()
                else None
            ),
            "max": (
                combined["date"].max().strftime("%Y-%m-%d")
                if combined["date"].notna().any()
                else None
            ),
            "observed_session_calendar": (
                "Union of valid dates observed in the configured source tickers; "
                "weekends and exchange holidays are not inferred."
            ),
        },
        "symbols": list(normalized_symbols),
        "stock_reports": stock_reports,
        "row_count": int(len(combined)),
        "row_counts_by_symbol": {
            symbol: int((combined["symbol"] == symbol).sum())
            for symbol in normalized_symbols
        },
        "missing_observed_sessions_by_symbol": missing_sessions,
        "exclusions": {
            "rule": (
                "For duplicate normalized ticker/date keys, retain the first "
                "source row by source_row_number; list every later source row here."
            ),
            "count": len(exclusions),
            "records": exclusions,
        },
        "row_quality_flags": [
            {
                "symbol": str(row.symbol),
                "date": row.date.strftime("%Y-%m-%d") if pd.notna(row.date) else None,
                "source_row_number": int(row.source_row_number),
                "flags": json.loads(row.quality_flags),
            }
            for row in combined.itertuples(index=False)
            if json.loads(row.quality_flags)
        ],
        "flag_explanations": FLAG_EXPLANATIONS,
        "checks": {
            "all_configured_symbols_represented": set(combined["symbol"].unique())
            == set(normalized_symbols),
            "no_duplicate_ticker_date_keys": duplicate_count == 0,
            "date_ordering": sorted_ok,
            "numeric_dtypes": dtype_checks,
            "source_csv_parquet_parity": input_parity,
            "silver_csv_parquet_parity": output_parity,
            "calculations": calculation_checks,
            "per_stock_rows_reconcile_to_combined": sum(
                stock_reports[symbol]["row_count"] for symbol in normalized_symbols
            )
            == len(combined),
            "data_quality": quality_checks,
        },
        "methodology": {
            "adjusted_close_source_column": "adj_close",
            "adjusted_close_present_in_sources": {
                symbol: any(
                    _column_alias(column) == "adj_close"
                    for column in pd.read_csv(
                        raw_path / f"{symbol}.csv", nrows=0
                    ).columns
                )
                for symbol in normalized_symbols
            },
            "return_formula": "adj_close[t] / adj_close[t-1] - 1",
            "log_return_formula": "ln(adj_close[t] / adj_close[t-1])",
            "return_requires_consecutive_observed_sessions": True,
            "no_price_or_return_filling": True,
            "rolling_volatility": (
                f"Trailing {rolling_window} consecutive row returns; sample "
                "standard deviation (ddof=1); no annualization."
            ),
            "rolling_warmup": (
                "rolling_volatility_warmup is true until a full window of valid "
                "returns is available."
            ),
            "missing_session_detection": (
                "Compare each symbol with the union of dates observed across "
                "the configured source symbols. Missing rows are reported, not "
                "synthesized; a session omitted by every source cannot be inferred."
            ),
            "corporate_action_caveat": (
                "Adjusted close is used as supplied by the source. This dataset "
                "does not independently establish a complete corporate-action "
                "history or guarantee corporate-action correctness."
            ),
        },
        "output_files": output_files,
    }
    check_values = [
        value
        for key, value in report["checks"].items()
        if key not in ("numeric_dtypes", "source_csv_parquet_parity", "silver_csv_parquet_parity", "calculations")
    ]
    check_values.extend(report["checks"]["numeric_dtypes"].values())
    check_values.extend(report["checks"]["source_csv_parquet_parity"].values())
    check_values.extend(report["checks"]["silver_csv_parquet_parity"].values())
    check_values.extend(
        value
        for key, value in report["checks"]["calculations"].items()
        if isinstance(value, bool)
    )
    check_values.extend(quality_checks.values())
    report["valid"] = (
        all(check_values)
        and not exclusions
        and not any(missing_sessions.values())
        and all(stock["flagged_rows"] == 0 for stock in stock_reports.values())
    )

    report_path = output_path / "silver_validation_report.json"
    output_files["silver_validation_report"] = {"json": report_path.name}
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    checksums_after = {
        entry["filename"]: _sha256(raw_path / entry["filename"])
        for entry in source_files
    }
    if checksums_after != checksums_before:
        raise RuntimeError("Raw source checksums changed during Silver processing.")

    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("pandas", "numpy", "pyarrow", "yfinance", "pytest")
    }
    manifest_path = output_path / "silver_dataset_manifest.json"
    output_checksums = {
        path.name: _sha256(path)
        for path in sorted(output_path.iterdir())
        if path.is_file() and path.name != "silver_dataset_manifest.json"
    }
    manifest: dict[str, Any] = {
        "processing_timestamp_utc": report_timestamp,
        "source_files": source_files,
        "source_checksums_unchanged_after_processing": True,
        "date_range": report["date_range"],
        "stock_symbols": list(normalized_symbols),
        "row_counts": report["row_counts_by_symbol"],
        "combined_row_count": len(combined),
        "output_filenames": {
            **output_files,
            "silver_dataset_manifest": {"json": manifest_path.name},
        },
        "output_sha256": output_checksums,
        "runtime_versions": {
            "python": platform.python_version(),
            "packages": package_versions,
        },
        "processing_configuration": {
            "rolling_window": rolling_window,
            "rolling_std_ddof": 1,
            "duplicate_rule": (
                "Keep earliest source_row_number per normalized symbol/date; "
                "record later rows in exclusions."
            ),
            "symbol_normalization": "uppercase; strip NSE: prefix and .NS suffix",
            "adjusted_close_column": "adj_close; never substitute close",
            "missing_session_calendar": "union of observed source dates",
            "fill_missing_values": False,
            "forward_fill_prices_or_returns": False,
            "excluded_source_rows": len(exclusions),
        },
        "manifest_checksum_note": (
            "This manifest does not include its own checksum. Output checksums "
            "cover all generated data files and silver_validation_report.json."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if not report["valid"]:
        LOGGER.warning(
            "Silver processing completed with validation findings; inspect %s",
            report_path,
        )
    return {"report": report, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local FinCloud Silver market-data layer."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nse"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/silver")
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    args = parser.parse_args()
    result = process_market_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        rolling_window=args.rolling_window,
    )
    print(f"Silver validation report: {args.output_dir / 'silver_validation_report.json'}")
    print(f"Silver manifest: {args.output_dir / 'silver_dataset_manifest.json'}")
    print(f"Overall validation: {'passed' if result['report']['valid'] else 'issues found'}")
    return 0 if result["report"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
