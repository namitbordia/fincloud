"""Download and store historical daily market data from configurable providers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.ingestion.config import DEFAULT_SYMBOLS, IngestionConfig
from src.ingestion.validators import validate_stock_data

LOGGER = logging.getLogger(__name__)
BASE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
YAHOO_COLUMN_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "adj_close",
    "adjustedclose": "adj_close",
    "volume": "volume",
}


class MarketDataProvider(Protocol):
    """Interface implemented by historical market-data providers."""

    def fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Return provider data for the requested ticker and date range."""


class YahooFinanceProvider:
    """Supplementary historical data adapter for Yahoo Finance via yfinance."""

    def __init__(self) -> None:
        try:
            import yfinance
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for downloads. Install project dependencies "
                "with: python -m pip install -r requirements.txt"
            ) from exc
        self._yfinance = yfinance

    def fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        self._yfinance.config.debug.hide_exceptions = False
        history = self._yfinance.Ticker(ticker).history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        return history


def _column_name(column: Any) -> str | None:
    candidates = column if isinstance(column, tuple) else (column,)
    for candidate in candidates:
        normalized = str(candidate).strip().lower().replace(" ", "").replace("_", "")
        if normalized in YAHOO_COLUMN_NAMES:
            return YAHOO_COLUMN_NAMES[normalized]
    return None


def normalize_provider_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Create the common output schema on a copy, retaining source rows as-is."""
    raw = data.copy(deep=True)
    if isinstance(raw.columns, pd.MultiIndex):
        normalized = {
            field: raw.iloc[:, position]
            for position, column in enumerate(raw.columns)
            if (field := _column_name(column)) is not None
        }
    else:
        normalized = {
            field: raw[column]
            for column in raw.columns
            if (field := _column_name(column)) is not None
        }

    date_column = next(
        (
            column
            for column in raw.columns
            if str(column).strip().lower() in {"date", "datetime"}
        ),
        None,
    )
    date_values = raw[date_column] if date_column is not None else raw.index
    parsed_dates = pd.to_datetime(date_values, errors="coerce")
    result = pd.DataFrame(index=raw.index)
    result["date"] = pd.Series(parsed_dates, index=raw.index).dt.strftime(
        "%Y-%m-%d"
    )
    result["symbol"] = symbol
    for column in ("open", "high", "low", "close"):
        if column in normalized:
            result[column] = normalized[column].to_numpy()
    if "adj_close" in normalized:
        result["adj_close"] = normalized["adj_close"].to_numpy()
    if "volume" in normalized:
        result["volume"] = normalized["volume"].to_numpy()

    columns = [column for column in BASE_COLUMNS if column in result]
    if "adj_close" in result:
        columns.insert(columns.index("volume"), "adj_close")
    return result.loc[:, columns].reset_index(drop=True)


def _empty_stock_data(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(columns=(*BASE_COLUMNS[:1], "symbol", *BASE_COLUMNS[2:])).assign(
        symbol=symbol
    )


def _save_frame(
    data: pd.DataFrame,
    path: Path,
    write_parquet: bool,
) -> dict[str, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    parquet_path: Path | None = None
    if write_parquet and importlib.util.find_spec("pyarrow") is not None:
        parquet_path = path.with_suffix(".parquet")
        data.to_parquet(parquet_path, index=False, engine="pyarrow")
    return {
        "csv": str(path),
        "parquet": str(parquet_path) if parquet_path is not None else None,
    }


def run_pipeline(
    config: IngestionConfig,
    provider: MarketDataProvider | None = None,
) -> dict[str, Any]:
    """Fetch configured symbols, write raw datasets, and save validation results."""
    market_data_provider = provider or YahooFinanceProvider()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    stock_data: dict[str, pd.DataFrame] = {}
    download_errors: dict[str, str] = {}
    output_files: dict[str, dict[str, str | None]] = {}

    for symbol, ticker in config.symbols.items():
        for attempt in range(1, config.max_attempts + 1):
            try:
                downloaded = market_data_provider.fetch(
                    ticker, config.start_date, config.end_date
                )
                normalized = normalize_provider_data(downloaded, symbol)
            except Exception as exc:
                LOGGER.warning(
                    "Download attempt %s/%s failed for %s (%s): %s",
                    attempt,
                    config.max_attempts,
                    symbol,
                    ticker,
                    exc,
                )
                if attempt == config.max_attempts:
                    download_errors[symbol] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    time.sleep(config.retry_delay_seconds)
            else:
                stock_data[symbol] = normalized
                break

        if symbol in stock_data:
            output_files[symbol] = _save_frame(
                stock_data[symbol],
                config.output_dir / f"{symbol}.csv",
                config.write_parquet,
            )

    if stock_data:
        combined = pd.concat(stock_data.values(), ignore_index=True, sort=False)
    else:
        combined = _empty_stock_data("")
    combined_path = config.output_dir / "all_stocks.csv"
    output_files["all_stocks"] = _save_frame(
        combined, combined_path, config.write_parquet
    )

    dataset_reports = {
        symbol: validate_stock_data(
            stock_data.get(symbol, _empty_stock_data(symbol)), symbol
        )
        for symbol in config.symbols
    }
    combined_report = validate_stock_data(combined)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "date_range": {
            "start_inclusive": config.start_date,
            "end_exclusive": config.end_date,
        },
        "datasets": dataset_reports,
        "combined": combined_report,
        "download_errors": download_errors,
        "output_files": output_files,
        "parquet_available": importlib.util.find_spec("pyarrow") is not None,
    }
    report["valid"] = (
        not download_errors
        and all(dataset["valid"] for dataset in dataset_reports.values())
        and combined_report["valid"]
    )
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical daily stock data into local CSV files."
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        required=True,
        help="End date, exclusive (YYYY-MM-DD), matching yfinance semantics.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=tuple(DEFAULT_SYMBOLS),
        default=list(DEFAULT_SYMBOLS),
        help="Canonical NSE symbols to download (default: all configured symbols).",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/nse"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/validation_report.json"),
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Write CSV only even when pyarrow is installed.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    config = IngestionConfig(
        start_date=args.start,
        end_date=args.end,
        symbols={symbol: DEFAULT_SYMBOLS[symbol] for symbol in args.symbols},
        output_dir=args.output_dir,
        report_path=args.report,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay,
        write_parquet=not args.no_parquet,
    )
    report = run_pipeline(config)
    print(f"Validation report: {config.report_path}")
    print(f"Overall validation: {'passed' if report['valid'] else 'issues found'}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
