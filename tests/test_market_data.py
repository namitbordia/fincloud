"""Offline tests for provider-independent market data ingestion."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.ingestion.config import IngestionConfig
from src.ingestion.market_data import (
    YahooFinanceProvider,
    normalize_provider_data,
    run_pipeline,
)


def provider_frame(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0 + offset, 101.0 + offset],
            "High": [105.0 + offset, 106.0 + offset],
            "Low": [99.0 + offset, 100.0 + offset],
            "Close": [103.0 + offset, 104.0 + offset],
            "Adj Close": [102.0 + offset, 103.0 + offset],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        self.calls.append(ticker)
        return provider_frame(offset=float(len(self.calls)))


class FailingThenSuccessfulProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("temporary network issue")
        return provider_frame()


class PartiallyFailingProvider(FakeProvider):
    def fetch(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        self.calls.append(ticker)
        if ticker == "RELIANCE.NS":
            raise TimeoutError("provider timed out")
        return provider_frame()


class TestMarketDataPipeline(unittest.TestCase):
    def test_yahoo_provider_preserves_error_behavior_without_deprecated_argument(
        self,
    ) -> None:
        call: dict[str, object] = {}

        class FakeTicker:
            def __init__(self, ticker: str) -> None:
                call["ticker"] = ticker

            def history(self, **kwargs: object) -> pd.DataFrame:
                call["kwargs"] = kwargs
                raise RuntimeError("provider error")

        fake_yfinance = SimpleNamespace(
            config=SimpleNamespace(
                debug=SimpleNamespace(hide_exceptions=True)
            ),
            Ticker=FakeTicker,
        )
        with patch.dict("sys.modules", {"yfinance": fake_yfinance}):
            provider = YahooFinanceProvider()
            with self.assertRaisesRegex(RuntimeError, "provider error"):
                provider.fetch("RELIANCE.NS", "2024-01-01", "2024-02-01")

        self.assertFalse(fake_yfinance.config.debug.hide_exceptions)
        self.assertEqual(call["ticker"], "RELIANCE.NS")
        self.assertEqual(
            call["kwargs"],
            {
                "start": "2024-01-01",
                "end": "2024-02-01",
                "interval": "1d",
                "auto_adjust": False,
                "actions": False,
            },
        )

    def test_normalization_keeps_rows_and_uses_consistent_columns(self) -> None:
        normalized = normalize_provider_data(provider_frame(), "RELIANCE")

        self.assertEqual(
            normalized.columns.tolist(),
            [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ],
        )
        self.assertEqual(normalized["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(len(normalized), 2)

    def test_normalization_accepts_a_date_column(self) -> None:
        frame = provider_frame().reset_index(names="Date")

        normalized = normalize_provider_data(frame, "TCS")

        self.assertEqual(normalized["date"].tolist(), ["2024-01-02", "2024-01-03"])

    def test_multiple_symbols_write_individual_and_combined_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provider = FakeProvider()
            config = IngestionConfig(
                start_date="2024-01-01",
                end_date="2024-02-01",
                symbols={"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS"},
                output_dir=root / "data" / "raw" / "nse",
                report_path=root / "data" / "processed" / "validation_report.json",
                retry_delay_seconds=0,
            )

            report = run_pipeline(config, provider=provider)

            combined = pd.read_csv(config.output_dir / "all_stocks.csv")
            self.assertTrue(report["valid"])
            self.assertEqual(provider.calls, ["RELIANCE.NS", "TCS.NS"])
            self.assertTrue((config.output_dir / "RELIANCE.csv").exists())
            self.assertTrue((config.output_dir / "TCS.csv").exists())
            if report["parquet_available"]:
                self.assertTrue((config.output_dir / "RELIANCE.parquet").exists())
            else:
                self.assertFalse((config.output_dir / "RELIANCE.parquet").exists())
            self.assertEqual(combined["symbol"].value_counts().to_dict(), {
                "RELIANCE": 2,
                "TCS": 2,
            })
            self.assertTrue(config.report_path.exists())

    def test_retry_succeeds_after_transient_provider_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provider = FailingThenSuccessfulProvider()
            config = IngestionConfig(
                start_date="2024-01-01",
                end_date="2024-02-01",
                symbols={"INFY": "INFY.NS"},
                output_dir=root / "raw",
                report_path=root / "processed" / "validation_report.json",
                max_attempts=2,
                retry_delay_seconds=0,
                write_parquet=False,
            )

            report = run_pipeline(config, provider=provider)

            self.assertTrue(report["valid"])
            self.assertEqual(provider.attempts, 2)
            self.assertEqual(report["download_errors"], {})

    def test_persistent_download_error_is_reported_and_other_symbols_continue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provider = PartiallyFailingProvider()
            config = IngestionConfig(
                start_date="2024-01-01",
                end_date="2024-02-01",
                symbols={"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS"},
                output_dir=root / "raw",
                report_path=root / "processed" / "validation_report.json",
                max_attempts=2,
                retry_delay_seconds=0,
                write_parquet=False,
            )

            report = run_pipeline(config, provider=provider)

            self.assertFalse(report["valid"])
            self.assertIn("RELIANCE", report["download_errors"])
            self.assertEqual(provider.calls, [
                "RELIANCE.NS",
                "RELIANCE.NS",
                "TCS.NS",
            ])
            self.assertFalse((config.output_dir / "RELIANCE.csv").exists())
            self.assertTrue((config.output_dir / "TCS.csv").exists())
            combined = pd.read_csv(config.output_dir / "all_stocks.csv")
            self.assertEqual(combined["symbol"].unique().tolist(), ["TCS"])


if __name__ == "__main__":
    unittest.main()
