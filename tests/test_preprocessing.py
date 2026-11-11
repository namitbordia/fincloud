"""Offline tests for the Silver market-data preprocessing pipeline."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import (
    _frames_equivalent,
    _process_stock,
    _source_frame,
    process_market_data,
)


def synthetic_frame(
    symbol: str,
    dates: pd.DatetimeIndex,
    adjusted_prices: np.ndarray | None = None,
) -> pd.DataFrame:
    if adjusted_prices is None:
        adjusted_prices = 100.0 + np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": f"{symbol}.NS",
            "open": adjusted_prices - 0.5,
            "high": adjusted_prices + 1.0,
            "low": adjusted_prices - 1.0,
            "close": adjusted_prices + 0.25,
            "adj_close": adjusted_prices,
            "volume": np.full(len(dates), 1000, dtype=np.int64),
        }
    )


def write_raw_sources(raw_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol, data in frames.items():
        data.to_csv(raw_dir / f"{symbol}.csv", index=False)
        data.to_parquet(raw_dir / f"{symbol}.parquet", index=False)
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(raw_dir / "all_stocks.csv", index=False)
    combined.to_parquet(raw_dir / "all_stocks.parquet", index=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSilverPreprocessing(unittest.TestCase):
    def test_adjusted_price_returns_log_returns_and_rolling_volatility(self) -> None:
        dates = pd.date_range("2024-01-01", periods=24, freq="D")
        returns = np.array(
            [np.nan] + [0.01 if index % 2 else -0.005 for index in range(1, 24)]
        )
        prices = np.empty(len(dates), dtype=float)
        prices[0] = 100.0
        for index in range(1, len(prices)):
            prices[index] = prices[index - 1] * (1.0 + returns[index])
        frame = synthetic_frame("AAA", dates, prices)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "AAA.csv"
            frame.to_csv(path, index=False)
            source = _source_frame(path, "AAA")
            processed, report = _process_stock(
                source, {date: index for index, date in enumerate(dates)}
            )

        np.testing.assert_allclose(
            processed["daily_return"].to_numpy()[1:],
            prices[1:] / prices[:-1] - 1.0,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            processed["log_return"].to_numpy()[1:],
            np.log(prices[1:] / prices[:-1]),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(np.isnan(processed.loc[0, "daily_return"]))
        self.assertTrue(processed["rolling_volatility_warmup"].iloc[:20].all())
        expected_volatility = pd.Series(
            processed["daily_return"]
        ).rolling(20, min_periods=20).std(ddof=1)
        np.testing.assert_allclose(
            processed["rolling_volatility_20d"],
            expected_volatility,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        self.assertEqual(report["rolling_volatility_count"], 4)

    def test_missing_values_invalid_ohlc_and_unsorted_dates_are_flagged(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = synthetic_frame("AAA", dates)
        frame.loc[1, "adj_close"] = np.nan
        frame.loc[2, "high"] = frame.loc[2, "low"] - 2
        frame = frame.iloc[::-1].reset_index(drop=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "AAA.csv"
            frame.to_csv(path, index=False)
            source = _source_frame(path, "AAA")
            processed, report = _process_stock(
                source, {date: index for index, date in enumerate(dates)}
            )

        self.assertTrue(processed["date"].is_monotonic_increasing)
        missing_row = processed.loc[processed["date"] == dates[1]].iloc[0]
        self.assertTrue(np.isnan(missing_row["adj_close"]))
        self.assertTrue(np.isnan(missing_row["daily_return"]))
        self.assertIn("missing_adjusted_close", missing_row["quality_flags"])
        invalid_row = processed.loc[processed["date"] == dates[2]].iloc[0]
        self.assertIn("invalid_ohlc", invalid_row["quality_flags"])
        self.assertEqual(report["missing_source_values"]["adj_close"], 1)
        self.assertEqual(report["invalid_ohlc_rows"], 1)

    def test_close_is_never_used_as_adjusted_close_fallback(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        frame = synthetic_frame("AAA", dates).drop(columns="adj_close")

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "AAA.csv"
            frame.to_csv(path, index=False)
            source = _source_frame(path, "AAA")
            processed, report = _process_stock(
                source, {date: index for index, date in enumerate(dates)}
            )

        self.assertTrue(processed["close"].notna().all())
        self.assertTrue(processed["adj_close"].isna().all())
        self.assertTrue(processed["daily_return"].isna().all())
        self.assertFalse(report["adjusted_close_available"])

    def test_duplicate_rows_are_excluded_and_missing_sessions_are_reported(self) -> None:
        dates = pd.date_range("2024-01-01", periods=22, freq="D")
        aaa = synthetic_frame("AAA", dates)
        duplicate = aaa.iloc[[4]].copy()
        duplicate["adj_close"] = 999.0
        aaa = pd.concat([aaa, duplicate], ignore_index=True)
        bbb = synthetic_frame("BBB", dates.delete(10))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            write_raw_sources(raw_dir, {"AAA": aaa, "BBB": bbb})
            result = process_market_data(
                raw_dir,
                root / "silver",
                symbols=("AAA", "BBB"),
            )
            silver = pd.read_csv(root / "silver" / "all_stocks.csv")

        report = result["report"]
        self.assertEqual(len(result["report"]["exclusions"]["records"]), 1)
        self.assertEqual(report["row_counts_by_symbol"], {"AAA": 22, "BBB": 21})
        self.assertEqual(len(silver), 43)
        self.assertFalse(silver.duplicated(["symbol", "date"]).any())
        self.assertEqual(report["missing_observed_sessions_by_symbol"]["BBB"], [
            dates[10].strftime("%Y-%m-%d")
        ])
        bbb_after_gap = silver.loc[
            (silver["symbol"] == "BBB")
            & (silver["date"] == dates[11].strftime("%Y-%m-%d"))
        ].iloc[0]
        self.assertIn("after_missing_session", bbb_after_gap["quality_flags"])
        self.assertTrue(np.isnan(bbb_after_gap["daily_return"]))
        retained_duplicate = silver.loc[
            (silver["symbol"] == "AAA")
            & (silver["date"] == dates[4].strftime("%Y-%m-%d"))
        ].iloc[0]
        self.assertEqual(retained_duplicate["adj_close"], 104.0)
        self.assertIn("duplicate_ticker_date", retained_duplicate["quality_flags"])
        self.assertFalse(report["valid"])

    def test_multiple_tickers_outputs_manifest_parity_and_raw_immutability(self) -> None:
        dates = pd.date_range("2024-02-01", periods=24, freq="D")
        frames = {
            "AAA": synthetic_frame("AAA", dates),
            "BBB": synthetic_frame("BBB", dates, 200 + np.arange(len(dates))),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            output_dir = root / "processed" / "silver"
            write_raw_sources(raw_dir, frames)
            source_hashes_before = {
                path.name: sha256(path) for path in raw_dir.iterdir()
            }

            result = process_market_data(
                raw_dir, output_dir, symbols=("AAA", "BBB")
            )

            source_hashes_after = {
                path.name: sha256(path) for path in raw_dir.iterdir()
            }
            manifest_path = output_dir / "silver_dataset_manifest.json"
            validation_path = output_dir / "silver_validation_report.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(validation_path.read_text(encoding="utf-8"))
            combined_csv = pd.read_csv(output_dir / "all_stocks.csv")
            combined_parquet = pd.read_parquet(output_dir / "all_stocks.parquet")

            self.assertEqual(source_hashes_before, source_hashes_after)
            self.assertTrue(manifest["source_checksums_unchanged_after_processing"])
            self.assertEqual(manifest["row_counts"], {"AAA": 24, "BBB": 24})
            self.assertEqual(manifest["combined_row_count"], 48)
            self.assertEqual(len(manifest["source_files"]), 6)
            self.assertTrue(report["valid"], json.dumps(report, indent=2))
            self.assertTrue(
                all(report["checks"]["silver_csv_parquet_parity"].values())
            )
            self.assertTrue(_frames_equivalent(combined_csv, combined_parquet))
            for name, checksum in manifest["output_sha256"].items():
                self.assertEqual(sha256(output_dir / name), checksum)


if __name__ == "__main__":
    unittest.main()
