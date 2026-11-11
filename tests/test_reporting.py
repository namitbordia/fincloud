"""Offline tests for analytics output generation and input immutability."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.reporting import run_analytics


def silver_fixture() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    frames = []
    for symbol, scale in (("AAA", 1.0), ("BBB", 1.5)):
        adjusted = 100.0 * np.cumprod(
            1.0 + scale * np.resize([0.01, -0.005, 0.002], len(dates))
        )
        returns = np.concatenate(([np.nan], adjusted[1:] / adjusted[:-1] - 1))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": adjusted,
                    "high": adjusted * 1.01,
                    "low": adjusted * 0.99,
                    "close": adjusted,
                    "adj_close": adjusted,
                    "volume": 1000,
                    "daily_return": returns,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestReporting(unittest.TestCase):
    def test_outputs_are_generated_reproducibly_without_modifying_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "silver"
            output_dir = root / "analytics"
            input_dir.mkdir()
            source = input_dir / "all_stocks.csv"
            silver_fixture().to_csv(source, index=False)
            before = checksum(source)

            first = run_analytics(
                input_dir,
                output_dir,
                symbols=("AAA", "BBB"),
                weights={"AAA": 0.6, "BBB": 0.4},
            )
            metrics_first = (output_dir / "stock_performance_metrics.csv").read_bytes()
            second = run_analytics(
                input_dir,
                output_dir,
                symbols=("AAA", "BBB"),
                weights={"AAA": 0.6, "BBB": 0.4},
            )

            expected_outputs = {
                "stock_performance_metrics.csv",
                "portfolio_daily_returns.csv",
                "portfolio_performance_metrics.json",
                "correlation_matrix.csv",
                "portfolio_drawdown.csv",
                "risk_metrics.csv",
                "analytics_validation_report.json",
                "analytics_manifest.json",
            }
            self.assertTrue(expected_outputs.issubset({p.name for p in output_dir.iterdir()}))
            self.assertTrue(first["report"]["valid"], json.dumps(first["report"]))
            self.assertTrue(second["report"]["valid"])
            self.assertEqual(
                first["report"]["excluded_portfolio_dates"]["count"], 1
            )
            self.assertEqual(
                first["report"]["excluded_portfolio_dates"]["records"][0][
                    "date"
                ],
                "2024-01-01",
            )
            self.assertEqual(metrics_first, (output_dir / "stock_performance_metrics.csv").read_bytes())
            self.assertEqual(before, checksum(source))
            self.assertEqual(first["manifest"]["configuration"]["portfolio_weights"], {
                "AAA": 0.6,
                "BBB": 0.4,
            })

    def test_missing_benchmark_yields_unavailable_beta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "silver"
            input_dir.mkdir()
            silver_fixture().to_csv(input_dir / "all_stocks.csv", index=False)

            result = run_analytics(
                input_dir,
                root / "analytics",
                symbols=("AAA", "BBB"),
            )

        self.assertFalse(result["report"]["benchmark"]["beta_available"])
        self.assertIn("unavailable_reason", result["report"]["benchmark"])

    def test_supplied_benchmark_is_aligned_and_included_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "silver"
            input_dir.mkdir()
            source = silver_fixture()
            source.to_csv(input_dir / "all_stocks.csv", index=False)
            benchmark_path = root / "benchmark.csv"
            source.loc[
                (source["symbol"] == "AAA")
                & source["daily_return"].notna(),
                ["date", "daily_return"],
            ].rename(columns={"daily_return": "return"}).to_csv(
                benchmark_path, index=False
            )

            result = run_analytics(
                input_dir,
                root / "analytics",
                symbols=("AAA", "BBB"),
                benchmark_csv=benchmark_path,
            )

        self.assertTrue(result["report"]["benchmark"]["beta_available"])
        self.assertAlmostEqual(
            result["report"]["benchmark"]["asset_beta"]["AAA"]["beta"], 1.0
        )
        self.assertIn(
            str(benchmark_path.resolve()),
            [entry["filename"] for entry in result["manifest"]["input_files"]],
        )


if __name__ == "__main__":
    unittest.main()
