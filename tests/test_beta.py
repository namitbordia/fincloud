"""Tests for benchmark validation and beta calculation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.beta import calculate_beta, load_benchmark_returns


class TestBeta(unittest.TestCase):
    def test_beta_matches_synthetic_benchmark(self) -> None:
        benchmark = pd.Series(
            [0.01, -0.02, 0.03, 0.00],
            index=pd.date_range("2024-01-01", periods=4),
        )
        asset = 2.0 * benchmark

        result = calculate_beta(asset, benchmark)

        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["beta"], 2.0)

    def test_beta_unavailable_without_benchmark_or_with_zero_variance(self) -> None:
        asset = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))

        self.assertFalse(calculate_beta(asset, None)["available"])
        result = calculate_beta(asset, pd.Series([0.01, 0.01], index=asset.index))
        self.assertFalse(result["available"])
        self.assertIn("variance", result["unavailable_reason"])

    def test_benchmark_csv_duplicate_and_missing_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "benchmark.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                    "return": [0.01, 0.02, np.nan],
                }
            ).to_csv(path, index=False)
            returns, exclusions = load_benchmark_returns(str(path))

        self.assertEqual(len(returns), 0)
        self.assertEqual(len(exclusions), 3)


if __name__ == "__main__":
    unittest.main()
