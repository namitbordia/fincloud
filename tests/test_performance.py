"""Tests for stock and portfolio performance statistics."""

import unittest

import numpy as np
import pandas as pd

from src.analytics.performance import calculate_performance_metrics


class TestPerformance(unittest.TestCase):
    def test_metrics_use_elapsed_calendar_time_and_sample_volatility(self) -> None:
        returns = np.array([0.01, -0.005, 0.02, 0.0])
        dates = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        )
        metrics = calculate_performance_metrics(
            returns, dates, annual_risk_free_rate=0.0
        )

        total = np.prod(1 + returns) - 1
        self.assertAlmostEqual(metrics["total_return"], total)
        self.assertAlmostEqual(
            metrics["cagr"], (1 + total) ** (365.25 / 3) - 1
        )
        self.assertAlmostEqual(
            metrics["annualized_volatility"], np.std(returns, ddof=1) * np.sqrt(252)
        )
        self.assertEqual(metrics["observation_count"], 4)

    def test_zero_volatility_and_insufficient_observations_are_unavailable(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        constant = calculate_performance_metrics([0.01, 0.01], dates)

        self.assertTrue(np.isnan(constant["sharpe_ratio"]))
        self.assertEqual(constant["annualized_volatility"], 0.0)
        single = calculate_performance_metrics([0.01], dates[:1])
        self.assertTrue(np.isnan(single["cagr"]))
        self.assertTrue(np.isnan(single["maximum_drawdown"]))


if __name__ == "__main__":
    unittest.main()
