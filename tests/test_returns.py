"""Tests for return calculations and return-input validation."""

import unittest

import numpy as np
import pandas as pd

from src.analytics.returns import (
    calculate_returns,
    risk_free_daily_rate,
    validate_return_frame,
)


class TestReturns(unittest.TestCase):
    def test_simple_returns_do_not_fill_missing_prices(self) -> None:
        prices = pd.Series([100.0, 110.0, np.nan, 121.0])

        result = calculate_returns(prices)

        np.testing.assert_allclose(
            result.to_numpy(),
            [np.nan, 0.1, np.nan, np.nan],
            equal_nan=True,
        )

    def test_validation_reports_missing_duplicate_and_invalid_returns(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
                "symbol": ["AAA"] * 4,
                "daily_return": [0.01, 0.02, np.nan, -1.2],
            }
        )

        result = validate_return_frame(frame)

        self.assertEqual(len(result.returns), 0)
        reasons = [item["reason"] for item in result.exclusions]
        self.assertEqual(reasons.count("duplicate_symbol_date"), 2)
        self.assertIn("missing_return", reasons)
        self.assertIn("return_below_minus_one", reasons)

    def test_risk_free_rate_conversion_and_validation(self) -> None:
        daily = risk_free_daily_rate(0.05, 252)

        self.assertAlmostEqual((1 + daily) ** 252 - 1, 0.05)
        with self.assertRaises(ValueError):
            risk_free_daily_rate(-1.0)


if __name__ == "__main__":
    unittest.main()
