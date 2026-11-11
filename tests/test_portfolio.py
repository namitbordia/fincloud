"""Tests for portfolio weight validation and daily rebalancing."""

import unittest

import numpy as np
import pandas as pd

from src.analytics.portfolio import (
    calculate_portfolio_returns,
    equal_weights,
    validate_weights,
)


class TestPortfolio(unittest.TestCase):
    def test_equal_and_custom_weights(self) -> None:
        symbols = ["AAA", "BBB"]
        self.assertEqual(equal_weights(symbols), {"AAA": 0.5, "BBB": 0.5})
        self.assertEqual(
            validate_weights({"AAA": 0.7, "BBB": 0.3}, symbols),
            {"AAA": 0.7, "BBB": 0.3},
        )

    def test_invalid_missing_negative_or_non_unit_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({"AAA": 1.0}, ["AAA", "BBB"])
        with self.assertRaises(ValueError):
            validate_weights({"AAA": 1.1, "BBB": -0.1}, ["AAA", "BBB"])
        with self.assertRaises(ValueError):
            validate_weights({"AAA": 0.6, "BBB": 0.3}, ["AAA", "BBB"])

    def test_constant_weight_daily_rebalanced_returns_and_missing_data(self) -> None:
        returns = pd.DataFrame(
            {"AAA": [0.10, np.nan], "BBB": [0.0, 0.05]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        result = calculate_portfolio_returns(
            returns, {"AAA": 0.5, "BBB": 0.5}
        )

        self.assertAlmostEqual(result.iloc[0], 0.05)
        self.assertTrue(np.isnan(result.iloc[1]))


if __name__ == "__main__":
    unittest.main()
