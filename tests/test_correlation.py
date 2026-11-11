"""Tests for pairwise-complete stock return correlation."""

import unittest

import numpy as np
import pandas as pd

from src.analytics.correlation import correlation_analysis


class TestCorrelation(unittest.TestCase):
    def test_correlation_aligns_pairwise_and_does_not_fill_missing_values(self) -> None:
        returns = pd.DataFrame(
            {
                "AAA": [0.01, 0.02, np.nan, 0.04],
                "BBB": [0.02, 0.04, 0.06, 0.08],
            }
        )

        matrix, details = correlation_analysis(returns)

        self.assertAlmostEqual(matrix.loc["AAA", "BBB"], 1.0)
        self.assertEqual(details["observation_counts"]["AAA"]["BBB"], 3)
        self.assertFalse(details["missing_values_filled"])


if __name__ == "__main__":
    unittest.main()
