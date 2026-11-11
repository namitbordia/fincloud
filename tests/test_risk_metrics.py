"""Tests for historical tail risk and drawdown calculations."""

import unittest

import numpy as np

from src.analytics.risk_metrics import drawdown_analysis, historical_var_cvar


class TestRiskMetrics(unittest.TestCase):
    def test_historical_var_and_cvar_use_positive_loss_convention(self) -> None:
        returns = np.array([0.02, 0.01, 0.0, -0.01, -0.04, -0.08])

        result = historical_var_cvar(returns, 0.95)

        self.assertTrue(result["available"])
        self.assertGreater(result["var"], 0.0)
        self.assertGreaterEqual(result["cvar"], result["var"])
        self.assertAlmostEqual(result["cvar"], 0.08)

    def test_insufficient_observations_are_explicit(self) -> None:
        result = historical_var_cvar([0.01], 0.99)

        self.assertFalse(result["available"])
        self.assertTrue(np.isnan(result["var"]))
        self.assertIn("required", result["unavailable_reason"])

    def test_drawdown_matches_running_peak_definition(self) -> None:
        path, maximum = drawdown_analysis([0.1, -0.2, 0.05], initial_value=100)

        np.testing.assert_allclose(
            path["wealth_index"], [110, 88, 92.4], rtol=1e-12
        )
        np.testing.assert_allclose(path["running_peak"], [110, 110, 110])
        self.assertAlmostEqual(maximum, -0.2)


if __name__ == "__main__":
    unittest.main()
