"""Offline tests for constrained optimization and walk-forward backtesting."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.optimization.backtest import walk_forward_backtest
from src.optimization.config import OptimizationConfig
from src.optimization.solvers import OptimizationError, optimize_weights


class TestOptimizationSolvers(unittest.TestCase):
    def setUp(self) -> None:
        self.mean = np.array([0.08, 0.12, 0.06])
        self.covariance = np.array(
            [
                [0.04, 0.006, 0.002],
                [0.006, 0.09, 0.004],
                [0.002, 0.004, 0.025],
            ]
        )
        random = np.random.default_rng(7)
        self.scenarios = random.multivariate_normal(
            [0.0003, 0.0005, 0.0002],
            self.covariance / 252,
            size=260,
        )

    def test_minimum_variance_and_max_sharpe_weights_are_feasible(self) -> None:
        for method in ("minimum_variance", "maximum_sharpe"):
            result = optimize_weights(
                method,
                self.mean,
                self.covariance,
                maximum_weight=0.6,
            )
            self.assertAlmostEqual(result.weights.sum(), 1.0, places=7)
            self.assertTrue((result.weights >= -1e-8).all())
            self.assertTrue((result.weights <= 0.6 + 1e-8).all())

    def test_cvar_linear_program_produces_constrained_weights(self) -> None:
        result = optimize_weights(
            "cvar_minimizing",
            self.mean,
            self.covariance,
            historical_returns=self.scenarios,
            cvar_confidence=0.95,
            maximum_weight=0.6,
        )
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=7)
        self.assertEqual(result.solver, "HiGHS linear programming")

    def test_maximum_holdings_and_infeasible_bounds_are_explicit(self) -> None:
        result = optimize_weights(
            "minimum_variance",
            self.mean,
            self.covariance,
            maximum_weight=0.8,
            maximum_holdings=2,
        )
        self.assertLessEqual(np.count_nonzero(result.weights), 2)
        with self.assertRaises(OptimizationError):
            optimize_weights(
                "minimum_variance",
                self.mean,
                self.covariance,
                minimum_weight=0.6,
                maximum_weight=0.7,
            )

    def test_fixed_asset_bounds_initialize_at_the_only_feasible_solution(self) -> None:
        result = optimize_weights(
            "minimum_variance",
            self.mean,
            self.covariance,
            minimum_weight=1.0 / 3.0,
            maximum_weight=1.0 / 3.0,
        )
        np.testing.assert_allclose(result.weights, np.full(3, 1.0 / 3.0))

    def test_cvar_requires_scenarios_and_covariance_must_be_psd(self) -> None:
        with self.assertRaises(ValueError):
            optimize_weights("cvar_minimizing", self.mean, self.covariance)
        invalid = self.covariance.copy()
        invalid[0, 0] = -1.0
        with self.assertRaises(ValueError):
            optimize_weights("minimum_variance", self.mean, invalid)


def synthetic_returns(periods: int = 900, seed: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    random = np.random.default_rng(seed)
    common = random.normal(0.0003, 0.009, periods)
    rows = []
    for index, symbol in enumerate(("AAA", "BBB", "CCC")):
        returns = common * (0.5 + index * 0.2) + random.normal(
            0.0001, 0.008 + index * 0.001, periods
        )
        returns[0] = np.nan
        rows.extend(
            {
                "date": date,
                "symbol": symbol,
                "daily_return": value,
            }
            for date, value in zip(dates, returns, strict=True)
        )
    return pd.DataFrame(rows)


class TestWalkForwardBacktest(unittest.TestCase):
    def test_backtest_is_chronological_and_costs_reduce_net_returns(self) -> None:
        frame = synthetic_returns()
        config = OptimizationConfig(
            estimation_windows=(60,),
            rebalance_frequencies=("monthly",),
            maximum_asset_weights=(0.8,),
            transaction_cost_bps=(0.0, 20.0),
            cvar_confidences=(0.95,),
            rebalance_start="2023-01-01",
        )
        result = walk_forward_backtest(frame, config=config)
        daily = result["daily_returns"]
        self.assertFalse(daily.empty)
        self.assertTrue(
            daily.groupby(["configuration_id", "strategy"])["date"]
            .apply(lambda dates: dates.is_monotonic_increasing)
            .all()
        )
        cost_scenarios = daily.loc[daily["transaction_cost_bps"] == 20.0]
        zero_cost = daily.loc[daily["transaction_cost_bps"] == 0.0]
        comparison_keys = [
            "estimation_window",
            "rebalance_frequency",
            "maximum_asset_weight",
            "cvar_confidence",
            "date",
            "strategy",
        ]
        merged = cost_scenarios.merge(
            zero_cost,
            on=comparison_keys,
            suffixes=("_cost", "_zero"),
        )
        self.assertFalse(merged.empty)
        self.assertTrue((merged["net_return_cost"] <= merged["net_return_zero"] + 1e-12).all())
        self.assertTrue(
            result["weights"]
            .loc[result["weights"]["optimization_status"] == "ok"]
            .groupby(["configuration_id", "rebalance_date", "strategy"])["weight"]
            .sum()
            .sub(1.0)
            .abs()
            .lt(1e-7)
            .all()
        )

    def test_future_changes_do_not_change_first_rebalance_weights(self) -> None:
        frame = synthetic_returns(periods=900)
        config = OptimizationConfig(
            estimation_windows=(60,),
            rebalance_frequencies=("quarterly",),
            maximum_asset_weights=(0.8,),
            transaction_cost_bps=(0.0,),
            cvar_confidences=(0.95,),
            rebalance_start="2023-01-01",
        )
        first = walk_forward_backtest(frame, config=config)
        changed = frame.copy()
        changed_dates = pd.to_datetime(changed["date"])
        changed.loc[changed_dates >= pd.Timestamp("2023-01-01"), "daily_return"] *= 2
        second = walk_forward_backtest(changed, config=config)
        first_decision = first["weights"]["rebalance_date"].min()
        first_weights = (
            first["weights"]
            .loc[
                (first["weights"]["rebalance_date"] == first_decision)
                & (first["weights"]["strategy"] == "minimum_variance")
            ]
            .sort_values("symbol")["weight"]
            .to_numpy()
        )
        second_weights = (
            second["weights"]
            .loc[
                (second["weights"]["rebalance_date"] == first_decision)
                & (second["weights"]["strategy"] == "minimum_variance")
            ]
            .sort_values("symbol")["weight"]
            .to_numpy()
        )
        np.testing.assert_allclose(first_weights, second_weights, rtol=1e-10, atol=1e-10)

    def test_forecast_aware_covariance_uses_prior_forecasts_and_past_correlation(self) -> None:
        frame = synthetic_returns(periods=900)
        config = OptimizationConfig(
            estimation_windows=(60,),
            rebalance_frequencies=("monthly",),
            maximum_asset_weights=(0.8,),
            transaction_cost_bps=(0.0,),
            cvar_confidences=(0.95,),
            rebalance_start="2023-01-01",
        )
        dates = pd.bdate_range("2022-12-01", "2024-06-30")
        forecasts = pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "model": "random_forest",
                    "forecast_status": "ok",
                    "forecast_volatility": 0.2 + index * 0.02,
                    "target_status": "insufficient_future_returns",
                    "actual_realized_volatility": np.nan,
                }
                for date in dates
                for index, symbol in enumerate(("AAA", "BBB", "CCC"))
            ]
        )
        result = walk_forward_backtest(
            frame,
            config=config,
            forecast_frame=forecasts,
            forecast_model="random_forest",
        )
        aware = result["weights"].loc[
            (result["weights"]["strategy"] == "minimum_variance_forecast_aware")
            & (result["weights"]["optimization_status"] == "ok")
        ]
        self.assertFalse(aware.empty)
        self.assertTrue(
            (
                pd.to_datetime(aware["forecast_origin"])
                < pd.to_datetime(aware["rebalance_date"])
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
