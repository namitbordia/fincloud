"""Offline tests for statistical volatility forecasting."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecasting.config import ForecastConfig
from src.forecasting.evaluation import forecast_metrics
from src.forecasting.models import (
    ewma_forecast,
    garch_forecast,
    historical_rolling_forecast,
)
from src.forecasting.reporting import run_forecasting
from src.forecasting.targets import (
    assign_split,
    build_forward_targets,
    forward_realized_volatility,
)


def synthetic_returns(
    *,
    start: str = "2020-01-01",
    periods: int = 1500,
    seed: int = 17,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    random = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for symbol, scale in (("AAA", 1.0), ("BBB", 1.4)):
        returns = random.normal(0.0002, 0.012 * scale, len(dates))
        returns[0] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "daily_return": returns,
                    "adj_close": 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestStatisticalForecasting(unittest.TestCase):
    def test_historical_rolling_volatility_is_annualized_sample_standard_deviation(
        self,
    ) -> None:
        returns = np.array([0.01, -0.02, 0.015, 0.005])
        result = historical_rolling_forecast(
            returns, window=4, trading_days_per_year=252
        )
        expected = np.std(returns, ddof=1) * np.sqrt(252)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.volatility, expected)

    def test_ewma_uses_sample_variance_initialization_and_recursion(self) -> None:
        returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
        decay = 0.8
        initial = np.var(returns[:3], ddof=1)
        variance = decay * initial + (1 - decay) * returns[3] ** 2
        variance = decay * variance + (1 - decay) * returns[4] ** 2
        result = ewma_forecast(
            returns,
            decay=decay,
            initialization_window=3,
            trading_days_per_year=252,
        )
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.volatility, np.sqrt(variance * 252))

    def test_garch_fits_synthetic_returns_and_forecasts_requested_horizon(self) -> None:
        random = np.random.default_rng(42)
        returns = random.normal(0.0003, 0.012, 500)
        result = garch_forecast(
            returns,
            horizon=5,
            min_observations=250,
            trading_days_per_year=252,
        )
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.convergence_flag, 0)
        self.assertTrue(np.isfinite(result.volatility))
        self.assertGreater(result.volatility, 0)

    def test_failed_garch_fit_is_reported_without_fallback(self) -> None:
        with patch(
            "src.forecasting.models.arch_model",
            side_effect=RuntimeError("synthetic optimizer failure"),
        ):
            result = garch_forecast(
                np.linspace(-0.02, 0.02, 300),
                min_observations=250,
            )
        self.assertEqual(result.status, "fit_failed")
        self.assertIsNone(result.volatility)
        self.assertIn("synthetic optimizer failure", result.error or "")

    def test_forward_realized_volatility_uses_exact_next_horizon(self) -> None:
        future = np.array([0.01, -0.02, 0.005])
        result = forward_realized_volatility(
            future, horizon=3, trading_days_per_year=252
        )
        expected = np.sqrt((252 / 3) * np.sum(future**2))
        self.assertAlmostEqual(result, expected)
        with self.assertRaises(ValueError):
            forward_realized_volatility(future[:2], horizon=3)

    def test_target_is_next_returns_only_and_forecast_prefix_is_causal(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=100)
        returns = np.linspace(-0.02, 0.02, len(dates))
        origin_position = 40
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": "AAA",
                "daily_return": returns,
            }
        )
        first_target = build_forward_targets(
            frame,
            horizon=20,
            train_end="2022-12-31",
            validation_end="2023-12-31",
        )
        origin = dates[origin_position].strftime("%Y-%m-%d")
        row = first_target.loc[first_target["date"] == origin].iloc[0]
        expected = forward_realized_volatility(
            returns[origin_position + 1 : origin_position + 21]
        )
        self.assertAlmostEqual(row["actual_realized_volatility"], expected)

        changed = frame.copy()
        changed.loc[origin_position + 1, "daily_return"] = 0.4
        second_target = build_forward_targets(
            changed,
            horizon=20,
            train_end="2022-12-31",
            validation_end="2023-12-31",
        )
        changed_row = second_target.loc[second_target["date"] == origin].iloc[0]
        self.assertNotAlmostEqual(
            row["actual_realized_volatility"],
            changed_row["actual_realized_volatility"],
        )
        before = historical_rolling_forecast(
            returns[: origin_position + 1], window=20
        )
        after = historical_rolling_forecast(
            changed["daily_return"].to_numpy()[: origin_position + 1], window=20
        )
        self.assertEqual(before.volatility, after.volatility)

    def test_targets_mark_split_boundary_and_invalid_future_returns(self) -> None:
        dates = pd.bdate_range("2023-12-01", periods=50)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": "AAA",
                "daily_return": np.full(len(dates), 0.01),
            }
        )
        targets = build_forward_targets(
            frame,
            horizon=20,
            train_end="2022-12-31",
            validation_end="2023-12-31",
        )
        last_validation = targets.loc[
            (targets["split"] == "validation")
            & (targets["target_status"] == "crosses_split_boundary")
        ]
        self.assertGreater(len(last_validation), 0)

        invalid = frame.copy()
        invalid.loc[invalid.index[15], "daily_return"] = -1.5
        invalid_targets = build_forward_targets(
            invalid,
            horizon=10,
            train_end="2022-12-31",
            validation_end="2023-12-31",
        )
        origin = invalid.loc[invalid.index[5], "date"].strftime("%Y-%m-%d")
        invalid_target = invalid_targets.loc[
            invalid_targets["date"] == origin
        ].iloc[0]
        self.assertEqual(
            invalid_target["target_status"], "missing_or_invalid_future_return"
        )

    def test_chronological_split_assigns_dates_without_shuffling(self) -> None:
        self.assertEqual(
            assign_split(
                pd.Timestamp("2022-12-30"),
                train_end="2022-12-31",
                validation_end="2023-12-31",
            ),
            "train",
        )
        self.assertEqual(
            assign_split(
                pd.Timestamp("2023-06-01"),
                train_end="2022-12-31",
                validation_end="2023-12-31",
            ),
            "validation",
        )
        self.assertEqual(
            assign_split(
                pd.Timestamp("2024-01-02"),
                train_end="2022-12-31",
                validation_end="2023-12-31",
            ),
            "test",
        )

    def test_metrics_mae_rmse_and_qlike_handle_zero_realized_variance(self) -> None:
        actual = np.array([0.0, 0.2])
        forecast = np.array([0.1, 0.1])
        metrics = forecast_metrics(
            actual, forecast, variance_floor=1e-12
        )
        self.assertAlmostEqual(metrics["mae"], 0.1)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.01))
        self.assertTrue(np.isfinite(metrics["qlike"]))
        self.assertEqual(metrics["valid_forecasts"], 2)
        positive_variance_metrics = forecast_metrics([0.2], [0.1])
        ratio = (0.2**2) / (0.1**2)
        self.assertAlmostEqual(
            positive_variance_metrics["qlike"], ratio - np.log(ratio) - 1
        )

    def test_missing_and_invalid_model_history_is_not_filled(self) -> None:
        returns = np.array([0.01] * 19 + [np.nan])
        for result in (
            historical_rolling_forecast(returns, window=20),
            ewma_forecast(returns, initialization_window=20),
        ):
            self.assertEqual(result.status, "invalid_history")
            self.assertIsNone(result.volatility)
        with self.assertRaises(ValueError):
            ewma_forecast([0.01, 0.02], decay=1.0)
        self.assertEqual(
            garch_forecast([0.01] * 20, min_observations=252).status,
            "insufficient_history",
        )

    def test_pipeline_outputs_are_reproducible_and_inputs_remain_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "silver"
            input_dir.mkdir()
            source = input_dir / "all_stocks.csv"
            synthetic_returns(periods=1250).loc[
                lambda frame: frame["symbol"] == "AAA"
            ].to_csv(source, index=False)
            before = sha256(source)
            config = ForecastConfig(
                min_garch_observations=252,
                train_end="2022-12-31",
                validation_end="2023-12-31",
            )
            output_dir = root / "forecasting" / "statistical"
            first = run_forecasting(
                input_dir,
                output_dir,
                raw_dir=None,
                symbols=("AAA",),
                config=config,
            )
            self.assertTrue(
                first["forecasts"]
                .loc[first["forecasts"]["model"] == "garch_1_1", "forecast_status"]
                .eq("ok")
                .all()
            )
            self.assertEqual(first["report"]["failed_fit_count"], 0)
            forecast_bytes = (output_dir / "statistical_forecasts.csv").read_bytes()
            metrics_bytes = (output_dir / "statistical_model_metrics.csv").read_bytes()
            second = run_forecasting(
                input_dir,
                output_dir,
                raw_dir=None,
                symbols=("AAA",),
                config=config,
            )
            expected = {
                "statistical_forecasts.csv",
                "statistical_model_metrics.csv",
                "statistical_stock_metrics.csv",
                "statistical_validation_report.json",
                "statistical_manifest.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))
            self.assertTrue(first["report"]["valid"])
            self.assertTrue(second["report"]["valid"])
            self.assertTrue(first["report"]["checks"]["features_cut_off_at_forecast_origin"])
            self.assertEqual(
                forecast_bytes, (output_dir / "statistical_forecasts.csv").read_bytes()
            )
            self.assertEqual(
                metrics_bytes, (output_dir / "statistical_model_metrics.csv").read_bytes()
            )
            self.assertEqual(before, sha256(source))
            self.assertEqual(first["manifest"]["model_configuration"]["ewma_lambda"], 0.94)
            self.assertTrue(
                (output_dir / "plots" / "actual_vs_forecast_test.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
