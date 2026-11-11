"""Offline tests for leakage-safe machine-learning volatility forecasting."""

from __future__ import annotations

import json
import platform
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.forecasting.ml.config import MLForecastConfig
from src.forecasting.ml.features import FEATURE_COLUMNS, build_features
from src.forecasting.ml.pipeline import (
    _fit_and_predict_symbol,
    run_ml_forecasting,
)
from src.forecasting.ml.targets import build_ml_targets


def synthetic_silver(periods: int = 1250, seed: int = 23) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    random = np.random.default_rng(seed)
    returns = random.normal(0.0002, 0.012, periods)
    returns[0] = np.nan
    prices = np.full(periods, 100.0)
    for index in range(1, periods):
        prices[index] = prices[index - 1] * (1.0 + returns[index])
    volume = random.integers(100_000, 500_000, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "daily_return": returns,
            "adj_close": prices,
            "close": prices,
            "volume": volume,
        }
    )


class TestMLForecasting(unittest.TestCase):
    def test_feature_generation_is_ordered_past_only_and_keeps_warmup_missing(self) -> None:
        source = synthetic_silver(periods=100)
        first = build_features(source)
        mutated = source.copy()
        mutated.loc[80, "daily_return"] = 0.5
        second = build_features(mutated)
        self.assertEqual(
            list(first.columns),
            ["symbol", "date", *FEATURE_COLUMNS],
        )
        self.assertTrue(first.iloc[0][list(FEATURE_COLUMNS)].isna().any())
        np.testing.assert_allclose(
            first.loc[:70, FEATURE_COLUMNS].to_numpy(dtype=float),
            second.loc[:70, FEATURE_COLUMNS].to_numpy(dtype=float),
            equal_nan=True,
        )
        self.assertIn("volume_change", FEATURE_COLUMNS)
        self.assertIn("rsi_14", FEATURE_COLUMNS)
        self.assertIn("volatility_60d", FEATURE_COLUMNS)

    def test_targets_match_phase5_formula_and_are_chronologically_bounded(self) -> None:
        source = synthetic_silver(periods=1250)
        targets = build_ml_targets(source)
        row = targets.loc[
            (targets["date"] == pd.Timestamp("2023-06-01"))
            & (targets["symbol"] == "AAA")
        ].iloc[0]
        position = int(source.index[source["date"] == pd.Timestamp("2023-06-01")][0])
        future = source.loc[position + 1 : position + 20, "daily_return"].to_numpy()
        expected = np.sqrt((252 / 20) * np.sum(future**2))
        self.assertAlmostEqual(row["actual_realized_volatility"], expected)
        self.assertLessEqual(pd.Timestamp(row["target_end_date"]), pd.Timestamp("2023-12-31"))
        train_targets = targets.loc[
            (targets["split"] == "train")
            & (targets["target_status"] == "available")
        ]
        self.assertTrue(
            pd.to_datetime(train_targets["target_end_date"]).dropna().le(
                pd.Timestamp("2022-12-31")
            ).all()
        )

    def test_ml_pipeline_compares_phase5_and_serializes_models_reproducibly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "silver"
            input_dir.mkdir()
            source = synthetic_silver()
            source.to_csv(input_dir / "all_stocks.csv", index=False)
            (input_dir / "silver_validation_report.json").write_text(
                json.dumps(
                    {
                        "methodology": {"adjusted_close_source_column": "adj_close"},
                        "checks": {
                            "calculations": {
                                "daily_returns_mathematically_correct": True
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            target_frame = source.copy()
            targets = build_ml_targets(target_frame)
            dates = pd.to_datetime(source["date"])
            baseline_rows = []
            for target in targets.loc[targets["date"] > pd.Timestamp("2022-12-31")].itertuples():
                for model in ("historical_rolling", "ewma", "garch_1_1"):
                    baseline_rows.append(
                        {
                            "date": target.date.strftime("%Y-%m-%d"),
                            "symbol": "AAA",
                            "split": target.split,
                            "target_end_date": target.target_end_date,
                            "target_status": target.target_status,
                            "actual_realized_volatility": target.actual_realized_volatility,
                            "model": model,
                            "forecast_volatility": 0.2,
                            "forecast_status": "ok",
                        }
                    )
            statistical = root / "statistical.csv"
            pd.DataFrame(baseline_rows).to_csv(statistical, index=False)
            config = MLForecastConfig(
                model_parameters={
                    "random_forest": {"n_estimators": 8, "max_depth": 3},
                    "xgboost": {
                        "n_estimators": 8,
                        "max_depth": 2,
                        "learning_rate": 0.1,
                    },
                    "ridge": {"alpha": 1.0},
                }
            )
            output_dir = root / "ml"
            model_dir = root / "models"
            before = (input_dir / "all_stocks.csv").read_bytes()
            first = run_ml_forecasting(
                input_dir,
                statistical,
                output_dir,
                model_dir,
                symbols=("AAA",),
                config=config,
            )
            predictions_first = first["forecasts"][
                ["date", "model", "forecast_volatility", "forecast_status"]
            ].copy()
            second = run_ml_forecasting(
                input_dir,
                statistical,
                output_dir,
                model_dir,
                symbols=("AAA",),
                config=config,
            )
            predictions_second = second["forecasts"][
                ["date", "model", "forecast_volatility", "forecast_status"]
            ].copy()
            pd.testing.assert_frame_equal(
                predictions_first, predictions_second, check_exact=False, rtol=1e-12
            )
            self.assertTrue(first["report"]["valid"])
            self.assertTrue(first["report"]["checks"]["target_is_strictly_future"])
            self.assertEqual(set(first["comparison"]["model"].unique()), {
                "historical_rolling",
                "ewma",
                "garch_1_1",
                "random_forest",
                "xgboost",
                "ridge",
            })
            self.assertEqual(before, (input_dir / "all_stocks.csv").read_bytes())
            metadata = json.loads(
                (model_dir / "model_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["feature_names_and_order"], list(FEATURE_COLUMNS))
            self.assertEqual(
                metadata["runtime_versions"]["python"], platform.python_version()
            )
            self.assertIn("validation_date_range", metadata)
            self.assertIn(
                metadata["validation_date_range"]["start"],
                set(source["date"].dt.strftime("%Y-%m-%d")),
            )
            artifacts = list(model_dir.glob("*.joblib"))
            self.assertEqual(len(artifacts), 3)
            restored = joblib.load(artifacts[0])
            self.assertTrue(hasattr(restored, "predict"))
            comparison = pd.read_csv(output_dir / "model_comparison.csv")
            self.assertEqual(comparison.model.nunique(), 6)
            self.assertTrue(
                (output_dir / "plots" / "actual_vs_predicted_volatility.png").is_file()
            )

    def test_failed_model_fits_remain_unavailable_without_fallback(self) -> None:
        from unittest.mock import patch

        source = synthetic_silver(periods=800)
        features = build_features(source)
        targets = build_ml_targets(source)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "src.forecasting.ml.pipeline._train_model",
                side_effect=ValueError("synthetic fitting failure"),
            ):
                forecasts, _, artifacts = _fit_and_predict_symbol(
                    "AAA",
                    source,
                    features,
                    targets,
                    config=MLForecastConfig(),
                    artifact_dir=Path(temporary_directory),
                )
        self.assertTrue(forecasts)
        self.assertEqual(
            {row["forecast_status"] for row in forecasts},
            {"fit_failed", "fit_unavailable"},
        )
        self.assertTrue(
            all(
                row["forecast_status"] == "fit_failed"
                for row in forecasts
                if row["date"] == "2023-01-02"
            )
        )
        self.assertTrue(
            all(row["forecast_volatility"] is None for row in forecasts)
        )
        self.assertTrue(artifacts["failures"])

    def test_invalid_duplicate_and_insufficient_silver_inputs_are_reported(self) -> None:
        source = synthetic_silver(periods=80)
        duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_features(duplicated)
        targets = build_ml_targets(source, horizon=20)
        self.assertTrue(
            targets.loc[targets["date"] >= pd.Timestamp("2020-01-01"), "target_status"]
            .eq("insufficient_future_returns")
            .any()
        )


if __name__ == "__main__":
    unittest.main()
