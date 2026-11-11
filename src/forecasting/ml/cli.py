"""CLI for machine-learning volatility forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.forecasting.ml.config import MLForecastConfig
from src.forecasting.ml.pipeline import run_ml_forecasting


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast realized volatility using chronological ML models."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/silver"))
    parser.add_argument(
        "--statistical-forecasts",
        type=Path,
        default=Path(
            "data/processed/forecasting/statistical/statistical_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/forecasting/ml"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/volatility"),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--refit-frequency", choices=("monthly", "quarterly"), default="monthly")
    args = parser.parse_args(arguments)
    config = MLForecastConfig(
        horizon=args.horizon,
        train_end=args.train_end,
        validation_end=args.validation_end,
        random_seed=args.random_seed,
        refit_frequency=args.refit_frequency,
    )
    result = run_ml_forecasting(
        args.input_dir,
        args.statistical_forecasts,
        args.output_dir,
        args.model_dir,
        symbols=args.symbols,
        config=config,
    )
    print(f"Validation: {'passed' if result['report']['valid'] else 'failed'}")
    print(f"Forecast rows: {len(result['forecasts'])}")
    print(f"Outputs: {Path(args.output_dir).resolve()}")
    print(f"Model artifacts: {Path(args.model_dir).resolve()}")
    return 0 if result["report"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
