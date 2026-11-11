"""Command-line entry point for statistical volatility forecasting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.forecasting.config import ForecastConfig
from src.forecasting.reporting import DEFAULT_SYMBOLS, run_forecasting


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast forward realized volatility using statistical models."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/silver"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/forecasting/statistical"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nse"))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--ewma-lambda", type=float, default=0.94)
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--min-garch-observations", type=int, default=252)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(arguments)
    config = ForecastConfig(
        horizon=args.horizon,
        ewma_lambda=args.ewma_lambda,
        train_end=args.train_end,
        validation_end=args.validation_end,
        min_garch_observations=args.min_garch_observations,
    )
    result = run_forecasting(
        args.input_dir,
        args.output_dir,
        raw_dir=args.raw_dir,
        symbols=args.symbols,
        config=config,
    )
    report = result["report"]
    print(f"Validation: {'passed' if report['valid'] else 'failed'}")
    print(f"Forecast rows: {report['forecast_rows']}")
    print(f"Outputs: {Path(args.output_dir).resolve()}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
