"""CLI for sensitivity runs and walk-forward portfolio optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.optimization.config import OptimizationConfig
from src.optimization.reporting import run_optimization


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize portfolios and run chronological walk-forward backtests."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/silver"))
    parser.add_argument(
        "--ml-comparison-csv",
        type=Path,
        default=Path("data/processed/forecasting/ml/model_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/optimization")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nse"))
    parser.add_argument("--forecast-model", default="random_forest")
    parser.add_argument("--windows", nargs="+", type=int, default=[60, 120, 252])
    parser.add_argument(
        "--frequencies",
        nargs="+",
        choices=("monthly", "quarterly"),
        default=["monthly", "quarterly"],
    )
    parser.add_argument("--maximum-weights", nargs="+", type=float, default=[0.6])
    parser.add_argument("--minimum-weight", type=float, default=0.0)
    parser.add_argument("--maximum-holdings", type=int)
    parser.add_argument("--transaction-cost-bps", nargs="+", type=float, default=[0, 10])
    parser.add_argument("--cvar-confidences", nargs="+", type=float, default=[0.95])
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--rebalance-start", default="2023-01-01")
    args = parser.parse_args(arguments)
    config = OptimizationConfig(
        estimation_windows=tuple(args.windows),
        rebalance_frequencies=tuple(args.frequencies),
        maximum_asset_weights=tuple(args.maximum_weights),
        minimum_asset_weight=args.minimum_weight,
        maximum_holdings=args.maximum_holdings,
        transaction_cost_bps=tuple(args.transaction_cost_bps),
        cvar_confidences=tuple(args.cvar_confidences),
        annual_risk_free_rate=args.risk_free_rate,
        rebalance_start=args.rebalance_start,
    )
    result = run_optimization(
        args.input_dir,
        args.ml_comparison_csv,
        args.output_dir,
        raw_dir=args.raw_dir,
        forecast_model=args.forecast_model,
        config=config,
    )
    report = result["validation_report"]
    print(f"Validation: {'passed' if report['valid'] else 'failed'}")
    print(f"Daily return rows: {report['daily_return_rows']}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
