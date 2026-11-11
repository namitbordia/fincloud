"""Build a compact, reproducible JSON summary from the checked-in outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        result.append({
            key: (float(value) if key in {"mae", "rmse", "qlike", "forecast_coverage"} and value else value)
            for key, value in row.items()
        })
    return result


def main() -> None:
    silver_dir = ROOT / "data/processed/silver"
    ml_dir = ROOT / "data/processed/forecasting/ml"
    output = {
        "dataset": read_json(silver_dir / "silver_dataset_manifest.json"),
        "dataset_validation": read_json(silver_dir / "silver_validation_report.json"),
        "ml_validation": read_json(ml_dir / "ml_validation_report.json"),
        "model_metrics": read_metrics(ml_dir / "ml_model_metrics.csv"),
        "optimization_validation": read_json(ROOT / "data/processed/optimization/backtest_validation_report.json"),
        "data_source_and_limitations": (
            "Daily OHLCV and adjusted close sourced through yfinance from Yahoo Finance. "
            "Supplementary research data; not official NSE-certified data. Adjusted-close "
            "and corporate-action handling follows the source series and should be independently verified."
        ),
    }
    destination = ml_dir / "fincloud_research_summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
