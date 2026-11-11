"""Environment-backed settings for the cloud API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    bucket: str
    region: str
    silver_key: str
    silver_manifest_key: str
    model_prefix: str
    results_prefix: str
    cors_origins: tuple[str, ...]
    experiments_key: str = "experiments/ml/fincloud_research_summary.json"

    @classmethod
    def from_environment(cls) -> Settings:
        bucket = os.environ.get(
            "FINCLOUD_S3_BUCKET", "fincloud-market-data-unique123"
        ).strip()
        if not bucket:
            raise ValueError("FINCLOUD_S3_BUCKET must not be empty.")
        region = os.environ.get(
            "FINCLOUD_AWS_REGION", os.environ.get("AWS_REGION", "ap-south-1")
        ).strip()
        if not region:
            raise ValueError("FINCLOUD_AWS_REGION must not be empty.")
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "FINCLOUD_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        return cls(
            bucket=bucket,
            region=region,
            silver_key=os.environ.get(
                "FINCLOUD_SILVER_KEY", "silver/all_stocks.csv"
            ).strip(),
            silver_manifest_key=os.environ.get(
                "FINCLOUD_SILVER_MANIFEST_KEY",
                "silver/silver_dataset_manifest.json",
            ).strip(),
            model_prefix=os.environ.get(
                "FINCLOUD_MODEL_PREFIX", "models/volatility"
            ).strip("/")
            + "/",
            results_prefix=os.environ.get(
                "FINCLOUD_RESULTS_PREFIX", "gold/api-results"
            ).strip("/")
            + "/",
            experiments_key=os.environ.get(
                "FINCLOUD_EXPERIMENTS_KEY",
                "experiments/ml/fincloud_research_summary.json",
            ).strip(),
            cors_origins=origins,
        )
