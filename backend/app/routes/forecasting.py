"""Forecast route."""

from typing import Any

from backend.app.services.forecast_service import run_forecast
from backend.app.services.s3_repository import S3Repository


def post_forecast(body: dict[str, Any], repository: S3Repository) -> dict[str, Any]:
    return run_forecast(body, repository)
