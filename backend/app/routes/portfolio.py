"""Risk-analysis route."""

from typing import Any

from backend.app.services.risk_service import run_risk
from backend.app.services.s3_repository import S3Repository


def post_risk(body: dict[str, Any], repository: S3Repository) -> dict[str, Any]:
    return run_risk(body, repository)
