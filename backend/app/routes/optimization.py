"""Portfolio-optimization route."""

from typing import Any

from backend.app.services.optimization_service import run_optimization
from backend.app.services.s3_repository import S3Repository


def post_optimization(
    body: dict[str, Any], repository: S3Repository
) -> dict[str, Any]:
    return run_optimization(body, repository)
