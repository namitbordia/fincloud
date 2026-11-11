"""Health route."""

from typing import Any

from backend.app.errors import S3ObjectNotFoundError, S3RepositoryError
from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository


def get_health(
    repository: S3Repository,
    *,
    api_gateway_verified: bool = False,
) -> tuple[int, dict[str, Any]]:
    try:
        repository.check_silver_access()
    except (S3ObjectNotFoundError, S3RepositoryError) as error:
        return 503, {
            "status": "degraded",
            "runtime_environment": (
                "aws_lambda" if repository.is_lambda else "local"
            ),
            "aws_region": repository.settings.region,
            "s3_connectivity": "unavailable",
            "error": str(error),
            "cloud_execution": cloud_execution(
                repository,
                repository.settings,
                data_verified=False,
            ),
            "api_gateway": "verified" if api_gateway_verified else "not_verified",
        }
    verified = repository.is_lambda
    return 200, {
        "status": "healthy",
        "runtime_environment": "aws_lambda" if verified else "local_mock",
        "aws_region": repository.settings.region,
        "s3_connectivity": "verified",
        "cloud_execution": cloud_execution(
            repository,
            repository.settings,
            data_verified=True,
        ),
        "api_gateway": "verified" if api_gateway_verified else "not_verified",
    }
