"""Truthful runtime and data-source indicators."""

from __future__ import annotations

from typing import Any

from backend.app.config import Settings
from backend.app.services.s3_repository import S3Repository


def cloud_execution(
    repository: S3Repository,
    settings: Settings,
    *,
    data_verified: bool,
    model_verified: bool = False,
) -> dict[str, Any]:
    verified = bool(repository.is_lambda and data_verified)
    return {
        "verified": verified,
        "data_source": "Amazon S3" if verified else "Mocked S3 (local development)",
        "compute": "AWS Lambda" if repository.is_lambda else "Local backend",
        "execution_location": settings.region if repository.is_lambda else "local",
        "model_source": (
            "Amazon S3"
            if verified and model_verified
            else "Not verified"
            if model_verified
            else None
        ),
        "region": settings.region if repository.is_lambda else None,
        "note": (
            None
            if verified
            else "Local/mock execution; AWS cloud execution has not been verified."
        ),
    }
