"""Environment-based configuration for the optional S3 data lake."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

DEFAULT_REGION = "ap-south-1"
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d$")


class S3ConfigurationError(ValueError):
    """Raised when required S3 settings are missing or malformed."""


def _validate_bucket_name(bucket: str) -> None:
    if not _BUCKET_PATTERN.fullmatch(bucket):
        raise S3ConfigurationError(
            "FINCLOUD_S3_BUCKET must be a valid 3-63 character S3 bucket name."
        )
    if ".." in bucket or ".-" in bucket or "-." in bucket:
        raise S3ConfigurationError("FINCLOUD_S3_BUCKET contains an invalid separator.")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", bucket):
        raise S3ConfigurationError("FINCLOUD_S3_BUCKET cannot be formatted as an IP address.")
    if bucket.startswith("xn--") or bucket.endswith(
        ("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3")
    ):
        raise S3ConfigurationError("FINCLOUD_S3_BUCKET uses a reserved bucket name.")


def _parse_boolean(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise S3ConfigurationError(f"{name} must be a boolean value.")


@dataclass(frozen=True)
class S3Config:
    """S3 target settings; credentials are supplied by boto3's provider chain."""

    bucket: str
    region: str = DEFAULT_REGION
    dry_run: bool = False

    def __post_init__(self) -> None:
        bucket = self.bucket.strip()
        region = self.region.strip()
        _validate_bucket_name(bucket)
        if not _REGION_PATTERN.fullmatch(region):
            raise S3ConfigurationError("AWS region is malformed.")
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "region", region)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        dry_run: bool | None = None,
    ) -> S3Config:
        """Load configuration without resolving credentials or contacting AWS."""
        values = os.environ if environ is None else environ
        bucket = values.get("FINCLOUD_S3_BUCKET", "").strip()
        if not bucket:
            raise S3ConfigurationError(
                "Set FINCLOUD_S3_BUCKET to the intended bucket name."
            )
        region = (
            values.get("FINCLOUD_AWS_REGION")
            or values.get("AWS_REGION")
            or DEFAULT_REGION
        ).strip()
        configured_dry_run = _parse_boolean(
            "FINCLOUD_S3_DRY_RUN", values.get("FINCLOUD_S3_DRY_RUN", "false")
        )
        return cls(
            bucket=bucket,
            region=region,
            dry_run=configured_dry_run if dry_run is None else dry_run,
        )
