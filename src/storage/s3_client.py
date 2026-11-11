"""Explicit, reusable S3 operations with injectable clients for offline tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.storage.s3_config import S3Config
from src.storage.s3_manifest import sha256_file

_MISSING_CODES = {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}


class S3StorageError(RuntimeError):
    """A safe, credential-free description of an S3 operation failure."""


def _error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def _is_missing(error: ClientError) -> bool:
    code = _error_code(error)
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _MISSING_CODES or status == 404


def _raise_storage_error(operation: str, error: Exception) -> S3StorageError:
    if isinstance(error, ClientError):
        code = _error_code(error) or "ClientError"
        if code in {"403", "AccessDenied", "AllAccessDisabled"}:
            message = f"{operation} was denied by S3."
        else:
            message = f"{operation} failed with S3 error {code}."
    elif isinstance(error, BotoCoreError):
        message = f"{operation} failed because of an AWS SDK or network error."
    else:
        message = f"{operation} failed ({type(error).__name__})."
    return S3StorageError(message)


class S3Storage:
    """S3 wrapper; creating/importing it does not resolve credentials or call AWS."""

    def __init__(
        self,
        config: S3Config,
        client: Any | None = None,
        identity_client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._identity_client = identity_client

    def _get_client(self) -> Any:
        if self.config.dry_run:
            raise S3StorageError("This operation is disabled in dry-run mode.")
        if self._client is None:
            self._client = boto3.client("s3", region_name=self.config.region)
        return self._client

    def _get_identity_client(self) -> Any:
        if self.config.dry_run:
            raise S3StorageError("This operation is disabled in dry-run mode.")
        if self._identity_client is None:
            self._identity_client = boto3.client("sts", region_name=self.config.region)
        return self._identity_client

    def check_connection(self) -> dict[str, Any]:
        """Read AWS caller identity and verify bucket access and configured region."""
        try:
            identity = self._get_identity_client().get_caller_identity()
        except (ClientError, BotoCoreError, OSError) as error:
            raise _raise_storage_error("AWS identity check", error) from error

        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.config.bucket)
        except ClientError as error:
            if _is_missing(error):
                return {
                    "account": identity.get("Account"),
                    "arn": identity.get("Arn"),
                    "bucket": self.config.bucket,
                    "configured_region": self.config.region,
                    "bucket_accessible": False,
                    "bucket_region": None,
                    "region_matches": False,
                }
            raise _raise_storage_error("Bucket access check", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("Bucket access check", error) from error

        try:
            location = client.get_bucket_location(Bucket=self.config.bucket)
        except (ClientError, BotoCoreError, OSError) as error:
            raise _raise_storage_error("Bucket region check", error) from error
        bucket_region = location.get("LocationConstraint") or "us-east-1"
        return {
            "account": identity.get("Account"),
            "arn": identity.get("Arn"),
            "bucket": self.config.bucket,
            "configured_region": self.config.region,
            "bucket_accessible": True,
            "bucket_region": bucket_region,
            "region_matches": bucket_region == self.config.region,
        }

    def check_bucket_accessible(self) -> bool:
        """Check bucket access without ever creating a bucket."""
        try:
            self._get_client().head_bucket(Bucket=self.config.bucket)
            return True
        except ClientError as error:
            if _is_missing(error):
                return False
            raise _raise_storage_error("Bucket access check", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("Bucket access check", error) from error

    def upload_file(
        self,
        local_path: str | Path,
        key: str,
    ) -> dict[str, Any]:
        """Conditionally upload one file, refusing to replace an existing object."""
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"Upload source does not exist: {source}")
        size_bytes = source.stat().st_size
        if size_bytes > 5_000_000_000:
            raise S3StorageError(
                "Conditional single-request uploads are limited to 5 GB; "
                "this file was not uploaded."
            )
        digest = sha256_file(source)
        result = {
            "bucket": self.config.bucket,
            "key": key,
            "size_bytes": size_bytes,
            "sha256": digest,
            "region": self.config.region,
            "upload_timestamp_utc": None,
            "dry_run": self.config.dry_run,
        }
        if self.config.dry_run:
            return result
        client = self._get_client()
        try:
            with source.open("rb") as body:
                client.put_object(
                    Bucket=self.config.bucket,
                    Key=key,
                    Body=body,
                    ContentLength=size_bytes,
                    IfNoneMatch="*",
                    ServerSideEncryption="AES256",
                    Metadata={"sha256": digest},
                )
        except ClientError as error:
            if _error_code(error) in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise S3StorageError(
                    f"Refusing to overwrite existing S3 object {key!r}; choose a new key."
                ) from error
            raise _raise_storage_error("S3 upload", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("S3 upload", error) from error
        result["upload_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return result

    def download_file(self, key: str, destination: str | Path) -> dict[str, Any]:
        """Stream an object to a new local file; refuse to overwrite any file."""
        client = self._get_client()
        target = Path(destination)
        if target.exists():
            raise FileExistsError(f"Download destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        response: dict[str, Any] | None = None
        body: BinaryIO | None = None
        created = False
        completed = False
        try:
            response = client.get_object(
                Bucket=self.config.bucket,
                Key=key,
            )
            body = response["Body"]
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                created = True
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            expected = response.get("Metadata", {}).get("sha256")
            actual = digest.hexdigest()
            if expected and expected.lower() != actual:
                raise S3StorageError("Downloaded object does not match its stored SHA-256 metadata.")
            completed = True
            return {
                "bucket": self.config.bucket,
                "key": key,
                "destination": str(target),
                "size_bytes": size,
                "sha256": actual,
                "remote_checksum_verified": bool(expected),
                "checksum_verification": (
                    "object_metadata_matched_downloaded_bytes"
                    if expected
                    else "no_remote_checksum_available"
                ),
            }
        except ClientError as error:
            if _is_missing(error):
                raise FileNotFoundError(f"S3 object does not exist: {key}") from error
            raise _raise_storage_error("S3 download", error) from error
        except BotoCoreError as error:
            raise _raise_storage_error("S3 download", error) from error
        except FileExistsError:
            raise
        except OSError as error:
            raise _raise_storage_error("S3 download", error) from error
        finally:
            if body is not None and hasattr(body, "close"):
                body.close()
            if created and not completed and target.exists():
                target.unlink(missing_ok=True)

    def list_objects(self, prefix: str = "") -> list[dict[str, Any]]:
        """List object keys and sizes under a prefix."""
        try:
            paginator = self._get_client().get_paginator("list_objects_v2")
            results: list[dict[str, Any]] = []
            for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
                results.extend(
                    {
                        "key": item["Key"],
                        "size_bytes": item["Size"],
                        "last_modified": item.get("LastModified"),
                    }
                    for item in page.get("Contents", [])
                )
            return results
        except (ClientError, BotoCoreError, OSError) as error:
            raise _raise_storage_error("S3 object listing", error) from error

    def object_exists(self, key: str) -> bool:
        """Return false only for a not-found response; surface access/network errors."""
        try:
            self._get_client().head_object(Bucket=self.config.bucket, Key=key)
            return True
        except ClientError as error:
            if _is_missing(error):
                return False
            raise _raise_storage_error("S3 object existence check", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("S3 object existence check", error) from error

    def get_object_metadata(self, key: str) -> dict[str, Any] | None:
        """Return safe object headers; ETag is exposed but never treated as SHA-256."""
        try:
            response = self._get_client().head_object(
                Bucket=self.config.bucket,
                Key=key,
                ChecksumMode="ENABLED",
            )
        except ClientError as error:
            if _is_missing(error):
                return None
            raise _raise_storage_error("S3 object metadata request", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("S3 object metadata request", error) from error
        return {
            "bucket": self.config.bucket,
            "key": key,
            "size_bytes": response.get("ContentLength"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified"),
            "server_side_encryption": response.get("ServerSideEncryption"),
            "metadata": response.get("Metadata", {}),
            "checksum_sha256": response.get("ChecksumSHA256"),
            "checksum_type": response.get("ChecksumType"),
        }

    def verify_local_checksum(self, local_path: str | Path, key: str) -> dict[str, Any]:
        """Compare local SHA-256 with bytes streamed from S3; ETags are not checksums."""
        local_digest = sha256_file(local_path)
        body: BinaryIO | None = None
        try:
            response = self._get_client().get_object(
                Bucket=self.config.bucket,
                Key=key,
            )
            body = response["Body"]
            remote_hash = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                remote_hash.update(chunk)
                size += len(chunk)
        except ClientError as error:
            if _is_missing(error):
                raise FileNotFoundError(f"S3 object does not exist: {key}") from error
            raise _raise_storage_error("S3 checksum verification", error) from error
        except (BotoCoreError, OSError) as error:
            raise _raise_storage_error("S3 checksum verification", error) from error
        finally:
            if body is not None and hasattr(body, "close"):
                body.close()
        remote_digest = remote_hash.hexdigest()
        return {
            "bucket": self.config.bucket,
            "key": key,
            "local_sha256": local_digest,
            "remote_sha256": remote_digest,
            "remote_size_bytes": size,
            "verified": local_digest == remote_digest,
            "verification_method": "sha256_of_streamed_remote_object_bytes",
        }
