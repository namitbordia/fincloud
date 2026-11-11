"""S3-only data and artifact access for deployed API execution."""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from backend.app.config import Settings
from backend.app.errors import S3ObjectNotFoundError, S3RepositoryError
from src.forecasting.ml.features import FEATURE_COLUMNS

_MISSING_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}
_RIDGE_TARGET = (
    "sqrt((trading_days_per_year / horizon) * "
    "sum(next horizon daily_return squared))"
)


class S3Repository:
    """Repository that never reads project-local data or artifacts."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self.is_lambda = bool(
            os.environ.get("AWS_LAMBDA_FUNCTION_NAME") and client is None
        )

    def _s3(self) -> Any:
        if self._client is None:
            self._client = boto3.client(
                "s3", region_name=self.settings.region
            )
        return self._client

    def _read_bytes(self, key: str) -> bytes:
        try:
            response = self._s3().get_object(
                Bucket=self.settings.bucket,
                Key=key,
            )
            body = response["Body"]
            try:
                data = body.read()
            finally:
                body.close()
            if not isinstance(data, bytes):
                raise S3RepositoryError(f"S3 object {key!r} did not return bytes.")
            return data
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in _MISSING_CODES:
                raise S3ObjectNotFoundError(
                    f"Required S3 object is missing: s3://{self.settings.bucket}/{key}"
                ) from error
            if code in {"403", "AccessDenied", "AllAccessDisabled"}:
                raise S3RepositoryError(
                    f"S3 access was denied for required object {key!r}."
                ) from error
            raise S3RepositoryError(
                f"S3 read failed for {key!r} (error {code or 'ClientError'})."
            ) from error
        except (BotoCoreError, OSError, KeyError) as error:
            raise S3RepositoryError(
                f"S3 read failed for {key!r} ({type(error).__name__})."
            ) from error

    def check_silver_access(self) -> bool:
        try:
            self._s3().head_object(
                Bucket=self.settings.bucket,
                Key=self.settings.silver_key,
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            status = error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if code in _MISSING_CODES or status == 404:
                raise S3ObjectNotFoundError(
                    f"Required S3 object is missing: "
                    f"s3://{self.settings.bucket}/{self.settings.silver_key}"
                ) from error
            if code in {"403", "AccessDenied", "AllAccessDisabled"}:
                raise S3RepositoryError(
                    f"S3 access was denied for {self.settings.silver_key!r}."
                ) from error
            raise S3RepositoryError(
                f"S3 connectivity check failed (error {code or 'ClientError'})."
            ) from error
        except (BotoCoreError, OSError) as error:
            raise S3RepositoryError(
                f"S3 connectivity check failed ({type(error).__name__})."
            ) from error
        return True

    def load_market_data(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        csv_bytes = self._read_bytes(self.settings.silver_key)
        manifest_bytes = self._read_bytes(self.settings.silver_manifest_key)
        try:
            manifest = json.loads(manifest_bytes)
            expected_hash = manifest["output_sha256"]["all_stocks.csv"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise S3RepositoryError(
                "Silver manifest is invalid or does not contain the combined CSV checksum."
            ) from error
        actual_hash = hashlib.sha256(csv_bytes).hexdigest()
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise S3RepositoryError(
                "Silver CSV SHA-256 does not match the S3 dataset manifest."
            )
        try:
            frame = pd.read_csv(io.BytesIO(csv_bytes))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise S3RepositoryError("Silver CSV could not be parsed.") from error
        required = {"date", "symbol", "daily_return", "adj_close", "volume"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise S3RepositoryError(
                f"Silver CSV is missing required columns: {missing}."
            )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
        for column in ("daily_return", "adj_close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame["date"].isna().any() or frame["symbol"].isna().any():
            raise S3RepositoryError("Silver CSV contains invalid date or symbol keys.")
        if frame.duplicated(["symbol", "date"]).any():
            raise S3RepositoryError("Silver CSV contains duplicate symbol/date rows.")
        symbols = sorted(str(symbol) for symbol in frame["symbol"].unique())
        manifest_symbols = sorted(manifest.get("stock_symbols", []))
        if symbols != manifest_symbols:
            raise S3RepositoryError(
                "Silver CSV symbol universe does not match its manifest."
            )
        expected_rows = manifest.get("combined_row_count")
        if expected_rows is not None and int(expected_rows) != len(frame):
            raise S3RepositoryError(
                "Silver CSV row count does not match its manifest."
            )
        frame = frame.sort_values(["symbol", "date"], kind="stable").reset_index(
            drop=True
        )
        metadata = {
            "bucket": self.settings.bucket,
            "key": self.settings.silver_key,
            "manifest_key": self.settings.silver_manifest_key,
            "sha256": actual_hash,
            "symbols": symbols,
            "row_count": len(frame),
            "start_date": frame["date"].min().date().isoformat(),
            "end_date": frame["date"].max().date().isoformat(),
        }
        return frame, metadata

    def load_model(
        self, symbol: str, model_name: str
    ) -> tuple[bytes, dict[str, Any], str]:
        metadata_key = self.settings.model_prefix + "model_metadata.json"
        try:
            metadata = json.loads(self._read_bytes(metadata_key))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise S3RepositoryError("Model metadata is not valid UTF-8 JSON.") from error
        try:
            if metadata.get("target_definition") != _RIDGE_TARGET:
                raise S3RepositoryError(
                    "Model metadata target definition is not the supported "
                    "forward realized-volatility target."
                )
            if metadata["feature_names_and_order"] != list(FEATURE_COLUMNS):
                raise S3RepositoryError(
                    "Model metadata feature order does not match the runtime."
                )
            artifact = metadata["model_artifacts"][symbol][model_name]
            feature_names = artifact["feature_names"]
            filename = artifact["filename"]
            expected_sha = artifact["sha256"]
            if feature_names != list(FEATURE_COLUMNS):
                raise S3RepositoryError(
                    "Saved model feature names do not match the runtime feature order."
                )
            if not artifact.get("fit_date") or int(artifact.get("training_observations", 0)) < 1:
                raise S3RepositoryError(
                    "Model metadata is missing valid training provenance."
                )
            if (
                not isinstance(filename, str)
                or not filename
                or "/" in filename
                or "\\" in filename
                or filename in {".", ".."}
            ):
                raise S3RepositoryError("Model metadata contains an unsafe filename.")
        except (KeyError, TypeError, ValueError) as error:
            raise S3ObjectNotFoundError(
                f"No {model_name} model is registered for {symbol} in S3 metadata."
            ) from error
        artifact_key = self.settings.model_prefix + filename
        model_bytes = self._read_bytes(artifact_key)
        actual_sha = hashlib.sha256(model_bytes).hexdigest()
        if not isinstance(expected_sha, str) or actual_sha != expected_sha:
            raise S3RepositoryError(
                f"Model artifact SHA-256 does not match S3 metadata: {artifact_key!r}."
            )
        return model_bytes, artifact, actual_sha

    def load_json(self, key: str) -> dict[str, Any]:
        try:
            value = json.loads(self._read_bytes(key))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise S3RepositoryError(f"S3 JSON object {key!r} is invalid.") from error
        if not isinstance(value, dict):
            raise S3RepositoryError(f"S3 JSON object {key!r} must contain an object.")
        return value

    def save_result(self, category: str, result: dict[str, Any]) -> str:
        import uuid

        from backend.app.schemas.responses import json_safe

        key = (
            self.settings.results_prefix
            + f"{category}/{uuid.uuid4().hex}.json"
        )
        body = json.dumps(
            json_safe(result),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        try:
            self._s3().put_object(
                Bucket=self.settings.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"403", "AccessDenied", "AllAccessDisabled"}:
                raise S3RepositoryError(
                    "S3 denied writing the analytics result."
                ) from error
            raise S3RepositoryError(
                f"S3 result write failed (error {code or 'ClientError'})."
            ) from error
        except (BotoCoreError, OSError) as error:
            raise S3RepositoryError(
                f"S3 result write failed ({type(error).__name__})."
            ) from error
        return key
