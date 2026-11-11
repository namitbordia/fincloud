"""Offline tests for S3 configuration, planning, and mocked client operations."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from src.storage.s3_cli import main
from src.storage.s3_client import S3Storage, S3StorageError
from src.storage.s3_config import (
    DEFAULT_REGION,
    S3Config,
    S3ConfigurationError,
)
from src.storage.s3_manifest import (
    DIRECTORY_MAPPINGS,
    PlannedFile,
    build_upload_plan,
    new_manifest,
    sha256_file,
    should_exclude,
    s3_key_for_path,
    validate_manifest,
    write_manifest,
)


def client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "stubbed response"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "StubbedOperation",
    )


class TestS3Configuration(unittest.TestCase):
    def test_environment_configuration_and_dry_run(self) -> None:
        config = S3Config.from_environment(
            {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
            dry_run=True,
        )
        self.assertEqual(config.bucket, "fincloud-research-lake")
        self.assertEqual(config.region, DEFAULT_REGION)
        self.assertTrue(config.dry_run)

        configured = S3Config.from_environment(
            {
                "FINCLOUD_S3_BUCKET": "fincloud-research-lake",
                "FINCLOUD_AWS_REGION": "ap-south-1",
                "AWS_REGION": "us-east-1",
                "FINCLOUD_S3_DRY_RUN": "yes",
            }
        )
        self.assertEqual(configured.region, "ap-south-1")
        self.assertTrue(configured.dry_run)

    def test_missing_and_malformed_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(S3ConfigurationError, "FINCLOUD_S3_BUCKET"):
            S3Config.from_environment({})
        for bucket in ("Bad_Name", "ab", "192.168.1.1", "invalid..bucket"):
            with self.subTest(bucket=bucket), self.assertRaises(S3ConfigurationError):
                S3Config(bucket)
        with self.assertRaises(S3ConfigurationError):
            S3Config("valid-bucket-name", region="not a region")
        with self.assertRaises(S3ConfigurationError):
            S3Config.from_environment(
                {
                    "FINCLOUD_S3_BUCKET": "valid-bucket-name",
                    "FINCLOUD_S3_DRY_RUN": "sometimes",
                }
            )

    def test_s3_import_and_default_cli_do_not_create_clients(self) -> None:
        with patch("src.storage.s3_client.boto3.client") as factory:
            self.assertEqual(main([]), 0)
            factory.assert_not_called()


class TestS3PlanningAndManifests(unittest.TestCase):
    def test_mapping_preserves_paths_and_rejects_traversal(self) -> None:
        self.assertEqual(
            s3_key_for_path("plots\\model.png", "gold/ml-forecasts"),
            "gold/ml-forecasts/plots/model.png",
        )
        with self.assertRaises(ValueError):
            s3_key_for_path("../secrets.env", "bronze/nse")
        with self.assertRaises(ValueError):
            s3_key_for_path("prices.csv", "../outside")
        with self.assertRaises(ValueError):
            s3_key_for_path("C:\\private\\secret.txt", "bronze/nse")
        self.assertEqual(
            dict(DIRECTORY_MAPPINGS)["data/raw/nse"],
            "bronze/nse",
        )

    def test_file_filter_excludes_credentials_caches_and_temporary_files(self) -> None:
        excluded = (
            ".env",
            ".gitkeep",
            "secrets.json",
            "api_credentials.txt",
            "private.pem",
            ".venv/Lib/site-packages/pkg.py",
            "data/__pycache__/module.pyc",
            "output.csv.tmp",
            "application.log",
            "~$report.xlsx",
        )
        for path in excluded:
            with self.subTest(path=path):
                self.assertTrue(should_exclude(path))
        self.assertFalse(should_exclude("data/processed/silver/all_stocks.csv"))

    def test_plan_is_limited_to_mappings_and_records_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "RELIANCE.csv").write_bytes(b"synthetic;not actual data")
            (source / "credentials.json").write_text("secret", encoding="utf-8")
            (root / "unrelated.csv").write_text("ignore", encoding="utf-8")
            plan, missing = build_upload_plan(
                root,
                mappings=(("data/raw/nse", "bronze/nse"), ("models/volatility", "models/volatility")),
            )
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0].relative_local_path, "data/raw/nse/RELIANCE.csv")
            self.assertEqual(plan[0].s3_key, "bronze/nse/RELIANCE.csv")
            self.assertEqual(plan[0].size_bytes, len(b"synthetic;not actual data"))
            self.assertEqual(
                plan[0].sha256,
                hashlib.sha256(b"synthetic;not actual data").hexdigest(),
            )
            self.assertEqual(missing, ["models/volatility"])
            self.assertEqual(sha256_file(source / "RELIANCE.csv"), plan[0].sha256)

    def test_manifest_validation_and_exclusive_write(self) -> None:
        config = S3Config("fincloud-research-lake", dry_run=True)
        planned = PlannedFile(
            "data/raw/nse/sample.csv",
            "bronze/nse/sample.csv",
            4,
            hashlib.sha256(b"data").hexdigest(),
        )
        manifest = new_manifest(config, [planned], status="dry-run")
        validate_manifest(manifest)
        self.assertEqual(manifest["files"][0]["checksum_verification"], "not_performed")
        self.assertEqual(manifest["files"][0]["bucket"], config.bucket)
        self.assertEqual(manifest["files"][0]["region"], config.region)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "manifest.json"
            write_manifest(destination, manifest)
            self.assertEqual(json.loads(destination.read_text())["status"], "dry-run")
            with self.assertRaises(FileExistsError):
                write_manifest(destination, manifest)
        manifest["files"][0]["remote_sha256"] = "0" * 64
        manifest["files"][0]["remote_checksum_verified"] = True
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_cli_dry_run_creates_manifest_without_constructing_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "sample.csv").write_text("synthetic", encoding="utf-8")
            manifest_path = root / "plan.json"
            with (
                patch.dict(
                    "os.environ",
                    {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                    clear=True,
                ),
                patch("src.storage.s3_client.boto3.client") as factory,
            ):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "dry-run",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 0)
            factory.assert_not_called()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "dry-run")
            self.assertEqual(manifest["files"][0]["s3_key"], "bronze/nse/sample.csv")

    def test_environment_dry_run_cannot_be_overridden_by_upload_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "sample.csv").write_text("synthetic", encoding="utf-8")
            manifest_path = root / "upload-plan.json"
            with (
                patch.dict(
                    "os.environ",
                    {
                        "FINCLOUD_S3_BUCKET": "fincloud-research-lake",
                        "FINCLOUD_S3_DRY_RUN": "true",
                    },
                    clear=True,
                ),
                patch("src.storage.s3_client.boto3.client") as factory,
            ):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "upload",
                        "--confirm",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 0)
            factory.assert_not_called()
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["status"],
                "dry-run",
            )

    def test_upload_requires_explicit_confirmation_before_client_creation(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                clear=True,
            ),
            patch("src.storage.s3_client.boto3.client") as factory,
            self.assertRaises(SystemExit),
        ):
            main(["upload"])
        factory.assert_not_called()

    def test_inaccessible_bucket_writes_plan_manifest_without_uploading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "sample.csv").write_text("synthetic", encoding="utf-8")
            manifest_path = root / "upload-attempt.json"
            storage = MagicMock()
            storage.check_connection.return_value = {
                "bucket_accessible": False,
                "region_matches": False,
            }
            with (
                patch.dict(
                    "os.environ",
                    {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                    clear=True,
                ),
                patch("src.storage.s3_cli.S3Storage", return_value=storage),
            ):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "upload",
                        "--confirm",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["status"],
                "partial",
            )
            storage.upload_file.assert_not_called()

    def test_check_command_reports_read_only_connection_result(self) -> None:
        storage = MagicMock()
        storage.check_connection.return_value = {
            "account": "123456789012",
            "arn": "arn:aws:iam::123456789012:user/demo",
            "bucket": "fincloud-research-lake",
            "configured_region": "ap-south-1",
            "bucket_accessible": True,
            "bucket_region": "ap-south-1",
            "region_matches": True,
        }
        with (
            patch.dict(
                "os.environ",
                {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                clear=True,
            ),
            patch("src.storage.s3_cli.S3Storage", return_value=storage),
        ):
            self.assertEqual(main(["check"]), 0)
        storage.check_connection.assert_called_once_with()
        storage.upload_file.assert_not_called()
        storage.download_file.assert_not_called()

    def test_upload_stops_before_writing_when_bucket_region_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "sample.csv").write_text("synthetic", encoding="utf-8")
            storage = MagicMock()
            storage.check_connection.return_value = {
                "bucket_accessible": True,
                "region_matches": False,
            }
            manifest_path = root / "region-mismatch.json"
            with (
                patch.dict(
                    "os.environ",
                    {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                    clear=True,
                ),
                patch("src.storage.s3_cli.S3Storage", return_value=storage),
            ):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "upload",
                        "--confirm",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 2)
            storage.upload_file.assert_not_called()

    def test_existing_upload_manifest_is_rejected_before_client_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "raw" / "nse"
            source.mkdir(parents=True)
            (source / "sample.csv").write_text("synthetic", encoding="utf-8")
            manifest_path = root / "existing.json"
            manifest_path.write_text("keep", encoding="utf-8")
            with (
                patch.dict(
                    "os.environ",
                    {"FINCLOUD_S3_BUCKET": "fincloud-research-lake"},
                    clear=True,
                ),
                patch("src.storage.s3_cli.S3Storage") as storage_factory,
            ):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "upload",
                        "--confirm",
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "keep")
            storage_factory.assert_not_called()


class TestS3ClientWithStubbedOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.config = S3Config("fincloud-research-lake", region="ap-south-1")
        self.client = MagicMock()
        self.storage = S3Storage(self.config, client=self.client)

    def test_bucket_accessible_and_missing_bucket(self) -> None:
        self.client.head_bucket.return_value = {}
        self.assertTrue(self.storage.check_bucket_accessible())
        self.client.head_bucket.side_effect = client_error("404", 404)
        self.assertFalse(self.storage.check_bucket_accessible())

    def test_read_only_connection_check_reports_identity_access_and_region(self) -> None:
        identity = MagicMock()
        identity.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/demo",
        }
        self.client.head_bucket.return_value = {}
        self.client.get_bucket_location.return_value = {
            "LocationConstraint": "ap-south-1"
        }
        storage = S3Storage(
            self.config,
            client=self.client,
            identity_client=identity,
        )

        result = storage.check_connection()

        self.assertEqual(result["account"], "123456789012")
        self.assertEqual(result["arn"], "arn:aws:iam::123456789012:user/demo")
        self.assertTrue(result["bucket_accessible"])
        self.assertEqual(result["bucket_region"], "ap-south-1")
        self.assertTrue(result["region_matches"])
        identity.get_caller_identity.assert_called_once_with()
        self.client.head_bucket.assert_called_once_with(
            Bucket="fincloud-research-lake"
        )
        self.client.get_bucket_location.assert_called_once_with(
            Bucket="fincloud-research-lake"
        )

    def test_connection_check_reports_region_mismatch_without_writes(self) -> None:
        identity = MagicMock()
        identity.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/demo",
        }
        self.client.head_bucket.return_value = {}
        self.client.get_bucket_location.return_value = {
            "LocationConstraint": "us-east-1"
        }
        storage = S3Storage(
            self.config,
            client=self.client,
            identity_client=identity,
        )

        result = storage.check_connection()

        self.assertTrue(result["bucket_accessible"])
        self.assertFalse(result["region_matches"])
        self.client.put_object.assert_not_called()
        self.client.create_bucket.assert_not_called()

    def test_upload_uses_sse_and_no_acl_then_stream_verifies_remote_bytes(self) -> None:
        content = b"synthetic financial data"
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "sample.csv"
            source.write_bytes(content)
            self.client.get_object.return_value = {"Body": io.BytesIO(content)}
            uploaded = self.storage.upload_file(source, "silver/sample.csv")
            verification = self.storage.verify_local_checksum(
                source, "silver/sample.csv"
            )
        self.assertEqual(uploaded["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(uploaded["region"], "ap-south-1")
        self.assertTrue(verification["verified"])
        self.client.put_object.assert_called_once()
        upload_args = self.client.put_object.call_args.kwargs
        self.assertEqual(upload_args["ServerSideEncryption"], "AES256")
        self.assertEqual(upload_args["Metadata"]["sha256"], uploaded["sha256"])
        self.assertEqual(upload_args["IfNoneMatch"], "*")
        self.assertNotIn("ACL", upload_args)
        self.assertEqual(upload_args["ContentLength"], len(content))

    def test_upload_refuses_to_replace_an_existing_object(self) -> None:
        self.client.put_object.side_effect = client_error("PreconditionFailed", 412)
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "sample.csv"
            source.write_text("data", encoding="utf-8")
            with self.assertRaisesRegex(S3StorageError, "Refusing to overwrite"):
                self.storage.upload_file(source, "silver/sample.csv")
        self.client.put_object.assert_called_once()
        self.assertEqual(
            self.client.put_object.call_args.kwargs["IfNoneMatch"],
            "*",
        )

    def test_dry_run_upload_never_creates_client_or_contacts_s3(self) -> None:
        dry_storage = S3Storage(
            S3Config("fincloud-research-lake", dry_run=True)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "sample.csv"
            source.write_text("synthetic", encoding="utf-8")
            result = dry_storage.upload_file(source, "bronze/nse/sample.csv")
        self.assertTrue(result["dry_run"])
        self.assertIsNone(result["upload_timestamp_utc"])
        self.assertIsNone(dry_storage._client)

    def test_download_streams_bytes_without_overwriting_and_verifies_metadata(self) -> None:
        content = b"stubbed object bytes"
        checksum = hashlib.sha256(content).hexdigest()
        self.client.get_object.return_value = {
            "Body": io.BytesIO(content),
            "Metadata": {"sha256": checksum},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "download.csv"
            result = self.storage.download_file("silver/download.csv", target)
            self.assertEqual(target.read_bytes(), content)
            self.assertTrue(result["remote_checksum_verified"])
            with self.assertRaises(FileExistsError):
                self.storage.download_file("silver/download.csv", target)

    def test_missing_and_corrupt_downloads_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "missing.csv"
            self.client.get_object.side_effect = client_error("NoSuchKey", 404)
            with self.assertRaises(FileNotFoundError):
                self.storage.download_file("absent.csv", target)
            self.client.get_object.side_effect = None
            self.client.get_object.return_value = {
                "Body": io.BytesIO(b"wrong bytes"),
                "Metadata": {"sha256": "0" * 64},
            }
            with self.assertRaises(S3StorageError):
                self.storage.download_file("corrupt.csv", target)
            self.assertFalse(target.exists())

    def test_access_denied_and_network_errors_are_not_silenced(self) -> None:
        self.client.head_bucket.side_effect = client_error("AccessDenied", 403)
        with self.assertRaisesRegex(S3StorageError, "denied"):
            self.storage.check_bucket_accessible()
        self.client.head_bucket.side_effect = TimeoutError("network timeout")
        with self.assertRaisesRegex(S3StorageError, "TimeoutError"):
            self.storage.check_bucket_accessible()

    def test_list_and_metadata_do_not_treat_etag_as_sha256(self) -> None:
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "silver/a.csv", "Size": 12}]}
        ]
        self.client.get_paginator.return_value = paginator
        self.client.head_object.return_value = {
            "ContentLength": 12,
            "ETag": '"md5-like-etag"',
            "Metadata": {"sha256": "a" * 64},
            "ServerSideEncryption": "AES256",
        }
        listed = self.storage.list_objects("silver/")
        metadata = self.storage.get_object_metadata("silver/a.csv")
        self.assertEqual(listed[0]["key"], "silver/a.csv")
        self.assertEqual(metadata["etag"], '"md5-like-etag"')
        self.assertNotIn("sha256", metadata)
        self.assertEqual(metadata["metadata"]["sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
