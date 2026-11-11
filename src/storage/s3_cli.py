"""Explicit command-line interface for the optional S3 data lake."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.storage.s3_client import S3Storage, S3StorageError
from src.storage.s3_config import S3Config, S3ConfigurationError
from src.storage.s3_manifest import (
    PlannedFile,
    build_upload_plan,
    default_manifest_path,
    new_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or explicitly operate FinCloud's configured S3 data lake."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("config", help="Display validated environment configuration.")
    subparsers.add_parser(
        "check",
        help="Read-only check of AWS identity, bucket access, and bucket region.",
    )
    subparsers.add_parser("check-bucket", help="Explicitly check bucket accessibility.")
    subparsers.add_parser("plan", help="List eligible mapped files without contacting AWS.")
    dry_run_parser = subparsers.add_parser(
        "dry-run", help="Plan and write a local manifest without contacting AWS."
    )
    dry_run_parser.add_argument("--manifest", type=Path)
    upload_parser = subparsers.add_parser(
        "upload", help="Upload mapped files; requires --confirm."
    )
    upload_parser.add_argument("--confirm", action="store_true")
    upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create a local upload plan and manifest without contacting AWS.",
    )
    upload_parser.add_argument("--manifest", type=Path)
    list_parser = subparsers.add_parser("list", help="List remote objects.")
    list_parser.add_argument("--prefix", default="")
    download_parser = subparsers.add_parser(
        "download", help="Download one object; requires --confirm."
    )
    download_parser.add_argument("--key", required=True)
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument("--confirm", action="store_true")
    verify_parser = subparsers.add_parser(
        "verify", help="Compare local and remotely streamed SHA-256 checksums."
    )
    verify_parser.add_argument("--key", required=True)
    verify_parser.add_argument("--file", type=Path, required=True)
    return parser


def _plan(root: Path) -> tuple[list[PlannedFile], list[str]]:
    files, missing = build_upload_plan(root)
    print(f"Eligible files: {len(files)}")
    print(f"Missing mapped directories: {len(missing)}")
    for item in files:
        print(
            f"{item.relative_local_path} -> {item.s3_key} "
            f"({item.size_bytes} bytes, sha256={item.sha256})"
        )
    if missing:
        print("Missing sources: " + ", ".join(missing))
    return files, missing


def _manifest_path(root: Path, configured: Path | None) -> Path:
    return configured if configured is not None else default_manifest_path(root)


def _save_local_plan(
    root: Path,
    config: S3Config,
    *,
    status: str,
    manifest_path: Path | None,
) -> Path:
    files, missing = _plan(root)
    manifest = new_manifest(config, files, status=status, missing_sources=missing)
    validate_manifest(manifest)
    destination = write_manifest(
        _manifest_path(root, manifest_path),
        manifest,
    )
    print(f"Local manifest: {destination.resolve()}")
    return destination


def _run_upload(
    root: Path,
    config: S3Config,
    *,
    manifest_path: Path | None,
) -> int:
    files, missing = build_upload_plan(root)
    manifest = new_manifest(
        config,
        files,
        status="completed",
        missing_sources=missing,
    )
    if not files:
        raise S3StorageError("No eligible mapped files were found; no AWS request was made.")
    destination = _manifest_path(root, manifest_path)
    if destination.exists():
        raise FileExistsError(
            f"Manifest destination already exists; refusing to overwrite it: {destination}"
        )
    storage = S3Storage(config)
    try:
        connection = storage.check_connection()
        if not connection["bucket_accessible"]:
            raise S3StorageError(
                "The configured bucket is missing or inaccessible; no bucket was created."
            )
        if not connection["region_matches"]:
            raise S3StorageError(
                "The configured region does not match the bucket's actual region; "
                "no files were uploaded."
            )
    except S3StorageError as error:
        manifest["status"] = "partial"
        manifest["error"] = str(error)
        validate_manifest(manifest)
        write_manifest(destination, manifest)
        print(f"Local manifest: {destination.resolve()}")
        raise
    failures: list[str] = []
    root_resolved = root.resolve()
    for entry, planned in zip(manifest["files"], files, strict=True):
        source = root_resolved / Path(planned.relative_local_path)
        try:
            if (
                source.stat().st_size != planned.size_bytes
                or sha256_file(source) != planned.sha256
            ):
                raise S3StorageError(
                    "Local file changed after planning; refusing to upload it."
                )
            uploaded = storage.upload_file(source, planned.s3_key)
            entry["upload_timestamp_utc"] = uploaded["upload_timestamp_utc"]
            entry["bucket"] = uploaded["bucket"]
            entry["region"] = uploaded["region"]
            entry["size_bytes"] = uploaded["size_bytes"]
            if uploaded["sha256"] != planned.sha256:
                raise S3StorageError(
                    "Local file changed during upload; the object is not marked verified."
                )
            verification = storage.verify_local_checksum(source, planned.s3_key)
            entry["remote_sha256"] = verification["remote_sha256"]
            entry["remote_checksum_verified"] = verification["verified"]
            entry["checksum_verification"] = verification["verification_method"]
            if not verification["verified"]:
                entry["checksum_verification"] = "mismatch"
                entry["error"] = "Remote bytes do not match the local SHA-256."
                failures.append(planned.s3_key)
            else:
                print(f"Uploaded and verified: {planned.s3_key}")
        except (S3StorageError, OSError, ValueError) as error:
            entry["checksum_verification"] = "upload_or_verification_failed"
            entry["error"] = str(error)
            failures.append(planned.s3_key)
            LOGGER.error("S3 operation failed for mapped object %s: %s", planned.s3_key, error)
    if failures:
        manifest["status"] = "partial"
        manifest["failed_keys"] = failures
    validate_manifest(manifest)
    write_manifest(destination, manifest)
    print(f"Local manifest: {destination.resolve()}")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        return 1
    print(f"Uploaded and verified files: {len(files)}")
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "plan":
        try:
            _plan(args.project_root)
            return 0
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
    try:
        cli_dry_run = bool(
            getattr(args, "dry_run", False) or args.command == "dry-run"
        )
        config = S3Config.from_environment(
            dry_run=True if cli_dry_run else None
        )
        if args.command == "config":
            print(
                json.dumps(
                    {
                        "bucket": config.bucket,
                        "region": config.region,
                        "dry_run": config.dry_run,
                        "credential_source": "boto3 standard credential provider chain",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "dry-run" or (
            args.command == "upload" and config.dry_run
        ):
            _save_local_plan(
                args.project_root,
                config,
                status="dry-run",
                manifest_path=args.manifest,
            )
            print("Dry run complete; no AWS client was created and no data was uploaded.")
            return 0
        if args.command == "upload":
            if not args.confirm:
                parser.error("upload requires --confirm; use `dry-run` to inspect first.")
            return _run_upload(
                args.project_root,
                config,
                manifest_path=args.manifest,
            )
        storage = S3Storage(config)
        if args.command == "check":
            result = storage.check_connection()
            print(json.dumps(result, indent=2))
            return 0 if result["bucket_accessible"] and result["region_matches"] else 1
        if args.command == "check-bucket":
            accessible = storage.check_bucket_accessible()
            print("Bucket accessible." if accessible else "Bucket not found.")
            return 0 if accessible else 1
        if args.command == "list":
            for item in storage.list_objects(args.prefix):
                print(f"{item['key']} ({item['size_bytes']} bytes)")
            return 0
        if args.command == "download":
            if not args.confirm:
                parser.error("download requires --confirm.")
            result = storage.download_file(args.key, args.destination)
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.command == "verify":
            result = storage.verify_local_checksum(args.file, args.key)
            print(json.dumps(result, indent=2))
            return 0 if result["verified"] else 1
        parser.error("Unknown command.")
    except (S3ConfigurationError, S3StorageError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
