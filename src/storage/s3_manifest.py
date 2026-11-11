"""Safe local-to-S3 planning and upload-manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from src.storage.s3_config import S3Config

DIRECTORY_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("data/raw/nse", "bronze/nse"),
    ("data/processed/silver", "silver"),
    ("data/processed/analytics", "gold/risk-analytics"),
    ("data/processed/forecasting/statistical", "gold/statistical-forecasts"),
    ("data/processed/forecasting/ml", "gold/ml-forecasts"),
    ("data/processed/optimization", "gold/portfolio-optimization"),
    ("models/volatility", "models/volatility"),
)

_EXCLUDED_COMPONENTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".aws",
    ".ssh",
    ".cache",
    ".ruff_cache",
    "node_modules",
    "temp",
    "tmp",
    ".pytest_cache",
    ".mypy_cache",
}
_SECRET_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
_SECRET_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".ppk"}
_TEMP_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".part",
    ".partial",
    ".swp",
    ".swo",
    ".bak",
    ".log",
    ".lock",
    ".download",
    ".crdownload",
}


@dataclass(frozen=True)
class PlannedFile:
    """One filtered local file and its deterministic S3 destination."""

    relative_local_path: str
    s3_key: str
    size_bytes: int
    sha256: str


def sha256_file(path: str | Path) -> str:
    """Return a local file's full SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s3_key_for_path(relative_path: str | Path, prefix: str) -> str:
    """Join a safe project-relative path to an S3 prefix using POSIX separators."""
    raw_path = str(relative_path).replace("\\", "/")
    path = PurePosixPath(raw_path)
    normalized_prefix = prefix.strip("/")
    if (
        path.is_absolute()
        or raw_path.startswith("//")
        or re.match(r"^[A-Za-z]:/", raw_path)
        or not raw_path
        or any(
        part in {"", ".", ".."} for part in raw_path.split("/")
        )
    ):
        raise ValueError("S3 object path must be a normalized relative path.")
    if not normalized_prefix or any(
        part in {"", ".", ".."} for part in normalized_prefix.split("/")
    ):
        raise ValueError("S3 prefix must be a normalized relative prefix.")
    return f"{normalized_prefix}/{path.as_posix()}"


def should_exclude(relative_path: str | Path) -> bool:
    """Filter credentials, local environments, caches, and temporary files."""
    path = PurePosixPath(str(relative_path).replace("\\", "/"))
    components = {component.lower() for component in path.parts}
    filename = path.name.lower()
    suffix = path.suffix.lower()
    if filename.startswith("."):
        return True
    if components & _EXCLUDED_COMPONENTS:
        return True
    if filename in _SECRET_NAMES or any(
        word in filename for word in ("secret", "credential", "access_key")
    ):
        return True
    if filename.startswith(".env") or suffix in _SECRET_EXTENSIONS:
        return True
    if suffix in _TEMP_EXTENSIONS or filename.endswith("~"):
        return True
    if filename in {".ds_store", "thumbs.db"} or filename.startswith("~$"):
        return True
    return False


def build_upload_plan(
    project_root: str | Path,
    *,
    mappings: Iterable[tuple[str, str]] = DIRECTORY_MAPPINGS,
) -> tuple[list[PlannedFile], list[str]]:
    """Inventory only mapped directories, preserving relative paths and names."""
    root = Path(project_root).resolve()
    planned: list[PlannedFile] = []
    missing_sources: list[str] = []
    seen_keys: set[str] = set()
    for local_directory, prefix in mappings:
        source_root = (root / local_directory).resolve()
        if root not in source_root.parents:
            raise ValueError(f"Mapped source escapes the project root: {local_directory}")
        if not source_root.is_dir():
            missing_sources.append(local_directory.replace("\\", "/"))
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if should_exclude(relative):
                continue
            relative_source_path = path.relative_to(source_root)
            key = s3_key_for_path(relative_source_path, prefix)
            if key in seen_keys:
                raise ValueError(f"Multiple local files map to S3 key {key!r}.")
            seen_keys.add(key)
            planned.append(
                PlannedFile(
                    relative_local_path=relative.as_posix(),
                    s3_key=key,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
    planned.sort(key=lambda item: (item.s3_key, item.relative_local_path))
    return planned, missing_sources


def new_manifest(
    config: S3Config,
    files: Iterable[PlannedFile],
    *,
    status: str = "planned",
    missing_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable manifest without implying remote verification."""
    if status not in {"planned", "completed", "partial", "dry-run"}:
        raise ValueError("Unsupported manifest status.")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bucket": config.bucket,
        "region": config.region,
        "dry_run": config.dry_run,
        "missing_source_directories": sorted(set(missing_sources)),
        "files": [
            {
                **asdict(item),
                "bucket": config.bucket,
                "region": config.region,
                "upload_timestamp_utc": None,
                "remote_sha256": None,
                "remote_checksum_verified": False,
                "checksum_verification": "not_performed",
            }
            for item in files
        ],
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Create a manifest exclusively; never overwrite a local file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return destination


def default_manifest_path(project_root: str | Path) -> Path:
    """Return a timestamped path that avoids replacing earlier manifests."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(project_root) / "data" / "processed" / "s3_manifests" / f"upload-{stamp}.json"


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate schema essentials and ensure unverified checksums stay explicit."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported S3 manifest schema_version.")
    if not manifest.get("bucket") or not manifest.get("region"):
        raise ValueError("S3 manifest must include its bucket and region.")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("S3 manifest files must be a list.")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("S3 manifest file entries must be objects.")
        if not {
            "relative_local_path",
            "s3_key",
            "bucket",
            "region",
            "size_bytes",
            "sha256",
            "upload_timestamp_utc",
            "remote_checksum_verified",
        }.issubset(item):
            raise ValueError("S3 manifest file entry is missing required fields.")
        if not re.fullmatch(r"[a-f0-9]{64}", str(item["sha256"])):
            raise ValueError("S3 manifest contains an invalid local SHA-256 checksum.")
        if not isinstance(item["remote_checksum_verified"], bool):
            raise ValueError("S3 manifest remote_checksum_verified must be boolean.")
        remote_sha = item.get("remote_sha256")
        if remote_sha is not None and not re.fullmatch(r"[a-f0-9]{64}", str(remote_sha)):
            raise ValueError("S3 manifest contains an invalid remote SHA-256 checksum.")
        if item["remote_checksum_verified"] and remote_sha != item["sha256"]:
            raise ValueError("A checksum cannot be marked verified unless both SHA-256 values match.")
