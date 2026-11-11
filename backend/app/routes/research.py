"""Curated research metadata routes; model and dataset objects stay private."""

from typing import Any

from backend.app.services.s3_repository import S3Repository


def get_experiments(repository: S3Repository) -> dict[str, Any]:
    return repository.load_json(repository.settings.experiments_key)


def get_models(repository: S3Repository) -> dict[str, Any]:
    metadata = repository.load_json(repository.settings.model_prefix + "model_metadata.json")
    artifacts = metadata.get("model_artifacts", {})
    return {
        "target_definition": metadata.get("target_definition"),
        "feature_names": metadata.get("feature_names_and_order", []),
        "training_date_range": metadata.get("training_date_range"),
        "validation_date_range": metadata.get("validation_date_range"),
        "test_date_range": metadata.get("test_date_range"),
        "models": [
            {"symbol": symbol, "name": name, **{k: v for k, v in artifact.items() if k != "filename"}}
            for symbol, entries in sorted(artifacts.items())
            for name, artifact in sorted(entries.items())
        ],
    }
