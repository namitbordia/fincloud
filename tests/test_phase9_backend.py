"""Offline tests for the S3-backed Lambda API using a mocked S3 client."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from botocore.exceptions import ClientError

from backend.app.config import Settings
from backend.app.handler import handle_event, lambda_handler
from backend.app.services.s3_repository import S3Repository

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MockS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.writes: dict[str, bytes] = {}
        self.read_keys: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        self.read_keys.append(Key)
        if Key not in self.objects:
            raise self._missing(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        if Key not in self.objects:
            raise self._missing(Key)
        return {"ContentLength": len(self.objects[Key])}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> dict[str, object]:
        del Bucket, ContentType
        self.writes[Key] = Body
        return {}

    @staticmethod
    def _missing(key: str) -> ClientError:
        return ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": f"No mock object: {key}"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )


def s3_fixture() -> MockS3:
    silver = (
        PROJECT_ROOT / "data/processed/silver/all_stocks.csv"
    ).read_bytes()
    silver_manifest = (
        PROJECT_ROOT / "data/processed/silver/silver_dataset_manifest.json"
    ).read_bytes()
    metadata = (
        PROJECT_ROOT / "models/volatility/model_metadata.json"
    ).read_bytes()
    objects = {
        "silver/all_stocks.csv": silver,
        "silver/silver_dataset_manifest.json": silver_manifest,
        "models/volatility/model_metadata.json": metadata,
    }
    for model_path in (PROJECT_ROOT / "models/volatility").glob("*.joblib"):
        objects[f"models/volatility/{model_path.name}"] = model_path.read_bytes()
    return MockS3(objects)


class FinCloudBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            bucket="fincloud-test-bucket",
            region="ap-south-1",
            silver_key="silver/all_stocks.csv",
            silver_manifest_key="silver/silver_dataset_manifest.json",
            model_prefix="models/volatility/",
            results_prefix="gold/api-results/",
            cors_origins=("http://localhost:5173",),
        )
        self.client = s3_fixture()
        self.repository = S3Repository(self.settings, self.client)
        self.repository.is_lambda = False

    @staticmethod
    def event(
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        version: str = "2.0",
        headers: dict[str, str] | None = None,
        base64_body: bool = False,
    ) -> dict[str, object]:
        encoded_body = json.dumps(body) if body is not None else None
        if encoded_body and base64_body:
            encoded_body = base64.b64encode(encoded_body.encode()).decode()
        if version == "2.0":
            return {
                "version": version,
                "rawPath": path,
                "headers": headers or {},
                "requestContext": {
                    "http": {"method": method, "path": path},
                    "stage": "$default",
                },
                "body": encoded_body,
                "isBase64Encoded": base64_body,
            }
        return {
            "httpMethod": method,
            "path": path,
            "headers": headers or {},
            "body": encoded_body,
            "isBase64Encoded": base64_body,
        }

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[int, dict[str, object], dict[str, str]]:
        response = handle_event(
            self.event(method, path, body, **kwargs), self.repository
        )
        return (
            int(response["statusCode"]),
            json.loads(response["body"]),
            response["headers"],
        )

    def test_s3_repository_verifies_actual_dataset_and_manifest(self) -> None:
        frame, metadata = self.repository.load_market_data()
        self.assertEqual(len(frame), 6190)
        self.assertEqual(metadata["symbols"], [
            "HDFCBANK", "ICICIBANK", "INFY", "RELIANCE", "TCS"
        ])
        self.assertEqual(
            hashlib.sha256(self.client.objects["silver/all_stocks.csv"]).hexdigest(),
            metadata["sha256"],
        )

    def test_missing_s3_objects_are_explicit_and_never_use_local_fallback(self) -> None:
        self.client.objects.pop("silver/all_stocks.csv")
        status, response, _ = self.request("GET", "/stocks")
        self.assertEqual(status, 503)
        self.assertIn("s3://fincloud-test-bucket/silver/all_stocks.csv", response["error"])
        self.assertEqual(self.client.read_keys, ["silver/all_stocks.csv"])

    def test_bad_dataset_checksum_is_rejected(self) -> None:
        manifest = json.loads(self.client.objects["silver/silver_dataset_manifest.json"])
        manifest["output_sha256"]["all_stocks.csv"] = "0" * 64
        self.client.objects["silver/silver_dataset_manifest.json"] = json.dumps(
            manifest
        ).encode()
        status, response, _ = self.request("GET", "/stocks")
        self.assertEqual(status, 502)
        self.assertIn("SHA-256", response["error"])

    def test_health_checks_s3_but_does_not_claim_lambda_execution_locally(self) -> None:
        status, response, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(response["s3_connectivity"], "verified")
        self.assertFalse(response["cloud_execution"]["verified"])
        self.assertEqual(response["cloud_execution"]["compute"], "Local backend")

    def test_injected_s3_client_never_claims_lambda_execution(self) -> None:
        with patch.dict("os.environ", {"AWS_LAMBDA_FUNCTION_NAME": "fincloud-api"}):
            repository = S3Repository(self.settings, self.client)
        self.assertFalse(repository.is_lambda)

    def test_risk_api_uses_real_functions_and_persists_result(self) -> None:
        body = {
            "symbols": ["RELIANCE", "TCS"],
            "weights": {"RELIANCE": 0.6, "TCS": 0.4},
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
        }
        status, response, _ = self.request("POST", "/portfolio/risk", body)
        self.assertEqual(status, 200, response)
        metrics = response["metrics"]
        self.assertGreater(metrics["observation_count"], 400)
        self.assertTrue(np.isfinite(metrics["total_return"]))
        self.assertIn("sortino_ratio", metrics)
        self.assertEqual(response["correlation"]["missing_values_filled"], False)
        self.assertGreater(len(response["daily_returns"]), 400)
        self.assertIn("gold/api-results/risk/", response["result_key"])
        self.assertTrue(response["result_persisted"])
        self.assertEqual(len(self.client.writes), 1)

    def test_risk_api_rejects_bad_weight_total_and_symbols(self) -> None:
        body = {
            "symbols": ["RELIANCE", "TCS"],
            "weights": {"RELIANCE": 0.8, "TCS": 0.3},
        }
        status, response, _ = self.request("POST", "/portfolio/risk", body)
        self.assertEqual(status, 400)
        self.assertIn("sum to 1", response["error"])
        body["weights"] = {"RELIANCE": 1.0, "NOPE": 0.0}
        status, response, _ = self.request("POST", "/portfolio/risk", body)
        self.assertEqual(status, 400)
        self.assertIn("exactly", response["error"])

    def test_ridge_forecast_loads_verified_real_artifact_from_mock_s3(self) -> None:
        status, response, _ = self.request(
            "POST", "/forecast", {"symbol": "TCS", "model": "ridge"}
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(response["model"], "ridge")
        self.assertGreaterEqual(
            response["forecasted_annualized_volatility"], 0
        )
        self.assertEqual(response["forecast_horizon_sessions"], 20)
        self.assertTrue(response["model_metadata"]["features_verified"])
        self.assertEqual(response["model_metadata"]["feature_count"], 22)
        self.assertEqual(
            response["model_metadata"]["target_definition"],
            "sqrt((trading_days_per_year / horizon) * "
            "sum(next horizon daily_return squared))",
        )
        self.assertIn(
            "models/volatility/TCS_ridge.joblib",
            self.client.read_keys,
        )
        self.assertFalse(response["cloud_execution"]["verified"])
        self.assertIsNone(response["evaluation_metrics"])

    def test_ewma_statistical_baseline_runs_without_model_artifact(self) -> None:
        status, response, _ = self.request(
            "POST", "/forecast", {"symbol": "INFY", "model": "ewma"}
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(response["model"], "ewma")
        self.assertGreater(response["forecasted_annualized_volatility"], 0)
        self.assertNotIn(
            "models/volatility/INFY_ridge.joblib", self.client.read_keys
        )

    def test_missing_model_artifact_does_not_fall_back_to_local_model(self) -> None:
        self.client.objects.pop("models/volatility/TCS_ridge.joblib")
        status, response, _ = self.request(
            "POST", "/forecast", {"symbol": "TCS", "model": "ridge"}
        )
        self.assertEqual(status, 503)
        self.assertIn("models/volatility/TCS_ridge.joblib", response["error"])

    def test_optimization_api_uses_existing_solver_and_compares_strategies(self) -> None:
        status, response, _ = self.request(
            "POST",
            "/portfolio/optimize",
            {
                "symbols": ["RELIANCE", "TCS", "INFY"],
                "strategy": "compare",
                "maximum_weight": 0.8,
                "start_date": "2022-01-01",
                "end_date": "2024-12-31",
            },
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(
            [row["strategy"] for row in response["strategies"]],
            ["minimum_variance", "maximum_sharpe", "cvar_minimizing"],
        )
        for result in response["strategies"]:
            self.assertAlmostEqual(sum(result["weights"].values()), 1.0, places=6)
            self.assertTrue(np.isfinite(result["annualized_volatility"]))
            self.assertIn(result["solver"], {"SLSQP", "HiGHS linear programming"})
        self.assertTrue(response["comparison"])
        self.assertIn("gold/api-results/optimization/", response["result_key"])

    def test_history_correlation_and_alias_routes_use_s3_silver(self) -> None:
        event = self.event("GET", "/stocks/RELIANCE/history")
        event["queryStringParameters"] = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        history_response = handle_event(event, self.repository)
        self.assertEqual(history_response["statusCode"], 200)
        history = json.loads(history_response["body"])
        self.assertGreater(history["count"], 0)
        self.assertLessEqual(history["history"][-1]["date"], "2024-01-31")

        correlation_response = handle_event(self.event(
            "POST", "/portfolio/correlation", {"symbols": ["RELIANCE", "TCS"]}
        ), self.repository)
        self.assertEqual(correlation_response["statusCode"], 200)
        self.assertEqual(len(json.loads(correlation_response["body"])["correlation"]), 2)

        alias_response = handle_event(self.event(
            "POST", "/portfolio/analyze", {"symbols": ["RELIANCE", "TCS"], "weights": {"RELIANCE": 0.5, "TCS": 0.5}}
        ), self.repository)
        self.assertEqual(alias_response["statusCode"], 200)

    def test_research_routes_return_curated_metadata_not_artifact_bytes(self) -> None:
        self.client.objects["experiments/ml/fincloud_research_summary.json"] = b'{"models":[],"data_range":{"start":"2020-01-01","end":"2024-12-31"}}'
        status, response, _ = self.request("GET", "/research/experiments")
        self.assertEqual(status, 200)
        self.assertEqual(response["data_range"]["start"], "2020-01-01")
        status, response, _ = self.request("GET", "/research/models")
        self.assertEqual(status, 200)
        self.assertGreater(len(response["models"]), 0)
        self.assertNotIn("filename", response["models"][0])

    def test_random_forest_model_artifact_runs_live_forecast(self) -> None:
        status, response, _ = self.request(
            "POST", "/ml/forecast", {"symbol": "RELIANCE", "model": "random_forest"}
        )
        self.assertEqual(status, 200, response)
        self.assertGreater(response["forecasted_annualized_volatility"], 0)
        self.assertTrue(response["cloud_execution"]["model_source"] is None or not response["cloud_execution"]["verified"])

    def test_http_api_v1_base64_and_stage_prefix_are_supported(self) -> None:
        event = self.event(
            "POST",
            "/prod/portfolio/risk",
            {
                "symbols": ["RELIANCE", "TCS"],
                "weights": {"RELIANCE": 0.5, "TCS": 0.5},
            },
            version="1.0",
            base64_body=True,
        )
        event["requestContext"] = {"stage": "prod"}
        response = handle_event(event, self.repository)
        self.assertEqual(response["statusCode"], 200)

    def test_response_format_cors_and_non_finite_json_are_safe(self) -> None:
        status, response, headers = self.request(
            "GET",
            "/stocks",
            headers={"Origin": "http://localhost:5173"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["access-control-allow-origin"], "http://localhost:5173")
        self.assertEqual(response["supported_stock_count"], 5)
        denied = handle_event(
            self.event(
                "OPTIONS",
                "/stocks",
                headers={"origin": "https://not-allowed.example"},
            ),
            self.repository,
        )
        self.assertEqual(denied["statusCode"], 403)
        invalid = handle_event(
            self.event("POST", "/portfolio/risk", {"invalid": float("nan")}),
            self.repository,
        )
        self.assertEqual(invalid["statusCode"], 400)
        self.assertIsNotNone(lambda_handler)

    def test_lambda_handler_parses_api_gateway_payload(self) -> None:
        event = self.event("GET", "/health")
        with (
            patch(
                "backend.app.handler.Settings.from_environment",
                return_value=self.settings,
            ),
            patch(
                "backend.app.handler.S3Repository",
                return_value=self.repository,
            ),
        ):
            response = lambda_handler(event, object())
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])["s3_connectivity"], "verified")


if __name__ == "__main__":
    unittest.main()
