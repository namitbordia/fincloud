"""API Gateway HTTP API payload v1/v2 handler and local dispatch entry point."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any

from backend.app.config import Settings
from backend.app.errors import (
    ForecastUnavailableError,
    OptimizationUnavailableError,
    RequestValidationError,
    S3ObjectNotFoundError,
    S3RepositoryError,
)
from backend.app.routes.forecasting import post_forecast
from backend.app.routes.correlation import post_correlation
from backend.app.routes.health import get_health
from backend.app.routes.optimization import post_optimization
from backend.app.routes.portfolio import post_risk
from backend.app.routes.research import get_experiments, get_models
from backend.app.routes.stocks import get_stock_history, get_stocks
from backend.app.schemas.requests import parse_object
from backend.app.schemas.responses import json_safe
from backend.app.services.s3_repository import S3Repository

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 1_000_000


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point for API Gateway HTTP API payload versions 1 and 2."""
    del context
    return handle_event(event)


def handle_event(
    event: dict[str, Any],
    repository: S3Repository | None = None,
) -> dict[str, Any]:
    settings = repository.settings if repository is not None else Settings.from_environment()
    origin = _header(event, "origin")
    headers = _response_headers(settings, origin)
    try:
        method, path = _request_target(event)
        if method == "OPTIONS":
            if origin and origin not in settings.cors_origins:
                return _response(403, {"error": "Origin is not allowed."}, headers)
            return _response(204, {}, headers)
        body = _request_body(event)
        active_repository = repository or S3Repository(settings)
        if method == "GET" and path == "/health":
            status, payload = get_health(
                active_repository,
                api_gateway_verified=bool(
                    event.get("requestContext", {}).get("apiId")
                ),
            )
            return _response(status, payload, headers)
        if method == "GET" and path == "/stocks":
            return _response(200, get_stocks(active_repository), headers)
        if method == "GET" and (match := re.fullmatch(r"/stocks/([^/]+)/history", path)):
            query = event.get("queryStringParameters") or {}
            return _response(200, get_stock_history(active_repository, match.group(1), query.get("start_date"), query.get("end_date")), headers)
        if method == "POST" and path in {"/portfolio/risk", "/portfolio/analyze"}:
            return _response(
                200, post_risk(parse_object(body), active_repository), headers
            )
        if method == "POST" and path == "/portfolio/correlation":
            return _response(200, post_correlation(parse_object(body), active_repository), headers)
        if method == "POST" and path in {"/forecast", "/ml/forecast"}:
            return _response(
                200, post_forecast(parse_object(body), active_repository), headers
            )
        if method == "GET" and path == "/research/experiments":
            return _response(200, get_experiments(active_repository), headers)
        if method == "GET" and path == "/research/models":
            return _response(200, get_models(active_repository), headers)
        if method == "POST" and path == "/portfolio/optimize":
            return _response(
                200,
                post_optimization(parse_object(body), active_repository),
                headers,
            )
        return _response(404, {"error": "Route not found."}, headers)
    except RequestValidationError as error:
        return _response(400, {"error": str(error)}, headers)
    except S3ObjectNotFoundError as error:
        return _response(503, {"error": str(error)}, headers)
    except S3RepositoryError as error:
        return _response(502, {"error": str(error)}, headers)
    except ForecastUnavailableError as error:
        return _response(422, {"error": str(error)}, headers)
    except OptimizationUnavailableError as error:
        return _response(422, {"error": str(error)}, headers)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return _response(400, {"error": f"Invalid JSON request ({type(error).__name__})."}, headers)
    except Exception:
        LOGGER.exception("Unhandled FinCloud API error.")
        return _response(500, {"error": "Internal server error."}, headers)


def _request_target(event: dict[str, Any]) -> tuple[str, str]:
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
    )
    path = event.get("rawPath") or event.get("path") or "/"
    stage = event.get("requestContext", {}).get("stage")
    if isinstance(stage, str) and stage not in {"", "$default"}:
        prefix = f"/{stage}"
        if path == prefix:
            path = "/"
        elif path.startswith(prefix + "/"):
            path = path[len(prefix) :]
    if not isinstance(method, str) or not isinstance(path, str):
        raise RequestValidationError("Unsupported API Gateway event shape.")
    return method.upper(), path.rstrip("/") or "/"


def _request_body(event: dict[str, Any]) -> Any:
    raw = event.get("body")
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise RequestValidationError("Request body must be JSON text.")
    try:
        data = (
            base64.b64decode(raw, validate=True)
            if event.get("isBase64Encoded")
            else raw.encode("utf-8")
        )
    except (ValueError, base64.binascii.Error) as error:
        raise RequestValidationError("Request body has invalid base64 encoding.") from error
    if len(data) > MAX_REQUEST_BYTES:
        raise RequestValidationError("Request body exceeds the 1 MB application limit.")
    try:
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise


def _header(event: dict[str, Any], name: str) -> str | None:
    headers = event.get("headers") or {}
    return next(
        (
            value
            for key, value in headers.items()
            if isinstance(key, str) and key.lower() == name
        ),
        None,
    )


def _response_headers(settings: Settings, origin: str | None) -> dict[str, str]:
    result = {
        "content-type": "application/json; charset=utf-8",
        "cache-control": "no-store",
        "vary": "Origin",
        "access-control-allow-methods": "GET,POST,OPTIONS",
        "access-control-allow-headers": "authorization,content-type",
        "access-control-max-age": "600",
    }
    if origin and origin in settings.cors_origins:
        result["access-control-allow-origin"] = origin
    return result


def _response(
    status_code: int,
    payload: Any,
    headers: dict[str, str],
) -> dict[str, Any]:
    safe = json_safe(payload)
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(safe, allow_nan=False, separators=(",", ":")),
        "isBase64Encoded": False,
    }
