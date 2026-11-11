"""Local-only HTTP server backed by an in-memory mock S3 client."""

from __future__ import annotations

import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from backend.app.config import Settings
from backend.app.handler import handle_event
from backend.app.services.s3_repository import S3Repository

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalMockS3:
    """Explicit local mock; production handler never constructs this client."""

    def __init__(self) -> None:
        self.objects = {
            "silver/all_stocks.csv": (
                PROJECT_ROOT / "data/processed/silver/all_stocks.csv"
            ).read_bytes(),
            "silver/silver_dataset_manifest.json": (
                PROJECT_ROOT
                / "data/processed/silver/silver_dataset_manifest.json"
            ).read_bytes(),
            "models/volatility/model_metadata.json": (
                PROJECT_ROOT / "models/volatility/model_metadata.json"
            ).read_bytes(),
        }
        for path in (PROJECT_ROOT / "models/volatility").glob("*.joblib"):
            self.objects[f"models/volatility/{path.name}"] = path.read_bytes()
        self.writes: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        if Key not in self.objects:
            raise _missing_object(Key)
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        if Key not in self.objects:
            raise _missing_object(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> dict[str, Any]:
        del Bucket, ContentType
        self.writes[Key] = Body
        return {}


def _missing_object(key: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "NoSuchKey", "Message": f"Mock object missing: {key}"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "GetObject",
    )


def main() -> None:
    settings = Settings.from_environment()
    repository = S3Repository(settings, client=LocalMockS3())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self._dispatch()

        def do_OPTIONS(self) -> None:
            self._dispatch()

        def _dispatch(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else None
            event = {
                "version": "2.0",
                "rawPath": self.path.split("?", 1)[0],
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "requestContext": {
                    "http": {"method": self.command, "path": self.path}
                },
                "body": body,
            }
            response = handle_event(event, repository)
            encoded = response["body"].encode("utf-8")
            self.send_response(response["statusCode"])
            for key, value in response["headers"].items():
                self.send_header(key, value)
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"local-mock-api {self.address_string()} - {format % args}")

    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("FinCloud local mock API: http://127.0.0.1:8000")
    print("Mocked S3 data comes from local project files; AWS execution is not verified.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
