"""Stock-universe route."""

from typing import Any
from datetime import date

from backend.app.errors import RequestValidationError

from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository


def get_stocks(repository: S3Repository) -> dict[str, Any]:
    frame, metadata = repository.load_market_data()
    latest = frame.sort_values("date", kind="stable").groupby("symbol", sort=True).tail(2)
    prices = {}
    for symbol, rows in latest.groupby("symbol", sort=True):
        rows = rows.sort_values("date", kind="stable")
        current = float(rows.iloc[-1]["adj_close"])
        previous = float(rows.iloc[-2]["adj_close"]) if len(rows) > 1 else current
        prices[str(symbol)] = {
            "price": current,
            "daily_change_percent": ((current / previous) - 1.0) * 100.0 if previous else 0.0,
            "date": rows.iloc[-1]["date"].date().isoformat(),
        }
    return {
        "symbols": metadata["symbols"],
        "supported_stock_count": len(metadata["symbols"]),
        "stocks": prices,
        "history": {
            "start_date": metadata["start_date"],
            "end_date": metadata["end_date"],
            "row_count": metadata["row_count"],
        },
        "data_source": {
            "service": "Amazon S3" if repository.is_lambda else "Mocked S3",
            "key": metadata["key"],
            "manifest_key": metadata["manifest_key"],
            "sha256_verified": True,
        },
        "cloud_execution": cloud_execution(
            repository, repository.settings, data_verified=True
        ),
    }


def get_stock_history(
    repository: S3Repository,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    if not symbol:
        raise RequestValidationError("symbol must not be empty.")
    try:
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
    except ValueError as error:
        raise RequestValidationError("Dates must use YYYY-MM-DD format.") from error
    if start and end and start > end:
        raise RequestValidationError("start_date must not be after end_date.")
    frame, metadata = repository.load_market_data()
    rows = frame.loc[frame["symbol"] == symbol].copy()
    if rows.empty:
        raise RequestValidationError(f"Unsupported symbol: {symbol}.")
    if start:
        rows = rows.loc[rows["date"].dt.date >= start]
    if end:
        rows = rows.loc[rows["date"].dt.date <= end]
    history = [
        {
            "date": row.date.date().isoformat(),
            "adjusted_close": float(row.adj_close),
            "daily_return": float(row.daily_return),
            "volume": float(row.volume),
        }
        for row in rows.sort_values("date", kind="stable").itertuples()
    ]
    return {"symbol": symbol, "history": history, "count": len(history), "data_source": metadata["key"]}
