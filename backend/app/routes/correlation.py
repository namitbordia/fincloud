"""Portfolio return correlation from the verified Silver market dataset."""

from typing import Any

import numpy as np

from backend.app.errors import RequestValidationError
from backend.app.schemas.requests import parse_dates, parse_symbols
from backend.app.services.execution import cloud_execution
from backend.app.services.s3_repository import S3Repository


def post_correlation(body: dict[str, Any], repository: S3Repository) -> dict[str, Any]:
    symbols = parse_symbols(body.get("symbols"))
    if len(symbols) < 2:
        raise RequestValidationError("At least two symbols are required for correlation.")
    start, end = parse_dates(body)
    frame, metadata = repository.load_market_data()
    missing = sorted(set(symbols) - set(metadata["symbols"]))
    if missing:
        raise RequestValidationError(f"Unsupported symbols: {missing}.")
    selected = frame.loc[frame["symbol"].isin(symbols)].copy()
    if start:
        selected = selected.loc[selected["date"].dt.date >= start]
    if end:
        selected = selected.loc[selected["date"].dt.date <= end]
    matrix = selected.pivot(index="date", columns="symbol", values="daily_return").reindex(columns=symbols).dropna()
    if len(matrix) < 2:
        raise RequestValidationError("At least two complete observations are required.")
    values = matrix.corr().to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    result = {
        "symbols": symbols,
        "correlation": values.tolist(),
        "observations": int(len(matrix)),
        "period": {"start_date": matrix.index.min().date().isoformat(), "end_date": matrix.index.max().date().isoformat()},
        "data_source": metadata["key"],
        "cloud_execution": cloud_execution(repository, repository.settings, data_verified=True),
    }
    return result
