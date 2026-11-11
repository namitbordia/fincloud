"""Configuration for historical market data downloads."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_SYMBOLS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
}


@dataclass(frozen=True)
class IngestionConfig:
    """Date range, symbols, and local output settings for an ingestion run."""

    start_date: str
    end_date: str
    symbols: dict[str, str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    output_dir: Path = Path("data/raw/nse")
    report_path: Path = Path("data/processed/validation_report.json")
    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    write_parquet: bool = True

    def __post_init__(self) -> None:
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date.")
        if not self.symbols:
            raise ValueError("At least one stock symbol must be configured.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
