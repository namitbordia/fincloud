"""Offline tests for historical market data validation."""

import unittest

import pandas as pd

from src.ingestion.validators import validate_stock_data


def valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["RELIANCE", "RELIANCE"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "adj_close": [102.0, 103.0],
            "volume": [1000, 1200],
        }
    )


class TestMarketDataValidators(unittest.TestCase):
    def test_valid_stock_data_passes_all_checks(self) -> None:
        result = validate_stock_data(valid_data(), "RELIANCE")

        self.assertTrue(result["valid"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["issues"], [])

    def test_missing_values_are_reported(self) -> None:
        data = valid_data()
        data.loc[0, "close"] = None

        result = validate_stock_data(data)

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["checks"]["missing_values"]["counts_by_column"]["close"], 1
        )

    def test_duplicate_dates_are_reported_without_removing_rows(self) -> None:
        data = valid_data()
        data.loc[1, "date"] = data.loc[0, "date"]

        result = validate_stock_data(data)

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(
            result["checks"]["duplicate_dates"]["duplicate_rows"], 2
        )

    def test_zero_and_negative_prices_are_reported(self) -> None:
        data = valid_data()
        data.loc[0, "open"] = 0
        data.loc[1, "close"] = -1

        result = validate_stock_data(data)

        self.assertEqual(
            result["checks"]["positive_prices"]["invalid_counts_by_column"],
            {"open": 1, "close": 1},
        )

    def test_invalid_ohlc_relationship_is_reported(self) -> None:
        data = valid_data()
        data.loc[0, "high"] = 98.0

        result = validate_stock_data(data)

        self.assertEqual(result["checks"]["ohlc_relationships"]["invalid_rows"], 1)

    def test_unsorted_dates_and_negative_volume_are_reported(self) -> None:
        data = valid_data().iloc[::-1].reset_index(drop=True)
        data.loc[0, "volume"] = -1

        result = validate_stock_data(data)

        self.assertFalse(result["checks"]["date_sorting"]["passed"])
        self.assertEqual(result["checks"]["non_negative_volume"]["invalid_rows"], 1)

    def test_missing_required_column_is_reported(self) -> None:
        result = validate_stock_data(valid_data().drop(columns=["volume"]))

        self.assertIn(
            "volume",
            result["checks"]["required_columns"]["missing"],
        )


if __name__ == "__main__":
    unittest.main()
