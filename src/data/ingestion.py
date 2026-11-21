"""Transaction data ingestion module."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.config import get_config

config = get_config()


class TransactionDataLoader:
    """Load and validate transaction data from various sources."""

    REQUIRED_COLUMNS = [
        "transaction_id",
        "timestamp",
        "portfolio_id",
        "asset_class",
        "transaction_type",
        "transaction_amount",
        "currency",
        "counterparty_id",
    ]

    def __init__(self, raw_data_path: str | None = None):
        self.raw_data_path = Path(raw_data_path or config["data"]["raw_path"])

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load transaction data from a CSV file."""
        filepath = self.raw_data_path / filename
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        return self._validate(df)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load transaction data from a Parquet file."""
        filepath = self.raw_data_path / filename
        logger.info(f"Loading data from {filepath}")
        df = pd.read_parquet(filepath)
        return self._validate(df)

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accept an existing DataFrame and validate it."""
        return self._validate(df.copy())

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check required columns and basic data quality."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        null_counts = df[self.REQUIRED_COLUMNS].isnull().sum()
        nulls = null_counts[null_counts > 0]
        if not nulls.empty:
            logger.warning(f"Null values found:\n{nulls}")

        logger.info(
            f"Loaded {len(df):,} transactions "
            f"({df['timestamp'].min()} → {df['timestamp'].max()})"
        )
        return df

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """Persist processed data to Parquet."""
        out_dir = Path(config["data"]["processed_path"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved processed data → {out_path}")
        return out_path
