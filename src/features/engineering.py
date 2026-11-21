"""Feature engineering for investment transaction data."""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


class TransactionFeatureEngineer:
    """Derive features from raw transaction records."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("timestamp").copy()

    def add_time_features(self) -> "TransactionFeatureEngineer":
        """Extract temporal features from the timestamp."""
        ts = self.df["timestamp"]
        self.df["hour_of_day"] = ts.dt.hour
        self.df["day_of_week"] = ts.dt.dayofweek
        self.df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        self.df["month"] = ts.dt.month
        self.df["quarter"] = ts.dt.quarter
        logger.debug("Added time features.")
        return self

    def add_rolling_stats(
        self, windows: list[int] = [7, 30], group_col: str = "portfolio_id"
    ) -> "TransactionFeatureEngineer":
        """Rolling mean and std of transaction amount per portfolio."""
        self.df = self.df.sort_values(["portfolio_id", "timestamp"])
        for w in windows:
            col_mean = f"rolling_mean_{w}d"
            col_std = f"rolling_std_{w}d"
            rolled = (
                self.df.groupby(group_col)["transaction_amount"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            self.df[col_mean] = rolled
            self.df[col_std] = (
                self.df.groupby(group_col)["transaction_amount"]
                .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )
        logger.debug(f"Added rolling stats for windows {windows}.")
        return self

    def add_amount_zscore(self) -> "TransactionFeatureEngineer":
        """Z-score of transaction amount per portfolio."""
        grp = self.df.groupby("portfolio_id")["transaction_amount"]
        self.df["amount_zscore"] = (
            (self.df["transaction_amount"] - grp.transform("mean"))
            / grp.transform("std").replace(0, np.nan)
        ).fillna(0)
        logger.debug("Added amount z-score.")
        return self

    def add_transaction_frequency(
        self, windows_hours: list[int] = [1, 24]
    ) -> "TransactionFeatureEngineer":
        """Count of transactions per portfolio in rolling time windows."""
        self.df = self.df.set_index("timestamp")
        for h in windows_hours:
            col = f"tx_freq_{h}h"
            self.df[col] = (
                self.df.groupby("portfolio_id")["transaction_amount"]
                .transform(lambda x: x.rolling(f"{h}h").count())
            )
        self.df = self.df.reset_index()
        logger.debug(f"Added transaction frequency for windows {windows_hours}h.")
        return self

    def add_counterparty_risk(
        self, risk_map: dict | None = None
    ) -> "TransactionFeatureEngineer":
        """Map counterparty IDs to risk scores (0–1)."""
        if risk_map is None:
            # Placeholder: assign random scores; replace with real lookup
            unique = self.df["counterparty_id"].unique()
            rng = np.random.default_rng(42)
            risk_map = {cp: rng.random() for cp in unique}
        self.df["counterparty_risk_score"] = (
            self.df["counterparty_id"].map(risk_map).fillna(0.5)
        )
        logger.debug("Added counterparty risk scores.")
        return self

    def build(self) -> pd.DataFrame:
        """Return the enriched DataFrame."""
        logger.info(f"Feature set built: {len(self.df.columns)} columns.")
        return self.df
