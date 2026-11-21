"""Unit tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineering import TransactionFeatureEngineer


@pytest.fixture
def sample_df():
    """Minimal valid transaction DataFrame."""
    n = 50
    rng = np.random.default_rng(0)
    base_time = datetime(2024, 1, 1)
    return pd.DataFrame({
        "transaction_id": [f"TX{i:04d}" for i in range(n)],
        "timestamp": [base_time + timedelta(hours=i) for i in range(n)],
        "portfolio_id": rng.choice(["P001", "P002", "P003"], n),
        "asset_class": rng.choice(["equity", "bond", "fx"], n),
        "transaction_type": rng.choice(["BUY", "SELL"], n),
        "transaction_amount": rng.uniform(1_000, 500_000, n),
        "currency": "USD",
        "counterparty_id": rng.choice(["CP01", "CP02", "CP03"], n),
    })


def test_time_features(sample_df):
    engineer = TransactionFeatureEngineer(sample_df)
    result = engineer.add_time_features().build()
    for col in ["hour_of_day", "day_of_week", "is_weekend", "month", "quarter"]:
        assert col in result.columns, f"Missing column: {col}"
    assert result["hour_of_day"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["is_weekend"].isin([0, 1]).all()


def test_rolling_stats(sample_df):
    result = (
        TransactionFeatureEngineer(sample_df)
        .add_rolling_stats(windows=[7])
        .build()
    )
    assert "rolling_mean_7d" in result.columns
    assert "rolling_std_7d" in result.columns
    assert result["rolling_mean_7d"].notna().all()


def test_amount_zscore(sample_df):
    result = TransactionFeatureEngineer(sample_df).add_amount_zscore().build()
    assert "amount_zscore" in result.columns
    assert result["amount_zscore"].notna().all()


def test_counterparty_risk(sample_df):
    result = TransactionFeatureEngineer(sample_df).add_counterparty_risk().build()
    assert "counterparty_risk_score" in result.columns
    assert result["counterparty_risk_score"].between(0, 1).all()


def test_chained_features(sample_df):
    result = (
        TransactionFeatureEngineer(sample_df)
        .add_time_features()
        .add_rolling_stats()
        .add_amount_zscore()
        .add_counterparty_risk()
        .build()
    )
    assert len(result) == len(sample_df)
    assert result.columns.tolist().__len__() > sample_df.columns.tolist().__len__()
