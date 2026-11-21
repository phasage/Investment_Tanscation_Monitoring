"""Unit tests for anomaly detector."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.anomaly_detector import AnomalyDetector
from src.features.engineering import TransactionFeatureEngineer


@pytest.fixture
def feature_df():
    n = 100
    rng = np.random.default_rng(1)
    base_time = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "transaction_id": [f"TX{i:04d}" for i in range(n)],
        "timestamp": [base_time + timedelta(hours=i) for i in range(n)],
        "portfolio_id": rng.choice(["P001", "P002"], n),
        "asset_class": "equity",
        "transaction_type": "BUY",
        "transaction_amount": rng.uniform(1_000, 100_000, n),
        "currency": "USD",
        "counterparty_id": rng.choice(["CP01", "CP02"], n),
    })
    return (
        TransactionFeatureEngineer(df)
        .add_time_features()
        .add_rolling_stats()
        .add_amount_zscore()
        .add_counterparty_risk()
        .build()
    )


def test_fit_predict(feature_df):
    detector = AnomalyDetector()
    detector.fit(feature_df)
    result = detector.predict(feature_df)
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert result["anomaly_score"].between(0, 1).all()
    assert result["is_anomaly"].isin([0, 1]).all()


def test_predict_before_fit_raises():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict(pd.DataFrame())


def test_anomaly_rate_reasonable(feature_df):
    detector = AnomalyDetector()
    detector.fit(feature_df)
    result = detector.predict(feature_df)
    rate = result["is_anomaly"].mean()
    # Anomaly rate should be under 20% on normal data
    assert rate < 0.20, f"Anomaly rate too high: {rate:.2%}"
