"""Real-time and batch inference pipeline."""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.features.engineering import TransactionFeatureEngineer
from src.models.anomaly_detector import AnomalyDetector
from src.models.risk_scorer import RiskScorer
from src.utils.config import get_settings

settings = get_settings()


class InferencePipeline:
    """Score incoming transactions for anomaly and risk."""

    def __init__(self):
        logger.info("Loading models for inference...")
        self.anomaly_detector = AnomalyDetector.load()
        self.risk_scorer = RiskScorer.load()
        logger.info("Inference pipeline ready.")

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of transactions.
        Returns original columns + anomaly_score, is_anomaly, risk_score, risk_label.
        """
        engineer = (
            TransactionFeatureEngineer(df)
            .add_time_features()
            .add_rolling_stats()
            .add_amount_zscore()
            .add_counterparty_risk()
        )
        df_features = engineer.build()

        df_features = self.anomaly_detector.predict(df_features)
        df_features = self.risk_scorer.predict(df_features)

        logger.info(
            f"Scored {len(df_features):,} transactions | "
            f"Anomalies: {df_features['is_anomaly'].sum()} | "
            f"High risk: {(df_features['risk_label'] == 'HIGH').sum()}"
        )
        return df_features

    def score_single(self, transaction: dict) -> dict:
        """Score a single transaction record."""
        df = pd.DataFrame([transaction])
        result = self.score(df)
        return result.iloc[0].to_dict()
