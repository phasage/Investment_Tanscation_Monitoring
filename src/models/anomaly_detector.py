"""Anomaly detection model for investment transactions."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.utils.config import get_config, get_settings

config = get_config()
settings = get_settings()


class AnomalyDetector:
    """Isolation Forest-based anomaly detection on transaction features."""

    FEATURE_COLS = [
        "transaction_amount",
        "amount_zscore",
        "rolling_mean_7d",
        "rolling_std_7d",
        "tx_freq_1h",
        "tx_freq_24h",
        "counterparty_risk_score",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
    ]

    def __init__(self):
        cfg = config["models"]["anomaly_detection"]
        self.model = IsolationForest(
            contamination=cfg.get("contamination", 0.05),
            n_estimators=cfg.get("n_estimators", 200),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.threshold = settings.anomaly_threshold
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        return df[available]

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = self._get_features(df)
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Training anomaly detector on {len(X):,} samples.")
        self.model.fit(X_scaled)
        self._is_fitted = True
        logger.info("Anomaly detector training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with anomaly_score and is_anomaly columns."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._get_features(df)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest: decision_function gives higher = more normal
        raw_scores = self.model.decision_function(X_scaled)
        # Normalize to 0–1 where 1 = most anomalous
        anomaly_scores = 1 - (
            (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        )

        result = df.copy()
        result["anomaly_score"] = anomaly_scores
        result["is_anomaly"] = (anomaly_scores >= self.threshold).astype(int)
        logger.info(
            f"Detected {result['is_anomaly'].sum():,} anomalies "
            f"out of {len(result):,} transactions."
        )
        return result

    def save(self, path: str | None = None) -> Path:
        out_path = Path(path or settings.model_registry_path) / "anomaly_detector.joblib"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, out_path)
        logger.info(f"Anomaly detector saved → {out_path}")
        return out_path

    @classmethod
    def load(cls, path: str | None = None) -> "AnomalyDetector":
        load_path = Path(path or get_settings().model_registry_path) / "anomaly_detector.joblib"
        artifact = joblib.load(load_path)
        detector = cls()
        detector.model = artifact["model"]
        detector.scaler = artifact["scaler"]
        detector._is_fitted = True
        logger.info(f"Anomaly detector loaded ← {load_path}")
        return detector
