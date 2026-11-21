"""Risk scoring model for investment transactions."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from loguru import logger

from src.utils.config import get_config, get_settings

config = get_config()
settings = get_settings()


class RiskScorer:
    """XGBoost classifier that assigns a risk probability to each transaction."""

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
        "anomaly_score",
    ]
    TARGET_COL = "is_high_risk"

    def __init__(self):
        cfg = config["models"]["risk_scoring"]
        self.model = XGBClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            eval_metric=cfg.get("eval_metric", "auc"),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 20),
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        return df[available]

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COL) -> "RiskScorer":
        X = self._get_features(df)
        y = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Training risk scorer on {len(X_train):,} samples.")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self._is_fitted = True

        preds = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        logger.info(f"Risk scorer validation AUC: {auc:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with risk_score and risk_label columns."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._get_features(df)
        risk_scores = self.model.predict_proba(X)[:, 1]

        result = df.copy()
        result["risk_score"] = risk_scores
        result["risk_label"] = pd.cut(
            risk_scores,
            bins=[-np.inf, settings.risk_score_medium, settings.risk_score_high, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        logger.info(
            f"Risk distribution: {result['risk_label'].value_counts().to_dict()}"
        )
        return result

    def save(self, path: str | None = None) -> Path:
        out_path = Path(path or settings.model_registry_path) / "risk_scorer.joblib"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)
        logger.info(f"Risk scorer saved → {out_path}")
        return out_path

    @classmethod
    def load(cls, path: str | None = None) -> "RiskScorer":
        load_path = Path(path or get_settings().model_registry_path) / "risk_scorer.joblib"
        scorer = cls()
        scorer.model = joblib.load(load_path)
        scorer._is_fitted = True
        logger.info(f"Risk scorer loaded ← {load_path}")
        return scorer
