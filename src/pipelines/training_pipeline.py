"""End-to-end training pipeline."""
from __future__ import annotations

from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger

from src.data.ingestion import TransactionDataLoader
from src.features.engineering import TransactionFeatureEngineer
from src.models.anomaly_detector import AnomalyDetector
from src.models.risk_scorer import RiskScorer
from src.utils.config import get_config, get_settings

config = get_config()
settings = get_settings()


def run_training_pipeline(raw_file: str, target_col: str = "is_high_risk") -> dict:
    """
    Full training pipeline:
    1. Ingest raw transaction data
    2. Engineer features
    3. Train anomaly detector
    4. Train risk scorer
    5. Log artifacts to MLflow
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="training_pipeline") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # --- 1. Ingest ---
        loader = TransactionDataLoader()
        df = loader.load_csv(raw_file)

        # --- 2. Feature Engineering ---
        engineer = (
            TransactionFeatureEngineer(df)
            .add_time_features()
            .add_rolling_stats()
            .add_amount_zscore()
            .add_counterparty_risk()
        )
        try:
            engineer.add_transaction_frequency()
        except Exception as e:
            logger.warning(f"Skipped tx frequency features: {e}")

        df_features = engineer.build()

        processed_path = loader.save_processed(df_features, "features.parquet")

        # --- 3. Anomaly Detection ---
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(df_features)
        df_features = anomaly_detector.predict(df_features)
        anomaly_path = anomaly_detector.save()
        mlflow.log_artifact(str(anomaly_path))

        anomaly_rate = df_features["is_anomaly"].mean()
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        logger.info(f"Anomaly rate: {anomaly_rate:.2%}")

        # --- 4. Risk Scoring (requires labelled data) ---
        artifacts = {}
        if target_col in df_features.columns:
            risk_scorer = RiskScorer()
            risk_scorer.fit(df_features, target_col=target_col)
            df_features = risk_scorer.predict(df_features)
            risk_path = risk_scorer.save()
            mlflow.log_artifact(str(risk_path))
            artifacts["risk_scorer"] = str(risk_path)
        else:
            logger.warning(
                f"Column '{target_col}' not found. Skipping risk scorer training."
            )

        mlflow.log_param("raw_file", raw_file)
        mlflow.log_param("n_transactions", len(df_features))

        artifacts.update({
            "anomaly_detector": str(anomaly_path),
            "processed_data": str(processed_path),
            "mlflow_run_id": run.info.run_id,
        })

    logger.info("Training pipeline complete.")
    return artifacts


if __name__ == "__main__":
    import sys
    file = sys.argv[1] if len(sys.argv) > 1 else "transactions.csv"
    results = run_training_pipeline(file)
    for k, v in results.items():
        print(f"  {k}: {v}")
