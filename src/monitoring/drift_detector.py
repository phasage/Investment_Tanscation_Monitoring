"""Model and data drift detection using Evidently AI."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger


class DriftDetector:
    """
    Detect feature and prediction drift between a reference dataset
    and the current production data window.
    """

    def __init__(self, reference_df: pd.DataFrame, feature_cols: list[str]):
        self.reference_df = reference_df
        self.feature_cols = feature_cols

    def detect(self, current_df: pd.DataFrame) -> dict:
        """
        Run drift detection. Returns a summary dict with drift flags per feature.
        Uses Evidently if available; falls back to basic KS-test otherwise.
        """
        try:
            return self._evidently_drift(current_df)
        except ImportError:
            logger.warning("Evidently not installed. Using KS-test fallback.")
            return self._ks_drift(current_df)

    def _evidently_drift(self, current_df: pd.DataFrame) -> dict:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_df[self.feature_cols],
            current_data=current_df[self.feature_cols],
        )
        result = report.as_dict()
        drift_summary = result["metrics"][0]["result"]
        logger.info(
            f"Drift detected: {drift_summary.get('dataset_drift', False)} | "
            f"Drifted features: {drift_summary.get('number_of_drifted_columns', 0)}"
        )
        return drift_summary

    def _ks_drift(self, current_df: pd.DataFrame) -> dict:
        from scipy.stats import ks_2samp

        results = {}
        for col in self.feature_cols:
            if col in self.reference_df.columns and col in current_df.columns:
                stat, pval = ks_2samp(
                    self.reference_df[col].dropna(),
                    current_df[col].dropna(),
                )
                drifted = pval < 0.05
                results[col] = {"statistic": stat, "p_value": pval, "drifted": drifted}
                if drifted:
                    logger.warning(f"Drift detected in '{col}' (p={pval:.4f})")

        n_drifted = sum(1 for v in results.values() if v["drifted"])
        return {
            "dataset_drift": n_drifted > 0,
            "number_of_drifted_columns": n_drifted,
            "features": results,
            "checked_at": datetime.utcnow().isoformat(),
        }

    def save_report(self, report: dict, output_dir: str = "reports/metrics") -> Path:
        import json
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"drift_report_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Drift report saved → {out_path}")
        return out_path
