"""
Tests for Day 7 concepts – ML Model Evaluation
Verifies that the evaluation workflow produces sensible metrics
on synthetic imbalanced investment-transaction data.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Shared fixture – synthetic imbalanced transaction dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluation_data():
    """
    2 000 synthetic transactions; 5 % require investigation (class 1).
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X, y = make_classification(
        n_samples=2_000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        weights=[0.95, 0.05],
        random_state=42,
    )
    cols = [
        "Transaction_Amount",
        "Beneficiary_History",
        "Transaction_Frequency",
        "Portfolio_Activity",
        "After_Hours_Flag",
    ]
    df = pd.DataFrame(X, columns=cols)
    df["Requires_Review"] = y

    X_all = df.drop("Requires_Review", axis=1)
    y_all = df["Requires_Review"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.40, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture(scope="module")
def trained_model(evaluation_data):
    X_train, _, _, y_train, _, _ = evaluation_data
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetProperties:
    def test_class_imbalance(self, evaluation_data):
        _, _, _, y_train, _, _ = evaluation_data
        minority_rate = y_train.mean()
        # Minority class should be roughly 5 % (allow 2–10 % range)
        assert 0.02 <= minority_rate <= 0.10, (
            f"Expected ~5 % minority class, got {minority_rate:.2%}"
        )

    def test_split_sizes(self, evaluation_data):
        X_train, X_val, X_test, _, _, _ = evaluation_data
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 2_000
        # Train ≈ 60 %, val ≈ 20 %, test ≈ 20 %
        assert abs(len(X_train) / total - 0.60) < 0.05
        assert abs(len(X_val)   / total - 0.20) < 0.05
        assert abs(len(X_test)  / total - 0.20) < 0.05


class TestModelTraining:
    def test_model_has_pipeline_steps(self, trained_model):
        step_names = [name for name, _ in trained_model.steps]
        assert "scaler" in step_names
        assert "clf" in step_names

    def test_model_predicts_probabilities(self, trained_model, evaluation_data):
        _, _, X_test, _, _, _ = evaluation_data
        proba = trained_model.predict_proba(X_test)
        assert proba.shape[1] == 2
        # Each row sums to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestEvaluationMetrics:
    def test_accuracy_above_baseline(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        # Baseline (always-normal model) = 95 %; balanced model trades some accuracy
        # for recall – expect > 60 %
        assert acc > 0.60, f"Accuracy too low: {acc:.4f}"

    def test_recall_above_zero(self, trained_model, evaluation_data):
        """
        Day 7 key lesson: a 95 % accuracy all-negative model has recall = 0.
        Our balanced model must detect at least some minority-class transactions.
        """
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        rec = recall_score(y_test, preds, zero_division=0)
        assert rec > 0.0, "Recall is 0 – model never predicts minority class"

    def test_f1_better_than_zero(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        assert f1 > 0.0

    def test_roc_auc_above_random(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        proba = trained_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        # Must do better than random (0.5)
        assert auc > 0.5, f"ROC-AUC not better than random: {auc:.4f}"

    def test_pr_auc_above_prior(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        proba = trained_model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, proba)
        prior = y_test.mean()   # random classifier baseline for PR-AUC
        assert pr_auc > prior, (
            f"PR-AUC {pr_auc:.4f} not better than class prior {prior:.4f}"
        )


class TestConfusionMatrix:
    def test_confusion_matrix_shape(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_sums_to_test_size(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        assert cm.sum() == len(y_test)

    def test_no_all_negative_model(self, trained_model, evaluation_data):
        """
        Day 7 scenario: 99 % accuracy / 20 % recall model should NOT be deployed.
        Our balanced model must have TP > 0.
        """
        _, _, X_test, _, _, y_test = evaluation_data
        preds = trained_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        tp = cm[1, 1]
        assert tp > 0, "Model never predicts minority class (TP = 0)"


class TestThresholdAnalysis:
    def test_lower_threshold_increases_recall(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        proba = trained_model.predict_proba(X_test)[:, 1]

        preds_low  = (proba >= 0.2).astype(int)
        preds_high = (proba >= 0.8).astype(int)

        recall_low  = recall_score(y_test, preds_low,  zero_division=0)
        recall_high = recall_score(y_test, preds_high, zero_division=0)

        assert recall_low >= recall_high, (
            f"Expected lower threshold to increase recall. "
            f"recall@0.2={recall_low:.4f}, recall@0.8={recall_high:.4f}"
        )

    def test_higher_threshold_increases_precision(self, trained_model, evaluation_data):
        _, _, X_test, _, _, y_test = evaluation_data
        proba = trained_model.predict_proba(X_test)[:, 1]

        preds_low  = (proba >= 0.2).astype(int)
        preds_high = (proba >= 0.8).astype(int)

        p_low  = precision_score(y_test, preds_low,  zero_division=0)
        p_high = precision_score(y_test, preds_high, zero_division=0)

        # Higher threshold → fewer but more certain positives → higher precision
        # Allow equal (edge case where all positives are correct at high threshold)
        assert p_high >= p_low or p_high == 0, (
            f"Expected higher threshold → higher precision. "
            f"p@0.2={p_low:.4f}, p@0.8={p_high:.4f}"
        )
