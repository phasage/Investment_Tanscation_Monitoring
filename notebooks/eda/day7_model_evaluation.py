"""
Day 7 – ML Model Evaluation for Investment Transaction Monitoring
=================================================================
Topics covered:
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC and PR-AUC
  - Classification threshold analysis
  - Model drift concepts

Domain: Investment transaction monitoring
         Identify transactions that may require investigation.
"""

# ---------------------------------------------------------------------------
# Step 1 – Install (run once)
# ---------------------------------------------------------------------------
# pip install pandas numpy scikit-learn

# ---------------------------------------------------------------------------
# Step 2 – Imports
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np

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
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Step 3 – Generate synthetic investment transaction data
# ---------------------------------------------------------------------------
# 2,000 transactions; only 5 % require investigation → imbalanced dataset
X_raw, y = make_classification(
    n_samples=2_000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    weights=[0.95, 0.05],   # 95 % normal, 5 % require review
    random_state=42,
)

FEATURE_NAMES = [
    "Transaction_Amount",
    "Beneficiary_History",
    "Transaction_Frequency",
    "Portfolio_Activity",
    "After_Hours_Flag",
]

df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
df["Requires_Review"] = y

print("── Dataset overview ──────────────────────────────")
print(df.head())
print()
print("Class distribution:")
print(df["Requires_Review"].value_counts())
print()

# ---------------------------------------------------------------------------
# Step 4 – Separate features and target
# ---------------------------------------------------------------------------
X = df.drop("Requires_Review", axis=1)
y = df["Requires_Review"]

# ---------------------------------------------------------------------------
# Step 5 – Three-way split: 60 % train / 20 % validation / 20 % test
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train size:      {len(X_train):,}")
print(f"Validation size: {len(X_val):,}")
print(f"Test size:       {len(X_test):,}")
print()

# ---------------------------------------------------------------------------
# Step 6 – Build and train the model
# ---------------------------------------------------------------------------
# class_weight="balanced" compensates for the minority class (5 %)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)
print("Model training complete.")
print()

# ---------------------------------------------------------------------------
# Step 7 – Threshold analysis on validation set
# ---------------------------------------------------------------------------
val_proba = model.predict_proba(X_val)[:, 1]

print("── Threshold analysis (validation set) ──────────")
print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
print("-" * 46)

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    preds = (val_proba >= threshold).astype(int)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    f = f1_score(y_val, preds, zero_division=0)
    print(f"{threshold:>10.1f}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")

print()

# ---------------------------------------------------------------------------
# Step 8 – Final evaluation on test set (threshold = 0.5)
# ---------------------------------------------------------------------------
# In a real project, choose the threshold using the validation set only.
THRESHOLD = 0.5
test_proba = model.predict_proba(X_test)[:, 1]
final_preds = (test_proba >= THRESHOLD).astype(int)

print(f"── Final test-set evaluation (threshold={THRESHOLD}) ──")
print(f"Accuracy  : {accuracy_score(y_test, final_preds):.4f}")
print(f"Precision : {precision_score(y_test, final_preds, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, final_preds, zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y_test, final_preds, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, test_proba):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, test_proba):.4f}")
print()

# ---------------------------------------------------------------------------
# Step 9 – Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel()

print("── Confusion Matrix ──────────────────────────────")
print(f"                 Predicted Normal  Predicted Review")
print(f"Actually Normal       {tn:>6}            {fp:>6}")
print(f"Actually Review       {fn:>6}            {tp:>6}")
print()
print(f"True  Positives (TP): {tp}  → correctly flagged for review")
print(f"True  Negatives (TN): {tn}  → correctly cleared as normal")
print(f"False Positives (FP): {fp}  → normal transactions incorrectly flagged")
print(f"False Negatives (FN): {fn}  → review transactions missed by model")
print()

# ---------------------------------------------------------------------------
# Step 10 – Detailed classification report
# ---------------------------------------------------------------------------
print("── Classification Report ─────────────────────────")
print(classification_report(y_test, final_preds, zero_division=0,
                             target_names=["Normal", "Requires Review"]))

# ---------------------------------------------------------------------------
# Step 11 – Model comparison (Day 7 scenario from docs)
# ---------------------------------------------------------------------------
print("── Scenario: Model A vs Model B ──────────────────")
comparison = {
    "Model": ["Model A", "Model B"],
    "Accuracy":  [0.98, 0.94],
    "Precision": [0.90, 0.80],
    "Recall":    [0.40, 0.92],
    "F1-Score":  [0.5538, 0.8558],
}
print(pd.DataFrame(comparison).to_string(index=False))
print()
print("Interview answer: Model B has significantly better Recall and F1-Score.")
print("If the goal is to catch suspicious transactions, recall matters more than accuracy.")

# ---------------------------------------------------------------------------
# Step 12 – Bonus: 99 % accuracy / 20 % recall challenge
# ---------------------------------------------------------------------------
print()
print("── Bonus: Would you deploy a 99 % accuracy / 20 % recall model? ──")
print("Answer: No.")
print("  20 % recall means the model misses 80 % of transactions requiring review.")
print("  Investigate: class imbalance, feature quality, threshold, training data.")
print("  Evaluate alternative models using validation data before any test-set use.")
