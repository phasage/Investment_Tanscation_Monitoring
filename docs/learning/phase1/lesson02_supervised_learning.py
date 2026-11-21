"""
+==================================================================+
|  PERSONAL AI MENTOR  -  Phase 1, Lesson 2                       |
|  Supervised Learning - Decision Trees, Random Forest, XGBoost   |
|  Audience : 16+ yrs .NET / AWS enterprise developer             |
|  Duration : ~60 minutes                                          |
+==================================================================+

WHAT YOU WILL LEARN
-------------------
1. Decision Trees - how they work and why they matter
2. Random Forest  - why ensembles beat single models
3. XGBoost        - the production standard for tabular financial data
4. Feature importance - explaining model decisions to compliance teams
5. Hyperparameter tuning - how to improve a model systematically
6. The full pipeline: data -> features -> train -> evaluate -> explain
7. Five interview questions

RUN THIS FILE
-------------
    pip install numpy pandas scikit-learn xgboost
    python lesson02_supervised_learning.py
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

SEP = "-" * 65

# ==================================================================
# CONCEPT: Decision Trees
# ==================================================================
"""
A Decision Tree is a flowchart of IF/ELSE conditions learned from data.

You already write decision trees manually in C# for compliance rules:

    if (transaction.Amount > 500_000)
        if (transaction.CounterpartyRiskScore > 0.7)
            return RiskLabel.High;
        else
            return RiskLabel.Medium;
    else
        return RiskLabel.Low;

A Decision Tree algorithm finds these thresholds AUTOMATICALLY from data.
It asks: "What split of feature X reduces classification error the most?"

The measure used: Gini impurity or Information Gain (entropy).

ADVANTAGE: Fully interpretable. Compliance teams can read the rules.
DISADVANTAGE: Overfits easily. High variance -- different training sets
              produce very different trees.
"""

# ==================================================================
# SECTION 1 - Generate realistic investment transaction dataset
# ==================================================================

def generate_transactions(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic investment transactions for a fund management platform.
    
    Features (what your .NET API would receive):
      transaction_amount     - GBP value of the trade
      counterparty_risk      - 0 (safe) to 1 (risky)
      after_hours_flag       - 1 if placed outside market hours
      new_counterparty_flag  - 1 if first time trading with this party
      portfolio_concentration- % of portfolio in this asset (0-1)
      order_frequency_24h    - number of orders from this portfolio today
    
    Target:
      is_high_risk - 1 if the transaction was later confirmed suspicious
    """
    rng = np.random.default_rng(seed)
    n_risk = int(n * 0.08)      # 8% high-risk (realistic class imbalance)
    n_normal = n - n_risk

    def make_records(count, risk_class):
        if risk_class == 0:
            return {
                "transaction_amount":      rng.lognormal(10, 1, count),
                "counterparty_risk":       rng.beta(2, 8, count),    # skew low
                "after_hours_flag":        rng.binomial(1, 0.05, count),
                "new_counterparty_flag":   rng.binomial(1, 0.08, count),
                "portfolio_concentration": rng.beta(2, 10, count),
                "order_frequency_24h":     rng.poisson(2, count).astype(float),
                "is_high_risk":            np.zeros(count, dtype=int),
            }
        else:
            return {
                "transaction_amount":      rng.lognormal(13, 1.5, count),  # larger
                "counterparty_risk":       rng.beta(8, 2, count),          # skew high
                "after_hours_flag":        rng.binomial(1, 0.40, count),   # more likely
                "new_counterparty_flag":   rng.binomial(1, 0.55, count),   # more likely
                "portfolio_concentration": rng.beta(8, 2, count),          # concentrated
                "order_frequency_24h":     rng.poisson(12, count).astype(float),
                "is_high_risk":            np.ones(count, dtype=int),
            }

    normal = pd.DataFrame(make_records(n_normal, 0))
    risky  = pd.DataFrame(make_records(n_risk, 1))
    df = pd.concat([normal, risky], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    return df


FEATURES = [
    "transaction_amount",
    "counterparty_risk",
    "after_hours_flag",
    "new_counterparty_flag",
    "portfolio_concentration",
    "order_frequency_24h",
]
TARGET = "is_high_risk"

df = generate_transactions(2000)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(SEP)
print("LESSON 2 - Supervised Learning: Decision Tree -> RF -> XGBoost")
print(SEP)
print(f"Dataset: {len(df):,} transactions | "
      f"High-risk: {y.sum()} ({y.mean():.1%}) | "
      f"Normal: {(1-y).sum()} ({(1-y).mean():.1%})")
print()


# ==================================================================
# SECTION 2 - Decision Tree
# ==================================================================

print(SEP)
print("SECTION 2 - Decision Tree")
print(SEP)

dt = DecisionTreeClassifier(
    max_depth=4,              # limit depth to prevent overfitting
    class_weight="balanced",  # compensate for class imbalance
    random_state=42,
)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

print("Decision rules learned (first 4 levels):")
print(export_text(dt, feature_names=FEATURES, max_depth=3))

print(f"Decision Tree ROC-AUC : {roc_auc_score(y_test, dt_proba):.4f}")
print(classification_report(
    y_test, dt_preds,
    target_names=["Normal", "High-Risk"],
    zero_division=0,
))

"""
WHAT TO NOTICE:
  The tree learned thresholds automatically -- counterparty_risk > 0.5,
  transaction_amount > some_threshold -- just like you'd write in C# compliance rules.
  
  The difference: it found the optimal thresholds from 1,600 examples.
  You didn't write a single IF statement.
"""


# ==================================================================
# SECTION 3 - Random Forest (Ensemble of Trees)
# ==================================================================

print(SEP)
print("SECTION 3 - Random Forest")
print(SEP)

"""
CONCEPT: Random Forest builds 100+ decision trees, each on a random
subset of data and features. The final prediction is the MAJORITY VOTE.

.NET analogy: like a distributed voting system where each node independently
processes a subset of the data and you aggregate results -- similar to how
you'd design a consensus algorithm in a distributed microservice.

WHY IT'S BETTER than a single tree:
  One tree:    high variance, overfits.
  100 trees:   individual errors cancel out -> lower variance.
  This is called the "wisdom of crowds" in ensemble learning.
"""

rf = RandomForestClassifier(
    n_estimators=200,         # 200 trees -- more is usually better (diminishing returns)
    max_depth=8,
    class_weight="balanced",
    n_jobs=-1,                # use all CPU cores -- like setting maxDegreeOfParallelism
    random_state=42,
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(f"Random Forest ROC-AUC : {roc_auc_score(y_test, rf_proba):.4f}")
print(classification_report(
    y_test, rf_preds,
    target_names=["Normal", "High-Risk"],
    zero_division=0,
))

# Feature importance -- critical for compliance explainability
print("Feature Importance (how much each feature drives the decision):")
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "#" * int(imp * 50)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

print()
print("INTERPRETATION:")
print("  High importance = this feature strongly predicts high-risk.")
print("  You can show this to your compliance team as evidence the model")
print("  is using the right signals, not spurious correlations.")


# ==================================================================
# SECTION 4 - XGBoost (Production Standard for Financial Data)
# ==================================================================

print()
print(SEP)
print("SECTION 4 - XGBoost")
print(SEP)

"""
CONCEPT: XGBoost (eXtreme Gradient Boosting) builds trees SEQUENTIALLY.
Each tree corrects the errors of the previous one.

Random Forest:   100 trees trained in PARALLEL, vote on final answer.
XGBoost:         100 trees trained in SEQUENCE, each focused on hard cases.

This is the algorithm behind most winning Kaggle financial tabular competitions.
It is what HSBC, JPMorgan, and other financial institutions use for
fraud detection and credit scoring at scale.

.NET analogy: like a chain of Decorator/Handler middleware where each handler
focuses on the cases the previous one got wrong.

KEY HYPERPARAMETERS (you will tune these):
  n_estimators:    number of trees (more = better, with diminishing returns)
  learning_rate:   how much each tree contributes (lower = more robust)
  max_depth:       tree depth (higher = more complex patterns, more overfitting risk)
  subsample:       fraction of training data each tree sees (adds randomness)
  scale_pos_weight: handles class imbalance (set to n_negative / n_positive)
"""

try:
    from xgboost import XGBClassifier

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale = n_neg / n_pos    # compensate for class imbalance

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,       # low LR = slow but robust
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,   # critical for imbalanced datasets
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # monitor test AUC during training
        verbose=False,
    )

    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    print(f"XGBoost ROC-AUC       : {roc_auc_score(y_test, xgb_proba):.4f}")
    print(classification_report(
        y_test, xgb_preds,
        target_names=["Normal", "High-Risk"],
        zero_division=0,
    ))

    print("XGBoost Feature Importance:")
    xgb_imp = pd.Series(
        xgb.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    for feat, imp in xgb_imp.items():
        bar = "#" * int(imp * 60)
        print(f"  {feat:<30} {imp:.4f}  {bar}")

except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    print("Skipping XGBoost section.")


# ==================================================================
# SECTION 5 - Cross-Validation (the right way to evaluate)
# ==================================================================

print()
print(SEP)
print("SECTION 5 - Cross-Validation")
print(SEP)

"""
PROBLEM: One train/test split gives you one number. Is it reliable?
What if you got lucky (or unlucky) with that particular split?

SOLUTION: K-Fold Cross-Validation
  Split data into K folds.
  Train on K-1 folds, test on the remaining fold.
  Repeat K times, rotating which fold is the test set.
  Average the K scores.

.NET analogy: running your integration tests against K different database
snapshots to ensure your code works across a range of data states, not just
one specific fixture.

This gives you a confidence interval on model performance.
"""

cv_scores = cross_val_score(
    RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    X, y,
    cv=5,                     # 5-fold cross-validation
    scoring="roc_auc",        # metric to report
    n_jobs=-1,
)

print(f"5-Fold Cross-Validation ROC-AUC scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\n  Mean  : {cv_scores.mean():.4f}")
print(f"  Std   : {cv_scores.std():.4f}  "
      f"(lower = more consistent across different data samples)")
print()
print("  RULE OF THUMB: If std > 0.05, your model is sensitive to")
print("  which data it trains on -- investigate class balance and")
print("  feature quality before deploying.")


# ==================================================================
# SECTION 6 - Algorithm Comparison
# ==================================================================

print()
print(SEP)
print("SECTION 6 - Algorithm Comparison for Financial Risk Scoring")
print(SEP)

COMPARISON = [
    ("Logistic Regression", "Linear boundary",    "High",        "Low",      "Baseline, audit trail"),
    ("Decision Tree",       "Non-linear",         "High",        "Medium",   "When rules must be readable"),
    ("Random Forest",       "Non-linear",         "Medium",      "High",     "Production risk scoring"),
    ("XGBoost",             "Non-linear",         "Low-Medium",  "Very High","Production fraud detection"),
    ("Neural Network",      "Highly non-linear",  "Low",         "Very High","Images, text, sequences"),
    ("LLM (GenAI)",         "Language patterns",  "Low",         "Very High","Unstructured text, emails"),
]

print(f"  {'Algorithm':<22} {'Boundary':<18} {'Explainability':<16} {'Accuracy':<12} Use Case")
print(f"  {'-'*22} {'-'*18} {'-'*16} {'-'*12} {'-'*30}")
for row in COMPARISON:
    print(f"  {row[0]:<22} {row[1]:<18} {row[2]:<16} {row[3]:<12} {row[4]}")

print()
print("  RECOMMENDATION for investment transaction risk scoring:")
print("  -> Start with Logistic Regression as a baseline.")
print("  -> Move to XGBoost for production (best accuracy/explainability balance).")
print("  -> Use LLMs only for unstructured inputs (emails, PDFs, notes).")


# ==================================================================
# SECTION 7 - Five Interview Questions
# ==================================================================

print()
print(SEP)
print("SECTION 7 - Interview Questions")
print(SEP)

QUESTIONS = [
    (
        "Q1. How does XGBoost differ from Random Forest?",
        """Random Forest: trains trees in PARALLEL, each on a random data/feature subset.
   Final answer = majority vote. Reduces variance.
   XGBoost: trains trees SEQUENTIALLY. Each tree focuses on the residual errors
   of the previous one (gradient boosting). Reduces both bias and variance.
   XGBoost generally outperforms Random Forest on tabular financial data.
   Random Forest is faster to train and less sensitive to hyperparameters.""",
    ),
    (
        "Q2. How do you handle class imbalance in fraud/risk detection?",
        """In financial datasets, 95-99% of transactions are normal. Options:
   1. class_weight='balanced' -- upweights minority class during training
   2. SMOTE -- synthesise minority class examples
   3. scale_pos_weight (XGBoost) -- penalise minority misclassification more
   4. Threshold adjustment -- lower threshold to increase recall
   5. PR-AUC instead of ROC-AUC -- more informative for imbalanced data
   I always check minority-class precision and recall separately, never just accuracy.""",
    ),
    (
        "Q3. What is ROC-AUC and when is it insufficient?",
        """ROC-AUC measures the model's ability to rank positive examples above negatives
   across all classification thresholds. 1.0 = perfect, 0.5 = random.
   It's insufficient for highly imbalanced datasets because it can be high even
   when the model performs poorly on the minority class.
   In those cases I use Precision-Recall AUC (average_precision_score).
   For financial risk systems I also calculate the business cost of FP and FN.""",
    ),
    (
        "Q4. How do you explain a model's decision to a compliance team?",
        """Several approaches depending on model complexity:
   1. Feature importance (Random Forest / XGBoost) -- which features drove the prediction
   2. Decision tree rules -- readable IF/ELSE conditions
   3. SHAP values -- per-prediction feature contributions
   4. LIME -- local surrogate model approximation
   For regulated financial decisions I combine:
     - Model feature importance (global explanation)
     - SHAP values (transaction-level explanation)
     - Confidence score and threshold documentation
   The LLM layer can then generate a human-readable summary of the SHAP values.""",
    ),
    (
        "Q5. How would you deploy an ML model in an AWS microservices architecture?",
        """My approach:
   1. Train and register model in MLflow (tracks version, metrics, artifacts)
   2. Serialise with joblib/ONNX -- language-agnostic model format
   3. Deploy as a containerised FastAPI service on ECS or EKS
   4. Expose via API Gateway -- existing .NET services call it like any REST API
   5. Alternatively: AWS SageMaker endpoint for managed inference scaling
   6. Use Lambda for lightweight, event-driven scoring of individual transactions
   7. Monitor with CloudWatch + Evidently AI for data drift
   8. Blue/green deployment for model updates -- same as your existing CI/CD pipeline""",
    ),
]

for question, answer in QUESTIONS:
    print(f"\n  {question}")
    print(f"  {'-' * (len(question) - 2)}")
    for line in answer.strip().split("\n"):
        print(f"  {line}")

print()
print(SEP)
print("EXERCISES")
print(SEP)
print("""
  EXERCISE 1: Change max_depth in the Decision Tree to 1, 3, 10.
    What happens to the printed decision rules and test ROC-AUC?
    At which depth does overfitting begin?

  EXERCISE 2: Remove 'counterparty_risk' from FEATURES.
    Retrain all three models. How much does ROC-AUC drop?
    This tells you how much business value that one feature carries.

  EXERCISE 3: Implement SHAP for XGBoost.
    pip install shap
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test[:5])
    print(shap_values)
    Can you interpret which features made the first transaction high-risk?

  EXERCISE 4: Connect to your data.
    Use the transactions generated by scripts/generate_sample_data.py.
    Train XGBoost. What ROC-AUC do you get?
    Identify the top 3 features driving risk in your synthetic dataset.
""")

print(SEP)
print("NEXT LESSON: lesson03_unsupervised_learning.py")
print("  -> K-Means clustering of portfolios")
print("  -> Isolation Forest for anomaly detection")
print("  -> DBSCAN for behaviour-based clustering")
print("  -> When to use unsupervised vs supervised learning")
print(SEP)
