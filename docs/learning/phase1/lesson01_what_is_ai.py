"""
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =================================================================
#  PERSONAL AI MENTOR  -  Phase 1, Lesson 1
#  What is AI, ML, Deep Learning and Generative AI?
#  Audience : 16+ yrs .NET / AWS enterprise developer
#  Duration : ~60 minutes
# =================================================================

WHAT YOU WILL LEARN
-------------------
1. The relationship between AI, ML, Deep Learning and GenAI
2. How ML differs from traditional software you already write
3. The three learning paradigms (supervised, unsupervised, reinforcement)
4. Where each technology fits in a real enterprise financial system
5. Hands-on: build your first ML model from scratch (no magic, pure maths)
6. Hands-on: see the same problem solved with scikit-learn
7. Five interview questions you will face as an AI Technical Lead

RUN THIS FILE
-------------
    pip install numpy pandas scikit-learn
    python lesson01_what_is_ai.py

NO PAID APIs REQUIRED.
"""

# ==================================================================
# SECTION 1 - The AI Landscape (READ THIS FIRST)
# ==================================================================
"""
You already know this from enterprise software:
-----------------------------------------------------------------
  Traditional software = explicit rules written by a programmer.

  Example in C#:
      if (transaction.Amount > 1_000_000 && transaction.IsNewCounterparty)
          FlagForReview(transaction);

  You encoded the domain knowledge as IF/ELSE logic.
  When rules change, a developer changes the code.

AI / ML flips this:
-----------------------------------------------------------------
  You give the system EXAMPLES of past decisions.
  The system learns the rules itself.

  Example:
      100,000 historical transactions, each labelled "normal" or "review".
      The model learns which combinations of features predict "review".
      No developer wrote those rules.

-----------------------------------------------------------------
  AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  Generative AI
-----------------------------------------------------------------

  AI          = any technique that makes a computer behave intelligently
  ML          = AI that learns patterns from data (no explicit rules)
  Deep Learning = ML using multi-layer neural networks (learns complex patterns)
  GenAI       = Deep Learning that GENERATES new content (text, code, images)

ENTERPRISE ANALOGY (from your world):
  AI          ~  the concept of "smart software"
  ML          ~  a self-configuring business rule engine
  Deep Learning ~ a rule engine that handles unstructured data (emails, PDFs)
  GenAI       ~  a rule engine that also drafts responses, summaries, and code

WHERE EACH FITS IN YOUR INVESTMENT PLATFORM:
  ML          -> flag suspicious transactions (numerical patterns)
  Deep Learning -> understand portfolio manager emails (language)
  GenAI       -> extract structured JSON from free-text emails, explain decisions
"""

# ==================================================================
# SECTION 2 - Three ML Learning Paradigms
# ==================================================================
"""
1. SUPERVISED LEARNING
   ---------------------
   You provide labelled examples: (input, correct_output).
   The model learns to map inputs -> outputs.

   .NET analogy: an ML model is like a trained IClassifier<Transaction, RiskLabel>
   whose weights were set by gradient descent instead of a human developer.

   Examples:
     • Predict whether a transaction requires review      (classification)
     • Predict next month's portfolio value               (regression)
     • Predict which broker to route an order to          (classification)

2. UNSUPERVISED LEARNING
   ----------------------
   No labels. The model discovers structure in raw data.

   .NET analogy: like running a GROUP BY on dimensions you haven't named yet.

   Examples:
     • Cluster portfolios by behaviour pattern
     • Detect anomalous transactions without prior examples
     • Discover topic groups in regulatory documents

3. REINFORCEMENT LEARNING
   ------------------------
   An agent takes actions, receives rewards/penalties, and learns a policy.
   Less common in everyday enterprise AI -- used in trading execution engines,
   recommendation systems and robotics.

   .NET analogy: like a retry policy that learns the optimal backoff strategy
   from production feedback rather than from configuration.
"""

# ==================================================================
# SECTION 3 - The Core ML Concept: What a Model Actually IS
# ==================================================================
"""
A model is just a MATHEMATICAL FUNCTION with learnable parameters (weights).

For a simple linear model:
    prediction = w1 * feature1 + w2 * feature2 + ... + bias

Before training:  weights are random.
After training:   weights have been tuned so the function approximates
                  the true relationship in the training data.

Training = an optimisation loop:
    1. Make a prediction.
    2. Measure how wrong it is (loss).
    3. Compute the direction to adjust weights (gradient).
    4. Adjust weights a small step in that direction (gradient descent).
    5. Repeat thousands of times.

This is the SAME optimisation concept as minimising latency in your
microservices -- you're just minimising prediction error instead.
"""

# ==================================================================
# SECTION 4 - HANDS-ON: Logistic Regression from scratch (NumPy only)
# ==================================================================
"""
PROBLEM: Predict whether a financial transaction is high-risk (1) or normal (0).

FEATURES:
  x1 = transaction_amount_normalised  (0-1)
  x2 = counterparty_risk_score        (0-1)

We will implement:
  • Sigmoid activation
  • Binary cross-entropy loss
  • Gradient descent training loop
  • Prediction and accuracy

After this, you will know exactly what scikit-learn does internally.
"""

import numpy as np

np.random.seed(42)
SEP = "-" * 65


# -- 4.1 Generate synthetic transaction data ---------------------

def generate_transaction_data(n: int = 200) -> tuple:
    """
    Synthetic dataset:
      normal transactions    -> low amount, low counterparty risk
      high-risk transactions -> high amount OR high counterparty risk
    """
    rng = np.random.default_rng(42)

    # Normal transactions (class 0)
    n_normal = int(n * 0.75)
    X_normal = rng.uniform(0, 0.4, (n_normal, 2))   # low features
    y_normal = np.zeros(n_normal)

    # High-risk transactions (class 1)
    n_risk = n - n_normal
    X_risk = rng.uniform(0.5, 1.0, (n_risk, 2))     # high features
    y_risk = np.ones(n_risk)

    X = np.vstack([X_normal, X_risk])
    y = np.concatenate([y_normal, y_risk])

    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


X, y = generate_transaction_data(200)
print(SEP)
print("SECTION 4 - Logistic Regression from Scratch")
print(SEP)
print(f"Dataset: {len(X)} transactions | Features: {X.shape[1]}")
print(f"Class distribution: {int(y.sum())} high-risk, {int((1-y).sum())} normal")
print()


# -- 4.2 Train/test split ----------------------------------------

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# -- 4.3 Sigmoid function ----------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Squashes any real number into (0, 1).
    Think of it as: 'how confident am I this is high-risk?'
    
    σ(z) = 1 / (1 + e^(-z))
    
    z -> -∞  means probability ~ 0  (very normal)
    z -> +∞  means probability ~ 1  (very high-risk)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# -- 4.4 Binary cross-entropy loss -------------------------------

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Measures how wrong our predictions are.
    Lower = better. Zero = perfect predictions.
    
    Loss = -mean( y*log(p) + (1-y)*log(1-p) )
    
    .NET analogy: like a custom ILossFunction<bool, float>
    """
    eps = 1e-9   # prevent log(0)
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )


# -- 4.5 Training loop (gradient descent) ------------------------

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.5,
    epochs: int = 1000,
) -> tuple[np.ndarray, float]:
    """
    The core ML training loop -- identical in concept to what
    scikit-learn, PyTorch and TensorFlow all do internally.

    Parameters
    ----------
    X              : feature matrix (n_samples, n_features)
    y              : labels (n_samples,)
    learning_rate  : how big each weight update step is
    epochs         : how many times we scan the full dataset

    Returns
    -------
    weights, bias  : the learned parameters
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)   # start at zero
    bias = 0.0

    print("Training progress:")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

    for epoch in range(epochs):
        # 1. Forward pass: compute predictions
        z = X @ weights + bias        # linear combination
        predictions = sigmoid(z)       # squash to (0,1)

        # 2. Compute loss
        loss = binary_cross_entropy(y, predictions)

        # 3. Compute gradients
        #    dL/dw = (1/n) * X^T * (predictions - y)
        #    dL/db = (1/n) * sum(predictions - y)
        error = predictions - y
        grad_w = (X.T @ error) / n_samples
        grad_b = np.mean(error)

        # 4. Update weights (gradient descent step)
        weights -= learning_rate * grad_w
        bias    -= learning_rate * grad_b

        # Log every 200 epochs
        if epoch % 200 == 0 or epoch == epochs - 1:
            acc = np.mean((predictions >= 0.5) == y)
            print(f"  {epoch:>6}  {loss:>10.4f}  {acc:>10.2%}")

    return weights, bias


weights, bias = train_logistic_regression(X_train, y_train)
print()


# -- 4.6 Evaluate on test set -------------------------------------

def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float = 0.5,
) -> dict:
    proba = sigmoid(X @ weights + bias)
    preds = (proba >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    accuracy  = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


results = evaluate(X_test, y_test, weights, bias)

print(SEP)
print("Test-set evaluation (from-scratch model)")
print(SEP)
print(f"  Accuracy  : {results['accuracy']:.2%}")
print(f"  Precision : {results['precision']:.2%}  "
      f"(of flagged transactions, % that are truly high-risk)")
print(f"  Recall    : {results['recall']:.2%}  "
      f"(of all high-risk transactions, % we caught)")
print(f"  F1-Score  : {results['f1']:.2%}")
print()
print(f"  Confusion matrix:")
print(f"    TP={results['tp']}  FP={results['fp']}")
print(f"    FN={results['fn']}  TN={results['tn']}")
print()
print("  Learned weights:")
print(f"    w1 (transaction_amount)   = {weights[0]:+.4f}")
print(f"    w2 (counterparty_risk)    = {weights[1]:+.4f}")
print(f"    bias                      = {bias:+.4f}")
print()
print("  INTERPRETATION: positive weights mean the feature pushes")
print("  the prediction toward 'high-risk'. Both features should")
print("  have positive weights because higher values = higher risk.")


# ==================================================================
# SECTION 5 - SAME PROBLEM with scikit-learn (what production uses)
# ==================================================================

print()
print(SEP)
print("SECTION 5 - Same model with scikit-learn")
print(SEP)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# In production you wrap in a Pipeline -- identical to .NET middleware pipeline
sk_model = Pipeline([
    ("scaler", StandardScaler()),          # normalise features
    ("clf", LogisticRegression(            # logistic regression
        max_iter=1000,
        random_state=42,
    )),
])

sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

print("scikit-learn classification report:")
print(classification_report(
    y_test, sk_preds,
    target_names=["Normal", "High-Risk"],
    zero_division=0,
))

print("KEY INSIGHT:")
print("  The from-scratch model and scikit-learn model solve the same")
print("  problem the same way. scikit-learn adds production robustness,")
print("  convergence guarantees, and feature scaling. Use it in production.")


# ==================================================================
# SECTION 6 - The AI ↔ .NET Architecture Mapping
# ==================================================================

print()
print(SEP)
print("SECTION 6 - AI concept ↔ .NET concept mapping")
print(SEP)

MAPPING = [
    ("ML Model",          "A class with learned parameters (like a configured service)"),
    ("Training",          "Dependency injection container setup -- but automated"),
    ("Inference",         "Calling a service method -- model.Predict(input)"),
    ("Features",          "The DTO fields you pass to a business rule engine"),
    ("Labels / targets",  "The expected return value you would unit-test against"),
    ("Loss function",     "A custom IComparable that measures prediction quality"),
    ("Gradient descent",  "An optimiser -- like a self-tuning retry/backoff policy"),
    ("Epoch",             "One full pass through training data -- like a migration batch"),
    ("Overfitting",       "A model that memorised training data -- like hardcoded test stubs"),
    ("Regularisation",    "Adding a penalty for complexity -- like SOLID/YAGNI principles"),
    ("Pipeline",          "ASP.NET middleware pipeline -- ordered transformations"),
    ("Validation set",    "Integration test suite -- unseen data before production"),
    ("Test set",          "Production smoke test -- final unbiased evaluation"),
    ("MLflow experiment", "Application Insights -- experiment and metric tracking"),
]

print(f"  {'AI / ML Concept':<30}  .NET Equivalent")
print(f"  {'-'*30}  {'-'*35}")
for ai_term, net_equiv in MAPPING:
    print(f"  {ai_term:<30}  {net_equiv}")


# ==================================================================
# SECTION 7 - Where GenAI fits on top of ML
# ==================================================================

print()
print(SEP)
print("SECTION 7 - GenAI vs Classical ML in your investment platform")
print(SEP)

COMPARISON = [
    ("Detect high-risk transactions",    "Classical ML",  "Logistic Regression / XGBoost"),
    ("Cluster portfolios by behaviour",  "Classical ML",  "K-Means / DBSCAN"),
    ("Score counterparty risk",          "Classical ML",  "Gradient Boosting"),
    ("Read a PDF fund prospectus",       "GenAI (LLM)",   "GPT-4 / Claude via Bedrock"),
    ("Extract order from email",         "GenAI (LLM)",   "GPT-4 / Claude + structured output"),
    ("Explain why a trade was flagged",  "GenAI (LLM)",   "LLM + RAG over transaction history"),
    ("Recommend broker routing",         "Classical ML",  "XGBoost trained on TCA data"),
    ("Summarise portfolio performance",  "GenAI (LLM)",   "LLM + portfolio context"),
    ("Detect data drift in features",    "Classical ML",  "KS-test / Evidently AI"),
    ("Write compliance commentary",      "GenAI (LLM)",   "LLM + regulatory document RAG"),
]

print(f"  {'Use Case':<42}  {'Technology':<18}  Implementation")
print(f"  {'-'*42}  {'-'*18}  {'-'*35}")
for use_case, tech, impl in COMPARISON:
    print(f"  {use_case:<42}  {tech:<18}  {impl}")

print()
print("  KEY TAKEAWAY: Classical ML and GenAI are complementary.")
print("  Your investment platform needs BOTH.")
print("  Classical ML for numerical patterns, GenAI for language and generation.")


# ==================================================================
# SECTION 8 - Five Technical Lead Interview Questions
# ==================================================================

print()
print(SEP)
print("SECTION 8 - Interview Questions for AI Technical Lead")
print(SEP)

QUESTIONS = [
    (
        "Q1. What is the difference between AI, ML, Deep Learning and GenAI?",
        """AI is the broad field of building intelligent systems.
   ML is a subset that learns patterns from data instead of explicit rules.
   Deep Learning is a subset of ML using multi-layer neural networks --
   especially useful for unstructured data like text and images.
   Generative AI is a subset of Deep Learning that creates new content.
   In my investment platform I use classical ML for transaction risk scoring
   and GenAI for extracting structured orders from unstructured emails.""",
    ),
    (
        "Q2. How does ML differ from traditional rule-based programming?",
        """Traditional programming: I write IF/ELSE rules encoding domain knowledge.
   ML: I provide labelled historical examples; the algorithm finds the rules.
   The advantage: ML discovers non-obvious patterns in high-dimensional data,
   adapts to new patterns with retraining, and doesn't require a developer
   to enumerate every edge case. The risk: the model can learn spurious patterns
   if the training data is unrepresentative.""",
    ),
    (
        "Q3. What is gradient descent and why does it matter?",
        """Gradient descent is the optimisation algorithm that trains a model.
   It iteratively adjusts the model weights in the direction that reduces the
   loss function -- the measure of prediction error.
   It matters because almost all production ML and deep learning models
   are trained using variants of gradient descent (SGD, Adam, AdamW).
   Understanding it helps me debug training instability, tune learning rates,
   and explain model behaviour to business stakeholders.""",
    ),
    (
        "Q4. What is overfitting and how do you prevent it?",
        """Overfitting means the model memorised the training data rather than
   learning generalisable patterns. It performs well on training data but
   poorly on new production data.
   Prevention strategies:
     • More diverse training data
     • Regularisation (L1/L2 penalty on large weights)
     • Dropout (for neural networks)
     • Early stopping (stop training when validation loss stops improving)
     • Cross-validation
     • Simpler model architecture
   In my investment platform I always evaluate on a held-out test set
   and monitor for data drift after deployment.""",
    ),
    (
        "Q5. When would you NOT use ML? When would you use it?",
        """NOT use ML:
     • When the rules are deterministic and well-known (e.g. settlement date = T+2)
     • When you have too little data
     • When the decision requires a full audit trail with explicit reasoning
     • When regulatory requirements mandate interpretable rules
   USE ML:
     • When patterns are too complex or high-dimensional for manual rules
     • When historical labelled data is available and representative
     • When the relationship between features and outcome changes over time
     • When you need to process unstructured data (text, images, audio)
   In financial services I use ML for risk scoring and anomaly detection,
   not for settlement computation or regulatory calculations.""",
    ),
]

for question, answer in QUESTIONS:
    print(f"\n  {question}")
    print(f"  {'-' * (len(question) - 2)}")
    for line in answer.strip().split("\n"):
        print(f"  {line}")


# ==================================================================
# SECTION 9 - Your Practical Exercise for Today
# ==================================================================

print()
print(SEP)
print("SECTION 9 - Today's Exercises")
print(SEP)
print("""
  EXERCISE 1 (10 min): Modify the training loop above.
    Change the threshold from 0.5 to 0.3.
    Observe: Recall goes up, Precision goes down. WHY?
    Hint: lower threshold = model is more aggressive about flagging risk.

  EXERCISE 2 (10 min): Add a third feature.
    Add x3 = after_hours_flag (0 or 1) to generate_transaction_data().
    High-risk transactions are more likely to occur after hours.
    Retrain the model. Do the weights reflect your domain intuition?

  EXERCISE 3 (15 min): Connect to your project.
    Open  scripts/generate_sample_data.py  in this repo.
    It already generates synthetic investment transactions.
    Load that data, pick two features, and train a logistic regression.
    Report precision/recall. Would you deploy it?

  EXERCISE 4 (5 min): Reflection.
    In your existing investment transaction system, write down:
    - Three decisions currently made by explicit IF/ELSE rules
    - Which of those could be improved by ML?
    - What data would you need to train it?
""")

print(SEP)
print("NEXT LESSON: lesson02_supervised_learning.py")
print("  -> Decision Trees, Random Forest, XGBoost")
print("  -> Feature importance and model explainability")
print("  -> Hyperparameter tuning")
print("  -> Applying XGBoost to transaction risk scoring")
print(SEP)
