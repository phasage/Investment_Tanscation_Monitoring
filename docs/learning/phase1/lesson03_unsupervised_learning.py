# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
# =================================================================
#  PERSONAL AI MENTOR  -  Phase 1, Lesson 3
#  Unsupervised Learning: Clustering and Anomaly Detection
#  Audience : 16+ yrs .NET / AWS enterprise developer
#  Duration : ~60 minutes
# =================================================================

WHAT YOU WILL LEARN
-------------------
1. What unsupervised learning is and when to prefer it over supervised
2. K-Means clustering  - group portfolios by behaviour
3. DBSCAN              - density-based clustering, finds arbitrary shapes
4. Isolation Forest    - anomaly detection without labels (your AnomalyDetector)
5. PCA                 - dimensionality reduction for visualisation
6. The right algorithm for each investment platform scenario
7. Five interview questions

RUN THIS FILE
-------------
    pip install numpy pandas scikit-learn
    python -X utf8 docs/learning/phase1/lesson03_unsupervised_learning.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

SEP = "-" * 65


# ==================================================================
# CONCEPT: Supervised vs Unsupervised Learning
# ==================================================================
"""
SUPERVISED LEARNING (Lessons 1-2):
  You provide labelled examples: (input, correct_output).
  The model learns to predict the label for new inputs.
  Requires: labelled historical data.
  Example: predict is_high_risk from past flagged transactions.

UNSUPERVISED LEARNING (this lesson):
  You provide inputs ONLY - no labels.
  The model discovers hidden structure by itself.
  Requires: data, but NO labels.

WHY THIS MATTERS FOR YOUR INVESTMENT PLATFORM:
  You often do NOT have labels.
  - Fraud labels: only 0.1% are confirmed, the rest unknown.
  - New attack patterns: by definition not in training labels.
  - Portfolio segmentation: no one has pre-labelled "portfolio types".

  Unsupervised learning lets you:
    1. Find natural groups in portfolios/clients (clustering)
    2. Flag unusual transactions WITHOUT needing confirmed fraud labels
    3. Reduce 50 features to 2D for visual inspection (PCA)
    4. Discover new risk patterns the business hasn't named yet

.NET ANALOGY:
  Supervised   = a trained IClassifier<Transaction, RiskLabel>
  Unsupervised = a self-organising GROUP BY that finds the grouping keys itself
"""

print(SEP)
print("LESSON 3 - Unsupervised Learning: Clustering + Anomaly Detection")
print(SEP)


# ==================================================================
# SECTION 1 - Generate realistic portfolio dataset
# ==================================================================

def generate_portfolio_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic portfolio behaviour data.
    
    Each row = one portfolio's behaviour over the last 30 days.
    
    Features:
      avg_daily_trades     - average number of trades per day
      avg_trade_size_gbp   - average trade size in GBP
      equity_pct           - % of portfolio in equities
      fx_exposure_pct      - % of portfolio in foreign currency
      after_hours_pct      - % of trades placed after market hours
      new_counterparty_pct - % of trades with new counterparties
      concentration_score  - Herfindahl index (0=diversified, 1=concentrated)
    
    Hidden structure (we know but the model does NOT):
      Group A - Large institutional, diversified, low risk
      Group B - Medium hedge fund, concentrated, high FX
      Group C - Retail/small, infrequent, domestic
      Group D - Anomalous / potentially suspicious (small cluster)
    """
    rng = np.random.default_rng(seed)

    groups = {
        "A_institutional": {
            "n": int(n * 0.35),
            "avg_daily_trades":     rng.normal(45, 8),
            "avg_trade_size_gbp":   rng.lognormal(12.5, 0.4),
            "equity_pct":           rng.beta(5, 3),
            "fx_exposure_pct":      rng.beta(3, 5),
            "after_hours_pct":      rng.beta(1, 15),
            "new_counterparty_pct": rng.beta(1, 12),
            "concentration_score":  rng.beta(2, 8),
        },
        "B_hedge_fund": {
            "n": int(n * 0.30),
            "avg_daily_trades":     rng.normal(120, 25),
            "avg_trade_size_gbp":   rng.lognormal(11.5, 0.6),
            "equity_pct":           rng.beta(3, 3),
            "fx_exposure_pct":      rng.beta(7, 2),
            "after_hours_pct":      rng.beta(2, 8),
            "new_counterparty_pct": rng.beta(2, 8),
            "concentration_score":  rng.beta(6, 3),
        },
        "C_retail": {
            "n": int(n * 0.30),
            "avg_daily_trades":     rng.normal(5, 2),
            "avg_trade_size_gbp":   rng.lognormal(9.0, 0.5),
            "equity_pct":           rng.beta(4, 4),
            "fx_exposure_pct":      rng.beta(1, 8),
            "after_hours_pct":      rng.beta(1, 20),
            "new_counterparty_pct": rng.beta(1, 20),
            "concentration_score":  rng.beta(3, 5),
        },
    }

    frames = []
    for group_name, params in groups.items():
        count = params["n"]
        frame = pd.DataFrame({
            "avg_daily_trades":     np.abs(params["avg_daily_trades"](count)
                                           if callable(params["avg_daily_trades"])
                                           else rng.normal(params["avg_daily_trades"], 1, count)),
            "avg_trade_size_gbp":   params["avg_trade_size_gbp"](count)
                                    if callable(params["avg_trade_size_gbp"])
                                    else rng.lognormal(10, 0.5, count),
            "equity_pct":           np.clip(params["equity_pct"](count)
                                            if callable(params["equity_pct"])
                                            else rng.beta(4, 4, count), 0, 1),
            "fx_exposure_pct":      np.clip(params["fx_exposure_pct"](count)
                                            if callable(params["fx_exposure_pct"])
                                            else rng.beta(2, 5, count), 0, 1),
            "after_hours_pct":      np.clip(params["after_hours_pct"](count)
                                            if callable(params["after_hours_pct"])
                                            else rng.beta(1, 15, count), 0, 1),
            "new_counterparty_pct": np.clip(params["new_counterparty_pct"](count)
                                            if callable(params["new_counterparty_pct"])
                                            else rng.beta(1, 12, count), 0, 1),
            "concentration_score":  np.clip(params["concentration_score"](count)
                                            if callable(params["concentration_score"])
                                            else rng.beta(3, 5, count), 0, 1),
            "true_group":           group_name,
        })
        frames.append(frame)

    # Add anomalous portfolios (5%)
    n_anomalous = int(n * 0.05)
    anomalous = pd.DataFrame({
        "avg_daily_trades":     rng.uniform(200, 500, n_anomalous),
        "avg_trade_size_gbp":   rng.lognormal(14, 1, n_anomalous),
        "equity_pct":           rng.uniform(0, 1, n_anomalous),
        "fx_exposure_pct":      rng.uniform(0.7, 1.0, n_anomalous),
        "after_hours_pct":      rng.uniform(0.4, 0.9, n_anomalous),
        "new_counterparty_pct": rng.uniform(0.5, 1.0, n_anomalous),
        "concentration_score":  rng.uniform(0.7, 1.0, n_anomalous),
        "true_group":           "D_anomalous",
    })
    frames.append(anomalous)

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


FEATURES = [
    "avg_daily_trades",
    "avg_trade_size_gbp",
    "equity_pct",
    "fx_exposure_pct",
    "after_hours_pct",
    "new_counterparty_pct",
    "concentration_score",
]

df = generate_portfolio_data(500)
X_raw = df[FEATURES].values
true_groups = df["true_group"].values

# Always scale before clustering - distances matter
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Dataset: {len(df)} portfolios | {len(FEATURES)} behavioural features")
print(f"True groups (hidden from model):")
for g, cnt in df["true_group"].value_counts().items():
    print(f"  {g:<25} {cnt:>4} portfolios")
print()


# ==================================================================
# SECTION 2 - K-Means Clustering
# ==================================================================
print(SEP)
print("SECTION 2 - K-Means Clustering")
print(SEP)

"""
K-Means: partition N points into K clusters by minimising
the sum of squared distances from each point to its cluster centre.

ALGORITHM:
  1. Place K centroids randomly.
  2. Assign each point to its nearest centroid.
  3. Move each centroid to the mean of its assigned points.
  4. Repeat 2-3 until centroids stop moving.

.NET ANALOGY:
  Imagine GROUP BY on a computed column where the column definition
  is learned iteratively. Each centroid is the "ideal representative"
  of its group.

CHOOSING K - The Elbow Method:
  Run K-Means for K=2 to K=10.
  Plot inertia (total within-cluster sum of squares).
  The "elbow" - where adding more clusters gives diminishing returns -
  is the optimal K.

LIMITATION:
  - Assumes spherical clusters of similar size.
  - Sensitive to outliers (anomalous portfolios pull centroids).
  - Must specify K in advance.
"""

print("Finding optimal K using Elbow Method + Silhouette Score:")
print(f"  {'K':>4}  {'Inertia':>12}  {'Silhouette':>12}  {'Recommendation'}")
print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*20}")

best_k = 3
best_silhouette = -1

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    sil = silhouette_score(X, labels)
    note = ""
    if sil > best_silhouette:
        best_silhouette = sil
        best_k = k
        note = "<-- best silhouette"
    print(f"  {k:>4}  {inertia:>12.1f}  {sil:>12.4f}  {note}")

print()
print(f"  Optimal K = {best_k} (highest silhouette score)")
print()

# Train final K-Means with optimal K
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(X)
df["kmeans_cluster"] = km_labels

print(f"K-Means clusters (K={best_k}):")
for cluster_id in sorted(df["kmeans_cluster"].unique()):
    mask = df["kmeans_cluster"] == cluster_id
    group_counts = df.loc[mask, "true_group"].value_counts()
    dominant = group_counts.index[0]
    print(f"  Cluster {cluster_id}: {mask.sum():>4} portfolios | "
          f"dominant group: {dominant}")

print()
print("INTERPRETATION:")
print("  K-Means found natural portfolio segments without being told they exist.")
print("  Compliance teams can now apply different monitoring rules per segment.")
print("  High-frequency hedge fund portfolios get tighter scrutiny than retail.")

# Cluster profile - what characterises each cluster
print()
print("Cluster profiles (mean feature values per cluster):")
cluster_means = df.groupby("kmeans_cluster")[FEATURES].mean()
for feature in ["avg_daily_trades", "avg_trade_size_gbp", "after_hours_pct"]:
    row = "  " + f"{feature:<28}"
    for c in sorted(cluster_means.index):
        val = cluster_means.loc[c, feature]
        row += f"  C{c}: {val:>8.3f}"
    print(row)


# ==================================================================
# SECTION 3 - DBSCAN
# ==================================================================
print()
print(SEP)
print("SECTION 3 - DBSCAN (Density-Based Clustering)")
print(SEP)

"""
DBSCAN: groups points that are closely packed, marks isolated points
        as noise/anomalies.

KEY DIFFERENCE FROM K-MEANS:
  - Does NOT require specifying K in advance.
  - Can find clusters of arbitrary shape (not just spheres).
  - Points that don't fit any cluster are labelled -1 (noise/anomalous).
  - Those noise points are your anomaly candidates.

TWO PARAMETERS:
  eps          - neighbourhood radius: how close must points be to be "neighbours"?
  min_samples  - minimum points in a neighbourhood to form a core point

.NET ANALOGY:
  Imagine a spatial index (like SQL Server's STContains) that finds
  groups of geographically close data points. Points with no neighbours
  within eps distance are flagged automatically.

ADVANTAGE FOR INVESTMENT MONITORING:
  - No need to decide K upfront.
  - Anomalous portfolios are automatically labelled -1 (noise).
  - Works well when clusters have different densities.

DISADVANTAGE:
  - Sensitive to eps and min_samples tuning.
  - Struggles in high-dimensional spaces (curse of dimensionality).
  - Slower than K-Means on very large datasets.
"""

# PCA to 2D first helps DBSCAN in high dimensions
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
variance_explained = pca.explained_variance_ratio_.sum()

db = DBSCAN(eps=0.8, min_samples=5)
db_labels = db.fit_predict(X_2d)
df["dbscan_cluster"] = db_labels

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = int((db_labels == -1).sum())

print(f"PCA variance explained by 2 components: {variance_explained:.1%}")
print(f"DBSCAN results (eps=0.8, min_samples=5):")
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {n_noise}  <-- these are anomaly candidates")
print()

# How many of the noise points are truly anomalous?
noise_mask = df["dbscan_cluster"] == -1
noise_groups = df.loc[noise_mask, "true_group"].value_counts()
print("  Composition of noise points (anomaly candidates):")
for group, count in noise_groups.items():
    print(f"    {group:<25} {count:>4}")

true_anomalous_in_noise = noise_groups.get("D_anomalous", 0)
total_anomalous = int((df["true_group"] == "D_anomalous").sum())
print()
print(f"  True anomalous portfolios: {total_anomalous}")
print(f"  Caught in noise:           {true_anomalous_in_noise} "
      f"({true_anomalous_in_noise/total_anomalous:.0%} recall)")


# ==================================================================
# SECTION 4 - Isolation Forest (Anomaly Detection)
# ==================================================================
print()
print(SEP)
print("SECTION 4 - Isolation Forest (Anomaly Detection)")
print(SEP)

"""
Isolation Forest: detects anomalies by isolating points.

CORE IDEA:
  Anomalies are rare and different. They are EASIER TO ISOLATE
  with random splits than normal points.

  Build N random decision trees. For each point, count how many
  splits it takes to isolate it in a tree (path length).
  Short path = isolated quickly = anomalous.
  Long path  = needs many splits = normal (similar to many others).

.NET ANALOGY:
  Like a binary search tree where easy-to-find items are anomalous.
  A unique portfolio (unusual feature values) gets partitioned away
  with just 1-2 splits. A typical portfolio needs 10-15 splits.

THIS IS EXACTLY WHAT src/models/anomaly_detector.py uses.
Now you understand what it does internally.

ADVANTAGES:
  - No labels required.
  - Linear time complexity O(n log n).
  - Works well in high dimensions.
  - Handles mixed anomaly types (not just outliers in one direction).

KEY PARAMETER:
  contamination - estimated fraction of anomalies (0.05 = expect 5%).
  Set this based on domain knowledge or use "auto".
"""

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # we know ~5% are anomalous
    random_state=42,
    n_jobs=-1,
)
iso_labels = iso.fit_predict(X)   # -1 = anomaly, 1 = normal
iso_scores = iso.decision_function(X)  # higher = more normal

# Convert to 0/1 flag: 1 = anomaly
df["iso_anomaly"] = (iso_labels == -1).astype(int)

# Normalise scores to 0-1 where 1 = most anomalous
df["iso_score"] = 1 - (
    (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
)

# Evaluate
flagged = df["iso_anomaly"].sum()
true_positives = int(((df["iso_anomaly"] == 1) & (df["true_group"] == "D_anomalous")).sum())
false_positives = int(((df["iso_anomaly"] == 1) & (df["true_group"] != "D_anomalous")).sum())
total_anomalous = int((df["true_group"] == "D_anomalous").sum())
missed = total_anomalous - true_positives

print(f"Isolation Forest results (contamination=0.05):")
print(f"  Total flagged       : {flagged}")
print(f"  True positives      : {true_positives}  (genuine anomalies caught)")
print(f"  False positives     : {false_positives}  (normal portfolios wrongly flagged)")
print(f"  Missed anomalies    : {missed}")
print()
precision = true_positives / flagged if flagged else 0
recall    = true_positives / total_anomalous if total_anomalous else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  Precision : {precision:.2%}  "
      f"(of flagged portfolios, % are truly anomalous)")
print(f"  Recall    : {recall:.2%}  "
      f"(of all anomalous portfolios, % were caught)")
print(f"  F1-Score  : {f1:.2%}")

# Show top anomalies with scores
print()
print("Top 10 most anomalous portfolios:")
print(f"  {'#':>3}  {'Anomaly Score':>14}  {'True Group':<25}  Flagged?")
print(f"  {'-'*3}  {'-'*14}  {'-'*25}  {'-'*8}")
top_anomalies = df.nlargest(10, "iso_score")[["iso_score", "true_group", "iso_anomaly"]]
for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
    flag = "YES" if row["iso_anomaly"] == 1 else "no"
    print(f"  {i:>3}  {row['iso_score']:>14.4f}  {row['true_group']:<25}  {flag}")


# ==================================================================
# SECTION 5 - PCA: Dimensionality Reduction
# ==================================================================
print()
print(SEP)
print("SECTION 5 - PCA: Dimensionality Reduction")
print(SEP)

"""
PCA (Principal Component Analysis): reduce many features to a smaller
number of components that capture most of the variance.

USE CASES IN YOUR PLATFORM:
  1. Visualise 7-dimensional portfolio data in 2D for dashboards.
  2. Remove correlated/redundant features before clustering.
  3. Compress feature vectors to reduce memory/compute cost.
  4. Understand which combinations of features explain portfolio variance.

HOW IT WORKS:
  Finds linear combinations of features (principal components) that
  maximise variance in each successive direction.
  PC1 = direction of maximum variance.
  PC2 = direction of maximum remaining variance, orthogonal to PC1.

.NET ANALOGY:
  Like a custom ORDER BY DENSE_RANK() across multiple correlated
  dimensions - collapsing redundant information while preserving
  the most discriminating signal.

IMPORTANT:
  PCA is a preprocessing step, NOT a model.
  You cannot ask "which portfolio is anomalous?" from PCA alone.
  Combine PCA -> Isolation Forest or PCA -> K-Means.
"""

for n_components in [2, 3, 4, 5, 7]:
    pca_n = PCA(n_components=n_components, random_state=42)
    pca_n.fit(X)
    total_var = pca_n.explained_variance_ratio_.sum()
    bar = "#" * int(total_var * 40)
    print(f"  {n_components} components: {total_var:.1%} variance explained  {bar}")

print()
print("  RECOMMENDATION: use 3-4 components for clustering (keeps ~85% variance)")
print("  while reducing noise from low-variance features.")

# Feature loadings for PC1 and PC2
pca_2 = PCA(n_components=2, random_state=42)
pca_2.fit(X)
print()
print("  PC1 loadings (which original features drive the first component):")
pc1_loadings = pd.Series(
    np.abs(pca_2.components_[0]), index=FEATURES
).sort_values(ascending=False)
for feat, loading in pc1_loadings.items():
    bar = "#" * int(loading * 30)
    print(f"    {feat:<28} {loading:.3f}  {bar}")


# ==================================================================
# SECTION 6 - Algorithm Comparison
# ==================================================================
print()
print(SEP)
print("SECTION 6 - Choosing the Right Unsupervised Algorithm")
print(SEP)

COMPARISON = [
    ("K-Means",          "Portfolio segmentation",        "Must specify K",         "Fast, interpretable"),
    ("DBSCAN",           "Find natural clusters + noise",  "Tune eps/min_samples",   "No K needed, finds noise"),
    ("Isolation Forest", "Anomaly/fraud detection",        "Tune contamination",     "No labels, scales well"),
    ("PCA",              "Visualisation, compression",     "Loses interpretability", "Preprocessing step only"),
    ("Autoencoders",     "Complex anomaly detection",      "Needs deep learning",    "Handles unstructured data"),
    ("UMAP",             "Visualisation (non-linear)",     "Non-deterministic",      "Better than PCA for viz"),
]

print(f"  {'Algorithm':<20} {'Best Use Case':<32} {'Weakness':<26} Strength")
print(f"  {'-'*20} {'-'*32} {'-'*26} {'-'*30}")
for row in COMPARISON:
    print(f"  {row[0]:<20} {row[1]:<32} {row[2]:<26} {row[3]}")

print()
print("  FOR YOUR INVESTMENT PLATFORM:")
print("  - Segment portfolios into risk tiers     -> K-Means (K=3-5)")
print("  - Flag unusual transaction patterns      -> Isolation Forest")
print("  - Discover new behaviour clusters        -> DBSCAN")
print("  - Visualise portfolio risk landscape     -> PCA or UMAP -> scatter plot")
print("  - All of the above combined in pipeline  -> PCA -> K-Means + Isolation Forest")


# ==================================================================
# SECTION 7 - Combined Pipeline (what production looks like)
# ==================================================================
print()
print(SEP)
print("SECTION 7 - Production Pipeline: Segment + Flag")
print(SEP)

"""
PRODUCTION PATTERN for investment portfolio monitoring:
  1. Scale features (StandardScaler)
  2. Reduce dimensions (PCA to 4 components)
  3. Cluster portfolios (K-Means, K=3)
  4. Assign cluster-specific monitoring thresholds
  5. Flag anomalies within each cluster (Isolation Forest)
  6. Alert operations team with cluster context

This is exactly the pattern used in src/models/anomaly_detector.py.
"""

from sklearn.pipeline import Pipeline as SKPipeline

# Build the production pipeline
pipeline = SKPipeline([
    ("scaler",    StandardScaler()),
    ("pca",       PCA(n_components=4, random_state=42)),
    ("iso_forest", IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_raw)
pipe_labels = pipeline.predict(X_raw)
df["pipeline_anomaly"] = (pipe_labels == -1).astype(int)

pipe_tp = int(((df["pipeline_anomaly"] == 1) & (df["true_group"] == "D_anomalous")).sum())
pipe_fp = int(((df["pipeline_anomaly"] == 1) & (df["true_group"] != "D_anomalous")).sum())
pipe_p  = pipe_tp / (pipe_tp + pipe_fp) if (pipe_tp + pipe_fp) > 0 else 0
pipe_r  = pipe_tp / total_anomalous if total_anomalous > 0 else 0

print("Pipeline: StandardScaler -> PCA(4) -> IsolationForest")
print(f"  Precision : {pipe_p:.2%}")
print(f"  Recall    : {pipe_r:.2%}")
print()
print("  This pipeline is serialisable with joblib:")
print("  import joblib")
print("  joblib.dump(pipeline, 'models/trained/portfolio_anomaly.joblib')")
print("  Loaded by inference_pipeline.py at API startup.")


# ==================================================================
# SECTION 8 - Five Interview Questions
# ==================================================================
print()
print(SEP)
print("SECTION 8 - Interview Questions")
print(SEP)

QA = [
    (
        "Q1. When would you choose unsupervised over supervised learning?",
        """When labelled data is unavailable, expensive or insufficient.
   For example, in fraud detection you may have 100,000 transactions
   but only 50 confirmed fraud cases. You cannot train a reliable
   supervised model on 50 positive examples.
   Isolation Forest can flag anomalies across all 100,000 without labels.
   Also use unsupervised learning for discovery tasks where you don't
   know the categories in advance (e.g. portfolio segmentation).""",
    ),
    (
        "Q2. What is the difference between K-Means and DBSCAN?",
        """K-Means: requires K upfront, assumes spherical clusters of similar
   size, sensitive to outliers. Fast, interpretable centroids.
   DBSCAN: discovers K automatically, finds arbitrary-shaped clusters,
   flags isolated points as noise/anomalies. Requires tuning eps and
   min_samples. Slower on large datasets.
   For portfolio segmentation I'd use K-Means (interpretable, fast).
   For transaction anomaly detection I'd use DBSCAN or Isolation Forest
   because I don't know K and I want the noise points flagged.""",
    ),
    (
        "Q3. How does Isolation Forest detect anomalies?",
        """It builds random decision trees and measures how many splits it takes
   to isolate each data point. Anomalies are rare and unusual, so they
   get isolated in fewer splits (shorter path length).
   Normal points blend in with the majority and need many splits.
   The anomaly score = average path length across all trees, normalised.
   It's efficient (O(n log n)), requires no labels, and works well in
   high dimensions - which is why it's the default in our anomaly_detector.py.""",
    ),
    (
        "Q4. What is PCA and when would you use it?",
        """PCA transforms features into orthogonal principal components ordered
   by variance explained. It's used for:
   1. Dimensionality reduction - reduce 50 features to 10 for faster ML
   2. Noise removal - drop low-variance components
   3. Visualisation - reduce to 2D for dashboards and reports
   4. Multicollinearity removal before regression
   It's a preprocessing step, not a model. The components are linear
   combinations of original features, which can reduce interpretability.""",
    ),
    (
        "Q5. How would you deploy a clustering model in production on AWS?",
        """The challenge: clustering is typically done offline (batch), not per-request.
   Approach:
   1. Run K-Means weekly via AWS Glue/Lambda on latest portfolio behaviour data.
   2. Store cluster assignments in RDS/DynamoDB per portfolio_id.
   3. Run Isolation Forest daily for anomaly scores.
   4. .NET 8 API reads cluster + anomaly score from cache (ElastiCache).
   5. Use SageMaker Batch Transform for large-scale weekly scoring.
   6. Monitor cluster drift: if % of portfolios in each cluster shifts >10%,
      trigger retraining via EventBridge -> Lambda -> SageMaker pipeline.""",
    ),
]

for question, answer in QA:
    print(f"\n  {question}")
    print(f"  {'-' * (len(question) - 2)}")
    for line in answer.strip().split("\n"):
        print(f"  {line}")


# ==================================================================
# SECTION 9 - Exercises
# ==================================================================
print()
print(SEP)
print("SECTION 9 - Exercises")
print(SEP)
print("""
  EXERCISE 1 (10 min): K-Means elbow exploration.
    Change the range to k=2..12. Where does the elbow appear?
    Try contamination=0.10 in Isolation Forest. What changes in
    precision and recall? When would a higher contamination be right?

  EXERCISE 2 (15 min): Connect to your project data.
    Run:  python -X utf8 scripts/generate_sample_data.py --n 5000
    Load data/raw/transactions.csv into this script.
    Group by portfolio_id, compute per-portfolio feature averages.
    Run K-Means + Isolation Forest on the result.
    How many portfolio clusters emerge? Which ones look anomalous?

  EXERCISE 3 (10 min): Compare Isolation Forest thresholds.
    For each flagged anomaly (iso_anomaly == 1), look at which features
    are most extreme using:
        anomalies = df[df['iso_anomaly'] == 1][FEATURES]
        print(anomalies.describe())
    Which features have the most extreme values in anomalous portfolios?

  EXERCISE 4 (5 min): The src/ connection.
    Open src/models/anomaly_detector.py.
    You now understand every line of code in that file.
    The fit() method = IsolationForest.fit() on scaled features.
    The predict() method = decision_function() -> anomaly_score.
    The save()/load() = joblib serialisation of the pipeline.
""")

print(SEP)
print("NEXT LESSON: lesson04_neural_networks.py")
print("  -> What a neuron actually is (not the biology)")
print("  -> Layers, activations, forward propagation")
print("  -> Backpropagation - the algorithm behind every LLM")
print("  -> Build a 2-layer neural network from scratch with NumPy")
print("  -> The same network in PyTorch")
print("  -> Direct path: MLP -> Transformer -> GPT-4")
print(SEP)
