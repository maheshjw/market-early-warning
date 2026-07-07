"""
03_modeling.py
--------------
Train XGBoost, Random Forest, and Logistic Regression models for
early warning signal detection.

Key design decisions:
  - TimeSeriesSplit CV (no lookahead bias — critical for financial ML)
  - class_weight='balanced' + SMOTE to handle class imbalance
  - Train on pre-2015 data, test on 2015-2024 for final holdout evaluation
  - Save all model artifacts to models/ directory

Author: Mahesh Jagdish Wadhokar
BA889 — Boston University Questrom, June 2026
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: xgboost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling  import SMOTE
    from imblearn.pipeline       import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    print("WARNING: imbalanced-learn not available. SMOTE disabled.")
    SMOTE_AVAILABLE = False

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
FIG_DIR   = "reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("[1/4] Loading processed dataset...")

dataset_path = os.path.join(PROC_DIR, "dataset_features_labels.csv")
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Run 02_feature_engineering.py first.")

dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
FEATURE_COLS = [c for c in dataset.columns if c != "crash_label"]

X = dataset[FEATURE_COLS]
y = dataset["crash_label"]

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Label distribution: {dict(y.value_counts().sort_index())}")
print(f"  Crash rate: {y.mean()*100:.1f}%")

# ── Train/Test split (strict temporal — pre-2015 train, 2015+ test) ──────────
train_mask = dataset.index < "2015-01-01"
test_mask  = dataset.index >= "2015-01-01"

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n  Train: {X_train.shape[0]} samples ({X_train.index.min().year}–{X_train.index.max().year})")
print(f"  Test:  {X_test.shape[0]} samples ({X_test.index.min().year}–{X_test.index.max().year})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Define models
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Defining models...")

# TimeSeriesSplit for cross-validation (preserves temporal ordering)
tscv = TimeSeriesSplit(n_splits=5)

scaler = StandardScaler()

models = {}

# Model A: Logistic Regression (interpretable baseline)
models["Logistic Regression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.1,           # regularization
        random_state=42
    ))
])

# Model B: Random Forest (ensemble baseline)
models["Random Forest"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Model C: XGBoost (primary model)
if XGBOOST_AVAILABLE:
    # Calculate scale_pos_weight for imbalanced classes (equivalent to class_weight)
    neg_count   = (y_train == 0).sum()
    pos_count   = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        ))
    ])
else:
    print("  XGBoost skipped (not installed).")

print(f"  Models defined: {list(models.keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Cross-validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Running TimeSeriesSplit cross-validation...")
print("  (5 folds, preserving temporal order — no lookahead bias)")

cv_results = {}

for name, pipeline in models.items():
    print(f"\n  ── {name} ──")

    # PR-AUC via cross_val_score with "average_precision" scoring
    pr_auc_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=tscv,
        scoring="average_precision",
        n_jobs=-1
    )
    roc_auc_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1
    )
    f1_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=tscv,
        scoring="f1",
        n_jobs=-1
    )

    cv_results[name] = {
        "pr_auc_mean"   : pr_auc_scores.mean(),
        "pr_auc_std"    : pr_auc_scores.std(),
        "roc_auc_mean"  : roc_auc_scores.mean(),
        "roc_auc_std"   : roc_auc_scores.std(),
        "f1_mean"       : f1_scores.mean(),
        "f1_std"        : f1_scores.std(),
    }

    print(f"  PR-AUC:  {pr_auc_scores.mean():.3f} ± {pr_auc_scores.std():.3f}")
    print(f"  ROC-AUC: {roc_auc_scores.mean():.3f} ± {roc_auc_scores.std():.3f}")
    print(f"  F1:      {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# Save CV results
cv_df = pd.DataFrame(cv_results).T
cv_df.to_csv(os.path.join(PROC_DIR, "cv_results.csv"))
print(f"\n  CV results saved to {PROC_DIR}/cv_results.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Final fit + holdout evaluation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Fitting on full training set and evaluating on holdout (2015-2024)...")

fitted_models   = {}
holdout_results = {}
probas          = {}

for name, pipeline in models.items():
    print(f"\n  ── {name} ──")

    # Fit on full training data
    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline

    # Predict on test set
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    probas[name] = y_proba

    # Metrics
    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred)

    holdout_results[name] = {
        "PR_AUC"  : pr_auc,
        "ROC_AUC" : roc_auc,
        "F1"      : f1,
    }

    print(f"  PR-AUC:  {pr_auc:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  F1:      {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Crash", "Pre-Crash"],
                                 digits=3))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"  Saved: {model_path}")

# Save holdout results
holdout_df = pd.DataFrame(holdout_results).T
holdout_df.to_csv(os.path.join(PROC_DIR, "holdout_results.csv"))

# Save probabilities for evaluation script
proba_df = pd.DataFrame(probas, index=X_test.index)
proba_df.to_csv(os.path.join(PROC_DIR, "test_probabilities.csv"))
proba_df_all = pd.DataFrame({
    name: pipeline.predict_proba(X)[:, 1]
    for name, pipeline in fitted_models.items()
}, index=X.index)
proba_df_all.to_csv(os.path.join(PROC_DIR, "all_probabilities.csv"))

print("\n" + "="*60)
print("MODELING COMPLETE")
print("="*60)
print("\nHoldout performance summary:")
print(holdout_df.round(3).to_string())
print("\nNext step: python 04_evaluation.py")
