"""
train_model.py
--------------
AgriLoan Credit Scorer — ML Model Training Script

Trains an XGBoost classifier on a synthetically generated agricultural-loan
dataset, prints evaluation metrics, and saves the trained model pipeline as
model/credit_model.pkl.

NOTE: The spec requires XGBoost. This script tries to import xgboost first.
If xgboost is not installed, it falls back to sklearn GradientBoostingClassifier
which is API-compatible (same predict_proba interface). Install xgboost via:
    pip install xgboost

Feature vector (10 features – must match services/feature_engineering.py):
    0  land_size           – acres
    1  crop_diversity      – number of crops
    2  average_yield       – quintals/acre (3-year average)
    3  soil_ph             – pH value
    4  rainfall            – mm/year
    5  irrigation_score    – 1 (rain-fed) to 4 (drip)
    6  ownership_score     – 1 (tenant) to 3 (owner)
    7  drought_risk_score  – 1 (low) to 3 (high)
    8  nitrogen_score      – 1 (low) to 3 (high)
    9  yield_trend         – slope of 3-year yield series

Run:
    python train_model.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Try to use real XGBoost if available ─────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _BASE_CLF = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    _MODEL_LABEL = "XGBoost (xgboost)"
except ImportError:
    _BASE_CLF = GradientBoostingClassifier(
        n_estimators  = 300,
        max_depth     = 5,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = 42,
    )
    _MODEL_LABEL = "GradientBoostingClassifier (XGBoost-compatible fallback)"

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES   = 8000
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "credit_model.pkl")

FEATURE_NAMES = [
    "land_size", "crop_diversity", "average_yield", "soil_ph",
    "rainfall", "irrigation_score", "ownership_score",
    "drought_risk_score", "nitrogen_score", "yield_trend",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic dataset generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic agricultural-loan repayment dataset.

    Each row represents one loan application. The binary target label
    (repaid=1 / defaulted=0) is derived from a weighted score of all
    features plus Gaussian noise, then thresholded.

    Args:
        n    (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame with feature columns + 'repaid' target column.
    """
    rng = np.random.default_rng(seed)

    # ── Generate feature distributions ────────────────────────────────────
    land_size           = rng.uniform(0.5,  50.0,  n)
    crop_diversity      = rng.integers(1, 9,       n).astype(float)
    average_yield       = rng.uniform(5.0,  60.0,  n)
    soil_ph             = rng.uniform(5.5,   8.5,  n)
    rainfall            = rng.uniform(300, 3000,   n)
    irrigation_score    = rng.integers(1, 5,        n).astype(float)
    ownership_score     = rng.integers(1, 4,        n).astype(float)
    drought_risk_score  = rng.integers(1, 4,        n).astype(float)
    nitrogen_score      = rng.integers(1, 4,        n).astype(float)
    yield_trend         = rng.uniform(-5.0, 5.0,   n)

    # ── Compute repayment score (domain knowledge-driven weights) ─────────
    # Each term normalised to [0,1] before weighting
    score = (
          0.18 * (land_size          / 50.0)
        + 0.10 * (crop_diversity     / 8.0)
        + 0.20 * (average_yield      / 60.0)
        + 0.08 * np.where(np.abs(soil_ph - 7.0) < 0.8, 1.0, 0.4)
        + 0.10 * (rainfall           / 3000.0)
        + 0.12 * (irrigation_score   / 4.0)
        + 0.12 * (ownership_score    / 3.0)
        - 0.06 * ((drought_risk_score - 1.0) / 2.0)   # high drought → bad
        + 0.06 * (nitrogen_score     / 3.0)
        + 0.04 * ((yield_trend + 5.0) / 10.0)
    )

    # Add Gaussian noise and binarise at 0.53 (slight class imbalance)
    score  += rng.normal(0, 0.06, n)
    repaid  = (score > 0.53).astype(int)

    df = pd.DataFrame({
        "land_size":          land_size,
        "crop_diversity":     crop_diversity,
        "average_yield":      average_yield,
        "soil_ph":            soil_ph,
        "rainfall":           rainfall,
        "irrigation_score":   irrigation_score,
        "ownership_score":    ownership_score,
        "drought_risk_score": drought_risk_score,
        "nitrogen_score":     nitrogen_score,
        "yield_trend":        yield_trend,
        "repaid":             repaid,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save() -> float:
    """
    Full training pipeline:
        1. Generate synthetic dataset
        2. Train/test split (80/20, stratified)
        3. Fit StandardScaler + classifier Pipeline
        4. Evaluate (accuracy, AUC, classification report)
        5. 5-fold cross-validation AUC
        6. Save Pipeline to MODEL_PATH via pickle

    Returns:
        float: Test-set accuracy.
    """
    _banner("AgriLoan Credit Scorer — Model Training")
    print(f"  Model     : {_MODEL_LABEL}")
    print(f"  Samples   : {N_SAMPLES:,}")
    print(f"  Output    : {MODEL_PATH}")

    # ── Generate data ──────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic dataset …")
    df = generate_dataset()
    X  = df[FEATURE_NAMES]
    y  = df["repaid"]
    print(f"       Shape : {X.shape}")
    print(f"       Repaid: {y.sum():,} / Defaulted: {(y==0).sum():,}  "
          f"(ratio {y.mean():.1%})")

    # ── Train / test split ─────────────────────────────────────────────────
    print("\n[2/5] Splitting data (80% train / 20% test) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = RANDOM_SEED,
        stratify     = y,
    )
    print(f"       Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── Build pipeline (scaler + model bundled together) ───────────────────
    print("\n[3/5] Building pipeline (StandardScaler + Classifier) …")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    _BASE_CLF),
    ])

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training model …")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating model …")
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)

    print(f"\n  ✅  Test Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  ✅  ROC-AUC Score  : {roc_auc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Defaulted", "Repaid"],
        digits=4,
    ))

    # 5-fold cross-validation on full dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  5-Fold CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Persist model ──────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n  💾  Model saved → {MODEL_PATH}")
    _banner("Training Complete")
    return accuracy


def _banner(title: str) -> None:
    width = 62
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    acc = train_and_save()
    sys.exit(0 if acc > 0.50 else 1)