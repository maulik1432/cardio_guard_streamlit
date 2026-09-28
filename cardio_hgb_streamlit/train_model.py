"""Train a HistGradientBoosting cardiovascular disease classifier.

Usage:
    python train_model.py
    python train_model.py cardio_train.csv

The training pipeline intentionally does NOT calculate a noise score or use
IQR-based outlier removal. It only handles missing numeric values and creates
stable features used by the Streamlit app.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "cardio_train.csv"
MODEL_OUT = "model.pkl"
METRICS_OUT = "metrics.json"

TARGET = "cardio"
ID_COL = "id"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    required = {
        "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact feature representation used by the web app."""
    x = df.copy()

    # The original dataset stores age in days. Convert to years for readability
    # and a more natural feature scale. No StandardScaler is required for HGB.
    x["age_years"] = x["age"] / 365.25

    # Useful deterministic clinical features; these are not noise calculations.
    height_m = x["height"] / 100.0
    x["bmi"] = x["weight"] / (height_m ** 2)
    x["pulse_pressure"] = x["ap_hi"] - x["ap_lo"]
    x["bp_ratio"] = x["ap_hi"] / x["ap_lo"].replace(0, np.nan)

    feature_cols = [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
        "bmi", "pulse_pressure", "bp_ratio"
    ]
    return x[feature_cols]


def main() -> None:
    print(f"Loading: {CSV_PATH}")
    df = load_data(CSV_PATH)

    X = make_features(df)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # HistGradientBoosting does not need feature scaling.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(random_state=42))
    ])

    param_dist = {
        "model__learning_rate": [0.03, 0.05, 0.07, 0.10],
        "model__max_iter": [200, 300, 400, 500],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 30, 50, 80],
        "model__l2_regularization": [0, 0.5, 1, 2, 5],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="accuracy",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("Tuning HistGradientBoosting...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nBest CV accuracy:", round(search.best_score_, 4))
    print("Test accuracy:", round(accuracy, 4))
    print("Test ROC-AUC:", round(auc, 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))
    print("Best parameters:")
    print(search.best_params_)

    bundle = {
        "model": best_model,
        "feature_cols": list(X.columns),
        "target": TARGET,
        "dataset_schema": list(df.columns),
        "model_name": "HistGradientBoostingClassifier",
    }
    joblib.dump(bundle, MODEL_OUT)

    metrics = {
        "model": "HistGradientBoostingClassifier",
        "best_cv_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy),
        "test_roc_auc": float(auc),
        "test_size": 0.20,
        "random_state": 42,
        "best_params": search.best_params_,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "noise_calculation": False,
        "iqr_outlier_filter": False,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved: {MODEL_OUT}")
    print(f"Saved: {METRICS_OUT}")
    print("Run Streamlit with: streamlit run app.py")


if __name__ == "__main__":
    main()
