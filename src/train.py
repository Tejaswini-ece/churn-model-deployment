import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# Business assumptions: adjust these with your internship mentor.
MISSED_CHURN_COST = 5.0
UNNECESSARY_OFFER_COST = 1.0

# 1. Load and validate data
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required = {"customerID", "Churn", "TotalCharges"}
if not required.issubset(df.columns):
    raise ValueError(f"Dataset is missing columns: {required - set(df.columns)}")

df = df.drop_duplicates().copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["Churn"])

# Do not use customerID as a predictive feature.
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Churn must contain only Yes/No labels.")

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 2. Separate train, validation and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

# Result: 60% train, 20% validation, 20% test.

# 3. Build preprocessing and classification pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    )),
])

# 4. Tune hyperparameters using cross-validation on training data only
search = GridSearchCV(
    pipeline,
    param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)
model = search.best_estimator_

# 5. Select threshold on validation data using business costs
val_prob = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.05, 0.96, 0.01)

def business_cost(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    actual = np.asarray(y_true)
    false_negatives = int(((actual == 1) & (predictions == 0)).sum())
    false_positives = int(((actual == 0) & (predictions == 1)).sum())
    cost = (
        MISSED_CHURN_COST * false_negatives
        + UNNECESSARY_OFFER_COST * false_positives
    )
    return cost, false_negatives, false_positives

threshold_results = [
    (business_cost(y_val, val_prob, t)[0], t)
    for t in thresholds
]
best_cost, threshold = min(threshold_results)

# 6. Evaluate once on the held-out test set
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

metrics = {
    "accuracy": float(accuracy_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_business_cost": float(
        MISSED_CHURN_COST * fn + UNNECESSARY_OFFER_COST * fp
    ),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

# 7. Save model and reproducibility metadata
model_path = ARTIFACT_DIR / "churn_model.joblib"
joblib.dump(model, model_path)

dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

metadata = {
    "model_version": "1.0.0",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": dataset_hash,
    "dataset_rows_after_cleaning": int(len(df)),
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "best_parameters": search.best_params_,
    "cross_validation_roc_auc": float(search.best_score_),
    "decision_threshold": float(threshold),
    "threshold_selection_validation_cost": float(best_cost),
    "missed_churn_cost_assumption": MISSED_CHURN_COST,
    "unnecessary_offer_cost_assumption": UNNECESSARY_OFFER_COST,
    "metrics": metrics,
    "library_versions": {
        "python": __import__("sys").version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    },
}

with open(ARTIFACT_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete!")
print("Best parameters:", search.best_params_)
print("Chosen threshold:", round(threshold, 2))
print("Validation business cost:", best_cost)
print("\nHeld-out test metrics:")
print(json.dumps(metrics, indent=2))
print("\nModel saved to:", model_path)
print("Metadata saved to:", ARTIFACT_DIR / "model_metadata.json")