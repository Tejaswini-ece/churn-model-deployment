
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = ROOT / "artifacts" / "drift_reference.json"

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop_duplicates()
df = df.drop(columns=["customerID", "Churn"])

reference = {}

for column in df.columns:
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna().astype(float).to_numpy()

        # Quantile boundaries for numeric PSI bins.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([values.min() - 1, values.max() + 1])

        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts / max(counts.sum(), 1)).tolist()

        reference[column] = {
            "type": "numeric",
            "edges": [
                float(x) if np.isfinite(x) else None for x in edges
            ],
            "proportions": proportions,
        }
    else:
        values = series.fillna("__MISSING__").astype(str)
        counts = values.value_counts(normalize=True)

        # Reserve a category for previously unseen values.
        proportions = counts.to_dict()
        proportions["__OTHER__"] = 0.0

        reference[column] = {
            "type": "categorical",
            "proportions": proportions,
        }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(reference, indent=2))

print(f"Saved drift reference: {OUTPUT_PATH}")
print(f"Features profiled: {len(reference)}")
