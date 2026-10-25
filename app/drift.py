
import numpy as np
import pandas as pd


def calculate_psi(expected, actual, epsilon=1e-6):
    """Calculate Population Stability Index for two distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)

    expected = expected / expected.sum()
    actual = actual / actual.sum()

    return float(np.sum(
        (actual - expected) * np.log(actual / expected)
    ))


def assess_drift(reference, recent_data):
    """Compare recent feature distributions with the training reference."""
    results = {}

    for column, profile in reference.items():
        if column not in recent_data.columns:
            results[column] = {
                "psi": None,
                "status": "missing_feature",
            }
            continue

        series = recent_data[column]
        expected_profile = profile["proportions"]

        if profile["type"] == "numeric":
            edges = np.array([
                -np.inf if x is None else
                np.inf if x is None and i == len(profile["edges"]) - 1
                else x
                for i, x in enumerate(profile["edges"])
            ], dtype=float)

            # JSON stores both infinite end boundaries as null.
            edges[0] = -np.inf
            edges[-1] = np.inf

            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                results[column] = {
                    "psi": None,
                    "status": "no_valid_values",
                }
                continue

            counts, _ = np.histogram(values, bins=edges)
            actual = counts / counts.sum()
            expected = np.asarray(expected_profile, dtype=float)

        else:
            values = series.fillna("__MISSING__").astype(str)
            categories = list(expected_profile.keys())

            counts = values.value_counts()
            total = len(values)

            actual = np.array([
                counts.get(category, 0) / total
                for category in categories
            ], dtype=float)

            # Account for values not seen during training.
            other_count = sum(
                int(count)
                for category, count in counts.items()
                if category not in categories
            )

            if "__OTHER__" in categories:
                other_index = categories.index("__OTHER__")
                actual[other_index] += other_count / total

            expected = np.asarray([
                expected_profile[category]
                for category in categories
            ], dtype=float)

        psi = calculate_psi(expected, actual)

        if psi >= 0.2:
            status = "high_drift"
        elif psi >= 0.1:
            status = "warning"
        else:
            status = "stable"

        results[column] = {
            "psi": round(psi, 6),
            "status": status,
        }

    return {
        "features_checked": len(results),
        "drift_detected": any(
            item["psi"] is not None and item["psi"] >= 0.1
            for item in results.values()
        ),
        "features": results,
    }
