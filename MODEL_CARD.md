# Model Card: Telecom Customer Churn Classifier

## Overview

- **Task:** Binary classification of telecom customer churn.
- **Model implementation:** scikit-learn pipeline.
- **Model artifact:** `artifacts/churn_model.joblib`.
- **Recorded model version:** 1.0.0.
- **Decision threshold:** 0.30.

## Intended Use

The model is intended to help explore customer churn risk and prioritize
customers for further review or retention outreach. It should support,
not replace, business judgment.

## Evaluation Results

Recorded evaluation metrics at threshold 0.30:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.842 |
| Recall | 0.930 |
| Precision | 0.433 |
| F1-score | 0.591 |
| Accuracy | 0.659 |

Confusion matrix counts: 580 true negatives, 455 false positives,
26 false negatives, and 348 true positives.

The threshold favors recall, meaning more actual churners are identified,
but more non-churners are also flagged.

## Decision-Cost Assumptions

The experiment used illustrative costs of 5 units per missed churner and
1 unit per unnecessary offer. These assumptions are not verified business
costs and should be replaced with validated operational estimates.

## Data

The project uses the IBM Telco Customer Churn dataset. The raw CSV is not
committed to the repository. Dataset provenance, permitted use, and any
redistribution requirements should be documented before public release.

## Monitoring

The API provides a PSI-based drift assessment endpoint using a saved
reference profile. Thresholds of 0.10 and 0.20 are monitoring heuristics,
not proof that a model has become inaccurate. Assess drift on a meaningful
sample and monitor predictive performance when labels become available.

## Limitations and Risks

- Results may change across data splits and customer populations.
- A high predicted probability does not guarantee churn.
- False positives may cause unnecessary retention outreach.
- False negatives may miss customers who later churn.
- Check model performance across relevant customer groups before operational
  use.
- Do not use predictions as the sole basis for consequential decisions.

## Versioning and Retraining

Keep the model, metadata, evaluation results, and reference drift profile
together. Retrain only with documented data and evaluation procedures.
Compare candidate models against the current version before deployment.
