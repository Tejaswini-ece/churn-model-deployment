# Telecom Customer Churn Prediction API

A machine-learning deployment project that predicts customer churn using
scikit-learn and exposes predictions through a secure FastAPI service.
The project includes input validation, batch predictions, health checks,
drift assessment, automated tests, and Docker deployment.

## Features

- Trained scikit-learn churn classification pipeline
- FastAPI REST API with API-key authentication
- Single-customer and batch prediction endpoints
- Input validation using Pydantic
- Health endpoint and request logging
- Population Stability Index (PSI)-based drift assessment
- Docker image with a non-root runtime user
- Automated API and drift tests
- OpenAPI specification at `openapi.json`

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── drift.py
├── artifacts/
│   ├── churn_model.joblib
│   ├── model_metadata.json
│   └── drift_reference.json
├── data/                  # Dataset excluded from Git
├── src/
│   ├── train.py
│   └── build_drift_reference.py
├── tests/
│   ├── test_api.py
│   └── benchmark_latency.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-prod.txt
├── openapi.json
└── README.md
```

## Dataset

This project uses the IBM Telco Customer Churn dataset, commonly distributed
as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Place the CSV at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The raw dataset is excluded from Git commits. Obtain the dataset from its
authorized source before retraining the model.

## Model Evaluation

The recorded evaluation at decision threshold 0.30 was:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.842 |
| Recall | 0.930 |
| Precision | 0.433 |
| F1-score | 0.591 |
| Accuracy | 0.659 |
| True negatives | 580 |
| False positives | 455 |
| False negatives | 26 |
| True positives | 348 |

The threshold prioritizes finding likely churners, at the expense of more
false-positive predictions. The recorded business-cost assumptions were
5 units for a missed churner and 1 unit for an unnecessary retention offer.
These are illustrative assumptions, not measured financial costs.

Metrics depend on the dataset split and evaluation procedure. They do not
guarantee performance on future customers.

## Setup for Development

Use a compatible Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Ensure the required model artifacts exist in `artifacts/` before starting
the API.

## Run the API

Set an API key in your terminal:

```bash
export CHURN_API_KEY="replace-with-a-strong-secret"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The example key used during local testing is for development only.
Do not use it in a public deployment.

Interactive API documentation is available at `/docs`.
The OpenAPI schema is available at `/openapi.json`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for multiple customers |
| POST | `/drift` | Assess input feature drift against the reference |

Prediction endpoints require the `X-API-Key` request header.
Check `/docs` for the exact request schemas and response formats.

Example health check:

```bash
curl http://localhost:8000/health
```

Example authenticated prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-a-strong-secret" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0
  }'
```

## Run Tests

Install development dependencies using `requirements.txt`, then run:

```bash
pytest -v
```

The current test run completed with 21 passing tests.

## Latency Benchmark

With the API running locally, execute:

```bash
python tests/benchmark_latency.py
```

The previously recorded benchmark reported a median latency of 10.75 ms
and a p95 latency of 15.21 ms over 100 measured requests. These are
environment-specific measurements, not a production latency guarantee.
Rerun the benchmark after deployment changes.

## Docker Deployment

Build the optimized image:

```bash
docker build -t churn-api:1.0.1 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e CHURN_API_KEY="replace-with-a-strong-secret" \
  churn-api:1.0.1
```

Then check `http://localhost:8000/health`.

The optimized image was measured at 470 MB in the development environment.
Image size may vary with the build environment and platform.

## Drift Monitoring

The `/drift` endpoint compares submitted feature distributions against the
saved reference profile using Population Stability Index (PSI).

The configured thresholds are intended as monitoring heuristics:
- PSI below 0.10: low drift
- PSI from 0.10 to below 0.20: warning
- PSI at or above 0.20: high drift

A single request is not sufficient to establish meaningful population drift.
Use an appropriate batch of recent production data and investigate flagged
features before deciding whether to retrain.

## Limitations and Responsible Use

- Predictions are estimates, not guarantees that a customer will leave.
- Evaluation results may not generalize to new populations or time periods.
- The illustrative business-cost assumptions should be validated with actual
  retention costs and customer value.
- Protect API keys, customer data, logs, and deployed endpoints.
- Review model quality and drift before making retention decisions.
- Retrain and evaluate the model using an approved, versioned dataset and
  compare the new model against the current version before replacing it.

## License and Data

Add an appropriate license before redistributing this project. Follow the
dataset provider's terms when using or sharing the dataset.
