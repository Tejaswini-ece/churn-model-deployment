
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.drift import calculate_psi


API_KEY = "test-api-key"

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
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
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CHURN_API_KEY", API_KEY)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY}


# 1
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


# 2
def test_health_reports_loaded_model(client):
    data = client.get("/health").json()
    assert data["model_loaded"] is True


# 3
def test_health_reports_model_version(client):
    data = client.get("/health").json()
    assert data["model_version"] == "1.0.0"


# 4
def test_prediction_without_key_is_rejected(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 401


# 5
def test_prediction_with_wrong_key_is_rejected(client):
    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


# 6
def test_prediction_with_valid_key_succeeds(client, auth_headers):
    response = client.post(
        "/predict", json=SAMPLE_CUSTOMER, headers=auth_headers
    )
    assert response.status_code == 200


# 7
def test_prediction_probability_is_valid(client, auth_headers):
    response = client.post(
        "/predict", json=SAMPLE_CUSTOMER, headers=auth_headers
    )
    probability = response.json()["churn_probability"]
    assert 0 <= probability <= 1


# 8
def test_prediction_contains_expected_fields(client, auth_headers):
    response = client.post(
        "/predict", json=SAMPLE_CUSTOMER, headers=auth_headers
    )
    assert {
        "churn_probability", "predicted_churn",
        "threshold", "model_version"
    }.issubset(response.json())


# 9
def test_prediction_uses_saved_threshold(client, auth_headers):
    response = client.post(
        "/predict", json=SAMPLE_CUSTOMER, headers=auth_headers
    )
    assert response.json()["threshold"] == pytest.approx(0.30)


# 10
def test_invalid_senior_citizen_is_rejected(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "SeniorCitizen": 2}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 422


# 11
def test_negative_tenure_is_rejected(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "tenure": -1}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 422


# 12
def test_negative_monthly_charges_are_rejected(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "MonthlyCharges": -1}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 422


# 13
def test_empty_category_is_rejected(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "gender": " "}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 422


# 14
def test_unexpected_field_is_rejected(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "unexpected": "value"}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 422


# 15
def test_unseen_category_does_not_crash(client, auth_headers):
    customer = {**SAMPLE_CUSTOMER, "gender": "UnseenCategory"}
    response = client.post(
        "/predict", json=customer, headers=auth_headers
    )
    assert response.status_code == 200


# 16
def test_batch_prediction_succeeds(client, auth_headers):
    response = client.post(
        "/predict/batch",
        json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


# 17
def test_empty_batch_is_rejected(client, auth_headers):
    response = client.post(
        "/predict/batch",
        json={"customers": []},
        headers=auth_headers,
    )
    assert response.status_code == 422


# 18
def test_batch_over_limit_is_rejected(client, auth_headers):
    response = client.post(
        "/predict/batch",
        json={"customers": [SAMPLE_CUSTOMER] * 501},
        headers=auth_headers,
    )
    assert response.status_code == 422


# 19
def test_drift_endpoint_succeeds(client, auth_headers):
    response = client.post(
        "/drift",
        json={"customers": [SAMPLE_CUSTOMER]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "features" in response.json()


# 20
def test_psi_is_near_zero_for_equal_distributions():
    proportions = np.array([0.2, 0.3, 0.5])
    psi = calculate_psi(proportions, proportions)
    assert psi == pytest.approx(0.0)


# 21
def test_psi_is_positive_for_different_distributions():
    expected = np.array([0.9, 0.1])
    actual = np.array([0.2, 0.8])
    assert calculate_psi(expected, actual) > 0
