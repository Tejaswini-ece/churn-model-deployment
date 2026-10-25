
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/predict"
API_KEY = "dev-only-change-this-key"

customer = {
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

headers = {"X-API-Key": API_KEY}
latencies = []

with httpx.Client(timeout=10) as client:
    for _ in range(5):
        response = client.post(URL, json=customer, headers=headers)
        response.raise_for_status()

    for _ in range(100):
        start = time.perf_counter()
        response = client.post(URL, json=customer, headers=headers)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies.append(elapsed)

print(f"Requests measured: {len(latencies)}")
print(f"Median latency: {statistics.median(latencies):.2f} ms")
print(f"95th percentile: {sorted(latencies)[94]:.2f} ms")
