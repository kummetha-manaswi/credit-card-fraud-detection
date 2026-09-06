"""
Tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

SAMPLE_FRAUD = {
    "V1": -1.548788, "V2": 1.808698, "V3": -0.953509, "V4": 2.213085,
    "V5": -2.015728, "V6": -0.913457, "V7": -2.356013, "V8": 1.197169,
    "V9": -1.678374, "V10": -3.538650, "V11": 3.102090, "V12": -3.993373,
    "V13": -1.937411, "V14": -3.822894, "V15": 0.830970, "V16": -2.475359,
    "V17": -5.211875, "V18": -0.413872, "V19": 0.933262, "V20": 0.390786,
    "V21": 0.855138, "V22": 0.774745, "V23": 0.059037, "V24": 0.343200,
    "V25": -0.468938, "V26": -0.278338, "V27": 0.625922, "V28": 0.395573,
    "Time": 74159.0, "Amount": 76.94
}

SAMPLE_LEGIT = {
    "V1": 1.228821, "V2": -0.063408, "V3": 0.274145, "V4": 0.647465,
    "V5": -0.048135, "V6": 0.372073, "V7": -0.224231, "V8": 0.079939,
    "V9": 0.640759, "V10": -0.273054, "V11": -1.252728, "V12": 0.465079,
    "V13": 0.400502, "V14": -0.292842, "V15": -0.101774, "V16": -0.399836,
    "V17": 0.034336, "V18": -0.783550, "V19": 0.141345, "V20": -0.096566,
    "V21": -0.129554, "V22": -0.083779, "V23": -0.151661, "V24": -0.700372,
    "V25": 0.598550, "V26": 0.491409, "V27": 0.002989, "V28": 0.001782,
    "Time": 61290.0, "Amount": 11.50
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["version"] == "2.0.0"
    assert data["threshold"] == 0.75


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["number_of_features"] == 30
    assert data["threshold"] == 0.75


def test_predict_legitimate_endpoint():
    response = client.post("/predict", json=SAMPLE_LEGIT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["risk_level"] == "LOW"
    assert data["fraud_probability"] < 0.30
    assert data["threshold"] == 0.75
    assert data["action"] == "APPROVE"


def test_predict_fraud_endpoint():
    response = client.post("/predict", json=SAMPLE_FRAUD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["risk_level"] == "HIGH"
    assert data["fraud_probability"] >= 0.75
    assert data["threshold"] == 0.75
    assert data["action"] == "BLOCK & INVESTIGATE"


def test_predict_validation_error():
    # Missing Amount field
    bad_payload = {k: v for k, v in SAMPLE_LEGIT.items() if k != "Amount"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch_endpoint():
    response = client.post("/predict/batch", json={"transactions": [SAMPLE_LEGIT, SAMPLE_FRAUD]})
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert data["high_risk_count"] == 1
    assert data["low_risk_count"] == 1
    assert len(data["results"]) == 2
