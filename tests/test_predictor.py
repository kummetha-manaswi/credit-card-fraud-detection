"""
Unit tests for FraudPredictor engine.
Verifies artifact loading, 30-feature ordering invariance,
threshold 0.75 enforcement, and risk tier categorization.
"""

import pytest
import pandas as pd
from src.predictor import FraudPredictor, get_predictor

# Test fixtures for representative transactions
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


def test_predictor_artifacts_loaded():
    predictor = get_predictor()
    assert predictor.model is not None, "Model should be loaded"
    assert predictor.scaler is not None, "Scaler should be loaded"
    assert len(predictor.features) == 30, "Should have exactly 30 features"
    assert predictor.threshold == 0.75, "Threshold must be exactly 0.75"
    assert "V1" in predictor.features
    assert "Amount" in predictor.features
    assert "Time" in predictor.features


def test_predict_single_legitimate():
    predictor = get_predictor()
    result = predictor.predict_single(SAMPLE_LEGIT)
    assert result["prediction"] == 0
    assert result["risk_level"] == "LOW"
    assert result["fraud_probability"] < 0.30
    assert result["risk_score"] < 30.0
    assert result["threshold"] == 0.75
    assert result["action"] == "APPROVE"


def test_predict_single_fraud():
    predictor = get_predictor()
    result = predictor.predict_single(SAMPLE_FRAUD)
    assert result["prediction"] == 1
    assert result["risk_level"] == "HIGH"
    assert result["fraud_probability"] >= 0.75
    assert result["risk_score"] >= 75.0
    assert result["threshold"] == 0.75
    assert result["action"] == "BLOCK & INVESTIGATE"


def test_feature_order_invariance():
    """Verify dictionary key ordering does not alter the prediction."""
    predictor = get_predictor()
    res1 = predictor.predict_single(SAMPLE_FRAUD)

    # Reverse key order
    reversed_keys = dict(reversed(list(SAMPLE_FRAUD.items())))
    res2 = predictor.predict_single(reversed_keys)

    assert res1["fraud_probability"] == res2["fraud_probability"]
    assert res1["prediction"] == res2["prediction"]
    assert res1["risk_level"] == res2["risk_level"]


def test_risk_tier_classification():
    predictor = get_predictor()
    assert predictor.get_risk_tier(0.95) == "HIGH"
    assert predictor.get_risk_tier(0.75) == "HIGH"
    assert predictor.get_risk_tier(0.7499) == "MEDIUM"
    assert predictor.get_risk_tier(0.30) == "MEDIUM"
    assert predictor.get_risk_tier(0.2999) == "LOW"
    assert predictor.get_risk_tier(0.01) == "LOW"


def test_predict_batch():
    predictor = get_predictor()
    batch_df = pd.DataFrame([SAMPLE_LEGIT, SAMPLE_FRAUD])
    scored = predictor.predict_batch(batch_df)

    assert len(scored) == 2
    assert "fraud_probability" in scored.columns
    assert "risk_level" in scored.columns
    assert "prediction" in scored.columns
    assert scored.iloc[0]["prediction"] == 0
    assert scored.iloc[1]["prediction"] == 1


def test_explainer():
    from src.explainer import get_explainer
    explainer = get_explainer()
    res = explainer.explain_transaction(SAMPLE_FRAUD, top_k=5)
    assert "top_factors" in res
    assert "pushing_fraud" in res
    assert "pushing_legit" in res
    assert len(res["top_factors"]) == 5
