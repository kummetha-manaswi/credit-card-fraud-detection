"""
Fraud Predictor Engine
Handles model artifact loading, input alignment, scaling, inference,
optimized threshold application (0.75), and risk tier categorization.
"""

import os
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd


class FraudPredictor:
    """
    Production fraud risk predictor using the trained Random Forest classifier
    and StandardScaler with validation-optimized threshold = 0.75.
    """

    def __init__(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is None:
            # Default to model_artifacts relative to repository root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifacts_dir = os.path.join(base_dir, "model_artifacts")

        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.features: List[str] = []
        self.threshold: float = 0.75
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load all serialized model artifacts."""
        model_path = os.path.join(self.artifacts_dir, "random_forest_model.pkl")
        scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")
        features_path = os.path.join(self.artifacts_dir, "features.pkl")
        threshold_path = os.path.join(self.artifacts_dir, "threshold.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = list(joblib.load(features_path))

        if os.path.exists(threshold_path):
            self.threshold = float(joblib.load(threshold_path))
        else:
            self.threshold = 0.75

    def get_risk_tier(self, probability: float) -> str:
        """
        Categorize transaction risk tier according to business rules:
        - HIGH: probability >= 0.75 (Action: Block & trigger investigation)
        - MEDIUM: 0.30 <= probability < 0.75 (Action: Step-up authentication / manual review)
        - LOW: probability < 0.30 (Action: Approve)
        """
        if probability >= 0.75:
            return "HIGH"
        elif probability >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for a single transaction dictionary.
        Enforces exact feature ordering and scaling.
        """
        # Create DataFrame with exact column order
        df = pd.DataFrame([transaction])[self.features]

        # Scale features
        scaled_array = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_array, columns=self.features)

        # Get fraud probability (Class 1)
        fraud_prob = float(self.model.predict_proba(scaled_df)[0, 1])

        # Apply optimized threshold
        prediction = int(fraud_prob >= self.threshold)
        risk_level = self.get_risk_tier(fraud_prob)
        risk_score = fraud_prob * 100.0

        return {
            "fraud_probability": round(fraud_prob, 4),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "prediction": prediction,
            "threshold": round(self.threshold, 2),
            "action": (
                "BLOCK & INVESTIGATE"
                if risk_level == "HIGH"
                else "MANUAL REVIEW REQUIRED"
                if risk_level == "MEDIUM"
                else "APPROVE"
            ),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for a pandas DataFrame.
        """
        data = df[self.features].copy()
        scaled = self.scaler.transform(data)
        scaled_df = pd.DataFrame(scaled, columns=self.features)

        probabilities = self.model.predict_proba(scaled_df)[:, 1]
        results_df = df.copy()
        results_df["fraud_probability"] = np.round(probabilities, 4)
        results_df["risk_score"] = np.round(probabilities * 100.0, 2)
        results_df["prediction"] = (probabilities >= self.threshold).astype(int)
        results_df["risk_level"] = [self.get_risk_tier(p) for p in probabilities]

        return results_df


# Module-level cached predictor instance
_predictor_instance: Optional[FraudPredictor] = None


def get_predictor(artifacts_dir: Optional[str] = None) -> FraudPredictor:
    """Return singleton or initialized instance of FraudPredictor."""
    global _predictor_instance
    if _predictor_instance is None or artifacts_dir is not None:
        _predictor_instance = FraudPredictor(artifacts_dir=artifacts_dir)
    return _predictor_instance
