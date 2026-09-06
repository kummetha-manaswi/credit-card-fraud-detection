"""
SHAP Explainability Module
Provides global feature importance and transaction-level explanation using TreeExplainer.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap
from src.predictor import FraudPredictor, get_predictor


class FraudExplainer:
    """
    SHAP TreeExplainer wrapper for interpreting the Random Forest fraud model.
    """

    def __init__(self, predictor: Optional[FraudPredictor] = None):
        self.predictor = predictor or get_predictor()
        self.explainer = shap.TreeExplainer(self.predictor.model)
        self.features = self.predictor.features

    def explain_transaction(
        self, transaction: Dict[str, Any], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute local SHAP attribution for an individual transaction.
        Returns top features pushing towards or away from fraud.
        """
        # Align features and scale
        df = pd.DataFrame([transaction])[self.features]
        scaled = self.predictor.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=self.features)

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(scaled_df)

        # Binary classification handling
        if isinstance(raw_shap, list):
            shap_values = raw_shap[1][0]  # Class 1 (Fraud)
        elif len(raw_shap.shape) == 3:
            shap_values = raw_shap[0, :, 1]
        else:
            shap_values = raw_shap[0]

        impact_df = pd.DataFrame({
            "Feature": self.features,
            "Raw_Value": [transaction.get(f, 0.0) for f in self.features],
            "SHAP_Impact": shap_values,
            "Abs_Impact": np.abs(shap_values)
        }).sort_values("Abs_Impact", ascending=False)

        top_factors = impact_df.head(top_k).to_dict(orient="records")
        positive_factors = impact_df[impact_df["SHAP_Impact"] > 0].head(top_k).to_dict(orient="records")
        negative_factors = impact_df[impact_df["SHAP_Impact"] < 0].head(top_k).to_dict(orient="records")

        # Base value (expected value for fraud class)
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            expected_val = float(base_value[1])
        else:
            expected_val = float(base_value)

        return {
            "expected_value": expected_val,
            "top_factors": top_factors,
            "pushing_fraud": positive_factors,
            "pushing_legit": negative_factors,
            "all_shap": dict(zip(self.features, shap_values.tolist()))
        }

    def get_global_importance(
        self, sample_df: pd.DataFrame, max_samples: int = 500
    ) -> pd.DataFrame:
        """
        Compute global mean absolute SHAP feature importance over a sample of transactions.
        """
        sample = sample_df[self.features].head(max_samples)
        scaled = self.predictor.scaler.transform(sample)
        scaled_df = pd.DataFrame(scaled, columns=self.features)

        raw_shap = self.explainer.shap_values(scaled_df)

        if isinstance(raw_shap, list):
            shap_fraud = raw_shap[1]
        elif len(raw_shap.shape) == 3:
            shap_fraud = raw_shap[:, :, 1]
        else:
            shap_fraud = raw_shap

        mean_abs = np.abs(shap_fraud).mean(axis=0)
        global_df = pd.DataFrame({
            "Feature": self.features,
            "Mean_Abs_SHAP": mean_abs
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        return global_df


_explainer_instance: Optional[FraudExplainer] = None


def get_explainer(predictor: Optional[FraudPredictor] = None) -> FraudExplainer:
    """Return singleton or initialized instance of FraudExplainer."""
    global _explainer_instance
    if _explainer_instance is None or predictor is not None:
        _explainer_instance = FraudExplainer(predictor=predictor)
    return _explainer_instance
