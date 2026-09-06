"""
Pydantic schemas for the Fraud Intelligence API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """
    Input schema representing a single credit card transaction with 30 features.
    """
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Time: float = Field(..., description="Elapsed seconds since first transaction")
    Amount: float = Field(..., ge=0.0, description="Transaction monetary amount")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "V1": -1.548788,
                "V2": 1.808698,
                "V3": -0.953509,
                "V4": 2.213085,
                "V5": -2.015728,
                "V6": -0.913457,
                "V7": -2.356013,
                "V8": 1.197169,
                "V9": -1.678374,
                "V10": -3.538650,
                "V11": 3.102090,
                "V12": -3.993373,
                "V13": -1.937411,
                "V14": -3.822894,
                "V15": 0.830970,
                "V16": -2.475359,
                "V17": -5.211875,
                "V18": -0.413872,
                "V19": 0.933262,
                "V20": 0.390786,
                "V21": 0.855138,
                "V22": 0.774745,
                "V23": 0.059037,
                "V24": 0.343200,
                "V25": -0.468938,
                "V26": -0.278338,
                "V27": 0.625922,
                "V28": 0.395573,
                "Time": 74159.0,
                "Amount": 76.94
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Response schema for fraud risk prediction.
    """
    fraud_probability: float
    risk_score: float
    risk_level: str
    prediction: int
    threshold: float
    action: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    number_of_features: int
    threshold: float
    version: str
