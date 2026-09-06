"""
FastAPI Microservice for AI Fraud Intelligence.
Provides endpoints for transaction health check, single & batch fraud prediction,
and SHAP local attribution.
"""

import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from src.predictor import get_predictor
from src.explainer import get_explainer

# Initialize FastAPI application
app = FastAPI(
    title="AI-Powered Credit Card Fraud Detection API",
    description="Production-grade Machine Learning API for real-time fraud risk prediction and explainability",
    version="2.0.0",
)

# CORS middleware for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = get_predictor()


@app.get("/", tags=["General"])
def home():
    """Service status and documentation gateway."""
    return {
        "message": "AI-Powered Credit Card Fraud Detection API",
        "status": "running",
        "version": "2.0.0",
        "docs_url": "/docs",
        "threshold": predictor.threshold,
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    """System health check and loaded model information."""
    return HealthResponse(
        status="healthy",
        model_loaded=(predictor.model is not None),
        number_of_features=len(predictor.features),
        threshold=predictor.threshold,
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(transaction: TransactionInput):
    """
    Score a single transaction for fraud risk.
    Applies exact feature ordering, StandardScaler transform,
    and the optimized 0.75 decision threshold.
    """
    try:
        data = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
        result = predictor.predict_single(data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictionRequest):
    """
    Score a batch of transactions in a single vectorized request.
    """
    try:
        import pandas as pd
        records = [
            t.model_dump() if hasattr(t, "model_dump") else t.dict()
            for t in payload.transactions
        ]
        df = pd.DataFrame(records)
        scored_df = predictor.predict_batch(df)

        results = []
        high_c = 0
        med_c = 0
        low_c = 0

        for _, row in scored_df.iterrows():
            tier = row["risk_level"]
            if tier == "HIGH":
                high_c += 1
            elif tier == "MEDIUM":
                med_c += 1
            else:
                low_c += 1

            results.append(
                PredictionResponse(
                    fraud_probability=float(row["fraud_probability"]),
                    risk_score=float(row["risk_score"]),
                    risk_level=tier,
                    prediction=int(row["prediction"]),
                    threshold=float(predictor.threshold),
                    action=(
                        "BLOCK & INVESTIGATE"
                        if tier == "HIGH"
                        else "MANUAL REVIEW REQUIRED"
                        if tier == "MEDIUM"
                        else "APPROVE"
                    ),
                )
            )

        return BatchPredictionResponse(
            total_processed=len(results),
            high_risk_count=high_c,
            medium_risk_count=med_c,
            low_risk_count=low_c,
            results=results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/explain", tags=["Explainability"])
def explain(transaction: TransactionInput):
    """
    Compute SHAP attribution factors for an individual transaction.
    """
    try:
        explainer = get_explainer(predictor=predictor)
        data = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
        explanation = explainer.explain_transaction(data, top_k=10)
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
