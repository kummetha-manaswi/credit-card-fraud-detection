# 💳 AI Fraud Guard — Real-Time Transaction Security Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen.svg)](https://github.com/slundberg/shap)
[![Tests](https://img.shields.io/badge/tests-25%20passed-success.svg)](https://pytest.org)

An enterprise-grade **AI Fraud Guard** platform designed as a modern FinTech product. This system provides autonomous transaction risk scoring, customer-facing virtual assistant support, game-theoretic explainability (SHAP), and high-throughput microservices.

---

## 📌 Executive Summary & Key Results

Credit card fraud detection is governed by extreme class imbalance: genuine transactions outnumber fraudulent events by more than **600 to 1** (fraud rate **0.1667%**). Standard models with default `0.50` decision thresholds generate unmanageable volumes of false alarms, causing customer churn and operational overload.

By coupling **SMOTE resampling (training set only)** with **validation-based decision threshold optimization (0.75)**, our Random Forest classifier achieves an exceptional precision-recall balance on the untouched test partition:

| Metric | Logistic Regression (Baseline) | Random Forest (Default Threshold 0.50) | Random Forest (Production @ Threshold 0.75) |
| :--- | :---: | :---: | :---: |
| **Precision** | 10.74% | 81.11% | **93.24%** |
| **Recall** | **87.37%** | 76.84% | **72.63%** |
| **F1-Score** | 19.12% | 78.92% | **81.66%** |
| **ROC-AUC** | 96.44% | 97.68% | **97.68%** |
| **PR-AUC** | 65.32% | 79.74% | **79.74%** |
| **False Positives (Alarms)** | 690 | 17 | **5 (99.3% reduction)** |

### 🎯 Test-Set Confusion Matrix (56,746 Transactions @ Threshold 0.75)

```text
                        Predicted Legitimate    Predicted Fraud
Actual Legitimate               56,646                  5   (False Positives)
Actual Fraud                        26                 69   (True Positives)
```
* **Only 5 False Alarms** out of 56,651 legitimate transactions (0.008% false positive rate).
* **69 Confirmed Fraud Cases Intercepted** in real time.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Data & Storage Layer
        CSV[Curated Sample CSV / Kaggle Dataset] --> SQLite[(SQLite Database: transactions)]
        Artifacts[model_artifacts/: RF Model, Scaler, Features, Threshold]
    end

    subgraph Core AI Engine
        Artifacts --> Predictor[src/predictor.py: FraudPredictor Engine]
        Artifacts --> Explainer[src/explainer.py: SHAP TreeExplainer]
        SQLite --> SQLAnalytics[src/sql_analytics.py: SQL Analytics Engine]
        Predictor --> Assistant[src/assistant.py: FraudGuard AI Assistant]
    end

    subgraph Serving & Microservice Layer
        Predictor --> API[api/app.py: FastAPI Service]
        API --> Docs[/docs: Swagger UI & OpenAPI Contract]
    end

    subgraph FinTech Web UI
        Predictor -. Local Fallback .-> Streamlit[streamlit_app.py: Streamlit FinTech App]
        API -- REST HTTP Requests --> Streamlit
        Explainer --> Streamlit
        SQLAnalytics --> Streamlit
        Assistant --> Streamlit
    end
```

---

## 🌟 Product Experience & Features

### 1. 💳 Modern Website Landing Experience (`Home`)
- **Hero Banner:** "Intelligent protection for every transaction."
- **Interactive Quick Actions:** Direct pathways to Check a Transaction, Explore Fraud, and Understand AI.
- **Key Financial KPIs:** Clean display of Analyzed Volume, Detected Fraud, Baseline Fraud Rate, and Precision.

### 2. 🔍 Streamlined Risk Checker (Simple & Advanced Modes)
- **Simple Mode (Default):** Evaluates observable business attributes (**Amount** and **Time of Day**) using pre-loaded empirical transaction scenarios (`🟢 Normal Purchase`, `🔴 Suspicious Purchase`, `🎲 Random Transaction`). Latent PCA features (V1–V28) remain cleanly managed under the hood.
- **Large Visual Results:** High-contrast risk banners (🔴 HIGH RISK, 🟠 MEDIUM RISK, 🟢 LOW RISK), numerical risk score ($/100$), calibrated probability, and operational action triggers (`BLOCK & INVESTIGATE`).
- **Compact Fraud Alert Summary:** Real-time investigation card outlining primary anomaly signals and recommended authorization steps.
- **"Why did AI flag this?" Attribution:** Instant SHAP push/pull factor tables showing which latent dimensions elevated or reduced risk.
- **Advanced Mode:** Full 30-feature inspection grid with zero fabrication of PCA features.

### 3. 🤖 FraudGuard AI Customer Assistant
- Persistent customer-support specialist helping users with declined payments, unrecognized transactions, and card safety practices.
- **Context-Aware Intelligence:** Automatically interprets the active page and the latest transaction evaluated in the Risk Checker.
- **100% Offline Fallback:** Operates with comprehensive built-in financial knowledge without requiring external LLM API keys.

### 4. 📊 Fraud Intelligence & SQL Workbench
- Amount segmentation charts revealing automated micro-probing attacks in the $\$0$–$\$1$ bracket.
- Nocturnal hourly curves displaying 2:00 AM off-peak fraud spikes ($1.45\%$).
- Embedded read-only SQL terminal for custom financial analytics on SQLite.

### 5. 🤖 AI Model Governance & Decision Threshold Simulator
- Executive scorecard with verified metrics ($93.24\%$ Precision, $72.63\%$ Recall, $97.68\%$ ROC-AUC).
- **Interactive Decision Threshold Simulator:** Slider ($0.10$ to $0.90$) allowing users to interactively simulate precision-recall trade-offs while keeping the production threshold locked at $0.75$.
- Developer API Playground showcasing `/predict` and `/health` request contracts.

### 6. 🧠 Explainable AI (SHAP)
- Global feature rankings highlighting top predictive dimensions (`V14`, `V17`, `V10`, `V12`, `V4`).
- Local transaction waterfalls and push/pull attribution tables.

---

## ⚡ Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kummetha-manaswi/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Tests

```bash
pytest -v
```
*Expected: 25 passed in ~4.6 seconds.*

### 3. Launch the FinTech Dashboard

```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`. Operates standalone out of the box with zero external dependencies.

### 4. Launch the FastAPI Microservice (Optional)

```bash
uvicorn api.app:app --reload --port 8000
```
* **Interactive OpenAPI Documentation:** Available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 📄 License & Attribution

This project is open-source under the MIT License. Built as an advanced agentic portfolio project demonstrating modern FinTech UX, robust machine learning engineering, and explainable AI.
