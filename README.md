# 💳 AI Fraud Guard

### AI-Powered Credit Card Fraud Detection Platform

A machine learning based fraud detection application that analyzes credit card transactions, estimates fraud risk, explains model decisions, and provides fraud analytics through an interactive FinTech-style web application.

> Built with Machine Learning, Explainable AI, SQL, FastAPI and Streamlit.

---

## 🚀 Live Demo

**Live Application:** `YOUR_STREAMLIT_URL`

**GitHub Repository:** `YOUR_GITHUB_URL`

---

## 🎯 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared with legitimate transactions.

This project builds an end-to-end fraud detection platform that goes beyond a simple **"Fraud / Not Fraud"** prediction.

The platform provides:

- 🔍 Transaction risk checking
- 📊 Fraud analytics
- 🤖 AI-powered customer assistant
- 🧠 Explainable AI using SHAP
- 🗄️ SQL-based transaction analysis
- ⚡ FastAPI prediction service
- 🧪 Automated testing
- 🎨 Interactive FinTech-style interface

---

## 📊 Model Performance

The dataset contains a highly imbalanced fraud class, with fraudulent transactions representing approximately **0.1667%** of the data.

The project uses:

- SMOTE on training data
- Random Forest classification
- Validation-based threshold optimization
- Production decision threshold of **0.75**

### Test Set Results

| Metric | Result |
|---|---:|
| Precision | **93.24%** |
| Recall | **72.63%** |
| F1-Score | **81.66%** |
| ROC-AUC | **97.68%** |
| PR-AUC | **79.74%** |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| **Actual Legitimate** | 56,646 | 5 |
| **Actual Fraud** | 26 | 69 |

At the production threshold of **0.75**, the model detected **69 fraudulent transactions** with only **5 false positives** on the test set.

---

## 🖥️ Features

### 🔍 Transaction Risk Checker

An interactive interface for evaluating transaction risk.

**Simple Mode**

Designed for normal users.

Users interact with:

- Transaction amount
- Transaction time/scenario
- Pre-loaded transaction examples

The underlying `V1–V28` features are handled by the application automatically.

**Advanced Mode**

Technical users can inspect the complete model input:

`Time`, `V1–V28`, and `Amount`.

---

### 📊 Fraud Intelligence

Interactive fraud analytics including:

- Fraud vs legitimate transactions
- Transaction amount analysis
- Time-based fraud patterns
- Fraud statistics
- SQL-powered analysis

---

### 🤖 FraudGuard AI

A customer-facing AI assistant designed around transaction and fraud support.

It can answer questions related to:

- Suspicious transactions
- Fraud risk
- Card security
- Declined payments
- Transaction results
- Application functionality

The application also includes an offline fallback so the assistant can function without requiring an external LLM API key.

---

### 🧠 Explainable AI

The application uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

It provides:

- Global feature importance
- Local transaction explanations
- Feature contribution analysis
- Individual prediction explanations

The dataset's `V1–V28` variables are anonymized PCA-transformed features and are not assigned artificial business meanings.

---

### 🤖 AI Model

The model analysis section provides:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Decision threshold analysis

A threshold simulator allows users to explore the precision-recall trade-off.

Production threshold:

`0.75`

---

### ⚡ FastAPI Prediction API

The project includes a FastAPI service for model inference.

Main endpoints:

- `GET /health`
- `POST /predict`

The API performs:

1. Input validation
2. Feature ordering
3. Scaling
4. Model inference
5. Fraud probability calculation
6. Threshold-based classification

Interactive Swagger/OpenAPI documentation is available through the `/docs` route when the API is running.

---

## 🏗️ Architecture

Transaction Data  
↓  
Data Cleaning  
↓  
Train / Validation / Test Split  
↓  
SMOTE on Training Data  
↓  
Feature Scaling  
↓  
Random Forest Model  
↓  
Fraud Probability  
↓  
Decision Threshold `0.75`  
↓  
Fraud / Legitimate Decision  
↓  
Streamlit / FastAPI / Explainable AI

---

## 🧠 Machine Learning Pipeline

The core machine learning workflow consists of:

1. Data cleaning
2. Duplicate handling
3. Train/validation/test splitting
4. Handling class imbalance using SMOTE
5. Feature scaling
6. Random Forest training
7. Probability prediction
8. Threshold optimization
9. Fraud classification
10. Explainability using SHAP

---

## 🔬 Understanding V1–V28

The dataset contains the following major variables:

- `Time`
- `V1–V28`
- `Amount`
- `Class`

`V1–V28` are anonymized PCA-transformed numerical features provided by the dataset.

They are **not** direct business attributes such as:

- Customer age
- Customer location
- Merchant
- Card type
- Customer income

Because their original meanings are anonymized, the application does not invent interpretations for them.

In Simple Mode, users do not need to manually enter these features.

---

## 🛠️ Technology Stack

**Programming**

- Python

**Data Science**

- Pandas
- NumPy

**Machine Learning**

- Scikit-learn
- Random Forest
- SMOTE

**Explainable AI**

- SHAP

**Backend**

- FastAPI
- Pydantic
- Uvicorn

**Database**

- SQLite
- SQL

**Frontend**

- Streamlit

**Testing**

- Pytest

**Development**

- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

    credit-card-fraud-detection/
    │
    ├── .streamlit/
    │   └── config.toml
    │
    ├── api/
    │   ├── __init__.py
    │   ├── app.py
    │   └── schemas.py
    │
    ├── assets/
    │   ├── card_hero.jpg
    │   ├── card_shield.jpg
    │   ├── neural_chip.jpg
    │   ├── payment_pos.jpg
    │   └── security_center.jpg
    │
    ├── data/
    │   └── sample_transactions.csv
    │
    ├── model_artifacts/
    │   ├── features.pkl
    │   ├── random_forest_model.pkl
    │   ├── scaler.pkl
    │   └── threshold.pkl
    │
    ├── src/
    │   ├── __init__.py
    │   ├── assistant.py
    │   ├── explainer.py
    │   ├── predictor.py
    │   └── sql_analytics.py
    │
    ├── tests/
    │   ├── __init__.py
    │   ├── test_api.py
    │   ├── test_assistant.py
    │   ├── test_predictor.py
    │   └── test_sql.py
    │
    ├── Credit_Card_Fraud_Detection.ipynb
    ├── app.py
    ├── streamlit_app.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

---

## 🚀 Run Locally

### Clone the repository

    git clone YOUR_GITHUB_URL
    cd credit-card-fraud-detection

### Install dependencies

    pip install -r requirements.txt

### Run tests

    pytest -v

### Start Streamlit

    streamlit run streamlit_app.py

### Start FastAPI

    uvicorn api.app:app --reload --port 8000

The FastAPI Swagger documentation is available through the `/docs` route.

---

## 🧪 Testing

Automated tests cover:

- Prediction pipeline
- FastAPI endpoints
- SQL analytics
- FraudGuard AI assistant

Run:

    pytest -v

---

## 📌 Dataset

This project uses the public credit card fraud detection dataset containing transaction records with anonymized PCA features.

The original raw dataset is not included in the GitHub repository because of its size.

A curated sample dataset is included for application demonstration and testing.

---

## 🌐 Deployment

The Streamlit application can be deployed using Streamlit Community Cloud directly from the GitHub repository.

**Application entry point:**

`streamlit_app.py`

After deployment, replace the placeholder below with the generated application URL:

`YOUR_STREAMLIT_URL`

---

## ⚠️ Limitations

This project is an educational and portfolio implementation and should not be treated as a production banking fraud prevention system.

The dataset's anonymized features and historical transaction distribution do not represent every real-world banking environment.

A production fraud detection system would additionally require:

- Real-time transaction streams
- Customer behavior history
- Merchant information
- Geographic signals
- Device information
- Authentication signals
- Continuous model monitoring
- Data privacy controls
- Security infrastructure
- Model drift monitoring

---

## 🎓 What This Project Demonstrates

This project combines:

**Data Science**  
**Machine Learning**  
**Imbalanced Classification**  
**Feature Engineering**  
**Explainable AI**  
**SQL Analytics**  
**REST APIs**  
**Interactive Web Applications**  
**Automated Testing**

The project demonstrates the complete journey from a machine learning model to an interactive application rather than stopping at model training alone.

---

## 👩‍💻 Author

**Kummetha Manaswi**

Data Science | Machine Learning | AI

---

## 📄 License

MIT License
