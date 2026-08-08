# Real-Time Fraud Detection System

An end-to-end fraud detection pipeline — from SQL-based feature engineering through model comparison, hyperparameter tuning, explainability, and deployment as a live API.

## Problem Statement

Credit card fraud costs financial institutions billions annually, and the challenge isn't just detecting fraud — it's doing so in a dataset where fraud represents just 0.17% of all transactions. This asymmetry means a model can achieve 99.83% accuracy while catching zero fraud cases, making naive accuracy-based approaches actively misleading. This project builds an end-to-end fraud detection pipeline — from SQL-based feature engineering through model comparison, tuning, explainability, and deployment — with particular attention to the precision-recall tradeoffs inherent to severely imbalanced classification.

## Dataset

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 anonymized European credit card transactions from September 2013, with 492 (0.17%) labeled as fraudulent. Features `V1`–`V28` are PCA-transformed for confidentiality; `Time`, `Amount`, and `Class` (target) remain in original form.

## Architecture

```
Raw CSV (Kaggle dataset)
    ↓
SQLite Database (data/fraud_detection.db)
    ↓
SQL Feature Engineering (window functions: rolling avg, transaction count, time since last)
    ↓
transaction_features table
    ↓
Train/Test Split (stratified, 80/20)
    ↓
Preprocessing Pipeline (StandardScaler)
    ↓
Model Training + Comparison (Logistic Regression, Random Forest, XGBoost, LightGBM)
    ↓
Imbalance Handling (class_weight, SMOTE) + Hyperparameter Tuning (RandomizedSearchCV)
    ↓
Best Model: Tuned Random Forest (63% recall)
    ↓
Explainability (SHAP)
    ↓
Model Persistence (joblib)
    ↓
FastAPI /predict endpoint → real-time fraud scoring
```

## Feature Engineering

Three features were engineered directly in SQL using window functions:

- **`rolling_avg_amount`** — rolling average transaction amount (`AVG() OVER (ORDER BY Time ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)`)
- **`transaction_count_last_10`** — rolling count of nearby transactions
- **`time_since_last_transaction`** — gap since the previous transaction, via `LAG()`

## Results

| Model | Precision (Fraud) | Recall (Fraud) | Accuracy | Notes |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.00 | 0.00 | 99.83% | Accuracy trap — catches zero fraud |
| Logistic Regression + class_weight | 0.00 | 0.57 | 65% | Overcorrected, too many false alarms |
| Logistic Regression + SMOTE | 0.00 | 0.57 | 66% | Same plateau as class weighting |
| Random Forest (default) | 0.02 | 0.13 | 99% | Default settings underperform |
| XGBoost (default) | 0.01 | 0.32 | 95% | |
| LightGBM (default) | 0.01 | 0.34 | 95% | Auto-dropped 1 low-signal feature |
| Random Forest (tuned, flawed CV) | 0.02 | 0.11 | 99% | Tuned against SMOTE folds — methodological error |
| **Random Forest (tuned, corrected)** | **0.01** | **0.63** | **79%** | **Best model — tuned against real imbalanced CV folds** |

## Key Findings

**1. Accuracy is dangerously misleading on imbalanced data.** A baseline model achieved 99.83% accuracy while catching 0% of actual fraud cases — simply by always predicting "legitimate." This motivated the shift to precision/recall/F1 as the real evaluation metrics throughout the rest of the project.

**2. Not every engineered feature carries signal.** Of three SQL window-function features built, `transaction_count_last_10` showed almost no difference between fraud and legitimate transactions (10.9998 vs 11.0000) — a direct consequence of the dataset lacking user-level identifiers, meaning this feature measured dataset-wide transaction density rather than individual behavior. This was independently confirmed later when LightGBM automatically dropped the same feature during training. `rolling_avg_amount` and `time_since_last_transaction`, by contrast, both showed meaningful separation between classes.

**3. Imbalance-handling technique matters less than model choice — up to a point.** Both `class_weight="balanced"` and SMOTE, applied to logistic regression, converged to nearly identical results (57% recall, near-zero precision), suggesting the bottleneck was the model's linear decision boundary rather than the imbalance-handling method itself.

**4. Default tree-model settings underperformed a tuned linear model.** Random Forest, XGBoost, and LightGBM all underperformed logistic regression + SMOTE using default hyperparameters, a reminder that "more powerful" algorithms aren't automatically better without proper tuning.

**5. Cross-validation methodology matters as much as the model itself.** An initial hyperparameter search tuned against SMOTE-balanced cross-validation folds produced a model that performed *worse* on real test data than an untuned default — because it was optimized for a synthetic distribution that didn't match production reality. Correcting this by validating against real, stratified, imbalanced folds (with `class_weight="balanced"` instead of SMOTE) produced the project's best model: 63% recall.

**6. There is no single "best" model — only a business-appropriate tradeoff point.** The final model catches 63% of fraud but at the cost of many false positives (1% precision). Whether this tradeoff is acceptable depends on the real-world cost of a false positive (customer friction) versus a false negative (direct financial loss) — a business decision, not a purely technical one.

## Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the tuned Random Forest's predictions, both globally (which features matter most overall) and locally (why a specific transaction was flagged). See `notebooks/shap_summary_plot.png`.

## API

The final model is served via a FastAPI application with a `/predict` endpoint that accepts transaction features as JSON and returns a fraud prediction with probability.

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 5000.0,
    "Amount": 250.00,
    "transaction_count_last_10": 11,
    "rolling_avg_amount": 88.5,
    "time_since_last_transaction": 1.2
  }'
```

Example response:
```json
{"is_fraud": true, "fraud_probability": 0.5505575678194787}
```

Interactive API documentation is available at `/docs` once the server is running.

## Setup & Reproduction

1. Clone this repository
2. Create the conda environment: `conda create --name fraud-detection python=3.11`
3. Activate it: `conda activate fraud-detection`
4. Install dependencies: `pip install -r requirements.txt`
5. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/`
6. Load data into SQLite: `python src/load_data.py`
7. Build engineered features: `python src/build_features_table.py`
8. Train and save the final model: `python src/train_final_model.py`
9. Start the API: `uvicorn api.main:app --reload`
10. Visit `http://127.0.0.1:8000/docs` to test predictions interactively

## Limitations & Honest Caveats

- **No true user-level features.** The dataset is anonymized with no `user_id`, so velocity/behavioral features (like `transaction_count_last_10`) reflect dataset-wide density rather than individual account behavior — a real production system would build these per-user.
- **Low precision at the final operating point.** The tuned model catches 63% of fraud but with only ~1% precision, meaning the vast majority of flagged transactions are false alarms. A production deployment would need threshold tuning and/or a human-review workflow to make this operationally viable.
- **Static, historical data.** This dataset represents ~2 days of transactions from September 2013. A real fraud system faces concept drift (fraud patterns evolve over time) that this project doesn't address.
- **Local deployment only.** The FastAPI service runs locally, not deployed to any cloud infrastructure — a genuine next step, not a finished production system.

## What I'd Do With More Time

- Explore precision-recall threshold tuning to find a more business-viable operating point
- Test additional resampling techniques (ADASYN, undersampling combined with SMOTE)
- Containerize the API with Docker for reproducible deployment
- Deploy to a cloud platform (Render, Railway, or AWS) for a live, publicly-accessible demo
- Build a simple Streamlit dashboard on top of the API for non-technical stakeholders

## Tech Stack

Python, Pandas, SQLite, Scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), SHAP, FastAPI, Uvicorn, Git
