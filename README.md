# Loan Disbursal Prediction System (End-to-End ML + API)

This project builds a complete machine learning pipeline to predict loan disbursal amounts using business features like login count, sanction ratio, and average sanction amount.

It includes data ingestion, model training, evaluation, and a production-ready Flask API for real-time predictions.

---

## Features

- Automated data ingestion from Cube.js API
- Time-series aggregation (weekly financial data)
- Regression-based prediction model (OLS)
- Model refitting with updated data
- Real-time prediction API
- Intelligent warning system for low-quality inputs
- Persistent model coefficients & dataset storage

---

## System Architecture

Data Source (Cube.js API) → Data Processing → Model Training → Flask API → Predictions

---

## Dataset

- Source: Internal Cube.js API
- Aggregation: Weekly
- Features:
  - `loginCount`
  - `sanctionRatio`
  - `avgSanctionAmount`
- Target:
  - `totalDisbursedAmount`

Data is fetched and saved as:
- `applications_data.json`
- `applications_data.csv`

---

## Model

### Regression Model
- Algorithm: **Ordinary Least Squares (OLS)** using statsmodels
- Predictors:
  - Login count
  - Sanction ratio
  - Avg sanction amount

### Enhancements

- Floor constraint using 10th percentile (prevents unrealistic predictions)
- Input validation warnings (detects low-quality inputs)
- Model metrics:
  - R² / Adjusted R²
  - MAE
  - MAPE
  - p-values

---

## Model Refit Pipeline

- Automatically retrains model using latest data
- Requires minimum data threshold
- Returns:
  - Updated coefficients
  - Performance metrics
  - Comparison with previous model

---

## API Endpoints

### Data Management
- `GET /api/data` → Fetch dataset
- `POST /api/data` → Add/update row
- `DELETE /api/data/<id>` → Delete row

---

### Model
- `POST /api/refit` → Retrain model
- `POST /api/refit/accept` → Save new model
- `GET /api/coefs` → Get model coefficients

---

### Prediction
- `POST /api/predict`

#### Example Input:
```json
{
  "loginCount": 100,
  "sanctionRatio": 0.35,
  "avgSanctionAmount": 1200000
}
``` id="pred-in"

#### Example Output:
```json
{
  "prediction": 54000000,
  "warnings": []
}
``` id="pred-out"
```
---

## Tech Stack

- Python
- Flask (API backend)
- Pandas / NumPy
- Statsmodels (OLS regression)
- Cube.js API
- JSON/CSV storage

---
