# 💳 Loan Approval Prediction with Interpretability

This project builds a machine learning pipeline to predict loan approval outcomes and analyze feature importance using interpretable techniques.

It combines ensemble learning (Random Forest), permutation-based feature importance, and decision tree classification for explainability.

---

##  Objective

- Predict loan approval status
- Identify key factors influencing decisions
- Build interpretable models for transparency

---

## Dataset

- Source: Kaggle Loan Approval Classification Dataset
- Contains applicant demographics, financial attributes, and loan details

Example features:
- Person income
- Loan interest rate
- Loan intent
- Home ownership
- Previous loan defaults

---

##  Methodology

### 1. Data Preprocessing
- Loaded dataset using Pandas
- Encoded categorical variables using Label Encoding
- Split data into training and validation sets

---

### 2. Model 1 – Random Forest Classifier
- Trained a Random Forest model (100 estimators)
- Used ensemble learning for better generalization

---

### 3. Feature Importance (Permutation Importance)
- Applied permutation importance using ELI5
- Evaluated feature impact by measuring performance drop when shuffled

 Top features:
- Previous loan defaults
- Loan interest rate
- Loan percent income
- Home ownership
- Income

---

### 4. Model 2 – Decision Tree Classifier (Interpretable Model)
- Trained on top important features
- Provides clear, human-readable decision rules

---

##  Evaluation Metrics

### Decision Tree Performance

- **Accuracy Score**
- **Classification Report (Precision, Recall, F1-score)**
- **Confusion Matrix**
- **ROC-AUC Score**

These metrics provide a comprehensive view of model performance and class-wise behavior.

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- ELI5 (Permutation Importance)

---
