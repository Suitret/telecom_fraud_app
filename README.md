# Telecom Fraud Detection & Streamlit Deployment

This project implements a **Telecom Fraud Detection** app. It features a machine learning pipeline that identifies fraudulent Call Detail Records (CDRs) and provides a user-friendly **Streamlit** interface for real-time inference.

---

## Project Overview

Telecommunication fraud (such as SIM-boxing, call masking, and international revenue share fraud) costs the industry billions annually. This project uses **Random Forest** and **Adaptive Multi-Feature Encoding** to classify transactions as "Genuine" or "Fraudulent" based on behavioral patterns.

### Key Features
* **Automated Feature Engineering:** Implements adaptive target and frequency encoding for categorical variables like `location_origin` and `call_type`.
* **Robust Classification:** Utilizes an ensemble learning approach to handle the non-linear nature of fraud patterns.
* **Interactive Dashboard:** A Streamlit-based web application allowing users to input call parameters and receive immediate risk scores.

---

## 📂 Project Structure
```
telecom_fraud_app/
├── app.py # Streamlit Web Application (UI/UX)
├── model_utils.py # Custom Encoding & Transformation Logic
├── model_new.pkl # Model
├── label_encoder.pkl # Label Encoder
├── train.py # Model Training & Serialization Script
├── requirements.txt # Project Dependencies
└── data/
        └── realtime_cdr_log.csv # Dataset (Kaggle Source)
```
---

## Installation & Setup

### 1. Clone the Environment
Ensure you have Python 3.8+ installed. Create a virtual environment and install dependencies:
```
bash pip install -r requirements.txt
```
### 2. Prepare the Data
Place the `realtime_cdr_log.csv` file inside the `data/` directory. The model expects columns such as `duration_sec`, `call_type`, `is_night_call`, and `transaction_status`.

### 3. Train the Model
Run the training script to generate the serialized model and encoder files:
```
bash python train.py
```
### 4. Launch the Application
Start the Streamlit server to use the web interface:
```
bash streamlit run app.py
```
---

## Methodology

### Data Encoding
We employ a custom `AdaptiveEncoder` class located in `model_utils.py`. This handles:
* **Target Mean Encoding:** Captures the historical fraud rate of specific locations/call types.
* **Frequency Encoding:** Accounts for the volume of calls, identifying anomalies in call frequency.

### Model Choice
A **Random Forest Classifier** is used due to its resilience against overfitting in imbalanced datasets (common in fraud detection) and its ability to provide feature importance rankings.

---

## Future Extensions

* **SMOTE Integration:** Implement Synthetic Minority Over-sampling Technique to better handle highly imbalanced fraud classes.
* **Real-time Batch Upload:** Add functionality to `app.py` for uploading `.csv` files and downloading a report of flagged transactions.
* **SHAP Explainability:** Integrate SHAP values to explain *why* a specific call was flagged as fraudulent (e.g., "High duration + Night call").
