# Major-Project-Labmnetrix
# Voyage Analytics – Prediction Modeler (Member 1)

## Overview

This repository contains the machine learning component of the **Voyage Analytics: Integrating MLOps in Travel Productionization of ML Systems** project.

**Role:** Prediction Modeler (Member 1)

The objective of this module is to develop, evaluate, and export production-ready machine learning models that can later be deployed by the Backend and DevOps teams.

---

# Tasks Completed

### Flight Price Prediction (Regression)

* Data preprocessing
* Missing value handling
* Feature encoding
* Feature scaling
* Linear Regression (Baseline)
* Random Forest Regressor
* Model evaluation
* MLflow experiment tracking
* Model artifact generation (.pkl)

### Gender Classification

* Data preprocessing
* Missing value handling
* Feature engineering
* Logistic Regression
* Random Forest Classifier
* Model evaluation
* MLflow experiment tracking
* Model artifact generation (.pkl)

---

# Project Structure

```
voyage/
│
├── data/
│   ├── flights.csv
│   └── users.csv
│
├── models/
│   ├── flight_price_model.pkl
│   └── gender_classifier.pkl
│
├── config.py
├── data_prep.py
├── train_regression.py
├── train_classification.py
├── evaluate.py
├── utils.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd voyage
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Placement

Create the following folder:

```
data/
```

Place the datasets inside it.

```
data/
    flights.csv
    users.csv
```

---

# Train Flight Price Model

```bash
python train_regression.py
```

This will:

* Train Linear Regression
* Train Random Forest Regressor
* Compare both models
* Log experiments to MLflow
* Save the best model

Output:

```
models/flight_price_model.pkl
```

---

# Train Gender Classification Model

```bash
python train_classification.py
```

This will:

* Train Logistic Regression
* Train Random Forest Classifier
* Compare both models
* Log experiments to MLflow
* Save the best model

Output:

```
models/gender_classifier.pkl
```

---

# Evaluate Saved Models

Run

```bash
python evaluate.py
```

Outputs

### Flight Model

* RMSE
* MAE
* R² Score

### Gender Model

* Accuracy
* Precision
* Recall
* F1 Score

---

# MLflow Tracking

All experiments are automatically logged.

Logged information includes

* Parameters
* Metrics
* Model artifacts
* Experiment history

If running locally

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

If configured with DagsHub, experiments will also appear in the shared MLflow dashboard.

---

# Model Artifacts

The following files are generated after training.

```
models/

flight_price_model.pkl

gender_classifier.pkl
```

These files are production-ready Scikit-learn pipelines and already include preprocessing.

No additional preprocessing is required before inference.

---

# Integration Guide

## For Member 2 (Streamlit)

Use the Flask API provided by Member 3.

Do not load the `.pkl` models directly inside the Streamlit application.

The Streamlit frontend should send user input to the Flask API and display the returned predictions.

---

## For Member 3 (Backend)

Load the saved models using Joblib.

```python
import joblib

flight_model = joblib.load("models/flight_price_model.pkl")

gender_model = joblib.load("models/gender_classifier.pkl")
```

Since preprocessing is already included in the pipeline, pass raw input data with the same feature names used during training.

No manual encoding or scaling is required.

---

## For Member 4 (DevOps)

The following files should be packaged inside the Flask Docker container.

```
models/

flight_price_model.pkl

gender_classifier.pkl
```

The backend container should expose prediction endpoints, while Kubernetes deployment should mount these model artifacts or include them in the image.

---

# Notes

* All preprocessing is implemented using Scikit-learn Pipelines.
* Models are exported as Joblib artifacts.
* Experiments are tracked using MLflow.
* The preprocessing pipeline is embedded within each model, ensuring consistent behavior during inference.
* Any new input must follow the same schema as the training dataset.

---

# Author

Prediction Modeler (Member 1)

Voyage Analytics – Integrating MLOps in Travel Productionization of ML Systems
