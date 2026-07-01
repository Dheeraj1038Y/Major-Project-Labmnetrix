"""
evaluate.py

Evaluate saved ML models
"""

import joblib
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import *

from data_prep import (
    load_flights_data,
    prepare_flight_data,
    load_users_data,
    prepare_user_data,
)


# ==========================================================
# FLIGHT MODEL
# ==========================================================

def evaluate_regression():

    print("\nEvaluating Flight Price Model...")

    model = joblib.load(FLIGHT_MODEL_PATH)

    df = load_flights_data()

    X, y = prepare_flight_data(df)

    predictions = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("=" * 50)
    print("Flight Price Regression")
    print("=" * 50)
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")
    print("=" * 50)


# ==========================================================
# GENDER MODEL
# ==========================================================

def evaluate_classification():

    print("\nEvaluating Gender Classifier...")

    model = joblib.load(GENDER_MODEL_PATH)

    df = load_users_data()

    # Uncomment if your classifier is only male/female
    # df = df[df["gender"] != "none"]

    X, y = prepare_user_data(df)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("=" * 50)
    print("Gender Classification")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 50)


# ==========================================================
# MAIN
# ==========================================================

def main():

    evaluate_regression()

    evaluate_classification()


if __name__ == "__main__":
    main()