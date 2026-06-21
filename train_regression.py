"""
train_regression.py

Flight Price Prediction Training Script
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from config import *

from data_prep import (
    load_flights_data,
    prepare_flight_data,
)

from utils import (
    log_parameters,
    log_regression_metrics,
    print_regression_results,
    save_model,
)


# ==========================================================
# CREATE PREPROCESSOR
# ==========================================================

def create_preprocessor(X):

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return rmse, mae, r2


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    mlflow.set_experiment(
        FLIGHT_EXPERIMENT
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        pipeline.fit(
            X_train,
            y_train,
        )

        rmse, mae, r2 = evaluate(
            pipeline,
            X_test,
            y_test,
        )

        log_parameters(model_name, model)
        log_regression_metrics(rmse, mae, r2)

        # Save a temporary local artifact for MLflow and reuse the helper
        save_model(pipeline, "models/temp_model.pkl")
        mlflow.log_artifact("models/temp_model.pkl")

        print_regression_results(
            model_name,
            rmse,
            mae,
            r2,
        )

        return pipeline, r2


# ==========================================================
# MAIN
# ==========================================================

def main():

    df = load_flights_data()

    X, y = prepare_flight_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = create_preprocessor(
        X_train
    )

    lr_pipeline, lr_score = train_model(
        LinearRegression(),
        "Linear Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    rf_pipeline, rf_score = train_model(
        RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    if rf_score > lr_score:

        best_model = rf_pipeline

        print("\nBest Model : Random Forest")

    else:

        best_model = lr_pipeline

        print("\nBest Model : Linear Regression")

    save_model(
        best_model,
        FLIGHT_MODEL_PATH,
    )

    print(
        f"\nModel Saved : {FLIGHT_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
    