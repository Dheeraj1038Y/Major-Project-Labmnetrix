"""
train_classification.py

Gender Classification Training Script
"""

import joblib
import mlflow
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import *

from data_prep import (
    load_users_data,
    prepare_user_data,
)

from utils import (
    log_parameters,
    log_classification_metrics,
    print_classification_results,
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
                OneHotEncoder(handle_unknown="ignore"),
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

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return accuracy, precision, recall, f1


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
        GENDER_EXPERIMENT
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        pipeline.fit(
            X_train,
            y_train,
        )

        accuracy, precision, recall, f1 = evaluate(
            pipeline,
            X_test,
            y_test,
        )

        log_parameters(model_name, model)
        log_classification_metrics(
            accuracy,
            precision,
            recall,
            f1,
        )

        print_classification_results(
            model_name,
            accuracy,
            precision,
            recall,
            f1,
        )

        return pipeline, f1


# ==========================================================
# MAIN
# ==========================================================

def main():

    df = load_users_data()

    X, y = prepare_user_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = create_preprocessor(
        X_train
    )

    lr_pipeline, lr_score = train_model(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    rf_pipeline, rf_score = train_model(
        RandomForestClassifier(
         n_estimators=500,
         max_depth=None,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
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

        print("\nBest Model : Logistic Regression")

    save_model(
        best_model,
        GENDER_MODEL_PATH,
    )

    print(
        f"\nModel Saved : {GENDER_MODEL_PATH}"
    )


if __name__ == "__main__":
    df = load_users_data()

    print("\nGender Distribution")
    print(df["gender"].value_counts())

    print("\nPercentage")
    print(df["gender"].value_counts(normalize=True) * 100)

    main()