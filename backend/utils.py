"""
utils.py

Common utility functions for Voyage Analytics
"""

import os
import joblib
import mlflow


# ==========================================================
# FILE I/O OPERATIONS
# ==========================================================

def create_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_model(model, path: str) -> None:
    """Save model using joblib."""
    create_directory(os.path.dirname(path))
    joblib.dump(model, path)
    print(f"\nModel saved successfully -> {path}")


def load_model(path: str):
    """Load saved model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ==========================================================
# MLFLOW LOGGING UTILITIES
# ==========================================================

def log_parameters(model_name: str, model) -> None:
    """
    Log model parameters to MLflow. 
    Dynamically extracts all parameters if the model supports get_params().
    """
    params = {"model_name": model_name}
    
    # Safely extract all hyperparameters if it's a scikit-learn compatible model
    if hasattr(model, "get_params"):
        params.update(model.get_params())
    else:
        # Fallback for specific attributes if get_params standard isn't available
        for attr in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
            if hasattr(model, attr):
                params[attr] = getattr(model, attr)

    mlflow.log_params(params)


def log_regression_metrics(rmse: float, mae: float, r2: float) -> None:
    """Log regression metrics to MLflow in a single batch call."""
    mlflow.log_metrics({
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


def log_classification_metrics(accuracy: float, precision: float, recall: float, f1: float) -> None:
    """Log classification metrics to MLflow in a single batch call."""
    mlflow.log_metrics({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ==========================================================
# CONSOLE OUTPUT FORMATTING
# ==========================================================

def print_regression_results(model_name: str, rmse: float, mae: float, r2: float) -> None:
    """Print formatted regression evaluation metrics."""
    border = "=" * 50
    output = (
        f"{border}\n"
        f"{model_name}\n"
        f"{border}\n"
        f"RMSE : {rmse:.2f}\n"
        f"MAE  : {mae:.2f}\n"
        f"R²   : {r2:.4f}\n"
        f"{border}"
    )
    print(output)


def print_classification_results(model_name: str, accuracy: float, precision: float, recall: float, f1: float) -> None:
    """Print formatted classification evaluation metrics."""
    border = "=" * 50
    output = (
        f"{border}\n"
        f"{model_name}\n"
        f"{border}\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
        f"{border}"
    )
    print(output)