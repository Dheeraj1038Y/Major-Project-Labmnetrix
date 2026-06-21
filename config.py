"""
config.py

Central configuration file for the Voyage Analytics MLOps project.
All paths, constants, and MLflow configuration are defined here.
"""

from pathlib import Path
import os

# ==========================================================
# PROJECT PATHS
# ==========================================================

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Models directory
MODEL_DIR = BASE_DIR / "models"

# MLflow artifact directory (used if local tracking is enabled)
MLRUNS_DIR = BASE_DIR / "mlruns"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# ==========================================================
# DATASET PATHS
# ==========================================================

FLIGHTS_DATA_PATH = DATA_DIR / "flights.csv"
USERS_DATA_PATH = DATA_DIR / "users.csv"
HOTELS_DATA_PATH = DATA_DIR / "hotels.csv"

# ==========================================================
# SAVED MODEL PATHS
# ==========================================================

FLIGHT_MODEL_PATH = MODEL_DIR / "flight_price_model.pkl"
GENDER_MODEL_PATH = MODEL_DIR / "gender_classifier.pkl"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

TEST_SIZE = 0.20

# ==========================================================
# MLFLOW CONFIGURATION (DagsHub)
# ==========================================================

# IMPORTANT:
# Replace these with environment variables before production.
# They are included here because your project handbook specifies them.

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/Dheeraj1038Y/Major-Project-Labmnetrix.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = "Dheeraj1038Y"

os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "ad86804ecf1b5d368fc14f5013432ab3e43c964e"
)

# Experiment names

FLIGHT_EXPERIMENT = "voyage_flight_prediction"
GENDER_EXPERIMENT = "voyage_gender_classification"

# ==========================================================
# REGRESSION TARGET
# ==========================================================

FLIGHT_TARGET = "price"

# ==========================================================
# CLASSIFICATION TARGET
# ==========================================================

GENDER_TARGET = "gender"

# ==========================================================
# MODEL NAMES
# ==========================================================

FLIGHT_MODEL_NAME = "flight_price_predictor"

GENDER_MODEL_NAME = "gender_classifier"

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

DATE_COLUMN = "date"

# Columns to drop

FLIGHT_DROP_COLUMNS = [
    "travelCode",
    "userCode"
]

USER_DROP_COLUMNS = [
    "code"
]

# ==========================================================
# REGRESSION METRICS
# ==========================================================

REGRESSION_METRICS = [
    "rmse",
    "mae",
    "r2"
]

# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

"""
config.py
Centralized configurations for Voyage Analytics MLOps Pipeline"""

