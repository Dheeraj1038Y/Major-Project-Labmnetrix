"""
data_prep.py

Handles:
1. Loading datasets
2. Basic cleaning
3. Feature engineering
4. Splitting features and target

NOTE:
Scaling, encoding, and imputation are NOT performed here.
Those belong inside the sklearn Pipeline.
"""

import pandas as pd

from config import (
    FLIGHTS_DATA_PATH,
    USERS_DATA_PATH,
    FLIGHT_TARGET,
    GENDER_TARGET,
    FLIGHT_DROP_COLUMNS,
    USER_DROP_COLUMNS
)


# ==========================================================
# LOAD DATASETS
# ==========================================================

def load_flights_data():
    """
    Load flights dataset.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(FLIGHTS_DATA_PATH)


def load_users_data():
    """
    Load users dataset.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(USERS_DATA_PATH)


# ==========================================================
# FLIGHT FEATURE ENGINEERING
# ==========================================================

def prepare_flight_data(df):
    """
    Perform feature engineering for flight dataset.

    - Remove ID columns
    - Convert date to datetime
    - Extract month/day/weekday
    - Separate X and y
    """

    df = df.copy()

    # Remove ID columns if present
    df.drop(
        columns=[col for col in FLIGHT_DROP_COLUMNS if col in df.columns],
        inplace=True,
        errors="ignore"
    )

    # Date features
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.day_name()

        df.drop(columns=["date"], inplace=True)

    # Features and Target
    X = df.drop(columns=[FLIGHT_TARGET])

    y = df[FLIGHT_TARGET]

    return X, y


# ==========================================================
# USER FEATURE ENGINEERING
# ==========================================================

def prepare_user_data(df):
    """
    Feature engineering for users dataset.

    - Remove ID column
    - Extract first name
    - Remove original name
    - Separate X and y
    """

    df = df.copy()

    # Remove ID
    df.drop(
        columns=[col for col in USER_DROP_COLUMNS if col in df.columns],
        inplace=True,
        errors="ignore"
    )

    # Extract first name
    if "name" in df.columns:
        df["first_name"] = (
            df["name"]
            .astype(str)
            .str.split()
            .str[0]
        )

        # Feature 1
        df["name_length"] = (
            df["first_name"]
            .str.len()
        )

        # Feature 2
        df["first_letter"] = (
            df["first_name"]
            .str[0]
        )

        # Feature 3
        df["last_letter"] = (
            df["first_name"]
            .str[-1]
        )

        # Remove original full name
        df.drop(columns=["name"], inplace=True)

    X = df.drop(columns=[GENDER_TARGET])

    y = df[GENDER_TARGET]

    return X, y


# ==========================================================
# DISPLAY DATA INFO
# ==========================================================

def dataset_summary(df, dataset_name="Dataset"):
    """
    Print basic information about a dataframe.
    """

    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Shape : {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("=" * 60)


# ==========================================================
# TEST MODULE
# ==========================================================

if __name__ == "__main__":

    flights = load_flights_data()
    users = load_users_data()

    dataset_summary(flights, "Flights Dataset")
    dataset_summary(users, "Users Dataset")

    X_flight, y_flight = prepare_flight_data(flights)
    X_user, y_user = prepare_user_data(users)

    print("\nFlight Features Shape :", X_flight.shape)
    print("Flight Target Shape   :", y_flight.shape)

    print("\nUser Features Shape   :", X_user.shape)
    print("User Target Shape     :", y_user.shape)