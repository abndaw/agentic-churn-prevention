# preprocess.py
# Preprocessing pipeline for the Telco Churn dataset


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(csv_path=None, scale_numeric=False):
    """
    Loads and preprocesses Telco Churn dataset.

    Returns:
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_names
    """

    # 1. Loading data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} rows")

    # 2. Saving customerID before dropping it for agent tracking
    customer_ids = df["customerID"].copy()

    # 3. Cleaning TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 4. Droping useless columns
    df = df.drop(columns=["customerID", "gender", "PhoneService"])

    # 5. Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 6. Feature engineering
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df = df.drop(columns=["TotalCharges"])

    # 7. Categorical encoding
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }
    df["Contract"] = df["Contract"].map(contract_map)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # 8. Split X / y
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    feature_names = X.columns.tolist()

    # 9. Train / Test split (with customerID tracking)
    X_train_full, X_test, y_train_full, y_test, ids_train_full, ids_test = train_test_split(
        X, y, customer_ids,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 10. Train / Validation split (with customerID tracking)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_full, y_train_full, ids_train_full,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 -> validation = 20%
        stratify=y_train_full,
        random_state=42
    )

    # 11. Assign customerID as DataFrame index
    X_train.index = ids_train.values
    X_val.index = ids_val.values
    X_test.index = ids_test.values
    y_train.index = ids_train.values
    y_val.index = ids_val.values
    y_test.index = ids_test.values

    # 12. Scaling (only for logistic regression)
    if scale_numeric:
        numeric_cols = [
            "tenure",
            "MonthlyCharges",
            "avg_monthly_charge",
            "SeniorCitizen"
        ]
        
        scaler = StandardScaler()
        
        # Create copies to avoid SettingWithCopyWarning
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.2%} | val: {y_val.mean():.2%} | test: {y_test.mean():.2%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names