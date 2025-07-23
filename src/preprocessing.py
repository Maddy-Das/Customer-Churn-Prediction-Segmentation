# ============================
# src/preprocessing.py
# ============================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.copy()

    # Drop ID column if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values in TotalCharges
    imputer = SimpleImputer(strategy='mean')
    df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Split into features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled