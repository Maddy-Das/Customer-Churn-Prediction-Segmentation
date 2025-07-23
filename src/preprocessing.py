import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.copy()

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled