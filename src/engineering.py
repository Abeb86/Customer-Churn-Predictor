from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

def convert_totalcharges_to_numeric(X):
    """Convert TotalCharges column to numeric, replacing invalid values with NaN"""
    X = X.copy()
    if isinstance(X, pd.DataFrame) and 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    return X

def transform_features(df):
    # lists of expected columns (will be filtered to only those present in df)
    categorical_features = ['InternetService', 'Contract', 'PaymentMethod']
    numeric_features = ['gender', 'tenure', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    # Filter the feature lists to only include columns that exist in the passed dataframe.
    # This prevents ColumnTransformer from being created with columns missing from the input.
    categorical_features = [c for c in categorical_features if c in df.columns]
    numeric_features = [c for c in numeric_features if c in df.columns and c not in categorical_features]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor
