import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from .engineering import transform_features, convert_totalcharges_to_numeric
from pathlib import Path

# load clean data

df = pd.read_csv('data/cleaned_telco_customer_churn.csv')

# split data into features and target
x = df.drop('Churn', axis=1) 
y = df['Churn'] 

# split data into training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# build preprocessor using training features (not full x)
preprocessor = transform_features(x_train)

clf = Pipeline(steps=[
    ('convert_totalcharges', FunctionTransformer(convert_totalcharges_to_numeric, validate=False)),
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5))
])

# 4. Train (with simple diagnostic for column mismatch)
try:
    clf.fit(x_train, y_train)
except ValueError as e:
    print("ValueError during fit():", e)
    print("x_train columns:", list(x_train.columns))
    try:
        print("Preprocessor column spec:", preprocessor.transformers)
    except Exception:
        pass
    raise

# ensure models dir exists and save the whole pipeline
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(clf, 'models/churn_pipeline.pkl')
print("Model Pipeline saved to models/churn_pipeline.pkl")
