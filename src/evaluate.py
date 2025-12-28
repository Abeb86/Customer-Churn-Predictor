import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_data_path):
    # 1. Load the saved Pipeline (Model + Preprocessor)
    model = joblib.load(model_path)
    
    # 2. Load the test data
    df = pd.read_csv(test_data_path)
    X_test = df.drop('Churn', axis=1)
    y_test = df['Churn']
    
    # 3. Make Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 4. Print the "Business Verdict"
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")
    
    # 5. Visualize the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Who did we miss?')
    plt.savefig('notebooks/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    # Note: Ensure you saved a test CSV during your training step!
    evaluate_model('models/churn_pipeline.pkl', 'data/cleaned_telco_customer_churn.csv') 