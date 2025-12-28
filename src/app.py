import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path so src module can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 1. Load the trained pipeline
model_path = project_root / 'models' / 'churn_pipeline.pkl'
model = joblib.load(model_path)

st.title("Customer Churn Prediction Tool")
st.write("Enter customer details below to predict the risk of cancellation.")

# Quick test scenarios
with st.expander("💡 Quick Test Scenarios"):
    st.write("**High Churn Risk:**")
    st.write("- Short tenure (0-6 months)")
    st.write("- Month-to-month contract")
    st.write("- Electronic check payment")
    st.write("- High monthly charges (>$80)")
    st.write("- No tech support")
    st.write("- Fiber optic internet")
    
    st.write("**Low Churn Risk:**")
    st.write("- Long tenure (>24 months)")
    st.write("- Two year contract")
    st.write("- Automatic payment (bank/credit card)")
    st.write("- Moderate charges")
    st.write("- Tech support: Yes")

# 2. Create the User Interface (Inputs) - Main Factors Only
st.sidebar.header("Key Customer Information")

# Main factors that drive churn
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

# 3. Create a DataFrame with main factors + default values for other required features
input_dict = {
    # Main factors (user input)
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaymentMethod': [payment_method],
    'InternetService': [internet],
    'TechSupport': [1 if tech_support == "Yes" else 0],
    
    # Default values for other required features
    'gender': [1],  # Default: Male
    'Partner': [0],  # Default: No
    'Dependents': [0],  # Default: No
    'PhoneService': [1],  # Default: Yes
    'MultipleLines': [0],  # Default: No
    'OnlineSecurity': [0],  # Default: No
    'OnlineBackup': [0],  # Default: No
    'DeviceProtection': [0],  # Default: No
    'StreamingTV': [0],  # Default: No
    'StreamingMovies': [0],  # Default: No
    'PaperlessBilling': [0],  # Default: No
    'SeniorCitizen': [0]  # Default: No
}

# Note: The pipeline handles the encoding, so we pass raw values!
input_df = pd.DataFrame(input_dict)

# 4. Prediction Logic
if st.button("Predict Churn"):
    try:
        # The pipeline automatically scales and encodes the input_df!
        prediction = model.predict(input_df)
        probability = float(model.predict_proba(input_df)[0][1])  # Convert to Python float
        churn_prob = probability * 100
        
        # Display results with more detail
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Show probability as a progress bar
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        st.progress(probability)  # Now it's a Python float
        
        # Classification with threshold
        threshold = 0.5
        if prediction[0] == 1 or churn_prob >= threshold * 100:
            st.error(f"⚠️ **HIGH RISK**: This customer is likely to CHURN!")
            st.info(f"Churn probability: **{churn_prob:.2f}%** (Threshold: {threshold*100:.0f}%)")
        else:
            st.success(f"✅ **LOW RISK**: This customer is likely to stay.")
            st.info(f"Churn probability: **{churn_prob:.2f}%** (Threshold: {threshold*100:.0f}%)")
        
        # Debug info (expandable)
        with st.expander("Debug Information"):
            st.write("**Prediction:**", "Churn" if prediction[0] == 1 else "No Churn")
            st.write("**Probability (Churn):**", f"{probability:.4f}")
            st.write("**Probability (Stay):**", f"{1-probability:.4f}")
            st.write("**Input DataFrame:**")
            st.dataframe(input_df)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)