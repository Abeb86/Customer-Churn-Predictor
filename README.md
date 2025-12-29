# Customer Churn Predictor

A machine learning project that predicts customer churn for a telecommunications company using XGBoost classifier. The project includes data preprocessing, model training, evaluation, and an interactive Streamlit web application for real-time predictions.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)

## ✨ Features

- **Data Preprocessing**: Automated data cleaning and feature engineering pipeline
- **Machine Learning Model**: XGBoost classifier for churn prediction
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Model Evaluation**: Comprehensive metrics and visualization tools
- **Production-Ready Pipeline**: End-to-end ML pipeline with preprocessing and prediction

## 📁 Project Structure

```
Customer-Churn-Predictor/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Raw data
│   └── cleaned_telco_customer_churn.csv         # Cleaned data
│
├── models/
│   └── churn_pipeline.pkl                       # Trained model pipeline
│
├── notebooks/
│   ├── 01_exploration.ipynb                     # Data exploration
│   ├── data_cleaning.ipynb                      # Data cleaning notebook
│   └── confusion_matrix.png                     # Evaluation visualization
│
├── src/
│   ├── app.py                                   # Streamlit web application
│   ├── train.py                                 # Model training script
│   ├── evaluate.py                              # Model evaluation script
│   ├── engineering.py                          # Feature engineering
│   └── data_prep.py                             # Data preparation utilities
│
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abeb86/Customer-Churn-Predictor.git
   cd Customer-Churn-Predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Data Preparation

The raw data should be placed in the `data/` directory. The cleaned data is already available, but if you need to re-clean:

- Use the Jupyter notebooks in `notebooks/` for data exploration and cleaning
- The cleaned data will be saved as `data/cleaned_telco_customer_churn.csv`

### 2. Train the Model

Train the machine learning model:

```bash
python -m src.train
```

This will:
- Load the cleaned data
- Split into training and testing sets
- Create a preprocessing pipeline
- Train an XGBoost classifier
- Save the model to `models/churn_pipeline.pkl`

### 3. Evaluate the Model

Evaluate the trained model:

```bash
python -m src.evaluate
```

This will display:
- Classification report
- ROC-AUC score
- Confusion matrix visualization

#### Understanding the Confusion Matrix

The confusion matrix is a key evaluation metric that shows how well the model performs on the test data. It visualizes the model's predictions compared to the actual outcomes.

**Confusion Matrix Structure:**
<img width="854" height="650" alt="image" src="https://github.com/user-attachments/assets/54cb0bf6-086e-4f4d-896a-def4120ebdc3" />

**Metrics Explained:**

- **True Positives (TP)**: Customers correctly predicted to churn
- **True Negatives (TN)**: Customers correctly predicted to stay
- **False Positives (FP)**: Customers incorrectly predicted to churn (Type I error)
- **False Negatives (FN)**: Customers incorrectly predicted to stay (Type II error)

**Key Insights:**

- **High TN + TP**: Model is performing well overall
- **High FP**: Model is too aggressive in predicting churn (false alarms)
- **High FN**: Model is missing customers who will actually churn (more critical for business)

The confusion matrix is automatically saved as `notebooks/confusion_matrix.png` when you run the evaluation script. This visualization helps identify:
- Model accuracy for each class
- Areas where the model needs improvement
- Balance between precision and recall

### 4. Run the Web Application

Launch the interactive Streamlit app:

```bash
streamlit run src/app.py
```

The app will open in your browser (typically at `http://localhost:8501`).

#### Using the Web App

1. **Enter Customer Information** in the sidebar:
   - Tenure (months)
   - Monthly Charges
   - Total Charges
   - Contract Type
   - Payment Method
   - Internet Service
   - Tech Support

2. **Click "Predict Churn"** to get:
   - Churn probability percentage
   - Risk classification (High/Low)
   - Visual progress bar
   - Detailed debug information

3. **Test Scenarios**:
   - **High Churn Risk**: Short tenure, month-to-month contract, electronic check payment, high charges
   - **Low Churn Risk**: Long tenure, two-year contract, automatic payment, moderate charges

## 🤖 Model Details

### Algorithm
- **XGBoost Classifier**
  - Number of estimators: 100
  - Learning rate: 0.1
  - Max depth: 5

### Model Evaluation

The model performance is evaluated using several metrics:

#### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's predictions:

| Metric | Description | Business Impact |
|--------|-------------|-----------------|
| **True Positives (TP)** | Correctly identified churners | Enables proactive retention efforts |
| **True Negatives (TN)** | Correctly identified non-churners | Prevents unnecessary intervention costs |
| **False Positives (FP)** | Incorrectly flagged as churners | Wasted retention resources |
| **False Negatives (FN)** | Missed actual churners | Lost customers and revenue |

**Interpretation:**
- A good model should minimize False Negatives (missing actual churners is costly)
- The confusion matrix visualization is saved to `notebooks/confusion_matrix.png`
- The heatmap shows the distribution of predictions vs. actual outcomes

#### Other Evaluation Metrics

- **Classification Report**: Provides precision, recall, and F1-score for each class
- **ROC-AUC Score**: Measures the model's ability to distinguish between churn and no-churn
- **Accuracy**: Overall percentage of correct predictions

### Features

The model uses 19 features:

**Numeric Features:**
- `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
- `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`, `PaperlessBilling`

**Categorical Features:**
- `InternetService` (DSL, Fiber optic, No)
- `Contract` (Month-to-month, One year, Two year)
- `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card)

### Preprocessing Pipeline

1. **TotalCharges Conversion**: Converts string values to numeric (handles missing/empty values)
2. **Imputation**: 
   - Numeric features: Median imputation
   - Categorical features: Most frequent imputation
3. **Scaling**: StandardScaler for numeric features
4. **Encoding**: OneHotEncoder for categorical features (drop='first')

## 🔑 Key Features

### Main Churn Factors

The model focuses on these key factors that drive churn:

1. **Tenure**: How long the customer has been with the company
2. **Contract Type**: Month-to-month contracts have higher churn risk
3. **Payment Method**: Electronic checks correlate with higher churn
4. **Monthly Charges**: Higher charges increase churn risk
5. **Internet Service**: Service type affects retention
6. **Tech Support**: Availability of support reduces churn

### Model Pipeline

The complete pipeline includes:
- Data type conversion
- Missing value imputation
- Feature scaling
- Categorical encoding
- Model prediction

All steps are automated and saved in the pipeline for consistent predictions.

## 🛠 Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities and preprocessing
- **XGBoost**: Gradient boosting classifier
- **Streamlit**: Web application framework
- **Joblib**: Model serialization
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive notebooks for exploration

## 📊 Data

The dataset used is the Telco Customer Churn dataset, which contains information about:
- Customer demographics
- Services subscribed
- Account information
- Churn status

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


**Note**: Make sure to have the trained model (`models/churn_pipeline.pkl`) before running the web application. If the model doesn't exist, run the training script first.
