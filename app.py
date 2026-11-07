
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")
st.write(
    "Performance evaluation of five machine learning "
    "classification models."
)

# ==========================================
# DEFINE FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

FEATURES = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]

# ==========================================
# MODEL SELECTION
# ==========================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Classification Model",
    list(MODEL_FILES.keys())
)

# ==========================================
# LOAD TEST DATA
# ==========================================

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing test features and target column",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success("Uploaded test dataset successfully!")
    except Exception as error:
        st.error(f"Unable to read CSV file: {error}")
        st.stop()
else:
    default_file = BASE_DIR / "test_data.csv"

    if default_file.exists():
        test_data = pd.read_csv(default_file)
        st.info("Using the default test_data.csv file.")
    else:
        st.warning("Please upload a test dataset to continue.")
        st.stop()

# ==========================================
# VALIDATE TEST DATA
# ==========================================

if "y" not in test_data.columns:
    st.error("The uploaded CSV must contain the target column 'y'.")
    st.stop()

missing_features = [
    feature for feature in FEATURES
    if feature not in test_data.columns
]

if missing_features:
    st.error(f"Missing required features: {missing_features}")
    st.stop()

X_test = test_data[FEATURES].copy()

# Convert target into binary format
y_test = test_data["y"].copy()

if y_test.dtype == "object":
    y_test = y_test.astype(str).str.strip().str.lower()
    y_test = y_test.map({"no": 0, "yes": 1})

y_test = pd.to_numeric(y_test, errors="coerce")

if y_test.isnull().any() or not set(y_test.unique()).issubset({0, 1}):
    st.error("Target column 'y' must contain only 0/1 or no/yes values.")
    st.stop()

y_test = y_test.astype(int)

if len(y_test) == 0:
    st.error("The uploaded test dataset is empty.")
    st.stop()

if y_test.nunique() != 2:
    st.error("The test dataset must contain both target classes for AUC.")
    st.stop()

st.write("Test Dataset Shape:", test_data.shape)
st.dataframe(test_data.head())

# ==========================================
# LOAD SELECTED MODEL
# ==========================================

st.header("Model Evaluation")

model_path = MODEL_DIR / MODEL_FILES[selected_model]

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# ==========================================
# GENERATE PREDICTIONS
# ==========================================

try:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

# ==========================================
# CALCULATE EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_score)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# ==========================================
# DISPLAY METRICS
# ==========================================

st.subheader(f"Performance Metrics: {selected_model}")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
)

st.dataframe(cm_df)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.round(4))

st.success("Model evaluation completed successfully!")
