import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

# Load the model and scaler
model = joblib.load("random_forest_model.pkl")

# Sidebar for user input
st.sidebar.header("Ethereum Fraud Detection")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Overview:")
    st.dataframe(data.head())

    # Preprocess the data (customize preprocessing as per your dataset)
    scaler = StandardScaler()
    X = data.drop(columns=["FLAG", "Index"], errors="ignore")  # Modify based on your dataset
    X_scaled = scaler.fit_transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    data["Predicted FLAG"] = predictions
    st.write("Predictions:")
    st.dataframe(data)

    # Display Evaluation Metrics
    st.write("### Feature Importance")
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
    st.pyplot(plt)

    # ROC Curve
    st.write("### ROC Curve")
    y_proba = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(data["FLAG"], y_proba)  # Assumes FLAG column exists
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    st.pyplot(plt)