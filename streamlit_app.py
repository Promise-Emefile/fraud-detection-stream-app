import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# ----------------------------
# Load model and scaler
# ----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("model/fraud_model.h5")   # path to model file
    scaler = joblib.load("scaler/fraud_detection_scaler.pkl")  # path to scaler file
    return model, scaler

model, scaler = load_trained_model()

# ----------------------------
# Load dataset to extract features
# ----------------------------
@st.cache_data
def get_feature_names():
    df = pd.read_csv("synthetic_fraud_dataset.csv")
    feature_names = df.drop(["Fraud_Label", "Transaction_ID", "User_ID","Location","Timestamp"], axis=1).columns.tolist()
    return feature_names

feature_names = get_feature_names()

# ----------------------------
# Streamlit App
# ----------------------------
st.title("💳 Fraud Detection App")
st.write("Predict whether a transaction is fraudulent using a trained deep learning model.")

# Collect inputs dynamically
st.sidebar.header("Enter Transaction Details")
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0)
    user_input.append(value)

# Convert input into numpy array
input_array = np.array(user_input).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Prediction
if st.sidebar.button("Predict Fraud"):
    prediction = model.predict(scaled_input)
    fraud_prob = float(prediction[0][0])

    st.subheader("🔎 Prediction Result")
    if fraud_prob > 0.5:
        st.error(f"⚠ This transaction is predicted FRAUDULENT (probability = {fraud_prob:.2f})")
    else:
        st.success(f"✅ This transaction is predicted LEGITIMATE (probability = {fraud_prob:.2f})")
