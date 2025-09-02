import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

#loading model and scalar
@st.cache_resource
def load_trained_model():
  model = load_model("fraud_detection_model.h5")
  scaler = joblib.load("fraud_detection_scaler.pkl")
  return model, scaler

model, scalar = load_trained_model()

#loading dataset to extract features
@st.cache_data
def get_feature_names():
  df = pd.read_csv("synthetic_fraud_dataset.csv")
  features_names = df.drop(["Fraud_Label", "Transaction_ID", "User_ID"], axis=1).columns.tolist()
  return feature_name

feature_names = get_feature_names()

st.title("Fraud Detection App")
st.write("Predict whether a transaction is fraudulent using trained deep learning model.")
#inputs
st.sidebar.header("Enter Transaction Details")
user_input = []
for feature in feature_names:
  value = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0)
  user_input.append(value)

#converting inputs into numpy array
input_array = np.array(user_input).reshape(1, -1)
scaled_input = scaler.transform(input_array)

#prediction
if st.sidebar.button("Predict Fraud"):
  prediction = model.predict(scaled_input)
  fraud_prob = float(prediction[0][0])
  st.subheader("Prediction Result")
  if fraud_prob > 0.5:
    st.error(f"This transaction is predicted FRAUDULENT (probability = {fraud_prob:.2f})")
  else:
    st.success(f" This transaction is predicted LEGITIMATE (probability = {fraud_prob:.2f})")
