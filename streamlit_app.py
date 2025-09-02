import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = load_model('fraud_model.h5')  # Your saved Keras model
scaler = pickle.load(open('fraud_detection_scaler.pkl', 'wb'))  # Your saved scaler

# App title
st.title("🕵️‍♀️ Fraud Detection Web App")
st.markdown("Upload transaction data to predict fraudulent activity.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Read data
    data = pd.read_csv(uploaded_file)

    # Display raw data
    st.subheader("📄 Uploaded Data")
    st.write(data.head())

    # Preprocess: drop non-numeric or irrelevant columns if needed
    features = data.select_dtypes(include=[np.number])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    predictions = model.predict(scaled_features)
    data['Fraud_Prediction'] = (predictions > 0.5).astype(int)

    # Show results
    st.subheader("🔍 Prediction Results")
    st.write(data[['Fraud_Prediction']].value_counts().rename("Count").reset_index())

    st.subheader("📊 Detailed Output")
    st.write(data)

    # Download option
    st.download_button("Download Results", data.to_csv(index=False), file_name="fraud_predictions.csv")

