import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# App title
st.title("🕵️‍♀️ Fraud Detection Web App")
st.markdown("Upload transaction data to predict fraudulent activity.")

# Load model and scaler with error handling
try:
    model = load_model('fraud_model.h5')
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

try:
    scaler = pickle.load(open('fraud_detection_scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Scaler loading failed: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()

    st.subheader("📄 Uploaded Data")
    st.write(data.head())

    # Validate expected features
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = data.select_dtypes(include=[np.number]).columns.tolist()

    missing = [feat for feat in expected_features if feat not in data.columns]
    if missing:
        st.warning(f"⚠️ Missing expected features: {missing}")
        st.stop()

    # Align and scale features
    try:
        features = data[expected_features]
        scaled_features = scaler.transform(features)
    except Exception as e:
        st.error(f"❌ Feature scaling failed: {e}")
        st.stop()

    # Predict
    try:
        predictions = model.predict(scaled_features)
        data['Fraud_Prediction'] = (predictions > 0.5).astype(int)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Show results
    st.subheader("🔍 Prediction Results")
    st.write(data['Fraud_Prediction'].value_counts().rename("Count").reset_index())

    st.subheader("📊 Detailed Output")
    st.write(data)

    # Download option
    st.download_button("📥 Download Results", data.to_csv(index=False), file_name="fraud_predictions.csv")
