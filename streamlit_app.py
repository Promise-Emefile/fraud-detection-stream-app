import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# App title
st.title("Fraud Detection Web App")
st.markdown("Upload transaction data or enter a single transaction to predict fraudulent activity.")

# Load model and scaler
try:
    model = load_model('model/fraud_model.h5')
    scaler = pickle.load(open('scaler/fraud_detection_scaler.pkl', 'rb'))  # MinMaxScaler
    feature_list = pickle.load(open('feature_list.pkl', 'rb'))
except Exception as e:
    st.error(f" Failed to load model or scaler: {e}")
    st.stop()

# Define expected features (excluding dropped columns)
expected_features = [
    'Transaction_Amount', 'Transaction_Type', 'Account_Balance', 'Device_Type',
    'Merchant_Category', 'IP_Address_Flag', 'Previous_Fraudulent_Activity',
    'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
    'Card_Type', 'Card_Age', 'Transaction_Distance', 'Authentication_Method',
    'Risk_Score', 'Is_Weekend'
]

# Batch Prediction
st.subheader("Upload CSV or Excel File")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.write("Preview of Uploaded Data")
    st.write(data.head())

    # Drop irrelevant columns
    drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'Location']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # Encode categorical columns
    data_encoded = pd.get_dummies(data)

    # Align with expected features
    for col in expected_features:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    data_encoded = data_encoded[expected_features]

    # Scale and predict
    try:
        scaled = scaler.transform(data_encoded)
        predictions = model.predict(scaled)
        data['Fraud_Prediction'] = (predictions > 0.5).astype(int)
        st.success("Predictions completed.")
        st.write(data[['Fraud_Prediction']].value_counts().rename("Count").reset_index())
        st.write(data)

        st.download_button("Download Results", data.to_csv(index=False), file_name="fraud_predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Single Transaction Entry 
st.subheader("Enter a Single Transaction")

with st.form("single_entry"):
    # Numerical inputs
    Transaction_Amount = st.number_input("Transaction Amount", min_value=0.0)
    Account_Balance = st.number_input("Account Balance", min_value=0.0)
    IP_Address_Flag = st.selectbox("IP Address Flag", [0, 1])
    Previous_Fraudulent_Activity = st.selectbox("Previous Fraudulent Activity", [0, 1])
    Daily_Transaction_Count = st.number_input("Daily Transaction Count", min_value=0)
    Avg_Transaction_Amount_7d = st.number_input("Avg Transaction Amount (7d)", min_value=0.0)
    Failed_Transaction_Count_7d = st.number_input("Failed Transaction Count (7d)", min_value=0)
    Card_Age = st.number_input("Card Age", min_value=0)
    Transaction_Distance = st.number_input("Transaction Distance", min_value=0.0)
    Risk_Score = st.slider("Risk Score", min_value=0.0, max_value=1.0, step=0.01)
    Is_Weekend = st.selectbox("Is Weekend", [0, 1])

    # Categorical inputs
    Transaction_Type = st.selectbox("Transaction Type", ['POS', 'Bank Transfer', 'Online', 'ATM Withdrawal'])
    Device_Type = st.selectbox("Device Type", ['Laptop', 'Mobile', 'Tablet'])
    Merchant_Category = st.selectbox("Merchant Category", ['Travel', 'Clothing', 'Restaurants', 'Electronics'])
    Card_Type = st.selectbox("Card Type", ['Amex', 'Mastercard', 'Visa'])
    Authentication_Method = st.selectbox("Authentication Method", ['Biometric', 'Password', 'OTP'])

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_dict = {
        'Transaction_Amount': Transaction_Amount,
        'Transaction_Type': Transaction_Type,
        'Account_Balance': Account_Balance,
        'Device_Type': Device_Type,
        'Merchant_Category': Merchant_Category,
        'IP_Address_Flag': IP_Address_Flag,
        'Previous_Fraudulent_Activity': Previous_Fraudulent_Activity,
        'Daily_Transaction_Count': Daily_Transaction_Count,
        'Avg_Transaction_Amount_7d': Avg_Transaction_Amount_7d,
        'Failed_Transaction_Count_7d': Failed_Transaction_Count_7d,
        'Card_Type': Card_Type,
        'Card_Age': Card_Age,
        'Transaction_Distance': Transaction_Distance,
        'Authentication_Method': Authentication_Method,
        'Risk_Score': Risk_Score,
        'Is_Weekend': Is_Weekend
    }

    input_df = pd.DataFrame([input_dict])
    encoded_input = pd.get_dummies(input_df)

    for col in feature_list:
        if col not in encoded_input.columns:
            encoded_input[col] = 0
    encoded_input = encoded_input[feature_list]

    try:
        scaled_input = scaler.transform(encoded_input)
        prediction = model.predict(scaled_input)[0][0]
        label = "Fraudulent" if prediction > 0.5 else "Legitimate"
        st.success(f" Prediction: {label} (Confidence: {prediction:.2f})")
    except Exception as e:
        st.error(f" Prediction failed: {e}")
