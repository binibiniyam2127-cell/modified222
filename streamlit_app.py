import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Load trained model and preprocessing objects ---
MODEL_ASSETS_PATH = 'model_assets'

try:
    model = joblib.load(os.path.join(MODEL_ASSETS_PATH, 'xgboost_model.pkl'))
    feature_columns = joblib.load(os.path.join(MODEL_ASSETS_PATH, 'feature_columns.pkl'))
    categorical_cols_for_encoding = joblib.load(os.path.join(MODEL_ASSETS_PATH, 'categorical_cols_for_encoding.pkl'))
    st.success("Model assets loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model assets not found. Make sure 'model_assets' folder is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- Streamlit UI ---
st.title('Transaction Fraud Detection App')
st.write('Enter transaction details to predict if it\'s fraudulent.')

# Input fields for transaction details
timestamp = st.number_input('Timestamp (e.g., 1678886400)', min_value=0, value=1678886400)
amount = st.number_input('Amount', min_value=0.0, value=1000.0, format="%.2f")
old_balance = st.number_input('Old Balance', min_value=0.0, value=5000.0, format="%.2f")
new_balance = st.number_input('New Balance', min_value=0.0, value=4000.0, format="%.2f")
is_international = st.selectbox('Is International?', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

# Categorical inputs
transaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
region = st.selectbox('Region', ['USA', 'Europe', 'Asia', 'Other'])
device_type = st.selectbox('Device Type', ['Mobile', 'Desktop', 'Tablet'])

if st.button('Predict Fraud'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Timestamp': [timestamp],
        'Transaction_Type': [transaction_type],
        'Amount': [amount],
        'Old_Balance': [old_balance],
        'New_Balance': [new_balance],
        'Region': [region],
        'Device_Type': [device_type],
        'Is_International': [is_international]
    })

    # Apply the same feature engineering steps
    input_data['Hour'] = input_data['Timestamp'] % 24
    input_data['Balance_Error'] = (input_data['Old_Balance'] - input_data['Amount']) - input_data['New_Balance']

    # Apply one-hot encoding
    df_encoded_input = pd.get_dummies(input_data, columns=categorical_cols_for_encoding, drop_first=True)

    # Ensure all expected columns from training are present and in the correct order
    # Add missing columns with value 0
    for col in feature_columns:
        if col not in df_encoded_input.columns:
            df_encoded_input[col] = 0
    # Drop any extra columns that weren't in the training set
    df_encoded_input = df_encoded_input[feature_columns]

    # Make prediction
    try:
        prediction_proba = model.predict_proba(df_encoded_input)[:, 1][0]
        predicted_fraud = (prediction_proba >= 0.5).astype(int)

        st.write(f"### Prediction Results:")
        if predicted_fraud == 1:
            st.error(f"\u26A0\uFE0F Fraud Detected! (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"\u2705 No Fraud Detected. (Probability: {prediction_proba:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")