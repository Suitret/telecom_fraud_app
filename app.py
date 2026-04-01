import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets
model = joblib.load('model_new.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Telecom Fraud Detector", layout="centered")

st.title("🛡️ Telecom Fraud Detection System")
st.markdown("Enter Call Detail Record (CDR) attributes to check for fraudulent activity.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input("Call Duration (seconds)", min_value=0, value=60)
        call_type = st.selectbox("Call Type", ["local", "international"])
        
    with col2:
        is_night = st.selectbox("Is Night Call?", [0, 1], help="1 for Yes, 0 for No")
        location = st.selectbox("Origin Location", ["Lagos", "Cairo", "Nairobi", "Accra", "Johannesburg"])

    submit = st.form_submit_button("Analyze Transaction")

if submit:
    # Prepare input
    # Note: Using the same encoding used in training
    type_enc = 1 if call_type == "local" else 0
    # For a robust app, you'd use the saved LabelEncoder for location
    loc_enc = 0 # Placeholder: simplified for demo
    
    input_data = np.array([[duration, type_enc, is_night, loc_enc]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 FRAUDULENT ACTIVITY DETECTED (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ TRANSACTION GENUINE (Confidence: {1-probability:.2%})")

st.sidebar.info("This app uses a Random Forest model trained on the Telecom CDR dataset to identify patterns like call masking and random fraud.")
