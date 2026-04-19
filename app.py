import streamlit as st
import pandas as pd
import pickle

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Encode inputs
def encode(val, mapping):
    return mapping[val]

gender_enc = encode(gender, {"Male": 1, "Female": 0})
partner_enc = encode(partner, {"Yes": 1, "No": 0})
dependents_enc = encode(dependents, {"Yes": 1, "No": 0})
phone_enc = encode(phone, {"Yes": 1, "No": 0})
internet_enc = encode(internet, {"DSL": 0, "Fiber optic": 1, "No": 2})
contract_enc = encode(contract, {"Month-to-month": 0, "One year": 1, "Two year": 2})

# Build input dataframe matching training columns exactly
input_data = pd.DataFrame([[gender_enc, senior, partner_enc, dependents_enc,
                             tenure, phone_enc, 0, internet_enc, 0, 0, 0, 0,
                             0, 0, 0, contract_enc, 0, monthly, total]],
                           columns=["gender", "SeniorCitizen", "Partner", "Dependents",
                                    "tenure", "PhoneService", "MultipleLines",
                                    "InternetService", "OnlineSecurity", "OnlineBackup",
                                    "DeviceProtection", "TechSupport", "StreamingTV",
                                    "StreamingMovies", "PaperlessBilling", "Contract",
                                    "PaymentMethod", "MonthlyCharges", "TotalCharges"])

# Match exact column order from training
input_data = input_data[model.get_booster().feature_names]

if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ This customer is likely to CHURN! Probability: {probability:.2%}")
    else:
        st.success(f"✅ This customer is likely to STAY! Probability of churn: {probability:.2%}")