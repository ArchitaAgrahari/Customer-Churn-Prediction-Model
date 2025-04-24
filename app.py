import streamlit as st
import numpy as np
import pickle

# Set up Streamlit
st.set_page_config(page_title="Churn Predictor")
st.title("💼 Bank Customer Churn Predictor")

# Load model and scaler
# Load model and scaler safely
import os

try:
    with open(os.path.join(os.path.dirname(__file__), "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please make sure 'xgb_model.pkl' and 'scaler.pkl' are in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error loading model/scaler: {e}")
    st.stop()



# Form for user input
st.markdown("### Fill in the customer details:")

credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, step=1000.0)
salary = st.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, step=1000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Encoding input
gender_val = 1 if gender == "Male" else 0
geo_val = {"France": 0, "Germany": 1, "Spain": 2}[geography]
cr_card_val = 1 if has_cr_card == "Yes" else 0
active_val = 1 if is_active_member == "Yes" else 0

# Prepare input array
input_data = np.array([[credit_score, age, tenure, balance, salary,
                        num_of_products, cr_card_val, active_val,
                        gender_val, geo_val]])

# Scale the first 5 numerical features only
input_scaled = input_data.copy()
input_scaled[:, :5] = scaler.transform(input_data[:, :5])

# Predict button
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This customer is likely to STAY (Probability: {1 - proba:.2f})")
