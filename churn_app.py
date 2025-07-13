import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("C:/Users/hp/OneDrive/Desktop/churn_app/rf_model.pkl")

st.title("💼 Bank Customer Churn Prediction")
st.write("Enter customer details below:")

# User Inputs
credit_score = st.slider("Credit Score", 300, 900, 600)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", value=50000.0)
products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.selectbox("Has Credit Card?", [1, 0])
active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Encode country and gender
country_encoded = {"France": 0, "Germany": 1, "Spain": 2}[country]
gender_encoded = {"Male": 1, "Female": 0}[gender]

# Prepare input
features = np.array([[credit_score, country_encoded, gender_encoded, age, tenure,
                      balance, products_number, credit_card, active_member, estimated_salary]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN!")
    else:
        st.success("✅ This customer is NOT likely to churn.")
