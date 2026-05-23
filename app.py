import streamlit as st
import numpy as np
import joblib

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("credit_model.pkl")   # your saved pipeline

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Credit Risk Prediction System")
st.write("Enter applicant details to predict credit default risk.")

# -----------------------
# INPUT SECTION
# -----------------------

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Monthly Income", min_value=0, value=5000)
    debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.5)

with col2:
    revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
    open_credit = st.slider("Open Credit Lines", 0, 30, 5)
    dependents = st.slider("Dependents", 0, 10, 1)

late_30_59 = st.slider("Late Payments (30-59 days)", 0, 10, 0)
late_60_89 = st.slider("Late Payments (60-89 days)", 0, 10, 0)
late_90 = st.slider("Late Payments (90+ days)", 0, 10, 0)
real_estate = st.slider("Real Estate Loans", 0, 10, 1)

# -----------------------
# PREPROCESSING (SAME AS TRAINING)
# -----------------------

# apply log transform (IMPORTANT if used before)
income = np.log1p(income)
debt_ratio = np.log1p(debt_ratio)

# -----------------------
# FINAL INPUT (ORDER MUST MATCH TRAINING)
# -----------------------

input_data = np.array([[
    revolving,
    age,
    late_30_59,
    debt_ratio,
    income,
    open_credit,
    late_90,
    real_estate,
    late_60_89,
    dependents
]])

# -----------------------
# PREDICTION
# -----------------------

if st.button("🔍 Predict Risk"):
    prob = model.predict_proba(input_data)[0][1]

    # use your best threshold (example: 0.4)
    threshold = 0.4
    pred = 1 if prob > threshold else 0

    st.subheader("Result:")

    if pred == 1:
        st.error(f"⚠️ High Risk of Default\n\nRisk Score: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Risk\n\nRisk Score: {prob*100:.2f}%")

# -----------------------
# FOOTER
# -----------------------

st.markdown("---")
st.caption("Model: Random Forest / Logistic | Built with Streamlit")