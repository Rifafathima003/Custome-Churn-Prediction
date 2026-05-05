"""
Customer Churn Prediction — Streamlit Web App
Run: streamlit run predict.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(os.path.dirname(__file__), 'best_model.pkl'))
    scaler   = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
    features = joblib.load(os.path.join(os.path.dirname(__file__), 'feature_names.pkl'))
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("""
> **Business Problem:** Predict which customers are likely to cancel their subscription,
> so the retention team can act before they leave.
""")
st.divider()

if not model_loaded:
    st.warning("⚠️ Model not found. Please run `model_training.ipynb` first to train and save the model.")
    st.stop()

# ── Sidebar — Customer Input ──────────────────────────────────────────────────
st.sidebar.header("🧑 Customer Profile")
st.sidebar.markdown("Fill in the customer details below:")

gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet        = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup   = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot     = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support    = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_mov   = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment         = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges   = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0,
                                           float(monthly_charges * tenure), step=1.0)

# ── Build input DataFrame ─────────────────────────────────────────────────────
def encode(val, mapping):
    return mapping.get(val, 0)

yes_no      = {"Yes": 1, "No": 0}
yes_no_ns   = {"Yes": 1, "No": 0, "No internet service": 2, "No phone service": 2}
contract_m  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_m  = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_m   = {
    "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
    "Electronic check": 2, "Mailed check": 3
}

tenure_group_val = (
    0 if tenure <= 12 else
    1 if tenure <= 24 else
    2 if tenure <= 48 else 3
)

input_data = {
    "gender":              [encode(gender, {"Male": 1, "Female": 0})],
    "SeniorCitizen":       [encode(senior, yes_no)],
    "Partner":             [encode(partner, yes_no)],
    "Dependents":          [encode(dependents, yes_no)],
    "tenure":              [tenure],
    "PhoneService":        [encode(phone_service, yes_no)],
    "MultipleLines":       [encode(multiple_lines, yes_no_ns)],
    "InternetService":     [encode(internet, internet_m)],
    "OnlineSecurity":      [encode(online_security, yes_no_ns)],
    "OnlineBackup":        [encode(online_backup, yes_no_ns)],
    "DeviceProtection":    [encode(device_prot, yes_no_ns)],
    "TechSupport":         [encode(tech_support, yes_no_ns)],
    "StreamingTV":         [encode(streaming_tv, yes_no_ns)],
    "StreamingMovies":     [encode(streaming_mov, yes_no_ns)],
    "Contract":            [encode(contract, contract_m)],
    "PaperlessBilling":    [encode(paperless, yes_no)],
    "PaymentMethod":       [encode(payment, payment_m)],
    "MonthlyCharges":      [monthly_charges],
    "TotalCharges":        [total_charges],
    "tenure_group":        [tenure_group_val],
    "avg_monthly_spend":   [total_charges / (tenure + 1)],
    "is_senior_monthly":   [1 if senior == "Yes" and contract == "Month-to-month" else 0],
}

input_df = pd.DataFrame(input_data)

# Align columns to training feature order
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# ── Prediction ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader("📋 Customer Summary")
    summary = {
        "Tenure":           f"{tenure} months",
        "Contract":         contract,
        "Monthly Charges":  f"${monthly_charges:.2f}",
        "Internet Service": internet,
        "Tech Support":     tech_support,
        "Payment Method":   payment,
    }
    for k, v in summary.items():
        st.markdown(f"**{k}:** {v}")

with col2:
    st.subheader("🔮 Prediction Result")
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = model.predict(input_df)[0]

    if churn_pred == 1:
        st.error(f"⚠️ **High Churn Risk**")
        st.metric("Churn Probability", f"{churn_prob * 100:.1f}%", delta="At Risk")
    else:
        st.success(f"✅ **Low Churn Risk**")
        st.metric("Churn Probability", f"{churn_prob * 100:.1f}%", delta="Retained")

    # Risk meter
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.barh(0, churn_prob, color='#E8714C' if churn_prob > 0.5 else '#4C9BE8', height=0.4)
    ax.barh(0, 1 - churn_prob, left=churn_prob, color='#eee', height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_yticks([])
    ax.set_title('Risk Meter', fontsize=9)
    st.pyplot(fig, use_container_width=False)

with col3:
    st.subheader("💡 Retention Recommendations")
    if churn_pred == 1:
        recs = []
        if contract == "Month-to-month":
            recs.append("🔄 Offer a discounted annual or 2-year contract")
        if tech_support == "No":
            recs.append("🛠️ Provide free tech support for 3 months")
        if monthly_charges > 70:
            recs.append("💰 Offer a loyalty discount or bundle deal")
        if tenure < 12:
            recs.append("🎁 Send a new customer appreciation gift or credit")
        if payment == "Electronic check":
            recs.append("💳 Incentivise auto-pay setup with a discount")
        if not recs:
            recs.append("📞 Personal outreach from retention team")
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.markdown("✅ Customer appears stable. Continue regular engagement.")
        st.markdown("- 🌟 Consider loyalty reward programme")
        st.markdown("- 📊 Monitor at next renewal period")

# ── KPI Footer ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Model Performance (on test set)")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Model",    "XGBoost (Tuned)")
kpi2.metric("Accuracy", "~82%")
kpi3.metric("ROC-AUC",  "~0.86")
kpi4.metric("Recall",   "~80%")
st.caption("Metrics from model_training.ipynb. Update after retraining.")
