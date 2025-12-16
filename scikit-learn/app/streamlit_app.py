import streamlit as st
import requests

st.title("🔮 Customer Churn Predictor")
st.write("Predict churn with 93.9% accuracy!")

# Input form
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 118, 70)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    # ... more inputs

if st.button("Predict Churn"):
    # Call API
    response = requests.post("http://localhost:8000/predict", json=customer_data)
    result = response.json()

    # Display results
    if result["churn_prediction"] == "Yes":
        st.error(f"⚠️ High Churn Risk: {result['churn_probability']:.1%}")
    else:
        st.success(f"✅ Low Churn Risk: {result['churn_probability']:.1%}")

    st.metric("Risk Level", result["risk_level"])
    st.write("**Risk Factors:**")
    for factor in result["key_risk_factors"]:
        st.write(f"- {factor}")
