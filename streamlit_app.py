import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model and preprocessor
model = tf.keras.models.load_model("model.keras")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="Azure Cost Prediction", page_icon="💰")

st.title("💰 Azure Cost Prediction App")

# -------------------------
# Inputs
# -------------------------
service_name = st.text_input("Service Name")
cost_usd = st.number_input("Cost USD", min_value=0.0, step=0.01)

currency = st.selectbox("Currency", ["USD", "INR", "GBP", "EUR"])

usage_date = st.date_input("Usage Date")

# -------------------------
# Convert date
# -------------------------
year = usage_date.year
month = usage_date.month
day = usage_date.day

# -------------------------
# Currency conversion (IMPORTANT)
# -------------------------
currency_map = {
    "USD": 83,
    "INR": 1,
    "GBP": 105,
    "EUR": 90
}

currency_value = currency_map[currency]

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):

    # Validation
    if service_name == "":
        st.error("Please enter Service Name")

    else:

        # MUST MATCH TRAINING FEATURES
        df = pd.DataFrame({
            "ServiceName": [service_name],
            "CostUSD": [cost_usd],
            "Currency": [currency_value],
            "Year": [year],
            "Month": [month],
            "Day": [day]
        })

        # Transform using trained preprocessor
        processed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(processed, verbose=0)

        st.success(f"💰 Predicted Cost: {prediction[0][0]:.2f}")
