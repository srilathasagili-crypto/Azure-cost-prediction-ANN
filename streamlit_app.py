import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model and preprocessor
model = tf.keras.models.load_model("model.keras")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="Azure Cost Prediction", page_icon="💰")

st.title("💰 Azure Cost Prediction")

service_name = st.text_input("Service Name")
cost_usd = st.number_input("Cost USD", min_value=0.0, step=0.01)
currency = st.selectbox("Currency", ["USD"])
usage_date = st.date_input("Usage Date")

if st.button("Predict"):

    df = pd.DataFrame({
        "ServiceName": [service_name],
        "CostUSD": [cost_usd],
        "Currency": [currency],
        "UsageDate": [usage_date]
    })

    df["UsageDate"] = pd.to_datetime(df["UsageDate"])
    df["Year"] = df["UsageDate"].dt.year
    df["Month"] = df["UsageDate"].dt.month
    df["Day"] = df["UsageDate"].dt.day
    df.drop("UsageDate", axis=1, inplace=True)

    processed = preprocessor.transform(df)

    prediction = model.predict(processed, verbose=0)

    st.success(f"Predicted Cost: {prediction[0][0]:.2f}")
