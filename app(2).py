from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model("model.keras")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/")
def home():
    return "Azure Cost Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df["UsageDate"] = pd.to_datetime(df["UsageDate"])
        df["Year"] = df["UsageDate"].dt.year
        df["Month"] = df["UsageDate"].dt.month
        df["Day"] = df["UsageDate"].dt.day
        df.drop("UsageDate", axis=1, inplace=True)
        processed_data = preprocessor.transform(df)
        prediction = model.predict(processed_data)
        return jsonify({"Predicted Cost": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
