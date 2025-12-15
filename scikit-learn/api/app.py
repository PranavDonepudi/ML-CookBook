from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("../models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    # Get customer data
    data = request.json

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify(
        {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": float(probability),
            "risk_level": "High"
            if probability > 0.7
            else "Medium"
            if probability > 0.4
            else "Low",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
