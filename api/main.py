from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

import os
model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict")
def predict(data: dict):

    # expected input features
    features = np.array([[
        data["Type_encoded"],
        data["Air_temperature_K"],
        data["Process_temperature_K"],
        data["Rotational_speed_rpm"],
        data["Torque_Nm"],
        data["Tool_wear_min"],
        data["temp_diff"],
        data["power"]
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "failure_prediction": int(prediction),
        "failure_probability": float(probability)
    }