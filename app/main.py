# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ✅ Initialize FastAPI
app = FastAPI(title="ML Inference API")

# ✅ Load trained model
model = joblib.load("app/model.pkl")

# ✅ Request schema
class StudentData(BaseModel):
    study_hours: float
    sleep_hours: float

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: StudentData):
    features = np.array([[data.study_hours, data.sleep_hours]])
    prediction = model.predict(features)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return {
        "study_hours": data.study_hours,
        "sleep_hours": data.sleep_hours,
        "prediction": result
    }
