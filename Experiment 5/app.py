from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI application
app = FastAPI(title="Iris ML Prediction API")


# Load trained model
model = joblib.load("model/model.pkl")


# Define input data
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris ML Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisData):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    class_names = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    return {
        "prediction": int(prediction),
        "class": class_names[prediction]
    }
