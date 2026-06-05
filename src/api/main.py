from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


MODEL_FILE_MAP = {
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost"
}

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }

# Define request schema
class HouseData(BaseModel):
    Model: str
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int
    BedroomAbvGr: int
    GarageArea: int
    HouseAge: int
    TotalArea: float


@app.post("/predict")
def predict(data: HouseData):

    payload = data.dict()
    model_name = payload.pop("Model")
    input_df = pd.DataFrame([payload])

    model_file = MODEL_FILE_MAP.get(model_name, model_name.lower().replace(" ", "_"))
    model_path = Path("models") / f"{model_file}.pkl"

    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not available. Expected file: {model_path}"
        )

    model = joblib.load(model_path)

    prediction = model.predict(input_df)

    predicted_price = float(np.expm1(prediction[0]))

    return {"predicted_price": round(predicted_price, 2)}