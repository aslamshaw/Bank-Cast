from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load trained model and label encoder
with open("stacking_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Run prediction
    pred = model.predict(df)
    label = le.inverse_transform(pred)[0]

    return {"prediction": label}


@app.get("/health")
def health():
    return {"status": "ok"}


# cd "Bank_Marketing"
# python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs

