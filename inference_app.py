import os
import json
import numpy as np
import mlflow
import mlflow.keras
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# This is just an example for Iris data
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

RUN_INFO_JSON = "run_info.json"  # Or pass via an env var if needed

if not os.path.exists(RUN_INFO_JSON):
    raise FileNotFoundError(f"{RUN_INFO_JSON} not found in container.")

# 1) Load run_info.json, take the top (most recent) entry
with open(RUN_INFO_JSON, "r") as f:
    run_history = json.load(f)

if len(run_history) == 0:
    raise ValueError("No entries found in run_info.json")

latest_run = run_history[0]
keras_model_uri = latest_run["keras_model_uri"]  # e.g. "file:///app/mlruns/abc123/artifacts/my_keras_model"

print(f"Loading Keras model from {keras_model_uri}")
model = mlflow.keras.load_model(keras_model_uri)

app = FastAPI()

@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array([[data.sepal_length, data.sepal_width,
                            data.petal_length, data.petal_width]], dtype=np.float32)
    prediction = model.predict(input_data)
    # Return raw prediction array
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    # Host on 0.0.0.0 so Docker can expose it, port 8000 by default
    uvicorn.run(app, host="0.0.0.0", port=8000)
