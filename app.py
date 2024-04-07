from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI()

# Load the trained linear regression model
lr_model = joblib.load('lr_model.joblib')

class PredictionRequest(BaseModel):
    X: List[float]

@app.post("/predict")
def predict_sales(request: PredictionRequest):
    # Extract input data from request
    X = request.X

    # Convert input data to numpy array and reshape it
    X_array = np.array(X).reshape(-1, 1)

    # Make predictions using the loaded model
    Y_pred = lr_model.predict(X_array)

    return {"predictions": Y_pred.tolist()}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)