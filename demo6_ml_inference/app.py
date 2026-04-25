from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from typing import List

app = FastAPI(
    title="ML Inference API",
    description="Simple ML model inference API for iris classification",
    version="1.0.0"
)

# Load model and metadata
MODEL_PATH = "models/iris_model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
TARGET_NAMES_PATH = "models/target_names.pkl"

# Check if models exist
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found! Run train_model.py first.")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURE_NAMES_PATH, "rb") as f:
    feature_names = pickle.load(f)

with open(TARGET_NAMES_PATH, "rb") as f:
    target_names = pickle.load(f)


# Define request/response models
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    features_used: List[str]


class BatchPredictionResponse(BaseModel):
    predictions: List[str]
    probabilities: List[float]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Inference API",
        "description": "Iris flower classification model",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "RandomForestClassifier",
        "classes": target_names.tolist(),
        "features": feature_names.tolist(),
        "n_features": len(feature_names),
        "n_classes": len(target_names)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: IrisData):
    """
    Make a prediction for a single iris flower

    Example:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    try:
        # Prepare features in the correct order
        features = [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]

        # Make prediction
        prediction_idx = model.predict([features])[0]
        prediction_label = target_names[prediction_idx]

        # Get prediction probability
        probabilities = model.predict_proba([features])[0]
        probability = float(probabilities[prediction_idx])

        return PredictionResponse(
            prediction=prediction_label,
            probability=probability,
            features_used=feature_names.tolist()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(data_list: List[IrisData]):
    """
    Make predictions for multiple iris flowers
    """
    try:
        if not data_list:
            raise HTTPException(status_code=400, detail="Empty input list")

        # Prepare all features
        features = []
        for data in data_list:
            features.append([
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width
            ])

        # Make predictions
        predictions_idx = model.predict(features)
        predictions = [target_names[idx] for idx in predictions_idx]

        # Get probabilities
        probabilities_array = model.predict_proba(features)
        probabilities = [float(probs[idx]) for probs, idx in zip(
            probabilities_array, predictions_idx)]

        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
