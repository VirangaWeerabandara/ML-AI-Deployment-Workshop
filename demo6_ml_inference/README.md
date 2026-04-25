# ML Model Inference with FastAPI

A complete tutorial on deploying a machine learning model using FastAPI for real-time inference.

## Overview

This demo shows:

- Training and saving an ML model (scikit-learn Random Forest)
- Building a FastAPI server for model inference
- Single and batch prediction endpoints
- Error handling and validation
- Testing the API
- Containerization with Docker

## Features

✨ **Model**: Random Forest Classifier for Iris flower classification
📊 **Single Prediction**: Classify a single iris flower
🔄 **Batch Prediction**: Classify multiple iris flowers at once
🔍 **Model Info**: Get information about the model and features
❤️ **Health Check**: Monitor API health
📝 **Comprehensive Documentation**: OpenAPI/Swagger docs included

## Project Structure

```
demo6_ml_inference/
├── train_model.py          # Train and save the model
├── app.py                  # FastAPI application
├── test_app.py             # Test suite
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## Installation

### 1. Create Virtual Environment (Optional but recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

```bash
python train_model.py
```

Output:

```
Model Accuracy: 1.0000
Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00       10
   virginica       1.00      1.00      1.00       10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Model saved to models/iris_model.pkl
```

This creates the `models/` directory with:

- `iris_model.pkl` - The trained model
- `feature_names.pkl` - Feature names
- `target_names.pkl` - Target class names

### Step 2: Start the API Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Output:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 3: Access the API

**Swagger Documentation**: http://localhost:8000/docs

**ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint

```
GET /
```

Returns API information and available endpoints.

### 2. Health Check

```
GET /health
```

Returns API health status.

**Response**:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Model Information

```
GET /model-info
```

Returns model details and features.

**Response**:

```json
{
  "model_type": "RandomForestClassifier",
  "classes": ["setosa", "versicolor", "virginica"],
  "features": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "n_features": 4,
  "n_classes": 3
}
```

### 4. Single Prediction

```
POST /predict
```

**Request**:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response**:

```json
{
  "prediction": "setosa",
  "probability": 0.99,
  "features_used": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ]
}
```

### 5. Batch Prediction

```
POST /predict-batch
```

**Request**:

```json
[
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  {
    "sepal_length": 7.0,
    "sepal_width": 3.2,
    "petal_length": 4.7,
    "petal_width": 1.4
  }
]
```

**Response**:

```json
{
  "predictions": ["setosa", "versicolor"],
  "probabilities": [0.99, 0.95]
}
```

## Testing

Run the complete test suite:

```bash
python test_app.py
```

This will:

- Test all endpoints
- Verify response formats
- Test error handling
- Run single and batch predictions

**Output**:

```
==================================================
ML INFERENCE API TEST SUITE
==================================================

=== Testing Root Endpoint ===
Status Code: 200
✓ Root endpoint passed!

=== Testing Health Check ===
Status Code: 200
✓ Health check passed!

=== Testing Model Info ===
Status Code: 200
✓ Model info endpoint passed!

=== Testing Single Predictions ===
Test Sample 1:
✓ Prediction: setosa (confidence: 0.99)

Test Sample 2:
✓ Prediction: versicolor (confidence: 0.95)

Test Sample 3:
✓ Prediction: virginica (confidence: 1.00)

=== Testing Batch Predictions ===
✓ Batch predictions passed!

=== Testing Error Handling ===
✓ Correctly rejected empty batch
✓ Correctly rejected invalid input

==================================================
✓ ALL TESTS PASSED!
==================================================
```

## Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '[
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}
  ]'
```

## Using Python Requests

```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
})
print(response.json())

# Batch prediction
response = requests.post('http://localhost:8000/predict-batch', json=[
    {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
    {'sepal_length': 7.0, 'sepal_width': 3.2, 'petal_length': 4.7, 'petal_width': 1.4}
])
print(response.json())
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t ml-inference-api .
```

### Run Container

```bash
docker run -p 8000:8000 ml-inference-api
```

The API will be available at `http://localhost:8000`

## Key Concepts

### 1. Model Persistence

The model is trained once and saved as a pickle file for fast loading in the API.

### 2. Pydantic Models

Used for request validation and automatic API documentation:

- `IrisData`: Single prediction input
- `PredictionResponse`: Single prediction output
- `BatchPredictionResponse`: Batch prediction output

### 3. Error Handling

- Input validation (missing/invalid fields)
- HTTP error responses with meaningful messages
- Empty batch rejection

### 4. Performance

- Model loaded once at startup
- Predictions are fast (~1-5ms for single prediction)
- Batch processing for multiple predictions

### 5. API Documentation

Automatic interactive docs generated from code:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI schema: `/openapi.json`

## Production Considerations

1. **Model Versioning**: Track model versions separately
2. **Caching**: Add caching for frequently used predictions
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Authentication**: Add API key or JWT authentication
5. **Logging**: Add comprehensive logging for monitoring
6. **Monitoring**: Set up metrics collection (Prometheus, etc.)
7. **Database**: Store predictions for audit/analysis
8. **Load Balancing**: Deploy multiple instances with load balancing

## Next Steps

- Try modifying the model (add more features, use different algorithm)
- Add authentication to the API
- Implement prediction caching
- Add database storage for predictions
- Deploy to cloud platform (AWS, GCP, Azure)
- Add monitoring and logging
- Create API client library

## Troubleshooting

**Error: "Model not found! Run train_model.py first"**

- Solution: Run `python train_model.py` before starting the API

**Error: "Cannot connect to API"**

- Solution: Ensure API is running with `python app.py`

**Port 8000 already in use**

- Solution: Use different port with `uvicorn app:app --port 8001`

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## License

This tutorial is provided as-is for educational purposes.
