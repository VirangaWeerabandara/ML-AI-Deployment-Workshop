from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import random
import time

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="🤖 ML Sentiment Classifier API",
    description="A demo API that simulates a sentiment analysis model. "
                "This showcases how to expose ML predictions via REST endpoints.",
    version="1.0.0",
)

# ─── CORS Configuration ──────────────────────────────────────
# Allow requests from the interactive dashboard (and localhost in general)
app.add_middleware(
    CORSMiddleware,
    # Allow all origins (for demo; restrict in production)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ─── Request / Response Schemas ──────────────────────────────
class TextRequest(BaseModel):
    """Input schema for the prediction endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to analyze for sentiment.",
        json_schema_extra={"examples": [
            "I absolutely love this product! It's amazing."]}
    )


class PredictionResponse(BaseModel):
    """Output schema returned by the prediction endpoint."""
    text: str
    prediction: str
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# ─── Mock Model ──────────────────────────────────────────────
# In production, you'd load a real sklearn / PyTorch / TF model here.
# This mock simulates the prediction pipeline.

POSITIVE_WORDS = {"love", "great", "amazing", "excellent",
                  "wonderful", "best", "fantastic", "happy", "good", "awesome"}
NEGATIVE_WORDS = {"hate", "terrible", "awful", "worst",
                  "bad", "horrible", "disgusting", "poor", "ugly", "boring"}


def mock_sentiment_model(text: str) -> dict:
    """
    Simulates a sentiment analysis model.
    In a real deployment, this would be:
        model = joblib.load("model.pkl")
        prediction = model.predict(processed_text)
    """
    words = set(text.lower().split())
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    if pos_count > neg_count:
        label = "positive"
        confidence = min(0.99, 0.7 + pos_count * 0.05 + random.uniform(0, 0.1))
    elif neg_count > pos_count:
        label = "negative"
        confidence = min(0.99, 0.7 + neg_count * 0.05 + random.uniform(0, 0.1))
    else:
        label = "neutral"
        confidence = 0.5 + random.uniform(0, 0.15)

    return {"prediction": label, "confidence": round(confidence, 4)}


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint – verifies the API is running."""
    return {
        "message": "Welcome to the ML Sentiment API!",
        "docs": "Visit /docs for interactive documentation.",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    In production, load balancers and orchestrators (Kubernetes)
    ping this to know if the service is alive.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: TextRequest):
    """
    🔮 Predict the sentiment of the input text.

    This is the core endpoint. In a real app you would:
    1. Preprocess the text (tokenization, cleaning)
    2. Pass it through the loaded model
    3. Post-process and return the result

    **Try it out** in the Swagger UI above!
    """
    start = time.perf_counter()

    result = mock_sentiment_model(request.text)

    latency = (time.perf_counter() - start) * 1000  # ms

    return PredictionResponse(
        text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(requests: list[TextRequest]):
    """
    📦 Batch prediction endpoint.

    Demonstrates the **batch inference** pattern:
    - Accept multiple texts at once
    - Process them together (in a real model, you'd vectorize the batch)
    - Return all results

    This is more efficient than calling /predict N times because
    it amortizes the overhead of each HTTP request.
    """
    start = time.perf_counter()
    results = []

    for req in requests:
        result = mock_sentiment_model(req.text)
        results.append({
            "text": req.text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        })

    total_latency = (time.perf_counter() - start) * 1000

    return {
        "count": len(results),
        "total_latency_ms": round(total_latency, 2),
        "avg_latency_ms": round(total_latency / len(results), 2) if results else 0,
        "results": results,
    }
