from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import time

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="🤗 HuggingFace Sentiment API",
    description="Serves a real HuggingFace sentiment analysis model via REST API.",
    version="1.0.0",
)

# ─── CORS Configuration ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model at Startup ──────────────────────────────────
# The model loads ONCE when the server starts, not on every request.
# This is a critical deployment pattern!
print("⏳ Loading HuggingFace model...")
start = time.perf_counter()
classifier = pipeline("sentiment-analysis")
print(f"✅ Model loaded in {time.perf_counter() - start:.1f}s")


# ─── Schemas ─────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class SentimentOutput(BaseModel):
    text: str
    label: str
    score: float
    latency_ms: float


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "HuggingFace Sentiment API", "status": "ready"}


@app.post("/analyze", response_model=SentimentOutput)
async def analyze_sentiment(request: TextInput):
    """Analyze the sentiment of input text using a real transformer model."""
    start = time.perf_counter()
    result = classifier(request.text)[0]
    latency = (time.perf_counter() - start) * 1000

    return SentimentOutput(
        text=request.text,
        label=result["label"],
        score=round(result["score"], 4),
        latency_ms=round(latency, 2),
    )


@app.post("/analyze/batch")
async def analyze_batch(requests: list[TextInput]):
    """Batch sentiment analysis – process multiple texts at once."""
    start = time.perf_counter()

    texts = [r.text for r in requests]
    results = classifier(texts)

    total_latency = (time.perf_counter() - start) * 1000

    return {
        "count": len(results),
        "total_latency_ms": round(total_latency, 2),
        "results": [
            {
                "text": text,
                "label": res["label"],
                "score": round(res["score"], 4),
            }
            for text, res in zip(texts, results)
        ],
    }
