from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_cpp import Llama
import time
import os

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="🦙 Local LLM API",
    description=(
        "Serves a quantized LLM (GGUF format) locally on CPU.\n\n"
        "**Key Concepts Demonstrated:**\n"
        "- Quantization (INT4) reduces 2.2GB model → ~600MB\n"
        "- CPU inference without GPU/CUDA\n"
        "- Streaming vs non-streaming responses\n"
    ),
    version="1.0.0",
)

# Allow CORS for the interactive dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ──────────────────────────────────────────
MODEL_PATH = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print("=" * 60)
    print("  ❌ Model file not found!")
    print(f"  Expected: {MODEL_PATH}")
    print("  Run: python download_model.py")
    print("=" * 60)
    # Use a fallback mock mode so students can still see the API structure
    llm = None
else:
    print("⏳ Loading quantized LLM...")
    start = time.perf_counter()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,        # Context window size
        n_threads=4,       # CPU threads (adjust to your machine)
        verbose=False,
    )
    print(f"✅ Model loaded in {time.perf_counter() - start:.1f}s")


# ─── Schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="The user's question or prompt.",
        json_schema_extra={"examples": ["Explain what an API is in simple terms."]}
    )
    max_tokens: int = Field(
        default=100,
        ge=10,
        le=512,
        description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness. Lower = more focused, higher = more creative."
    )


class ChatResponse(BaseModel):
    prompt: str
    response: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float


class ModelInfo(BaseModel):
    model_file: str
    model_loaded: bool
    quantization: str
    context_window: int
    description: str


# ─── Mock Fallback ───────────────────────────────────────────
def mock_generate(prompt: str, max_tokens: int) -> dict:
    """Fallback when model file isn't downloaded yet."""
    return {
        "choices": [{
            "text": (
                f" [MOCK RESPONSE] This is a simulated response because the "
                f"GGUF model file hasn't been downloaded yet. In a real deployment, "
                f"the LLM would generate a response to: '{prompt[:50]}...'. "
                f"Run 'python download_model.py' to get the real model."
            )
        }],
        "usage": {"completion_tokens": 42},
    }


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Local LLM API",
        "model_loaded": llm is not None,
        "docs": "/docs",
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_file=MODEL_PATH,
        model_loaded=llm is not None,
        quantization="Q4_K_M (4-bit)",
        context_window=2048,
        description=(
            "TinyLlama 1.1B Chat – A compact language model quantized to "
            "4-bit precision (GGUF format). Original: ~2.2GB → Quantized: ~637MB. "
            "Runs on CPU without GPU."
        ),
    )


@app.post("/generate", response_model=ChatResponse, tags=["Generation"])
async def generate_text(request: ChatRequest):
    """
    🤖 Generate text from the LLM.

    The prompt is formatted in the ChatML/Llama-2 chat template:
    ```
    USER: {prompt}
    ASSISTANT:
    ```

    **Parameters to experiment with:**
    - `max_tokens`: More tokens = longer response but slower
    - `temperature`: 0.1 = focused/deterministic, 1.5 = creative/random
    """
    formatted_prompt = f"USER: {request.prompt}\nASSISTANT:"

    start = time.perf_counter()

    if llm is not None:
        output = llm(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["USER:", "\n\n\n"],
            echo=False,
        )
    else:
        output = mock_generate(request.prompt, request.max_tokens)

    latency = (time.perf_counter() - start) * 1000
    response_text = output["choices"][0]["text"].strip()
    tokens = output.get("usage", {}).get("completion_tokens", len(response_text.split()))
    tps = (tokens / (latency / 1000)) if latency > 0 else 0

    return ChatResponse(
        prompt=request.prompt,
        response=response_text,
        tokens_generated=tokens,
        latency_ms=round(latency, 2),
        tokens_per_second=round(tps, 1),
    )


@app.post("/compare-temperatures", tags=["Experiments"])
async def compare_temperatures(prompt: str = "What is machine learning?", max_tokens: int = 60):
    """
    🔬 Compare model outputs at different temperatures.

    This endpoint generates the same prompt at 3 different temperatures
    to demonstrate how temperature affects output randomness.

    Great for showing students the effect of this parameter!
    """
    temperatures = [0.1, 0.7, 1.5]
    results = []

    for temp in temperatures:
        formatted = f"USER: {prompt}\nASSISTANT:"
        start = time.perf_counter()

        if llm is not None:
            output = llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temp,
                stop=["USER:", "\n\n\n"],
                echo=False,
            )
            text = output["choices"][0]["text"].strip()
        else:
            text = f"[MOCK @ temp={temp}] Download model to see real differences."

        latency = (time.perf_counter() - start) * 1000

        results.append({
            "temperature": temp,
            "label": {0.1: "Focused/Deterministic", 0.7: "Balanced", 1.5: "Creative/Random"}[temp],
            "response": text,
            "latency_ms": round(latency, 2),
        })

    return {"prompt": prompt, "comparisons": results}
