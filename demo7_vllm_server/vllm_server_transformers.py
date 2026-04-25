"""
TinyLlama Inference Server - Windows-Compatible Version
Uses HuggingFace Transformers (no vLLM C extensions needed)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="⚡ TinyLlama Inference Server",
    description=(
        "High-performance LLM inference using Transformers + TinyLlama.\n\n"
        "**Features:**\n"
        "- TinyLlama: 1.1B parameter lightweight model\n"
        "- CPU & GPU support (auto-detected)\n"
        "- Batch processing capability\n"
        "- Windows-compatible (no CUDA build issues)\n"
    ),
    version="2.0.0",
)

# ─── CORS Configuration ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ──────────────────────────────────────────
print("⏳ Loading TinyLlama...")
start = time.perf_counter()

try:
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device.upper()}")

    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1b-chat-v1.0"
    print(f"📥 Loading from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    # Load model - simplified approach
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Move to device after loading
    model = model.to(device)
    if device == "cuda":
        model = model.half()  # Convert to float16 on GPU

    model.eval()  # Evaluation mode
    print(
        f"✅ Model loaded in {time.perf_counter() - start:.1f}s on {device.upper()}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise


# ─── Request/Response Schemas ───────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1,
                        description="Text prompt for generation")
    max_tokens: int = Field(
        100, ge=1, le=500, description="Max tokens to generate")
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0,
                         description="Nucleus sampling parameter")


class GenerateResponse(BaseModel):
    prompt: str
    response: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float


class ComparisonResult(BaseModel):
    temperature: float
    label: str
    response: str
    latency_ms: float


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Welcome endpoint."""
    return {
        "service": "TinyLlama Inference Server",
        "status": "ready",
        "model": "TinyLlama/TinyLlama-1.1b-chat-v1.0",
        "device": device.upper(),
        "version": "2.0.0 (Transformers)",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device.upper(),
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """🤖 Generate text using TinyLlama."""
    start = time.perf_counter()

    try:
        # Tokenize
        inputs = tokenizer.encode(
            request.prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = outputs.shape[1] - inputs.shape[1]

        latency_ms = (time.perf_counter() - start) * 1000
        tokens_per_second = (num_tokens / latency_ms) * \
            1000 if latency_ms > 0 else 0

        return GenerateResponse(
            prompt=request.prompt,
            response=result_text,
            tokens_generated=num_tokens,
            latency_ms=round(latency_ms, 2),
            tokens_per_second=round(tokens_per_second, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/batch", tags=["Generation"])
async def generate_batch(requests: list[GenerateRequest]):
    """📦 Batch generation."""
    start = time.perf_counter()
    results = []

    try:
        for req in requests:
            inputs = tokenizer.encode(
                req.prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            result_text = tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            results.append({
                "prompt": req.prompt,
                "response": result_text,
                "tokens_generated": outputs.shape[1] - inputs.shape[1],
            })

        total_latency = (time.perf_counter() - start) * 1000

        return {
            "count": len(results),
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(total_latency / len(results), 2) if results else 0,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.post("/compare-temperatures", tags=["Analysis"])
async def compare_temperatures(
    prompt: str = Field(..., description="Prompt for comparison"),
    max_tokens: int = Field(50, ge=1, le=200, description="Max tokens"),
):
    """🌡️ Compare different temperature settings."""
    temperatures = [0.2, 0.7, 1.5]
    comparisons = []

    try:
        for temp in temperatures:
            temp_start = time.perf_counter()

            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            result_text = tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            latency = (time.perf_counter() - temp_start) * 1000

            label = "Focused" if temp < 0.5 else "Balanced" if temp < 1.0 else "Creative"

            comparisons.append(
                ComparisonResult(
                    temperature=temp,
                    label=label,
                    response=result_text,
                    latency_ms=round(latency, 2),
                )
            )

        return {
            "prompt": prompt,
            "comparisons": [c.dict() for c in comparisons],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Comparison failed: {str(e)}")
