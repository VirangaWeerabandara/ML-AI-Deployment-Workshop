from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import time
import os

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="⚡ vLLM TinyLlama Server",
    description=(
        "High-performance LLM inference using vLLM + TinyLlama.\n\n"
        "**Key Concepts:**\n"
        "- vLLM uses PagedAttention for 10-20x throughput improvement\n"
        "- TinyLlama is a lightweight, quantized model perfect for edge deployment\n"
        "- Batch processing for efficient resource utilization\n"
    ),
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

# ─── Model Loading ──────────────────────────────────────────
# Load TinyLlama using vLLM
print("⏳ Loading TinyLlama with vLLM...")
start = time.perf_counter()

try:
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1b-chat-v1.0",
        tensor_parallel_size=1,  # Single GPU/CPU
        gpu_memory_utilization=0.7,  # Adjust based on your hardware
        dtype="auto",
    )
    print(f"✅ Model loaded in {time.perf_counter() - start:.1f}s")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Make sure you have enough GPU/CPU memory and CUDA installed (if using GPU)")
    raise


# ─── Request/Response Schemas ───────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1,
                        description="Text prompt for generation")
    max_tokens: int = Field(
        100, ge=1, le=1000, description="Max tokens to generate")
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
    """Welcome endpoint – confirms the vLLM server is running."""
    return {
        "service": "vLLM TinyLlama Server",
        "status": "ready",
        "model": "TinyLlama/TinyLlama-1.1b-chat-v1.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    🤖 Generate text using TinyLlama via vLLM.

    vLLM provides:
    - 10-20x faster inference than naive implementations
    - Efficient KV cache management (PagedAttention)
    - Support for batch processing
    """
    start = time.perf_counter()

    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Generate using vLLM
        outputs = llm.generate(
            [request.prompt],
            sampling_params=sampling_params,
        )

        # Extract result
        result_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)

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
    """
    📦 Batch generation – vLLM's strength!

    Process multiple prompts efficiently using vLLM's batching.
    This demonstrates the throughput advantage of vLLM.
    """
    start = time.perf_counter()

    try:
        results = []

        for req in requests:
            sampling_params = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )

            outputs = llm.generate(
                [req.prompt], sampling_params=sampling_params)
            result_text = outputs[0].outputs[0].text

            results.append({
                "prompt": req.prompt,
                "response": result_text,
                "tokens_generated": len(outputs[0].outputs[0].token_ids),
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
    prompt: str = Field(..., description="Prompt to generate for"),
    max_tokens: int = Field(
        50, ge=1, le=200, description="Max tokens per generation"),
):
    """
    🌡️ Compare outputs at different temperature settings.

    Demonstrates how temperature affects model creativity:
    - Low (0.2): Deterministic, focused
    - Medium (0.7): Balanced
    - High (1.5): Creative, varied
    """
    start = time.perf_counter()
    temperatures = [0.2, 0.7, 1.5]
    comparisons = []

    try:
        for temp in temperatures:
            sampling_params = SamplingParams(
                temperature=temp,
                top_p=0.9,
                max_tokens=max_tokens,
            )

            outputs = llm.generate([prompt], sampling_params=sampling_params)
            result_text = outputs[0].outputs[0].text
            latency = (time.perf_counter() - start) * 1000

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
