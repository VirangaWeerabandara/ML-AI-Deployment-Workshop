# ⚡ vLLM TinyLlama Server

High-performance LLM inference using **vLLM** and **TinyLlama**.

## 🎯 What is vLLM?

**vLLM** is a state-of-the-art LLM inference engine that provides:

- **10-20x faster** inference than naive implementations
- **PagedAttention**: Efficient KV cache management (like virtual memory for GPUs)
- **Batch processing**: Maximize GPU throughput
- **Easy integration**: Drop-in replacement for HuggingFace

**TinyLlama**: Lightweight, 1.1B parameter model perfect for edge deployment.

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (CPU) or 2GB+ VRAM (GPU)
- CUDA 11.8+ (for GPU acceleration, optional)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd demo7_vllm_server
pip install -r requirements.txt
```

### 2. Run the Server

**CPU Mode (slower but works everywhere):**

```bash
uvicorn vllm_server:app --reload --host 127.0.0.1 --port 8001
```

**GPU Mode (if CUDA available):**

```bash
# First run will download the model (~2GB)
uvicorn vllm_server:app --reload --host 127.0.0.1 --port 8001
```

You'll see:

```
✅ Model loaded in 45.2s
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### 3. Test the Server

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Use in Dashboard

1. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) (or wherever your dashboard is)
2. Go to **⚡ vLLM Chat** tab
3. Ensure URL is set to `http://127.0.0.1:8001`
4. Type your prompt and click **🤖 Generate**

## 📚 API Endpoints

### `POST /generate`

Generate text with custom parameters.

**Request:**

```json
{
  "prompt": "Tell me a joke",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**

```json
{
  "prompt": "Tell me a joke",
  "response": "Why did the AI go to school? To improve its learning model!",
  "tokens_generated": 15,
  "latency_ms": 234.5,
  "tokens_per_second": 64.0
}
```

### `POST /generate/batch`

Batch generation for multiple prompts.

**Request:**

```json
[
  { "prompt": "Hello", "max_tokens": 50, "temperature": 0.7 },
  { "prompt": "Hi", "max_tokens": 50, "temperature": 0.7 }
]
```

### `POST /compare-temperatures?prompt=YOUR_PROMPT&max_tokens=50`

Compare outputs at different temperatures (0.2, 0.7, 1.5).

**Response:**

```json
{
  "prompt": "Explain AI",
  "comparisons": [
    {
      "temperature": 0.2,
      "label": "Focused",
      "response": "AI is...",
      "latency_ms": 150
    },
    ...
  ]
}
```

## 🎚️ Parameter Guide

- **prompt** (str): The text to generate from
- **max_tokens** (int): Maximum tokens to generate (1-1000)
- **temperature** (float): Controls randomness
  - `0.0` = Always the same (greedy)
  - `0.7` = Balanced (default)
  - `2.0` = Very random
- **top_p** (float): Nucleus sampling (0.0-1.0)

## 📊 Performance Tips

### For Faster Inference:

1. **Use GPU** (if available):
   - Install CUDA: https://developer.nvidia.com/cuda-downloads
   - vLLM will auto-detect and use GPU

2. **Reduce max_tokens**:
   - Fewer tokens = faster generation

3. **Lower temperature**:
   - Lower temp = simpler sampling = faster

4. **Batch requests**:
   - Multiple prompts at once = better throughput

### Troubleshooting:

**"Model not found" error:**

```bash
# First time? vLLM downloads the model (2-3 minutes on good internet)
# Be patient!
```

**Out of memory:**

```bash
# Use CPU mode or reduce batch size
# Check: nvidia-smi  (if GPU available)
```

**Slow generation:**

- Running on CPU? This is normal. Consider using GPU.
- Check system resources: `top` or Task Manager

## 🔗 Connecting to Dashboard

The dashboard (demo5) expects the vLLM server at `http://127.0.0.1:8001`.

To run both:

**Terminal 1 - Sentiment API (Port 8000):**

```bash
cd demo1_fastapi_basics
uvicorn app:app --reload --port 8000
```

**Terminal 2 - vLLM Server (Port 8001):**

```bash
cd demo7_vllm_server
uvicorn vllm_server:app --reload --port 8001
```

**Terminal 3 - Dashboard (Port 5000 or open HTML):**

```bash
# Open demo5_interactive_dashboard/index.html in browser
# OR serve with Python:
cd demo5_interactive_dashboard
python -m http.server 5000
```

Then visit: **http://127.0.0.1:5000**

## 📖 Learning Resources

- **vLLM**: https://github.com/lm-sys/vllm
- **TinyLlama**: https://huggingface.co/TinyLlama/TinyLlama-1.1b-chat-v1.0
- **PagedAttention Paper**: https://arxiv.org/abs/2309.06180

## ⚙️ Advanced Configuration

Edit `vllm_server.py`:

```python
llm = LLM(
    model="TinyLlama/TinyLlama-1.1b-chat-v1.0",
    tensor_parallel_size=1,  # Multi-GPU support
    gpu_memory_utilization=0.7,  # Memory % to use
    dtype="auto",  # or "float16", "float32"
)
```

---

**Happy inferencing!** 🚀
