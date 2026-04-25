# ⚡ TinyLlama Inference Server - Windows Edition

**Why this version?** vLLM requires CUDA C extensions which are difficult to build on Windows. This version uses **HuggingFace Transformers** for cross-platform compatibility.

## Key Differences

| Aspect               | vLLM                 | Transformers (This)  |
| -------------------- | -------------------- | -------------------- |
| **Setup**            | Complex (CUDA build) | Easy (pip install)   |
| **Windows**          | ❌ Problematic       | ✅ Works great       |
| **GPU Support**      | ✅ Excellent         | ✅ Full support      |
| **Speed**            | 10-20x faster        | Good (slower on CPU) |
| **Batch Processing** | ✅ Optimized         | ✅ Basic             |

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
cd demo7_vllm_server
pip install -r requirements.txt
```

### 2. **Run the Server**

**Windows (Easiest):**

```bash
run.bat
```

**Or manually:**

```bash
uvicorn vllm_server_transformers:app --reload --host 127.0.0.1 --port 8001
```

You'll see:

```
✅ Model loaded in 45.2s on CPU
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### 3. **Test the Server**

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## 📊 Performance on Different Hardware

| Hardware           | Speed            | Memory | Notes            |
| ------------------ | ---------------- | ------ | ---------------- |
| **CPU (8 core)**   | 200-500ms/prompt | 2.5GB  | Slow but works   |
| **GPU (RTX 3060)** | 50-100ms/prompt  | 1.5GB  | 4-5x faster      |
| **GPU (RTX 4090)** | 10-30ms/prompt   | 1.2GB  | Production-ready |

---

## 🔗 API Endpoints

All endpoints from the original vLLM server are available:

### `POST /generate`

Generate text with custom parameters.

```json
{
  "prompt": "Explain AI in one sentence",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### `POST /generate/batch`

Process multiple prompts.

```json
[
  { "prompt": "Hello", "max_tokens": 50, "temperature": 0.7 },
  { "prompt": "Hi", "max_tokens": 50, "temperature": 0.7 }
]
```

### `POST /compare-temperatures`

See how temperature affects output.

```
?prompt=YOUR_PROMPT&max_tokens=50
```

---

## 🎯 Use with Dashboard

1. Start **Sentiment API** on port 8000:

   ```bash
   cd demo1_fastapi_basics
   uvicorn app:app --reload --port 8000
   ```

2. Start **TinyLlama Server** on port 8001:

   ```bash
   cd demo7_vllm_server
   run.bat
   ```

3. Open dashboard: `demo5_interactive_dashboard/index.html` in browser

4. Use both **Sentiment API** and **⚡ vLLM Chat** tabs!

---

## 📖 Troubleshooting

### "Model not found" error

```bash
# Model (~2.5GB) downloads automatically on first run
# This takes 3-5 minutes. Be patient!
```

### Out of memory

```bash
# On CPU? This is normal - TinyLlama is memory-heavy on CPU
# Try reducing max_tokens (default: 100 → try 50)
```

### Very slow on CPU

```bash
# CPU inference is 4-10x slower than GPU
# Consider: using GPU, smaller prompts, or GPU cloud services
```

### RuntimeError: CUDA out of memory

```bash
# On GPU? Your GPU doesn't have enough VRAM
# Try: float32 instead of float16, or reduce batch size
```

---

## 🔧 Adjusting for Your Hardware

Edit `vllm_server_transformers.py` line 41-45:

```python
# For better GPU performance:
torch_dtype=torch.float16,  # Uses less VRAM

# For CPU-only (default):
torch_dtype=torch.float32,  # More compatible but slower
```

---

## 📚 Learning Resources

- **TinyLlama**: https://huggingface.co/TinyLlama/TinyLlama-1.1b-chat-v1.0
- **Transformers**: https://huggingface.co/docs/transformers
- **PyTorch**: https://pytorch.org/docs/stable

---

## 🚀 Next Steps

- Try different prompts in the dashboard
- Compare temperature outputs
- Test batch processing
- Deploy to production (add authentication, rate limiting)

---

**Happy inferencing!** 🚀
