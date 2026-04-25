# 🚀 Workshop Session 2: ML + LLM Model Deployment

**Time:** 1:00 PM – 4:00 PM  
**Focus:** Bridging the gap between training a model and putting it into users' hands.

---

## 📁 Project Structure

```
codes/
├── demo1_fastapi_basics/        # 1:45-2:30 PM – Containerization & APIs
│   ├── app.py                   # FastAPI sentiment API (mock model)
│   ├── test_api.py              # Test client with latency demos
│   ├── requirements.txt
│   └── Dockerfile               # Docker containerization example
│
├── demo2_huggingface_pipelines/ # 2:30-3:15 PM – LLM Deployment (Easy Way)
│   ├── hf_pipeline_demo.py      # HuggingFace pipeline demos (standalone)
│   ├── hf_api_server.py         # HuggingFace model served via FastAPI
│   ├── test_hf_api.py           # Test client
│   └── requirements.txt
│
├── demo3_llm_server/            # 3:15-4:00 PM – Hands-on Deployment
│   ├── llm_server.py            # Quantized LLM (GGUF) via FastAPI
│   ├── download_model.py        # Downloads TinyLlama Q4 model (~637MB)
│   ├── test_llm_api.py          # Test client with throughput measurement
│   └── requirements.txt
│
├── demo4_quantization/          # 2:30-3:15 PM – Quantization Concepts
│   ├── quantization_demo.py     # NumPy-based quantization simulation
│   └── requirements.txt         # (no GPU needed!)
│
├── demo5_interactive_dashboard/ # Visual API tester (open in browser)
│   ├── index.html               # Interactive dashboard
│   ├── style.css
│   └── script.js
│
├── demo6_ml_inference/          # ML Model Inference (Sklearn)
│   ├── app.py                   # FastAPI iris classification
│   ├── train_model.py           # Train sklearn model
│   ├── example_client.py        # Example client
│   ├── models/                  # Pre-trained models
│   └── requirements.txt
│
├── demo7_vllm_server/           # ⚡ HIGH-PERFORMANCE LLM INFERENCE
│   ├── vllm_server.py           # vLLM + TinyLlama (10-20x faster)
│   ├── requirements.txt
│   ├── run.bat                  # Quick start script (Windows)
│   └── README.md                # Complete vLLM documentation
│
└── README.md                    # This file
```

---

## ⚡ Quick Start Guide

### Demo 1: FastAPI Basics (Mock Model)

```bash
cd demo1_fastapi_basics
pip install -r requirements.txt
uvicorn app:app --reload
# Open: http://127.0.0.1:8000/docs

# In another terminal:
python test_api.py
```

### Demo 2: HuggingFace Pipelines

```bash
cd demo2_huggingface_pipelines

# Standalone demo (downloads models automatically):
pip install -r requirements.txt
python hf_pipeline_demo.py

# As an API server:
uvicorn hf_api_server:app --reload
python test_hf_api.py
```

### Demo 4: Quantization Concepts

```bash
cd demo4_quantization
pip install -r requirements.txt
python quantization_demo.py     # No GPU needed!
```

### Demo 5: Interactive Dashboard

```
Open demo5_interactive_dashboard/index.html in a browser.
Start any API server (Demo 1 or Demo 3) first.
The dashboard connects to http://127.0.0.1:8000 by default.
```

### Demo 6: ML Model Inference (Sklearn)

```bash
cd demo6_ml_inference
pip install -r requirements.txt

# Train the model first:
python train_model.py

# Run the API:
uvicorn app:app --reload
# Open: http://127.0.0.1:8000/docs

# In another terminal:
python test_app.py
```

### Demo 7: vLLM High-Performance Inference ⚡

```bash
cd demo7_vllm_server
pip install -r requirements.txt

# Quick start (Windows):
run.bat

# Or manual start:
uvicorn vllm_server:app --reload --host 127.0.0.1 --port 8001

# In another terminal, test:
curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_tokens": 100, "temperature": 0.7}'
```

**Use with Dashboard:**

- Run Demo 1 on **port 8000**: `uvicorn app:app --reload --port 8000`
- Run Demo 7 on **port 8001**: `uvicorn vllm_server:app --reload --port 8001`
- Open dashboard: `index.html` in browser
- Access both APIs: Sentiment on 8000, vLLM Chat on 8001

---

## 🎯 Key Concepts Covered

| Concept                     | Demo          | Key Takeaway                                       |
| --------------------------- | ------------- | -------------------------------------------------- |
| **API Design**              | 1, 2, 3, 6, 7 | REST endpoints, request/response schemas           |
| **Model Serving**           | 1, 2, 3, 6, 7 | FastAPI + Uvicorn                                  |
| **Containerization**        | 1             | Docker for reproducible deployments                |
| **HuggingFace Integration** | 2             | Easy model loading and pipeline setup              |
| **Quantization**            | 4             | Reduce model size 4-10x with minimal accuracy loss |
| **LLM Inference**           | 3, 7          | Running local LLMs efficiently                     |
| **vLLM & PagedAttention**   | 7             | 10-20x faster inference, batch processing          |
| **Latency vs Throughput**   | 1, 5, 7       | Real-time vs batch trade-offs                      |
| **Edge Deployment**         | 3, 7          | Running models on consumer hardware                |

---

## 📊 Performance Comparison

| Method                   | Model        | Speed      | Memory | Use Case                   |
| ------------------------ | ------------ | ---------- | ------ | -------------------------- |
| **FastAPI Mock**         | N/A          | <1ms       | ~100MB | Testing, demos             |
| **HuggingFace Pipeline** | BERT-base    | 50-100ms   | ~500MB | Quick sentiment analysis   |
| **Quantized LLM**        | TinyLlama Q4 | 500-1000ms | ~800MB | Local edge deployment      |
| **vLLM**                 | TinyLlama    | 200-500ms  | ~1GB   | Production-grade, batching |

---

---
