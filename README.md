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

### Demo 3: Local LLM Server (GGUF)
```bash
cd demo3_llm_server
pip install -r requirements.txt
python download_model.py        # Downloads ~637MB model
uvicorn llm_server:app --reload
# Open: http://127.0.0.1:8000/docs

# In another terminal:
python test_llm_api.py
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

---

