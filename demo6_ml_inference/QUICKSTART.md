# Quick Start Guide

Get up and running with the ML Inference API in 5 minutes!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python train_model.py
```

This creates a trained model in the `models/` folder.

### 3. Start API Server

```bash
python app.py
```

The API is now running at `http://localhost:8000`

### 4. Test the API

**Option A: Using the Web UI**

- Open browser: http://localhost:8000/docs
- Try out endpoints directly in Swagger UI

**Option B: Using test suite**

```bash
# In another terminal
python test_app.py
```

**Option C: Using examples**

```bash
# In another terminal
python example_client.py
```

## 📊 API Overview

| Endpoint         | Method | Purpose           |
| ---------------- | ------ | ----------------- |
| `/`              | GET    | API info          |
| `/health`        | GET    | Health check      |
| `/model-info`    | GET    | Model details     |
| `/predict`       | POST   | Single prediction |
| `/predict-batch` | POST   | Batch predictions |

## 💡 Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## 📦 What's Included

- ✅ **train_model.py** - Train and save ML model
- ✅ **app.py** - FastAPI server
- ✅ **test_app.py** - Complete test suite
- ✅ **example_client.py** - Usage examples
- ✅ **Dockerfile** - Container setup
- ✅ **README.md** - Full documentation

## 🐳 Docker

```bash
# Build
docker build -t ml-inference-api .

# Run
docker run -p 8000:8000 ml-inference-api
```

## 📚 Learn More

See **README.md** for:

- Detailed documentation
- Advanced examples
- Production considerations
- Troubleshooting

---

**Happy deploying! 🎉**
