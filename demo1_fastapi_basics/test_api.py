import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_root():
    separator("TEST 1: Root Endpoint (GET /)")
    resp = requests.get(f"{BASE_URL}/")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")


def test_health():
    separator("TEST 2: Health Check (GET /health)")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")


def test_single_prediction():
    separator("TEST 3: Single Prediction (POST /predict)")

    test_cases = [
        "I absolutely love this product! It's amazing and wonderful.",
        "This is the worst experience I've ever had. Terrible service.",
        "The weather is cloudy today.",
    ]

    for text in test_cases:
        resp = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text},
        )
        data = resp.json()
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(data["prediction"], "❓")
        print(f"\n  Input:      \"{text[:60]}...\"")
        print(f"  Prediction: {emoji} {data['prediction']} ({data['confidence']:.2%})")
        print(f"  Latency:    {data['latency_ms']:.2f} ms")


def test_batch_prediction():
    separator("TEST 4: Batch Prediction (POST /predict/batch)")

    batch = [
        {"text": "Great quality, fast delivery!"},
        {"text": "Broken on arrival, very disappointed."},
        {"text": "It's okay, nothing special."},
        {"text": "Best purchase I've made this year! Absolutely fantastic!"},
        {"text": "Horrible customer support, will never buy again."},
    ]

    resp = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    data = resp.json()

    print(f"  Processed {data['count']} items")
    print(f"  Total latency:   {data['total_latency_ms']:.2f} ms")
    print(f"  Average latency: {data['avg_latency_ms']:.2f} ms")
    print()

    for r in data["results"]:
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(r["prediction"], "❓")
        print(f"    {emoji}  {r['prediction']:>8s} ({r['confidence']:.2%})  │  \"{r['text'][:45]}\"")


def test_latency_demo():
    separator("TEST 5: Latency Comparison – Single vs Batch")

    import time

    texts = [{"text": f"Sample review number {i} with some great content"} for i in range(20)]

    # Single requests (one by one)
    start = time.perf_counter()
    for t in texts:
        requests.post(f"{BASE_URL}/predict", json=t)
    single_time = (time.perf_counter() - start) * 1000

    # Batch request (all at once)
    start = time.perf_counter()
    requests.post(f"{BASE_URL}/predict/batch", json=texts)
    batch_time = (time.perf_counter() - start) * 1000

    print(f"  20 single requests: {single_time:.1f} ms total")
    print(f"  1 batch request:    {batch_time:.1f} ms total")
    print(f"  Speedup:            {single_time / batch_time:.1f}x faster with batching!")
    print()
    print("  💡 Key Takeaway: Batching reduces HTTP overhead significantly.")
    print("     In real ML, batch processing also lets the GPU process")
    print("     multiple inputs in parallel (vectorized computation).")


if __name__ == "__main__":
    print("\n🚀 ML Sentiment API – Test Suite")
    print("   Make sure the server is running: uvicorn app:app --reload\n")

    try:
        test_root()
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_latency_demo()

        separator("✅ ALL TESTS PASSED")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to the server!")
        print("   Start it first with: uvicorn app:app --reload")
