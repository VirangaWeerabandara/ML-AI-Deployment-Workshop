import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_single():
    separator("Single Sentiment Analysis")
    texts = [
        "FastAPI makes it incredibly easy to deploy ML models!",
        "I can't believe how slow this model is on my old laptop.",
        "The conference room is on the second floor.",
    ]
    for text in texts:
        resp = requests.post(f"{BASE_URL}/analyze", json={"text": text})
        data = resp.json()
        emoji = "😊" if data["label"] == "POSITIVE" else "😠"
        print(f"\n  {emoji} {data['label']} ({data['score']:.2%}) | {data['latency_ms']:.0f}ms")
        print(f"     \"{text[:65]}\"")


def test_batch():
    separator("Batch Sentiment Analysis")
    batch = [
        {"text": "I love deploying models with Docker!"},
        {"text": "Kubernetes is too complicated for beginners."},
        {"text": "The API documentation was very clear and helpful."},
        {"text": "My server crashed after the third request."},
        {"text": "The weather is nice today."},
    ]
    resp = requests.post(f"{BASE_URL}/analyze/batch", json=batch)
    data = resp.json()
    print(f"\n  Processed {data['count']} texts in {data['total_latency_ms']:.0f}ms\n")
    for r in data["results"]:
        emoji = "😊" if r["label"] == "POSITIVE" else "😠"
        print(f"    {emoji} {r['label']:>8s} ({r['score']:.2%})  │  \"{r['text'][:50]}\"")


if __name__ == "__main__":
    print("\n🤗 HuggingFace Sentiment API – Test Suite\n")
    try:
        test_single()
        test_batch()
        separator("✅ ALL TESTS PASSED")
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect! Start server: uvicorn hf_api_server:app --reload")
