import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_model_info():
    separator("TEST 1: Model Information")
    resp = requests.get(f"{BASE_URL}/model/info")
    data = resp.json()
    print(f"  Model File:    {data['model_file']}")
    print(f"  Loaded:        {'✅ Yes' if data['model_loaded'] else '❌ No (mock mode)'}")
    print(f"  Quantization:  {data['quantization']}")
    print(f"  Context:       {data['context_window']} tokens")
    print(f"  Description:   {data['description'][:80]}...")


def test_generation():
    separator("TEST 2: Text Generation")

    prompts = [
        ("Explain what an API is in simple terms.", 80),
        ("What is Docker and why do we use it?", 80),
        ("Write a haiku about machine learning.", 50),
    ]

    for prompt, max_tokens in prompts:
        print(f"\n  📝 Prompt: \"{prompt}\"")
        print(f"     Max tokens: {max_tokens}")

        resp = requests.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7},
        )
        data = resp.json()

        print(f"     🤖 Response: \"{data['response'][:120]}\"")
        print(f"     📊 Tokens: {data['tokens_generated']} | "
              f"Latency: {data['latency_ms']:.0f}ms | "
              f"Speed: {data['tokens_per_second']:.1f} tok/s")


def test_temperature_comparison():
    separator("TEST 3: Temperature Comparison")

    resp = requests.post(
        f"{BASE_URL}/compare-temperatures",
        params={"prompt": "What makes a good software engineer?", "max_tokens": 60},
    )
    data = resp.json()

    print(f"\n  Prompt: \"{data['prompt']}\"\n")

    for comp in data["comparisons"]:
        print(f"  🌡️  Temperature {comp['temperature']} ({comp['label']})")
        print(f"     Response: \"{comp['response'][:100]}\"")
        print(f"     Latency: {comp['latency_ms']:.0f}ms\n")


def test_throughput():
    separator("TEST 4: Throughput Measurement")

    prompt = "What is machine learning?"
    num_requests = 5

    print(f"  Sending {num_requests} sequential requests...\n")

    latencies = []
    for i in range(num_requests):
        start = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": 30, "temperature": 0.5},
        )
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        data = resp.json()
        print(f"    Request {i+1}: {latency:.0f}ms | {data['tokens_per_second']:.1f} tok/s")

    avg = sum(latencies) / len(latencies)
    print(f"\n  📊 Average latency: {avg:.0f}ms")
    print(f"  📊 Throughput: {1000 / avg:.2f} requests/second")
    print(f"\n  💡 With vLLM's PagedAttention, throughput can be 10-20x higher")
    print(f"     because it batches multiple requests efficiently on GPU!")


if __name__ == "__main__":
    print("\n🦙 Local LLM API – Test Suite")
    print("   Make sure the server is running: uvicorn llm_server:app --reload\n")

    try:
        test_model_info()
        test_generation()
        test_temperature_comparison()
        test_throughput()

        separator("✅ ALL TESTS COMPLETE")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to the server!")
        print("   Start it first with: uvicorn llm_server:app --reload")
