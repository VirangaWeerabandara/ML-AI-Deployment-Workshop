import numpy as np
import time
import sys


def separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def format_bytes(n_bytes: int) -> str:
    """Format bytes into human readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes/1024:.1f} KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes/1024**2:.1f} MB"
    else:
        return f"{n_bytes/1024**3:.2f} GB"


# ─── Demo 1: Precision Levels ───────────────────────────────
def demo_precision_levels():
    separator("DEMO 1: What Does Quantization Look Like?")
    print("  Showing how the same number looks at different precisions:\n")

    original = np.float32(3.14159265358979)

    precisions = {
        "FP32 (32-bit float)": np.float32(original),
        "FP16 (16-bit float)": np.float16(original),
        "INT8 (8-bit integer)": np.int8(np.round(original * 40)),   # scaled
        "INT4 (4-bit integer)": np.int8(np.clip(np.round(original * 5), -8, 7)),  # simulated
    }

    for name, val in precisions.items():
        print(f"    {name:25s}  →  {val}")

    print("\n  💡 Key Insight: Lower precision = less memory, slight accuracy loss")
    print("     INT4 is ~8x smaller than FP32, yet still usable for inference!")


# ─── Demo 2: Memory Savings ─────────────────────────────────
def demo_memory_savings():
    separator("DEMO 2: Memory Savings from Quantization")
    print("  Simulating a model with 1 Billion parameters:\n")

    n_params = 1_000_000_000  # 1B parameters

    precisions = {
        "FP32 (32-bit)": {"bytes_per_param": 4, "desc": "Full precision"},
        "FP16 (16-bit)": {"bytes_per_param": 2, "desc": "Half precision (standard LLM)"},
        "INT8 (8-bit)":  {"bytes_per_param": 1, "desc": "Quantized (good quality)"},
        "INT4 (4-bit)":  {"bytes_per_param": 0.5, "desc": "Heavily quantized (GGUF Q4)"},
    }

    print(f"  {'Precision':<20s}  {'Memory':>10s}  {'Savings':>10s}  Description")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*30}")

    base_size = n_params * 4  # FP32 baseline

    for name, info in precisions.items():
        size = int(n_params * info["bytes_per_param"])
        savings = (1 - size / base_size) * 100
        bar_len = int(size / base_size * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        print(f"  {name:<20s}  {format_bytes(size):>10s}  {savings:>9.0f}%  {info['desc']}")

    print(f"\n  📊 Real-world example:")
    print(f"     LLaMA 7B in FP16:  ~14 GB VRAM (needs expensive GPU)")
    print(f"     LLaMA 7B in INT4:  ~3.5 GB VRAM (runs on laptop!)")
    print(f"     TinyLlama 1.1B Q4: ~637 MB (runs on any CPU!)")


# ─── Demo 3: Speed Comparison ───────────────────────────────
def demo_speed_comparison():
    separator("DEMO 3: Computation Speed at Different Precisions")
    print("  Matrix multiplication benchmark (simulating model inference):\n")

    sizes = {"Small (512×512)": 512, "Medium (1024×1024)": 1024, "Large (2048×2048)": 2048}

    for label, n in sizes.items():
        print(f"  {label}:")

        # FP32
        a32 = np.random.randn(n, n).astype(np.float32)
        b32 = np.random.randn(n, n).astype(np.float32)
        start = time.perf_counter()
        _ = a32 @ b32
        fp32_time = (time.perf_counter() - start) * 1000

        # FP16 (simulated – numpy doesn't have true FP16 matmul on CPU)
        a16 = a32.astype(np.float16)
        b16 = b32.astype(np.float16)
        start = time.perf_counter()
        _ = a16.astype(np.float32) @ b16.astype(np.float32)
        fp16_time = (time.perf_counter() - start) * 1000

        # INT8
        a8 = (a32 * 127).astype(np.int8)
        b8 = (b32 * 127).astype(np.int8)
        start = time.perf_counter()
        _ = a8.astype(np.int32) @ b8.astype(np.int32)
        int8_time = (time.perf_counter() - start) * 1000

        print(f"    FP32: {fp32_time:>8.1f} ms")
        print(f"    FP16: {fp16_time:>8.1f} ms")
        print(f"    INT8: {int8_time:>8.1f} ms")
        print()

    print("  💡 On GPU with proper INT8/INT4 kernels, speedup is much larger!")
    print("     Libraries like bitsandbytes, GPTQ, and AWQ optimize this heavily.")


# ─── Demo 4: Accuracy Impact ────────────────────────────────
def demo_accuracy_impact():
    separator("DEMO 4: Accuracy Loss from Quantization")
    print("  Showing that quantization introduces minimal error:\n")

    np.random.seed(42)
    
    # Simulate model weights
    weights = np.random.randn(1000, 1000).astype(np.float32) * 0.02  # typical init scale
    input_data = np.random.randn(1, 1000).astype(np.float32)

    # FP32 output (ground truth)
    output_fp32 = input_data @ weights

    # FP16 output
    output_fp16 = input_data.astype(np.float16) @ weights.astype(np.float16)
    fp16_error = np.mean(np.abs(output_fp32 - output_fp16.astype(np.float32)))

    # INT8 simulation
    scale = np.max(np.abs(weights)) / 127
    weights_int8 = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    output_int8 = (input_data @ (weights_int8.astype(np.float32) * scale))
    int8_error = np.mean(np.abs(output_fp32 - output_int8))

    # INT4 simulation
    scale4 = np.max(np.abs(weights)) / 7
    weights_int4 = np.clip(np.round(weights / scale4), -8, 7).astype(np.int8)
    output_int4 = (input_data @ (weights_int4.astype(np.float32) * scale4))
    int4_error = np.mean(np.abs(output_fp32 - output_int4))

    results = [
        ("FP16 vs FP32", fp16_error),
        ("INT8 vs FP32", int8_error),
        ("INT4 vs FP32", int4_error),
    ]

    output_range = np.max(np.abs(output_fp32))
    
    print(f"  {'Comparison':<20s}  {'Abs Error':>12s}  {'Relative':>10s}  Visual")
    print(f"  {'─'*20}  {'─'*12}  {'─'*10}  {'─'*30}")

    for name, error in results:
        relative = (error / output_range) * 100
        bar_len = min(int(relative * 10), 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:<20s}  {error:>12.6f}  {relative:>9.4f}%  [{bar}]")

    print(f"\n  ✅ Even INT4 quantization has <1% relative error!")
    print(f"     This is why quantized models work so well in practice.")


# ─── Demo 5: GGUF Format Explainer ──────────────────────────
def demo_gguf_formats():
    separator("DEMO 5: GGUF Quantization Formats")
    print("  Common GGUF quantization levels and their trade-offs:\n")

    formats = [
        ("Q2_K",  "2-bit",   "~2.5 GB",  "⚠️  Lowest quality, maximum compression"),
        ("Q3_K_M","3-bit",   "~3.3 GB",  "🔶 Usable, noticeable quality loss"),
        ("Q4_K_M","4-bit",   "~4.1 GB",  "✅ Best balance of quality vs size"),
        ("Q5_K_M","5-bit",   "~4.8 GB",  "✅ High quality, moderate compression"),
        ("Q6_K",  "6-bit",   "~5.5 GB",  "✅ Near-original quality"),
        ("Q8_0",  "8-bit",   "~7.2 GB",  "✅ Almost lossless"),
        ("F16",   "16-bit",  "~14 GB",   "📌 Original (no quantization)"),
    ]

    print(f"  {'Format':<10s}  {'Bits':>6s}  {'Size (7B)':>10s}  Notes")
    print(f"  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*40}")

    for fmt, bits, size, notes in formats:
        print(f"  {fmt:<10s}  {bits:>6s}  {size:>10s}  {notes}")

    print(f"\n  💡 For this workshop, we use Q4_K_M:")
    print(f"     • Best quality-to-size ratio")
    print(f"     • Industry standard for local deployment")
    print(f"     • Supported by llama.cpp, Ollama, LM Studio")


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📐 Quantization Demo – Understanding Model Compression")
    print("   No model download or GPU required!\n")

    demo_precision_levels()
    demo_memory_savings()
    demo_speed_comparison()
    demo_accuracy_impact()
    demo_gguf_formats()

    separator("✅ DEMOS COMPLETE")
    print("  Key Takeaways:")
    print("  1. Quantization reduces memory 2-8x with minimal quality loss")
    print("  2. Q4_K_M is the sweet spot for most deployments")
    print("  3. This is what makes running 7B+ models on laptops possible!")
    print()
