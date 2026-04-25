from transformers import pipeline
import time
import textwrap


def separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ─── Demo 1: Text Generation with GPT-2 ─────────────────────
def demo_text_generation():
    separator("DEMO 1: Text Generation (GPT-2 – 124M params)")
    print("  Loading model... (first run downloads ~550MB)")

    start = time.perf_counter()
    generator = pipeline("text-generation", model="gpt2")
    load_time = time.perf_counter() - start
    print(f"  ✅ Model loaded in {load_time:.1f}s\n")

    prompts = [
        "Artificial intelligence will change the world by",
        "The most important skill for a software engineer is",
        "Model deployment is critical because",
    ]

    for prompt in prompts:
        start = time.perf_counter()
        output = generator(
            prompt,
            max_length=60,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )
        latency = (time.perf_counter() - start) * 1000

        generated = output[0]["generated_text"]
        print(f"  📝 Prompt:    \"{prompt}\"")
        print(f"     Output:    \"{textwrap.shorten(generated, 120)}\"")
        print(f"     Latency:   {latency:.0f} ms")
        print()


# ─── Demo 2: Sentiment Analysis (Pre-trained Classifier) ────
def demo_sentiment():
    separator("DEMO 2: Sentiment Analysis (distilbert-base)")
    print("  Loading model...")

    classifier = pipeline("sentiment-analysis")

    texts = [
        "I absolutely love this workshop! Learning so much about deployment.",
        "The documentation was confusing and I couldn't get it to work.",
        "The food at the cafeteria was alright, nothing special.",
        "FastAPI is an incredible framework for building APIs!",
        "I'm disappointed with the lack of GPU support on my laptop.",
    ]

    print()
    for text in texts:
        result = classifier(text)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😠"
        bar_len = int(result["score"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {emoji} {result['label']:>8s}  [{bar}] {result['score']:.2%}")
        print(f"     \"{textwrap.shorten(text, 70)}\"")
        print()


# ─── Demo 3: Zero-Shot Classification ───────────────────────
def demo_zero_shot():
    separator("DEMO 3: Zero-Shot Classification")
    print("  This model can classify text into ANY categories")
    print("  you define — without any training!\n")

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    text = "The new MacBook Pro features an M3 chip with 18-hour battery life."
    labels = ["technology", "sports", "politics", "food", "entertainment"]

    result = classifier(text, candidate_labels=labels)

    print(f"  Text: \"{text}\"\n")
    print(f"  Category Scores:")
    for label, score in zip(result["labels"], result["scores"]):
        bar_len = int(score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"    {label:>15s}  [{bar}] {score:.2%}")


# ─── Demo 4: Summarization ──────────────────────────────────
def demo_summarization():
    separator("DEMO 4: Text Summarization")

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    article = """
    Machine learning model deployment is the process of making a trained
    machine learning model available for use in a production environment.
    This involves packaging the model, creating APIs for inference,
    setting up monitoring and logging, and ensuring the model can handle
    real-world traffic at scale. Common challenges include managing
    dependencies, handling model versioning, ensuring low latency
    responses, and dealing with data drift over time. Modern deployment
    strategies often involve containerization with Docker, orchestration
    with Kubernetes, and serving frameworks like TensorFlow Serving,
    TorchServe, or vLLM for large language models.
    """

    print(f"  Original ({len(article.split())} words):")
    print(textwrap.fill(article.strip(), width=60, initial_indent="    ", subsequent_indent="    "))

    result = summarizer(article, max_length=60, min_length=20, do_sample=False)
    summary = result[0]["summary_text"]

    print(f"\n  Summary ({len(summary.split())} words):")
    print(textwrap.fill(summary, width=60, initial_indent="    ", subsequent_indent="    "))


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤗 HuggingFace Pipeline Demos")
    print("   Showing how easy it is to use pre-trained models\n")

    # Run each demo – comment out any you want to skip
    demo_text_generation()
    demo_sentiment()
    # Uncomment below for more demos (these download larger models):
    # demo_zero_shot()
    # demo_summarization()

    separator("✅ DEMOS COMPLETE")
    print("  💡 Key Takeaways:")
    print("     • pipeline() abstracts tokenization + model loading")
    print("     • Great for prototyping, but not optimized for production")
    print("     • For production LLM serving, use vLLM or similar\n")
