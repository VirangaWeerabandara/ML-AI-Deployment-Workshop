import urllib.request
import os
import sys
import time


MODEL_URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
    "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def download_with_progress(url: str, filename: str):
    """Download a file with a progress bar."""
    
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ Model already exists: {filename} ({size_mb:.0f} MB)")
        return

    print(f"📥 Downloading: {filename}")
    print(f"   From: {url}")
    print(f"   This may take a few minutes (~637 MB)...\n")

    start = time.perf_counter()

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.perf_counter() - start
            speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
            sys.stdout.write(
                f"\r   [{bar}] {percent:5.1f}%  "
                f"{downloaded_mb:.0f}/{total_mb:.0f} MB  "
                f"({speed:.1f} MB/s)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        elapsed = time.perf_counter() - start
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"\n\n✅ Download complete!")
        print(f"   File: {filename}")
        print(f"   Size: {size_mb:.0f} MB")
        print(f"   Time: {elapsed:.0f}s")
    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        print("\n   Alternative: Download manually from:")
        print(f"   {url}")
        print(f"   Save as: {filename}")
        if os.path.exists(filename):
            os.remove(filename)
        sys.exit(1)


if __name__ == "__main__":
    print("\n🦙 TinyLlama GGUF Model Downloader")
    print("=" * 50)
    download_with_progress(MODEL_URL, MODEL_FILE)
    print("\n🚀 Next step: uvicorn llm_server:app --reload\n")
