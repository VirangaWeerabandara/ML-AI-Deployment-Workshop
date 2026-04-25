from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os

# Step 1: Authenticate with Hugging Face
# Set HF_TOKEN environment variable before running this script
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ Authenticated with Hugging Face Hub")
else:
    print("⚠️  HF_TOKEN not set. Use `huggingface-cli login` to authenticate.")

# Step 2: Load model and tokenizer
model_name = "distilbert-base-uncased"
print(f"📥 Loading {model_name}...")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Model and tokenizer loaded")

# Step 3: Save locally
print("💾 Saving model locally...")
model.save_pretrained("demo-model")
tokenizer.save_pretrained("demo-model")
print("✅ Model saved to ./demo-model")

# Step 4: Push to Hugging Face Hub (optional)
# Uncomment the lines below and set your username
"""
USERNAME = "your-huggingface-username"  # Change this to your HF username
REPO_NAME = f"{USERNAME}/demo-model"

if hf_token:
    print(f"🚀 Pushing to {REPO_NAME}...")
    model.push_to_hub(REPO_NAME)
    tokenizer.push_to_hub(REPO_NAME)
    print(f"✅ Model deployed to https://huggingface.co/{REPO_NAME}")
else:
    print("⚠️  Skipping hub upload - authenticate first with HF_TOKEN")
"""
