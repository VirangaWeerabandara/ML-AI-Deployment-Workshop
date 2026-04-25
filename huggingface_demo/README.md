# Hugging Face Model Deployment Guide

This repository demonstrates how to deploy machine learning models to the Hugging Face Hub.

## Overview

This project shows how to:

- Load a pre-trained model from Hugging Face
- Save it locally
- Deploy it to the Hugging Face Hub for sharing and easy access

## Prerequisites

Install the required packages:

```bash
pip install torch torchvision transformers huggingface-hub
```

**Important:** Ensure version compatibility between PyTorch and torchvision:

- PyTorch 2.6.0 works with torchvision 0.17.0+
- For torch 2.6.0, use: `pip install torchvision==0.17.0`

## Getting Started

### 1. Set Up Authentication

Create a Hugging Face account at https://huggingface.co and generate an API token:

1. Go to https://huggingface.co/settings/tokens
2. Click "New token" and select "Write" access
3. Copy the token and save it securely

**NEVER commit API tokens to version control.** Use environment variables instead:

```bash
# Windows (PowerShell)
$env:HF_TOKEN = "your_token_here"

# Windows (Command Prompt)
set HF_TOKEN=your_token_here

# Linux/macOS
export HF_TOKEN="your_token_here"
```

Or store it in a `.env` file (add `.env` to `.gitignore`):

```
HF_TOKEN=your_token_here
```

### 2. Load and Save a Model Locally

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"

# Download and load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("demo-model")
tokenizer.save_pretrained("demo-model")
```

### 3. Push Model to Hugging Face Hub

#### Option A: Using Python Script

```python
from huggingface_hub import HfApi, login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Login (do this once)
login()

# Or programmatically
login(token="your_hf_token")

# Load your saved model
model = AutoModelForSequenceClassification.from_pretrained("./demo-model")
tokenizer = AutoTokenizer.from_pretrained("./demo-model")

# Push to hub
model.push_to_hub("your-username/demo-model")
tokenizer.push_to_hub("your-username/demo-model")
```

#### Option B: Using CLI

```bash
# Login to Hugging Face
huggingface-cli login

# Push to hub
huggingface-cli repo create demo-model
git clone https://huggingface.co/your-username/demo-model
cd demo-model
cp -r ../demo-model/* .
git add .
git commit -m "Add model files"
git push
```

#### Option C: Using `push_to_hub()` Method

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

login()  # Authenticate with your token

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./demo-model")
tokenizer = AutoTokenizer.from_pretrained("./demo-model")

# Push to hub in one command
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

## Deployment Checklist

- [ ] **Authentication**: Set up your HF_TOKEN securely
- [ ] **Model Testing**: Test your model locally before uploading
- [ ] **Documentation**: Create a model card (README.md in the repo)
- [ ] **Licenses**: Ensure proper license declarations
- [ ] **Privacy**: Never commit sensitive information (tokens, credentials)
- [ ] **Version Compatibility**: Check PyTorch and torchvision compatibility
- [ ] **Files**: Include model weights, tokenizer files, and config.json
- [ ] **Metadata**: Add tags and model information to the README

## Creating a Model Card on Hugging Face

Create a `README.md` in your model repository on the hub:

````markdown
# Model Card for your-model-name

## Model Details

- **Model Type**: Text Classification / Sequence Classification
- **Base Model**: DistilBERT
- **Training Data**: [Your training data description]
- **License**: Apache 2.0

## How to Use

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/your-model-name")
result = classifier("This is a great example")
print(result)
```
````

## Performance

- **Accuracy**: XX%
- **F1 Score**: XX

## Limitations

- [List any limitations]

## Ethical Considerations

- [Any ethical considerations]

````

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: Could not import module 'DistilBertForSequenceClassification'`

**Solution**: Fix torchvision version mismatch:
```bash
pip install torchvision==0.17.0 --force-reinstall
````

### Issue: `RuntimeError: operator torchvision::nms does not exist`

**Solution**: Ensure compatible versions:

```bash
pip install torch==2.6.0 torchvision==0.17.0 transformers
```

### Issue: `Authentication required but no token found`

**Solution**: Login with your token:

```bash
huggingface-cli login
# Or set environment variable
$env:HF_TOKEN = "your_token"
```

### Issue: `RepositoryNotFoundError`

**Solution**: Ensure your repository exists and you have write access:

```bash
# Check if repo exists
huggingface-cli repo list

# Create repo if needed
huggingface-cli repo create my-model-name
```

## Best Practices

1. **Version Control**: Use git to track model development
2. **Documentation**: Provide clear README and model cards
3. **Testing**: Test model inference before deployment
4. **Security**: Never commit API tokens or credentials
5. **Organization**: Use meaningful model names (e.g., `company/model-purpose-v1`)
6. **Tags**: Add relevant tags for discoverability
7. **Versioning**: Use semantic versioning (v1.0, v1.1, etc.)

## Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Hugging Face Model Card Guide](https://huggingface.co/docs/hub/model-cards)
- [Transformers Library Documentation](https://huggingface.co/docs/transformers)
- [huggingface_hub Python Library](https://github.com/huggingface/huggingface_hub)

## Quick Command Reference

```bash
# Login
huggingface-cli login

# List your repositories
huggingface-cli repo list

# Create new repository
huggingface-cli repo create my-model-name

# Clone a repository
git clone https://huggingface.co/username/model-name

# View model details
huggingface-cli repo view username/model-name
```

## Example Workflow

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os

# 1. Authenticate
token = os.getenv("HF_TOKEN")
login(token=token)

# 2. Load model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Fine-tune (optional - your training code here)
# ...

# 4. Save locally
model.save_pretrained("./demo-model")
tokenizer.save_pretrained("./demo-model")

# 5. Push to Hub
repo_name = "your-username/your-model-name"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✅ Model deployed to: https://huggingface.co/{repo_name}")
```

## Support

For issues or questions:

- Check [Hugging Face Discord](https://discord.gg/JfAtqhgAVk)
- Visit [GitHub Issues](https://github.com/huggingface/transformers/issues)
- Read [Documentation](https://huggingface.co/docs)

---

**Last Updated**: April 2026
