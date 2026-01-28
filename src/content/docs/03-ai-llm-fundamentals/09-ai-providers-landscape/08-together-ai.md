---
title: "Together AI"
---

# Together AI

## Introduction

Together AI offers a wide selection of open-source models through a unified API, along with fine-tuning services and optimized inference. Great for accessing many models through one platform.

### What We'll Cover

- Model selection
- API usage
- Fine-tuning services
- Pricing

---

## Available Models

### Model Categories

| Category | Models |
|----------|--------|
| Chat | Llama 3.3 70B, Qwen 72B, Mistral |
| Code | CodeLlama, DeepSeek Coder |
| Embedding | UAE-Large, BGE |
| Image | FLUX, Stable Diffusion XL |
| Reasoning | Qwen QwQ, DeepSeek R1 |

### Popular Models

```python
popular_models = {
    "chat": [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "mistralai/Mixtral-8x22B-Instruct-v0.1"
    ],
    "code": [
        "codellama/CodeLlama-70b-Instruct-hf",
        "deepseek-ai/deepseek-coder-33b-instruct"
    ],
    "fast": [
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
}
```

---

## API Usage

### Basic Chat

```python
from together import Together

client = Together()

def together_chat(prompt: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

print(together_chat("Hello!"))
```

### Streaming

```python
def together_stream(prompt: str):
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### Embeddings

```python
def get_embeddings(texts: list) -> list:
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        input=texts
    )
    return [e.embedding for e in response.data]
```

---

## Image Generation

### FLUX Models

```python
def generate_image(prompt: str) -> str:
    response = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell",
        prompt=prompt,
        width=1024,
        height=1024,
        n=1
    )
    return response.data[0].url

# Generate image
url = generate_image("A futuristic city at sunset, cyberpunk style")
```

---

## Fine-Tuning

### Custom Model Training

```python
# Upload training data
file = client.files.upload(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Start fine-tuning
job = client.fine_tuning.create(
    training_file=file.id,
    model="meta-llama/Llama-3.1-8B-Instruct",
    n_epochs=3,
    learning_rate=1e-5
)

# Check status
status = client.fine_tuning.retrieve(job.id)
print(f"Status: {status.status}")

# Use fine-tuned model
response = client.chat.completions.create(
    model=status.fine_tuned_model,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Pricing

### Current Pricing (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| Llama 3.3 70B Turbo | $0.88 | $0.88 |
| Llama 3.1 8B Turbo | $0.18 | $0.18 |
| Qwen 72B Turbo | $0.90 | $0.90 |
| Mixtral 8x22B | $1.20 | $1.20 |

### Value Proposition

```python
together_value = {
    "variety": "Access many models through one API",
    "turbo_models": "Optimized versions for speed",
    "fine_tuning": "Easy custom model training",
    "pricing": "Competitive, often cheaper than alternatives"
}
```

---

## Turbo Models

### Optimized Inference

```python
# Together offers "Turbo" versions of popular models
# - Optimized for faster inference
# - Same quality as original
# - Often cheaper

turbo_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # vs regular
    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo"
]
```

---

## Summary

✅ **Wide model selection**: Many open models in one place

✅ **Turbo optimization**: Faster inference versions

✅ **Fine-tuning**: Easy custom training

✅ **Unified API**: OpenAI-compatible

✅ **Image generation**: FLUX and SD models

**Next:** [Fireworks AI](./09-fireworks.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Groq](./07-groq.md) | [AI Providers](./00-ai-providers-landscape.md) | [Fireworks AI](./09-fireworks.md) |

