---
title: "Replicate"
---

# Replicate

## Introduction

Replicate is a model marketplace and cloud inference platform where developers can run thousands of open-source AI models through a simple API. The platform enables anyone to publish models, creating a rich ecosystem of community-contributed and official models. Now part of Cloudflare, Replicate is known for its simplicity and pay-per-second pricing.

### What We'll Cover

- Model marketplace and discovery
- Running models with Python and Node.js
- Image, video, and audio generation
- Custom model deployment with Cog
- Webhooks for async processing
- Pricing and cost optimization

### Prerequisites

- Python 3.8+ or Node.js 18+
- Replicate API token from [replicate.com](https://replicate.com/)
- Basic understanding of async operations

---

## Model Marketplace

### Categories and Popular Models

```
┌─────────────────────────────────────────────────────────────────┐
│                 REPLICATE MODEL ECOSYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Language Models                                                 │
│  ├── meta/llama-3.1-405b-instruct                              │
│  ├── meta/llama-3.2-90b-vision-instruct                        │
│  └── mistralai/mixtral-8x7b-instruct                           │
│                                                                  │
│  Image Generation                                                │
│  ├── black-forest-labs/flux-1.1-pro (Latest FLUX)              │
│  ├── black-forest-labs/flux-schnell (Fast)                     │
│  ├── stability-ai/sdxl                                          │
│  └── bytedance/sdxl-lightning-4step                            │
│                                                                  │
│  Video Generation                                                │
│  ├── minimax/video-01                                           │
│  ├── tencent/hunyuan-video                                      │
│  └── stability-ai/stable-video-diffusion                       │
│                                                                  │
│  Audio & Music                                                   │
│  ├── openai/whisper (Transcription)                            │
│  ├── cjwbw/musicgen (Music)                                     │
│  └── lucataco/xtts-v2 (TTS)                                    │
│                                                                  │
│  Image Editing                                                   │
│  ├── tencentarc/gfpgan (Face restore)                          │
│  ├── nightmareai/real-esrgan (Upscale)                         │
│  └── cjwbw/rembg (Background removal)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Discovery

| Category | Examples | Count |
|----------|----------|-------|
| Language | Llama, Mistral, Qwen | 500+ |
| Image Generation | FLUX, SDXL, Midjourney-style | 2000+ |
| Video | Stable Video, Runway-style | 200+ |
| Audio | Whisper, MusicGen, TTS | 300+ |
| Code | CodeLlama, StarCoder | 100+ |
| 3D | Point-E, Shap-E | 50+ |

---

## API Usage

### Installation

```bash
pip install replicate
# or
npm install replicate
```

### Running Models (Python)

```python
import replicate

def run_llm(prompt: str) -> str:
    """Run a language model"""
    output = replicate.run(
        "meta/llama-3.1-405b-instruct",
        input={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    # Most LLMs stream by default, so join the output
    return "".join(output)

result = run_llm("Write a haiku about machine learning")
print(result)
```

### Image Generation

```python
def generate_image(prompt: str, style: str = "photo") -> str:
    """Generate images with FLUX"""
    
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "webp",
            "output_quality": 80
        }
    )
    return output[0] if isinstance(output, list) else output

# Returns URL to generated image
image_url = generate_image(
    "A cozy coffee shop in Tokyo during cherry blossom season, "
    "warm lighting, film photography style"
)
print(f"Image: {image_url}")
```

### Video Generation

```python
def generate_video(prompt: str) -> str:
    """Generate video from text"""
    
    output = replicate.run(
        "minimax/video-01",
        input={
            "prompt": prompt,
            "duration": 5,
            "resolution": "720p"
        }
    )
    return output

video_url = generate_video(
    "A drone shot flying over mountains at sunrise"
)
```

### Audio Transcription

```python
def transcribe_audio(audio_url: str) -> dict:
    """Transcribe audio with Whisper"""
    
    output = replicate.run(
        "openai/whisper",
        input={
            "audio": audio_url,
            "model": "large-v3",
            "language": "en",
            "translate": False
        }
    )
    return output

result = transcribe_audio("https://example.com/podcast.mp3")
print(result["transcription"])
```

### Streaming

```python
def stream_llm(prompt: str):
    """Stream LLM responses in real-time"""
    
    for event in replicate.stream(
        "meta/llama-3.1-70b-instruct",
        input={
            "prompt": prompt,
            "max_tokens": 500
        }
    ):
        print(str(event), end="", flush=True)
    print()

stream_llm("Explain quantum computing in simple terms")
```

---

## Async Processing with Webhooks

### Creating Predictions

```python
import replicate

def create_async_prediction(prompt: str, webhook_url: str):
    """Start a prediction and get notified when done"""
    
    prediction = replicate.predictions.create(
        model="meta/llama-3.1-405b-instruct",
        input={"prompt": prompt, "max_tokens": 1000},
        webhook=webhook_url,
        webhook_events_filter=["completed"]
    )
    
    return prediction.id

# Later, check status
def check_prediction(prediction_id: str):
    prediction = replicate.predictions.get(prediction_id)
    return {
        "status": prediction.status,
        "output": prediction.output,
        "error": prediction.error
    }
```

### Webhook Handler (Flask)

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def handle_replicate_webhook():
    data = request.json
    
    prediction_id = data["id"]
    status = data["status"]
    output = data.get("output")
    
    if status == "succeeded":
        # Process the completed result
        print(f"Prediction {prediction_id} completed: {output}")
    elif status == "failed":
        print(f"Prediction {prediction_id} failed: {data.get('error')}")
    
    return "OK", 200
```

---

## Custom Deployments

### Deploy Your Own Model

```python
# 1. Create a deployment for dedicated capacity
deployment = replicate.deployments.create(
    owner="your-org",
    name="my-production-llm",
    model="meta/llama-3.1-70b-instruct",
    hardware="gpu-a100-large",
    min_instances=1,
    max_instances=5
)

# 2. Run predictions on your deployment
output = replicate.run(
    f"{deployment.owner}/{deployment.name}",
    input={"prompt": "Hello!"}
)
```

### Cog: Package Your Own Models

```yaml
# cog.yaml - Define your model
build:
  gpu: true
  python_version: "3.11"
  python_packages:
    - torch==2.1.0
    - transformers==4.35.0

predict: "predict.py:Predictor"
```

```python
# predict.py - Implement the Predictor class
from cog import BasePredictor, Input
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load model into memory"""
        self.model = torch.load("model.pt")
    
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_tokens: int = Input(default=100, ge=1, le=1000)
    ) -> str:
        """Run inference"""
        return self.model.generate(prompt, max_tokens)
```

```bash
# Push to Replicate
cog push r8.im/your-username/your-model
```

---

## Pricing

### Pay-Per-Second Model

```python
replicate_pricing = {
    "billing": "Per second of compute time",
    "minimum": "No minimum, pay only for what you use",
    "cold_start": "~10-30 seconds for some models (not billed)",
    "deployments": "Dedicated hardware at hourly rates"
}
```

### Example Costs

| Model Type | Approx. Cost | Notes |
|------------|--------------|-------|
| Llama 3.1 70B | ~$0.0025/sec | ~$0.18/1K tokens |
| FLUX.1 Pro | ~$0.05/image | High quality |
| FLUX Schnell | ~$0.003/image | Fast |
| Whisper Large | ~$0.0035/sec | Transcription |
| SDXL | ~$0.002/image | Stable Diffusion |

### Hardware Options

| GPU | Price/Hour | Best For |
|-----|------------|----------|
| CPU | $0.10 | Lightweight tasks |
| T4 | $0.55 | Inference, small models |
| A40 | $1.50 | Medium models |
| A100 40GB | $2.30 | Large models |
| A100 80GB | $3.50 | Very large models |

---

## Unique Features

```python
replicate_features = {
    "marketplace": "Thousands of community models",
    "simple_api": "One-liner run() interface",
    "versioning": "Pin to specific model versions for reproducibility",
    "webhooks": "Async processing with callbacks",
    "predictions_api": "Long-running job management",
    "fine_tuning": "Train custom FLUX and other models",
    "cloudflare": "CDN integration for fast delivery"
}
```

### Version Pinning

```python
# Always use a specific version for production
output = replicate.run(
    "black-forest-labs/flux-1.1-pro:abc123def456",  # Version hash
    input={"prompt": "..."}
)
```

---

## Best Practices

### When to Choose Replicate

```python
choose_replicate_when = [
    "Need variety - explore many model types",
    "Building prototypes quickly",
    "Want pay-per-use without commitments",
    "Need image/video/audio generation",
    "Publishing your own models",
    "Prefer simple API over complex SDKs"
]

consider_alternatives_when = [
    "Need lowest LLM latency (consider Groq)",
    "Production LLM at scale (consider dedicated providers)",
    "Require SLA guarantees (use deployments)",
    "Cost-sensitive high volume (consider self-hosting)"
]
```

---

## Summary

✅ **Huge marketplace**: Thousands of models across all modalities

✅ **Simple API**: `replicate.run()` for any model

✅ **Pay-per-use**: No minimums, billed per second

✅ **Custom deployment**: Scale with dedicated hardware

✅ **Cog framework**: Package and publish your own models

✅ **Webhooks**: Async processing for long-running tasks

**Next:** [DeepSeek](./11-deepseek.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Fireworks](./09-fireworks.md) | [AI Providers](./00-ai-providers-landscape.md) | [DeepSeek](./11-deepseek.md) |

