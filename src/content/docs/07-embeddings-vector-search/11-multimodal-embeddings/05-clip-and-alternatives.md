---
title: "CLIP and Alternatives"
---

# CLIP and Alternatives

## Introduction

While Cohere embed-v4.0 is a leading commercial option, several other multimodal embedding models exist—including open-source alternatives you can run locally. Understanding the landscape helps you choose the right model for your use case.

This lesson covers **OpenAI CLIP**, **Google's multimodal embeddings**, **SigLIP**, and open-source implementations.

### What We'll Cover

- OpenAI CLIP architecture and usage
- Google Vertex AI multimodal embeddings
- SigLIP improvements over CLIP
- Open-source options (OpenCLIP, Sentence Transformers)
- Model comparison and selection guide

### Prerequisites

- [What Are Multimodal Embeddings](./01-what-are-multimodal-embeddings.md)
- Understanding of vector embeddings
- Python environment setup

---

## OpenAI CLIP

### What is CLIP?

**CLIP** (Contrastive Language-Image Pre-training) was released by OpenAI in 2021 and revolutionized multimodal AI. It learns visual concepts from natural language supervision.

```
┌─────────────────────────────────────────────────────────┐
│                    CLIP Architecture                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Text Input                    Image Input             │
│       │                             │                   │
│       ▼                             ▼                   │
│  ┌─────────────┐           ┌─────────────────┐         │
│  │    Text     │           │     Vision      │         │
│  │  Transformer│           │   Transformer   │         │
│  │  (GPT-like) │           │   (ViT-based)   │         │
│  └─────────────┘           └─────────────────┘         │
│       │                             │                   │
│       ▼                             ▼                   │
│  Text Embedding              Image Embedding            │
│       │                             │                   │
│       └──────────┬──────────────────┘                   │
│                  ▼                                      │
│           Shared Vector Space                           │
│         (512 or 768 dimensions)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Details |
|---------|---------|
| **Training data** | 400M image-text pairs from internet |
| **Approach** | Contrastive learning |
| **Variants** | ViT-B/32, ViT-B/16, ViT-L/14 |
| **Dimensions** | 512 (B/32, B/16) or 768 (L/14) |
| **License** | MIT (open source) |

### Using CLIP with OpenAI's Package

```python
import torch
import clip
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Embed text
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Text embeddings shape: {text_features.shape}")
# Output: Text embeddings shape: torch.Size([2, 512])
```

### Embedding Images with CLIP

```python
# Load and preprocess image
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

print(f"Image embedding shape: {image_features.shape}")
# Output: Image embedding shape: torch.Size([1, 512])
```

### Zero-Shot Classification

CLIP's famous capability—classify images with arbitrary text labels:

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image = preprocess(Image.open("mystery_animal.jpg")).unsqueeze(0).to(device)

# Define candidate labels
labels = ["a cat", "a dog", "a bird", "a fish", "a horse"]
text = clip.tokenize([f"a photo of {label}" for label in labels]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Print results
for label, score in zip(labels, similarity[0]):
    print(f"{label}: {score.item():.2%}")
```

**Output:**
```
a cat: 94.23%
a dog: 3.45%
a bird: 1.12%
a fish: 0.78%
a horse: 0.42%
```

---

## Google Vertex AI Multimodal Embeddings

### Overview

Google offers multimodal embeddings through Vertex AI, supporting **text, images, and video** in the same space.

| Feature | Details |
|---------|---------|
| **Model** | `multimodalembedding@001` |
| **Dimensions** | 1408 (default), 128, 256, 512 available |
| **Modalities** | Text, Image, Video |
| **Languages** | English only |
| **Max text** | 32 tokens |

### Setup

```bash
pip install google-cloud-aiplatform
```

```python
from google.cloud import aiplatform
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image

# Initialize
vertexai.init(project="your-project", location="us-central1")
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
```

### Text Embedding

```python
# Embed text
embeddings = model.get_embeddings(
    contextual_text="A golden retriever playing in the park"
)

text_embedding = embeddings.text_embedding
print(f"Dimensions: {len(text_embedding)}")  # 1408
```

### Image Embedding

```python
# Embed image
image = Image.load_from_file("dog.jpg")

embeddings = model.get_embeddings(image=image)

image_embedding = embeddings.image_embedding
print(f"Dimensions: {len(image_embedding)}")  # 1408
```

### Video Embedding

Unique to Google—embed video segments:

```python
from vertexai.vision_models import Video, VideoSegmentConfig

# Load video
video = Video.load_from_file("product_demo.mp4")

# Configure segment extraction
segment_config = VideoSegmentConfig(
    start_offset_sec=0,
    end_offset_sec=30,
    interval_sec=10  # Extract every 10 seconds
)

embeddings = model.get_embeddings(
    video=video,
    video_segment_config=segment_config
)

# Get embeddings for each segment
for segment in embeddings.video_embeddings:
    print(f"Segment {segment.start_offset_sec}-{segment.end_offset_sec}s")
    print(f"  Embedding: {len(segment.embedding)} dimensions")
```

### Lower Dimensions

```python
# Use 512 dimensions for faster search
embeddings = model.get_embeddings(
    contextual_text="A sunset over the ocean",
    dimension=512  # 128, 256, 512, or 1408
)

print(f"Dimensions: {len(embeddings.text_embedding)}")  # 512
```

---

## SigLIP

### What is SigLIP?

**SigLIP** (Sigmoid Loss for Language-Image Pre-training) improves on CLIP by using sigmoid loss instead of softmax, enabling better batch efficiency and quality.

```
┌────────────────────────────────────────────────────┐
│                CLIP vs SigLIP Loss                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  CLIP (Softmax):                                   │
│    - Requires large batch sizes (32K+)            │
│    - All negatives compete for probability        │
│    - Memory intensive                              │
│                                                    │
│  SigLIP (Sigmoid):                                 │
│    - Works with smaller batches                   │
│    - Binary classification per pair               │
│    - More efficient training                       │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Key Improvements

| Aspect | CLIP | SigLIP |
|--------|------|--------|
| Loss function | Softmax | Sigmoid |
| Batch size needed | Large (32K) | Smaller OK |
| Zero-shot accuracy | Baseline | +2-3% better |
| Training efficiency | Standard | Higher |

### Using SigLIP via Transformers

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

# Load SigLIP model
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Prepare inputs
image = Image.open("photo.jpg")
texts = ["a cat", "a dog", "a bird"]

inputs = processor(
    text=texts,
    images=image,
    padding="max_length",
    return_tensors="pt"
)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Access embeddings
image_embeds = outputs.image_embeds  # [1, dim]
text_embeds = outputs.text_embeds    # [3, dim]

print(f"Image embedding: {image_embeds.shape}")
print(f"Text embeddings: {text_embeds.shape}")
```

### Zero-Shot with SigLIP

```python
# Calculate similarity (using sigmoid, not softmax)
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.2%}")
```

---

## OpenCLIP

### What is OpenCLIP?

**OpenCLIP** is an open-source reproduction of CLIP with additional model variants and training datasets.

### Installation

```bash
pip install open_clip_torch
```

### Available Models

```python
import open_clip

# List available models
models = open_clip.list_pretrained()
print(f"Available models: {len(models)}")

# Popular options:
# - ('ViT-B-32', 'laion2b_s34b_b79k')
# - ('ViT-L-14', 'laion2b_s32b_b82k')
# - ('ViT-H-14', 'laion2b_s32b_b79k')
# - ('ViT-bigG-14', 'laion2b_s39b_b160k')
```

### Using OpenCLIP

```python
import open_clip
import torch
from PIL import Image

# Load model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare inputs
image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
text = tokenizer(["a cat", "a dog", "a bird"])

# Get embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate similarity
similarity = (image_features @ text_features.T)
print(f"Similarities: {similarity}")
```

### Model Size Comparison

| Model | Parameters | Dimensions | LAION-5B Score |
|-------|------------|------------|----------------|
| ViT-B/32 | 151M | 512 | Baseline |
| ViT-B/16 | 149M | 512 | +2% |
| ViT-L/14 | 428M | 768 | +5% |
| ViT-H/14 | 986M | 1024 | +8% |
| ViT-bigG/14 | 2.5B | 1280 | +10% |

---

## Sentence Transformers Multimodal

### CLIP via Sentence Transformers

The `sentence-transformers` library provides easy-to-use multimodal models:

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load CLIP model through sentence-transformers
model = SentenceTransformer('clip-ViT-B-32')

# Embed text (returns numpy array)
text_embeddings = model.encode([
    "a photo of a cat",
    "a photo of a dog"
])

# Embed images
image = Image.open("cat.jpg")
image_embedding = model.encode(image)

print(f"Text shape: {text_embeddings.shape}")    # (2, 512)
print(f"Image shape: {image_embedding.shape}")   # (512,)
```

### Available Multimodal Models

```python
# Some multimodal models in sentence-transformers
multimodal_models = [
    "clip-ViT-B-32",           # OpenAI CLIP
    "clip-ViT-B-16",           # Larger CLIP
    "clip-ViT-L-14",           # Large CLIP
    "clip-ViT-B-32-multilingual-v1",  # Multilingual
]

# Load any model
model = SentenceTransformer('clip-ViT-L-14')
```

### Unified Interface

```python
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

class UnifiedEmbedder:
    """Simple interface for multimodal embeddings."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed text strings."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed single image."""
        image = Image.open(image_path)
        return self.model.encode(image, convert_to_numpy=True)
    
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Embed multiple images."""
        images = [Image.open(p) for p in image_paths]
        return self.model.encode(images, convert_to_numpy=True)


# Usage
embedder = UnifiedEmbedder("clip-ViT-L-14")

texts = embedder.embed_text(["cat", "dog"])
image = embedder.embed_image("photo.jpg")

# Same space - can compare directly
similarity = np.dot(image, texts.T)
```

---

## Model Comparison

### Feature Comparison

| Model | Provider | Dimensions | Languages | Video | Open Source |
|-------|----------|------------|-----------|-------|-------------|
| **embed-v4.0** | Cohere | 256-1536 | 100+ | ❌ | ❌ |
| **CLIP** | OpenAI | 512-768 | English | ❌ | ✅ |
| **SigLIP** | Google | 512-1024 | English | ❌ | ✅ |
| **Vertex AI** | Google | 128-1408 | English | ✅ | ❌ |
| **OpenCLIP** | Community | 512-1280 | English | ❌ | ✅ |

### When to Use Each

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Production API** | Cohere embed-v4.0 | Best quality, easy API |
| **Multilingual** | Cohere embed-v4.0 | 100+ languages |
| **Video content** | Google Vertex AI | Only option for video |
| **On-premise/local** | OpenCLIP or SigLIP | Open source, self-hosted |
| **Low latency** | CLIP ViT-B/32 | Smallest, fastest |
| **Highest quality** | OpenCLIP ViT-bigG | Best open-source accuracy |
| **Research/academic** | Any open-source | Modifiable, transparent |

### Quality vs Speed Trade-off

```
Quality
   ▲
   │      ┌──────────────┐
   │      │ ViT-bigG-14  │ ← Highest quality, slowest
   │      └──────────────┘
   │           ┌──────────────┐
   │           │ embed-v4.0   │ ← Production sweet spot
   │           └──────────────┘
   │      ┌──────────────┐
   │      │  ViT-L-14    │
   │      └──────────────┘
   │           ┌──────────────┐
   │           │  SigLIP-B    │
   │           └──────────────┘
   │                ┌──────────────┐
   │                │  ViT-B-32    │ ← Fast, good enough
   │                └──────────────┘
   └─────────────────────────────────────► Speed
```

---

## Practical Selection Guide

### Decision Tree

```
Do you need multilingual support?
├── YES → Cohere embed-v4.0
└── NO
    │
    Do you need video embedding?
    ├── YES → Google Vertex AI
    └── NO
        │
        Can you use cloud APIs?
        ├── YES → Cohere embed-v4.0 (quality) or Google (if GCP)
        └── NO (need on-premise)
            │
            What's your priority?
            ├── Quality → OpenCLIP ViT-H-14 or ViT-bigG-14
            ├── Speed → CLIP ViT-B-32
            └── Balance → SigLIP or OpenCLIP ViT-L-14
```

### Cost Considerations

| Option | Pricing Model | Approximate Cost |
|--------|---------------|------------------|
| Cohere | Per 1K embeddings | ~$0.10/1K |
| Google Vertex | Per 1K chars/images | ~$0.05-0.15/1K |
| OpenCLIP/CLIP | Self-hosted | Compute costs only |

---

## Summary

✅ **CLIP** pioneered multimodal embeddings with contrastive learning  
✅ **SigLIP** improves CLIP with sigmoid loss (better efficiency)  
✅ **Google Vertex AI** is the only option for **video embeddings**  
✅ **OpenCLIP** provides high-quality **open-source** alternatives  
✅ **Cohere embed-v4.0** leads for **multilingual** production use  
✅ Choice depends on: **language needs, hosting, quality vs speed, budget**

---

**Next:** [Multimodal Use Cases →](./06-use-cases.md)

---

<!-- 
Sources Consulted:
- OpenAI CLIP: https://openai.com/index/clip/
- Google Vertex AI Multimodal Embeddings: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
- SigLIP Paper: https://arxiv.org/abs/2303.15343
- OpenCLIP: https://github.com/mlfoundations/open_clip
- Sentence Transformers: https://www.sbert.net/
-->
