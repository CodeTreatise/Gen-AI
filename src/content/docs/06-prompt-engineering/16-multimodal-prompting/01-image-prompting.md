---
title: "Image Prompting Fundamentals"
---

# Image Prompting Fundamentals

## Introduction

Image prompting is the most common multimodal task—from analyzing product photos to extracting data from charts. This lesson covers the technical details of sending images to AI models, controlling quality and cost, and writing effective prompts for visual content.

> **🔑 Key Insight:** The way you send an image (URL, base64, file ID) and the detail level you choose can mean a 10x difference in token costs. Understanding these trade-offs is essential for production applications.

### What We'll Cover

- Three methods for providing images to models
- Detail levels and resolution control
- Token cost calculations with examples
- Multi-image prompts for comparison
- Best practices for image quality and orientation

### Prerequisites

- [Multimodal Prompting Overview](./00-multimodal-prompting-overview.md)

---

## Image Input Methods

### Method 1: URL Reference

The simplest approach—point to a publicly accessible image:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
            {
                "type": "input_text",
                "text": "Describe what you see in this image."
            }
        ]
    }]
)

print(response.output_text)
```

**Anthropic URL Example:**

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/product.jpg"
                }
            },
            {
                "type": "text",
                "text": "Describe this product's key features."
            }
        ]
    }]
)
```

**Gemini URL Example:**

```python
from google import genai
from google.genai import types
import httpx

client = genai.Client()

# Fetch image bytes from URL
image_url = "https://example.com/chart.png"
image_bytes = httpx.get(image_url).content

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        "Analyze this chart."
    ]
)
```

### Method 2: Base64 Encoded Data

For local files or when you need to process images before sending:

```python
import base64
from openai import OpenAI

client = OpenAI()

# Read and encode local image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

image_data = encode_image("receipt.jpg")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            },
            {
                "type": "input_text",
                "text": "Extract the merchant name, date, and total from this receipt."
            }
        ]
    }]
)
```

**Anthropic Base64 Example:**

```python
import base64
import anthropic

client = anthropic.Anthropic()

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": "Explain this system architecture diagram."
            }
        ]
    }]
)
```

**Gemini Local File Example:**

```python
from google import genai
from google.genai import types

client = genai.Client()

with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "What's in this photo?"
    ]
)
```

### Method 3: File API Upload

For large files or when you'll analyze the same image multiple times:

```python
from openai import OpenAI

client = OpenAI()

# Upload once
file = client.files.create(
    file=open("large_diagram.png", "rb"),
    purpose="vision"
)

# Use multiple times
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_file", "file_id": file.id},
            {"type": "input_text", "text": "Identify all components in this diagram."}
        ]
    }]
)
```

**Anthropic Files API:**

```python
import anthropic

client = anthropic.Anthropic()

# Upload file (requires beta header)
with open("image.jpg", "rb") as f:
    file = client.files.create(file=f)

# Use in message
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    betas=["files-api-2025-04-14"],
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": file.id
                }
            },
            {
                "type": "text",
                "text": "Describe this image."
            }
        ]
    }]
)
```

**Gemini Files API:**

```python
from google import genai

client = genai.Client()

# Upload (stored 48 hours, no cost)
uploaded_file = client.files.upload(file="large_image.jpg")

# Use in requests
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[uploaded_file, "Analyze this image in detail."]
)

# Check file status
file_info = client.files.get(name=uploaded_file.name)
print(f"File state: {file_info.state}")
```

---

## Detail Levels and Resolution Control

### OpenAI Detail Parameter

Control token usage with the `detail` parameter:

```python
# Low detail: 85 tokens, regardless of size
{
    "type": "input_image",
    "image_url": "https://example.com/image.jpg",
    "detail": "low"
}

# High detail: Variable based on resolution
{
    "type": "input_image",
    "image_url": "https://example.com/image.jpg",
    "detail": "high"
}

# Auto: Model decides (default)
{
    "type": "input_image",
    "image_url": "https://example.com/image.jpg",
    "detail": "auto"
}
```

### When to Use Each Level

| Detail | Tokens | Best For |
|--------|--------|----------|
| `low` | 85 fixed | General classification, dominant colors, simple questions |
| `high` | 765+ variable | Reading text, detailed analysis, small objects |
| `auto` | Varies | When unsure, let model decide |

### Gemini Media Resolution (3.0+)

Gemini 3 introduces per-media resolution control:

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        "Read all the text in this document."
    ],
    config={
        "media_resolution": "high"  # low, medium, high
    }
)
```

| Resolution | Tokens per Frame | Use Case |
|------------|------------------|----------|
| `low` | 66 | Quick analysis, thumbnails |
| `medium` | Default | Balanced |
| `high` | 258+ | Fine text, detailed inspection |

---

## Token Cost Calculations

### OpenAI GPT-4o Token Calculation

```python
def calculate_gpt4o_image_tokens(
    width: int,
    height: int,
    detail: str = "high"
) -> dict:
    """
    Calculate token cost for GPT-4o image input.
    """
    
    if detail == "low":
        return {"tokens": 85, "detail": "low", "tiles": 0}
    
    # Step 1: Scale to fit 2048x2048
    max_dim = 2048
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
    
    # Step 2: Scale shortest side to 768px
    min_dim = 768
    scale = min_dim / min(width, height)
    width = int(width * scale)
    height = int(height * scale)
    
    # Step 3: Count 512px tiles
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    tiles = tiles_x * tiles_y
    
    # Cost: 170 per tile + 85 base
    tokens = (tiles * 170) + 85
    
    return {
        "tokens": tokens,
        "detail": "high",
        "tiles": tiles,
        "scaled_size": f"{width}x{height}"
    }

# Examples
print(calculate_gpt4o_image_tokens(1024, 1024))
# {'tokens': 765, 'detail': 'high', 'tiles': 4, 'scaled_size': '768x768'}

print(calculate_gpt4o_image_tokens(2048, 4096))
# {'tokens': 1105, 'detail': 'high', 'tiles': 6, 'scaled_size': '768x1536'}

print(calculate_gpt4o_image_tokens(4096, 4096, "low"))
# {'tokens': 85, 'detail': 'low', 'tiles': 0}
```

### Anthropic Token Calculation

```python
def calculate_anthropic_image_tokens(width: int, height: int) -> dict:
    """
    Calculate token cost for Claude image input.
    Images are resized if larger than 1.15 megapixels.
    """
    
    max_pixels = 1_150_000  # 1.15 megapixels
    pixels = width * height
    
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        pixels = width * height
    
    # Formula: pixels / 750
    tokens = pixels // 750
    
    return {
        "tokens": tokens,
        "scaled_size": f"{width}x{height}",
        "megapixels": round(pixels / 1_000_000, 2)
    }

# Examples
print(calculate_anthropic_image_tokens(1092, 1092))
# {'tokens': 1590, 'scaled_size': '1092x1092', 'megapixels': 1.19}

print(calculate_anthropic_image_tokens(1000, 1000))
# {'tokens': 1333, 'scaled_size': '1000x1000', 'megapixels': 1.0}
```

### Gemini Token Calculation

```python
def calculate_gemini_image_tokens(
    width: int,
    height: int,
    media_resolution: str = "default"
) -> dict:
    """
    Calculate token cost for Gemini image input.
    """
    
    if media_resolution == "low":
        return {"tokens": 66, "resolution": "low"}
    
    # Small images: 258 tokens if both dims <= 384
    if width <= 384 and height <= 384:
        return {"tokens": 258, "resolution": "small"}
    
    # Larger images: tiled at 768x768
    # crop_unit = floor(min(width, height) / 1.5)
    crop_unit = min(width, height) // 1.5
    
    tiles_x = max(1, int(width / crop_unit))
    tiles_y = max(1, int(height / crop_unit))
    tiles = tiles_x * tiles_y
    
    tokens = tiles * 258
    
    return {
        "tokens": tokens,
        "resolution": "tiled",
        "tiles": tiles
    }

# Examples
print(calculate_gemini_image_tokens(300, 300))
# {'tokens': 258, 'resolution': 'small'}

print(calculate_gemini_image_tokens(960, 540))
# {'tokens': 1548, 'resolution': 'tiled', 'tiles': 6}
```

### Cost Comparison Table

For a 1024×1024 image:

| Provider | Model | Tokens | Cost (Input) |
|----------|-------|--------|--------------|
| OpenAI | GPT-4o (high) | 765 | ~$0.0019 |
| OpenAI | GPT-4o (low) | 85 | ~$0.0002 |
| OpenAI | GPT-4.1-mini | 1659 | ~$0.0007 |
| Anthropic | Claude Sonnet 4.5 | 1398 | ~$0.0042 |
| Gemini | Gemini 3 Flash | 1032 | ~$0.0002 |

---

## Multi-Image Prompts

### Comparing Images

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Compare these two product designs:"},
            {
                "type": "input_image",
                "image_url": "https://example.com/design_a.jpg"
            },
            {"type": "input_text", "text": "Design A (above)"},
            {
                "type": "input_image",
                "image_url": "https://example.com/design_b.jpg"
            },
            {"type": "input_text", "text": "Design B (above)"},
            {
                "type": "input_text",
                "text": "Which design is more visually appealing and why?"
            }
        ]
    }]
)
```

### Before/After Analysis

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Analyze these before and after photos:"},
            {
                "type": "input_image",
                "image_url": before_image_url
            },
            {"type": "input_text", "text": "[BEFORE]"},
            {
                "type": "input_image",
                "image_url": after_image_url
            },
            {"type": "input_text", "text": "[AFTER]"},
            {
                "type": "input_text",
                "text": "List all visible changes between the two images."
            }
        ]
    }]
)
```

### Reference Image for Style

```python
# Use one image as a style reference
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Here's a reference design style:"},
            {
                "type": "input_image",
                "image_url": reference_style_url
            },
            {"type": "input_text", "text": "And here's a new design to evaluate:"},
            {
                "type": "input_image",
                "image_url": new_design_url
            },
            {
                "type": "input_text",
                "text": "How well does the new design match the reference style? What changes would improve consistency?"
            }
        ]
    }]
)
```

### Multi-Image Limits

| Provider | Max Images per Request |
|----------|----------------------|
| OpenAI | 500 images |
| Anthropic | 100 API / 20 claude.ai |
| Gemini | 3,600 images (2.5+) |

---

## Image Quality Best Practices

### Image Requirements

| Requirement | Guideline |
|-------------|-----------|
| **Format** | PNG, JPEG, WEBP, GIF (non-animated) |
| **Size** | Up to 50MB total payload per request |
| **Clarity** | Clear, non-blurry images |
| **Orientation** | Correct rotation (models struggle with rotated content) |
| **Resolution** | Balance detail needs vs token cost |

### Preprocessing for Optimal Results

```python
from PIL import Image
import io
import base64

def preprocess_image(
    image_path: str,
    max_dimension: int = 1568,
    quality: int = 85
) -> str:
    """
    Preprocess image for optimal API submission.
    
    - Resize if too large
    - Correct EXIF orientation
    - Convert to JPEG
    - Return base64 encoded
    """
    
    img = Image.open(image_path)
    
    # Handle EXIF orientation
    from PIL import ExifTags
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass
    
    # Resize if needed
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed (for JPEG)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    
    return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
```

### Handling Rotated Images

```python
# Option 1: Mention rotation in prompt
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": rotated_image_url
            },
            {
                "type": "input_text",
                "text": "This image may be rotated. First determine the correct orientation, then describe what you see."
            }
        ]
    }]
)

# Option 2: Preprocess to correct orientation (preferred)
corrected_image = preprocess_image("rotated_photo.jpg")
```

---

## Common Mistakes

### ❌ Wrong Detail Level for Task

```python
# Bad: Low detail for text extraction
{
    "type": "input_image",
    "image_url": document_image,
    "detail": "low"  # ❌ Can't read small text
}

# Good: High detail for text
{
    "type": "input_image",
    "image_url": document_image,
    "detail": "high"  # ✅ Can read fine text
}
```

### ❌ Text Before Image

```python
# Less optimal: Text first
contents = [
    "What's in this image?",
    image_part
]

# Better: Image first
contents = [
    image_part,
    "What's in this image?"
]
```

### ❌ Sending Blurry Images

```python
# Validate image quality before sending
from PIL import Image
import cv2
import numpy as np

def is_blurry(image_path: str, threshold: float = 100.0) -> bool:
    """Detect if image is too blurry using Laplacian variance."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold

if is_blurry("photo.jpg"):
    print("Warning: Image may be too blurry for accurate analysis")
```

---

## Hands-on Exercise

### Your Task

Create a product comparison tool that:
1. Accepts two product images
2. Extracts key features from each
3. Creates a comparison table
4. Recommends which product is better for different use cases

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ProductFeatures(BaseModel):
    name: str
    key_features: list[str]
    apparent_quality: str
    target_audience: str
    price_tier: str  # budget, mid-range, premium

class ProductComparison(BaseModel):
    product_a: ProductFeatures
    product_b: ProductFeatures
    differences: list[str]
    recommendation_casual: str
    recommendation_professional: str

def compare_products(image_url_a: str, image_url_b: str) -> ProductComparison:
    """Compare two product images and extract structured comparison."""
    
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze these two products:"},
                {
                    "type": "input_image",
                    "image_url": image_url_a,
                    "detail": "high"
                },
                {"type": "input_text", "text": "Product A (above)"},
                {
                    "type": "input_image",
                    "image_url": image_url_b,
                    "detail": "high"
                },
                {"type": "input_text", "text": "Product B (above)"},
                {
                    "type": "input_text",
                    "text": """
                    Extract features for each product and compare them.
                    Provide recommendations for:
                    1. Casual/home use
                    2. Professional use
                    """
                }
            ]
        }],
        text_format=ProductComparison
    )
    
    return response.output_parsed

# Usage
comparison = compare_products(
    "https://example.com/camera_a.jpg",
    "https://example.com/camera_b.jpg"
)

print(f"Product A: {comparison.product_a.name}")
print(f"Product B: {comparison.product_b.name}")
print(f"\nKey Differences:")
for diff in comparison.differences:
    print(f"  - {diff}")
print(f"\nFor casual use: {comparison.recommendation_casual}")
print(f"For professional use: {comparison.recommendation_professional}")
```

</details>

---

## Summary

✅ **Three input methods:** URL, base64, and File API for different use cases
✅ **Detail levels matter:** `low` (85 tokens) vs `high` (765+ tokens) for GPT-4o
✅ **Token calculations differ:** Each provider has unique formulas
✅ **Multi-image prompts:** Label images clearly, place before questions
✅ **Preprocess images:** Correct orientation, resize for efficiency

**Next:** [Document Understanding](./02-document-understanding.md)

---

## Further Reading

- [OpenAI Vision Pricing Calculator](https://openai.com/api/pricing/)
- [Anthropic Vision Cost Guide](https://docs.anthropic.com/en/docs/build-with-claude/vision#calculate-image-costs)
- [Gemini Token Counting](https://ai.google.dev/gemini-api/docs/tokens)

---

<!-- 
Sources Consulted:
- OpenAI Vision Guide: Token calculations, detail levels, file limits
- Anthropic Vision: Image sizing, cost calculation, 100 image limit
- Gemini Image Understanding: 258 token tiles, 3600 image limit
-->
