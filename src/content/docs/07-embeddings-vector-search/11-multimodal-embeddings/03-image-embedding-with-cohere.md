---
title: "Image Embedding with Cohere"
---

# Image Embedding with Cohere

## Introduction

Cohere's embed-v4.0 supports **direct image embedding** through base64-encoded data URLs. This allows you to embed images into the same vector space as text, enabling powerful cross-modal search applications.

This lesson covers how to prepare images, encode them properly, and generate embeddings that live in the unified text-image space.

### What We'll Cover

- Base64 encoding for images
- Supported formats and size limits
- Embedding images with the Cohere API
- Image preprocessing best practices

### Prerequisites

- [Cohere embed-v4.0](./02-cohere-embed-v4.md)
- Python with `cohere` and `Pillow` packages
- Understanding of base64 encoding

---

## Image Requirements

### Supported Formats

| Format | Extension | MIME Type | Notes |
|--------|-----------|-----------|-------|
| PNG | `.png` | `image/png` | Lossless, supports transparency |
| JPEG | `.jpg`, `.jpeg` | `image/jpeg` | Lossy, smaller files |
| WebP | `.webp` | `image/webp` | Modern format, good compression |
| GIF | `.gif` | `image/gif` | Single frame only (not animated) |

### Size Limits

| Constraint | Limit |
|------------|-------|
| Max file size | 5 MB |
| Min dimensions | No minimum (but small images = poor quality) |
| Recommended | 224×224 to 1024×1024 pixels |
| Aspect ratio | Any (model handles internally) |

> **Note:** For best results, use images at least 224×224 pixels. Very small images may produce lower-quality embeddings.

---

## Base64 Encoding

### What is Base64?

Base64 converts binary data (like images) into ASCII text that can be safely transmitted in JSON:

```
Binary Image → Base64 String → Data URL
```

### Data URL Format

The embed API expects a **data URL** format:

```
data:image/{format};base64,{encoded_data}
```

Example:
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...
```

---

## Encoding Images in Python

### Method 1: From File Path

```python
import base64
from pathlib import Path

def image_to_base64_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    path = Path(image_path)
    
    # Determine MIME type from extension
    extension = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    mime_type = mime_types.get(extension, 'image/png')
    
    # Read and encode
    with open(path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.standard_b64encode(image_data).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"


# Usage
data_url = image_to_base64_data_url("photos/sunset.jpg")
print(f"Data URL length: {len(data_url)} characters")
```

**Output:**
```
Data URL length: 145234 characters
```

### Method 2: From URL (Download First)

```python
import base64
import requests
from io import BytesIO

def url_to_base64_data_url(image_url: str) -> str:
    """Download image from URL and convert to base64 data URL."""
    response = requests.get(image_url)
    response.raise_for_status()
    
    # Get content type from response
    content_type = response.headers.get('Content-Type', 'image/jpeg')
    
    # Encode to base64
    base64_data = base64.standard_b64encode(response.content).decode('utf-8')
    
    return f"data:{content_type};base64,{base64_data}"


# Usage
url = "https://example.com/images/photo.jpg"
data_url = url_to_base64_data_url(url)
```

### Method 3: From PIL Image Object

```python
import base64
from io import BytesIO
from PIL import Image

def pil_to_base64_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_data = buffer.getvalue()
    
    mime_type = f"image/{format.lower()}"
    base64_data = base64.standard_b64encode(image_data).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"


# Usage
img = Image.open("photo.jpg")
data_url = pil_to_base64_data_url(img, format="JPEG")
```

---

## Embedding Images with Cohere

### Basic Image Embedding

```python
import cohere

co = cohere.ClientV2()

# Prepare image as data URL
data_url = image_to_base64_data_url("sunset.jpg")

# Embed using input_type="image" and images parameter
response = co.embed(
    model="embed-v4.0",
    input_type="image",
    embedding_types=["float"],
    images=[data_url]  # List of data URLs
)

embedding = response.embeddings.float_[0]
print(f"Image embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

**Output:**
```
Image embedding dimensions: 1536
First 5 values: [0.0234, -0.0156, 0.0478, -0.0312, 0.0089]
```

### Key Parameters for Images

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` | `"embed-v4.0"` | Must use v4.0 for images |
| `input_type` | `"image"` | Required for image-only embedding |
| `images` | `[data_url]` | List of base64 data URLs |
| `embedding_types` | `["float"]` | Same options as text |

> **Warning:** You can only embed **1 image per request** when using the images parameter. For batch processing, make multiple API calls.

---

## Complete Embedding Pipeline

### Image Embedding Class

```python
import base64
import cohere
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Union, Optional


class ImageEmbedder:
    """Helper class for embedding images with Cohere."""
    
    MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    
    def __init__(self, api_key: Optional[str] = None):
        self.co = cohere.ClientV2(api_key=api_key) if api_key else cohere.ClientV2()
    
    def _validate_image(self, image_path: Path) -> None:
        """Validate image meets requirements."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {image_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        if image_path.stat().st_size > self.MAX_SIZE_BYTES:
            raise ValueError(
                f"Image too large: {image_path.stat().st_size / 1024 / 1024:.1f}MB. "
                f"Max: 5MB"
            )
    
    def _to_data_url(self, image_path: Union[str, Path]) -> str:
        """Convert image file to base64 data URL."""
        path = Path(image_path)
        self._validate_image(path)
        
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        mime_type = mime_types[path.suffix.lower()]
        
        with open(path, 'rb') as f:
            image_data = f.read()
        
        base64_data = base64.standard_b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"
    
    def embed(
        self,
        image_path: Union[str, Path],
        dimensions: int = 1536
    ) -> list[float]:
        """
        Embed a single image.
        
        Args:
            image_path: Path to image file
            dimensions: Output dimensions (256, 512, 1024, or 1536)
        
        Returns:
            List of floats representing the embedding
        """
        data_url = self._to_data_url(image_path)
        
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            output_dimension=dimensions,
            images=[data_url]
        )
        
        return response.embeddings.float_[0]
    
    def embed_batch(
        self,
        image_paths: list[Union[str, Path]],
        dimensions: int = 1536,
        show_progress: bool = True
    ) -> list[list[float]]:
        """
        Embed multiple images (one API call per image).
        
        Args:
            image_paths: List of paths to image files
            dimensions: Output dimensions
            show_progress: Whether to print progress
        
        Returns:
            List of embeddings
        """
        embeddings = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if show_progress:
                print(f"Embedding {i+1}/{total}: {Path(path).name}")
            
            emb = self.embed(path, dimensions)
            embeddings.append(emb)
        
        return embeddings


# Usage
embedder = ImageEmbedder()

# Single image
embedding = embedder.embed("products/shoe_001.jpg")
print(f"Embedding shape: {len(embedding)}")

# Multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
embeddings = embedder.embed_batch(images)
print(f"Embedded {len(embeddings)} images")
```

---

## Image Preprocessing

### Resizing Large Images

Large images waste bandwidth without improving embeddings:

```python
from PIL import Image
from io import BytesIO
import base64

def preprocess_image(
    image_path: str,
    max_size: int = 1024,
    quality: int = 85
) -> str:
    """
    Preprocess image: resize if needed, compress, return data URL.
    
    Args:
        image_path: Path to original image
        max_size: Maximum dimension (width or height)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 data URL of processed image
    """
    img = Image.open(image_path)
    
    # Convert RGBA to RGB (JPEG doesn't support transparency)
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if larger than max_size
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"Resized from {Image.open(image_path).size} to {new_size}")
    
    # Save to buffer as JPEG
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    
    # Check size
    size_kb = buffer.tell() / 1024
    print(f"Processed image size: {size_kb:.1f} KB")
    
    # Convert to data URL
    buffer.seek(0)
    base64_data = base64.standard_b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_data}"


# Usage
data_url = preprocess_image("large_photo.jpg", max_size=768)
```

**Output:**
```
Resized from (4000, 3000) to (768, 576)
Processed image size: 87.3 KB
```

### Handling Different Color Modes

```python
from PIL import Image

def normalize_image(image: Image.Image) -> Image.Image:
    """Convert image to RGB, handling various input modes."""
    if image.mode == 'RGB':
        return image
    
    if image.mode == 'RGBA':
        # Handle transparency
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    
    if image.mode == 'P':
        # Palette mode (e.g., some GIFs)
        return image.convert('RGB')
    
    if image.mode == 'L':
        # Grayscale
        return image.convert('RGB')
    
    if image.mode == 'CMYK':
        # Print format
        return image.convert('RGB')
    
    # Fallback
    return image.convert('RGB')
```

---

## Cross-Modal Search Example

### Text-to-Image Search

```python
import numpy as np
import cohere
from pathlib import Path

co = cohere.ClientV2()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Index images
image_dir = Path("product_images")
image_index = []

for img_path in image_dir.glob("*.jpg"):
    # Convert to data URL
    data_url = image_to_base64_data_url(str(img_path))
    
    # Get embedding
    response = co.embed(
        model="embed-v4.0",
        input_type="image",
        embedding_types=["float"],
        images=[data_url]
    )
    
    image_index.append({
        "path": str(img_path),
        "embedding": response.embeddings.float_[0]
    })
    print(f"Indexed: {img_path.name}")

# 2. Search with text query
query = "red running shoes"

query_response = co.embed(
    model="embed-v4.0",
    input_type="search_query",  # Text query input type
    embedding_types=["float"],
    texts=[query]
)
query_emb = query_response.embeddings.float_[0]

# 3. Find similar images
results = []
for item in image_index:
    score = cosine_similarity(query_emb, item["embedding"])
    results.append({"path": item["path"], "score": score})

results.sort(key=lambda x: x["score"], reverse=True)

# 4. Show top results
print(f"\nTop images for '{query}':")
for i, result in enumerate(results[:5]):
    print(f"  {i+1}. {result['path']} (score: {result['score']:.4f})")
```

**Output:**
```
Indexed: shoe_001.jpg
Indexed: shoe_002.jpg
Indexed: shoe_003.jpg
...

Top images for 'red running shoes':
  1. product_images/red_nike_runner.jpg (score: 0.8234)
  2. product_images/red_adidas_running.jpg (score: 0.7956)
  3. product_images/maroon_trainers.jpg (score: 0.7412)
  4. product_images/blue_running_shoes.jpg (score: 0.6234)
  5. product_images/red_heels.jpg (score: 0.5891)
```

---

## Common Issues and Solutions

### Issue 1: File Too Large

```python
# ❌ Error: Image too large
response = co.embed(images=[huge_data_url])
# APIError: Image size exceeds 5MB limit

# ✅ Solution: Preprocess first
data_url = preprocess_image("huge.jpg", max_size=1024, quality=80)
response = co.embed(images=[data_url])
```

### Issue 2: Wrong Input Type

```python
# ❌ Error: Wrong input_type for images
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",  # Wrong for images!
    images=[data_url]
)

# ✅ Correct: Use input_type="image" for image-only
response = co.embed(
    model="embed-v4.0",
    input_type="image",  # Correct
    images=[data_url]
)
```

### Issue 3: Invalid Base64

```python
# ❌ Error: Plain base64 without data URL prefix
response = co.embed(images=["iVBORw0KGgoAAAANSUhEUg..."])

# ✅ Correct: Include full data URL format
response = co.embed(images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."])
```

---

## Summary

✅ **Base64 data URLs** are required: `data:image/{format};base64,{data}`  
✅ Supported formats: **PNG, JPEG, WebP, GIF** (max 5MB)  
✅ Use `input_type="image"` with `images` parameter  
✅ **1 image per request** for image-only embedding  
✅ Preprocess large images to **reduce bandwidth** and latency  
✅ Images and text share the **same vector space** for cross-modal search

---

**Next:** [Mixed Content Embeddings →](./04-mixed-content-embeddings.md)

---

<!-- 
Sources Consulted:
- Cohere Embed API Reference: https://docs.cohere.com/reference/embed
- Cohere Multimodal Embeddings Guide: https://docs.cohere.com/docs/multimodal-embeddings
- PIL/Pillow Documentation: https://pillow.readthedocs.io/
-->
