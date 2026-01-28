---
title: "Image Generation Models"
---

# Image Generation Models

## Introduction

Image generation models create images from text descriptions. From DALL-E to Stable Diffusion, these models enable AI art, design automation, and creative applications.

### What We'll Cover

- Major image generation models
- Style and quality differences
- Resolution and aspect ratios
- Control mechanisms

---

## Major Models

### DALL-E 3 (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

def generate_dalle(prompt: str, size: str = "1024x1024") -> str:
    """Generate image with DALL-E 3"""
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,  # 1024x1024, 1792x1024, 1024x1792
        quality="standard",  # or "hd"
        n=1
    )
    
    return response.data[0].url

url = generate_dalle("A serene mountain lake at sunset, photorealistic")
print(f"Image URL: {url}")
```

**Strengths:**
- Excellent prompt understanding
- Safe by design
- High quality
- Automatic prompt enhancement

### Stable Diffusion (Stability AI)

```python
import requests

def generate_stable_diffusion(prompt: str, api_key: str) -> bytes:
    """Generate image with Stable Diffusion"""
    
    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        }
    )
    
    return response.json()["artifacts"][0]["base64"]
```

**Strengths:**
- Open source versions available
- Highly customizable
- Large community
- Many fine-tuned variants

### Flux (Black Forest Labs)

```python
import replicate

def generate_flux(prompt: str) -> str:
    """Generate image with Flux"""
    
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt}
    )
    
    return output[0]  # Image URL
```

**Strengths:**
- Excellent text rendering
- Fast generation (Schnell variant)
- High quality (Pro variant)
- Good prompt adherence

### Midjourney

```
// Midjourney is accessed via Discord
// No direct API - use Discord bot commands

/imagine prompt: A serene mountain lake at sunset, photorealistic --v 6 --ar 16:9
```

**Strengths:**
- Exceptional artistic quality
- Strong community
- Easy to use
- Great for creative work

---

## Model Comparison

| Feature | DALL-E 3 | SDXL | Flux | Midjourney |
|---------|----------|------|------|------------|
| API Access | ✅ | ✅ | ✅ | Discord only |
| Open Source | ❌ | ✅ | Partial | ❌ |
| Text in Images | Good | Poor | Excellent | Good |
| Photorealism | High | High | High | Medium |
| Artistic Style | Good | Excellent | Good | Excellent |
| Speed | Medium | Fast | Very Fast | Medium |

---

## Resolution and Aspect Ratios

### DALL-E 3 Options

```python
# Available sizes
sizes = {
    "square": "1024x1024",
    "landscape": "1792x1024",
    "portrait": "1024x1792"
}

# Generate each
for name, size in sizes.items():
    response = client.images.generate(
        model="dall-e-3",
        prompt="Mountain landscape",
        size=size
    )
    print(f"{name}: {response.data[0].url}")
```

### Stable Diffusion Flexibility

```python
# SDXL supports many resolutions
# Optimal: multiples of 64, base ~1MP

optimal_sizes = [
    (1024, 1024),   # 1:1 square
    (1152, 896),    # 4:3 landscape
    (896, 1152),    # 3:4 portrait
    (1216, 832),    # 3:2 landscape
    (832, 1216),    # 2:3 portrait
    (1344, 768),    # 16:9 landscape
    (768, 1344),    # 9:16 portrait
]
```

### Upscaling

```python
def upscale_image(image_url: str, scale: int = 2) -> bytes:
    """Upscale image using Stability AI"""
    
    response = requests.post(
        "https://api.stability.ai/v1/generation/esrgan-v1-x2plus/image-to-image/upscale",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"image": requests.get(image_url).content}
    )
    
    return response.json()["artifacts"][0]["base64"]
```

---

## Quality Settings

### DALL-E Quality Modes

```python
# Standard quality (faster, cheaper)
standard = client.images.generate(
    model="dall-e-3",
    prompt="A cat sitting on a window sill",
    quality="standard",
    size="1024x1024"
)

# HD quality (more detailed)
hd = client.images.generate(
    model="dall-e-3",
    prompt="A cat sitting on a window sill",
    quality="hd",
    size="1024x1024"
)
```

### Stable Diffusion Parameters

```python
generation_params = {
    "cfg_scale": 7,        # 1-35, how closely to follow prompt
    "steps": 30,           # 10-50, more = higher quality, slower
    "sampler": "K_DPM_2_ANCESTRAL",  # Sampling algorithm
    "seed": 12345,         # For reproducibility
}

# Higher quality settings
hq_params = {
    "cfg_scale": 9,
    "steps": 50,
    "sampler": "K_EULER_ANCESTRAL",
}
```

---

## Inpainting and Outpainting

### Inpainting (Edit Part of Image)

```python
def inpaint_dalle(
    image_path: str,
    mask_path: str,
    prompt: str
) -> str:
    """Edit part of an image with DALL-E"""
    
    response = client.images.edit(
        model="dall-e-2",  # DALL-E 2 for editing
        image=open(image_path, "rb"),
        mask=open(mask_path, "rb"),  # White = edit, Black = keep
        prompt=prompt,
        size="1024x1024"
    )
    
    return response.data[0].url

# Usage: Replace a person's outfit
url = inpaint_dalle(
    "person.png",
    "outfit_mask.png",
    "Person wearing a red dress"
)
```

### Outpainting (Extend Image)

```python
def outpaint_stable_diffusion(
    image_base64: str,
    direction: str,  # "left", "right", "up", "down"
    prompt: str
) -> bytes:
    """Extend image boundaries"""
    
    # Implementation depends on specific API
    # Generally involves:
    # 1. Create larger canvas
    # 2. Place original image
    # 3. Mask empty areas
    # 4. Run img2img on masked areas
    pass
```

---

## Control Mechanisms

### ControlNet (Stable Diffusion)

```python
def generate_with_controlnet(
    prompt: str,
    control_image: bytes,
    control_type: str = "canny"  # canny, depth, pose, etc.
) -> bytes:
    """Generate with structural control"""
    
    response = requests.post(
        "https://api.stability.ai/v1/generation/controlnet",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "text_prompts": [{"text": prompt}],
            "control_type": control_type,
            "control_image": control_image,
            "control_strength": 0.8,
        }
    )
    
    return response.json()["artifacts"][0]["base64"]

# Control types:
# - canny: Edge detection
# - depth: Depth map
# - pose: Human pose
# - scribble: Hand-drawn lines
# - segmentation: Semantic regions
```

### Style Reference

```python
# Many models support style transfer
style_transfer = {
    "prompt": "A mountain landscape",
    "style_reference": "uploaded_artwork.jpg",
    "style_strength": 0.7  # 0-1, how much to apply style
}
```

---

## Hands-on Exercise

### Your Task

Build an image generation wrapper:

```python
from openai import OpenAI
from enum import Enum
from dataclasses import dataclass

client = OpenAI()

class AspectRatio(Enum):
    SQUARE = "1024x1024"
    LANDSCAPE = "1792x1024"
    PORTRAIT = "1024x1792"

@dataclass
class GeneratedImage:
    url: str
    prompt: str
    revised_prompt: str
    size: str

class ImageGenerator:
    """Wrapper for DALL-E 3 generation"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate(
        self,
        prompt: str,
        aspect_ratio: AspectRatio = AspectRatio.SQUARE,
        hd: bool = False,
        style: str = "vivid"  # "vivid" or "natural"
    ) -> GeneratedImage:
        """Generate image with specified settings"""
        
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=aspect_ratio.value,
            quality="hd" if hd else "standard",
            style=style,
            n=1
        )
        
        return GeneratedImage(
            url=response.data[0].url,
            prompt=prompt,
            revised_prompt=response.data[0].revised_prompt,
            size=aspect_ratio.value
        )
    
    def generate_variations(
        self,
        prompt: str,
        count: int = 3
    ) -> list:
        """Generate multiple variations of same prompt"""
        
        results = []
        for i in range(count):
            # Add variation instruction
            varied_prompt = f"{prompt} (variation {i+1}, unique interpretation)"
            result = self.generate(varied_prompt)
            results.append(result)
        
        return results

# Test
generator = ImageGenerator()

# Single generation
result = generator.generate(
    "A cozy coffee shop interior with plants and natural light",
    aspect_ratio=AspectRatio.LANDSCAPE,
    hd=True
)
print(f"Generated: {result.url}")
print(f"Revised prompt: {result.revised_prompt}")

# Multiple variations
variations = generator.generate_variations(
    "A futuristic city skyline",
    count=3
)
for i, v in enumerate(variations):
    print(f"Variation {i+1}: {v.url}")
```

---

## Summary

✅ **DALL-E 3**: Best prompt understanding, safe

✅ **Stable Diffusion**: Open source, customizable

✅ **Flux**: Best text rendering, fast

✅ **Midjourney**: Best artistic quality

✅ **ControlNet**: Structural control over generation

✅ **Inpainting/Outpainting**: Edit and extend images

**Next:** [Image Understanding Models](./08-image-understanding-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Moderation Models](./06-moderation-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Image Understanding Models](./08-image-understanding-models.md) |

