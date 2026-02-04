---
title: "Vision Capabilities and Object Detection"
---

# Vision Capabilities and Object Detection

## Introduction

Beyond simple image description, modern vision models can locate objects, segment regions, and answer spatial questions. This lesson explores the detection and analysis capabilities available, along with their limitations and optimal prompting strategies.

> **🔑 Key Insight:** Gemini 2.0+ offers structured object detection with normalized coordinates. For other models, you can prompt for location descriptions, but results are less precise.

### What We'll Cover

- Object detection with bounding boxes
- Image segmentation
- Spatial reasoning and relationships
- Visual question answering patterns
- Known limitations and workarounds

### Prerequisites

- [Image Prompting Fundamentals](./01-image-prompting.md)

---

## Object Detection

### Gemini Object Detection (2.0+)

Gemini 2.0 and later can return bounding boxes in a normalized `[y_min, x_min, y_max, x_max]` format, where coordinates range from 0-1000:

```python
from google import genai
from google.genai import types

client = genai.Client()

# Load image
with open("street_scene.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Detect all vehicles in this image. Return bounding boxes."
    ],
    config={
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_name": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["object_name", "box_2d"]
            }
        }
    }
)

import json
detections = json.loads(response.text)

for obj in detections:
    name = obj["object_name"]
    y_min, x_min, y_max, x_max = obj["box_2d"]
    print(f"{name}: ({x_min}, {y_min}) to ({x_max}, {y_max})")
```

**Output:**
```
car: (120, 340) to (380, 520)
truck: (450, 300) to (720, 550)
motorcycle: (780, 400) to (850, 480)
```

### Converting Normalized Coordinates to Pixels

```python
def normalize_to_pixels(
    box: list[int],
    image_width: int,
    image_height: int
) -> dict:
    """
    Convert Gemini's 0-1000 normalized coordinates to pixels.
    
    Args:
        box: [y_min, x_min, y_max, x_max] in 0-1000 range
        image_width: Original image width in pixels
        image_height: Original image height in pixels
    
    Returns:
        Dictionary with pixel coordinates
    """
    y_min, x_min, y_max, x_max = box
    
    return {
        "x_min": int(x_min * image_width / 1000),
        "y_min": int(y_min * image_height / 1000),
        "x_max": int(x_max * image_width / 1000),
        "y_max": int(y_max * image_height / 1000),
        "width": int((x_max - x_min) * image_width / 1000),
        "height": int((y_max - y_min) * image_height / 1000)
    }

# Example: 1920x1080 image
box = [340, 120, 520, 380]  # [y_min, x_min, y_max, x_max]
pixels = normalize_to_pixels(box, 1920, 1080)

print(f"Top-left: ({pixels['x_min']}, {pixels['y_min']})")
print(f"Size: {pixels['width']} x {pixels['height']}")
```

**Output:**
```
Top-left: (230, 367)
Size: 499 x 194
```

### Visualizing Detection Results

```python
from PIL import Image, ImageDraw, ImageFont
import json

def draw_detections(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes on image."""
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    
    for i, obj in enumerate(detections):
        name = obj["object_name"]
        y_min, x_min, y_max, x_max = obj["box_2d"]
        
        # Convert to pixels
        box = [
            x_min * width / 1000,
            y_min * height / 1000,
            x_max * width / 1000,
            y_max * height / 1000
        ]
        
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1] - 20), name, fill=color)
    
    img.save(output_path)
    return output_path

# Usage
draw_detections("street.jpg", detections, "street_annotated.jpg")
```

### Disabling Thinking for Better Detection

For Gemini 2.5+ with thinking enabled, object detection works better with thinking disabled:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Detect all people in this image. Return bounding boxes."
    ],
    config={
        "thinking_config": {"thinking_budget": 0},  # Disable thinking
        "response_mime_type": "application/json",
        "response_schema": detection_schema
    }
)
```

---

## Image Segmentation

### Gemini Segmentation (2.5+)

Gemini 2.5+ can return segmentation masks as base64-encoded PNG images:

```python
from google import genai
from google.genai import types
import base64
from PIL import Image
from io import BytesIO

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Segment all people in this image. Return masks."
    ],
    config={
        "thinking_config": {"thinking_budget": 0},
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_name": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "mask": {"type": "string"}  # base64 PNG
                },
                "required": ["object_name", "box_2d", "mask"]
            }
        }
    }
)

import json
segments = json.loads(response.text)

for segment in segments:
    name = segment["object_name"]
    mask_b64 = segment["mask"]
    
    # Decode and display mask
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(BytesIO(mask_bytes))
    
    # Mask is same size as bounding box
    print(f"{name}: mask size {mask_img.size}")
```

### Applying Segmentation Masks

```python
def apply_mask_to_image(
    original_image: Image.Image,
    mask: Image.Image,
    box: list[int],
    color: tuple = (255, 0, 0, 128)
) -> Image.Image:
    """
    Apply segmentation mask overlay to original image.
    
    Args:
        original_image: PIL Image of original
        mask: PIL Image of mask (same size as bounding box)
        box: [y_min, x_min, y_max, x_max] in 0-1000 range
        color: RGBA color for overlay
    
    Returns:
        Image with mask overlay
    """
    width, height = original_image.size
    
    # Convert normalized coords to pixels
    y_min, x_min, y_max, x_max = box
    box_px = (
        int(x_min * width / 1000),
        int(y_min * height / 1000),
        int(x_max * width / 1000),
        int(y_max * height / 1000)
    )
    
    box_width = box_px[2] - box_px[0]
    box_height = box_px[3] - box_px[1]
    
    # Resize mask to match bounding box
    mask_resized = mask.resize((box_width, box_height))
    
    # Create colored overlay
    overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    
    # Apply mask as alpha channel
    colored_mask = Image.new('RGBA', (box_width, box_height), color)
    mask_l = mask_resized.convert('L')
    colored_mask.putalpha(mask_l)
    
    overlay.paste(colored_mask, box_px[:2])
    
    # Composite with original
    result = Image.alpha_composite(original_image.convert('RGBA'), overlay)
    
    return result
```

---

## Spatial Reasoning

### Location Questions

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": scene_url, "detail": "high"},
            {
                "type": "input_text",
                "text": """
                Analyze the spatial layout of this room:
                1. What's in the foreground vs background?
                2. What objects are on the left vs right side?
                3. What's near the window?
                4. Describe the arrangement of furniture.
                """
            }
        ]
    }]
)
```

### Relative Position Queries

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": photo_url, "detail": "high"},
            {
                "type": "input_text",
                "text": """
                Looking at the people in this photo:
                1. Who is standing in the center?
                2. Who is to the left of the person in red?
                3. Who is closest to the camera?
                4. Who is in the back row?
                """
            }
        ]
    }]
)
```

### Spatial Reasoning Limitations

> **⚠️ Warning:** Vision models have known limitations with spatial reasoning:

| Task | Accuracy | Notes |
|------|----------|-------|
| Object identification | High | Very reliable |
| Left/right positioning | Medium | Sometimes confused |
| Counting objects | Low-Medium | Approximate for 5+ items |
| Precise distances | Low | Estimates only |
| Rotation/orientation | Medium | May struggle with rotated images |

### Improving Spatial Accuracy

```python
# Technique 1: Ask for step-by-step reasoning
prompt = """
Before answering the spatial question, first:
1. Identify all objects/people in the image
2. Describe their approximate positions
3. Then answer: Who is standing to the left of the red car?
"""

# Technique 2: Use reference points
prompt = """
Using the door as a reference point:
- What is directly to the left of the door?
- What is above the door?
- What is in front of the door?
"""

# Technique 3: Grid-based reasoning
prompt = """
Divide this image into a 3x3 grid.
For each cell (top-left, top-center, top-right, etc.),
list what objects appear in that section.
"""
```

---

## Visual Question Answering

### Basic VQA Patterns

```python
# Yes/No Questions
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": image_url},
            {"type": "input_text", "text": "Is there a dog in this image? Answer only Yes or No."}
        ]
    }]
)

# Counting Questions
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": image_url},
            {
                "type": "input_text",
                "text": "Count the number of chairs visible. If more than 10, say 'more than 10'."
            }
        ]
    }]
)

# Identification Questions
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": image_url},
            {
                "type": "input_text",
                "text": "What brand is the laptop in this image? Look for logos or distinctive design features."
            }
        ]
    }]
)
```

### Structured VQA with Pydantic

```python
from pydantic import BaseModel

class SceneAnalysis(BaseModel):
    setting: str  # indoor, outdoor, etc.
    time_of_day: str | None  # morning, afternoon, night, unknown
    weather: str | None  # if outdoor
    main_subjects: list[str]
    activities: list[str]
    mood: str
    notable_objects: list[str]

response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": image_url, "detail": "high"},
            {"type": "input_text", "text": "Analyze this scene comprehensively."}
        ]
    }],
    text_format=SceneAnalysis
)

scene = response.output_parsed
print(f"Setting: {scene.setting}")
print(f"Main subjects: {', '.join(scene.main_subjects)}")
print(f"Activities: {', '.join(scene.activities)}")
```

### Multi-Turn Visual Conversations

```python
# Initial analysis
messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": image_url, "detail": "high"},
            {"type": "input_text", "text": "Describe this image in detail."}
        ]
    }
]

response = client.responses.create(
    model="gpt-4.1-mini",
    input=messages
)

# Follow-up question (same image in context)
messages.append({"role": "assistant", "content": response.output_text})
messages.append({
    "role": "user",
    "content": [{"type": "input_text", "text": "What color is the car in the background?"}]
})

response = client.responses.create(
    model="gpt-4.1-mini",
    input=messages
)
```

---

## Known Limitations

### What Vision Models Can't Do

| Limitation | Details |
|------------|---------|
| **Medical diagnosis** | Not reliable for clinical interpretation of X-rays, MRIs, etc. |
| **Non-Latin text** | Reduced accuracy for Chinese, Japanese, Korean, Arabic, etc. |
| **Precise counting** | Struggles with 10+ similar objects |
| **Small text** | May miss text under ~12pt without high detail |
| **CAPTCHAs** | Deliberately refuse for safety |
| **Rotated images** | Often misinterpret unless corrected |
| **Faces** | Will not identify specific individuals |

### Working Around Limitations

```python
# Limitation: Counting many objects
# Solution: Ask for estimate ranges

prompt = """
Count the birds in this flock. Provide:
- An estimate if you can't count exactly
- A range (e.g., "approximately 50-75")
- Never state an exact number for groups over 20
"""

# Limitation: Non-Latin text
# Solution: Use specialized OCR first, then analyze

import pytesseract
from PIL import Image

def extract_multilang_text(image_path: str) -> str:
    """Use Tesseract for non-Latin OCR before sending to vision model."""
    img = Image.open(image_path)
    # Use appropriate language pack
    text = pytesseract.image_to_string(img, lang='chi_sim+jpn+kor')
    return text

# Then include extracted text with image
extracted = extract_multilang_text("document.png")
prompt = f"""
This document contains the following text (from OCR):
{extracted}

Based on the image and this text, summarize the document's contents.
"""

# Limitation: Rotated images
# Solution: Preprocess to correct orientation

from PIL import Image

def auto_rotate(image_path: str) -> Image.Image:
    """Attempt to auto-correct rotation using EXIF data."""
    img = Image.open(image_path)
    
    # Try to get EXIF orientation
    try:
        from PIL.ExifTags import TAGS
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Orientation':
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass
    
    return img
```

---

## Common Mistakes

### ❌ Relying on Precise Counts

```python
# Bad: Expecting exact count
prompt = "Exactly how many people are in this crowd?"

# Good: Asking for estimates
prompt = "Estimate the number of people in this crowd. Provide a range."
```

### ❌ Asking for Face Identification

```python
# Bad: Identifying individuals
prompt = "Who is this person?"

# Good: Describing attributes
prompt = "Describe the person's apparent age, clothing, and expression."
```

### ❌ Expecting Medical Diagnosis

```python
# Bad: Clinical interpretation
prompt = "Does this X-ray show pneumonia?"

# Good: Educational description
prompt = "Describe what's visible in this chest X-ray image for educational purposes. Note: This is not a medical diagnosis."
```

---

## Hands-on Exercise

### Your Task

Build an inventory scanner that:
1. Takes an image of a shelf or display
2. Detects visible products
3. Returns estimated counts and locations
4. Identifies any empty shelf spaces

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from google import genai
from google.genai import types
from pydantic import BaseModel
import json

class ProductDetection(BaseModel):
    product_name: str
    brand: str | None
    estimated_count: int
    count_confidence: str  # exact, approximate, rough estimate
    position: str  # description like "top shelf, left side"
    box_2d: list[int] | None

class ShelfAnalysis(BaseModel):
    total_products_detected: int
    products: list[ProductDetection]
    empty_spaces: list[str]  # descriptions of gaps
    shelf_fullness_percent: int
    notes: list[str]

def analyze_shelf_inventory(image_path: str) -> ShelfAnalysis:
    """Analyze shelf inventory from image."""
    
    client = genai.Client()
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # First pass: detect and locate products
    detection_response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            """
            Detect all products on this shelf/display.
            For each product:
            1. Identify the product type and brand if visible
            2. Estimate the count (use ranges for stacks)
            3. Note the position on shelf
            4. Return bounding box coordinates
            
            Also identify any empty spaces where products could be stocked.
            """
        ],
        config={
            "thinking_config": {"thinking_budget": 0},
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "total_products_detected": {"type": "integer"},
                    "products": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_name": {"type": "string"},
                                "brand": {"type": "string"},
                                "estimated_count": {"type": "integer"},
                                "count_confidence": {"type": "string"},
                                "position": {"type": "string"},
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            }
                        }
                    },
                    "empty_spaces": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "shelf_fullness_percent": {"type": "integer"},
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    )
    
    data = json.loads(detection_response.text)
    
    return ShelfAnalysis(**data)

# Usage
analysis = analyze_shelf_inventory("store_shelf.jpg")

print(f"Products detected: {analysis.total_products_detected}")
print(f"Shelf fullness: {analysis.shelf_fullness_percent}%")

print("\nProducts:")
for product in analysis.products:
    count_str = f"{product.estimated_count} ({product.count_confidence})"
    print(f"  - {product.product_name}: {count_str} at {product.position}")

if analysis.empty_spaces:
    print("\nEmpty spaces:")
    for space in analysis.empty_spaces:
        print(f"  - {space}")
```

</details>

---

## Summary

✅ **Gemini 2.0+ offers structured detection:** Bounding boxes with normalized 0-1000 coordinates
✅ **Gemini 2.5+ adds segmentation:** Base64-encoded masks within bounding boxes
✅ **Disable thinking for detection:** `thinking_budget: 0` improves accuracy
✅ **Spatial reasoning is limited:** Use step-by-step prompting for better results
✅ **Know the limitations:** Counting, non-Latin text, medical images are problematic

**Next:** [Video and Audio Prompting](./04-video-audio-prompting.md)

---

## Further Reading

- [Gemini Object Detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection)
- [Gemini Segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation)
- [OpenAI Vision Best Practices](https://platform.openai.com/docs/guides/vision#best-practices)

---

<!-- 
Sources Consulted:
- Gemini Image Understanding: box_2d format, 0-1000 normalization, segmentation masks
- OpenAI Vision Guide: Limitations, spatial reasoning challenges
- Anthropic Vision: Image positioning best practices
-->
