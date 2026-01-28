---
title: "Image Understanding Models"
---

# Image Understanding Models

## Introduction

Image understanding models analyze and describe images. Integrated into LLMs like GPT-4V and Claude Vision, these models enable image analysis, OCR, and visual reasoning.

### What We'll Cover

- Vision capabilities in LLMs
- Image analysis and description
- OCR and text extraction
- Multi-image comparison

---

## Vision-Capable LLMs

### GPT-4 Vision

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_image_url(image_url: str, prompt: str) -> str:
    """Analyze image from URL"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        max_tokens=500
    )
    
    return response.choices[0].message.content

def analyze_image_file(image_path: str, prompt: str) -> str:
    """Analyze image from local file"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Example
description = analyze_image_url(
    "https://example.com/photo.jpg",
    "Describe this image in detail"
)
```

### Claude Vision

```python
from anthropic import Anthropic
import base64

client = Anthropic()

def analyze_with_claude(image_path: str, prompt: str) -> str:
    """Analyze image with Claude"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Determine media type
    if image_path.endswith(".png"):
        media_type = "image/png"
    elif image_path.endswith(".gif"):
        media_type = "image/gif"
    else:
        media_type = "image/jpeg"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    return response.content[0].text
```

### Gemini Vision

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR_KEY")

def analyze_with_gemini(image_path: str, prompt: str) -> str:
    """Analyze image with Gemini"""
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    image = Image.open(image_path)
    
    response = model.generate_content([prompt, image])
    
    return response.text
```

---

## Image Analysis Tasks

### General Description

```python
def describe_image(image_url: str) -> dict:
    """Get comprehensive image description"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this image and provide:
1. Brief description (1-2 sentences)
2. Main subjects/objects
3. Colors and mood
4. Notable details
5. Possible context/setting

Format as JSON."""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

### Object Detection Style

```python
def identify_objects(image_url: str) -> list:
    """Identify objects in image"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """List all distinct objects visible in this image.
For each object, provide:
- name
- approximate location (top-left, center, etc.)
- estimated size relative to image (small/medium/large)
- confidence (high/medium/low)

Return as JSON array."""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

---

## OCR and Text Extraction

### Basic Text Extraction

```python
def extract_text(image_path: str) -> str:
    """Extract all visible text from image"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract ALL text visible in this image. Maintain formatting where possible."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]
    )
    
    return response.choices[0].message.content
```

### Structured Document Extraction

```python
def extract_receipt(image_path: str) -> dict:
    """Extract receipt information"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Extract receipt information:
- Store name
- Date
- Items (name, quantity, price)
- Subtotal
- Tax
- Total
- Payment method

Return as JSON."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
                }
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)

def extract_business_card(image_path: str) -> dict:
    """Extract business card information"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Extract business card information:
- Name
- Title
- Company
- Email
- Phone
- Address
- Website
- Social media

Return as JSON."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
                }
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

---

## Diagram Understanding

### Chart Analysis

```python
def analyze_chart(image_url: str) -> dict:
    """Analyze chart or graph"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this chart/graph:
1. Chart type (bar, line, pie, etc.)
2. Title and axis labels
3. Data points or values (as many as you can read)
4. Key insights/trends
5. Data source (if visible)

Return as JSON."""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

### Flowchart/Diagram Parsing

```python
def parse_diagram(image_url: str) -> dict:
    """Parse flowchart or diagram"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Parse this diagram:
1. Diagram type (flowchart, architecture, UML, etc.)
2. All nodes/boxes and their labels
3. All connections/arrows between nodes
4. Overall purpose/meaning

Return as JSON with nodes and edges arrays."""
                },
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

---

## Multi-Image Comparison

### Compare Two Images

```python
def compare_images(image1_url: str, image2_url: str) -> dict:
    """Compare two images"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Compare these two images:
1. Similarities
2. Differences
3. Which is better for [quality/clarity/composition]
4. Any notable observations

Return as JSON."""
                },
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}}
            ]
        }],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

### Before/After Analysis

```python
def analyze_before_after(before_url: str, after_url: str, context: str) -> str:
    """Analyze before/after images"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""These are before (first) and after (second) images.
Context: {context}

Analyze:
1. What changed
2. Quality of the changes
3. Any issues or concerns
4. Overall assessment"""
                },
                {"type": "image_url", "image_url": {"url": before_url}},
                {"type": "image_url", "image_url": {"url": after_url}}
            ]
        }]
    )
    
    return response.choices[0].message.content
```

---

## Hands-on Exercise

### Your Task

Build an image analysis toolkit:

```python
from openai import OpenAI
import base64

client = OpenAI()

class ImageAnalyzer:
    """Comprehensive image analysis toolkit"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def describe(self, image_path: str) -> str:
        """Get general description"""
        return self._analyze(image_path, "Describe this image in detail.")
    
    def extract_text(self, image_path: str) -> str:
        """Extract all text"""
        return self._analyze(image_path, "Extract all visible text.")
    
    def identify_objects(self, image_path: str) -> list:
        """List all objects"""
        result = self._analyze(
            image_path,
            "List all objects as JSON array: [{\"object\": \"...\", \"location\": \"...\"}]"
        )
        import json
        try:
            return json.loads(result)
        except:
            return [{"raw": result}]
    
    def analyze_for_task(self, image_path: str, task: str) -> str:
        """Custom analysis for specific task"""
        return self._analyze(image_path, task)
    
    def _analyze(self, image_path: str, prompt: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"
                        }
                    }
                ]
            }]
        )
        return response.choices[0].message.content

# Test with a sample image
analyzer = ImageAnalyzer()

# Save a test image first, then:
# print(analyzer.describe("test_image.jpg"))
# print(analyzer.extract_text("document.png"))
# print(analyzer.identify_objects("photo.jpg"))
```

---

## Summary

✅ **GPT-4V, Claude Vision, Gemini** all support images

✅ **Base64 or URL** for image input

✅ **OCR** extracts text from documents

✅ **Diagram understanding** parses charts and flowcharts

✅ **Multi-image** enables comparison and analysis

**Next:** [Audio Models](./09-audio-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Image Generation Models](./07-image-generation-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Audio Models](./09-audio-models.md) |

