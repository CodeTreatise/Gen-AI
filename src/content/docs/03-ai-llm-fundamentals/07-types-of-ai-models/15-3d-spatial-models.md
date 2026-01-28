---
title: "3D & Spatial Models (Emerging)"
---

# 3D & Spatial Models (Emerging)

## Introduction

3D and spatial AI models are an emerging field enabling 3D object generation, scene understanding, and spatial reasoning. While still developing, these capabilities are rapidly advancing.

### What We'll Cover

- 3D object generation
- Scene understanding
- Spatial reasoning
- Current state and future directions

---

## 3D Object Generation

### Text-to-3D

```python
# Conceptual API - actual implementations vary
def generate_3d_object(prompt: str) -> bytes:
    """Generate 3D model from text description"""
    
    response = requests.post(
        "https://api.3d-generator.com/generate",
        json={
            "prompt": prompt,
            "format": "glb",  # glTF binary
            "quality": "high",
            "texture": True
        }
    )
    
    return response.content  # 3D model file

# Example
model = generate_3d_object("A wooden chair with curved armrests")
with open("chair.glb", "wb") as f:
    f.write(model)
```

### Current Tools

| Tool | Type | Status |
|------|------|--------|
| Point-E (OpenAI) | Text/Image to 3D | Research |
| Shap-E (OpenAI) | Text to 3D | Research |
| DreamFusion | Text to 3D | Research |
| GET3D (NVIDIA) | Image to 3D | Research |
| Meshy | Text to 3D | Commercial |
| Luma AI | Image to 3D | Commercial |

### Image-to-3D

```python
def image_to_3d(image_path: str) -> bytes:
    """Convert 2D image to 3D model"""
    
    # Using Luma AI (example)
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://api.lumalabs.ai/v1/3d",
            files={"image": f},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
    
    return response.content

# Create 3D from product photo
model = image_to_3d("product_photo.jpg")
```

---

## Scene Understanding

### 3D Scene Analysis

```python
def analyze_3d_scene(point_cloud: bytes) -> dict:
    """Analyze 3D scene from point cloud data"""
    
    # Hypothetical API
    response = requests.post(
        "https://api.scene-ai.com/analyze",
        files={"data": point_cloud},
        json={
            "detect_objects": True,
            "estimate_dimensions": True,
            "classify_room": True
        }
    )
    
    return response.json()

# Result example:
# {
#     "room_type": "living_room",
#     "objects": [
#         {"type": "sofa", "dimensions": [2.1, 0.9, 0.8], "position": [1.5, 2.0, 0]},
#         {"type": "table", "dimensions": [1.2, 0.6, 0.5], "position": [2.5, 2.0, 0]}
#     ],
#     "floor_area_sqm": 25.5
# }
```

### Depth Estimation

```python
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image

def estimate_depth(image_path: str) -> np.ndarray:
    """Estimate depth from 2D image"""
    
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Normalize to 0-255 for visualization
    depth_map = predicted_depth.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map
```

### NeRF (Neural Radiance Fields)

```python
# NeRF creates 3D scenes from 2D images
def create_nerf_scene(images: list, camera_poses: list):
    """Create NeRF scene from images"""
    
    # Using Luma AI or similar
    response = requests.post(
        "https://api.lumalabs.ai/v1/nerf",
        json={
            "images": [encode_base64(img) for img in images],
            "camera_poses": camera_poses
        }
    )
    
    return response.json()["scene_id"]

# View from any angle after creation
def render_nerf_view(scene_id: str, camera_position: list, camera_target: list):
    """Render novel view from NeRF scene"""
    
    response = requests.post(
        f"https://api.lumalabs.ai/v1/nerf/{scene_id}/render",
        json={
            "position": camera_position,
            "target": camera_target,
            "resolution": [1920, 1080]
        }
    )
    
    return response.content  # Rendered image
```

---

## Spatial Reasoning

### LLMs with Spatial Understanding

```python
from openai import OpenAI

client = OpenAI()

def spatial_reasoning(scene_description: str, question: str) -> str:
    """Answer spatial questions about a scene"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": """You are an expert at spatial reasoning. 
When given a scene description, you can:
- Understand relative positions
- Calculate distances
- Predict movements
- Identify spatial relationships"""
        }, {
            "role": "user",
            "content": f"""Scene: {scene_description}

Question: {question}"""
        }]
    )
    
    return response.choices[0].message.content

# Example
scene = """
A rectangular room 5m x 4m. 
A desk is against the north wall, 1m from the corner.
A chair is in front of the desk.
The door is on the south wall, centered.
"""

answer = spatial_reasoning(scene, "If I enter the room, what's the shortest path to the desk?")
```

### Visual Spatial Reasoning

```python
def analyze_spatial_image(image_path: str, question: str) -> str:
    """Spatial reasoning from image"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Spatial question: {question}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]
    )
    
    return response.choices[0].message.content

# Example
answer = analyze_spatial_image(
    "room_photo.jpg",
    "Would a 2-meter sofa fit between the window and the door?"
)
```

---

## Current Capabilities

### What Works

| Capability | Quality | Notes |
|------------|---------|-------|
| Simple 3D shapes | ⭐⭐⭐⭐ | Good |
| Furniture/objects | ⭐⭐⭐ | Decent |
| Characters | ⭐⭐ | Improving |
| Textures | ⭐⭐⭐ | Variable |
| Animation | ⭐⭐ | Limited |
| Physics | ⭐ | Early stage |

### Limitations

```python
current_limitations = {
    "geometry_accuracy": "Complex shapes often deformed",
    "texture_quality": "Can be blurry or inconsistent",
    "generation_time": "Can take minutes to hours",
    "format_compatibility": "Various formats, conversion needed",
    "fine_control": "Limited control over output",
    "consistency": "Hard to generate consistent series",
}
```

---

## Practical Applications

### E-commerce Product Visualization

```python
def create_product_3d_view(product_images: list) -> str:
    """Create 3D view from product photos"""
    
    # Combine multiple angles into 3D model
    response = requests.post(
        "https://api.lumalabs.ai/v1/capture",
        json={
            "images": [encode_base64(img) for img in product_images],
            "type": "object"
        }
    )
    
    return response.json()["model_url"]
```

### Architecture/Interior Design

```python
def generate_room_layout(description: str) -> dict:
    """Generate 3D room layout from description"""
    
    # Use LLM for layout planning
    layout_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Create a room layout in JSON format for:
{description}

Include: room dimensions, furniture positions, materials"""
        }],
        response_format={"type": "json_object"}
    )
    
    layout = json.loads(layout_response.choices[0].message.content)
    
    # Could then generate 3D visualization
    return layout
```

### Gaming/VR Assets

```python
def generate_game_asset(description: str, style: str = "realistic"):
    """Generate 3D game asset"""
    
    # Using Meshy or similar
    response = requests.post(
        "https://api.meshy.ai/v1/generate",
        json={
            "prompt": description,
            "style": style,  # realistic, cartoon, low-poly
            "rigged": True,  # Include animation rig
            "pbr_textures": True  # PBR material textures
        }
    )
    
    return response.json()
```

---

## Future Directions

### Emerging Capabilities

```python
future_capabilities = {
    "2025-2026": [
        "More accurate text-to-3D",
        "Real-time generation",
        "Better physics simulation",
        "Improved texture quality",
    ],
    "2027+": [
        "Full scene generation",
        "Animation from text",
        "Interactive 3D worlds",
        "Seamless AR/VR integration",
    ]
}
```

### Research Areas

| Area | Description |
|------|-------------|
| Gaussian Splatting | Faster, higher quality 3D reconstruction |
| Diffusion 3D | Applying diffusion models to 3D |
| Physics-aware | Understanding physical properties |
| Multi-view consistency | Consistent objects from all angles |

---

## Hands-on Exercise

### Your Task

Explore spatial reasoning with current tools:

```python
from openai import OpenAI
import json

client = OpenAI()

class SpatialReasoner:
    """Spatial reasoning assistant"""
    
    def describe_scene(self, image_description: str) -> dict:
        """Create structured scene representation"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Convert this scene description to structured JSON:
                
{image_description}

Include:
- room_dimensions (if inferable)
- objects: list with name, position, dimensions
- relationships: spatial relationships between objects"""
            }],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def answer_spatial_question(self, scene: dict, question: str) -> str:
        """Answer question about scene"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Scene: {json.dumps(scene)}

Question: {question}

Reason step by step about spatial relationships."""
            }]
        )
        
        return response.choices[0].message.content
    
    def suggest_placement(self, scene: dict, new_object: str) -> str:
        """Suggest where to place new object"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Scene: {json.dumps(scene)}

Where should I place: {new_object}

Consider:
- Available space
- Logical placement
- Accessibility"""
            }]
        )
        
        return response.choices[0].message.content

# Test
reasoner = SpatialReasoner()

scene = reasoner.describe_scene("""
A bedroom with a queen bed centered on the north wall.
A nightstand on each side of the bed.
A dresser against the east wall.
The closet door is on the west wall.
""")

print("Scene:", json.dumps(scene, indent=2))

answer = reasoner.answer_spatial_question(
    scene,
    "Is there room for a desk between the dresser and the closet?"
)
print("Answer:", answer)
```

---

## Summary

✅ **3D generation**: Emerging but improving rapidly

✅ **Scene understanding**: Depth estimation, object detection

✅ **Spatial reasoning**: LLMs show promising capabilities

✅ **Current limits**: Quality, speed, fine control

✅ **Future**: Real-time generation, physics, full scenes

**Lesson Complete!** Return to [Types of AI Models Overview](./00-types-of-ai-models.md)

---

## Navigation

| Previous | Up | Next Lesson |
|----------|-------|------|
| [Computer Use Models](./14-computer-use-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Model Selection Criteria](../08-model-selection-criteria/00-model-selection-criteria.md) |

<!-- 
Sources Consulted:
- OpenAI Point-E: https://github.com/openai/point-e
- Luma AI: https://lumalabs.ai/
- Intel DPT: https://huggingface.co/Intel/dpt-large
-->

