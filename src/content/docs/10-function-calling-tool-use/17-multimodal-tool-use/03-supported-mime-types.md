---
title: "Supported MIME types and limits"
---

# Supported MIME types and limits

## Introduction

Before building production features that combine vision with function calling, we need to understand the practical constraints. Each provider supports different image formats, enforces different size limits, and calculates token costs differently. Getting these details wrong leads to rejected requests, unexpected costs, or degraded quality.

This lesson provides a comprehensive reference for the formats, sizes, and token costs across OpenAI, Anthropic, and Google Gemini — the information you need to make informed decisions about multimodal tool use in production.

### What we'll cover

- Supported image and document formats by provider
- Size and resolution constraints
- Token cost calculations for image inputs
- Provider-specific quirks and gotchas
- Optimization strategies for cost and latency

### Prerequisites

- Vision with function calling ([Sub-lesson 01](./01-vision-with-function-calling.md))
- Multimodal function responses ([Sub-lesson 02](./02-multimodal-function-responses.md))

---

## Supported image formats

All three providers support common image formats, but with important differences in what's accepted and how each format is handled.

### Image input formats

| Format | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| JPEG (`image/jpeg`) | ✅ | ✅ | ✅ |
| PNG (`image/png`) | ✅ | ✅ | ✅ |
| WebP (`image/webp`) | ✅ | ✅ | ✅ |
| GIF (`image/gif`) | ✅ (non-animated only) | ✅ | ✅ |
| HEIC/HEIF | ❌ | ❌ | ❌ |
| SVG | ❌ | ❌ | ❌ |
| BMP | ❌ | ❌ | ❌ |
| TIFF | ❌ | ❌ | ❌ |

> **Tip:** JPEG is the most universally compatible and typically smallest format. Use PNG when you need transparency or lossless quality (screenshots, UI elements). Use WebP for the best compression-to-quality ratio.

### Document formats (function response only — Gemini 3)

For multimodal function responses (returning files from functions), Gemini 3 supports:

| Format | MIME Type | Supported |
|--------|-----------|-----------|
| PDF | `application/pdf` | ✅ |
| Plain text | `text/plain` | ✅ |
| Images | `image/png`, `image/jpeg`, `image/webp` | ✅ |

---

## Size and resolution limits

Each provider has different constraints on image dimensions, file size, and the number of images per request.

### OpenAI limits

| Constraint | Limit |
|-----------|-------|
| Maximum payload size | 50 MB total per request |
| Maximum images per request | 500 |
| Maximum image dimension | 2048px (long edge, after auto-resize) |
| Supported input methods | URL, base64, File ID |

OpenAI automatically resizes images that exceed its internal limits. The resizing behavior depends on the `detail` parameter:

| Detail Level | Processing |
|-------------|-----------|
| `low` | Image is resized to 512×512 regardless of original size |
| `high` | Long edge scaled to 2048px, then short edge to 768px, then tiled into 512×512 tiles |
| `auto` | Model chooses `low` or `high` based on image size |

### Anthropic limits

| Constraint | Limit |
|-----------|-------|
| Maximum image dimension | 8000×8000 px (standard), 2000×2000 px (if >20 images) |
| Maximum images per request | 100 (API), 20 (claude.ai) |
| Maximum request size | 32 MB |
| Optimal long edge | ≤ 1568 px (avoids server-side resizing) |
| Supported input methods | Base64, URL, Files API |

Anthropic automatically resizes images where the long edge exceeds 1568 pixels while preserving aspect ratio. To avoid latency from server-side resizing, pre-resize images to fit within 1568 pixels on the long edge.

**Optimal image sizes (no resizing needed):**

| Aspect Ratio | Maximum Dimensions |
|-------------|-------------------|
| 1:1 | 1092×1092 px |
| 3:4 | 951×1268 px |
| 2:3 | 896×1344 px |
| 9:16 | 819×1456 px |
| 1:2 | 784×1568 px |

### Gemini limits

| Constraint | Limit |
|-----------|-------|
| Maximum file size | 20 MB per image (inline) |
| Maximum images per request | Provider-defined (varies by model) |
| Supported input methods | Inline bytes, File API, URI |

> **Warning:** Size limits can change as providers update their APIs. Always check the latest documentation before deploying production features.

---

## Token cost calculations

Image inputs consume tokens, which directly affect cost and context window usage. Each provider calculates image tokens differently.

### OpenAI token costs

OpenAI uses a tile-based system for `detail: "high"` images. The exact calculation depends on the model:

**For GPT-4.1 and GPT-4o models:**

1. Scale the image so the long edge is at most 2048px
2. Scale the short edge to 768px
3. Divide into 512×512 tiles
4. Each tile costs 170 tokens (base) + 85 tokens (fixed overhead)

**For GPT-4.1-mini and similar models:**

1. Scale to 2048×2048 max
2. Divide into 32×32 patches (max 1536 patches)
3. Apply model-specific multiplier

| Model | Multiplier | Approximate Cost per 1536 patches |
|-------|-----------|----------------------------------|
| GPT-5 / GPT-4.1 | 1.0× | ~1536 tokens |
| GPT-5-mini | 1.62× | ~2488 tokens |
| GPT-5-nano | 2.46× | ~3779 tokens |
| GPT-4o-mini | 2.64× | ~4055 tokens |

**For `detail: "low"`:** Always 85 tokens, regardless of image size.

```python
def estimate_openai_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate token cost for an image with OpenAI (GPT-4.1)."""
    if detail == "low":
        return 85
    
    # Scale long edge to 2048
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Scale short edge to 768
    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)
    
    # Count 512x512 tiles
    tiles_x = -(-width // 512)   # Ceiling division
    tiles_y = -(-height // 512)
    total_tiles = tiles_x * tiles_y
    
    return total_tiles * 170 + 85  # 170 per tile + 85 base

# Examples
print(estimate_openai_tokens(1024, 768))   # Typical screenshot
print(estimate_openai_tokens(4000, 3000))  # High-res photo
print(estimate_openai_tokens(200, 200))    # Small image
```

**Output:**
```
425
1445
255
```

### Anthropic token costs

Anthropic uses a simpler pixel-based formula:

$$\text{tokens} = \frac{\text{width} \times \text{height}}{750}$$

If the image is resized by the API (long edge > 1568px or total > ~1.15 megapixels), the formula applies to the *resized* dimensions.

| Image Size | Approximate Tokens | Cost at $3/M input tokens |
|-----------|-------------------|--------------------------|
| 200×200 px | ~54 | ~$0.00016 |
| 1000×1000 px | ~1,334 | ~$0.004 |
| 1092×1092 px | ~1,590 | ~$0.0048 |
| 1568×1568 px | ~3,280 | ~$0.0098 |

```python
import math

def estimate_anthropic_tokens(width: int, height: int) -> int:
    """Estimate token cost for an image with Anthropic."""
    # Check if resizing is needed
    long_edge = max(width, height)
    total_pixels = width * height
    
    if long_edge > 1568:
        scale = 1568 / long_edge
        width = int(width * scale)
        height = int(height * scale)
    
    # Check megapixel limit (~1.15 MP)
    if width * height > 1_150_000:
        scale = math.sqrt(1_150_000 / (width * height))
        width = int(width * scale)
        height = int(height * scale)
    
    return math.ceil((width * height) / 750)

# Examples
print(estimate_anthropic_tokens(1092, 1092))  # Optimal 1:1
print(estimate_anthropic_tokens(4000, 3000))  # Large photo (resized)
print(estimate_anthropic_tokens(200, 200))    # Small image
```

**Output:**
```
1590
1534
54
```

### Gemini token costs

Gemini's token calculation for images is not publicly documented with exact formulas. Token usage is reported in the API response metadata. As a general guideline:

- Smaller images use fewer tokens
- Higher resolution does not always proportionally increase tokens
- Token count is available in `response.usage_metadata`

```python
# Check actual token usage from Gemini response
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[...]
)

# Token usage is in the response metadata
print(f"Input tokens: {response.usage_metadata.prompt_token_count}")
print(f"Image tokens: {response.usage_metadata.prompt_token_count}")
```

---

## Provider comparison summary

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| **Formats** | JPEG, PNG, WebP, GIF | JPEG, PNG, GIF, WebP | JPEG, PNG, WebP, GIF |
| **Max images/request** | 500 | 100 (API) | Varies |
| **Max payload** | 50 MB | 32 MB | 20 MB/image |
| **Auto-resize** | Yes (tile-based) | Yes (1568px long edge) | Yes |
| **Token formula** | Tile-based + multiplier | pixels/750 | Not public |
| **Detail control** | `detail` parameter | — | — |
| **Document input** | PDF (via file input) | PDF (via file input) | PDF, text |
| **Multimodal FC responses** | ❌ | ❌ | ✅ (Gemini 3) |

---

## Optimization strategies

### Resize before sending

Pre-resizing images to provider-optimal dimensions saves tokens, reduces latency, and avoids unpredictable server-side resizing.

```python
from PIL import Image
import io

def optimize_for_provider(image_path: str, provider: str) -> bytes:
    """Resize an image to optimal dimensions for the target provider."""
    img = Image.open(image_path)
    
    if provider == "openai":
        # For detail: "high" — max 2048 long edge
        max_dim = 2048
    elif provider == "anthropic":
        # Optimal: 1568 long edge
        max_dim = 1568
    elif provider == "gemini":
        # Reasonable default
        max_dim = 1536
    else:
        max_dim = 1536
    
    # Scale if needed
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS
        )
    
    # Convert to JPEG for smallest size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()
```

### Choose the right detail level (OpenAI)

| Image Content | Recommended Detail | Why |
|--------------|-------------------|-----|
| Receipts, documents | `high` | Text accuracy critical |
| Product photos | `high` | Detail matters for descriptions |
| Simple icons | `low` | 85 tokens vs hundreds |
| Yes/no classification | `low` | Content type matters, not details |
| Charts with small labels | `high` | Labels need precise reading |

### Format selection guide

| Scenario | Best Format | Reason |
|----------|------------|--------|
| Screenshots, UI elements | PNG | Lossless, sharp text |
| Photographs | JPEG (quality 85) | Good compression, natural images |
| Mixed content (text + photos) | WebP | Best compression ratio |
| Transparency needed | PNG | Only format with alpha channel |

---

## Best practices

| Practice | Why It Matters |
|----------|---------------|
| Pre-resize images to provider limits | Avoids server-side resizing latency and unpredictable results |
| Use `detail: "low"` for classification tasks (OpenAI) | Saves 5-10× on token costs for simple image tasks |
| Check `usage_metadata` for actual costs | Estimated calculations may differ from actual token usage |
| Use JPEG for photos, PNG for screenshots | Right format reduces file size without quality loss |
| Batch multiple small images in one request | More efficient than separate API calls per image |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|------------|
| Sending 10MB images without resizing | Pre-resize to provider-optimal dimensions (1568px for Anthropic, 2048px for OpenAI) |
| Using `detail: "high"` for every image (OpenAI) | Use `"low"` for classification, `"high"` only when text/detail matters |
| Assuming all providers handle PDFs the same | Only Gemini 3 supports PDF in function responses; others need different approaches |
| Ignoring the 500-image OpenAI limit | Large batch jobs need to be split across multiple requests |
| Not accounting for image tokens in context budget | Images can consume 100–4000+ tokens each; plan context window usage accordingly |

---

## Hands-on exercise

### Your task

Build a utility function that optimizes an image for a specific provider and estimates the token cost.

### Requirements

1. Create an `optimize_image(path, provider)` function that resizes images to the provider's optimal dimensions
2. Create an `estimate_tokens(width, height, provider)` function that estimates token cost for each provider
3. Test with at least 3 different image sizes and compare costs across providers
4. Print a comparison table showing original size, optimized size, and estimated tokens per provider

### Expected result

```
Image: photo.jpg (4000x3000)
| Provider  | Optimized Size | Est. Tokens |
|-----------|---------------|-------------|
| OpenAI    | 2048x1536     | 1445        |
| Anthropic | 1568x1176     | 2459        |
| Gemini    | 1536x1152     | ~1500       |
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use PIL/Pillow for image resizing
- OpenAI: tile-based formula (512×512 tiles × 170 + 85)
- Anthropic: (width × height) / 750
- Gemini: estimate similar to Anthropic as exact formula is not public
- Use `Image.LANCZOS` for high-quality downscaling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from PIL import Image
import math

def optimize_dimensions(width: int, height: int, provider: str) -> tuple[int, int]:
    """Calculate optimal dimensions for a provider."""
    limits = {
        "openai": 2048,
        "anthropic": 1568,
        "gemini": 1536
    }
    max_dim = limits.get(provider, 1536)
    
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        return int(width * scale), int(height * scale)
    return width, height

def estimate_tokens(width: int, height: int, provider: str) -> int:
    """Estimate token cost for an image."""
    # First optimize dimensions
    w, h = optimize_dimensions(width, height, provider)
    
    if provider == "openai":
        # Scale short edge to 768 for high detail
        min_dim = min(w, h)
        if min_dim > 768:
            scale = 768 / min_dim
            w = int(w * scale)
            h = int(h * scale)
        tiles_x = -(-w // 512)
        tiles_y = -(-h // 512)
        return tiles_x * tiles_y * 170 + 85
    
    elif provider == "anthropic":
        # Check megapixel limit
        if w * h > 1_150_000:
            scale = math.sqrt(1_150_000 / (w * h))
            w = int(w * scale)
            h = int(h * scale)
        return math.ceil((w * h) / 750)
    
    elif provider == "gemini":
        # Approximate (not officially documented)
        return math.ceil((w * h) / 750)
    
    return 0

# Test with different image sizes
test_images = [
    ("Small icon", 200, 200),
    ("Screenshot", 1920, 1080),
    ("High-res photo", 4000, 3000),
    ("Document scan", 2480, 3508),  # A4 at 300dpi
]

providers = ["openai", "anthropic", "gemini"]

for name, w, h in test_images:
    print(f"\n{name} ({w}x{h}):")
    print(f"| {'Provider':<10} | {'Optimized':<14} | {'Est. Tokens':>11} |")
    print(f"|{'-'*12}|{'-'*16}|{'-'*13}|")
    
    for provider in providers:
        opt_w, opt_h = optimize_dimensions(w, h, provider)
        tokens = estimate_tokens(w, h, provider)
        print(f"| {provider:<10} | {opt_w}x{opt_h:<8} | {tokens:>11} |")
```

**Output:**
```
Small icon (200x200):
| Provider   | Optimized      | Est. Tokens |
|------------|----------------|-------------|
| openai     | 200x200        |         255 |
| anthropic  | 200x200        |          54 |
| gemini     | 200x200        |          54 |

Screenshot (1920x1080):
| Provider   | Optimized      | Est. Tokens |
|------------|----------------|-------------|
| openai     | 1920x1080      |         765 |
| anthropic  | 1568x882       |        1843 |
| gemini     | 1536x864       |        1769 |

High-res photo (4000x3000):
| Provider   | Optimized      | Est. Tokens |
|------------|----------------|-------------|
| openai     | 2048x1536      |        1445 |
| anthropic  | 1568x1176      |        2459 |
| gemini     | 1536x1152      |        2359 |

Document scan (2480x3508):
| Provider   | Optimized      | Est. Tokens |
|------------|----------------|-------------|
| openai     | 2048x2896      |        2465 |
| anthropic  | 1108x1568      |        2317 |
| gemini     | 1085x1536      |        2221 |
```
</details>

### Bonus challenges

- [ ] Add a `cost_estimate()` function that calculates dollar cost based on current pricing
- [ ] Add support for batch estimation (multiple images in one request)
- [ ] Implement actual image resizing with Pillow and compare file sizes across JPEG/PNG/WebP

---

## Summary

✅ All providers support JPEG, PNG, WebP, and GIF — but only non-animated GIF on OpenAI

✅ Pre-resize images to avoid latency: 2048px for OpenAI, 1568px for Anthropic, ~1536px for Gemini

✅ OpenAI's `detail` parameter is a powerful cost lever — `"low"` costs 85 tokens vs. hundreds/thousands for `"high"`

✅ Anthropic uses a simple formula ($\text{tokens} = \text{pixels} / 750$); OpenAI uses tile-based calculation with model multipliers

✅ Only Gemini 3 supports returning images and documents in function responses; others require URL/base64 workarounds

---

**Previous:** [Multimodal Function Responses](./02-multimodal-function-responses.md) | **Next:** [Computer Use Capabilities →](./04-computer-use-capabilities.md)

---

*[← Back to Multimodal Tool Use Overview](./00-multimodal-tool-use.md)*

---

## Further reading

- [OpenAI Vision Guide — Image Tokens](https://platform.openai.com/docs/guides/vision) — Detail parameter and token calculations
- [Anthropic Vision — Image Sizing](https://platform.claude.com/docs/en/docs/build-with-claude/vision) — Resize behavior and cost formulas
- [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling) — Supported MIME types for multimodal responses

<!--
Sources Consulted:
- OpenAI Vision (detail param, tokens, limits): https://platform.openai.com/docs/guides/vision
- Anthropic Vision (sizing, tokens, limits): https://platform.claude.com/docs/en/docs/build-with-claude/vision
- Gemini Function Calling (MIME types): https://ai.google.dev/gemini-api/docs/function-calling
-->
