---
title: "Response Metadata"
---

# Response Metadata

## Introduction

Beyond the main content, API responses include valuable metadata: annotations, refusals, service tier information, and more. Understanding these fields helps build more robust applications.

### What We'll Cover

- Annotations field in messages
- Refusal handling
- Service tier information
- System fingerprint (deprecated)
- Parsed output for structured data

### Prerequisites

- Response structure basics
- Understanding of content extraction

---

## Annotations

Messages can include annotations linking to external sources:

### URL Citations

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What are the latest AI developments?"}],
    tools=[{"type": "web_search_preview"}]  # Enables web search
)

message = response.choices[0].message

# Check for annotations
if hasattr(message, "annotations") and message.annotations:
    print("Sources:")
    for annotation in message.annotations:
        if annotation.type == "url_citation":
            print(f"  - {annotation.url_citation.title}")
            print(f"    URL: {annotation.url_citation.url}")
            print(f"    Text: {annotation.text[:50]}...")
```

### Annotation Structure

```json
{
  "message": {
    "role": "assistant",
    "content": "Recent AI developments include...[1]",
    "annotations": [
      {
        "type": "url_citation",
        "text": "[1]",
        "start_index": 35,
        "end_index": 38,
        "url_citation": {
          "url": "https://example.com/ai-news",
          "title": "AI News Article"
        }
      }
    ]
  }
}
```

### Rendering Annotations

```python
def render_with_citations(message) -> str:
    """Render message content with citation links."""
    content = message.content
    annotations = getattr(message, "annotations", []) or []
    
    if not annotations:
        return content
    
    # Sort by position (reverse to not mess up indices)
    sorted_annotations = sorted(
        [a for a in annotations if a.type == "url_citation"],
        key=lambda a: a.start_index,
        reverse=True
    )
    
    # Replace citation markers with links
    for ann in sorted_annotations:
        citation = ann.url_citation
        link = f'[{ann.text}]({citation.url} "{citation.title}")'
        content = (
            content[:ann.start_index] + 
            link + 
            content[ann.end_index:]
        )
    
    return content


# Usage
rendered = render_with_citations(response.choices[0].message)
print(rendered)
```

---

## File Citations (Responses API)

When using file search, responses include file citations:

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What does our Q4 report say about revenue?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }]
)

for item in response.output:
    if item.type == "message":
        for content in item.content:
            if content.type == "output_text":
                # Check for annotations
                for ann in content.annotations or []:
                    if ann.type == "file_citation":
                        print(f"Source: {ann.filename}")
                        print(f"Quote: {ann.quote}")
```

---

## Refusal Handling

When the model declines a request, the refusal field explains why:

### Detecting Refusals

```python
def check_refusal(response) -> dict:
    """Check if response contains a refusal."""
    message = response.choices[0].message
    
    refusal = getattr(message, "refusal", None)
    
    if refusal:
        return {
            "refused": True,
            "reason": refusal,
            "content": message.content  # May still have some content
        }
    
    # Also check for content filter
    if response.choices[0].finish_reason == "content_filter":
        return {
            "refused": True,
            "reason": "Content policy violation",
            "content": None
        }
    
    return {
        "refused": False,
        "reason": None,
        "content": message.content
    }


# Usage
result = check_refusal(response)
if result["refused"]:
    print(f"Request declined: {result['reason']}")
else:
    print(result["content"])
```

### Refusal Response Format

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "refusal": "I can't help with that request as it may involve harmful content."
    },
    "finish_reason": "stop"
  }]
}
```

### User-Friendly Refusal Handling

```python
def handle_response_with_refusal(response):
    """Handle response with graceful refusal messaging."""
    message = response.choices[0].message
    
    if message.refusal:
        # Log for analysis
        log_refusal(response.id, message.refusal)
        
        # Return user-friendly message
        return {
            "success": False,
            "message": "I'm not able to help with that particular request. "
                      "Could you try rephrasing or asking something else?",
            "internal_reason": message.refusal
        }
    
    return {
        "success": True,
        "content": message.content
    }


def log_refusal(request_id: str, reason: str):
    """Log refusal for monitoring."""
    import json
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "refusal_reason": reason
    }
    
    # Append to log file or send to monitoring
    with open("refusals.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

---

## Service Tier

The `service_tier` field indicates which capacity tier served the request:

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}],
    service_tier="auto"  # or "default" or "flex"
)

print(f"Service tier used: {response.service_tier}")
# Output: "default" or "scale"
```

### Service Tier Options

| Tier | Description | Use Case |
|------|-------------|----------|
| `default` | Standard tier | Real-time applications |
| `flex` | Lower cost, variable latency | Batch processing |
| `auto` | Let API choose | Balanced approach |

### Monitoring Tier Usage

```python
from collections import defaultdict

class TierMonitor:
    def __init__(self):
        self.tier_counts = defaultdict(int)
        self.tier_latencies = defaultdict(list)
    
    def record(self, response, latency_ms: int):
        tier = getattr(response, "service_tier", "unknown")
        self.tier_counts[tier] += 1
        self.tier_latencies[tier].append(latency_ms)
    
    def report(self):
        print("Service Tier Usage:")
        for tier, count in self.tier_counts.items():
            latencies = self.tier_latencies[tier]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            print(f"  {tier}: {count} requests, avg {avg_latency:.0f}ms")


# Usage
monitor = TierMonitor()

import time
start = time.time()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)
latency = int((time.time() - start) * 1000)

monitor.record(response, latency)
monitor.report()
```

---

## System Fingerprint (Deprecated)

> **Note:** `system_fingerprint` is deprecated and may be removed in future API versions.

Previously used to track model version:

```python
# Deprecated - don't rely on this
fingerprint = getattr(response, "system_fingerprint", None)
if fingerprint:
    print(f"System fingerprint: {fingerprint}")
    # fp_44709d6fcb
```

### Migration

Instead, use the `model` field which now includes version info:

```python
# Current approach - use model version
print(f"Model: {response.model}")
# Output: gpt-4.1-2025-04-14
```

---

## Parsed Output (Structured Data)

When using structured outputs, `output_parsed` provides typed access:

### With Pydantic

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

response = client.responses.parse(
    model="gpt-4.1",
    input="Write an article about AI",
    text_format=Article
)

# Typed access via output_parsed
article = response.output_parsed
print(f"Title: {article.title}")
print(f"Tags: {', '.join(article.tags)}")
```

### Raw vs Parsed

```python
def get_structured_output(response, expected_type=None):
    """Get structured output, preferring parsed version."""
    
    # Try parsed first (typed)
    if hasattr(response, "output_parsed") and response.output_parsed:
        return {
            "source": "parsed",
            "data": response.output_parsed,
            "typed": True
        }
    
    # Fall back to raw text parsing
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    try:
                        import json
                        data = json.loads(content.text)
                        return {
                            "source": "raw_json",
                            "data": data,
                            "typed": False
                        }
                    except json.JSONDecodeError:
                        pass
    
    return {"source": None, "data": None, "typed": False}
```

---

## Complete Metadata Extractor

```python
from dataclasses import dataclass
from typing import Optional, List, Any
from datetime import datetime

@dataclass
class Annotation:
    type: str
    text: str
    start_index: int
    end_index: int
    url: Optional[str] = None
    title: Optional[str] = None
    filename: Optional[str] = None
    quote: Optional[str] = None

@dataclass
class ResponseMetadata:
    id: str
    model: str
    created: datetime
    service_tier: Optional[str]
    finish_reason: str
    refusal: Optional[str]
    annotations: List[Annotation]
    has_tool_calls: bool
    has_structured_output: bool


class MetadataExtractor:
    """Extract all metadata from API responses."""
    
    @staticmethod
    def extract(response, provider: str = "openai") -> ResponseMetadata:
        if provider == "openai":
            return MetadataExtractor._extract_openai(response)
        elif provider == "anthropic":
            return MetadataExtractor._extract_anthropic(response)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def _extract_openai(response) -> ResponseMetadata:
        message = response.choices[0].message if response.choices else None
        
        # Extract annotations
        annotations = []
        if message and hasattr(message, "annotations") and message.annotations:
            for ann in message.annotations:
                annotation = Annotation(
                    type=ann.type,
                    text=getattr(ann, "text", ""),
                    start_index=getattr(ann, "start_index", 0),
                    end_index=getattr(ann, "end_index", 0)
                )
                
                if ann.type == "url_citation" and hasattr(ann, "url_citation"):
                    annotation.url = ann.url_citation.url
                    annotation.title = ann.url_citation.title
                
                annotations.append(annotation)
        
        # Created timestamp
        created = datetime.fromtimestamp(response.created)
        
        return ResponseMetadata(
            id=response.id,
            model=response.model,
            created=created,
            service_tier=getattr(response, "service_tier", None),
            finish_reason=response.choices[0].finish_reason if response.choices else "unknown",
            refusal=getattr(message, "refusal", None) if message else None,
            annotations=annotations,
            has_tool_calls=bool(message and message.tool_calls),
            has_structured_output=False  # Not applicable for chat
        )
    
    @staticmethod
    def _extract_anthropic(response) -> ResponseMetadata:
        return ResponseMetadata(
            id=response.id,
            model=response.model,
            created=datetime.now(),  # Anthropic doesn't provide created
            service_tier=None,
            finish_reason=response.stop_reason,
            refusal=None,  # Anthropic handles differently
            annotations=[],
            has_tool_calls=any(b.type == "tool_use" for b in response.content),
            has_structured_output=False
        )


# Usage
metadata = MetadataExtractor.extract(response, "openai")
print(f"Model: {metadata.model}")
print(f"Created: {metadata.created}")
print(f"Tier: {metadata.service_tier}")
print(f"Annotations: {len(metadata.annotations)}")

if metadata.refusal:
    print(f"⚠️ Refusal: {metadata.refusal}")
```

---

## JavaScript Implementation

```javascript
class MetadataExtractor {
    static extract(response, provider = 'openai') {
        if (provider === 'openai') {
            return this.extractOpenAI(response);
        } else if (provider === 'anthropic') {
            return this.extractAnthropic(response);
        }
        throw new Error(`Unknown provider: ${provider}`);
    }
    
    static extractOpenAI(response) {
        const message = response?.choices?.[0]?.message;
        
        // Extract annotations
        const annotations = (message?.annotations || []).map(ann => ({
            type: ann.type,
            text: ann.text,
            startIndex: ann.start_index,
            endIndex: ann.end_index,
            url: ann.url_citation?.url,
            title: ann.url_citation?.title
        }));
        
        return {
            id: response.id,
            model: response.model,
            created: new Date(response.created * 1000),
            serviceTier: response.service_tier ?? null,
            finishReason: response.choices?.[0]?.finish_reason ?? 'unknown',
            refusal: message?.refusal ?? null,
            annotations,
            hasToolCalls: Boolean(message?.tool_calls?.length),
            hasStructuredOutput: false
        };
    }
    
    static extractAnthropic(response) {
        return {
            id: response.id,
            model: response.model,
            created: new Date(),
            serviceTier: null,
            finishReason: response.stop_reason,
            refusal: null,
            annotations: [],
            hasToolCalls: response.content?.some(b => b.type === 'tool_use'),
            hasStructuredOutput: false
        };
    }
    
    static formatMetadata(metadata) {
        const lines = [
            `ID: ${metadata.id}`,
            `Model: ${metadata.model}`,
            `Created: ${metadata.created.toISOString()}`,
            `Finish: ${metadata.finishReason}`
        ];
        
        if (metadata.serviceTier) {
            lines.push(`Tier: ${metadata.serviceTier}`);
        }
        
        if (metadata.refusal) {
            lines.push(`⚠️ Refusal: ${metadata.refusal}`);
        }
        
        if (metadata.annotations.length > 0) {
            lines.push(`📎 Annotations: ${metadata.annotations.length}`);
        }
        
        return lines.join('\n');
    }
}

// Usage
const metadata = MetadataExtractor.extract(response, 'openai');
console.log(MetadataExtractor.formatMetadata(metadata));
```

---

## Hands-on Exercise

### Your Task

Build a response analyzer that extracts and displays all metadata.

### Requirements

1. Extract all metadata fields
2. Format annotations with source links
3. Flag refusals prominently
4. Show service tier and timing

### Expected Result

```
Response Analysis
=================
ID: chatcmpl-abc123
Model: gpt-4.1-2025-04-14
Created: 2025-01-15 10:30:00
Tier: default
Status: stop

📎 Citations (2):
  [1] AI News Today - https://example.com/ai
  [2] Research Paper - https://arxiv.org/...

✅ No refusal
🔧 No tool calls
```

<details>
<summary>💡 Hints</summary>

- Check for annotations on message object
- Use `datetime.fromtimestamp()` for created
- Format annotations as numbered list
</details>

<details>
<summary>✅ Solution</summary>

```python
from datetime import datetime

def analyze_response(response) -> str:
    """Generate complete response analysis."""
    lines = ["Response Analysis", "=" * 40]
    
    # Basic info
    lines.append(f"ID: {response.id}")
    lines.append(f"Model: {response.model}")
    
    created = datetime.fromtimestamp(response.created)
    lines.append(f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tier = getattr(response, "service_tier", "unknown")
    lines.append(f"Tier: {tier}")
    
    finish = response.choices[0].finish_reason if response.choices else "unknown"
    lines.append(f"Status: {finish}")
    lines.append("")
    
    # Annotations
    message = response.choices[0].message if response.choices else None
    annotations = getattr(message, "annotations", []) or []
    
    if annotations:
        lines.append(f"📎 Citations ({len(annotations)}):")
        for i, ann in enumerate(annotations, 1):
            if ann.type == "url_citation":
                title = ann.url_citation.title
                url = ann.url_citation.url
                lines.append(f"  [{i}] {title}")
                lines.append(f"      {url}")
    else:
        lines.append("📎 No citations")
    
    lines.append("")
    
    # Refusal
    refusal = getattr(message, "refusal", None) if message else None
    if refusal:
        lines.append(f"⚠️ REFUSAL: {refusal}")
    else:
        lines.append("✅ No refusal")
    
    # Tool calls
    tool_calls = getattr(message, "tool_calls", None) if message else None
    if tool_calls:
        lines.append(f"🔧 Tool calls: {len(tool_calls)}")
        for tc in tool_calls:
            lines.append(f"   - {tc.function.name}")
    else:
        lines.append("🔧 No tool calls")
    
    return "\n".join(lines)


# Test
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)

print(analyze_response(response))
```

</details>

---

## Summary

✅ Annotations provide source citations for web/file search  
✅ Refusal field explains why requests are declined  
✅ Service tier indicates which capacity pool served the request  
✅ `output_parsed` provides typed access to structured outputs  
✅ Extract all metadata for comprehensive logging and monitoring

**Previous:** [Content Extraction Patterns](./06-content-extraction.md)

**Back to:** [Response Handling Overview](./00-response-handling.md)

---

## Further Reading

- [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search) — Citation annotations
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Parsed output
- [File Search](https://platform.openai.com/docs/assistants/tools/file-search) — File citations

<!-- 
Sources Consulted:
- OpenAI Web Search: https://platform.openai.com/docs/guides/tools-web-search
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
