---
title: "Cacheable Content"
---

# Cacheable Content

## Introduction

Understanding what types of content can be cached is essential for maximizing caching benefits. Both OpenAI and Anthropic support caching for various content types including messages, images, tool definitions, and structured output schemas.

### What We'll Cover

- Message content caching
- System/developer message caching
- Image caching (URLs and base64)
- Tool definition caching
- Structured output schema caching
- Content size requirements

### Prerequisites

- Understanding of OpenAI and Anthropic caching
- API access to either provider
- Python development environment

---

## Message Content Caching

### Text Messages

```python
from dataclasses import dataclass
from typing import List, Literal, Optional
import hashlib

@dataclass
class CacheableMessage:
    role: Literal["system", "user", "assistant", "developer"]
    content: str
    tokens_estimate: int = 0
    
    def __post_init__(self):
        # Rough estimate: 4 chars per token
        if self.tokens_estimate == 0:
            self.tokens_estimate = len(self.content) // 4
    
    @property
    def is_cacheable(self) -> bool:
        """Check if message meets minimum token threshold."""
        return self.tokens_estimate >= 1024
    
    @property
    def content_hash(self) -> str:
        """Generate hash for cache key matching."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


def analyze_message_cacheability(messages: List[dict]) -> dict:
    """Analyze which messages are cacheable."""
    
    results = []
    total_tokens = 0
    cacheable_tokens = 0
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle structured content
            content = " ".join(
                block.get("text", "") for block in content 
                if isinstance(block, dict) and block.get("type") == "text"
            )
        
        tokens = len(content) // 4
        is_cacheable = tokens >= 256  # Per-message threshold lower
        
        results.append({
            "index": i,
            "role": msg.get("role"),
            "tokens": tokens,
            "cacheable": is_cacheable,
            "recommendation": "Include in prefix" if is_cacheable else "Dynamic content"
        })
        
        total_tokens += tokens
        if is_cacheable:
            cacheable_tokens += tokens
    
    return {
        "messages": results,
        "total_tokens": total_tokens,
        "cacheable_tokens": cacheable_tokens,
        "cache_potential": f"{cacheable_tokens / total_tokens * 100:.1f}%" if total_tokens > 0 else "0%"
    }


# Example
messages = [
    {"role": "system", "content": "You are an expert assistant. " * 100},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
    {"role": "user", "content": "What is Python?"}
]

analysis = analyze_message_cacheability(messages)
print(f"Cache potential: {analysis['cache_potential']}")
```

### Conversation History Caching

```python
from typing import List

class ConversationCache:
    """Manage cacheable conversation history."""
    
    def __init__(self, min_prefix_tokens: int = 1024):
        self.min_prefix_tokens = min_prefix_tokens
        self.messages: List[dict] = []
    
    def add_message(self, role: str, content: str):
        """Add message to conversation."""
        self.messages.append({"role": role, "content": content})
    
    def get_cache_split(self) -> tuple:
        """Split messages into cached prefix and dynamic suffix."""
        
        # Calculate cumulative tokens
        cumulative = []
        running_total = 0
        
        for msg in self.messages:
            content = msg.get("content", "")
            tokens = len(content) // 4
            running_total += tokens
            cumulative.append(running_total)
        
        # Find split point
        split_index = 0
        for i, total in enumerate(cumulative):
            if total >= self.min_prefix_tokens:
                split_index = i + 1
                break
        
        prefix = self.messages[:split_index]
        suffix = self.messages[split_index:]
        
        return prefix, suffix
    
    def format_for_openai(self) -> List[dict]:
        """Format for OpenAI Chat Completions."""
        return self.messages
    
    def format_for_anthropic(self) -> tuple:
        """Format for Anthropic with cache control."""
        
        prefix, suffix = self.get_cache_split()
        
        # Mark prefix messages for caching
        cached_messages = []
        for i, msg in enumerate(prefix):
            formatted = {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"]
                    }
                ]
            }
            
            # Add cache control to last prefix message
            if i == len(prefix) - 1:
                formatted["content"][0]["cache_control"] = {"type": "ephemeral"}
            
            cached_messages.append(formatted)
        
        # Add suffix without caching
        for msg in suffix:
            cached_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return cached_messages


# Usage
convo = ConversationCache(min_prefix_tokens=1024)

# Add large context
convo.add_message("system", "Detailed instructions... " * 200)
convo.add_message("user", "First question")
convo.add_message("assistant", "First answer")
convo.add_message("user", "Follow-up")  # Dynamic

prefix, suffix = convo.get_cache_split()
print(f"Prefix messages: {len(prefix)} (cached)")
print(f"Suffix messages: {len(suffix)} (dynamic)")
```

---

## System Message Caching

### OpenAI Developer Messages

```python
# OpenAI: System messages cached automatically
# Responses API uses "instructions" parameter

def create_cached_system_prompt(
    base_instructions: str,
    knowledge_base: str,
    rules: List[str]
) -> str:
    """Create a cache-optimized system prompt."""
    
    sections = []
    
    # Static sections (cached together)
    sections.append("## Role and Identity")
    sections.append(base_instructions)
    
    sections.append("\n## Knowledge Base")
    sections.append(knowledge_base)
    
    sections.append("\n## Rules and Guidelines")
    for i, rule in enumerate(rules, 1):
        sections.append(f"{i}. {rule}")
    
    return "\n".join(sections)


# Example
system_prompt = create_cached_system_prompt(
    base_instructions="You are an expert Python developer and educator.",
    knowledge_base="Reference documentation: " + "Python 3.12 features..." * 100,
    rules=[
        "Always provide working code examples",
        "Explain the 'why' behind recommendations",
        "Use type hints in all code",
        "Follow PEP 8 style guidelines"
    ]
)

print(f"System prompt tokens: ~{len(system_prompt) // 4}")
```

### Anthropic System Blocks

```python
def create_anthropic_system_blocks(
    instructions: str,
    knowledge: str,
    dynamic_context: Optional[str] = None,
    ttl_hours: int = 1
) -> List[dict]:
    """Create system blocks with caching for Anthropic."""
    
    blocks = []
    ttl = ttl_hours * 3600
    
    # Instructions block (cached)
    blocks.append({
        "type": "text",
        "text": instructions,
        "cache_control": {"type": "ephemeral", "ttl": ttl}
    })
    
    # Knowledge block (cached)
    if knowledge:
        blocks.append({
            "type": "text",
            "text": knowledge,
            "cache_control": {"type": "ephemeral", "ttl": ttl}
        })
    
    # Dynamic context (not cached)
    if dynamic_context:
        blocks.append({
            "type": "text",
            "text": dynamic_context
            # No cache_control
        })
    
    return blocks


# Example
from datetime import datetime

system_blocks = create_anthropic_system_blocks(
    instructions="You are an expert assistant. " * 100,
    knowledge="Documentation: " + "Reference content..." * 200,
    dynamic_context=f"Current time: {datetime.now().isoformat()}"
)

print(f"Total blocks: {len(system_blocks)}")
print(f"Cached blocks: {sum(1 for b in system_blocks if 'cache_control' in b)}")
```

---

## Image Caching

### URL-Based Images

```python
@dataclass
class CacheableImage:
    """Image content that can be cached."""
    
    url: Optional[str] = None
    base64_data: Optional[str] = None
    media_type: str = "image/png"
    detail: Literal["auto", "low", "high"] = "auto"
    
    @property
    def token_estimate(self) -> int:
        """Estimate tokens for image."""
        # Based on detail level
        if self.detail == "low":
            return 85  # Fixed for low detail
        elif self.detail == "high":
            return 765  # Per 512x512 tile
        else:
            return 300  # Average for auto
    
    def to_openai_content(self) -> dict:
        """Format for OpenAI API."""
        if self.url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": self.url,
                    "detail": self.detail
                }
            }
        else:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.media_type};base64,{self.base64_data}",
                    "detail": self.detail
                }
            }
    
    def to_anthropic_content(self, cache: bool = False) -> dict:
        """Format for Anthropic API."""
        if self.url:
            block = {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": self.url
                }
            }
        else:
            block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": self.base64_data
                }
            }
        
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        
        return block


# Example usage
image = CacheableImage(
    url="https://example.com/diagram.png",
    detail="high"
)

print(f"Estimated tokens: {image.token_estimate}")
print("OpenAI format:", image.to_openai_content())
```

### Base64 Image Caching

```python
import base64
from pathlib import Path

def prepare_cached_images(
    image_paths: List[str],
    detail: str = "auto"
) -> List[CacheableImage]:
    """Prepare images for caching."""
    
    images = []
    
    for path in image_paths:
        path_obj = Path(path)
        
        # Determine media type
        suffix = path_obj.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(suffix, "image/png")
        
        # Read and encode
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        
        images.append(CacheableImage(
            base64_data=data,
            media_type=media_type,
            detail=detail
        ))
    
    return images


def build_image_message(
    text: str,
    images: List[CacheableImage],
    cache_images: bool = True
) -> dict:
    """Build message with images for Anthropic."""
    
    content = [{"type": "text", "text": text}]
    
    for img in images:
        content.append(img.to_anthropic_content(cache=cache_images))
    
    return {"role": "user", "content": content}
```

### Image Caching Best Practices

```python
@dataclass
class ImageCachingConfig:
    """Configuration for image caching."""
    
    cache_static_images: bool = True
    consistent_detail_level: str = "high"
    max_images_per_request: int = 10
    
    def validate_request(self, images: List[CacheableImage]) -> dict:
        """Validate image caching configuration."""
        
        issues = []
        recommendations = []
        
        # Check count
        if len(images) > self.max_images_per_request:
            issues.append(f"Too many images ({len(images)})")
        
        # Check detail consistency
        details = set(img.detail for img in images)
        if len(details) > 1:
            issues.append("Inconsistent detail levels break caching")
            recommendations.append("Use consistent detail level across all images")
        
        # Estimate tokens
        total_tokens = sum(img.token_estimate for img in images)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "total_image_tokens": total_tokens,
            "cacheable": total_tokens >= 1024
        }


# Validate
config = ImageCachingConfig(consistent_detail_level="high")

images = [
    CacheableImage(url="https://example.com/img1.png", detail="high"),
    CacheableImage(url="https://example.com/img2.png", detail="low"),  # Inconsistent!
]

validation = config.validate_request(images)
print(f"Valid: {validation['valid']}")
print(f"Issues: {validation['issues']}")
```

---

## Tool Definition Caching

### Caching Function Definitions

```python
from typing import Dict, Any

@dataclass
class CacheableTool:
    """Tool definition optimized for caching."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_openai_tool(self) -> dict:
        """Format for OpenAI."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def to_anthropic_tool(self) -> dict:
        """Format for Anthropic."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


class ToolRegistry:
    """Registry of cacheable tools."""
    
    def __init__(self):
        self.tools: Dict[str, CacheableTool] = {}
        self._version = "1.0"
    
    def register(self, tool: CacheableTool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_openai_tools(self) -> List[dict]:
        """Get tools formatted for OpenAI."""
        return [tool.to_openai_tool() for tool in self.tools.values()]
    
    def get_anthropic_tools(self, cache: bool = True) -> List[dict]:
        """Get tools formatted for Anthropic with optional caching."""
        tools = [tool.to_anthropic_tool() for tool in self.tools.values()]
        
        if cache and tools:
            # Add cache control to last tool
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        
        return tools
    
    def get_cache_key(self) -> str:
        """Generate cache key for tool definitions."""
        # Include version and tool names for cache invalidation
        tool_sig = "|".join(sorted(self.tools.keys()))
        return f"tools_v{self._version}_{hashlib.sha256(tool_sig.encode()).hexdigest()[:8]}"
    
    def bump_version(self):
        """Bump version when tools change (invalidates cache)."""
        major, minor = self._version.split(".")
        self._version = f"{major}.{int(minor) + 1}"


# Example
registry = ToolRegistry()

registry.register(CacheableTool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
))

registry.register(CacheableTool(
    name="search_docs",
    description="Search documentation",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
))

print(f"Cache key: {registry.get_cache_key()}")
print(f"Tools: {len(registry.tools)}")
```

### Tool Caching Strategies

```python
class ToolCacheStrategy:
    """Strategies for caching tool definitions."""
    
    @staticmethod
    def consistent_ordering(tools: List[CacheableTool]) -> List[CacheableTool]:
        """Sort tools for consistent cache keys."""
        return sorted(tools, key=lambda t: t.name)
    
    @staticmethod
    def minimal_descriptions(tools: List[CacheableTool]) -> List[CacheableTool]:
        """Truncate descriptions to save tokens while preserving meaning."""
        processed = []
        for tool in tools:
            processed.append(CacheableTool(
                name=tool.name,
                description=tool.description[:200] if len(tool.description) > 200 else tool.description,
                parameters=tool.parameters
            ))
        return processed
    
    @staticmethod
    def estimate_tool_tokens(tools: List[CacheableTool]) -> int:
        """Estimate token count for tools."""
        import json
        
        total = 0
        for tool in tools:
            # Estimate from JSON serialization
            tool_json = json.dumps(tool.to_openai_tool())
            total += len(tool_json) // 4
        
        return total


# Analyze tool caching potential
tools = registry.get_openai_tools()
tokens = ToolCacheStrategy.estimate_tool_tokens(list(registry.tools.values()))

print(f"Tool tokens: {tokens}")
print(f"Cacheable: {tokens >= 1024}")
```

---

## Structured Output Schema Caching

### Caching Response Schemas

```python
from typing import Type
from pydantic import BaseModel

class CacheableSchema:
    """Manage cacheable output schemas."""
    
    def __init__(self, model: Type[BaseModel], version: str = "1.0"):
        self.model = model
        self.version = version
        self._schema = model.model_json_schema()
    
    @property
    def schema(self) -> dict:
        """Get JSON schema."""
        return self._schema
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for schema."""
        import json
        schema_str = json.dumps(self._schema, sort_keys=True)
        return f"schema_{self.model.__name__}_v{self.version}_{hashlib.sha256(schema_str.encode()).hexdigest()[:8]}"
    
    def to_openai_format(self) -> dict:
        """Format for OpenAI structured outputs."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.model.__name__,
                "schema": self._schema,
                "strict": True
            }
        }
    
    def token_estimate(self) -> int:
        """Estimate tokens for schema."""
        import json
        return len(json.dumps(self._schema)) // 4


# Example with Pydantic
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int
    forecast: List[str]


class DetailedAnalysis(BaseModel):
    summary: str
    key_points: List[str]
    confidence: float
    sources: List[str]
    recommendations: List[str]


# Create cacheable schemas
weather_schema = CacheableSchema(WeatherResponse, version="1.0")
analysis_schema = CacheableSchema(DetailedAnalysis, version="2.1")

print(f"Weather schema tokens: {weather_schema.token_estimate()}")
print(f"Analysis schema tokens: {analysis_schema.token_estimate()}")
print(f"Weather cache key: {weather_schema.cache_key}")
```

### Schema Registry

```python
class SchemaRegistry:
    """Registry of cacheable output schemas."""
    
    def __init__(self):
        self.schemas: Dict[str, CacheableSchema] = {}
    
    def register(self, name: str, schema: CacheableSchema):
        """Register a schema."""
        self.schemas[name] = schema
    
    def get(self, name: str) -> Optional[CacheableSchema]:
        """Get schema by name."""
        return self.schemas.get(name)
    
    def combined_cache_key(self) -> str:
        """Generate combined cache key for all schemas."""
        keys = sorted(s.cache_key for s in self.schemas.values())
        combined = "|".join(keys)
        return f"schemas_{hashlib.sha256(combined.encode()).hexdigest()[:12]}"
    
    def total_tokens(self) -> int:
        """Total tokens for all schemas."""
        return sum(s.token_estimate() for s in self.schemas.values())


# Usage
schema_registry = SchemaRegistry()
schema_registry.register("weather", weather_schema)
schema_registry.register("analysis", analysis_schema)

print(f"Combined key: {schema_registry.combined_cache_key()}")
print(f"Total tokens: {schema_registry.total_tokens()}")
```

---

## Content Size Requirements

### Token Thresholds

```python
@dataclass
class CachingThreshold:
    """Caching threshold configuration."""
    
    provider: str
    min_tokens: int
    recommended_min: int
    notes: str


THRESHOLDS = {
    "openai": CachingThreshold(
        provider="OpenAI",
        min_tokens=1024,
        recommended_min=2000,
        notes="Automatic caching, no configuration needed"
    ),
    "anthropic": CachingThreshold(
        provider="Anthropic",
        min_tokens=1024,
        recommended_min=2000,
        notes="Explicit cache_control blocks required"
    )
}


def check_caching_viability(
    content: str,
    provider: str,
    content_type: str = "text"
) -> dict:
    """Check if content is viable for caching."""
    
    threshold = THRESHOLDS.get(provider.lower())
    if not threshold:
        return {"error": f"Unknown provider: {provider}"}
    
    tokens = len(content) // 4
    
    meets_minimum = tokens >= threshold.min_tokens
    meets_recommended = tokens >= threshold.recommended_min
    
    return {
        "provider": provider,
        "content_type": content_type,
        "tokens": tokens,
        "min_threshold": threshold.min_tokens,
        "meets_minimum": meets_minimum,
        "meets_recommended": meets_recommended,
        "recommendation": (
            "Excellent caching candidate" if meets_recommended else
            "Viable but consider adding more context" if meets_minimum else
            f"Need {threshold.min_tokens - tokens} more tokens for caching"
        )
    }


# Example
content = "System prompt content " * 300
result = check_caching_viability(content, "openai")
print(f"Tokens: {result['tokens']}")
print(f"Recommendation: {result['recommendation']}")
```

---

## Hands-on Exercise

### Your Task

Build a content analyzer that determines cacheability.

### Requirements

1. Analyze mixed content (text, images, tools, schemas)
2. Calculate total cacheable tokens
3. Provide optimization recommendations
4. Generate cache keys

<details>
<summary>💡 Hints</summary>

- Sum tokens from all content types
- Check threshold per content type
- Combine cache keys for composite content
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import hashlib
import json

@dataclass
class ContentItem:
    """Base class for cacheable content."""
    content_type: str
    tokens: int
    cacheable: bool
    cache_key: str


@dataclass
class TextContent(ContentItem):
    text: str
    
    def __init__(self, text: str):
        tokens = len(text) // 4
        super().__init__(
            content_type="text",
            tokens=tokens,
            cacheable=tokens >= 256,
            cache_key=hashlib.sha256(text.encode()).hexdigest()[:12]
        )
        self.text = text


@dataclass
class ImageContent(ContentItem):
    url: Optional[str] = None
    detail: str = "auto"
    
    def __init__(self, url: str = None, detail: str = "auto"):
        token_map = {"low": 85, "high": 765, "auto": 300}
        tokens = token_map.get(detail, 300)
        
        super().__init__(
            content_type="image",
            tokens=tokens,
            cacheable=True,  # Images always cacheable
            cache_key=hashlib.sha256((url or "").encode()).hexdigest()[:12]
        )
        self.url = url
        self.detail = detail


@dataclass
class ToolContent(ContentItem):
    tools: List[dict]
    
    def __init__(self, tools: List[dict]):
        tool_json = json.dumps(tools, sort_keys=True)
        tokens = len(tool_json) // 4
        
        super().__init__(
            content_type="tools",
            tokens=tokens,
            cacheable=tokens >= 256,
            cache_key=hashlib.sha256(tool_json.encode()).hexdigest()[:12]
        )
        self.tools = tools


@dataclass
class SchemaContent(ContentItem):
    schema: dict
    
    def __init__(self, schema: dict):
        schema_json = json.dumps(schema, sort_keys=True)
        tokens = len(schema_json) // 4
        
        super().__init__(
            content_type="schema",
            tokens=tokens,
            cacheable=tokens >= 128,
            cache_key=hashlib.sha256(schema_json.encode()).hexdigest()[:12]
        )
        self.schema = schema


class ContentAnalyzer:
    """Analyze content for caching optimization."""
    
    MIN_CACHE_THRESHOLD = 1024
    
    def __init__(self):
        self.items: List[ContentItem] = []
    
    def add_text(self, text: str) -> "ContentAnalyzer":
        self.items.append(TextContent(text))
        return self
    
    def add_image(self, url: str, detail: str = "auto") -> "ContentAnalyzer":
        self.items.append(ImageContent(url, detail))
        return self
    
    def add_tools(self, tools: List[dict]) -> "ContentAnalyzer":
        self.items.append(ToolContent(tools))
        return self
    
    def add_schema(self, schema: dict) -> "ContentAnalyzer":
        self.items.append(SchemaContent(schema))
        return self
    
    def analyze(self) -> dict:
        """Analyze all content for cacheability."""
        
        total_tokens = sum(item.tokens for item in self.items)
        cacheable_tokens = sum(
            item.tokens for item in self.items 
            if item.cacheable
        )
        
        by_type = {}
        for item in self.items:
            if item.content_type not in by_type:
                by_type[item.content_type] = {"count": 0, "tokens": 0}
            by_type[item.content_type]["count"] += 1
            by_type[item.content_type]["tokens"] += item.tokens
        
        meets_threshold = total_tokens >= self.MIN_CACHE_THRESHOLD
        
        return {
            "total_items": len(self.items),
            "total_tokens": total_tokens,
            "cacheable_tokens": cacheable_tokens,
            "cache_potential": f"{cacheable_tokens / total_tokens * 100:.1f}%" if total_tokens > 0 else "0%",
            "meets_threshold": meets_threshold,
            "by_type": by_type,
            "combined_cache_key": self.get_combined_cache_key()
        }
    
    def get_combined_cache_key(self) -> str:
        """Generate combined cache key."""
        keys = sorted(item.cache_key for item in self.items if item.cacheable)
        combined = "|".join(keys)
        return f"content_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        
        recommendations = []
        analysis = self.analyze()
        
        # Threshold check
        if not analysis["meets_threshold"]:
            deficit = self.MIN_CACHE_THRESHOLD - analysis["total_tokens"]
            recommendations.append(
                f"Add ~{deficit} more tokens to enable caching"
            )
        
        # Content type recommendations
        by_type = analysis["by_type"]
        
        if "text" in by_type and by_type["text"]["tokens"] < 500:
            recommendations.append(
                "Consider adding more detailed instructions for better caching"
            )
        
        if "image" in by_type:
            images = [i for i in self.items if i.content_type == "image"]
            details = set(i.detail for i in images)
            if len(details) > 1:
                recommendations.append(
                    "Use consistent image detail level for better caching"
                )
        
        if "tools" in by_type and by_type["tools"]["tokens"] > 500:
            recommendations.append(
                "Large tool definitions are good caching candidates"
            )
        
        # General recommendations
        if analysis["cache_potential"] == "100.0%":
            recommendations.append(
                "Excellent! All content is cacheable"
            )
        elif float(analysis["cache_potential"].rstrip("%")) < 50:
            recommendations.append(
                "Consider restructuring to increase cacheable content"
            )
        
        return recommendations if recommendations else ["No optimizations needed"]
    
    def format_for_openai(self) -> dict:
        """Format content for OpenAI API."""
        
        messages = []
        tools = []
        response_format = None
        
        for item in self.items:
            if item.content_type == "text":
                messages.append({
                    "role": "system",
                    "content": item.text
                })
            elif item.content_type == "tools":
                tools.extend(item.tools)
            elif item.content_type == "schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"schema": item.schema}
                }
        
        result = {"messages": messages}
        if tools:
            result["tools"] = tools
        if response_format:
            result["response_format"] = response_format
        
        return result
    
    def format_for_anthropic(self) -> dict:
        """Format content for Anthropic API with caching."""
        
        system_blocks = []
        tools = []
        
        for item in self.items:
            if item.content_type == "text":
                block = {"type": "text", "text": item.text}
                if item.cacheable:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
            elif item.content_type == "tools":
                for tool in item.tools:
                    tools.append(tool)
        
        # Add cache control to last tool
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        
        result = {"system": system_blocks}
        if tools:
            result["tools"] = tools
        
        return result


# Usage example
analyzer = ContentAnalyzer()

# Add various content
analyzer.add_text("You are an expert Python developer. " * 100)
analyzer.add_text("Reference documentation: " + "Technical content..." * 200)
analyzer.add_image("https://example.com/diagram.png", detail="high")
analyzer.add_tools([
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search documentation",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    }
])
analyzer.add_schema({
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    }
})

# Analyze
print("=== Content Analysis ===")
analysis = analyzer.analyze()
for key, value in analysis.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")

print("\n=== Recommendations ===")
for rec in analyzer.get_recommendations():
    print(f"• {rec}")

print("\n=== Combined Cache Key ===")
print(analyzer.get_combined_cache_key())
```

</details>

---

## Summary

✅ Messages, images, tools, and schemas are all cacheable  
✅ Consistent content ordering improves cache hits  
✅ Images require consistent detail levels for caching  
✅ Tool definitions should be sorted alphabetically  
✅ Minimum 1024 tokens required for caching

**Next:** [Cache-Friendly Design](./05-cache-friendly-design.md)

---

## Further Reading

- [OpenAI Vision](https://platform.openai.com/docs/guides/vision) — Image handling
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/tool-use) — Tool definitions
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Schema caching
