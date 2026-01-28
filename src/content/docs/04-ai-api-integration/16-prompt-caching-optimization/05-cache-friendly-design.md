---
title: "Cache-Friendly Design"
---

# Cache-Friendly Design

## Introduction

Designing prompts and API requests for optimal cache performance requires understanding how prefix-based caching works. The key principle is simple: static content first, dynamic content last. This lesson covers architectural patterns for maximizing cache hits.

### What We'll Cover

- Static prefix, dynamic suffix pattern
- Prompt structure optimization
- Consistent tool and schema definitions
- Multi-turn conversation design
- Cache-aware request ordering

### Prerequisites

- Understanding of caching mechanisms
- Experience with LLM API calls
- Python development environment

---

## Static Prefix Pattern

### Core Concept

```python
"""
Cache matching works on PREFIXES. Content must match exactly from the start.

GOOD (high cache hit rate):
┌─────────────────────────────────┐
│ Static System Instructions      │ ← CACHED (same every request)
│ Static Knowledge Base           │ ← CACHED
│ Static Tool Definitions         │ ← CACHED
├─────────────────────────────────┤
│ Dynamic User Context            │ ← Not cached (changes)
│ Current Query                   │ ← Not cached
└─────────────────────────────────┘

BAD (low cache hit rate):
┌─────────────────────────────────┐
│ Current Timestamp               │ ← BREAKS CACHE (changes first!)
│ User-specific Context           │ ← Everything after is uncached
│ Static System Instructions      │ ← Could be cached but isn't
│ Static Knowledge Base           │ ← Could be cached but isn't
└─────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PromptComponent:
    """Component of a prompt with caching metadata."""
    
    content: str
    is_static: bool
    order_priority: int  # Lower = earlier in prompt
    description: str = ""
    
    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


def optimize_prompt_order(components: List[PromptComponent]) -> List[PromptComponent]:
    """Sort components for optimal caching."""
    
    # Static components first (sorted by priority)
    # Dynamic components last (sorted by priority)
    return sorted(components, key=lambda c: (not c.is_static, c.order_priority))


# Example
components = [
    PromptComponent("Current time: 2024-01-15", is_static=False, order_priority=1, description="Timestamp"),
    PromptComponent("You are an expert Python tutor...", is_static=True, order_priority=1, description="Instructions"),
    PromptComponent("User preferences: dark mode", is_static=False, order_priority=2, description="User prefs"),
    PromptComponent("Python documentation reference...", is_static=True, order_priority=2, description="Knowledge"),
]

optimized = optimize_prompt_order(components)
for c in optimized:
    status = "STATIC" if c.is_static else "DYNAMIC"
    print(f"[{status}] {c.description}: {c.content[:30]}...")
```

### Implementation Pattern

```python
class CacheOptimizedPrompt:
    """Build prompts optimized for caching."""
    
    def __init__(self):
        self._static_sections: List[str] = []
        self._dynamic_sections: List[str] = []
    
    def add_static(self, content: str, label: str = "") -> "CacheOptimizedPrompt":
        """Add static content (cached)."""
        if label:
            self._static_sections.append(f"## {label}\n{content}")
        else:
            self._static_sections.append(content)
        return self
    
    def add_dynamic(self, content: str, label: str = "") -> "CacheOptimizedPrompt":
        """Add dynamic content (not cached)."""
        if label:
            self._dynamic_sections.append(f"## {label}\n{content}")
        else:
            self._dynamic_sections.append(content)
        return self
    
    def build(self) -> str:
        """Build the optimized prompt."""
        # Static first, dynamic last
        all_sections = self._static_sections + self._dynamic_sections
        return "\n\n".join(all_sections)
    
    def get_cache_stats(self) -> dict:
        """Get caching statistics."""
        static_tokens = sum(len(s) // 4 for s in self._static_sections)
        dynamic_tokens = sum(len(s) // 4 for s in self._dynamic_sections)
        total = static_tokens + dynamic_tokens
        
        return {
            "static_tokens": static_tokens,
            "dynamic_tokens": dynamic_tokens,
            "total_tokens": total,
            "cache_ratio": f"{static_tokens / total * 100:.1f}%" if total > 0 else "0%",
            "cacheable": static_tokens >= 1024
        }


# Usage
prompt = (
    CacheOptimizedPrompt()
    .add_static("You are an expert Python developer and educator.", "Role")
    .add_static("Always provide working code examples with type hints.", "Guidelines")
    .add_static("Python 3.12 reference documentation..." * 50, "Knowledge Base")
    .add_dynamic(f"Current session started: {datetime.now()}", "Session")
    .add_dynamic("User is working on a web scraping project", "Context")
)

print(prompt.get_cache_stats())
print("\nPrompt preview:")
print(prompt.build()[:500] + "...")
```

---

## Message Structure Optimization

### Chat Completions Structure

```python
from typing import List, Dict

class OptimizedMessageBuilder:
    """Build message arrays optimized for caching."""
    
    def __init__(self):
        self.system_content: List[str] = []
        self.conversation_history: List[dict] = []
        self.current_query: str = ""
    
    def set_system(self, *contents: str) -> "OptimizedMessageBuilder":
        """Set system message components (order matters!)."""
        self.system_content = list(contents)
        return self
    
    def add_history(self, role: str, content: str) -> "OptimizedMessageBuilder":
        """Add conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        return self
    
    def set_query(self, query: str) -> "OptimizedMessageBuilder":
        """Set current user query."""
        self.current_query = query
        return self
    
    def build(self) -> List[dict]:
        """Build optimized message array."""
        
        messages = []
        
        # System message (static, cached)
        if self.system_content:
            messages.append({
                "role": "system",
                "content": "\n\n".join(self.system_content)
            })
        
        # Conversation history (partially cached)
        messages.extend(self.conversation_history)
        
        # Current query (dynamic)
        if self.current_query:
            messages.append({
                "role": "user",
                "content": self.current_query
            })
        
        return messages
    
    def analyze_cacheability(self) -> dict:
        """Analyze cache potential of messages."""
        
        system_tokens = sum(len(c) // 4 for c in self.system_content)
        history_tokens = sum(len(m["content"]) // 4 for m in self.conversation_history)
        query_tokens = len(self.current_query) // 4
        
        # Prefix = system + history (potentially cached)
        prefix_tokens = system_tokens + history_tokens
        
        return {
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "query_tokens": query_tokens,
            "prefix_tokens": prefix_tokens,
            "cache_potential": f"{prefix_tokens / (prefix_tokens + query_tokens) * 100:.1f}%"
                if (prefix_tokens + query_tokens) > 0 else "0%"
        }


# Usage
builder = (
    OptimizedMessageBuilder()
    .set_system(
        "You are an expert Python tutor.",
        "Always provide code examples.",
        "Reference: " + "Python documentation..." * 100
    )
    .add_history("user", "What is a decorator?")
    .add_history("assistant", "A decorator is a function that modifies another function...")
    .set_query("Show me an example of a decorator with arguments")
)

messages = builder.build()
print(f"Message count: {len(messages)}")
print(builder.analyze_cacheability())
```

### Anthropic Block Optimization

```python
class AnthropicBlockBuilder:
    """Build Anthropic system blocks for optimal caching."""
    
    def __init__(self, ttl_hours: int = 1):
        self.ttl = ttl_hours * 3600
        self.static_blocks: List[dict] = []
        self.dynamic_blocks: List[dict] = []
    
    def add_static_text(self, text: str) -> "AnthropicBlockBuilder":
        """Add static text block (cached)."""
        self.static_blocks.append({
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral", "ttl": self.ttl}
        })
        return self
    
    def add_dynamic_text(self, text: str) -> "AnthropicBlockBuilder":
        """Add dynamic text block (not cached)."""
        self.dynamic_blocks.append({
            "type": "text",
            "text": text
        })
        return self
    
    def build(self) -> List[dict]:
        """Build system blocks array."""
        # Static blocks first, dynamic blocks last
        return self.static_blocks + self.dynamic_blocks
    
    def get_cache_summary(self) -> dict:
        """Get cache configuration summary."""
        return {
            "cached_blocks": len(self.static_blocks),
            "dynamic_blocks": len(self.dynamic_blocks),
            "ttl_seconds": self.ttl,
            "cached_tokens": sum(len(b["text"]) // 4 for b in self.static_blocks),
            "dynamic_tokens": sum(len(b["text"]) // 4 for b in self.dynamic_blocks)
        }


# Usage
blocks = (
    AnthropicBlockBuilder(ttl_hours=1)
    .add_static_text("You are an expert assistant. " * 100)
    .add_static_text("Knowledge base: " + "Reference content..." * 200)
    .add_dynamic_text(f"Current time: {datetime.now().isoformat()}")
    .add_dynamic_text("User context: Premium subscription")
)

system_blocks = blocks.build()
print(f"Total blocks: {len(system_blocks)}")
print(blocks.get_cache_summary())
```

---

## Tool Definition Consistency

### Consistent Tool Ordering

```python
from typing import Dict, Any
import json
import hashlib

class ConsistentToolSet:
    """Maintain consistent tool definitions for caching."""
    
    def __init__(self):
        self._tools: Dict[str, dict] = {}
        self._version = "1.0.0"
    
    def add_tool(
        self,
        name: str,
        description: str,
        parameters: dict
    ) -> "ConsistentToolSet":
        """Add tool with validation."""
        
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        return self
    
    def get_tools(self) -> List[dict]:
        """Get tools in consistent order (sorted by name)."""
        return [self._tools[name] for name in sorted(self._tools.keys())]
    
    def get_cache_key(self) -> str:
        """Get stable cache key for tool set."""
        # Sorted, consistent JSON
        tools_json = json.dumps(self.get_tools(), sort_keys=True)
        hash_val = hashlib.sha256(tools_json.encode()).hexdigest()[:12]
        return f"tools_v{self._version}_{hash_val}"
    
    def bump_version(self):
        """Bump version when tools change."""
        parts = self._version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self._version = ".".join(parts)


# Usage
tools = (
    ConsistentToolSet()
    .add_tool(
        "search_docs",
        "Search documentation",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    )
    .add_tool(
        "get_weather",
        "Get weather for location",
        {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
    )
    .add_tool(
        "calculate",
        "Perform calculation",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    )
)

# Tools always in same order
tool_list = tools.get_tools()
print(f"Tool count: {len(tool_list)}")
print(f"Cache key: {tools.get_cache_key()}")
print(f"First tool: {tool_list[0]['function']['name']}")  # Always 'calculate'
```

### Schema Consistency

```python
class ConsistentSchemaRegistry:
    """Maintain consistent schema definitions."""
    
    def __init__(self):
        self._schemas: Dict[str, dict] = {}
    
    def register(self, name: str, schema: dict) -> "ConsistentSchemaRegistry":
        """Register schema."""
        # Normalize schema for consistency
        self._schemas[name] = self._normalize_schema(schema)
        return self
    
    def _normalize_schema(self, schema: dict) -> dict:
        """Normalize schema for consistent representation."""
        if not isinstance(schema, dict):
            return schema
        
        # Sort keys for consistency
        normalized = {}
        for key in sorted(schema.keys()):
            value = schema[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_schema(value)
            elif isinstance(value, list):
                normalized[key] = [
                    self._normalize_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                normalized[key] = value
        
        return normalized
    
    def get_schema(self, name: str) -> Optional[dict]:
        """Get normalized schema."""
        return self._schemas.get(name)
    
    def get_openai_format(self, name: str) -> dict:
        """Get schema in OpenAI structured output format."""
        schema = self.get_schema(name)
        if not schema:
            raise ValueError(f"Schema '{name}' not found")
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": True
            }
        }


# Usage
registry = ConsistentSchemaRegistry()

registry.register("analysis_result", {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["summary", "confidence"]
})

schema = registry.get_openai_format("analysis_result")
print(json.dumps(schema, indent=2))
```

---

## Multi-Turn Conversation Design

### Stable Prefix Strategy

```python
class CacheAwareConversation:
    """Conversation manager optimized for caching."""
    
    def __init__(
        self,
        system_prompt: str,
        max_history_tokens: int = 4000
    ):
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.messages: List[dict] = []
        self._cache_boundary = 0  # Index where cache ends
    
    def add_turn(self, user_message: str, assistant_response: str):
        """Add a conversation turn."""
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_response})
        
        # Update cache boundary
        self._update_cache_boundary()
    
    def _update_cache_boundary(self):
        """Update where cached prefix ends."""
        # Cache boundary is the last exchange that fits in cache window
        total_tokens = 0
        
        for i in range(len(self.messages) - 2, -1, -2):
            exchange_tokens = (
                len(self.messages[i]["content"]) // 4 +
                len(self.messages[i + 1]["content"]) // 4
            )
            
            if total_tokens + exchange_tokens > self.max_history_tokens:
                break
            
            total_tokens += exchange_tokens
            self._cache_boundary = i
    
    def get_messages_for_request(self, current_query: str) -> List[dict]:
        """Get messages optimized for cache hit."""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Include conversation history
        messages.extend(self.messages)
        
        # Add current query
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def get_cache_analysis(self) -> dict:
        """Analyze cache potential."""
        
        system_tokens = len(self.system_prompt) // 4
        history_tokens = sum(len(m["content"]) // 4 for m in self.messages)
        cached_tokens = system_tokens + sum(
            len(m["content"]) // 4 
            for m in self.messages[:self._cache_boundary]
        )
        
        return {
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "cached_tokens": cached_tokens,
            "cache_boundary": self._cache_boundary,
            "total_turns": len(self.messages) // 2
        }


# Usage
convo = CacheAwareConversation(
    system_prompt="You are an expert Python tutor. " * 100,  # Large system prompt
    max_history_tokens=2000
)

# Add conversation history
convo.add_turn("What is a list?", "A list is a mutable sequence...")
convo.add_turn("How do I sort?", "Use the sort() method or sorted()...")
convo.add_turn("What about reverse?", "Use reverse() or [::-1]...")

# Get analysis
analysis = convo.get_cache_analysis()
print(f"Cached tokens: {analysis['cached_tokens']}")
print(f"Cache boundary at turn: {analysis['cache_boundary'] // 2}")
```

### Sliding Window with Cache

```python
class SlidingWindowConversation:
    """Conversation with sliding window for cache optimization."""
    
    def __init__(
        self,
        system_prompt: str,
        window_size: int = 10,  # Number of turns to keep
        summary_threshold: int = 5  # When to summarize
    ):
        self.system_prompt = system_prompt
        self.window_size = window_size
        self.summary_threshold = summary_threshold
        
        self.recent_messages: List[dict] = []
        self.summary: str = ""
    
    def add_turn(self, user: str, assistant: str):
        """Add a turn, maintaining window."""
        
        self.recent_messages.append({"role": "user", "content": user})
        self.recent_messages.append({"role": "assistant", "content": assistant})
        
        # Check if we need to slide window
        turns = len(self.recent_messages) // 2
        if turns > self.window_size:
            self._slide_window()
    
    def _slide_window(self):
        """Slide window and update summary."""
        
        # Messages to summarize
        to_summarize = self.recent_messages[:self.summary_threshold * 2]
        
        # Update summary (in practice, use LLM to summarize)
        summary_text = f"Previous discussion covered: "
        for msg in to_summarize:
            if msg["role"] == "user":
                summary_text += f"{msg['content'][:50]}... "
        
        self.summary = summary_text
        
        # Keep remaining messages
        self.recent_messages = self.recent_messages[self.summary_threshold * 2:]
    
    def get_messages(self, query: str) -> List[dict]:
        """Get optimized messages for request."""
        
        messages = []
        
        # System prompt (cached)
        system_content = self.system_prompt
        if self.summary:
            system_content += f"\n\nConversation context: {self.summary}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Recent messages (potentially cached)
        messages.extend(self.recent_messages)
        
        # Current query (dynamic)
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def get_stats(self) -> dict:
        """Get conversation statistics."""
        return {
            "recent_turns": len(self.recent_messages) // 2,
            "has_summary": bool(self.summary),
            "system_tokens": len(self.system_prompt) // 4,
            "summary_tokens": len(self.summary) // 4,
            "history_tokens": sum(len(m["content"]) // 4 for m in self.recent_messages)
        }


# Usage
sliding = SlidingWindowConversation(
    system_prompt="Expert Python tutor..." * 100,
    window_size=5,
    summary_threshold=3
)

# Simulate long conversation
for i in range(8):
    sliding.add_turn(f"Question {i}", f"Answer {i} with details...")

stats = sliding.get_stats()
print(f"Recent turns: {stats['recent_turns']}")
print(f"Has summary: {stats['has_summary']}")
```

---

## Request Ordering Strategy

### Optimized Request Builder

```python
@dataclass
class RequestComponent:
    """Component of an API request."""
    
    name: str
    content: Any
    is_static: bool
    tokens: int
    order: int  # Lower = earlier


class OptimizedRequestBuilder:
    """Build API requests optimized for caching."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.components: List[RequestComponent] = []
    
    def add_component(
        self,
        name: str,
        content: Any,
        is_static: bool = True,
        tokens: int = 0
    ) -> "OptimizedRequestBuilder":
        """Add a request component."""
        
        # Calculate tokens if not provided
        if tokens == 0:
            if isinstance(content, str):
                tokens = len(content) // 4
            elif isinstance(content, list):
                tokens = len(json.dumps(content)) // 4
            elif isinstance(content, dict):
                tokens = len(json.dumps(content)) // 4
        
        # Order: static components first
        order = 0 if is_static else 100
        
        self.components.append(RequestComponent(
            name=name,
            content=content,
            is_static=is_static,
            tokens=tokens,
            order=order
        ))
        
        return self
    
    def build_openai_request(self) -> dict:
        """Build OpenAI Chat Completions request."""
        
        # Sort components
        sorted_components = sorted(self.components, key=lambda c: c.order)
        
        request = {"messages": []}
        
        for comp in sorted_components:
            if comp.name == "system":
                request["messages"].insert(0, {
                    "role": "system",
                    "content": comp.content
                })
            elif comp.name == "user":
                request["messages"].append({
                    "role": "user",
                    "content": comp.content
                })
            elif comp.name == "tools":
                request["tools"] = comp.content
            elif comp.name == "response_format":
                request["response_format"] = comp.content
        
        return request
    
    def build_anthropic_request(self) -> dict:
        """Build Anthropic request with caching."""
        
        sorted_components = sorted(self.components, key=lambda c: c.order)
        
        request = {"system": [], "messages": []}
        
        for comp in sorted_components:
            if comp.name == "system":
                block = {"type": "text", "text": comp.content}
                if comp.is_static:
                    block["cache_control"] = {"type": "ephemeral"}
                request["system"].append(block)
            elif comp.name == "user":
                request["messages"].append({
                    "role": "user",
                    "content": comp.content
                })
            elif comp.name == "tools":
                request["tools"] = comp.content
        
        return request
    
    def get_cache_analysis(self) -> dict:
        """Analyze cache potential."""
        
        static_tokens = sum(c.tokens for c in self.components if c.is_static)
        dynamic_tokens = sum(c.tokens for c in self.components if not c.is_static)
        total = static_tokens + dynamic_tokens
        
        return {
            "static_components": sum(1 for c in self.components if c.is_static),
            "dynamic_components": sum(1 for c in self.components if not c.is_static),
            "static_tokens": static_tokens,
            "dynamic_tokens": dynamic_tokens,
            "cache_ratio": f"{static_tokens / total * 100:.1f}%" if total > 0 else "0%",
            "cacheable": static_tokens >= 1024
        }


# Usage
builder = (
    OptimizedRequestBuilder(provider="openai")
    .add_component("system", "You are an expert Python tutor. " * 100, is_static=True)
    .add_component("tools", [
        {"type": "function", "function": {"name": "search", "parameters": {}}}
    ], is_static=True)
    .add_component("user", "What is a decorator?", is_static=False)
)

request = builder.build_openai_request()
print("Request keys:", list(request.keys()))
print("Cache analysis:", builder.get_cache_analysis())
```

---

## Hands-on Exercise

### Your Task

Build a cache-optimized prompt management system.

### Requirements

1. Organize content by static/dynamic classification
2. Automatically order for maximum caching
3. Generate cache keys for content
4. Provide optimization recommendations

<details>
<summary>💡 Hints</summary>

- Use consistent hashing for cache keys
- Sort static content first
- Track token counts per section
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import hashlib
import json

class ContentType(Enum):
    SYSTEM = "system"
    KNOWLEDGE = "knowledge"
    TOOLS = "tools"
    SCHEMA = "schema"
    HISTORY = "history"
    CONTEXT = "context"
    QUERY = "query"


@dataclass
class ContentSection:
    """Section of content with caching metadata."""
    
    name: str
    content_type: ContentType
    content: Any
    is_static: bool
    tokens: int = 0
    hash: str = ""
    
    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = self._estimate_tokens()
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _estimate_tokens(self) -> int:
        if isinstance(self.content, str):
            return len(self.content) // 4
        return len(json.dumps(self.content)) // 4
    
    def _compute_hash(self) -> str:
        if isinstance(self.content, str):
            data = self.content
        else:
            data = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class CacheOptimizedPromptManager:
    """Manage prompts for optimal caching."""
    
    # Order priority by content type
    TYPE_PRIORITY = {
        ContentType.SYSTEM: 1,
        ContentType.KNOWLEDGE: 2,
        ContentType.TOOLS: 3,
        ContentType.SCHEMA: 4,
        ContentType.HISTORY: 5,
        ContentType.CONTEXT: 6,
        ContentType.QUERY: 7,
    }
    
    # Whether content type is typically static
    TYPE_STATIC = {
        ContentType.SYSTEM: True,
        ContentType.KNOWLEDGE: True,
        ContentType.TOOLS: True,
        ContentType.SCHEMA: True,
        ContentType.HISTORY: False,  # Grows over time
        ContentType.CONTEXT: False,
        ContentType.QUERY: False,
    }
    
    def __init__(self):
        self.sections: List[ContentSection] = []
    
    def add(
        self,
        name: str,
        content_type: ContentType,
        content: Any,
        is_static: Optional[bool] = None
    ) -> "CacheOptimizedPromptManager":
        """Add a content section."""
        
        # Use default static classification if not provided
        if is_static is None:
            is_static = self.TYPE_STATIC.get(content_type, False)
        
        self.sections.append(ContentSection(
            name=name,
            content_type=content_type,
            content=content,
            is_static=is_static
        ))
        
        return self
    
    def get_ordered_sections(self) -> List[ContentSection]:
        """Get sections in cache-optimized order."""
        
        def sort_key(section: ContentSection) -> tuple:
            # Static first, then by type priority
            return (
                not section.is_static,
                self.TYPE_PRIORITY.get(section.content_type, 99)
            )
        
        return sorted(self.sections, key=sort_key)
    
    def build_system_prompt(self) -> str:
        """Build optimized system prompt string."""
        
        ordered = self.get_ordered_sections()
        
        parts = []
        for section in ordered:
            if section.content_type in [ContentType.SYSTEM, ContentType.KNOWLEDGE]:
                if isinstance(section.content, str):
                    parts.append(f"## {section.name}\n{section.content}")
        
        return "\n\n".join(parts)
    
    def build_openai_request(self, query: str, model: str = "gpt-4o") -> dict:
        """Build complete OpenAI request."""
        
        ordered = self.get_ordered_sections()
        
        # Combine system content
        system_parts = []
        for section in ordered:
            if section.content_type in [ContentType.SYSTEM, ContentType.KNOWLEDGE]:
                if isinstance(section.content, str):
                    system_parts.append(section.content)
        
        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": query}
        ]
        
        request = {"model": model, "messages": messages}
        
        # Add tools if present
        for section in ordered:
            if section.content_type == ContentType.TOOLS:
                request["tools"] = section.content
            elif section.content_type == ContentType.SCHEMA:
                request["response_format"] = section.content
        
        return request
    
    def build_anthropic_request(
        self,
        query: str,
        model: str = "claude-sonnet-4-20250514"
    ) -> dict:
        """Build complete Anthropic request with caching."""
        
        ordered = self.get_ordered_sections()
        
        system_blocks = []
        tools = []
        
        for section in ordered:
            if section.content_type in [ContentType.SYSTEM, ContentType.KNOWLEDGE]:
                block = {"type": "text", "text": section.content}
                if section.is_static:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
            elif section.content_type == ContentType.TOOLS:
                tools = section.content
        
        request = {
            "model": model,
            "max_tokens": 1024,
            "system": system_blocks,
            "messages": [{"role": "user", "content": query}]
        }
        
        if tools:
            # Add cache control to last tool
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            request["tools"] = tools
        
        return request
    
    def get_cache_key(self) -> str:
        """Generate cache key for static content."""
        
        static_hashes = sorted(
            s.hash for s in self.sections if s.is_static
        )
        
        combined = "|".join(static_hashes)
        return f"prompt_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def get_analysis(self) -> dict:
        """Analyze cache optimization."""
        
        ordered = self.get_ordered_sections()
        
        static_tokens = sum(s.tokens for s in ordered if s.is_static)
        dynamic_tokens = sum(s.tokens for s in ordered if not s.is_static)
        total = static_tokens + dynamic_tokens
        
        by_type = {}
        for section in ordered:
            type_name = section.content_type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "tokens": 0, "static": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["tokens"] += section.tokens
            if section.is_static:
                by_type[type_name]["static"] += section.tokens
        
        return {
            "total_sections": len(ordered),
            "static_sections": sum(1 for s in ordered if s.is_static),
            "dynamic_sections": sum(1 for s in ordered if not s.is_static),
            "static_tokens": static_tokens,
            "dynamic_tokens": dynamic_tokens,
            "total_tokens": total,
            "cache_ratio": f"{static_tokens / total * 100:.1f}%" if total > 0 else "0%",
            "cacheable": static_tokens >= 1024,
            "by_type": by_type,
            "cache_key": self.get_cache_key()
        }
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        
        recommendations = []
        analysis = self.get_analysis()
        
        # Check cache threshold
        if not analysis["cacheable"]:
            deficit = 1024 - analysis["static_tokens"]
            recommendations.append(
                f"Add ~{deficit} more static tokens to enable caching"
            )
        
        # Check cache ratio
        ratio = float(analysis["cache_ratio"].rstrip("%"))
        if ratio < 50:
            recommendations.append(
                "Low cache ratio. Move more content to static sections."
            )
        elif ratio >= 80:
            recommendations.append(
                "Excellent cache ratio! Prompt is well-optimized."
            )
        
        # Check content organization
        by_type = analysis["by_type"]
        
        if "context" in by_type and by_type["context"]["static"] > 0:
            recommendations.append(
                "Context marked as static but typically changes. Verify classification."
            )
        
        if "knowledge" in by_type and by_type["knowledge"]["tokens"] < 500:
            recommendations.append(
                "Consider adding more knowledge content for better caching benefit."
            )
        
        # Check section count
        if analysis["total_sections"] > 10:
            recommendations.append(
                "Many sections. Consider consolidating for simpler cache key."
            )
        
        return recommendations if recommendations else ["No optimizations needed"]


# Usage example
manager = CacheOptimizedPromptManager()

# Add content sections
manager.add(
    "base_instructions",
    ContentType.SYSTEM,
    "You are an expert Python developer and educator. " * 50
)

manager.add(
    "documentation",
    ContentType.KNOWLEDGE,
    "Python 3.12 Reference: " + "Documentation content..." * 100
)

manager.add(
    "tools",
    ContentType.TOOLS,
    [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search documentation",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }
    ]
)

manager.add(
    "user_context",
    ContentType.CONTEXT,
    f"Session started: {datetime.now().isoformat()}"
)

# Analyze
print("=== Cache Analysis ===")
analysis = manager.get_analysis()
for key, value in analysis.items():
    if key == "by_type":
        print(f"\n{key}:")
        for t, stats in value.items():
            print(f"  {t}: {stats}")
    else:
        print(f"{key}: {value}")

print("\n=== Recommendations ===")
for rec in manager.get_recommendations():
    print(f"• {rec}")

print("\n=== OpenAI Request Preview ===")
request = manager.build_openai_request("What is a decorator?")
print(f"Messages: {len(request['messages'])}")
print(f"Has tools: {'tools' in request}")

print("\n=== Anthropic Request Preview ===")
request = manager.build_anthropic_request("What is a decorator?")
print(f"System blocks: {len(request['system'])}")
cached_blocks = sum(1 for b in request["system"] if "cache_control" in b)
print(f"Cached blocks: {cached_blocks}")
```

</details>

---

## Summary

✅ Static content first, dynamic content last  
✅ Consistent tool ordering with sorted names  
✅ Stable schema representations for caching  
✅ Sliding window for long conversations  
✅ Cache keys based on content hashes

**Next:** [Monitoring Performance](./06-monitoring-performance.md)

---

## Further Reading

- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — Best practices
- [Anthropic Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Design patterns
- [Cache Optimization](https://platform.openai.com/docs/guides/latency-optimization) — Latency tips
