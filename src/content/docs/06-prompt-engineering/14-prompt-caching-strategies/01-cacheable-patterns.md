---
title: "Identifying Cacheable Prompt Patterns"
---

# Identifying Cacheable Prompt Patterns

## Introduction

Not all prompts benefit equally from caching. The key to maximizing cache efficiency is recognizing which parts of your prompts are stable enough to cache and structuring them correctly. This lesson teaches you to identify cacheable patterns and design prompts for optimal cache utilization.

> **🔑 Key Insight:** Cache hits require an *exact prefix match*. Any difference in the static portion—even a single character—creates a cache miss.

### What We'll Cover

- Static vs dynamic prompt components
- Cacheable pattern recognition
- Template-based cache optimization
- Common patterns by use case
- Anti-patterns that break caching

### Prerequisites

- [Prompt Caching Overview](./00-prompt-caching-overview.md)
- Understanding of prompt structure

---

## Static vs Dynamic Content

### The Cacheability Spectrum

```mermaid
graph LR
    A[Never Changes] --> B[Rarely Changes] --> C[Changes Daily] --> D[Changes Per Request]
    
    style A fill:#90EE90
    style B fill:#90EE90
    style C fill:#FFFF00
    style D fill:#FFB6C1
```

| Category | Examples | Cacheability |
|----------|----------|--------------|
| **Never changes** | System prompt, tool definitions, company guidelines | ✅ Excellent |
| **Rarely changes** | Reference docs, few-shot examples, templates | ✅ Good |
| **Changes daily** | Daily reports, fresh data, updated context | ⚠️ Moderate |
| **Changes per request** | User query, timestamps, user IDs | ❌ Don't cache |

### Anatomy of a Cacheable Prompt

```python
# ┌────────────────────────────────────────────┐
# │           CACHE-FRIENDLY PROMPT            │
# ├────────────────────────────────────────────┤
# │                                            │
# │  ┌──────────────────────────────────┐      │
# │  │ SYSTEM INSTRUCTIONS              │      │
# │  │ - Role definition                │      │
# │  │ - Output format requirements     │ ← CACHE THIS
# │  │ - Constraints and rules          │      │
# │  └──────────────────────────────────┘      │
# │                 ↓                          │
# │  ┌──────────────────────────────────┐      │
# │  │ TOOL DEFINITIONS                 │      │
# │  │ - Function schemas               │ ← CACHE THIS
# │  │ - Available actions              │      │
# │  └──────────────────────────────────┘      │
# │                 ↓                          │
# │  ┌──────────────────────────────────┐      │
# │  │ REFERENCE CONTEXT                │      │
# │  │ - Documents                      │ ← CACHE THIS
# │  │ - Examples                       │      │
# │  │ - Background info                │      │
# │  └──────────────────────────────────┘      │
# │                 ↓                          │
# │  ═══════════ CACHE BREAKPOINT ═══════════  │
# │                 ↓                          │
# │  ┌──────────────────────────────────┐      │
# │  │ DYNAMIC CONTENT                  │      │
# │  │ - Conversation history           │ ← DON'T CACHE
# │  │ - Current user query             │      │
# │  │ - Session-specific data          │      │
# │  └──────────────────────────────────┘      │
# │                                            │
# └────────────────────────────────────────────┘
```

---

## Pattern Recognition

### Pattern 1: Large Static System Prompt

**Scenario:** Customer service bot with extensive guidelines

```python
# ❌ BAD: Dynamic content mixed with static
system_prompt = f"""
You are a customer service agent for Acme Corp.
Today's date is {datetime.now()}.  # ← BREAKS CACHE
Current promotions: {get_promotions()}  # ← BREAKS CACHE

Guidelines:
1. Always greet customers warmly...
[2000 more tokens of static instructions]
"""

# ✅ GOOD: Static first, dynamic separate
static_system_prompt = """
You are a customer service agent for Acme Corp.

Guidelines:
1. Always greet customers warmly...
[2000 tokens of static instructions]
"""  # ← This gets cached

dynamic_context = f"""
Context for this conversation:
- Current date: {datetime.now()}
- Active promotions: {get_promotions()}
"""  # ← This goes after cache breakpoint
```

### Pattern 2: Document Q&A

**Scenario:** Answering questions about a large document

```python
# Cache-optimized structure
def create_document_qa_prompt(document: str, question: str) -> list:
    return [
        {
            "type": "text",
            "text": "You are a document analyst. Answer questions based only on the provided document."
        },
        {
            "type": "text",
            "text": f"DOCUMENT:\n{document}",
            "cache_control": {"type": "ephemeral"}  # Cache the document
        },
        # After breakpoint - not cached
        {
            "type": "text", 
            "text": f"QUESTION: {question}"
        }
    ]

# First request: Cache miss (writes document to cache)
# Subsequent requests: Cache hit (90% savings on document tokens)
```

### Pattern 3: Tool-Heavy Agent

**Scenario:** Agent with 20+ tool definitions

```python
# Tools are ideal for caching - they rarely change
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "input_schema": {...}
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "input_schema": {...}
    },
    # ... 18 more tools
]

# Structure for caching
response = client.messages.create(
    model="claude-sonnet-4-5",
    tools=tools,  # Cached first in the hierarchy
    system=[
        {
            "type": "text",
            "text": agent_instructions,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=conversation_history
)
```

### Pattern 4: Few-Shot Examples

**Scenario:** Consistent examples across all requests

```python
# Few-shot examples are perfect for caching
few_shot_examples = """
Example 1:
Input: "The food was amazing but service was slow"
Output: {"sentiment": "mixed", "food": "positive", "service": "negative"}

Example 2:
Input: "Terrible experience, never coming back"
Output: {"sentiment": "negative", "food": null, "service": null}

Example 3:
Input: "Quick service and delicious meals"
Output: {"sentiment": "positive", "food": "positive", "service": "positive"}
"""

# Cache structure
system = [
    {"type": "text", "text": "Analyze restaurant reviews."},
    {
        "type": "text",
        "text": few_shot_examples,
        "cache_control": {"type": "ephemeral"}
    }
]
# User message comes after - different each time
```

### Pattern 5: Multi-Turn Conversation

**Scenario:** Long chat with stable history

```python
def build_conversation_prompt(history: list, new_message: str) -> dict:
    """Build prompt that maximizes cache hits for conversation."""
    
    # Strategy: Cache everything except the last user message
    cached_history = history[:-1] if history else []
    
    messages = []
    
    for i, msg in enumerate(cached_history):
        message = {"role": msg["role"], "content": msg["content"]}
        # Cache breakpoint at end of stable history
        if i == len(cached_history) - 1:
            message["content"] = [
                {
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        messages.append(message)
    
    # New message - not cached
    messages.append({"role": "user", "content": new_message})
    
    return {"messages": messages}
```

---

## Template-Based Caching

### Prompt Templates for Consistent Caching

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class CacheablePromptTemplate:
    """Template that separates static and dynamic content."""
    
    static_system: str
    static_context: str | None = None
    few_shot_examples: str | None = None
    
    def build(
        self,
        dynamic_context: str = None,
        user_message: str = None
    ) -> dict:
        """Build prompt with cache-optimized structure."""
        
        system_parts = []
        
        # Static instructions (always same)
        system_parts.append({
            "type": "text",
            "text": self.static_system
        })
        
        # Static context/examples (cacheable)
        if self.static_context:
            system_parts.append({
                "type": "text",
                "text": self.static_context
            })
        
        if self.few_shot_examples:
            system_parts.append({
                "type": "text",
                "text": self.few_shot_examples,
                "cache_control": {"type": "ephemeral"}  # Cache breakpoint
            })
        
        # Dynamic context (after cache breakpoint)
        messages = []
        if dynamic_context:
            messages.append({
                "role": "user",
                "content": f"Current context:\n{dynamic_context}"
            })
            messages.append({
                "role": "assistant", 
                "content": "I understand the context. How can I help?"
            })
        
        # User query
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })
        
        return {
            "system": system_parts,
            "messages": messages
        }

# Usage
template = CacheablePromptTemplate(
    static_system="You are a code review assistant...",
    static_context=code_style_guide,  # 5000 tokens
    few_shot_examples=example_reviews   # 3000 tokens
)

# Every request benefits from cached 8000 tokens
prompt = template.build(
    dynamic_context=f"Repository: {repo_name}",
    user_message="Review this PR: ..."
)
```

### Version-Aware Templates

```python
class VersionedPromptTemplate:
    """Template with version tracking for cache management."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self._static_content = None
        self._content_hash = None
    
    def set_static_content(self, content: str):
        """Set static content and compute hash for cache key."""
        import hashlib
        self._static_content = content
        self._content_hash = hashlib.sha256(
            content.encode()
        ).hexdigest()[:16]
    
    @property
    def cache_key(self) -> str:
        """Unique cache key for this template version."""
        return f"{self.name}:{self.version}:{self._content_hash}"
    
    def content_changed(self, new_content: str) -> bool:
        """Check if content has changed (would invalidate cache)."""
        import hashlib
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]
        return new_hash != self._content_hash
```

---

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Timestamp in System Prompt

```python
# BAD: Breaks cache every second
system = f"You are an assistant. Current time: {datetime.now()}"

# GOOD: Keep timestamps in user message
system = "You are an assistant."
user = f"Current time: {datetime.now()}. My question is: ..."
```

### ❌ Anti-Pattern 2: User-Specific Data in Cached Section

```python
# BAD: Different cache for every user
system = f"""
You are helping user {user.name} (ID: {user.id}).
User preferences: {user.preferences}
[Long static instructions...]
"""

# GOOD: Generic instructions cached, user data dynamic
system = "[Long static instructions...]"
user = f"""
User context:
- Name: {user.name}
- ID: {user.id}
- Preferences: {user.preferences}

Question: {question}
"""
```

### ❌ Anti-Pattern 3: Random or Rotating Examples

```python
# BAD: Different examples each request
examples = random.sample(all_examples, 5)
system = f"Examples:\n{format_examples(examples)}"

# GOOD: Fixed examples for consistent caching
examples = all_examples[:5]  # Always the same 5
system = f"Examples:\n{format_examples(examples)}"
```

### ❌ Anti-Pattern 4: Unstable JSON Key Order

```python
# BAD: Dict key order may vary (Python <3.7 or some serializers)
tool = {"name": "search", "description": "...", "params": {...}}

# GOOD: Ensure consistent ordering
import json
tool = json.dumps(tool_dict, sort_keys=True)
```

### ❌ Anti-Pattern 5: Including Request Metadata

```python
# BAD: Request ID changes every time
system = f"""
Request ID: {uuid4()}
Debug mode: {settings.debug}
[Instructions...]
"""

# GOOD: No per-request metadata in cached section
system = "[Instructions...]"
# Log request ID separately
```

---

## Use Case Patterns

### Chatbot/Customer Service

| Component | Placement | Cache Strategy |
|-----------|-----------|----------------|
| Brand guidelines | System (start) | Always cache |
| Knowledge base | System (middle) | Cache with breakpoint |
| Current promotions | User context | Don't cache |
| User message | User | Don't cache |

### RAG (Retrieval-Augmented Generation)

| Component | Placement | Cache Strategy |
|-----------|-----------|----------------|
| RAG instructions | System | Always cache |
| Retrieved chunks | System (end) | Cache if same query repeated |
| User query | User | Don't cache |

### Code Assistant

| Component | Placement | Cache Strategy |
|-----------|-----------|----------------|
| Coding guidelines | System | Always cache |
| Repository context | System | Cache per session |
| Current file | User context | Cache if editing same file |
| User request | User | Don't cache |

### Agentic Workflow

| Component | Placement | Cache Strategy |
|-----------|-----------|----------------|
| Tool definitions | Tools array | Always cache (first in hierarchy) |
| Agent instructions | System | Always cache |
| Task context | System (end) | Cache per task |
| Step history | Messages | Grows each step |

---

## Hands-on Exercise

### Your Task

Analyze the following prompt and restructure it for optimal caching:

```python
# Current (suboptimal) prompt
def create_prompt(user_id, question, documents):
    return f"""
You are a helpful assistant for user {user_id}.
Today is {datetime.now().strftime('%Y-%m-%d')}.

REFERENCE DOCUMENTS:
{documents}  # 50,000 tokens

INSTRUCTIONS:
1. Only answer based on the documents above
2. Cite your sources
3. If unsure, say so

Question: {question}
"""
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Identify what changes every request vs what stays the same
- Documents are large and likely reused for same user
- User ID and date break the cache
- Structure for Anthropic's cache_control

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
def create_optimized_prompt(user_id: str, question: str, documents: str) -> dict:
    """Cache-optimized prompt structure."""
    
    # Static instructions - never change
    static_instructions = """
INSTRUCTIONS:
1. Only answer based on the documents provided
2. Cite your sources with [Doc N] notation
3. If the answer isn't in the documents, say "I don't have that information"
"""
    
    return {
        "system": [
            # Static instructions (cacheable)
            {
                "type": "text",
                "text": static_instructions
            },
            # Documents - large, cacheable if same docs used
            {
                "type": "text",
                "text": f"REFERENCE DOCUMENTS:\n{documents}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            # Dynamic context - user-specific, date changes
            {
                "role": "user",
                "content": f"""
Context:
- User ID: {user_id}
- Date: {datetime.now().strftime('%Y-%m-%d')}

Question: {question}
"""
            }
        ]
    }

# Now when same documents are used:
# - First request: Cache miss (~50K tokens processed, cached)
# - Subsequent requests: Cache hit (90% savings on 50K tokens)

# Example API call
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    **create_optimized_prompt(user_id, question, documents)
)

# Monitor effectiveness
print(f"Cache read: {response.usage.cache_read_input_tokens}")
print(f"Cache write: {response.usage.cache_creation_input_tokens}")
print(f"Uncached: {response.usage.input_tokens}")
```

**Improvements:**
- Static instructions separated from dynamic data
- Documents cached with breakpoint
- User ID and date moved to user message
- 50K tokens cached instead of reprocessed

**Cost comparison (10 requests, Sonnet 4.5 @ $3/MTok):**
- Before: 10 × 50,000 tokens = 500K tokens × $3 = $1.50
- After: 1 write (62.5K cost-equivalent) + 9 reads (5K cost-equivalent each) = ~$0.32
- **Savings: ~79%**

</details>

---

## Summary

✅ **Separate static from dynamic:** Static content first, dynamic last
✅ **Avoid per-request data in cached sections:** No timestamps, user IDs, or random values
✅ **Use templates:** Ensure consistent structure across requests
✅ **Cache large contexts:** Documents, examples, and tool definitions benefit most
✅ **Monitor cache metrics:** Track hit rates to validate strategy
✅ **Match pattern to use case:** Different applications have different cacheable components

**Next:** [Provider Caching Features](./02-provider-features.md)

---

## Further Reading

- [Anthropic Prompt Caching Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI Prompt Structuring Guide](https://platform.openai.com/docs/guides/prompt-caching)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Caching: Structuring prompts with static content first
- Anthropic Prompt Caching: cache_control placement and hierarchy
- Practical patterns from production implementations
-->
