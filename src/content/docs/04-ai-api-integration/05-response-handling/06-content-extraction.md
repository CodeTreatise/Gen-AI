---
title: "Content Extraction Patterns"
---

# Content Extraction Patterns

## Introduction

Extracting content from API responses requires careful handling of nested structures, optional fields, and varying response formats. This lesson covers safe navigation patterns and robust extraction strategies.

### What We'll Cover

- Safe navigation with optional chaining
- Default values for missing content
- Type validation techniques
- Content normalization
- Multi-provider extraction

### Prerequisites

- Response structure knowledge
- Python/JavaScript fundamentals

---

## Safe Navigation

### The Problem

Responses can have missing or null fields:

```python
# This can crash!
content = response.choices[0].message.content  # What if content is None?
text = response.output[0].content[0].text      # What if arrays are empty?
```

### Python Safe Access

```python
from typing import Optional

def safe_get_content(response) -> Optional[str]:
    """Safely extract content from Chat Completions response."""
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        
        message = getattr(choices[0], "message", None)
        if not message:
            return None
        
        return getattr(message, "content", None)
    except (IndexError, AttributeError):
        return None


# Using getattr with defaults
content = getattr(
    getattr(
        response.choices[0] if response.choices else None,
        "message", 
        None
    ),
    "content",
    ""
)
```

### Python Optional Chaining Pattern

```python
def get_nested(obj, *keys, default=None):
    """Navigate nested attributes/keys safely."""
    current = obj
    
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                current = current[key] if len(current) > key else None
            else:
                current = getattr(current, key, None)
            
            if current is None:
                return default
        except (IndexError, KeyError, TypeError):
            return default
    
    return current


# Usage
content = get_nested(response, "choices", 0, "message", "content", default="")
tokens = get_nested(response, "usage", "total_tokens", default=0)
```

### JavaScript Optional Chaining

```javascript
// Native optional chaining (?.)
const content = response?.choices?.[0]?.message?.content ?? "";
const tokens = response?.usage?.total_tokens ?? 0;

// With nullish coalescing for defaults
function extractContent(response) {
    return response?.choices?.[0]?.message?.content 
        ?? response?.output?.[0]?.content?.[0]?.text 
        ?? "";
}
```

---

## Default Values

### Type-Safe Defaults

```python
from typing import TypeVar, Optional
from dataclasses import dataclass

T = TypeVar('T')

def with_default(value: Optional[T], default: T) -> T:
    """Return value if not None, otherwise default."""
    return value if value is not None else default


@dataclass
class ExtractedResponse:
    content: str = ""
    finish_reason: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def extract_with_defaults(response) -> ExtractedResponse:
    """Extract response fields with safe defaults."""
    choice = response.choices[0] if response.choices else None
    
    return ExtractedResponse(
        content=with_default(
            getattr(getattr(choice, "message", None), "content", None),
            ""
        ),
        finish_reason=with_default(
            getattr(choice, "finish_reason", None),
            "unknown"
        ),
        input_tokens=with_default(
            getattr(response.usage, "prompt_tokens", None),
            0
        ),
        output_tokens=with_default(
            getattr(response.usage, "completion_tokens", None),
            0
        ),
        model=with_default(response.model, "")
    )
```

### Context-Aware Defaults

```python
def extract_content_with_context(
    response,
    context: str = "general",
    fallback_message: str = "No response generated."
) -> str:
    """Extract content with context-appropriate defaults."""
    
    content = safe_get_content(response)
    
    if content:
        return content
    
    # Check for refusal
    message = response.choices[0].message if response.choices else None
    if message and hasattr(message, "refusal") and message.refusal:
        return f"[Declined: {message.refusal}]"
    
    # Check finish reason for context
    finish_reason = response.choices[0].finish_reason if response.choices else None
    
    if finish_reason == "content_filter":
        return "[Response filtered due to content policy]"
    
    if finish_reason == "length":
        return "[Response truncated - no content captured]"
    
    return fallback_message
```

---

## Type Validation

### Runtime Type Checking

```python
from typing import Any

def validate_string(value: Any, field_name: str = "field") -> str:
    """Validate and convert value to string."""
    if value is None:
        return ""
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float, bool)):
        return str(value)
    
    if isinstance(value, (list, dict)):
        import json
        return json.dumps(value)
    
    raise TypeError(f"{field_name} must be string-like, got {type(value)}")


def validate_content(response) -> str:
    """Extract and validate content as string."""
    raw_content = safe_get_content(response)
    return validate_string(raw_content, "content")
```

### Pydantic Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    
    @validator("text", pre=True, always=True)
    def ensure_text(cls, v, values):
        if values.get("type") == "output_text" and v is None:
            return ""
        return v

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    
    @validator("content", pre=True)
    def normalize_content(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            # Handle content blocks
            return " ".join(
                block.get("text", "") 
                for block in v 
                if isinstance(block, dict)
            )
        return str(v)

class ValidatedResponse(BaseModel):
    id: str
    model: str
    content: str = ""
    finish_reason: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    
    @classmethod
    def from_chat_completion(cls, response) -> "ValidatedResponse":
        choice = response.choices[0] if response.choices else None
        
        return cls(
            id=response.id,
            model=response.model,
            content=choice.message.content if choice else "",
            finish_reason=choice.finish_reason if choice else "unknown",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0
        )
```

---

## Content Normalization

### Whitespace Handling

```python
import re

def normalize_content(content: str) -> str:
    """Normalize whitespace and formatting in content."""
    if not content:
        return ""
    
    # Replace multiple whitespace with single space
    content = re.sub(r'\s+', ' ', content)
    
    # Preserve paragraph breaks
    content = re.sub(r' *\n *\n *', '\n\n', content)
    
    # Trim leading/trailing whitespace
    content = content.strip()
    
    # Remove trailing whitespace from each line
    content = '\n'.join(line.rstrip() for line in content.split('\n'))
    
    return content


def normalize_code_blocks(content: str) -> str:
    """Ensure code blocks are properly formatted."""
    # Fix unclosed code blocks
    if content.count('```') % 2 != 0:
        content += '\n```'
    
    # Ensure newlines around code blocks
    content = re.sub(r'```(\w*)\n?', r'\n```\1\n', content)
    content = re.sub(r'\n?```(\s*\n)', r'\n```\1', content)
    
    return content.strip()
```

### Encoding Normalization

```python
def normalize_unicode(content: str) -> str:
    """Normalize Unicode characters."""
    import unicodedata
    
    # Normalize to NFC form
    content = unicodedata.normalize('NFC', content)
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    return content
```

---

## Multi-Provider Extraction

### Universal Extractor

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class Provider(Enum):
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"

@dataclass
class UnifiedContent:
    text: str
    role: str = "assistant"
    tool_calls: List[dict] = None
    refusal: Optional[str] = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class ContentExtractor:
    """Extract content from any provider's response format."""
    
    @staticmethod
    def extract(response, provider: Provider) -> UnifiedContent:
        extractors = {
            Provider.OPENAI_CHAT: ContentExtractor._extract_chat,
            Provider.OPENAI_RESPONSES: ContentExtractor._extract_responses,
            Provider.ANTHROPIC: ContentExtractor._extract_anthropic
        }
        
        extractor = extractors.get(provider)
        if not extractor:
            raise ValueError(f"Unknown provider: {provider}")
        
        return extractor(response)
    
    @staticmethod
    def _extract_chat(response) -> UnifiedContent:
        if not response.choices:
            return UnifiedContent(text="")
        
        message = response.choices[0].message
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
        
        return UnifiedContent(
            text=message.content or "",
            role=message.role,
            tool_calls=tool_calls,
            refusal=getattr(message, "refusal", None)
        )
    
    @staticmethod
    def _extract_responses(response) -> UnifiedContent:
        text_parts = []
        tool_calls = []
        
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text_parts.append(content.text)
                    elif content.type == "refusal":
                        return UnifiedContent(
                            text="",
                            refusal=content.refusal
                        )
            
            elif item.type == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments
                })
        
        return UnifiedContent(
            text="".join(text_parts),
            tool_calls=tool_calls
        )
    
    @staticmethod
    def _extract_anthropic(response) -> UnifiedContent:
        text_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        
        return UnifiedContent(
            text="".join(text_parts),
            role="assistant",
            tool_calls=tool_calls
        )
```

---

## JavaScript Implementation

```javascript
class ContentExtractor {
    static extract(response, provider = 'openai') {
        const extractors = {
            openai: this.extractOpenAI,
            openai_responses: this.extractResponses,
            anthropic: this.extractAnthropic
        };
        
        const extractor = extractors[provider];
        if (!extractor) {
            throw new Error(`Unknown provider: ${provider}`);
        }
        
        return extractor(response);
    }
    
    static extractOpenAI(response) {
        const message = response?.choices?.[0]?.message;
        
        if (!message) {
            return { text: "", role: "assistant", toolCalls: [] };
        }
        
        const toolCalls = (message.tool_calls || []).map(tc => ({
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments
        }));
        
        return {
            text: message.content ?? "",
            role: message.role,
            toolCalls,
            refusal: message.refusal ?? null
        };
    }
    
    static extractResponses(response) {
        const textParts = [];
        const toolCalls = [];
        
        for (const item of response?.output || []) {
            if (item.type === "message") {
                for (const content of item.content || []) {
                    if (content.type === "output_text") {
                        textParts.push(content.text);
                    }
                }
            } else if (item.type === "function_call") {
                toolCalls.push({
                    id: item.call_id,
                    name: item.name,
                    arguments: item.arguments
                });
            }
        }
        
        return {
            text: textParts.join(""),
            role: "assistant",
            toolCalls
        };
    }
    
    static extractAnthropic(response) {
        const textParts = [];
        const toolCalls = [];
        
        for (const block of response?.content || []) {
            if (block.type === "text") {
                textParts.push(block.text);
            } else if (block.type === "tool_use") {
                toolCalls.push({
                    id: block.id,
                    name: block.name,
                    arguments: block.input
                });
            }
        }
        
        return {
            text: textParts.join(""),
            role: "assistant",
            toolCalls
        };
    }
}

// Usage with optional chaining and defaults
function safeExtract(response, provider) {
    try {
        const result = ContentExtractor.extract(response, provider);
        return {
            success: true,
            ...result
        };
    } catch (error) {
        return {
            success: false,
            text: "",
            role: "assistant",
            toolCalls: [],
            error: error.message
        };
    }
}
```

---

## Defensive Extraction Wrapper

```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def safe_extraction(default_value=""):
    """Decorator for safe content extraction."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result if result is not None else default_value
            except Exception as e:
                logger.warning(f"Extraction failed: {e}")
                return default_value
        return wrapper
    return decorator


@safe_extraction(default_value="")
def extract_text(response) -> str:
    """Extract text content from response."""
    # Chat Completions
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    
    # Responses API
    if hasattr(response, "output"):
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text
    
    return ""


@safe_extraction(default_value=0)
def extract_tokens(response) -> int:
    """Extract total token count."""
    return response.usage.total_tokens
```

---

## Hands-on Exercise

### Your Task

Build a universal content extractor that safely handles all providers.

### Requirements

1. Extract text content from any provider
2. Handle missing/null fields gracefully
3. Normalize whitespace
4. Return consistent structure

### Expected Result

```python
result = extract_universal(response, provider="openai")
# {
#   "text": "Hello, how can I help?",
#   "role": "assistant",
#   "tool_calls": [],
#   "tokens": 25
# }
```

<details>
<summary>💡 Hints</summary>

- Use try/except for safety
- Check provider to choose extraction path
- Normalize text before returning
</details>

<details>
<summary>✅ Solution</summary>

```python
import re
from typing import Optional, List

def extract_universal(response, provider: str = "openai") -> dict:
    """Extract content from any provider's response."""
    
    result = {
        "text": "",
        "role": "assistant",
        "tool_calls": [],
        "tokens": 0,
        "error": None
    }
    
    try:
        # Extract based on provider
        if provider == "openai":
            result.update(extract_openai_chat(response))
        elif provider == "openai_responses":
            result.update(extract_openai_responses(response))
        elif provider == "anthropic":
            result.update(extract_anthropic(response))
        else:
            result["error"] = f"Unknown provider: {provider}"
        
        # Normalize text
        result["text"] = normalize_text(result["text"])
        
        # Extract tokens
        if hasattr(response, "usage"):
            result["tokens"] = getattr(
                response.usage, 
                "total_tokens", 
                0
            )
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def extract_openai_chat(response) -> dict:
    message = response.choices[0].message if response.choices else None
    
    if not message:
        return {"text": ""}
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments
            })
    
    return {
        "text": message.content or "",
        "role": message.role,
        "tool_calls": tool_calls
    }


def extract_openai_responses(response) -> dict:
    text_parts = []
    tool_calls = []
    
    for item in getattr(response, "output", []):
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text_parts.append(content.text)
        elif item.type == "function_call":
            tool_calls.append({
                "id": item.call_id,
                "name": item.name,
                "arguments": item.arguments
            })
    
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls
    }


def extract_anthropic(response) -> dict:
    text_parts = []
    tool_calls = []
    
    for block in getattr(response, "content", []):
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "arguments": block.input
            })
    
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls
    }


def normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# Test
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)

result = extract_universal(response, "openai")
print(f"Text: {result['text']}")
print(f"Tokens: {result['tokens']}")
```

</details>

---

## Summary

✅ Use optional chaining (`?.` in JS) or `getattr()` in Python for safe access  
✅ Always provide sensible default values for missing fields  
✅ Validate and normalize content before using  
✅ Build universal extractors for multi-provider applications  
✅ Wrap extraction in try/except for defensive coding

**Next:** [Response Metadata](./07-response-metadata.md)

---

## Further Reading

- [Python getattr](https://docs.python.org/3/library/functions.html#getattr) — Safe attribute access
- [JS Optional Chaining](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Optional_chaining) — MDN
- [Pydantic Validators](https://docs.pydantic.dev/latest/concepts/validators/) — Data validation

<!-- 
Sources Consulted:
- Python getattr: https://docs.python.org/3/library/functions.html#getattr
- MDN Optional Chaining: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Optional_chaining
-->
