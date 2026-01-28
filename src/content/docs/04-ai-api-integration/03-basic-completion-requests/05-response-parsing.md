---
title: "Response Parsing"
---

# Response Parsing

## Introduction

AI API responses contain rich metadata beyond just the generated text. Understanding response structure helps you extract usage statistics, track performance, handle edge cases, and build robust applications.

### What We'll Cover

- Response object structure
- Extracting metadata (IDs, timestamps)
- Usage statistics and token counting
- Finish reasons and their meanings
- Provider-specific response formats
- `output_text` helper in SDKs

### Prerequisites

- Basic API request knowledge
- JSON parsing familiarity

---

## Response Object Structure

### OpenAI Responses API

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What is Python?"
)

# Response object structure
print(type(response))  # Response object
print(response.id)     # "resp_abc123..."
print(response.model)  # "gpt-4.1"
print(response.output) # List of output items
```

Full response structure:

```python
{
    "id": "resp_abc123...",
    "object": "response",
    "created_at": 1706000000,
    "model": "gpt-4.1",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Python is a programming language..."
                }
            ]
        }
    ],
    "usage": {
        "input_tokens": 15,
        "output_tokens": 150,
        "total_tokens": 165
    }
}
```

### OpenAI Chat Completions

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is Python?"}]
)

# Response structure
{
    "id": "chatcmpl-abc123...",
    "object": "chat.completion",
    "created": 1706000000,
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Python is..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 150,
        "total_tokens": 165
    }
}
```

### Anthropic Messages

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is Python?"}]
)

# Response structure
{
    "id": "msg_abc123...",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Python is..."
        }
    ],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 150
    }
}
```

---

## Extracting Response Metadata

### Response ID

```python
# Track responses for debugging and logging
response_id = response.id
print(f"Response ID: {response_id}")

# Use for support tickets or audit logs
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "response_id": response_id,
    "model": response.model
}
```

### Creation Timestamp

```python
from datetime import datetime

# OpenAI returns Unix timestamp
created_at = response.created_at  # or response.created
timestamp = datetime.fromtimestamp(created_at)
print(f"Created: {timestamp}")
```

### Model Information

```python
# Actual model used (may differ from requested)
print(f"Requested: gpt-4.1")
print(f"Actual: {response.model}")
# Could show versioned name like "gpt-4.1-2025-04-14"
```

---

## Usage Statistics

Track token usage for cost monitoring and optimization:

### OpenAI Usage

```python
usage = response.usage

print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total tokens: {usage.total_tokens}")

# Calculate cost (example rates)
INPUT_COST_PER_1K = 0.01  # $0.01 per 1K input tokens
OUTPUT_COST_PER_1K = 0.03  # $0.03 per 1K output tokens

cost = (usage.input_tokens / 1000 * INPUT_COST_PER_1K +
        usage.output_tokens / 1000 * OUTPUT_COST_PER_1K)
print(f"Estimated cost: ${cost:.4f}")
```

### Anthropic Usage

```python
usage = response.usage

# Anthropic separates input/output
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")

# Cache usage (if using prompt caching)
if hasattr(usage, 'cache_creation_input_tokens'):
    print(f"Cache creation: {usage.cache_creation_input_tokens}")
    print(f"Cache read: {usage.cache_read_input_tokens}")
```

### Usage Tracker Class

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class UsageTracker:
    """Track API usage across requests."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    cost_per_input_1k: float = 0.01
    cost_per_output_1k: float = 0.03
    
    def add(self, usage):
        """Add usage from a response."""
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.request_count += 1
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def total_cost(self) -> float:
        return (self.input_tokens / 1000 * self.cost_per_input_1k +
                self.output_tokens / 1000 * self.cost_per_output_1k)
    
    def report(self) -> Dict:
        return {
            "requests": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": f"${self.total_cost:.4f}"
        }

# Usage
tracker = UsageTracker()
tracker.add(response.usage)
print(tracker.report())
```

---

## Finish Reasons

Understand why generation stopped:

### OpenAI Finish Reasons

| Reason | Meaning | Action |
|--------|---------|--------|
| `stop` | Natural completion | Normal |
| `length` | Hit max_tokens | May be truncated |
| `content_filter` | Content blocked | Review prompt |
| `tool_calls` | Model wants to call tools | Process tool calls |

```python
# Chat Completions
finish_reason = response.choices[0].finish_reason

if finish_reason == "length":
    print("⚠️ Response was truncated. Consider increasing max_tokens.")
elif finish_reason == "content_filter":
    print("⚠️ Content was filtered. Review your prompt.")
elif finish_reason == "tool_calls":
    print("Model requested tool calls. Process them.")
```

### Anthropic Stop Reasons

| Reason | Meaning |
|--------|---------|
| `end_turn` | Natural completion |
| `max_tokens` | Hit limit |
| `stop_sequence` | Hit stop sequence |
| `tool_use` | Requested tool use |

```python
stop_reason = response.stop_reason

if stop_reason == "max_tokens":
    print("⚠️ Response may be incomplete")
```

---

## Quick Content Access

### Using `output_text` Helper

The Responses API provides a convenient helper:

```python
# Instead of navigating nested structure
text = response.output[0].content[0].text

# Use the helper
text = response.output_text  # Same result, cleaner
```

### Chat Completions Pattern

```python
# Standard pattern
content = response.choices[0].message.content
```

### Anthropic Pattern

```python
# Standard pattern
text = response.content[0].text
```

---

## Response Parser Class

Create a unified parser for multiple providers:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ParsedResponse:
    """Unified parsed response structure."""
    
    id: str
    model: str
    content: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    raw_response: Any
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def was_truncated(self) -> bool:
        return self.finish_reason in ("length", "max_tokens")

class ResponseParser:
    """Parse responses from multiple providers."""
    
    @staticmethod
    def parse_openai_responses(response) -> ParsedResponse:
        """Parse OpenAI Responses API response."""
        return ParsedResponse(
            id=response.id,
            model=response.model,
            content=response.output_text,
            finish_reason=getattr(response, 'finish_reason', 'stop'),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response
        )
    
    @staticmethod
    def parse_openai_chat(response) -> ParsedResponse:
        """Parse OpenAI Chat Completions response."""
        choice = response.choices[0]
        return ParsedResponse(
            id=response.id,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            raw_response=response
        )
    
    @staticmethod
    def parse_anthropic(response) -> ParsedResponse:
        """Parse Anthropic response."""
        content = ""
        if response.content:
            content = response.content[0].text
        
        return ParsedResponse(
            id=response.id,
            model=response.model,
            content=content,
            finish_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response
        )

# Usage
parser = ResponseParser()

# Parse OpenAI Responses API
openai_response = client.responses.create(...)
parsed = parser.parse_openai_responses(openai_response)
print(f"Content: {parsed.content}")
print(f"Tokens: {parsed.total_tokens}")
print(f"Truncated: {parsed.was_truncated}")
```

---

## Handling Multiple Choices

Some requests return multiple completions:

```python
# Request multiple choices
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Give me a creative name for a startup"}],
    n=3  # Request 3 different completions
)

# Process all choices
for i, choice in enumerate(response.choices):
    print(f"Option {i + 1}: {choice.message.content}")
    print(f"  Finish reason: {choice.finish_reason}")
```

### Selecting Best Choice

```python
def select_best_choice(response, criteria="first"):
    """Select from multiple choices based on criteria."""
    choices = response.choices
    
    if criteria == "first":
        return choices[0].message.content
    
    elif criteria == "longest":
        return max(choices, key=lambda c: len(c.message.content)).message.content
    
    elif criteria == "shortest":
        return min(choices, key=lambda c: len(c.message.content)).message.content
```

---

## Error Detection in Responses

Check for potential issues:

```python
def analyze_response(response) -> Dict[str, Any]:
    """Analyze response for potential issues."""
    issues = []
    
    # Check for truncation
    finish_reason = getattr(response.choices[0], 'finish_reason', 'stop')
    if finish_reason == "length":
        issues.append("Response was truncated due to token limit")
    
    # Check for content filter
    if finish_reason == "content_filter":
        issues.append("Content was filtered")
    
    # Check for empty response
    content = response.choices[0].message.content
    if not content or not content.strip():
        issues.append("Response was empty")
    
    # Check for refusal
    if content and any(phrase in content.lower() for phrase in 
                       ["i cannot", "i can't", "i'm not able"]):
        issues.append("Possible refusal detected")
    
    return {
        "content": content,
        "finish_reason": finish_reason,
        "issues": issues,
        "has_issues": len(issues) > 0
    }
```

---

## JSON Response Parsing

When expecting structured data:

```python
import json

def parse_json_response(response) -> Optional[Dict]:
    """Parse JSON from response content."""
    content = response.output_text
    
    # Try direct parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code block
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        json_str = content[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try extracting from generic code block
    if "```" in content:
        start = content.find("```") + 3
        # Skip language identifier if present
        newline = content.find("\n", start)
        start = newline + 1
        end = content.find("```", start)
        json_str = content[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None
```

---

## Hands-on Exercise

### Your Task

Build a `ResponseAnalyzer` that provides comprehensive response analysis.

### Requirements

1. Parse responses from OpenAI and Anthropic
2. Extract all metadata (ID, model, tokens, timing)
3. Detect potential issues (truncation, empty, refusal)
4. Calculate and track costs
5. Provide summary statistics

### Expected Result

```python
analyzer = ResponseAnalyzer()
result = analyzer.analyze(response)
print(result.summary())
# {
#     "id": "resp_abc123",
#     "model": "gpt-4.1",
#     "tokens": {"input": 50, "output": 150, "total": 200},
#     "cost": "$0.0055",
#     "issues": [],
#     "finish_reason": "stop"
# }
```

<details>
<summary>💡 Hints</summary>

- Use hasattr() to handle different response structures
- Create separate methods for each provider
- Store cost rates in a config dictionary
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class AnalysisResult:
    id: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    issues: List[str] = field(default_factory=list)
    provider: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> Dict[str, Any]:
        cost = self._calculate_cost()
        return {
            "id": self.id,
            "model": self.model,
            "provider": self.provider,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.input_tokens + self.output_tokens
            },
            "cost": f"${cost:.4f}",
            "finish_reason": self.finish_reason,
            "issues": self.issues,
            "has_issues": len(self.issues) > 0
        }
    
    def _calculate_cost(self) -> float:
        # Simplified cost calculation
        rates = {
            "gpt-4.1": (0.01, 0.03),
            "gpt-4o": (0.005, 0.015),
            "claude-sonnet-4": (0.003, 0.015),
        }
        
        # Find matching rate
        for model_prefix, (input_rate, output_rate) in rates.items():
            if model_prefix in self.model:
                return (self.input_tokens / 1000 * input_rate +
                        self.output_tokens / 1000 * output_rate)
        
        # Default rate
        return (self.input_tokens + self.output_tokens) / 1000 * 0.01

class ResponseAnalyzer:
    def analyze(self, response) -> AnalysisResult:
        # Detect provider and parse
        if hasattr(response, 'output_text'):
            return self._analyze_openai_responses(response)
        elif hasattr(response, 'choices'):
            return self._analyze_openai_chat(response)
        elif hasattr(response, 'stop_reason'):
            return self._analyze_anthropic(response)
        else:
            raise ValueError("Unknown response format")
    
    def _analyze_openai_responses(self, response) -> AnalysisResult:
        content = response.output_text
        issues = self._detect_issues(content, "stop")
        
        return AnalysisResult(
            id=response.id,
            model=response.model,
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason="stop",
            issues=issues,
            provider="openai"
        )
    
    def _analyze_openai_chat(self, response) -> AnalysisResult:
        choice = response.choices[0]
        content = choice.message.content or ""
        issues = self._detect_issues(content, choice.finish_reason)
        
        return AnalysisResult(
            id=response.id,
            model=response.model,
            content=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=choice.finish_reason,
            issues=issues,
            provider="openai"
        )
    
    def _analyze_anthropic(self, response) -> AnalysisResult:
        content = response.content[0].text if response.content else ""
        issues = self._detect_issues(content, response.stop_reason)
        
        return AnalysisResult(
            id=response.id,
            model=response.model,
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            issues=issues,
            provider="anthropic"
        )
    
    def _detect_issues(self, content: str, finish_reason: str) -> List[str]:
        issues = []
        
        if finish_reason in ("length", "max_tokens"):
            issues.append("Response truncated")
        if finish_reason == "content_filter":
            issues.append("Content filtered")
        if not content.strip():
            issues.append("Empty response")
        if any(phrase in content.lower() for phrase in ["i cannot", "i can't"]):
            issues.append("Possible refusal")
        
        return issues

# Test
analyzer = ResponseAnalyzer()
# result = analyzer.analyze(response)
# print(result.summary())
```

</details>

---

## Summary

✅ Responses contain rich metadata: ID, model, timestamps, usage  
✅ Track token usage for cost monitoring and optimization  
✅ Check finish reasons to detect truncation or filtering  
✅ Use `output_text` helper for quick content access  
✅ Create unified parsers for multi-provider support  
✅ Analyze responses for potential issues and edge cases

**Next:** [Extracting Generated Content](./06-extracting-content.md)

---

## Further Reading

- [OpenAI Response Object](https://platform.openai.com/docs/api-reference/responses/object) — Response structure
- [Usage Tracking](https://platform.openai.com/docs/guides/rate-limits) — Token counting
- [Anthropic Response](https://docs.anthropic.com/en/api/messages) — Anthropic response format

<!-- 
Sources Consulted:
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat
- Anthropic Messages: https://docs.anthropic.com/en/api/messages
-->
