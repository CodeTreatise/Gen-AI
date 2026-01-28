---
title: "Streaming vs Non-Streaming Trade-offs"
---

# Streaming vs Non-Streaming Trade-offs

## Introduction

Choosing between streaming and non-streaming modes involves trade-offs in user experience, code complexity, error handling, and use case fit. This lesson helps you make the right choice for your application.

### What We'll Cover

- Streaming advantages and challenges
- Non-streaming advantages and challenges
- Decision framework
- Hybrid approaches

---

## Streaming: Pros and Cons

### Advantages

```python
streaming_advantages = {
    "perceived_speed": "Users see content immediately",
    "engagement": "Users stay focused on progressive output",
    "early_abort": "Users can stop wrong answers early",
    "read_along": "Natural reading pace matches generation",
    "feedback": "System feels responsive and alive",
}

# Perceived speed improvement
# Total time: 5 seconds
# Non-streaming: 5 seconds of waiting
# Streaming: ~0.5 seconds to first token, then continuous flow
```

### Challenges

```python
streaming_challenges = {
    "complexity": "More code to handle chunks",
    "error_handling": "Partial responses on failure",
    "state_management": "Track accumulated content",
    "testing": "Harder to test streaming behavior",
    "debugging": "Async issues harder to trace",
}
```

### Code Complexity Comparison

```python
# Non-streaming: Simple
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
result = response.choices[0].message.content

# Streaming: More complex
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    stream=True
)

chunks = []
for chunk in response:
    if chunk.choices[0].delta.content:
        chunks.append(chunk.choices[0].delta.content)
        print(chunk.choices[0].delta.content, end="", flush=True)

result = "".join(chunks)
```

---

## Non-Streaming: Pros and Cons

### Advantages

```python
non_streaming_advantages = {
    "simplicity": "One request, one response",
    "error_handling": "All-or-nothing response",
    "testing": "Easy to mock and verify",
    "caching": "Response can be cached directly",
    "processing": "Full response for post-processing",
}
```

### Challenges

```python
non_streaming_challenges = {
    "perceived_slow": "Users wait for full generation",
    "anxiety": "Users wonder if it's working",
    "abandonment": "Users may leave during long waits",
    "no_preview": "Can't see partial results",
}
```

### When Non-Streaming is Better

```python
# 1. Background processing
def process_batch(items: list):
    """Process items without user waiting"""
    results = []
    for item in items:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Process: {item}"}],
            stream=False  # Simpler for batch
        )
        results.append(response.choices[0].message.content)
    return results

# 2. Post-processing required
def extract_and_validate(text: str):
    """Need full response before processing"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Extract JSON: {text}"}],
        response_format={"type": "json_object"},
        stream=False  # Need complete JSON
    )
    
    import json
    data = json.loads(response.choices[0].message.content)
    return validate_schema(data)

# 3. Caching responses
cache = {}

def cached_query(query: str):
    if query in cache:
        return cache[query]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
        stream=False  # Complete response for caching
    )
    
    result = response.choices[0].message.content
    cache[query] = result
    return result
```

---

## Decision Framework

### Use Streaming When

| Scenario | Reasoning |
|----------|-----------|
| **Interactive chat UI** | Users expect real-time feedback |
| **Long responses** | >2-3 seconds generation |
| **User-facing applications** | Perceived speed matters |
| **Conversational interfaces** | Natural feel |
| **Code generation UIs** | Watch code appear |

### Use Non-Streaming When

| Scenario | Reasoning |
|----------|-----------|
| **Background jobs** | No one is watching |
| **Batch processing** | Processing many items |
| **Post-processing needed** | Need complete response first |
| **Caching** | Store complete responses |
| **API endpoints** | Return complete response |
| **Short responses** | <1 second generation |

### Decision Tree

```
Is a user actively watching?
├── No → Non-streaming
│   └── Simpler code, easier caching
│
└── Yes → Is response time > 2 seconds?
    ├── No → Either works, prefer simpler
    │
    └── Yes → Streaming
        └── Better UX, worth complexity
```

---

## Hybrid Approaches

### Buffered Streaming

```python
def buffered_streaming(messages: list, buffer_threshold: int = 100):
    """
    Collect some content before showing.
    Balance between responsiveness and smoothness.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    buffer = ""
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            
            # Release when buffer is full or contains complete sentence
            if len(buffer) >= buffer_threshold or buffer.endswith(('.', '!', '?', '\n')):
                yield buffer
                buffer = ""
    
    # Don't forget remaining content
    if buffer:
        yield buffer
```

### Streaming with Fallback

```python
async def stream_with_fallback(messages: list, timeout: float = 30.0):
    """
    Try streaming, fall back to non-streaming on issues.
    """
    
    try:
        # Try streaming
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True,
            timeout=timeout
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as streaming_error:
        print(f"Streaming failed: {streaming_error}")
        
        # Fall back to non-streaming
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=False
        )
        yield response.choices[0].message.content
```

### Stream Then Process

```python
def stream_then_process(messages: list, processor):
    """
    Stream to user while collecting for processing.
    """
    
    full_content = ""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            print(content, end="", flush=True)  # Stream to user
    
    print()
    
    # Now process complete content
    processed = processor(full_content)
    
    return {
        "raw": full_content,
        "processed": processed
    }

# Usage
result = stream_then_process(
    messages,
    processor=lambda x: extract_code_blocks(x)
)
```

---

## Error Handling Differences

### Non-Streaming Error Handling

```python
def non_streaming_with_errors(messages: list):
    """Error handling is straightforward"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    
    except openai.RateLimitError:
        return "Rate limited. Please try again."
    
    except openai.APIError as e:
        return f"API error: {e}"
    
    except Exception as e:
        return f"Unexpected error: {e}"
```

### Streaming Error Handling

```python
def streaming_with_errors(messages: list):
    """Error handling with partial content recovery"""
    
    collected_content = ""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print()
        return {"success": True, "content": collected_content}
    
    except Exception as e:
        # We may have partial content!
        print(f"\n[Error after {len(collected_content)} chars: {e}]")
        
        return {
            "success": False,
            "partial_content": collected_content,
            "error": str(e)
        }

# Decision: What to do with partial content?
# - Show user what we got
# - Retry from checkpoint
# - Discard and show error
```

---

## Performance Comparison

### Timing Breakdown

```python
import time

def compare_timing(messages: list):
    """Compare streaming vs non-streaming timing"""
    
    # Non-streaming timing
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=False
    )
    non_stream_time = time.time() - start
    non_stream_content = response.choices[0].message.content
    
    # Streaming timing
    start = time.time()
    first_token_time = None
    stream_content = ""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - start
            stream_content += chunk.choices[0].delta.content
    
    stream_total_time = time.time() - start
    
    return {
        "non_streaming": {
            "total_time": non_stream_time,
            "time_to_content": non_stream_time,  # Same as total
        },
        "streaming": {
            "total_time": stream_total_time,
            "time_to_first_token": first_token_time,
            "perceived_improvement": f"{(1 - first_token_time/non_stream_time)*100:.1f}%"
        }
    }
```

---

## Hands-on Exercise

### Your Task

Build an adaptive streaming system:

```python
from openai import OpenAI
import time

client = OpenAI()

def adaptive_response(
    messages: list,
    expected_length: str = "unknown"  # "short", "medium", "long", "unknown"
) -> str:
    """
    Automatically choose streaming based on expected length.
    Short responses (<1s): Non-streaming
    Long responses (>2s): Streaming
    Unknown: Start non-streaming, switch if slow
    """
    
    if expected_length == "short":
        # Non-streaming for quick responses
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=50,
            stream=False
        )
        return response.choices[0].message.content
    
    elif expected_length == "long":
        # Streaming for long responses
        content = ""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return content
    
    else:
        # Your implementation: Adaptive approach
        # Hint: Try non-streaming with short timeout
        # If slow, switch to streaming
        pass

# Test with different types
test_cases = [
    {"messages": [{"role": "user", "content": "What is 2+2?"}], "expected": "short"},
    {"messages": [{"role": "user", "content": "Write a detailed essay about AI"}], "expected": "long"},
    {"messages": [{"role": "user", "content": "Explain recursion"}], "expected": "unknown"},
]

for test in test_cases:
    print(f"\n{'='*50}")
    print(f"Expected length: {test['expected']}")
    print('='*50)
    result = adaptive_response(test["messages"], test["expected"])
    print(f"\nResult length: {len(result)} chars")
```

---

## Summary

✅ **Streaming**: Better UX, more complex code

✅ **Non-streaming**: Simpler code, worse perceived speed

✅ **Use streaming** for interactive, user-facing applications

✅ **Use non-streaming** for background jobs and batch processing

✅ **Hybrid approaches** offer flexibility

✅ **Error handling differs**: streaming may have partial content

**Next:** [First Token Latency](./05-first-token-latency.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Transport Mechanisms](./03-transport-mechanisms.md) | [Streaming Response Modes](./00-streaming-response-modes.md) | [First Token Latency](./05-first-token-latency.md) |

