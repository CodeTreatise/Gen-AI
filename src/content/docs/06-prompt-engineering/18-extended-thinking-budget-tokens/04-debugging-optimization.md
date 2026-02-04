---
title: "Debugging and Optimization"
---

# Debugging and Optimization

## Introduction

Extended thinking adds new debugging capabilities and optimization considerations. You can inspect the model's reasoning to understand its decisions, but you also need to manage caching behavior, handle edge cases like redacted thinking, and optimize for cost and latency.

This lesson covers practical debugging techniques and optimization strategies for extended thinking.

### What We'll Cover

- Inspecting thinking output for debugging
- Understanding summarized vs. full thinking
- Handling redacted thinking blocks
- Caching behavior with thinking
- Batch processing for large budgets
- Token counting and cost optimization

### Prerequisites

- [Extended Thinking Overview](./00-extended-thinking-overview.md)
- [Thinking Budget Configuration](./01-thinking-budget-configuration.md)
- [Thinking with Tools and Streaming](./03-thinking-with-tools-streaming.md)

---

## Inspecting Thinking Output

### Why Inspect Thinking?

The thinking block reveals:
- How the model interpreted your prompt
- What approaches it considered
- Where it may have gone wrong
- Why it chose a particular answer

### Basic Inspection Pattern

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "What's 23 × 47?"}]
)

# Separate thinking from response
thinking_blocks = []
text_blocks = []

for block in response.content:
    if block.type == "thinking":
        thinking_blocks.append(block.thinking)
    elif block.type == "text":
        text_blocks.append(block.text)

print("=== THINKING ===")
for thinking in thinking_blocks:
    print(thinking)
    
print("\n=== RESPONSE ===")
for text in text_blocks:
    print(text)
```

### Debugging Workflow

```
1. Run with extended thinking
       ↓
2. Get unexpected result?
       ↓
3. Inspect thinking block
       ↓
4. Identify where reasoning went wrong
       ↓
5. Adjust prompt or budget
       ↓
6. Retry
```

### Common Issues Revealed by Thinking

| Thinking Pattern | Issue | Fix |
|------------------|-------|-----|
| Repeating same point | Budget too low | Increase `budget_tokens` |
| Missing key constraint | Unclear prompt | Add explicit requirements |
| Wrong interpretation | Ambiguous wording | Rephrase more clearly |
| Stopped mid-reasoning | Hit token limit | Increase `max_tokens` |
| Considered right answer, chose wrong | Confidence issue | Add verification step |

---

## Summarized vs. Full Thinking

### Claude 4 Models (Summarized)

Claude 4 models (Opus 4.5, Sonnet 4, etc.) provide **summarized thinking**:

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",  # Claude 4 model
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)

# Thinking is summarized for external consumption
for block in response.content:
    if block.type == "thinking":
        # This is a summary, not raw internal reasoning
        print(block.thinking)
```

### Claude Sonnet 3.7 (Full Thinking)

Claude Sonnet 3.7 provides **full, unfiltered thinking**:

```python
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",  # Claude 3.7
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)

# Thinking is the complete internal reasoning
for block in response.content:
    if block.type == "thinking":
        # This is raw, unfiltered thinking
        print(block.thinking)
```

### Implications

| Aspect | Summarized (Claude 4) | Full (Claude 3.7) |
|--------|----------------------|-------------------|
| Debugging depth | Less granular | Very detailed |
| Token output | Shorter | Longer |
| Privacy | More protected | Raw thoughts visible |
| Debug use case | High-level issues | Detailed trace |

---

## Redacted Thinking

### What Is Redacted Thinking?

Sometimes the model's thinking triggers safety systems. When this happens, you receive a `redacted_thinking` block instead of the thinking content:

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": prompt}]
)

for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
    elif block.type == "redacted_thinking":
        # Thinking was flagged by safety systems
        print("Thinking was redacted")
        print(f"Data: {block.data}")  # Encrypted content
```

### Handling Redacted Thinking

```python
def process_response(response):
    """Handle response with potential redacted thinking."""
    
    has_redaction = False
    thinking_content = []
    text_content = []
    
    for block in response.content:
        if block.type == "thinking":
            thinking_content.append(block.thinking)
        elif block.type == "redacted_thinking":
            has_redaction = True
            # Log for investigation (don't expose to users)
            print(f"[DEBUG] Redacted thinking block encountered")
        elif block.type == "text":
            text_content.append(block.text)
    
    if has_redaction:
        # The response may still be valid
        # but debugging is limited
        print("[WARNING] Some thinking was redacted")
    
    return {
        "thinking": thinking_content,
        "text": text_content,
        "has_redaction": has_redaction
    }
```

### Why Redaction Happens

Redaction occurs when thinking content triggers Anthropic's safety systems. This doesn't necessarily mean the response is unsafe—just that the internal reasoning touched on sensitive topics.

> **Note:** Redacted blocks still preserve conversation continuity. You can pass them back in multi-turn conversations.

---

## Caching Behavior

### How Caching Works with Thinking

Anthropic caches prompt prefixes for efficiency. With extended thinking, caching has specific behaviors:

| Scenario | Caching Behavior |
|----------|-----------------|
| Previous turns' thinking | Stripped from cache |
| Current turn's thinking | Counts toward `max_tokens` |
| Claude Opus 4.5 | Previous thinking NOT stripped |

### Cache Invalidation

Changing thinking parameters invalidates the cache:

```python
# Request 1
response1 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)  # Creates cache

# Request 2 - Different budget invalidates cache!
response2 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 15000},  # Different
    ...
)  # Cache miss

# Request 3 - Same parameters, cache hit
response3 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 10000},  # Same as request 1
    ...
)  # Cache hit (assuming same prompt prefix)
```

### Optimizing Cache Usage

```python
# Keep thinking parameters consistent across requests
THINKING_CONFIG = {"type": "enabled", "budget_tokens": 10000}

def make_request(messages):
    return client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking=THINKING_CONFIG,  # Consistent
        messages=messages
    )
```

### Context Window Considerations

```
Without Extended Thinking:
[System] + [Previous Messages] + [New Response] ≤ Context Window

With Extended Thinking:
[System] + [Previous Messages (thinking stripped)] + [Current Thinking] + [Response] ≤ Context Window
```

The current turn's thinking counts toward the context window, but previous turns' thinking is stripped (except for Opus 4.5).

---

## Batch Processing

### When to Use Batch API

For large thinking budgets (>32,000 tokens), use the Batch API to avoid timeouts:

```python
# Individual request with large budget may timeout
response = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 50000},  # Large budget
    ...
)  # Risk of timeout

# Use Batch API instead
batch_request = client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 16000,
                "thinking": {"type": "enabled", "budget_tokens": 50000},
                "messages": [{"role": "user", "content": prompt}]
            }
        }
    ]
)
```

### Batch Processing Pattern

```python
import time

def process_with_large_budget(prompts: list[str], budget: int = 50000):
    """Process multiple prompts with large thinking budgets via batch API."""
    
    # Create batch
    requests = [
        {
            "custom_id": f"request-{i}",
            "params": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 16000,
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        for i, prompt in enumerate(prompts)
    ]
    
    batch = client.messages.batches.create(requests=requests)
    
    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        time.sleep(30)  # Check every 30 seconds
    
    # Retrieve results
    results = list(client.messages.batches.results(batch.id))
    return results
```

---

## Token Counting and Cost Optimization

### Token Usage Metrics

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": prompt}]
)

# Access usage metrics
usage = response.usage
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")

# For Claude 4 models, thinking tokens are included in output
# For models with separate counting:
if hasattr(usage, "thinking_tokens"):
    print(f"Thinking tokens: {usage.thinking_tokens}")
```

### Gemini Token Counting

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=GenerateContentConfig(
        thinking_config={"thinking_budget": 8000}
    )
)

# Access usage metadata
usage = response.usage_metadata
print(f"Prompt tokens: {usage.prompt_token_count}")
print(f"Response tokens: {usage.candidates_token_count}")
print(f"Thinking tokens: {usage.thoughts_token_count}")  # Separate count
print(f"Total: {usage.total_token_count}")
```

### Cost Optimization Strategies

| Strategy | Implementation | Savings |
|----------|----------------|---------|
| Right-size budget | Match budget to task complexity | 10-50% |
| Use cheaper models | Sonnet vs Opus for simple tasks | 30-60% |
| Disable for simple tasks | No thinking for basic queries | 50-80% |
| Batch process | Use Batch API for 50% discount | 50% |
| Cache effectively | Keep params consistent | Variable |

### Budget Right-Sizing Pattern

```python
def get_thinking_config(task_complexity: str) -> dict:
    """Return appropriate thinking config based on task complexity."""
    
    configs = {
        "simple": None,  # No extended thinking
        "moderate": {"type": "enabled", "budget_tokens": 5000},
        "complex": {"type": "enabled", "budget_tokens": 15000},
        "very_complex": {"type": "enabled", "budget_tokens": 30000}
    }
    
    return configs.get(task_complexity, configs["moderate"])

def make_optimized_request(prompt: str, complexity: str):
    thinking = get_thinking_config(complexity)
    
    kwargs = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if thinking:
        kwargs["thinking"] = thinking
    
    return client.messages.create(**kwargs)
```

---

## Debugging Patterns

### Print Thinking for Development

```python
import os

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def debug_request(prompt: str):
    """Make request and optionally print thinking for debugging."""
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": prompt}]
    )
    
    if DEBUG:
        for block in response.content:
            if block.type == "thinking":
                print(f"\n{'='*60}")
                print("THINKING:")
                print(f"{'='*60}")
                print(block.thinking)
                print(f"{'='*60}\n")
    
    return response
```

### Logging Thinking to File

```python
import json
from datetime import datetime

def log_response(prompt: str, response, log_file: str = "thinking_log.jsonl"):
    """Log thinking to file for later analysis."""
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:500],  # Truncate for storage
        "thinking": [],
        "response": [],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }
    
    for block in response.content:
        if block.type == "thinking":
            entry["thinking"].append(block.thinking)
        elif block.type == "text":
            entry["response"].append(block.text)
        elif block.type == "redacted_thinking":
            entry["thinking"].append("[REDACTED]")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### A/B Testing Thinking Budgets

```python
import random

def ab_test_budget(prompt: str, budgets: list[int], trials: int = 5):
    """Compare response quality across different budgets."""
    
    results = {budget: [] for budget in budgets}
    
    for _ in range(trials):
        for budget in budgets:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": budget},
                messages=[{"role": "user", "content": prompt}]
            )
            
            results[budget].append({
                "thinking_length": sum(
                    len(b.thinking) for b in response.content if b.type == "thinking"
                ),
                "response": next(
                    (b.text for b in response.content if b.type == "text"), ""
                ),
                "output_tokens": response.usage.output_tokens
            })
    
    # Analyze results
    for budget, data in results.items():
        avg_thinking = sum(d["thinking_length"] for d in data) / len(data)
        avg_tokens = sum(d["output_tokens"] for d in data) / len(data)
        print(f"Budget {budget}: avg thinking chars = {avg_thinking:.0f}, avg tokens = {avg_tokens:.0f}")
    
    return results
```

---

## Performance Monitoring

### Timing Extended Thinking Requests

```python
import time

def timed_request(prompt: str, budget: int):
    """Make request and measure timing."""
    
    start = time.time()
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": prompt}]
    )
    
    elapsed = time.time() - start
    
    return {
        "response": response,
        "elapsed_seconds": elapsed,
        "tokens_per_second": response.usage.output_tokens / elapsed,
        "budget_used_estimate": sum(
            len(b.thinking) / 4  # Rough token estimate
            for b in response.content if b.type == "thinking"
        )
    }
```

### Monitoring Dashboard Metrics

Track these metrics for production monitoring:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Thinking budget utilization | % of budget actually used | < 20% (wasteful) |
| Redaction rate | % of responses with redacted thinking | > 5% |
| Timeout rate | % of requests timing out | > 1% |
| Cache hit rate | % of requests hitting cache | < 80% |
| Average thinking time | Time spent on thinking | > 30s |

---

## Hands-on Exercise

### Your Task

Create a debugging utility that:
1. Makes a request with extended thinking
2. Logs the thinking to a file
3. Detects potential issues (low utilization, redaction)
4. Provides recommendations

<details>
<summary>💡 Hints (click to expand)</summary>

- Check if thinking length is much shorter than budget
- Look for `redacted_thinking` blocks
- Track token usage vs. budget

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from datetime import datetime
from dataclasses import dataclass
from anthropic import Anthropic

@dataclass
class ThinkingAnalysis:
    thinking_chars: int
    budget_tokens: int
    utilization_estimate: float
    has_redaction: bool
    recommendations: list[str]

def analyze_thinking_response(response, budget_tokens: int) -> ThinkingAnalysis:
    """Analyze a thinking response for potential issues."""
    
    thinking_chars = 0
    has_redaction = False
    recommendations = []
    
    for block in response.content:
        if block.type == "thinking":
            thinking_chars += len(block.thinking)
        elif block.type == "redacted_thinking":
            has_redaction = True
    
    # Estimate utilization (rough: 4 chars per token)
    estimated_tokens = thinking_chars / 4
    utilization = estimated_tokens / budget_tokens if budget_tokens > 0 else 0
    
    # Generate recommendations
    if utilization < 0.2:
        recommendations.append(
            f"Low budget utilization ({utilization:.0%}). "
            f"Consider reducing budget_tokens from {budget_tokens} to {int(budget_tokens * 0.5)}"
        )
    
    if utilization > 0.95:
        recommendations.append(
            f"High budget utilization ({utilization:.0%}). "
            f"Model may have been constrained. Consider increasing budget."
        )
    
    if has_redaction:
        recommendations.append(
            "Thinking was redacted by safety systems. "
            "Review prompt for potentially sensitive content."
        )
    
    if not recommendations:
        recommendations.append("No issues detected. Configuration looks optimal.")
    
    return ThinkingAnalysis(
        thinking_chars=thinking_chars,
        budget_tokens=budget_tokens,
        utilization_estimate=utilization,
        has_redaction=has_redaction,
        recommendations=recommendations
    )

def debug_thinking_request(
    client: Anthropic,
    prompt: str,
    budget_tokens: int = 10000,
    log_file: str = "thinking_debug.jsonl"
):
    """Make request, analyze, and log with recommendations."""
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": budget_tokens},
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Analyze
    analysis = analyze_thinking_response(response, budget_tokens)
    
    # Log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:200],
        "budget_tokens": budget_tokens,
        "thinking_chars": analysis.thinking_chars,
        "utilization_estimate": analysis.utilization_estimate,
        "has_redaction": analysis.has_redaction,
        "recommendations": analysis.recommendations,
        "usage": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Print analysis
    print(f"\n{'='*60}")
    print("THINKING ANALYSIS")
    print(f"{'='*60}")
    print(f"Budget: {budget_tokens} tokens")
    print(f"Thinking output: ~{analysis.thinking_chars} characters")
    print(f"Estimated utilization: {analysis.utilization_estimate:.0%}")
    print(f"Redacted: {'Yes ⚠️' if analysis.has_redaction else 'No ✅'}")
    print(f"\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"  • {rec}")
    print(f"{'='*60}\n")
    
    return response, analysis

# Usage
if __name__ == "__main__":
    client = Anthropic()
    
    response, analysis = debug_thinking_request(
        client,
        "What is 2 + 2?",
        budget_tokens=10000
    )
    
    # This simple prompt will show low utilization warning
```

</details>

---

## Summary

✅ **Inspect thinking** to debug model reasoning
✅ **Handle redacted thinking** gracefully—it doesn't break continuity
✅ **Keep parameters consistent** for cache optimization
✅ **Use Batch API** for budgets > 32,000 tokens
✅ **Right-size budgets** based on task complexity
✅ **Log and monitor** thinking metrics in production

---

## Lesson 18 Complete ✅

You've completed the Extended Thinking & Budget Tokens lesson. You now understand:

1. **What extended thinking is** and when to use it
2. **How to configure budgets** across different providers
3. **Prompting techniques** optimized for thinking models
4. **Tool use and streaming** with extended thinking
5. **Debugging and optimization** strategies

**Next Lesson:** [Grounding and Attribution](../19-grounding-attribution/00-grounding-attribution-overview.md)

---

## Further Reading

- [Anthropic Token Counting API](https://docs.anthropic.com/en/api/counting-tokens)
- [Anthropic Batch API](https://docs.anthropic.com/en/api/creating-message-batches)
- [Anthropic Extended Thinking Guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)

---

<!-- 
Sources Consulted:
- Anthropic Extended Thinking: Summarized thinking, redacted thinking, caching behavior
- Anthropic Batch API: Large budget processing
- Gemini Thinking: thoughtsTokenCount for usage
- Anthropic Extended Thinking Tips: Batch processing recommendations
-->
