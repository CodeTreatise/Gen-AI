---
title: "Reasoning Configuration: Effort, Budgets, and Context Management"
---

# Reasoning Configuration: Effort, Budgets, and Context Management

## Introduction

Reasoning models offer fine-grained control over how much "thinking" they perform. This lesson covers the configuration options across providers—from effort levels and token budgets to reasoning summaries and context management strategies.

> **🔑 Key Insight:** More reasoning isn't always better. The right configuration balances quality, latency, and cost for your specific use case. A simple classification might need `low` effort, while a complex planning task needs `high`.

### What We'll Cover

- Effort and budget configuration across providers
- Reasoning summaries and transparency
- Context window management strategies
- Handling reasoning tokens in responses
- Tool use with reasoning models
- Production optimization patterns

### Prerequisites

- [Prompting Best Practices](./03-prompting-best-practices.md)
- Familiarity with token counting

---

## Effort and Budget Configuration

### OpenAI: Reasoning Effort

OpenAI uses a simple `effort` parameter with three levels:

```python
from openai import OpenAI
client = OpenAI()

# Low effort: Quick, simpler reasoning
response_low = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    input=[{"role": "user", "content": "Summarize this article."}]
)

# Medium effort: Balanced (default)
response_medium = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": "Analyze this code for issues."}]
)

# High effort: Maximum reasoning
response_high = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    input=[{"role": "user", "content": "Create a comprehensive migration strategy."}]
)
```

### Effort Level Guidelines

| Effort | Typical Reasoning Tokens | Use Cases |
|--------|-------------------------|-----------|
| `low` | 1,000 - 5,000 | Simpler analysis, classification, summarization |
| `medium` | 5,000 - 15,000 | Standard analysis, code review, planning |
| `high` | 15,000 - 50,000+ | Complex reasoning, research, multi-step planning |

### Anthropic: Budget Tokens

Anthropic gives explicit control via token budget:

```python
import anthropic
client = anthropic.Anthropic()

# Minimum budget (1,024 tokens)
response_min = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 1024  # Minimum allowed
    },
    messages=[{"role": "user", "content": "Quick classification task."}]
)

# Moderate budget
response_mid = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000
    },
    messages=[{"role": "user", "content": "Analyze this system design."}]
)

# High budget
response_high = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=32000,
    thinking={
        "type": "enabled",
        "budget_tokens": 20000
    },
    messages=[{"role": "user", "content": "Create a comprehensive architecture proposal."}]
)
```

### Budget Token Rules

```python
# Important constraints for Anthropic
max_tokens = 16000
budget_tokens = 10000

# Rule 1: budget_tokens must be >= 1,024
assert budget_tokens >= 1024, "Minimum budget is 1,024"

# Rule 2: budget_tokens must be < max_tokens (standard mode)
assert budget_tokens < max_tokens, "Budget must be less than max_tokens"

# Rule 3: With interleaved thinking, budget can equal max_tokens
# (beta feature: interleaved-thinking-2025-05-14)
```

### Gemini: Thinking Levels and Budgets

Gemini offers two configuration styles depending on model version:

```python
from google import genai
client = genai.Client()

# Gemini 3 series: Use thinking_level
response = client.models.generate_content(
    model="gemini-3-flash",
    contents="Analyze this problem.",
    config={
        "thinking_config": {
            "thinking_level": "medium"  # minimal, low, medium, high
        }
    }
)

# Gemini 2.5 series: Use thinking_budget
response = client.models.generate_content(
    model="gemini-2.5-flash-preview",
    contents="Analyze this problem.",
    config={
        "thinking_config": {
            "thinking_budget": 8000  # 0-32768, or -1 for dynamic
        }
    }
)
```

### Gemini Thinking Level Details

| Level | Behavior | Approximate Tokens |
|-------|----------|-------------------|
| `minimal` | Least thinking, fastest | ~500-1,000 |
| `low` | Light reasoning | ~1,000-3,000 |
| `medium` | Balanced | ~3,000-8,000 |
| `high` | Maximum reasoning (default) | Dynamic, potentially high |

```python
# Dynamic thinking budget (Gemini 2.5)
response = client.models.generate_content(
    model="gemini-2.5-flash-preview",
    contents="Complex planning task.",
    config={
        "thinking_config": {
            "thinking_budget": -1  # Model decides
        }
    }
)
```

---

## Reasoning Summaries

### OpenAI: Reasoning Summary Configuration

```python
# Get reasoning summaries
response = client.responses.create(
    model="gpt-5",
    reasoning={
        "effort": "high",
        "summary": "auto"  # "auto" | "none" | "detailed"
    },
    input=[{"role": "user", "content": "Analyze this business strategy."}]
)

# Access summaries in output
for item in response.output:
    if item.type == "reasoning":
        # Summary is an array of strings
        for summary_part in item.summary:
            print(f"Reasoning: {summary_part}")
    elif item.type == "message":
        print(f"Response: {item.content[0].text}")
```

### Summary Options

| Option | Behavior |
|--------|----------|
| `"auto"` | Include summary when beneficial |
| `"none"` | No summary (just final output) |
| `"detailed"` | More comprehensive summary |

### Anthropic: Direct Thinking Access

Anthropic provides the actual thinking content (summarized for Claude 4 models):

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[{"role": "user", "content": "Solve this problem."}]
)

# Access thinking blocks
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
        # Note: For Claude 4, this is summarized
        # For Claude 4.5 Opus, raw thinking tokens are preserved
    elif block.type == "text":
        print(f"Response: {block.text}")
```

### Claude 4 vs Claude 4.5 Opus Thinking

| Model | Thinking Content |
|-------|-----------------|
| Claude 4, Sonnet 4, Haiku 4.5 | Summarized (not raw tokens) |
| Claude 4.5 Opus | Raw thinking tokens preserved |

### Gemini: Thought Summaries

```python
# Request thought summaries
response = client.models.generate_content(
    model="gemini-3-flash",
    contents="Analyze this dataset.",
    config={
        "thinking_config": {
            "thinking_level": "high",
            "include_thoughts": True
        }
    }
)

# Access thoughts
for part in response.candidates[0].content.parts:
    if hasattr(part, 'thought') and part.thought:
        print(f"Thought: {part.text}")
    else:
        print(f"Response: {part.text}")
```

---

## Context Window Management

### Token Reservation Strategy

Reasoning tokens come from the same context window as output. Reserve space accordingly:

```python
def calculate_token_allocation(
    context_window: int,
    input_tokens: int,
    expected_output: int,
    reasoning_effort: str
) -> dict:
    """
    Calculate token allocation for reasoning models.
    """
    
    # Estimate reasoning tokens by effort
    reasoning_estimates = {
        "low": 5000,
        "medium": 15000,
        "high": 25000
    }
    
    expected_reasoning = reasoning_estimates[reasoning_effort]
    
    # Total needed
    total_needed = input_tokens + expected_reasoning + expected_output
    
    # Check fit
    fits = total_needed <= context_window
    
    # Calculate max_output_tokens setting
    # Must include both reasoning and actual output
    recommended_max_output = expected_reasoning + expected_output + 2000  # Buffer
    
    return {
        "fits": fits,
        "input_tokens": input_tokens,
        "estimated_reasoning": expected_reasoning,
        "estimated_output": expected_output,
        "total_estimate": total_needed,
        "context_window": context_window,
        "headroom": context_window - total_needed,
        "recommended_max_output_tokens": recommended_max_output
    }

# Example: Complex analysis task
allocation = calculate_token_allocation(
    context_window=128000,  # GPT-5 context
    input_tokens=10000,     # Large document
    expected_output=2000,   # Detailed response
    reasoning_effort="high"
)

print(f"Fits: {allocation['fits']}")
print(f"Headroom: {allocation['headroom']} tokens")
print(f"Set max_output_tokens to: {allocation['recommended_max_output_tokens']}")
```

### OpenAI Token Reservation

```python
# Reserve tokens for reasoning
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    max_output_tokens=32000,  # Includes reasoning + output
    input=[{"role": "user", "content": "[large input]"}]
)

# Check actual usage
print(f"Reasoning tokens: {response.usage.reasoning_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total output: {response.usage.reasoning_tokens + response.usage.completion_tokens}")
```

### Handling Long Reasoning Completion

Sometimes reasoning exhausts the available tokens. Handle gracefully:

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    max_output_tokens=16000,
    input=[{"role": "user", "content": "Very complex task..."}]
)

# Check for incomplete response
if response.status == "incomplete":
    if response.incomplete_details.reason == "max_output_tokens":
        print("Response truncated - consider increasing max_output_tokens")
        
        # Retry with more tokens
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "high"},
            max_output_tokens=32000,  # Increased
            input=[{"role": "user", "content": "Very complex task..."}]
        )
```

### Context Window by Model

| Provider | Model | Context Window |
|----------|-------|----------------|
| OpenAI | GPT-5 | 128,000 |
| OpenAI | GPT-5-mini | 128,000 |
| OpenAI | o3 | 200,000 |
| Anthropic | Claude Sonnet 4 | 200,000 |
| Anthropic | Claude Opus 4 | 200,000 |
| Gemini | Gemini 3 Flash | 1,000,000 |
| Gemini | Gemini 3 Pro | 1,000,000 |

---

## Tool Use with Reasoning Models

### OpenAI: Preserving Reasoning Items

When using function calling, reasoning items help the model make consistent decisions across tool calls:

```python
# Define tools
tools = [{
    "type": "function",
    "name": "get_stock_price",
    "description": "Get current stock price",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"}
        },
        "required": ["symbol"]
    }
}]

# Initial request
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    tools=tools,
    input=[{
        "role": "user",
        "content": "Compare the valuations of AAPL and MSFT"
    }]
)

# If tool use requested, execute and continue
if response.output[0].type == "function_call":
    tool_call = response.output[0]
    
    # Execute tool
    result = get_stock_price(tool_call.arguments["symbol"])
    
    # Continue with reasoning preserved
    # The model's reasoning about the comparison strategy persists
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "medium"},
        tools=tools,
        input=[
            {"role": "user", "content": "Compare the valuations of AAPL and MSFT"},
            tool_call,  # Original tool call
            {"role": "tool", "content": result, "tool_call_id": tool_call.id}
        ]
    )
```

### Anthropic: Preserving Thinking Blocks in Tool Use

> **Critical:** You must include thinking blocks when continuing a conversation with tool results.

```python
tools = [{
    "name": "search_database",
    "description": "Search the customer database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}]

# First request
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    tools=tools,
    messages=[{
        "role": "user",
        "content": "Find customers who might churn this month"
    }]
)

if response1.stop_reason == "tool_use":
    # Find tool use block
    tool_block = next(b for b in response1.content if b.type == "tool_use")
    
    # Execute tool
    search_result = search_database(tool_block.input["query"])
    
    # Continue - MUST include all content including thinking blocks
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        tools=tools,
        messages=[
            {"role": "user", "content": "Find customers who might churn this month"},
            {"role": "assistant", "content": response1.content},  # Includes thinking!
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": search_result
            }]}
        ]
    )
```

### Gemini: Thought Signatures for Function Calling

```python
# First turn with tool
response1 = client.models.generate_content(
    model="gemini-3-flash",
    contents="Get the weather in Tokyo and recommend what to wear",
    config={
        "thinking_config": {"thinking_level": "medium"},
        "tools": [weather_tool]
    }
)

# Extract function call and thought signature
function_call = response1.candidates[0].content.parts[0].function_call

# Execute function
weather_data = get_weather(function_call.args["location"])

# Continue with signature preserved
# The model uses the signature to maintain reasoning context
history = [
    {"role": "user", "parts": [{"text": "Get weather in Tokyo and recommend..."}]},
    {"role": "model", "parts": response1.candidates[0].content.parts},  # Includes signature
    {"role": "user", "parts": [{
        "function_response": {
            "name": "get_weather",
            "response": weather_data
        }
    }]}
]

response2 = client.models.generate_content(
    model="gemini-3-flash",
    contents=history,
    config={
        "thinking_config": {"thinking_level": "medium"}
    }
)
```

---

## Production Optimization Patterns

### Adaptive Effort Selection

```python
from typing import Literal
import tiktoken

def select_reasoning_effort(
    prompt: str,
    task_type: str,
    latency_budget_ms: int
) -> Literal["low", "medium", "high"]:
    """
    Dynamically select reasoning effort based on task characteristics.
    """
    
    # Estimate prompt complexity
    enc = tiktoken.encoding_for_model("gpt-4")
    token_count = len(enc.encode(prompt))
    
    # Complexity indicators
    complex_keywords = [
        "analyze", "compare", "evaluate", "synthesize", 
        "strategy", "architecture", "optimize", "debug"
    ]
    complexity_score = sum(1 for kw in complex_keywords if kw in prompt.lower())
    
    # Task type weights
    task_weights = {
        "classification": 0,
        "summarization": 0,
        "analysis": 1,
        "planning": 2,
        "debugging": 2,
        "research": 2
    }
    task_weight = task_weights.get(task_type, 1)
    
    # Calculate total score
    total_score = complexity_score + task_weight + (token_count > 5000)
    
    # Latency constraint
    if latency_budget_ms < 5000:
        return "low"
    elif latency_budget_ms < 15000:
        return "medium" if total_score < 3 else "low"
    else:
        if total_score >= 4:
            return "high"
        elif total_score >= 2:
            return "medium"
        else:
            return "low"

# Usage
effort = select_reasoning_effort(
    prompt="Analyze this codebase and create a refactoring strategy",
    task_type="planning",
    latency_budget_ms=30000
)
# Returns: "high"
```

### Caching with Reasoning Models

OpenAI offers extended cache retention for reasoning models:

```python
# Enable 24-hour cache retention
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    prompt_cache_retention="24h",  # Extended caching
    input=[
        {
            "role": "developer",
            "content": "[Long system prompt or context that repeats across calls]"
        },
        {
            "role": "user",
            "content": "Analyze this specific scenario..."
        }
    ]
)

# Cached prompts reduce costs significantly
# The developer message context is cached across calls
```

### Batching for Cost Efficiency

```python
from openai import OpenAI
import json

client = OpenAI()

def create_batch_reasoning_request(tasks: list[dict]) -> str:
    """
    Create a batch file for reasoning tasks.
    """
    
    requests = []
    for i, task in enumerate(tasks):
        requests.append({
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-5",
                "reasoning": {"effort": task.get("effort", "medium")},
                "input": [{"role": "user", "content": task["prompt"]}]
            }
        })
    
    # Write to JSONL file
    with open("batch_input.jsonl", "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    
    return "batch_input.jsonl"

# Submit batch
batch_file = create_batch_reasoning_request([
    {"prompt": "Analyze document 1", "effort": "high"},
    {"prompt": "Analyze document 2", "effort": "high"},
    {"prompt": "Analyze document 3", "effort": "high"}
])

# Upload and create batch
file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
batch = client.batches.create(input_file_id=file.id, endpoint="/v1/responses")

# 50% discount on batch processing
print(f"Batch ID: {batch.id}")
```

### Streaming for Better UX

```python
async def stream_reasoning_response(prompt: str):
    """
    Stream response with progress indication.
    """
    
    thinking_started = False
    response_started = False
    
    stream = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "high", "summary": "auto"},
        input=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for event in stream:
        if event.type == "response.reasoning_summary.delta":
            if not thinking_started:
                print("\n🤔 Thinking...\n")
                thinking_started = True
            # Optionally show reasoning summary
            
        elif event.type == "response.output_text.delta":
            if not response_started:
                print("\n📝 Response:\n")
                response_started = True
            print(event.delta, end="", flush=True)
    
    print()  # Final newline
```

---

## Common Configuration Mistakes

### ❌ Insufficient max_output_tokens

```python
# Bad: max_output_tokens doesn't account for reasoning
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    max_output_tokens=1000,  # ❌ Way too low for high effort
    input=[...]
)

# Good: Reserve space for reasoning
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "high"},
    max_output_tokens=32000,  # ✅ Includes reasoning headroom
    input=[...]
)
```

### ❌ Mismatched Anthropic Budget

```python
# Bad: budget_tokens >= max_tokens
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # ❌ Exceeds max_tokens
    },
    messages=[...]
)

# Good: budget_tokens < max_tokens
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # ✅ Less than max_tokens
    },
    messages=[...]
)
```

### ❌ Inconsistent Thinking Settings

```python
# Bad: Toggling thinking mid-conversation (Anthropic)
response1 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[...]
)

response2 = client.messages.create(
    thinking={"type": "disabled"},  # ❌ Can't toggle
    messages=[...]
)

# Good: Keep thinking consistent
response2 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 5000},  # ✅ Same setting
    messages=[...]
)
```

---

## Hands-on Exercise

### Your Task

Create a configuration manager that dynamically selects reasoning settings based on task requirements and budget constraints.

**Requirements:**
1. Accept task description, budget (low/medium/high), and latency requirements
2. Return provider-specific configuration
3. Handle constraints (minimum budgets, token limits)
4. Include cost estimation

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ReasoningConfiguration:
    provider: str
    model: str
    config: dict
    estimated_cost_per_1k_input: float
    estimated_latency_seconds: tuple[int, int]  # min, max
    notes: list[str]

class ReasoningConfigManager:
    """
    Configure reasoning models based on task requirements.
    """
    
    # Pricing per 1M tokens (input + output estimated)
    PRICING = {
        "openai": {"gpt-5": 40, "gpt-5-mini": 20},
        "anthropic": {"claude-sonnet-4": 18, "claude-opus-4": 90},
        "gemini": {"gemini-3-flash": 0.75, "gemini-3-pro": 11}
    }
    
    def configure(
        self,
        task_complexity: Literal["simple", "moderate", "complex"],
        cost_budget: Literal["low", "medium", "high"],
        latency_tolerance: Literal["strict", "moderate", "relaxed"],
        provider_preference: Optional[str] = None
    ) -> ReasoningConfiguration:
        """
        Generate optimal configuration based on requirements.
        """
        
        notes = []
        
        # Select provider based on cost and preference
        if provider_preference:
            provider = provider_preference
        elif cost_budget == "low":
            provider = "gemini"
            notes.append("Selected Gemini for lowest cost")
        elif latency_tolerance == "strict":
            provider = "gemini"
            notes.append("Selected Gemini for fastest response")
        else:
            provider = "openai"
            notes.append("Selected OpenAI for balanced performance")
        
        # Select model within provider
        if provider == "openai":
            model = "gpt-5-mini" if cost_budget == "low" else "gpt-5"
            config = self._configure_openai(task_complexity, latency_tolerance)
        elif provider == "anthropic":
            model = "claude-sonnet-4" if cost_budget != "high" else "claude-opus-4"
            config = self._configure_anthropic(task_complexity, latency_tolerance)
        else:
            model = "gemini-3-flash" if cost_budget != "high" else "gemini-3-pro"
            config = self._configure_gemini(task_complexity, latency_tolerance)
        
        # Estimate latency
        latency = self._estimate_latency(task_complexity, provider)
        
        return ReasoningConfiguration(
            provider=provider,
            model=model,
            config=config,
            estimated_cost_per_1k_input=self.PRICING[provider][model] / 1000,
            estimated_latency_seconds=latency,
            notes=notes
        )
    
    def _configure_openai(self, complexity: str, latency: str) -> dict:
        effort_map = {
            ("simple", "strict"): "low",
            ("simple", "moderate"): "low",
            ("simple", "relaxed"): "medium",
            ("moderate", "strict"): "low",
            ("moderate", "moderate"): "medium",
            ("moderate", "relaxed"): "high",
            ("complex", "strict"): "medium",
            ("complex", "moderate"): "high",
            ("complex", "relaxed"): "high"
        }
        
        effort = effort_map.get((complexity, latency), "medium")
        
        return {
            "reasoning": {
                "effort": effort,
                "summary": "auto"
            },
            "max_output_tokens": {"low": 8000, "medium": 16000, "high": 32000}[effort]
        }
    
    def _configure_anthropic(self, complexity: str, latency: str) -> dict:
        budget_map = {
            "simple": 2000,
            "moderate": 5000,
            "complex": 15000
        }
        
        # Reduce budget for strict latency
        base_budget = budget_map[complexity]
        if latency == "strict":
            base_budget = max(1024, base_budget // 2)
        
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": base_budget
            },
            "max_tokens": base_budget + 8000  # Room for output
        }
    
    def _configure_gemini(self, complexity: str, latency: str) -> dict:
        level_map = {
            ("simple", "strict"): "minimal",
            ("simple", "moderate"): "low",
            ("simple", "relaxed"): "medium",
            ("moderate", "strict"): "low",
            ("moderate", "moderate"): "medium",
            ("moderate", "relaxed"): "high",
            ("complex", "strict"): "medium",
            ("complex", "moderate"): "high",
            ("complex", "relaxed"): "high"
        }
        
        level = level_map.get((complexity, latency), "medium")
        
        return {
            "thinking_config": {
                "thinking_level": level,
                "include_thoughts": complexity != "simple"
            }
        }
    
    def _estimate_latency(self, complexity: str, provider: str) -> tuple[int, int]:
        # Rough estimates in seconds
        estimates = {
            ("simple", "gemini"): (1, 3),
            ("simple", "openai"): (2, 5),
            ("simple", "anthropic"): (2, 5),
            ("moderate", "gemini"): (3, 8),
            ("moderate", "openai"): (5, 15),
            ("moderate", "anthropic"): (5, 15),
            ("complex", "gemini"): (5, 15),
            ("complex", "openai"): (10, 45),
            ("complex", "anthropic"): (10, 45)
        }
        
        return estimates.get((complexity, provider), (5, 20))

# Usage
manager = ReasoningConfigManager()

config = manager.configure(
    task_complexity="complex",
    cost_budget="medium",
    latency_tolerance="relaxed",
    provider_preference=None
)

print(f"Provider: {config.provider}")
print(f"Model: {config.model}")
print(f"Config: {config.config}")
print(f"Estimated latency: {config.estimated_latency_seconds[0]}-{config.estimated_latency_seconds[1]}s")
print(f"Notes: {config.notes}")
```

</details>

---

## Summary

✅ **Effort levels:** OpenAI (low/medium/high), Anthropic (budget_tokens), Gemini (thinking_level)
✅ **Token reservation:** Reserve 25,000+ tokens for high-effort reasoning
✅ **Summaries:** OpenAI has `summary`, Anthropic provides raw thinking, Gemini has `include_thoughts`
✅ **Tool use:** Preserve reasoning items/thinking blocks across tool calls
✅ **Optimization:** Adaptive effort selection, caching, and batching reduce costs

**Next:** [Lesson Overview](./00-prompting-reasoning-models-overview.md) | [Back to Unit Overview](../00-overview.md)

---

## Further Reading

- [OpenAI Context Length Guide](https://platform.openai.com/docs/guides/context-length)
- [Anthropic Token Counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [Gemini Context Windows](https://ai.google.dev/gemini-api/docs/models/gemini)

---

<!-- 
Sources Consulted:
- OpenAI Reasoning Guide: effort levels, summaries, token reservation
- Anthropic Extended Thinking: budget_tokens constraints, signature preservation
- Gemini Thinking: thinking_level, thinking_budget, thought signatures
-->
