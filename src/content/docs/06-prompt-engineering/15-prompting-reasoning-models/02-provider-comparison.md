---
title: "Provider Comparison: Reasoning APIs"
---

# Provider Comparison: Reasoning APIs

## Introduction

Each major AI provider implements reasoning capabilities differently—different APIs, parameters, and behaviors. This lesson provides a comprehensive comparison of OpenAI, Anthropic, and Google's reasoning implementations, with working code examples for each.

> **🔑 Key Insight:** While the underlying concept is similar (let models "think" before responding), the implementations vary significantly in parameter names, configuration options, and billing models.

### What We'll Cover

- OpenAI Responses API with reasoning configuration
- Anthropic Messages API with extended thinking
- Google Gemini with thinking modes
- Side-by-side API comparison
- Streaming implementations for each provider
- Tool use with reasoning across providers

### Prerequisites

- [When to Use Reasoning Models](./01-when-to-use-reasoning-models.md)
- API keys for providers you want to use

---

## OpenAI Reasoning API

### Responses API Overview

OpenAI's reasoning models use the newer **Responses API** (not Chat Completions) for full control over reasoning behavior.

```python
from openai import OpenAI

client = OpenAI()

# Basic reasoning request
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        {"role": "user", "content": "Analyze the trade-offs between microservices and monolithic architecture."}
    ]
)

print(response.output_text)
```

### Available Models

| Model | Best For | Reasoning Capability |
|-------|----------|---------------------|
| `gpt-5` | Complex reasoning, highest quality | Full |
| `gpt-5-mini` | Balanced cost/performance | Full |
| `gpt-5-nano` | High volume, simpler reasoning | Limited |
| `o3` | Specialized reasoning tasks | Full |
| `o4-mini` | Cost-effective reasoning | Full |

### Reasoning Configuration Options

```python
# Full reasoning configuration
response = client.responses.create(
    model="gpt-5",
    reasoning={
        "effort": "high",           # "low" | "medium" | "high"
        "summary": "auto"           # "auto" | "none" | "detailed"
    },
    max_output_tokens=16000,        # Reserve tokens for reasoning
    input=[
        {
            "role": "developer",    # Use "developer" not "system"
            "content": "You are a senior software architect."
        },
        {
            "role": "user",
            "content": "Review this system design for scalability issues."
        }
    ]
)
```

### Understanding Reasoning Effort

| Effort Level | Reasoning Tokens | Use Case | Latency |
|--------------|------------------|----------|---------|
| `low` | Minimal | Simpler reasoning tasks | Fastest |
| `medium` | Moderate | Balanced quality/speed | Medium |
| `high` | Extensive | Complex analysis, planning | Slowest |

```python
# Effort level comparison
efforts = ["low", "medium", "high"]

for effort in efforts:
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": effort},
        input=[{
            "role": "user",
            "content": "What's the optimal database choice for a time-series IoT application?"
        }]
    )
    
    print(f"\n{effort.upper()} effort:")
    print(f"  Reasoning tokens: {response.usage.reasoning_tokens}")
    print(f"  Output tokens: {response.usage.completion_tokens}")
```

### Accessing Reasoning Summaries

```python
# Request reasoning summary
response = client.responses.create(
    model="gpt-5",
    reasoning={
        "effort": "high",
        "summary": "auto"
    },
    input=[{
        "role": "user",
        "content": "Analyze whether we should migrate to Kubernetes."
    }]
)

# Access the response content
for item in response.output:
    if item.type == "reasoning":
        print("Reasoning Summary:")
        print(item.summary)  # Array of summary strings
    elif item.type == "message":
        print("\nFinal Response:")
        print(item.content[0].text)
```

### Developer Messages

> **Important:** OpenAI reasoning models use `developer` role instead of `system` for instructions.

```python
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        {
            "role": "developer",
            "content": """
            You are a code review assistant.
            Focus on security vulnerabilities and performance issues.
            Be specific about file and line numbers.
            """
        },
        {
            "role": "user",
            "content": "Review this authentication module: [code]"
        }
    ]
)
```

### OpenAI Streaming

```python
# Streaming with reasoning
stream = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": "Create a deployment strategy."}],
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.reasoning_summary.delta":
        # Handle reasoning summary streaming
        pass
```

---

## Anthropic Extended Thinking

### Messages API with Thinking

Anthropic's Claude models use `thinking` configuration in the Messages API:

```python
import anthropic

client = anthropic.Anthropic()

# Basic extended thinking request
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[
        {"role": "user", "content": "Analyze the security implications of this API design."}
    ]
)

# Process response blocks
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"\nResponse: {block.text}")
```

### Available Models

| Model | Thinking Support | Notes |
|-------|------------------|-------|
| `claude-opus-4-20250514` | Full | Highest quality, thinking preserved by default |
| `claude-sonnet-4-20250514` | Full | Balanced performance |
| `claude-sonnet-4-5-20250514` | Full | Latest Sonnet |
| `claude-haiku-4-5-20250514` | Full | Fast, cost-effective |
| Claude 4 models | Summarized | Thinking content is summarized, not raw |

### Budget Tokens Configuration

```python
# Minimum budget: 1,024 tokens
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # Must be >= 1024, < max_tokens
    },
    messages=[...]
)

# Usage information
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
# Note: Anthropic bundles thinking tokens into output tokens
```

### Interleaved Thinking (Beta)

For complex multi-step tasks, Claude can think multiple times throughout a response:

```python
# Enable interleaved thinking with beta header
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{
        "role": "user",
        "content": "Solve this step by step, showing your work: [complex problem]"
    }]
)

# Response may contain alternating thinking and text blocks
for block in response.content:
    print(f"[{block.type}]: {block.thinking if block.type == 'thinking' else block.text}")
```

### Thinking Signatures for Multi-Turn

```python
# First turn with thinking
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[{"role": "user", "content": "Analyze this codebase for refactoring opportunities."}]
)

# Extract all blocks including signature
all_content = response1.content

# Second turn - must include thinking blocks with signatures
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[
        {"role": "user", "content": "Analyze this codebase."},
        {"role": "assistant", "content": all_content},  # Includes thinking with signature
        {"role": "user", "content": "Focus on the authentication module specifically."}
    ]
)
```

### Anthropic Streaming

```python
# Streaming extended thinking
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[{"role": "user", "content": "Create a migration plan."}]
) as stream:
    current_type = None
    
    for event in stream:
        if hasattr(event, 'type'):
            if event.type == "content_block_start":
                current_type = event.content_block.type
                print(f"\n[{current_type.upper()}]")
            elif event.type == "content_block_delta":
                if hasattr(event.delta, 'thinking'):
                    print(event.delta.thinking, end="", flush=True)
                elif hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
```

### Tool Use with Extended Thinking

> **Critical:** When using tools with extended thinking, you must preserve thinking blocks in the conversation.

```python
# Tool definition
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]

# First request - may return tool_use
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}]
)

# Check for tool use
if response1.stop_reason == "tool_use":
    # Find tool use block
    tool_block = next(b for b in response1.content if b.type == "tool_use")
    
    # Execute tool
    weather_result = get_weather(tool_block.input["location"])
    
    # Continue conversation - MUST include all content blocks
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": response1.content},  # Includes thinking!
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": weather_result
            }]}
        ]
    )
```

---

## Google Gemini Thinking

### Thinking Configuration

Gemini offers two approaches depending on the model version:

```python
from google import genai

client = genai.Client()

# Gemini 3 series: Use thinking_level
response = client.models.generate_content(
    model="gemini-3-flash",
    contents="Analyze the performance bottlenecks in this code.",
    config={
        "thinking_config": {
            "thinking_level": "medium"  # minimal | low | medium | high
        }
    }
)

# Gemini 2.5 series: Use thinking_budget
response = client.models.generate_content(
    model="gemini-2.5-flash-preview",
    contents="Create an optimization plan.",
    config={
        "thinking_config": {
            "thinking_budget": 8000  # 0-32768, or -1 for dynamic
        }
    }
)
```

### Available Models

| Model | Configuration | Default Thinking |
|-------|---------------|------------------|
| `gemini-3-pro` | `thinking_level` | Cannot be disabled |
| `gemini-3-flash` | `thinking_level` | High (dynamic) |
| `gemini-2.5-pro-preview` | `thinking_budget` | Enabled |
| `gemini-2.5-flash-preview` | `thinking_budget` | Enabled |

### Thinking Levels

```python
# Gemini 3 thinking levels
levels = ["minimal", "low", "medium", "high"]

for level in levels:
    response = client.models.generate_content(
        model="gemini-3-flash",
        contents="Explain quantum entanglement.",
        config={
            "thinking_config": {"thinking_level": level}
        }
    )
    
    print(f"\n{level.upper()}:")
    print(f"  Thinking tokens: {response.usage_metadata.thinking_token_count}")
```

### Accessing Thought Summaries

```python
# Request thought summaries
response = client.models.generate_content(
    model="gemini-3-flash",
    contents="Design a caching strategy for this API.",
    config={
        "thinking_config": {
            "thinking_level": "high",
            "include_thoughts": True  # Include thought summaries
        }
    }
)

# Access thoughts in response
for part in response.candidates[0].content.parts:
    if hasattr(part, 'thought') and part.thought:
        print(f"Thought: {part.text}")
    else:
        print(f"Response: {part.text}")
```

### Thought Signatures for Multi-Turn

```python
# First turn
response1 = client.models.generate_content(
    model="gemini-3-flash",
    contents="Analyze this dataset for anomalies.",
    config={
        "thinking_config": {"thinking_level": "medium"}
    }
)

# Must preserve thought signatures in conversation history
history = [
    {"role": "user", "parts": [{"text": "Analyze this dataset."}]},
    {"role": "model", "parts": response1.candidates[0].content.parts}  # Includes signatures
]

# Second turn
response2 = client.models.generate_content(
    model="gemini-3-flash",
    contents=[
        *history,
        {"role": "user", "parts": [{"text": "Focus on the outliers in column A."}]}
    ],
    config={
        "thinking_config": {"thinking_level": "medium"}
    }
)
```

### Gemini Streaming

```python
# Streaming with thinking
for chunk in client.models.generate_content_stream(
    model="gemini-3-flash",
    contents="Create a step-by-step deployment plan.",
    config={
        "thinking_config": {
            "thinking_level": "high",
            "include_thoughts": True
        }
    }
):
    for part in chunk.candidates[0].content.parts:
        if hasattr(part, 'thought') and part.thought:
            print(f"[THINKING] {part.text}", end="", flush=True)
        else:
            print(part.text, end="", flush=True)
```

---

## Side-by-Side Comparison

### Configuration Reference

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| **API** | Responses API | Messages API | Generate Content |
| **Parameter** | `reasoning.effort` | `thinking.budget_tokens` | `thinking_level` / `thinking_budget` |
| **Values** | "low", "medium", "high" | 1024+ integer | "minimal"..."high" / 0-32768 |
| **System Prompt** | `developer` role | `system` parameter | System instruction |
| **Summaries** | `reasoning.summary` | N/A (raw thinking) | `include_thoughts` |
| **Disable** | N/A (always on) | Remove `thinking` | `thinking_budget: 0` (some models) |
| **Minimum** | N/A | 1,024 tokens | N/A |

### Unified Wrapper Example

```python
from typing import Literal, Optional
from dataclasses import dataclass

@dataclass
class ReasoningConfig:
    effort: Literal["low", "medium", "high"] = "medium"
    include_summary: bool = False

class UnifiedReasoningClient:
    """
    Unified interface for reasoning models across providers.
    """
    
    def __init__(self, provider: Literal["openai", "anthropic", "gemini"]):
        self.provider = provider
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client()
    
    def _effort_to_budget(self, effort: str) -> int:
        """Convert effort level to token budget for Anthropic."""
        budgets = {"low": 2000, "medium": 5000, "high": 10000}
        return budgets[effort]
    
    def _effort_to_level(self, effort: str) -> str:
        """Convert effort level to Gemini thinking level."""
        levels = {"low": "low", "medium": "medium", "high": "high"}
        return levels[effort]
    
    def generate(
        self,
        prompt: str,
        config: ReasoningConfig,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with reasoning."""
        
        if self.provider == "openai":
            return self._generate_openai(prompt, config, system_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, config, system_prompt)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, config, system_prompt)
    
    def _generate_openai(self, prompt, config, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.responses.create(
            model="gpt-5",
            reasoning={
                "effort": config.effort,
                "summary": "auto" if config.include_summary else "none"
            },
            input=messages
        )
        return response.output_text
    
    def _generate_anthropic(self, prompt, config, system_prompt):
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": self._effort_to_budget(config.effort)
            },
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        # Extract text content
        for block in response.content:
            if block.type == "text":
                return block.text
    
    def _generate_gemini(self, prompt, config, system_prompt):
        generation_config = {
            "thinking_config": {
                "thinking_level": self._effort_to_level(config.effort),
                "include_thoughts": config.include_summary
            }
        }
        
        if system_prompt:
            generation_config["system_instruction"] = system_prompt
        
        response = self.client.models.generate_content(
            model="gemini-3-flash",
            contents=prompt,
            config=generation_config
        )
        
        return response.text

# Usage
client = UnifiedReasoningClient("openai")
response = client.generate(
    prompt="Analyze the trade-offs of this architecture.",
    config=ReasoningConfig(effort="high", include_summary=True),
    system_prompt="You are a senior architect."
)
```

---

## Token Economics Comparison

### Pricing Structure

| Provider | Model | Input | Output | Reasoning |
|----------|-------|-------|--------|-----------|
| OpenAI | GPT-5 | $10/M | $30/M | Included in output |
| OpenAI | GPT-5-mini | $5/M | $15/M | Included in output |
| Anthropic | Sonnet 4 | $3/M | $15/M | Included in output |
| Anthropic | Opus 4 | $15/M | $75/M | Included in output |
| Gemini | 3 Flash | $0.15/M | $0.60/M | Separate (lower rate) |
| Gemini | 3 Pro | $1.25/M | $10/M | Separate (lower rate) |

### Cost Calculation Example

```python
def compare_provider_costs(
    input_tokens: int,
    reasoning_tokens: int,
    output_tokens: int
) -> dict:
    """
    Compare costs across providers for the same task.
    """
    
    costs = {}
    
    # OpenAI GPT-5
    costs["openai_gpt5"] = (
        (input_tokens / 1_000_000) * 10 +
        ((reasoning_tokens + output_tokens) / 1_000_000) * 30
    )
    
    # Anthropic Sonnet 4
    costs["anthropic_sonnet"] = (
        (input_tokens / 1_000_000) * 3 +
        ((reasoning_tokens + output_tokens) / 1_000_000) * 15
    )
    
    # Gemini 3 Flash (reasoning billed at output rate)
    costs["gemini_flash"] = (
        (input_tokens / 1_000_000) * 0.15 +
        ((reasoning_tokens + output_tokens) / 1_000_000) * 0.60
    )
    
    return {
        provider: f"${cost:.4f}"
        for provider, cost in costs.items()
    }

# Example task
print(compare_provider_costs(
    input_tokens=2000,
    reasoning_tokens=5000,
    output_tokens=500
))

# Output:
# {
#     "openai_gpt5": "$0.1850",
#     "anthropic_sonnet": "$0.0885",
#     "gemini_flash": "$0.0036"
# }
```

---

## Common Mistakes

### ❌ Using Wrong Role Names

```python
# Wrong: Using "system" with OpenAI reasoning models
response = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "system", "content": "Be helpful"},  # ❌ Wrong
        {"role": "user", "content": "Help me"}
    ]
)

# Right: Use "developer" for OpenAI
response = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "developer", "content": "Be helpful"},  # ✅ Correct
        {"role": "user", "content": "Help me"}
    ]
)
```

### ❌ Insufficient Token Budget

```python
# Wrong: Budget less than minimum
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 500  # ❌ Below 1,024 minimum
    },
    messages=[...]
)

# Right: Use at least 1,024
thinking={
    "type": "enabled",
    "budget_tokens": 1024  # ✅ Minimum
}
```

### ❌ Not Preserving Thinking in Multi-Turn

```python
# Wrong: Only including text in follow-up
messages=[
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": text_only},  # ❌ Missing thinking blocks
    {"role": "user", "content": "Follow-up"}
]

# Right: Include all content blocks
messages=[
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": response1.content},  # ✅ Full content
    {"role": "user", "content": "Follow-up"}
]
```

---

## Hands-on Exercise

### Your Task

Create a function that queries all three providers for the same task and compares results.

**Requirements:**
1. Accept a task description
2. Query each provider with equivalent settings
3. Return response text and token usage
4. Handle provider-specific errors

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class ProviderResult:
    provider: str
    response: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    error: Optional[str] = None

async def compare_providers(task: str) -> list[ProviderResult]:
    """Query all providers and compare results."""
    
    results = []
    
    # OpenAI
    try:
        from openai import OpenAI
        client = OpenAI()
        
        response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "medium"},
            input=[{"role": "user", "content": task}]
        )
        
        results.append(ProviderResult(
            provider="OpenAI GPT-5-mini",
            response=response.output_text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning_tokens=response.usage.reasoning_tokens
        ))
    except Exception as e:
        results.append(ProviderResult(
            provider="OpenAI GPT-5-mini",
            response="",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            error=str(e)
        ))
    
    # Anthropic
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            thinking={"type": "enabled", "budget_tokens": 5000},
            messages=[{"role": "user", "content": task}]
        )
        
        text = next(
            b.text for b in response.content if b.type == "text"
        )
        
        results.append(ProviderResult(
            provider="Anthropic Claude Sonnet 4",
            response=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=0  # Bundled in output
        ))
    except Exception as e:
        results.append(ProviderResult(
            provider="Anthropic Claude Sonnet 4",
            response="",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            error=str(e)
        ))
    
    # Gemini
    try:
        from google import genai
        client = genai.Client()
        
        response = client.models.generate_content(
            model="gemini-3-flash",
            contents=task,
            config={
                "thinking_config": {"thinking_level": "medium"}
            }
        )
        
        results.append(ProviderResult(
            provider="Google Gemini 3 Flash",
            response=response.text,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
            reasoning_tokens=response.usage_metadata.thinking_token_count
        ))
    except Exception as e:
        results.append(ProviderResult(
            provider="Google Gemini 3 Flash",
            response="",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            error=str(e)
        ))
    
    return results

# Usage
task = "What are the key considerations when designing a rate limiter?"

results = asyncio.run(compare_providers(task))

for r in results:
    print(f"\n{'='*50}")
    print(f"Provider: {r.provider}")
    if r.error:
        print(f"Error: {r.error}")
    else:
        print(f"Tokens - Input: {r.input_tokens}, Output: {r.output_tokens}, Reasoning: {r.reasoning_tokens}")
        print(f"Response preview: {r.response[:200]}...")
```

</details>

---

## Summary

✅ **OpenAI:** Responses API with `reasoning.effort` and developer messages
✅ **Anthropic:** Messages API with `thinking.budget_tokens` and signature preservation
✅ **Gemini:** `thinking_level` for 3.x or `thinking_budget` for 2.5 series
✅ **Key differences:** Parameter names, minimum budgets, multi-turn handling
✅ **Unified wrapper:** Abstract provider differences for portable code

**Next:** [Prompting Best Practices](./03-prompting-best-practices.md)

---

## Further Reading

- [OpenAI Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
- [Anthropic Extended Thinking Guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Gemini Thinking Documentation](https://ai.google.dev/gemini-api/docs/thinking)

---

<!-- 
Sources Consulted:
- OpenAI Reasoning Guide: platform.openai.com/docs/guides/reasoning
- OpenAI Responses API Reference: platform.openai.com/docs/api-reference/responses
- Anthropic Extended Thinking: docs.anthropic.com/en/docs/build-with-claude/extended-thinking
- Gemini Thinking Mode: ai.google.dev/gemini-api/docs/thinking
-->
