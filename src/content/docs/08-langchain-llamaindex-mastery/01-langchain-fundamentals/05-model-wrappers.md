---
title: "Model Wrappers"
---

# Model Wrappers

## Introduction

LangChain provides a unified interface for interacting with language models from any provider. Whether you're using OpenAI's GPT-4, Anthropic's Claude, Google's Gemini, or self-hosted models, you work with the same methods and patterns.

This abstraction layer is one of LangChain's most powerful features—it enables provider portability, easy experimentation across models, and production patterns like fallbacks and retries.

### What We'll Cover

- `init_chat_model` for quick model initialization
- Provider-specific model classes (ChatOpenAI, ChatAnthropic, etc.)
- Model configuration (temperature, max_tokens, timeout)
- Streaming responses
- Model fallbacks and error handling
- Token counting and usage tracking
- Binding tools and structured output

### Prerequisites

- LangChain and at least one provider package installed
- API keys configured in environment variables
- Understanding of LCEL fundamentals

---

## init_chat_model: The Universal Entry Point

`init_chat_model` is the recommended way to initialize any chat model in LangChain. It automatically selects the right provider class based on the model name.

### Basic Usage

```python
from langchain.chat_models import init_chat_model

# OpenAI (auto-detected from model name)
gpt4 = init_chat_model("gpt-4o")

# Anthropic
claude = init_chat_model("claude-sonnet-4-5-20250929")

# Google Gemini
gemini = init_chat_model("gemini-1.5-pro")

# Use the model
response = gpt4.invoke("What is LangChain?")
print(response.content)
```

### Explicit Provider Specification

Use the `provider:model` format for clarity:

```python
from langchain.chat_models import init_chat_model

# Explicit provider
model = init_chat_model("openai:gpt-4o")
model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
model = init_chat_model("google_genai:gemini-1.5-pro")
```

### Configuration Parameters

Pass common parameters directly:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o",
    temperature=0.7,      # Creativity (0-2)
    max_tokens=1000,      # Max response length
    timeout=30,           # Request timeout in seconds
    max_retries=3         # Retry on transient failures
)

response = model.invoke("Write a creative story opening")
print(response.content)
```

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `temperature` | `float` | Controls randomness (0=deterministic, 2=very creative) | Model-dependent |
| `max_tokens` | `int` | Maximum tokens in response | Model-dependent |
| `timeout` | `float` | Request timeout in seconds | None |
| `max_retries` | `int` | Number of retry attempts | 2 |
| `api_key` | `str` | API key (overrides environment variable) | From env |

---

## Provider-Specific Model Classes

For advanced configuration, use provider-specific classes directly:

### ChatOpenAI

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=2000,
    timeout=30,
    max_retries=3,
    # OpenAI-specific options
    frequency_penalty=0.5,
    presence_penalty=0.5,
    seed=42,  # For reproducibility
)

response = model.invoke("Explain quantum computing")
print(response.content)
```

#### OpenAI-Specific Features

```python
from langchain_openai import ChatOpenAI

# Response format (JSON mode)
model = ChatOpenAI(
    model="gpt-4o",
    response_format={"type": "json_object"}
)

# Using the new Responses API (for specific features)
model = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True
)
```

### ChatAnthropic

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    max_tokens=4096,
    timeout=60,
    # Anthropic-specific options
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)

response = model.invoke("What are the benefits of RAG?")
print(response.content)
```

### ChatGoogleGenerativeAI

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_output_tokens=2048,
    # Google-specific options
    top_p=0.9,
    top_k=40
)

response = model.invoke("Describe the Gemini model architecture")
print(response.content)
```

### Provider Comparison

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Best Model | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro |
| Max Context | 128K | 200K | 2M |
| Streaming | ✅ | ✅ | ✅ |
| Tool Calling | ✅ | ✅ | ✅ |
| JSON Mode | ✅ | ✅ | ✅ |
| Vision | ✅ | ✅ | ✅ |
| Prompt Caching | Implicit | Explicit | ✅ |

---

## Streaming Responses

Streaming is essential for responsive applications:

### Basic Streaming

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

print("Streaming response: ", end="")
for chunk in model.stream("Write a haiku about programming"):
    print(chunk.content, end="", flush=True)
print()
```

**Output:**
```
Streaming response: Code flows like water
Bugs lurk in the shadows deep
Debug, fix, repeat
```

### Streaming with Chunks

Each chunk is an `AIMessageChunk` that can be aggregated:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

full_response = None
token_count = 0

for chunk in model.stream("Explain machine learning briefly"):
    if full_response is None:
        full_response = chunk
    else:
        full_response = full_response + chunk
    
    token_count += 1
    print(chunk.content, end="")

print(f"\n\nTotal chunks: {token_count}")
print(f"Full response length: {len(full_response.content)} chars")
```

### Async Streaming

For web applications:

```python
import asyncio
from langchain.chat_models import init_chat_model

async def stream_response(prompt: str):
    model = init_chat_model("gpt-4o")
    
    async for chunk in model.astream(prompt):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_response("What is AI?"))
```

### Streaming in FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model

app = FastAPI()
model = init_chat_model("gpt-4o")

@app.get("/stream")
async def stream_endpoint(prompt: str):
    async def generate():
        async for chunk in model.astream(prompt):
            yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")
```

---

## Model Fallbacks

Implement resilience with fallback models:

### Simple Fallback

```python
from langchain.chat_models import init_chat_model

# Primary model
primary = init_chat_model("gpt-4o")

# Fallback model (cheaper, more available)
fallback = init_chat_model("gpt-4o-mini")

# Create model with fallback
resilient_model = primary.with_fallbacks([fallback])

# If primary fails, fallback is used automatically
response = resilient_model.invoke("What is Python?")
print(response.content)
```

### Multiple Fallbacks

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o").with_fallbacks([
    init_chat_model("gpt-4o-mini"),
    init_chat_model("claude-haiku-3-5-20241022")
])

# Tries each model in order until one succeeds
response = model.invoke("Explain neural networks")
```

### Fallback with Different Providers

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o")
fallback = ChatAnthropic(model="claude-sonnet-4-5-20250929")

resilient = primary.with_fallbacks([fallback])

# Works even if OpenAI is down
response = resilient.invoke("What is deep learning?")
```

### Retry Configuration

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o",
    max_retries=5,  # Retry up to 5 times
)

# Or add retry to existing model
model_with_retry = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True  # Exponential backoff with jitter
)
```

---

## Token Usage and Counting

### Accessing Token Usage

Token usage is included in response metadata:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")
response = model.invoke("What is Python?")

# Access usage metadata
if response.usage_metadata:
    print(f"Input tokens: {response.usage_metadata['input_tokens']}")
    print(f"Output tokens: {response.usage_metadata['output_tokens']}")
    print(f"Total tokens: {response.usage_metadata['total_tokens']}")
```

### Tracking Usage Across Calls

Use the callback handler for aggregate tracking:

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model = init_chat_model("gpt-4o")
callback = UsageMetadataCallbackHandler()

# Track across multiple calls
result1 = model.invoke("Hello", config={"callbacks": [callback]})
result2 = model.invoke("How are you?", config={"callbacks": [callback]})

print("Aggregate usage:")
print(callback.usage_metadata)
```

### Estimating Tokens Before Calling

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

# Get number of tokens for a message
messages = [{"role": "user", "content": "What is the meaning of life?"}]
token_count = model.get_num_tokens_from_messages(messages)
print(f"Estimated tokens: {token_count}")
```

---

## Binding Tools

Models can be configured to call tools:

### Basic Tool Binding

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

model = init_chat_model("gpt-4o")

# Bind tools to model
model_with_tools = model.bind_tools([get_weather, calculate])

# Model can now request tool calls
response = model_with_tools.invoke("What's 15 * 23?")

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
```

### Forcing Tool Use

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

# Force the model to use a specific tool
model_forced = model.bind_tools(
    [get_weather, calculate],
    tool_choice="get_weather"  # Force this tool
)

# Or require any tool (not optional)
model_required = model.bind_tools(
    [get_weather, calculate],
    tool_choice="required"  # Must use some tool
)
```

---

## Structured Output

Get responses in a specific format:

### Using Pydantic Models

```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    """A movie review with structured fields."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary")

model = init_chat_model("gpt-4o")
structured_model = model.with_structured_output(MovieReview)

review = structured_model.invoke(
    "Review the movie 'Inception' by Christopher Nolan"
)

print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
```

### Using TypedDict

```python
from typing import TypedDict
from langchain.chat_models import init_chat_model

class Person(TypedDict):
    name: str
    age: int
    occupation: str

model = init_chat_model("gpt-4o")
structured_model = model.with_structured_output(Person)

person = structured_model.invoke(
    "Extract info: John Smith is a 35-year-old software engineer"
)

print(person)
# {"name": "John Smith", "age": 35, "occupation": "software engineer"}
```

---

## Model Profiles

Check model capabilities programmatically:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

# Access model profile (requires langchain>=1.1)
if hasattr(model, 'profile'):
    profile = model.profile
    print(f"Max input tokens: {profile.get('max_input_tokens')}")
    print(f"Supports images: {profile.get('image_inputs')}")
    print(f"Supports tool calling: {profile.get('tool_calling')}")
    print(f"Supports reasoning: {profile.get('reasoning_output')}")
```

---

## Configurable Models

Create models that can be configured at runtime:

```python
from langchain.chat_models import init_chat_model

# Create a configurable model (no default model specified)
configurable = init_chat_model(temperature=0)

# Switch models at runtime
result1 = configurable.invoke(
    "Hello",
    config={"configurable": {"model": "gpt-4o"}}
)

result2 = configurable.invoke(
    "Hello",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use init_chat_model** | Provides flexibility and cleaner code |
| **Configure fallbacks** | Always have a backup for production |
| **Set timeouts** | Prevent hanging requests |
| **Track token usage** | Monitor costs and optimize prompts |
| **Use streaming** | For user-facing applications |
| **Pin model versions** | Use specific model versions in production |

### Production Configuration

```python
from langchain.chat_models import init_chat_model

production_model = init_chat_model(
    "gpt-4o-2024-08-06",  # Pin specific version
    temperature=0,         # Deterministic for consistency
    max_tokens=2000,       # Reasonable limit
    timeout=30,            # Don't hang forever
    max_retries=3          # Handle transient failures
).with_fallbacks([
    init_chat_model("gpt-4o-mini", timeout=15)
])
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using unpinned model names | Pin versions: `gpt-4o-2024-08-06` |
| No timeout configuration | Always set `timeout` parameter |
| Missing fallbacks | Add `.with_fallbacks()` for production |
| Ignoring token limits | Check `max_tokens` vs model's context window |
| Not tracking usage | Use `UsageMetadataCallbackHandler` |

---

## Hands-on Exercise

### Your Task

Build a multi-model comparison system that:
1. Sends the same prompt to 3 different models
2. Streams responses from each in parallel (or sequence)
3. Compares token usage and response times
4. Implements fallbacks for each model

### Requirements

1. Use at least 2 different providers (or model tiers)
2. Track token usage for each model
3. Measure response time
4. Implement fallback for at least one model
5. Display a comparison table

### Expected Result

```
Prompt: "Explain machine learning in one sentence"

Results:
| Model          | Response                    | Tokens | Time   |
|----------------|-----------------------------| -------|--------|
| gpt-4o         | Machine learning is...      | 45     | 1.2s   |
| gpt-4o-mini    | ML is a subset of AI...     | 38     | 0.8s   |
| claude-haiku   | Machine learning enables... | 42     | 0.9s   |
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `RunnableParallel` to call models concurrently
- Wrap timing logic in a `RunnableLambda`
- Access `usage_metadata` from responses
- Use `time.time()` for simple timing

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnableLambda

def timed_invoke(model, name: str):
    """Wrap model invocation with timing."""
    def invoke_with_timing(prompt: str):
        start = time.time()
        response = model.invoke(prompt)
        elapsed = time.time() - start
        
        return {
            "name": name,
            "response": response.content[:50] + "...",
            "tokens": response.usage_metadata.get("total_tokens", "N/A") if response.usage_metadata else "N/A",
            "time": f"{elapsed:.2f}s"
        }
    return RunnableLambda(invoke_with_timing)

# Initialize models with fallbacks
gpt4o = init_chat_model("gpt-4o", timeout=10).with_fallbacks([
    init_chat_model("gpt-4o-mini", timeout=5)
])
gpt4o_mini = init_chat_model("gpt-4o-mini", timeout=10)

# If you have Anthropic installed:
try:
    claude = init_chat_model("claude-haiku-3-5-20241022", timeout=10)
    models = {
        "gpt4o": timed_invoke(gpt4o, "gpt-4o"),
        "gpt4o_mini": timed_invoke(gpt4o_mini, "gpt-4o-mini"),
        "claude": timed_invoke(claude, "claude-haiku"),
    }
except:
    # Fallback if Anthropic not available
    gpt35 = init_chat_model("gpt-3.5-turbo", timeout=10)
    models = {
        "gpt4o": timed_invoke(gpt4o, "gpt-4o"),
        "gpt4o_mini": timed_invoke(gpt4o_mini, "gpt-4o-mini"),
        "gpt35": timed_invoke(gpt35, "gpt-3.5-turbo"),
    }

# Create parallel comparison chain
comparison = RunnableParallel(**models)

# Run comparison
prompt = "Explain machine learning in one sentence"
print(f"Prompt: {prompt}\n")
print("Results:")
print("-" * 70)
print(f"| {'Model':<15} | {'Response':<30} | {'Tokens':<7} | {'Time':<6} |")
print("-" * 70)

results = comparison.invoke(prompt)

for model_key, result in results.items():
    print(f"| {result['name']:<15} | {result['response']:<30} | {result['tokens']:<7} | {result['time']:<6} |")

print("-" * 70)
```

</details>

### Bonus Challenges

- [ ] Add streaming support and measure time-to-first-token
- [ ] Implement cost estimation based on token usage
- [ ] Add a "best response" selector using another model as judge

---

## Summary

✅ **init_chat_model** is the universal entry point for any model provider  
✅ Provider-specific classes offer advanced configuration options  
✅ **Streaming** is essential for responsive user experiences  
✅ Use **with_fallbacks()** and **with_retry()** for production reliability  
✅ Track **token usage** via `usage_metadata` or callback handlers  
✅ **bind_tools()** and **with_structured_output()** enable advanced capabilities  

**Next:** [Debugging and Tracing](./06-debugging-and-tracing.md) — Use LangSmith and callbacks for observability

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Core Abstractions](./04-core-abstractions.md) | [LangChain Fundamentals](./00-langchain-fundamentals.md) | [Debugging and Tracing](./06-debugging-and-tracing.md) |

<!-- 
Sources Consulted:
- LangChain Models: https://docs.langchain.com/oss/python/langchain/models
- LangChain Agents: https://docs.langchain.com/oss/python/langchain/agents
- LangChain Quickstart: https://docs.langchain.com/oss/python/langchain/quickstart
-->
