---
title: "LiteLLM Model Support"
---

# LiteLLM Model Support

## Introduction

The OpenAI Agents SDK isn't locked to OpenAI models. Through **LiteLLM integration**, we can run agents with 100+ models from providers like Anthropic, Google, Mistral, Cohere, and more — using the exact same agent code. Change one line (the model name) and your agent runs on Claude, Gemini, or any LiteLLM-supported model.

### What we'll cover

- Installing and configuring LiteLLM support
- Using `LitellmModel` with different providers
- Provider-specific configuration (API keys, endpoints)
- Usage tracking with `include_usage`
- Tracing with non-OpenAI models
- Troubleshooting common issues

### Prerequisites

- [Agent Class Fundamentals](./01-agent-class-fundamentals.md)
- [Runner Execution Model](./02-runner-execution-model.md)
- Install LiteLLM extras: `pip install 'openai-agents[litellm]'`

---

## Setting up LiteLLM

Install the LiteLLM extension and configure your provider API keys:

```bash
pip install 'openai-agents[litellm]'
```

```python
import os

# Set API keys for your providers
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["GOOGLE_API_KEY"] = "AIza..."
os.environ["MISTRAL_API_KEY"] = "..."
```

---

## Using LitellmModel

Replace the default model with `LitellmModel` to use any supported provider:

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Anthropic Claude
claude_agent = Agent(
    name="Claude Agent",
    instructions="You are helpful. Respond concisely.",
    model=LitellmModel(model="anthropic/claude-sonnet-4-20250514"),
)

result = Runner.run_sync(claude_agent, "What is the capital of France?")
print(result.final_output)
```

**Output:**
```
The capital of France is Paris.
```

### Multiple providers, same agent code

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Define the agent once
def create_agent(model_name: str) -> Agent:
    return Agent(
        name="Assistant",
        instructions="Be helpful and concise.",
        model=LitellmModel(model=model_name),
    )

# Run with different providers
openai_agent = create_agent("openai/gpt-4o")
claude_agent = create_agent("anthropic/claude-sonnet-4-20250514")
gemini_agent = create_agent("gemini/gemini-2.0-flash")
mistral_agent = create_agent("mistral/mistral-large-latest")

# Same code, different models
for agent in [openai_agent, claude_agent, gemini_agent, mistral_agent]:
    result = Runner.run_sync(agent, "Say hello in one sentence.")
    print(f"{agent.model.model}: {result.final_output}")
```

---

## Supported providers

LiteLLM supports 100+ providers. Here are the most commonly used:

| Provider | Model format | Example |
|----------|-------------|---------|
| **OpenAI** | `openai/model-name` | `openai/gpt-4o` |
| **Anthropic** | `anthropic/model-name` | `anthropic/claude-sonnet-4-20250514` |
| **Google Gemini** | `gemini/model-name` | `gemini/gemini-2.0-flash` |
| **Mistral** | `mistral/model-name` | `mistral/mistral-large-latest` |
| **Cohere** | `cohere/model-name` | `cohere/command-r-plus` |
| **AWS Bedrock** | `bedrock/model-name` | `bedrock/anthropic.claude-3-sonnet` |
| **Azure OpenAI** | `azure/deployment-name` | `azure/gpt-4o-deployment` |
| **Ollama** | `ollama/model-name` | `ollama/llama3.1` |
| **Together AI** | `together_ai/model-name` | `together_ai/meta-llama/Meta-Llama-3.1-70B` |

> **💡 Tip:** The full model format is `provider/model-name`. Check [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers) for the complete list.

---

## Provider-specific configuration

### Custom API keys

```python
from agents.extensions.models.litellm_model import LitellmModel

# Pass API key directly (instead of environment variable)
model = LitellmModel(
    model="anthropic/claude-sonnet-4-20250514",
    api_key="sk-ant-your-key-here",
)
```

### Custom endpoints

```python
# Self-hosted or proxy endpoints
model = LitellmModel(
    model="openai/gpt-4o",
    api_base="https://my-proxy.example.com/v1",
    api_key="my-proxy-key",
)
```

### Ollama (local models)

```python
# Run models locally with Ollama
model = LitellmModel(
    model="ollama/llama3.1",
    api_base="http://localhost:11434",
)

agent = Agent(
    name="Local Agent",
    instructions="Be helpful.",
    model=model,
)
```

---

## Usage tracking

Track token usage across providers with `ModelSettings`:

```python
from agents import Agent, Runner, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

agent = Agent(
    name="Usage Tracker",
    instructions="Be concise.",
    model=LitellmModel(model="anthropic/claude-sonnet-4-20250514"),
    model_settings=ModelSettings(include_usage=True),
)

result = Runner.run_sync(agent, "Hello!")
print(result.final_output)

# Check token usage from raw responses
for response in result.raw_responses:
    if hasattr(response, 'usage') and response.usage:
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
```

---

## Tracing with non-OpenAI models

By default, traces are sent to OpenAI's dashboard. When using non-OpenAI models, we need to configure a separate API key for tracing:

```python
from agents.tracing import set_tracing_export_api_key

# Use an OpenAI key specifically for trace export
set_tracing_export_api_key("sk-your-openai-key-for-tracing")

# Now traces from Claude, Gemini, etc. appear in the OpenAI dashboard
agent = Agent(
    name="Claude Agent",
    model=LitellmModel(model="anthropic/claude-sonnet-4-20250514"),
    instructions="Be helpful.",
)
result = Runner.run_sync(agent, "Hello!")
```

Alternatively, disable tracing entirely:

```python
from agents import RunConfig

config = RunConfig(tracing_disabled=True)
result = Runner.run_sync(agent, "Hello!", run_config=config)
```

---

## Tools with non-OpenAI models

Tools work across providers, but some models handle function calling differently:

```python
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city.
    
    Args:
        city: The city name.
    """
    return f"72°F and sunny in {city}"

# Tools work with any provider that supports function calling
agent = Agent(
    name="Weather Bot",
    instructions="Help with weather queries.",
    model=LitellmModel(model="anthropic/claude-sonnet-4-20250514"),
    tools=[get_weather],
)

result = Runner.run_sync(agent, "Weather in London?")
print(result.final_output)
```

> **Warning:** Not all models support function calling equally well. Claude, GPT-4, and Gemini have strong support. Smaller or older models may struggle with complex tool schemas.

---

## Troubleshooting

### Serialization issues

Some providers have serialization quirks. Enable the LiteLLM serializer patch:

```bash
export OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH=true
```

Or in Python:

```python
import os
os.environ["OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH"] = "true"
```

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` | Missing or invalid API key | Set the correct `PROVIDER_API_KEY` env var |
| `Model not found` | Wrong model format | Use `provider/model-name` format |
| `Connection refused` | Ollama not running | Start Ollama: `ollama serve` |
| `Tool calling not supported` | Model doesn't support tools | Use a model with function calling support |
| Serialization errors | Provider-specific format issues | Set `OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH=true` |

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use `provider/model` format consistently | Prevents ambiguous model resolution |
| Set API keys via environment variables | Keeps secrets out of code |
| Test tools with your target model | Function calling support varies by provider |
| Enable serializer patch proactively | Prevents hard-to-debug serialization errors |
| Track usage across providers | Different pricing — monitor costs per provider |
| Use `set_tracing_export_api_key` for traces | Non-OpenAI models won't send traces without it |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting the provider prefix | Use `anthropic/claude-sonnet-4-20250514`, not just `claude-sonnet-4-20250514` |
| Hardcoding API keys | Use `os.environ` or `.env` files |
| Assuming all models support tools | Check provider docs for function calling support |
| Not installing the extras | Run `pip install 'openai-agents[litellm]'` |
| Using OpenAI tracing with non-OpenAI models | Set `set_tracing_export_api_key()` or disable tracing |

---

## Hands-on exercise

### Your task

Build a **model comparison agent** that runs the same prompt across multiple providers and compares responses.

### Requirements

1. Create agents for 3 different providers (e.g., OpenAI, Anthropic, Gemini)
2. Send the same prompt to all three
3. Compare response quality, length, and (if available) token usage
4. Print a formatted comparison table

### Expected result

A side-by-side comparison of how different models respond to the same prompt.

<details>
<summary>💡 Hints (click to expand)</summary>

- Create a list of `(name, model_string)` tuples
- Use `LitellmModel(model=model_string)` for each
- Measure `len(result.final_output)` for response length
- Use `time.time()` to measure latency
- Format as a table with columns: Provider, Output (truncated), Length, Time

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

models = [
    ("GPT-4o", "openai/gpt-4o"),
    ("Claude Sonnet", "anthropic/claude-sonnet-4-20250514"),
    ("Gemini Flash", "gemini/gemini-2.0-flash"),
]

prompt = "Explain what an API is in exactly 2 sentences."

print(f"Prompt: {prompt}\n")
print(f"{'Provider':<16} {'Response':<60} {'Length':>6} {'Time':>6}")
print("-" * 92)

for name, model_name in models:
    agent = Agent(
        name=f"{name} Agent",
        instructions="Follow instructions precisely.",
        model=LitellmModel(model=model_name),
    )
    
    start = time.time()
    try:
        result = Runner.run_sync(agent, prompt)
        elapsed = time.time() - start
        output = result.final_output
        truncated = output[:57] + "..." if len(output) > 60 else output
        print(f"{name:<16} {truncated:<60} {len(output):>6} {elapsed:>5.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"{name:<16} {'ERROR: ' + str(e)[:53]:<60} {'N/A':>6} {elapsed:>5.1f}s")
```

</details>

### Bonus challenges

- [ ] Add structured output and compare which models produce valid Pydantic objects
- [ ] Test tool calling accuracy across providers with a multi-tool agent
- [ ] Add Ollama/local model to the comparison (requires Ollama running locally)

---

## Summary

✅ `LitellmModel(model="provider/model")` enables 100+ models from any provider

✅ Same agent code works across OpenAI, Anthropic, Google, Mistral, Cohere, and more

✅ Set `OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH=true` to avoid serialization issues

✅ Use `set_tracing_export_api_key()` to send traces from non-OpenAI models to the dashboard

✅ Not all models support function calling — test tools with your target provider

**Next:** [Computer Use Capabilities](./12-computer-use-capabilities.md)

---

## Further reading

- [LiteLLM integration docs](https://openai.github.io/openai-agents-python/models/litellm/) — SDK LiteLLM guide
- [LiteLLM providers](https://docs.litellm.ai/docs/providers) — Full list of supported providers
- [LiteLLM GitHub](https://github.com/BerriAI/litellm) — LiteLLM source and docs

---

*[Back to OpenAI Agents SDK Overview](./00-openai-agents-sdk.md)*

<!-- 
Sources Consulted:
- LiteLLM integration: https://openai.github.io/openai-agents-python/models/litellm/
- LiteLLM providers: https://docs.litellm.ai/docs/providers
- OpenAI Agents SDK models: https://openai.github.io/openai-agents-python/models/
-->
