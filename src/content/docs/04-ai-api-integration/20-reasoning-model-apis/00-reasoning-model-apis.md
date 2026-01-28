---
title: "Reasoning Model APIs"
---

# Reasoning Model APIs

## Overview

This lesson covers OpenAI's reasoning models—LLMs trained with reinforcement learning to "think before they answer" by producing internal chains of thought. We'll explore the o-series models, reasoning-specific parameters, token management, and best practices for integrating reasoning capabilities into your applications.

### Topics Covered

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Reasoning Models Overview](./01-reasoning-models-overview.md) | o-series models, GPT-5 reasoning, test-time compute |
| 2 | [Reasoning Parameters](./02-reasoning-parameters.md) | effort settings, max_output_tokens, unsupported params |
| 3 | [Reasoning Tokens](./03-reasoning-tokens.md) | Token types, usage tracking, cost implications |
| 4 | [Multi-Turn Reasoning](./04-multi-turn-reasoning.md) | Passing reasoning items, context preservation |
| 5 | [Encrypted Reasoning](./05-encrypted-reasoning.md) | ZDR support, encrypted items, stateless reasoning |
| 6 | [Reasoning Summaries](./06-reasoning-summaries.md) | Summary parameter, viewing reasoning process |
| 7 | [Best Practices](./07-best-practices.md) | Prompting techniques, monitoring, optimization |

### Quick Comparison: Reasoning vs Standard Models

| Feature | Standard (GPT-4o) | Reasoning (GPT-5, o-series) |
|---------|------------------|---------------------------|
| Think before answering | No | Yes (reasoning tokens) |
| Best for | General tasks | Complex problems |
| Token types | Input + Output | Input + Reasoning + Output |
| Prompt style | Detailed instructions | High-level guidance |
| API | Chat Completions | Responses API (preferred) |
| Cost | Lower per token | Higher (reasoning tokens billed) |

### Quick Start

```python
from openai import OpenAI

client = OpenAI()

# Basic reasoning model usage
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},
    input=[
        {
            "role": "user",
            "content": "Solve: If a train leaves at 9am at 60mph and another at 10am at 80mph, when do they meet?"
        }
    ]
)

print(response.output_text)

# Check reasoning token usage
print(f"Reasoning tokens: {response.usage.output_tokens_details.reasoning_tokens}")
```

### When to Use Reasoning Models

| ✅ Use Reasoning Models For | ❌ Use Standard Models For |
|-----------------------------|---------------------------|
| Complex problem solving | Simple Q&A |
| Multi-step planning | Basic text generation |
| Scientific reasoning | Classification tasks |
| Code implementation | Summarization |
| Agentic workflows | Formatting/extraction |

### Prerequisites

- Completion of AI API Integration fundamentals
- Understanding of tokens and API costs
- Familiarity with the Responses API

---

**Start Learning:** [Reasoning Models Overview](./01-reasoning-models-overview.md)
