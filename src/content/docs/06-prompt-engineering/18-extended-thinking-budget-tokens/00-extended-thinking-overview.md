---
title: "Extended Thinking & Budget Tokens"
---

# Extended Thinking & Budget Tokens

## Introduction

Extended thinking gives AI models the ability to reason through complex problems before responding. Instead of generating output immediately, the model first works through the problem step-by-step in a "thinking" phase, then produces a more carefully considered response.

This capability dramatically improves performance on tasks requiring multi-step reasoning: complex math, intricate coding problems, nuanced analysis, and strategic planning.

> **🔑 Key Insight:** Extended thinking isn't just "thinking longer"—it's a fundamentally different mode where the model can explore, backtrack, and refine its approach before committing to a response.

### What We'll Cover

- What extended thinking is and how it works
- Comparing thinking capabilities across providers
- When to enable extended thinking
- Budget tokens and cost considerations
- Task complexity matching

### Prerequisites

- [Prompting Reasoning Models](../15-prompting-reasoning-models/00-prompting-reasoning-models.md)
- Basic understanding of API usage

---

## What Is Extended Thinking?

### Standard Generation vs Extended Thinking

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STANDARD GENERATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   User Prompt  ─────────────────────────►  Model Response            │
│                      (Direct output)                                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      EXTENDED THINKING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   User Prompt  ────►  [Thinking Phase]  ────►  Model Response        │
│                       │                                               │
│                       ├─ Analyze problem                              │
│                       ├─ Consider approaches                          │
│                       ├─ Work through steps                           │
│                       ├─ Check reasoning                              │
│                       └─ Refine answer                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works Internally

When extended thinking is enabled:

1. **Model receives prompt** with thinking enabled and a token budget
2. **Thinking phase begins**: Model generates internal reasoning tokens
3. **Reasoning continues** until the model determines it has a solution (or budget exhausted)
4. **Response generated**: Model produces final output informed by its thinking
5. **Thinking returned**: API returns both thinking content and final response

---

## Provider Comparison

### Supported Models and Features

| Provider | Models | Thinking Control | Output Visibility |
|----------|--------|------------------|-------------------|
| **Anthropic** | Claude Sonnet 4.5, Opus 4.5, Opus 4, Sonnet 4, Haiku 4.5 | `budget_tokens` | Summarized (Claude 4) or Full (Sonnet 3.7) |
| **Google** | Gemini 3 Pro/Flash, Gemini 2.5 Pro/Flash | `thinkingLevel` or `thinkingBudget` | Thought summaries |
| **OpenAI** | o3, o4-mini, o1 series | `reasoning_effort` | Hidden (summary available in some cases) |

### API Structure Comparison

**Anthropic (Claude):**
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[{"role": "user", "content": "Complex problem..."}]
)
```

**Google (Gemini 3):**
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Complex problem...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="high"  # minimal, low, medium, high
        )
    )
)
```

**Google (Gemini 2.5):**
```python
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Complex problem...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=10000  # Token count
        )
    )
)
```

**OpenAI (o-series):**
```python
response = client.responses.create(
    model="o3",
    reasoning={"effort": "high"},  # low, medium, high
    input=[{"role": "user", "content": "Complex problem..."}]
)
```

---

## When to Use Extended Thinking

### Task Complexity Guide

| Task Complexity | Extended Thinking | Examples |
|-----------------|-------------------|----------|
| **Easy** | Off or minimal | Fact retrieval, simple classification, greetings |
| **Medium** | Default/moderate | Comparisons, analogies, explanations, summarization |
| **Hard** | High budget | Complex math, multi-step coding, strategic planning |
| **Very Hard** | Maximum budget | Research-level problems, novel algorithm design |

### Ideal Use Cases

**✅ Enable extended thinking for:**

- **Complex mathematics**: Multi-step proofs, optimization problems
- **Intricate coding**: Algorithm design, debugging complex systems
- **Strategic analysis**: Business decisions with multiple variables
- **Legal/financial reasoning**: Policy interpretation, compliance analysis
- **Scientific problems**: Hypothesis evaluation, experimental design
- **Ambiguous tasks**: Where the model needs to interpret limited information

**❌ Don't use extended thinking for:**

- Simple Q&A or fact lookup
- Basic text generation
- Straightforward formatting tasks
- High-volume, low-complexity workloads
- Real-time chat where latency matters

### Decision Framework

```python
def should_use_extended_thinking(task: dict) -> tuple[bool, str]:
    """Decide whether to enable extended thinking."""
    
    # Factors that suggest extended thinking
    complexity_signals = [
        task.get("requires_math", False),
        task.get("requires_multi_step_reasoning", False),
        task.get("has_ambiguous_input", False),
        task.get("needs_code_generation", False),
        task.get("involves_analysis", False),
        len(task.get("constraints", [])) > 3,
    ]
    
    # Factors that suggest standard mode
    simplicity_signals = [
        task.get("is_factual_lookup", False),
        task.get("is_simple_formatting", False),
        task.get("requires_low_latency", False),
        task.get("high_volume", False),
    ]
    
    complexity_score = sum(complexity_signals)
    simplicity_score = sum(simplicity_signals)
    
    if simplicity_score > complexity_score:
        return False, "Use standard mode for this task"
    elif complexity_score >= 3:
        return True, "high"  # High thinking budget
    elif complexity_score >= 1:
        return True, "medium"  # Moderate budget
    else:
        return False, "Standard mode sufficient"
```

---

## Understanding Thinking Output

### Anthropic Response Structure

```python
# Claude extended thinking response
{
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me analyze this step by step...",
            "signature": "WaUjzkypQ2mUEVM36O2T..."  # Verification signature
        },
        {
            "type": "text",
            "text": "Based on my analysis, the answer is..."
        }
    ]
}
```

### Gemini Response Structure

```python
# Gemini with thought summaries
for part in response.candidates[0].content.parts:
    if part.thought:
        print("Thought summary:", part.text)
    else:
        print("Answer:", part.text)
```

### What You Get Back

| Provider | Thinking Visibility | What's Included |
|----------|---------------------|-----------------|
| **Claude (Sonnet 3.7)** | Full thinking | Complete internal reasoning |
| **Claude (Claude 4)** | Summarized | Key ideas, condensed for readability |
| **Gemini** | Thought summaries | Incremental reasoning summaries |
| **OpenAI o-series** | Hidden | Reasoning token count only |

> **Note:** Claude 4 models return summarized thinking by default. The full thinking tokens are still billed, but you see a condensed version for efficiency and safety.

---

## Cost and Latency Considerations

### How Thinking Tokens Are Billed

| Provider | Input Billing | Output Billing |
|----------|---------------|----------------|
| **Anthropic** | Standard input | Thinking tokens + output tokens |
| **Gemini** | Standard input | Full thinking tokens (not summary) |
| **OpenAI** | Standard input | Reasoning + output tokens |

### Latency Impact

Extended thinking increases response time:

```
Standard Mode:     ~500ms - 2s
Extended Thinking: ~3s - 60s+ (depending on budget)
```

**Latency management strategies:**

1. **Use streaming** to show progress
2. **Set appropriate budgets** for task complexity
3. **Use batch processing** for thinking budgets >32K tokens
4. **Implement timeouts** for user-facing applications

### Cost Optimization

```python
def estimate_thinking_cost(
    model: str,
    input_tokens: int,
    thinking_budget: int,
    estimated_output: int
) -> float:
    """Estimate cost for extended thinking request."""
    
    # Example rates (check current pricing)
    rates = {
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-opus-4-5": {"input": 0.015, "output": 0.075},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    }
    
    rate = rates.get(model, {"input": 0.003, "output": 0.015})
    
    input_cost = (input_tokens / 1000) * rate["input"]
    # Thinking tokens are billed as output
    thinking_cost = (thinking_budget / 1000) * rate["output"]
    output_cost = (estimated_output / 1000) * rate["output"]
    
    return input_cost + thinking_cost + output_cost
```

---

## Lesson Structure

This lesson is organized into the following files:

| File | Topic |
|------|-------|
| [01-thinking-budget-configuration.md](./01-thinking-budget-configuration.md) | Budget tokens, thinking levels, limits |
| [02-prompting-for-extended-thinking.md](./02-prompting-for-extended-thinking.md) | Prompting techniques for thinking mode |
| [03-thinking-with-tools-streaming.md](./03-thinking-with-tools-streaming.md) | Tool use, interleaved thinking, streaming |
| [04-debugging-optimization.md](./04-debugging-optimization.md) | Inspecting thinking, caching, optimization |

---

## Summary

✅ **Extended thinking** enables deeper reasoning before response generation
✅ **Match budget to complexity**—start small, increase as needed
✅ **Provider differences matter**—Anthropic uses `budget_tokens`, Gemini uses `thinkingLevel` or `thinkingBudget`
✅ **Thinking tokens cost money**—budget appropriately
✅ **Latency increases**—use streaming for better UX

**Next:** [Thinking Budget Configuration](./01-thinking-budget-configuration.md)

---

## Further Reading

- [Anthropic Extended Thinking Guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Gemini Thinking Documentation](https://ai.google.dev/gemini-api/docs/thinking)
- [OpenAI Reasoning Best Practices](https://platform.openai.com/docs/guides/reasoning-best-practices)

---

<!-- 
Sources Consulted:
- Anthropic Extended Thinking: budget_tokens, summarized thinking, model support
- Gemini Thinking: thinkingLevel, thinkingBudget, thought summaries
- OpenAI Reasoning: reasoning effort, when to use reasoning models
-->
