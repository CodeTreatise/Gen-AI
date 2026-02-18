---
title: "Temperature Settings for Function Calling"
---

# Temperature Settings for Function Calling

## Introduction

Temperature controls how "creative" or "deterministic" a model's output is. For most text generation, higher temperature means more variety. But function calling is different — you want the model to pick the **right** tool with the **exact** parameters, not the most creative ones.

All three major providers recommend lower temperature for function calling, but with an important exception for Google's newest models that we'll examine in detail.

### What we'll cover

- How temperature affects tool selection and argument generation
- Provider-specific temperature recommendations
- The Gemini 3 temperature caveat
- Testing methodology for finding optimal temperature
- When higher temperature is acceptable

### Prerequisites

- [Lesson 05: Model Parameters and Settings](../../03-ai-llm-fundamentals/05-model-parameters-settings/) — General temperature concepts
- [System Prompt Guidance](./03-system-prompt-guidance.md) — Controlling model behavior

---

## How temperature affects function calling

Temperature (typically 0.0 to 2.0) controls the probability distribution the model samples from when generating the next token:

| Temperature | Behavior | Effect on Function Calling |
|-------------|----------|---------------------------|
| 0.0 | Greedy — always picks highest-probability token | Most deterministic tool selection |
| 0.3 | Very low randomness | Slight variation, still reliable |
| 0.7 | Moderate randomness | May explore alternative tools |
| 1.0 | Default randomness | More varied argument generation |
| 1.5+ | High randomness | Unreliable — may hallucinate tool names |

For function calling, randomness is the enemy. When the model generates `{"name": "get_order", "arguments": {"order_id": "ORD-12345"}}`, every token needs to be correct. A "creative" `order_id` value is a wrong value.

```python
from openai import OpenAI
client = OpenAI()

# ✅ Low temperature for deterministic function calling
response = client.responses.create(
    model="gpt-4.1",
    input="What's the status of order ORD-98765?",
    tools=[{
        "type": "function",
        "name": "get_order_status",
        "description": "Get the current status of an order by its ID",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID in the format ORD-XXXXX"
                }
            },
            "required": ["order_id"]
        }
    }],
    temperature=0.0  # Deterministic tool selection
)
```

---

## Provider-specific recommendations

### OpenAI

OpenAI recommends lower temperature for reliable function calling:

```python
from openai import OpenAI
client = OpenAI()

# OpenAI recommendation: low temperature
response = client.responses.create(
    model="gpt-4.1",
    input="Search for flights from NYC to London on March 15",
    tools=flight_tools,
    temperature=0.0  # Most deterministic
)
```

OpenAI's strict mode (auto-enabled in the Responses API) already constrains output to valid JSON matching your schema, so temperature primarily affects **which tool** is selected and **how arguments are interpreted** from the user's message — not whether the output is valid JSON.

### Anthropic

```python
import anthropic
client = anthropic.Anthropic()

# Anthropic: lower temperature for tool use
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Search for flights from NYC to London on March 15"
    }],
    tools=flight_tools_anthropic,
    temperature=0.0  # Deterministic
)
```

### Gemini — standard models

```python
from google import genai

client = genai.Client()

# Gemini recommendation: "Use a low temperature (e.g., 0)"
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Search for flights from NYC to London on March 15",
    config=genai.types.GenerateContentConfig(
        tools=[flight_tool_gemini],
        temperature=0.0  # Gemini docs: "Use a low temperature"
    )
)
```

Google's documentation states: *"Use a low temperature (e.g., 0) for more deterministic and reliable function calls."*

---

## The Gemini 3 temperature caveat

> **⚠️ Warning:** Gemini 3 models handle temperature differently. Read this carefully.

Google's documentation includes a critical exception for Gemini 3:

> *"For Gemini 3: keep temperature at the default 1.0. Changing the temperature (setting it below 1.0) may lead to unexpected behavior, such as looping or degraded performance, particularly in complex mathematical or reasoning tasks."*

This means:

```python
# ✅ Gemini 2.5 Flash — use low temperature
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_message,
    config=genai.types.GenerateContentConfig(
        tools=[tools],
        temperature=0.0  # Works well for 2.x models
    )
)

# ✅ Gemini 3 Flash Preview — keep default temperature
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=user_message,
    config=genai.types.GenerateContentConfig(
        tools=[tools],
        temperature=1.0  # Required for Gemini 3 models
    )
)

# ❌ Gemini 3 with low temperature — may cause looping
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=user_message,
    config=genai.types.GenerateContentConfig(
        tools=[tools],
        temperature=0.0  # DON'T do this with Gemini 3
    )
)
```

### A provider-aware helper

```python
def get_tool_calling_temperature(model: str) -> float:
    """Return the recommended temperature for function calling by model."""
    
    # Gemini 3 models need temperature=1.0
    if model.startswith("gemini-3"):
        return 1.0
    
    # All other models benefit from low temperature
    # OpenAI gpt-4.1/gpt-5, Anthropic Claude, Gemini 2.x
    return 0.0


# Usage
model = "gemini-3-flash-preview"
temp = get_tool_calling_temperature(model)

response = client.models.generate_content(
    model=model,
    contents=user_message,
    config=genai.types.GenerateContentConfig(
        tools=[tools],
        temperature=temp
    )
)
```

### Quick reference

| Model Family | Recommended Temperature | Reason |
|-------------|------------------------|--------|
| OpenAI gpt-4.1 / gpt-5 | 0.0 | Deterministic selection |
| Anthropic Claude | 0.0 | Deterministic selection |
| Gemini 2.5 Flash/Pro | 0.0 | Google recommends "low temperature (e.g., 0)" |
| **Gemini 3 Flash/Pro** | **1.0 (default)** | **Lower values cause looping / degraded performance** |

---

## Testing methodology

Don't just pick a temperature — test it. Run the same prompts multiple times at different temperatures and measure:

```python
import json
from collections import Counter

async def test_temperature(
    client,
    model: str,
    tools: list,
    test_prompts: list[str],
    temperatures: list[float],
    runs_per_setting: int = 20
) -> dict:
    """Test function calling accuracy across temperatures."""
    
    results = {}
    
    for temp in temperatures:
        tool_calls = []
        errors = 0
        
        for prompt in test_prompts:
            for _ in range(runs_per_setting):
                try:
                    response = client.responses.create(
                        model=model,
                        input=prompt,
                        tools=tools,
                        temperature=temp
                    )
                    
                    # Extract tool call
                    for item in response.output:
                        if item.type == "function_call":
                            tool_calls.append({
                                "name": item.name,
                                "args": json.loads(item.arguments)
                            })
                except Exception:
                    errors += 1
        
        # Analyze consistency
        tool_names = [tc["name"] for tc in tool_calls]
        name_distribution = Counter(tool_names)
        
        # Calculate consistency score (0-1)
        # Higher = more consistent tool selection
        most_common_count = name_distribution.most_common(1)[0][1]
        consistency = most_common_count / len(tool_calls) if tool_calls else 0
        
        results[temp] = {
            "total_calls": len(tool_calls),
            "errors": errors,
            "consistency": round(consistency, 3),
            "tool_distribution": dict(name_distribution)
        }
    
    return results
```

**Output:**
```
Temperature 0.0: consistency=1.000, distribution={"get_order_status": 20}
Temperature 0.3: consistency=1.000, distribution={"get_order_status": 20}
Temperature 0.7: consistency=0.950, distribution={"get_order_status": 19, "search_orders": 1}
Temperature 1.0: consistency=0.850, distribution={"get_order_status": 17, "search_orders": 2, "get_customer": 1}
```

Consistency drops as temperature rises. For most applications, 0.0 is the right choice.

---

## When higher temperature is acceptable

Low temperature isn't always the answer. Some scenarios benefit from moderate temperature:

| Scenario | Recommended Temp | Why |
|----------|-----------------|-----|
| Standard tool calling | 0.0 | Maximum reliability |
| Ambiguous queries (multiple valid tools) | 0.3 | Allows the model to consider alternatives |
| Creative content generation with tool support | 0.7 | Tool calls are secondary to creative output |
| Gemini 3 models (any scenario) | 1.0 | Required to avoid looping |
| Brainstorming with data lookups | 0.5 | Creative flow + occasional data retrieval |

```python
# Example: creative writing assistant that can look up facts
# Higher temperature for creative text, but tool calls still need to be correct

response = client.responses.create(
    model="gpt-4.1",
    input="Write a story set in Tokyo and include real landmarks",
    tools=[{
        "type": "function",
        "name": "get_landmark_info",
        "description": "Get factual information about a real-world landmark",
        "parameters": {
            "type": "object",
            "properties": {
                "landmark_name": {"type": "string"}
            },
            "required": ["landmark_name"]
        }
    }],
    temperature=0.7  # Creative output, but strict mode ensures valid tool calls
)
```

> **Note:** With OpenAI's strict mode, higher temperature only affects **which** tool is called and the narrative text — the JSON structure of tool calls is always valid.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Default to temperature 0.0 for function calling | Maximum determinism for tool selection |
| Use 1.0 for Gemini 3 models specifically | Prevents looping and degraded performance |
| Test with 20+ runs per temperature setting | Statistical significance requires volume |
| Measure consistency, not just correctness | The same input should produce the same tool call |
| Document your temperature choice per model | Future team members need to know why |
| Separate creative temp from tool-calling temp | If possible, use different temperatures for different turns |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using default temperature (1.0) for all function calling | Set to 0.0 unless you have a specific reason not to |
| Setting Gemini 3 temperature to 0.0 | Keep at 1.0 — low values cause looping |
| Assuming one temperature works for all models | Check provider documentation for each model family |
| Never testing temperature impact | Run consistency tests with your actual tools and prompts |
| Using high temperature because "it works most of the time" | "Most of the time" means occasional failures in production |

---

## Hands-on exercise

### Your task

Build a temperature testing harness for a weather tool set that measures:
1. **Tool selection consistency** — does the same prompt always call the same tool?
2. **Argument accuracy** — are extracted values (city name, date) correct?
3. **Comparison across 4 temperature settings** — 0.0, 0.3, 0.7, 1.0

### Requirements

1. Define 2 weather tools (`get_current_weather`, `get_forecast`)
2. Write 5 test prompts with known expected tool calls
3. Implement a scoring function that rates consistency and accuracy
4. Present results in a comparison table

### Expected result

A script that produces a temperature comparison table showing consistency score, accuracy, and any anomalies at each setting.

<details>
<summary>💡 Hints (click to expand)</summary>

- "What's the weather in Paris?" → `get_current_weather`
- "Will it rain in Tokyo next week?" → `get_forecast`
- Score consistency as: (most_common_tool_count / total_calls)
- Score accuracy as: (correct_tool_calls / total_calls)
- Run each prompt 10+ times per temperature for meaningful data

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from collections import Counter
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()

# Tool definitions
weather_tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": (
            "Get current weather conditions for a city. "
            "Returns temperature, humidity, and conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "type": "function",
        "name": "get_forecast",
        "description": (
            "Get weather forecast for a city for the next 7 days. "
            "Returns daily high/low temperatures and conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {
                    "type": "integer",
                    "description": "Number of forecast days (1-7)"
                }
            },
            "required": ["city"]
        }
    }
]

# Test cases: (prompt, expected_tool, expected_city)
test_cases = [
    ("What's the weather in Paris right now?", "get_current_weather", "Paris"),
    ("Will it rain in Tokyo next week?", "get_forecast", "Tokyo"),
    ("Current temperature in London?", "get_current_weather", "London"),
    ("What's the 5-day forecast for Berlin?", "get_forecast", "Berlin"),
    ("Is it sunny in Sydney today?", "get_current_weather", "Sydney"),
]

@dataclass
class TestResult:
    temperature: float
    total_calls: int
    correct_tool: int
    correct_city: int
    consistency: float

def run_temperature_test(
    temp: float,
    runs_per_case: int = 10
) -> TestResult:
    """Test function calling at a specific temperature."""
    correct_tool = 0
    correct_city = 0
    all_tool_names = []
    total = 0
    
    for prompt, expected_tool, expected_city in test_cases:
        for _ in range(runs_per_case):
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                tools=weather_tools,
                temperature=temp
            )
            
            for item in response.output:
                if item.type == "function_call":
                    total += 1
                    all_tool_names.append(item.name)
                    
                    if item.name == expected_tool:
                        correct_tool += 1
                    
                    args = json.loads(item.arguments)
                    if args.get("city", "").lower() == expected_city.lower():
                        correct_city += 1
    
    # Consistency: how often does the most common tool appear?
    name_counts = Counter(all_tool_names)
    most_common = name_counts.most_common(1)[0][1] if name_counts else 0
    consistency = most_common / total if total else 0
    
    return TestResult(
        temperature=temp,
        total_calls=total,
        correct_tool=correct_tool,
        correct_city=correct_city,
        consistency=round(consistency, 3)
    )

# Run tests at each temperature
temperatures = [0.0, 0.3, 0.7, 1.0]
results = [run_temperature_test(t) for t in temperatures]

# Print comparison table
print(f"{'Temp':<6} {'Tool Acc':<10} {'City Acc':<10} {'Consistency':<12}")
print("-" * 38)
for r in results:
    tool_acc = f"{r.correct_tool}/{r.total_calls}"
    city_acc = f"{r.correct_city}/{r.total_calls}"
    print(f"{r.temperature:<6} {tool_acc:<10} {city_acc:<10} {r.consistency:<12}")
```

</details>

### Bonus challenges

- [ ] Add Gemini 3 to the test and verify the temperature=1.0 requirement
- [ ] Test with `top_p` variations alongside temperature
- [ ] Create a visualization (bar chart) of consistency scores across temperatures

---

## Summary

✅ **Default to temperature 0.0** for function calling — determinism is more important than creativity

✅ **Gemini 3 is the exception** — keep at 1.0 to avoid looping and degraded performance

✅ **Test empirically** — run 20+ trials per temperature setting to measure consistency

✅ **Strict mode provides a safety net** — but temperature still affects which tool is selected

✅ **Document your temperature choice** per model so the team knows why

✅ Higher temperature is acceptable only when creative output matters more than tool reliability

**Next:** [Idempotency →](./06-idempotency.md)

---

[← Previous: Safe Defaults](./04-safe-defaults.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- Google Gemini Function Calling (Temperature Guidance): https://ai.google.dev/gemini-api/docs/function-calling
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
-->
