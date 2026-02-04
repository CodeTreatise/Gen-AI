---
title: "JSON Mode in API Calls"
---

# JSON Mode in API Calls

## Introduction

JSON mode is an API feature that guarantees the model's output is valid JSON. Unlike prompt-based approaches that "encourage" JSON output, JSON mode provides a parsing guarantee—the response will always be syntactically valid JSON that can be parsed without errors.

> **🤖 AI Context:** JSON mode is available across most major providers but works differently in each. Understanding these differences helps you build portable applications.

### What We'll Cover

- Enabling JSON mode in OpenAI, Anthropic, and Google APIs
- Provider-specific parameters and behavior
- What JSON mode guarantees (and doesn't)
- Common requirements and gotchas

### Prerequisites

- [Output Formatting & Structured Prompting](../05-output-formatting-structured-prompting/)

---

## What JSON Mode Guarantees

| Guarantee | JSON Mode | Standard Mode |
|-----------|-----------|---------------|
| Valid JSON syntax | ✅ Yes | ❌ No |
| Parseable output | ✅ Yes | ⚠️ Usually |
| Schema compliance | ❌ No | ❌ No |
| Specific fields | ❌ No | ❌ No |
| Field types | ❌ No | ❌ No |

JSON mode ensures the output is valid JSON—not that it matches your expected structure.

---

## OpenAI JSON Mode

### Enabling JSON Mode

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Extract product info. Return JSON with name, price, and category."
        },
        {
            "role": "user",
            "content": "Apple AirPods Pro - $249, wireless earbuds"
        }
    ],
    response_format={"type": "json_object"}
)

# Guaranteed to be valid JSON
import json
data = json.loads(response.choices[0].message.content)
print(data)
```

**Output:**
```json
{
  "name": "Apple AirPods Pro",
  "price": 249,
  "category": "wireless earbuds"
}
```

### Critical Requirement: Mention JSON

> **Warning:** You MUST mention "JSON" somewhere in your messages. Without it, the API may throw an error or produce unexpected behavior.

```python
# ❌ WRONG - No JSON mention
messages=[
    {"role": "system", "content": "Extract product info."},
    {"role": "user", "content": "..."}
]

# ✅ CORRECT - JSON mentioned
messages=[
    {"role": "system", "content": "Extract product info. Return valid JSON."},
    {"role": "user", "content": "..."}
]
```

### Responses API Syntax

```python
# Using the newer Responses API
response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "developer", "content": "Return JSON with extracted entities."},
        {"role": "user", "content": "John works at TechCorp in Seattle."}
    ],
    text={"format": {"type": "json_object"}}
)
```

---

## Anthropic (Claude) JSON Mode

Anthropic doesn't have a dedicated JSON mode parameter. Instead, use:

### 1. Tool Use for Structured Output

```python
import anthropic

client = anthropic.Anthropic()

# Define a "tool" that describes your schema
tools = [{
    "name": "extract_product",
    "description": "Extract product information",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "category": {"type": "string"}
        },
        "required": ["name", "price", "category"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_product"},
    messages=[
        {"role": "user", "content": "Apple AirPods Pro - $249, wireless earbuds"}
    ]
)

# Access structured output from tool use
tool_use = response.content[0]
data = tool_use.input  # Already parsed dict
```

### 2. Prompt-Based JSON

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": """Extract product info from: Apple AirPods Pro - $249, wireless earbuds

Return ONLY valid JSON with this structure:
{"name": "...", "price": ..., "category": "..."}"""
        }
    ]
)

# Parse the response
import json
data = json.loads(response.content[0].text)
```

---

## Google Gemini JSON Mode

### Using response_mime_type

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "Extract: Apple AirPods Pro - $249. Return JSON with name, price, category.",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json"
    )
)

import json
data = json.loads(response.text)
```

### With Schema (Gemini 1.5+)

```python
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "category": {"type": "string"}
    },
    "required": ["name", "price", "category"]
}

response = model.generate_content(
    "Extract: Apple AirPods Pro - $249, wireless earbuds",
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        response_schema=schema
    )
)
```

---

## Provider Comparison

| Feature | OpenAI | Anthropic | Google Gemini |
|---------|--------|-----------|---------------|
| **JSON mode** | `response_format` | Tool use | `response_mime_type` |
| **Schema support** | Structured Outputs | Tool schemas | `response_schema` |
| **Strict mode** | Yes | N/A | Yes (Gemini 1.5+) |
| **Requires JSON mention** | Yes | No | No |

---

## Edge Cases to Handle

### 1. Maximum Token Limit

JSON may be truncated if it exceeds `max_tokens`:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_object"},
    max_tokens=100  # May truncate JSON!
)

# Check for truncation
if response.choices[0].finish_reason == "length":
    print("Warning: Response may be truncated")
```

### 2. Empty or Minimal JSON

The model might return minimal valid JSON:

```python
# Possible response if input is unclear
{"error": "Unable to extract information"}
# or
{}
```

Handle these cases:

```python
data = json.loads(response.choices[0].message.content)
if not data or "error" in data:
    # Handle gracefully
    pass
```

### 3. Refusals

When the model refuses for safety reasons:

```python
# Model might return:
{"refusal": "I cannot process this request"}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always mention "JSON" | Required for OpenAI, helps other providers |
| Set sufficient max_tokens | Prevent truncated JSON |
| Check finish_reason | Detect truncation |
| Validate after parsing | JSON mode doesn't guarantee schema |
| Handle edge cases | Empty objects, refusals |
| Include schema in prompt | Guide structure even without Structured Outputs |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No JSON mention in prompt | Always include "JSON" in messages |
| max_tokens too low | Calculate expected output size |
| Assuming schema compliance | Validate fields after parsing |
| Ignoring finish_reason | Check for "length" truncation |
| Not handling refusals | Check for refusal keys |

---

## Hands-on Exercise

### Your Task

Enable JSON mode for a sentiment analysis API across two providers.

### Requirements

1. Use OpenAI with response_format
2. Use Anthropic with tool use
3. Extract: sentiment, confidence, keywords
4. Handle potential edge cases

<details>
<summary>💡 Hints (click to expand)</summary>

- Remember OpenAI requires "JSON" in the prompt
- Anthropic tool_choice forces the tool to be called
- What if the text has no clear sentiment?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**OpenAI:**
```python
from openai import OpenAI
import json

client = OpenAI()

def analyze_sentiment_openai(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyze sentiment. Return JSON:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0-1.0,
  "keywords": ["word1", "word2"]
}"""
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        max_tokens=200
    )
    
    # Check for truncation
    if response.choices[0].finish_reason == "length":
        raise ValueError("Response truncated")
    
    data = json.loads(response.choices[0].message.content)
    
    # Validate required fields
    required = ["sentiment", "confidence", "keywords"]
    if not all(k in data for k in required):
        raise ValueError(f"Missing fields: {set(required) - set(data.keys())}")
    
    return data
```

**Anthropic:**
```python
import anthropic

client = anthropic.Anthropic()

def analyze_sentiment_anthropic(text):
    tools = [{
        "name": "sentiment_analysis",
        "description": "Analyze text sentiment",
        "input_schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["sentiment", "confidence", "keywords"]
        }
    }]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        tools=tools,
        tool_choice={"type": "tool", "name": "sentiment_analysis"},
        messages=[
            {"role": "user", "content": f"Analyze sentiment: {text}"}
        ]
    )
    
    # Extract from tool use response
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    
    raise ValueError("No tool use in response")

# Usage
text = "This product is amazing! Best purchase ever."
print(analyze_sentiment_openai(text))
print(analyze_sentiment_anthropic(text))
```

</details>

### Bonus Challenge

- [ ] Add a third provider (Google Gemini)
- [ ] Create a unified interface that works with any provider

---

## Summary

✅ **JSON mode guarantees** valid, parseable JSON output

✅ **Provider differences** require different configuration

✅ **Mention "JSON"** in prompts for OpenAI compatibility

✅ **Handle edge cases** like truncation and refusals

✅ **JSON mode ≠ schema compliance** — validate structure separately

**Next:** [Structured Outputs with Schemas](./02-structured-outputs-schemas.md)

---

## Further Reading

- [OpenAI JSON Mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Gemini JSON Mode](https://ai.google.dev/gemini-api/docs/json-mode)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
-->
