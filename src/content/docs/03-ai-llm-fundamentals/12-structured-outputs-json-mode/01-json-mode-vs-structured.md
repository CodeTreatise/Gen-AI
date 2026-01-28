---
title: "JSON Mode vs Structured Outputs"
---

# JSON Mode vs Structured Outputs

## Introduction

LLMs can output responses in JSON format through two approaches: JSON mode and structured outputs. Understanding the differences helps you choose the right tool for your use case.

### What We'll Cover

- JSON mode: Guaranteed valid JSON
- Structured outputs: Schema-enforced responses
- When to use each approach
- Provider support comparison

---

## JSON Mode

### What It Does

JSON mode ensures the model's output is valid JSON, but doesn't guarantee it matches any particular schema.

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract product info. Respond in JSON."},
        {"role": "user", "content": "iPhone 15 Pro, $999, Space Black"}
    ],
    response_format={"type": "json_object"}
)

# Guaranteed valid JSON, but schema varies
result = json.loads(response.choices[0].message.content)
```

**Output might be:**
```json
{
  "product": "iPhone 15 Pro",
  "price": 999,
  "color": "Space Black"
}
```

Or:
```json
{
  "name": "iPhone 15 Pro",
  "cost": "$999",
  "variant": "Space Black"
}
```

### Key Characteristics

| Aspect | JSON Mode |
|--------|-----------|
| Valid JSON | ✅ Guaranteed |
| Schema match | ❌ Not guaranteed |
| Field names | May vary |
| Type consistency | May vary |
| Missing fields | Possible |

---

## Structured Outputs

### What It Does

Structured outputs guarantee the response matches a specific JSON schema you provide.

```python
from openai import OpenAI
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    color: str

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract product info."},
        {"role": "user", "content": "iPhone 15 Pro, $999, Space Black"}
    ],
    response_format=Product
)

product = response.choices[0].message.parsed
print(product.name)   # "iPhone 15 Pro"
print(product.price)  # 999.0
print(product.color)  # "Space Black"
```

**Output is always:**
```json
{
  "name": "iPhone 15 Pro",
  "price": 999.0,
  "color": "Space Black"
}
```

### Key Characteristics

| Aspect | Structured Outputs |
|--------|-------------------|
| Valid JSON | ✅ Guaranteed |
| Schema match | ✅ Guaranteed |
| Field names | Exactly as specified |
| Type consistency | Exactly as specified |
| Missing fields | Not possible (all required fields included) |

---

## Comparison

```
┌────────────────────────────────────────────────────────────┐
│                  JSON MODE vs STRUCTURED                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  JSON MODE:                                                │
│  ┌──────────────┐     ┌──────────────────────────┐        │
│  │ Model Output │ ──→ │ Valid JSON (any schema)  │        │
│  └──────────────┘     └──────────────────────────┘        │
│                                                            │
│  STRUCTURED OUTPUTS:                                       │
│  ┌──────────────┐     ┌──────────────────────────┐        │
│  │ Model Output │ ──→ │ Valid JSON + Your Schema │        │
│  └──────────────┘     └──────────────────────────┘        │
│                            ↑                               │
│                       Schema provided                      │
│                       at request time                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Scenario | Use JSON Mode | Use Structured Outputs |
|----------|---------------|----------------------|
| Flexible extraction | ✅ | |
| Strict schema required | | ✅ |
| Variable response structure | ✅ | |
| API integration | | ✅ |
| Database insertion | | ✅ |
| Exploratory analysis | ✅ | |
| Production pipelines | | ✅ |

---

## Provider Support

### JSON Mode Support

| Provider | JSON Mode | Notes |
|----------|-----------|-------|
| OpenAI | ✅ | All GPT-4 and GPT-3.5 models |
| Anthropic | ✅ | Via prefilling technique |
| Google | ✅ | Gemini models |
| Mistral | ✅ | All models |
| Groq | ✅ | All models |

### Structured Outputs Support

| Provider | Structured Outputs | Notes |
|----------|-------------------|-------|
| OpenAI | ✅ | GPT-4o and newer |
| Anthropic | ⚠️ | Tool use workaround |
| Google | ✅ | Gemini 1.5+ |
| Mistral | ⚠️ | Limited support |
| Groq | ⚠️ | Partial support |

---

## Examples

### JSON Mode Example

```python
# Good for: "I just need some valid JSON"
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user", 
            "content": "List 3 programming languages with their use cases"
        }
    ],
    response_format={"type": "json_object"}
)

# Output structure is not guaranteed
# Could be: {"languages": [...]} 
# Or: {"items": [...]}
# Or: {"programming_languages": [...]}
```

### Structured Output Example

```python
from pydantic import BaseModel

class Language(BaseModel):
    name: str
    use_cases: list[str]
    year_created: int

class LanguageList(BaseModel):
    languages: list[Language]

# Good for: "I need exactly this structure"
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "List 3 programming languages with their use cases"
        }
    ],
    response_format=LanguageList
)

# Output structure is guaranteed
languages = response.choices[0].message.parsed
for lang in languages.languages:
    print(f"{lang.name} ({lang.year_created}): {lang.use_cases}")
```

---

## Best Practices

### When Using JSON Mode

```python
# Always instruct the model about JSON in the prompt
messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant. Always respond in valid JSON format."
    },
    {"role": "user", "content": user_query}
]

# Add error handling for parsing
try:
    result = json.loads(response.choices[0].message.content)
except json.JSONDecodeError:
    # Handle malformed JSON (rare but possible)
    pass
```

### When Using Structured Outputs

```python
# Define clear, specific schemas
class ExtractedEntity(BaseModel):
    """Entity extracted from text"""
    name: str
    entity_type: Literal["person", "organization", "location"]
    confidence: float  # 0.0 to 1.0

# Use Pydantic validators for additional constraints
from pydantic import Field

class ValidatedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
```

---

## Summary

✅ **JSON mode**: Guarantees valid JSON, flexible structure

✅ **Structured outputs**: Guarantees both valid JSON and schema compliance

✅ **Choose JSON mode** for exploratory or flexible use cases

✅ **Choose structured outputs** for production, APIs, databases

**Next:** [Response Format Parameter](./02-response-format.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Overview](./00-structured-outputs-json-mode.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [Response Format](./02-response-format.md) |
