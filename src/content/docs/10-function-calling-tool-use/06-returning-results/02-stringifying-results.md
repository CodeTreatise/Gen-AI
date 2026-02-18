---
title: "Stringifying Results"
---

# Stringifying Results

## Introduction

Your functions return Python objects — dictionaries, lists, dataclass instances, datetime objects, database rows. But OpenAI and Anthropic expect **strings**. How you convert complex objects to strings determines whether the model understands your result clearly or wastes tokens on irrelevant data.

This lesson covers serialization strategies: from basic `json.dumps()` to selective field extraction, truncation for oversized results, and summary generation that gives the model the information it needs without the bloat.

### What we'll cover

- Basic JSON serialization with `json.dumps()`
- Handling non-serializable Python objects (datetime, Decimal, sets)
- Key field selection to reduce token usage
- Truncation strategies for long strings and large lists
- Summary generation for complex results
- Building a reusable result serializer

### Prerequisites

- Result format structure ([Lesson 06-01](./01-result-format-structure.md))
- Python JSON module basics

---

## Basic JSON serialization

The simplest approach — convert the result to a JSON string:

```python
import json

def get_user_profile(user_id: str) -> dict:
    """Fetch a user profile from the database."""
    return {
        "user_id": user_id,
        "name": "Alice Chen",
        "email": "alice@example.com",
        "role": "admin",
        "created_at": "2024-01-15",
        "last_login": "2025-02-06"
    }

result = get_user_profile("user_123")
output = json.dumps(result)
print(output)
```

**Output:**
```
{"user_id": "user_123", "name": "Alice Chen", "email": "alice@example.com", "role": "admin", "created_at": "2024-01-15", "last_login": "2025-02-06"}
```

This works for simple dicts with strings, numbers, booleans, and None. But real applications return more complex objects.

---

## Handling non-serializable types

Python's `json.dumps()` fails on many common types. Here's a custom encoder that handles them:

```python
import json
from datetime import datetime, date
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID
from pathlib import Path


class ResultEncoder(json.JSONEncoder):
    """JSON encoder that handles common Python types."""
    
    def default(self, obj):
        # Date and time
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Decimal (common in financial data)
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Sets → lists
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        
        # Enums → their value
        if isinstance(obj, Enum):
            return obj.value
        
        # UUID → string
        if isinstance(obj, UUID):
            return str(obj)
        
        # Path → string
        if isinstance(obj, Path):
            return str(obj)
        
        # Dataclasses → dict
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        
        # Bytes → skip (or base64 encode if needed)
        if isinstance(obj, bytes):
            return f"<binary data, {len(obj)} bytes>"
        
        return super().default(obj)


def stringify_result(result) -> str:
    """Convert any function result to a JSON string."""
    if isinstance(result, str):
        return result
    return json.dumps(result, cls=ResultEncoder, ensure_ascii=False)
```

**Usage:**
```python
from datetime import datetime
from decimal import Decimal

order_result = {
    "order_id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
    "total": Decimal("149.99"),
    "currency": "USD",
    "placed_at": datetime(2025, 2, 6, 14, 30, 0),
    "items": {"widget", "gadget", "doohickey"},
    "status": "confirmed"
}

output = stringify_result(order_result)
print(output)
```

**Output:**
```json
{"order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "total": 149.99, "currency": "USD", "placed_at": "2025-02-06T14:30:00", "items": ["doohickey", "gadget", "widget"], "status": "confirmed"}
```

> **Tip:** Always test your encoder with actual function return values. Database query results, API responses, and ORM models often contain unexpected types.

---

## Key field selection

Functions often return more data than the model needs. A database query might return 20 columns, but the model only needs 3 to answer the user's question. Selecting key fields reduces token usage and improves response quality.

### Static field selection

Define which fields to include for each function:

```python
# Configuration: which fields to keep per function
RESULT_FIELDS = {
    "search_products": ["name", "price", "in_stock", "rating"],
    "get_user": ["name", "email", "role"],
    "list_orders": ["order_id", "total", "status", "date"],
}


def select_fields(result: dict | list, function_name: str) -> dict | list:
    """Select only the fields the model needs from a result."""
    fields = RESULT_FIELDS.get(function_name)
    
    if not fields:
        return result  # No field config — return everything
    
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if k in fields}
    
    if isinstance(result, list):
        return [
            {k: v for k, v in item.items() if k in fields}
            for item in result
            if isinstance(item, dict)
        ]
    
    return result
```

**Usage:**
```python
# Full database result
db_product = {
    "id": 42,
    "name": "Wireless Mouse",
    "price": 29.99,
    "in_stock": True,
    "rating": 4.5,
    "sku": "WM-2025-BLK",
    "warehouse_location": "A3-B7",
    "supplier_id": "SUP-001",
    "internal_cost": 12.50,
    "created_at": "2024-03-15",
    "updated_at": "2025-01-20"
}

# Filtered result — only what the model needs
filtered = select_fields(db_product, "search_products")
print(json.dumps(filtered, indent=2))
```

**Output:**
```json
{
  "name": "Wireless Mouse",
  "price": 29.99,
  "in_stock": true,
  "rating": 4.5
}
```

> **🤖 AI Context:** Filtering out internal fields like `warehouse_location`, `supplier_id`, and `internal_cost` isn't just about token efficiency — it prevents the model from leaking sensitive business data in its responses.

### Dynamic field selection

For functions where the relevant fields depend on the user's question, you can pass a hint:

```python
def select_fields_dynamic(
    result: dict,
    relevant_fields: list[str] | None = None
) -> dict:
    """Select fields based on what's relevant to the current query."""
    if not relevant_fields:
        return result
    
    return {k: v for k, v in result.items() if k in relevant_fields}


# For a pricing question, only price-related fields matter
pricing_view = select_fields_dynamic(
    db_product,
    relevant_fields=["name", "price", "in_stock"]
)
print(json.dumps(pricing_view))
```

**Output:**
```json
{"name": "Wireless Mouse", "price": 29.99, "in_stock": true}
```

---

## Truncation strategies

Sometimes results are too long — a search returns 1,000 records, a log query returns megabytes of text, or a function returns a massive nested structure. Truncation keeps results within reasonable token limits.

### String truncation

```python
def truncate_string(
    text: str,
    max_length: int = 2000,
    suffix: str = "... [truncated, {remaining} more characters]"
) -> str:
    """Truncate a string to a maximum length with an informative suffix."""
    if len(text) <= max_length:
        return text
    
    remaining = len(text) - max_length
    truncation_notice = suffix.format(remaining=remaining)
    # Leave room for the suffix
    cut_point = max_length - len(truncation_notice)
    return text[:cut_point] + truncation_notice
```

**Usage:**
```python
long_text = "A" * 5000
truncated = truncate_string(long_text, max_length=100)
print(truncated)
print(f"Length: {len(truncated)}")
```

**Output:**
```
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA... [truncated, 4900 more characters]
100
```

### List truncation

```python
def truncate_list(
    items: list,
    max_items: int = 10,
    include_count: bool = True
) -> list | dict:
    """Truncate a list and indicate how many items were omitted."""
    if len(items) <= max_items:
        return items
    
    truncated = items[:max_items]
    
    if include_count:
        return {
            "items": truncated,
            "total_count": len(items),
            "showing": max_items,
            "omitted": len(items) - max_items
        }
    
    return truncated
```

**Usage:**
```python
# 50 search results — show only the top 5
all_results = [{"name": f"Product {i}", "price": i * 10} for i in range(1, 51)]
truncated = truncate_list(all_results, max_items=5)
print(json.dumps(truncated, indent=2))
```

**Output:**
```json
{
  "items": [
    {"name": "Product 1", "price": 10},
    {"name": "Product 2", "price": 20},
    {"name": "Product 3", "price": 30},
    {"name": "Product 4", "price": 40},
    {"name": "Product 5", "price": 50}
  ],
  "total_count": 50,
  "showing": 5,
  "omitted": 45
}
```

### Nested structure depth limiting

```python
def limit_depth(obj, max_depth: int = 3, current_depth: int = 0):
    """Limit the nesting depth of a result object."""
    if current_depth >= max_depth:
        if isinstance(obj, dict):
            return f"{{... {len(obj)} keys}}"
        elif isinstance(obj, list):
            return f"[... {len(obj)} items]"
        return obj
    
    if isinstance(obj, dict):
        return {
            k: limit_depth(v, max_depth, current_depth + 1)
            for k, v in obj.items()
        }
    
    if isinstance(obj, list):
        return [
            limit_depth(item, max_depth, current_depth + 1)
            for item in obj
        ]
    
    return obj
```

**Usage:**
```python
deep_result = {
    "user": {
        "profile": {
            "settings": {
                "notifications": {
                    "email": True,
                    "sms": False
                }
            }
        }
    }
}

limited = limit_depth(deep_result, max_depth=2)
print(json.dumps(limited, indent=2))
```

**Output:**
```json
{
  "user": {
    "profile": "{... 1 keys}"
  }
}
```

---

## Summary generation

For very large results, generating a summary can be more effective than truncation. The summary gives the model the key facts without overwhelming it.

```python
def summarize_search_results(results: list[dict]) -> dict:
    """Generate a summary of search results instead of sending all data."""
    if not results:
        return {"summary": "No results found", "count": 0}
    
    # Extract key stats
    prices = [r.get("price", 0) for r in results if "price" in r]
    
    summary = {
        "total_results": len(results),
        "top_results": results[:3],  # First 3 results in full
        "price_range": {
            "min": min(prices) if prices else None,
            "max": max(prices) if prices else None,
            "average": round(sum(prices) / len(prices), 2) if prices else None
        },
        "available_count": sum(1 for r in results if r.get("in_stock", False)),
    }
    
    return summary
```

**Usage:**
```python
products = [
    {"name": f"Product {i}", "price": 10 + i * 5, "in_stock": i % 3 != 0}
    for i in range(1, 101)
]

summary = summarize_search_results(products)
print(json.dumps(summary, indent=2))
```

**Output:**
```json
{
  "total_results": 100,
  "top_results": [
    {"name": "Product 1", "price": 15, "in_stock": true},
    {"name": "Product 2", "price": 20, "in_stock": true},
    {"name": "Product 3", "price": 25, "in_stock": false}
  ],
  "price_range": {
    "min": 15,
    "max": 510,
    "average": 262.5
  },
  "available_count": 67
}
```

> **🤖 AI Context:** Summaries work especially well when the model needs to answer aggregate questions like "Are there affordable options?" or "How many are in stock?" The model doesn't need to see all 100 products to answer these questions — a summary with key statistics is more useful and far cheaper in tokens.

---

## Building a reusable result serializer

Here's a complete serializer that combines all the strategies:

```python
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SerializerConfig:
    """Configuration for result serialization."""
    max_string_length: int = 4000      # Max chars for string values
    max_list_items: int = 20           # Max items in lists
    max_depth: int = 4                 # Max nesting depth
    selected_fields: list[str] | None = None  # Fields to include (None = all)
    summarizer: Callable | None = None # Custom summary function


class ResultSerializer:
    """Serialize function results with configurable limits."""
    
    def __init__(self, config: SerializerConfig | None = None):
        self.config = config or SerializerConfig()
        self.encoder = ResultEncoder()
    
    def serialize(self, result: Any) -> str:
        """Serialize a result to a string with all configured limits applied."""
        if result is None:
            return '{"status": "success", "result": null}'
        
        if isinstance(result, str):
            return truncate_string(result, self.config.max_string_length)
        
        # Apply field selection
        processed = result
        if self.config.selected_fields and isinstance(processed, (dict, list)):
            processed = self._apply_field_selection(processed)
        
        # Apply custom summarizer if provided
        if self.config.summarizer and isinstance(processed, list):
            processed = self.config.summarizer(processed)
        
        # Apply list truncation
        if isinstance(processed, list):
            processed = truncate_list(processed, self.config.max_list_items)
        
        # Apply depth limiting
        processed = limit_depth(processed, self.config.max_depth)
        
        # Serialize to string
        output = json.dumps(processed, cls=ResultEncoder, ensure_ascii=False)
        
        # Final string truncation as safety net
        return truncate_string(output, self.config.max_string_length)
    
    def _apply_field_selection(self, data):
        """Filter dict fields based on config."""
        fields = self.config.selected_fields
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in fields}
        if isinstance(data, list):
            return [
                {k: v for k, v in item.items() if k in fields}
                for item in data if isinstance(item, dict)
            ]
        return data
```

**Usage:**
```python
# Configure per-function serialization
search_serializer = ResultSerializer(SerializerConfig(
    max_list_items=5,
    selected_fields=["name", "price", "in_stock"],
    max_string_length=2000
))

user_serializer = ResultSerializer(SerializerConfig(
    selected_fields=["name", "email", "role"],
    max_depth=2
))

# Use the appropriate serializer
products = [
    {"name": f"Item {i}", "price": i * 10, "in_stock": True, "sku": f"SKU-{i}"}
    for i in range(50)
]

output = search_serializer.serialize(products)
print(output)
```

**Output:**
```json
{"items": [{"name": "Item 0", "price": 0, "in_stock": true}, {"name": "Item 1", "price": 10, "in_stock": true}, {"name": "Item 2", "price": 20, "in_stock": true}, {"name": "Item 3", "price": 30, "in_stock": true}, {"name": "Item 4", "price": 40, "in_stock": true}], "total_count": 50, "showing": 5, "omitted": 45}
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Use a custom JSON encoder for non-standard types | Prevents `TypeError` crashes at serialization time |
| Select only the fields the model needs | Reduces token usage and prevents leaking internal data |
| Set reasonable truncation limits | Prevents context window overflow on large results |
| Include metadata about truncation | Tells the model "there's more data available" |
| Use summaries for aggregate data | More useful than 100 raw records for questions like "how many?" |
| Test with real function outputs | Synthetic data often misses edge cases like nested None values |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| `json.dumps()` crashes on datetime objects | Use a custom encoder: `cls=ResultEncoder` |
| Sending entire database rows (30+ columns) | Select only the 3-5 fields the model needs |
| Truncating without telling the model | Always include `[truncated, N more items]` metadata |
| Returning raw bytes or binary data | Convert to a description: `"<binary data, 1.2MB>"` |
| Using `str()` instead of `json.dumps()` | `str()` produces Python repr (single quotes, etc.) — not valid JSON |
| No upper limit on result size | Always have a `max_string_length` safety net |

---

## Hands-on exercise

### Your task

Build a `ResultProcessor` class that takes raw function results and prepares them for returning to the model. It should handle serialization, field selection, and truncation.

### Requirements

1. Accept a result (any Python type) and a function name
2. Look up field selection config for the function name
3. Apply field selection if configured
4. Serialize to JSON string using a custom encoder (handle datetime, Decimal, set)
5. Truncate the result string if it exceeds a configurable maximum
6. Return the final string ready for the provider's result format

### Expected result

```python
processor = ResultProcessor(max_length=500)
processor.configure_fields("search_products", ["name", "price", "rating"])

result = processor.process(
    [{"name": "Widget", "price": Decimal("9.99"), "rating": 4.5, "sku": "W-001"}],
    function_name="search_products"
)
# → '[{"name": "Widget", "price": 9.99, "rating": 4.5}]'
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Store field configs in a dictionary on the class
- Use `isinstance` checks to handle different result types
- Apply field selection before serialization
- Apply truncation after serialization (it's a string operation)

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from datetime import datetime, date
from decimal import Decimal


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        return super().default(obj)


class ResultProcessor:
    def __init__(self, max_length: int = 4000):
        self.max_length = max_length
        self._field_configs: dict[str, list[str]] = {}
    
    def configure_fields(self, function_name: str, fields: list[str]):
        """Set which fields to include for a given function."""
        self._field_configs[function_name] = fields
    
    def process(self, result, function_name: str | None = None) -> str:
        """Process a raw result into a model-ready string."""
        if result is None:
            return '{"status": "success"}'
        
        if isinstance(result, str):
            return self._truncate(result)
        
        # Apply field selection
        processed = result
        if function_name and function_name in self._field_configs:
            fields = self._field_configs[function_name]
            processed = self._select_fields(processed, fields)
        
        # Serialize
        output = json.dumps(processed, cls=CustomEncoder, ensure_ascii=False)
        
        # Truncate
        return self._truncate(output)
    
    def _select_fields(self, data, fields: list[str]):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in fields}
        if isinstance(data, list):
            return [
                {k: v for k, v in item.items() if k in fields}
                for item in data if isinstance(item, dict)
            ]
        return data
    
    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_length:
            return text
        remaining = len(text) - self.max_length
        suffix = f"... [truncated, {remaining} more chars]"
        return text[:self.max_length - len(suffix)] + suffix


# Test it
processor = ResultProcessor(max_length=500)
processor.configure_fields("search_products", ["name", "price", "rating"])

products = [
    {
        "name": "Wireless Mouse",
        "price": Decimal("29.99"),
        "rating": 4.5,
        "sku": "WM-001",
        "warehouse": "A3"
    },
    {
        "name": "USB-C Hub",
        "price": Decimal("49.99"),
        "rating": 4.2,
        "sku": "UH-002",
        "warehouse": "B1"
    }
]

output = processor.process(products, "search_products")
print(output)
```

**Output:**
```json
[{"name": "Wireless Mouse", "price": 29.99, "rating": 4.5}, {"name": "USB-C Hub", "price": 49.99, "rating": 4.2}]
```

</details>

### Bonus challenges

- [ ] Add a `configure_summarizer(function_name, summarizer_fn)` method that applies a custom summary function before serialization
- [ ] Handle nested field selection (e.g., `"user.name"` to select nested keys)
- [ ] Add token estimation (rough: `len(text) / 4` for English) and warn if the result exceeds a token budget

---

## Summary

✅ Use `json.dumps()` with a custom encoder to handle datetime, Decimal, sets, and other non-serializable types

✅ Select only the fields the model needs — it reduces tokens and prevents data leakage

✅ Truncate strings and lists with informative metadata so the model knows data was omitted

✅ Generate summaries for large datasets — statistics are more useful than raw records for many queries

✅ Build a reusable serializer with configurable limits per function

**Next:** [Multimodal Results →](./03-multimodal-results.md) — Returning images and documents in Gemini function responses

---

[← Previous: Result Format Structure](./01-result-format-structure.md) | [Back to Lesson Overview](./00-returning-results.md)

<!-- 
Sources Consulted:
- Python json module: https://docs.python.org/3/library/json.html
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- OpenAI Token Counting: https://platform.openai.com/docs/guides/tokens
-->
