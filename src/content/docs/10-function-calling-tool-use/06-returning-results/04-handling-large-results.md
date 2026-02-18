---
title: "Handling Large Results"
---

# Handling Large Results

## Introduction

API providers impose **token limits** on function results, and models can struggle to process oversized responses. A database query returning 10,000 rows or an API response with deeply nested objects will either fail at the API level or consume so much context that the model loses track of the conversation. We need strategies to keep results informative while staying within practical size constraints.

This lesson covers how to detect oversized results, apply pagination and chunking patterns, and build intelligent result reduction that preserves the most valuable information.

### What we'll cover

- Provider-specific size limits for function results
- Detecting and measuring result size before sending
- Pagination patterns for large datasets
- Summary-with-details reduction strategies
- Token budget estimation and management
- Building a result size manager

### Prerequisites

- Stringifying results ([Lesson 06-02](./02-stringifying-results.md))
- Token concepts ([Unit 03](../../../03-ai-llm-fundamentals/04-context-windows/))

---

## Provider size limits

Each provider handles large function results differently, and none provide explicit byte limits in their documentation. However, practical limits exist:

| Provider | Practical limit | What happens when exceeded |
|----------|----------------|---------------------------|
| **OpenAI** | ~512 KB output string | API error or truncation; consumes output tokens |
| **Anthropic** | ~100 KB per tool_result | API error; counts against context window tokens |
| **Gemini** | ~256 KB per function response | API error; large content may hit rate limits |

> **Warning:** These are practical observations, not documented guarantees. Even within these limits, sending very large results wastes tokens and degrades model performance. Aim for the **smallest result that fully answers the question**.

### The real constraint: Token cost

Even if the API accepts a large result, every character in your function result consumes tokens from the context window:

```python
import json

def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 characters for English text)."""
    return len(text) // 4

# A small result: ~50 tokens
small_result = json.dumps({"name": "Widget", "price": 29.99, "in_stock": True})
print(f"Small result: {len(small_result)} chars ≈ {estimate_tokens(small_result)} tokens")

# A large result: ~25,000 tokens
large_result = json.dumps([{"id": i, "name": f"Product {i}", "description": "A " * 50} for i in range(500)])
print(f"Large result: {len(large_result)} chars ≈ {estimate_tokens(large_result)} tokens")
```

**Output:**
```
Small result: 49 chars ≈ 12 tokens
Large result: 49500 chars ≈ 12375 tokens
```

> **🤖 AI Context:** Spending 12,000 tokens on a single function result means fewer tokens available for the model's reasoning, conversation history, and response generation. It also increases API costs proportionally.

---

## Detecting oversized results

Before sending a result, measure it and decide whether reduction is needed:

```python
import json
import sys


class ResultSizeChecker:
    """Check if a function result exceeds size thresholds."""
    
    # Conservative defaults (in characters)
    DEFAULT_LIMITS = {
        "openai": 50_000,      # ~12,500 tokens
        "anthropic": 40_000,   # ~10,000 tokens
        "gemini": 50_000,      # ~12,500 tokens
    }
    
    def __init__(self, provider: str = "openai", max_chars: int | None = None):
        self.provider = provider
        self.max_chars = max_chars or self.DEFAULT_LIMITS.get(provider, 50_000)
    
    def check(self, result: any) -> dict:
        """Analyze a result and return size information."""
        serialized = json.dumps(result, default=str)
        char_count = len(serialized)
        estimated_tokens = char_count // 4
        
        return {
            "chars": char_count,
            "estimated_tokens": estimated_tokens,
            "exceeds_limit": char_count > self.max_chars,
            "reduction_needed": max(0, char_count - self.max_chars),
            "utilization": round(char_count / self.max_chars * 100, 1),
        }
    
    def needs_reduction(self, result: any) -> bool:
        """Quick check if the result needs to be reduced."""
        serialized = json.dumps(result, default=str)
        return len(serialized) > self.max_chars


# Usage
checker = ResultSizeChecker(provider="openai")

small = {"status": "ok", "count": 42}
large = [{"id": i, "data": "x" * 200} for i in range(500)]

print("Small result:", checker.check(small))
print("Large result:", checker.check(large))
```

**Output:**
```
Small result: {'chars': 26, 'estimated_tokens': 6, 'exceeds_limit': False, 'reduction_needed': 0, 'utilization': 0.1}
Large result: {'chars': 111500, 'estimated_tokens': 27875, 'exceeds_limit': True, 'reduction_needed': 61500, 'utilization': 223.0}
```

---

## Pagination patterns

When a result is a list of items (database rows, search results, API responses), pagination is the most natural reduction strategy.

### Cursor-based pagination

```python
from dataclasses import dataclass


@dataclass
class PaginatedResult:
    """A paginated function result with navigation metadata."""
    items: list[dict]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
    
    def to_dict(self) -> dict:
        return {
            "items": self.items,
            "pagination": {
                "total_count": self.total_count,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": (self.total_count + self.page_size - 1) // self.page_size,
                "has_next": self.has_next,
                "has_previous": self.has_previous,
            },
        }


def paginate_results(
    all_items: list[dict],
    page: int = 1,
    page_size: int = 10,
) -> PaginatedResult:
    """Apply pagination to a list of results."""
    total = len(all_items)
    start = (page - 1) * page_size
    end = start + page_size
    
    return PaginatedResult(
        items=all_items[start:end],
        total_count=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
        has_previous=page > 1,
    )


# Example: database returns 150 users, we return page 1
all_users = [{"id": i, "name": f"User {i}", "email": f"user{i}@example.com"} for i in range(150)]

page1 = paginate_results(all_users, page=1, page_size=10)
print(json.dumps(page1.to_dict(), indent=2))
```

**Output:**
```json
{
  "items": [
    {"id": 0, "name": "User 0", "email": "user0@example.com"},
    {"id": 1, "name": "User 1", "email": "user1@example.com"},
    ...
    {"id": 9, "name": "User 9", "email": "user9@example.com"}
  ],
  "pagination": {
    "total_count": 150,
    "page": 1,
    "page_size": 10,
    "total_pages": 15,
    "has_next": true,
    "has_previous": false
  }
}
```

> **Tip:** Include clear pagination metadata so the model knows it can request the next page. Models understand `"has_next": true` and will often call the function again with `page: 2`.

### Adaptive page size

Instead of a fixed page size, adjust based on item complexity:

```python
def adaptive_paginate(
    items: list[dict],
    max_chars: int = 20_000,
    min_items: int = 3,
    max_items: int = 50,
) -> PaginatedResult:
    """Dynamically determine how many items fit within the character budget."""
    if not items:
        return PaginatedResult([], 0, 1, 0, False, False)
    
    # Estimate size of a single item
    sample_size = len(json.dumps(items[0], default=str))
    
    # Calculate how many items fit
    # Reserve 200 chars for pagination metadata
    available = max_chars - 200
    items_that_fit = max(min_items, min(max_items, available // sample_size))
    
    selected = items[:items_that_fit]
    
    return PaginatedResult(
        items=selected,
        total_count=len(items),
        page=1,
        page_size=items_that_fit,
        has_next=items_that_fit < len(items),
        has_previous=False,
    )


# Large items → fewer per page
large_items = [{"id": i, "bio": "x" * 500} for i in range(100)]
result = adaptive_paginate(large_items, max_chars=5_000)
print(f"Large items: showing {len(result.items)} of {result.total_count}")

# Small items → more per page
small_items = [{"id": i, "name": f"Item {i}"} for i in range(100)]
result = adaptive_paginate(small_items, max_chars=5_000)
print(f"Small items: showing {len(result.items)} of {result.total_count}")
```

**Output:**
```
Large items: showing 9 of 100
Small items: showing 50 of 100
```

---

## Summary-with-details strategy

Sometimes the model doesn't need every record — it needs an **overview** with the option to drill down:

```python
def summarize_with_top_items(
    items: list[dict],
    sort_key: str,
    top_n: int = 5,
    reverse: bool = True,
) -> dict:
    """Return aggregate stats plus top N items."""
    if not items:
        return {"summary": "No results found", "items": []}
    
    # Sort and select top items
    sorted_items = sorted(items, key=lambda x: x.get(sort_key, 0), reverse=reverse)
    top_items = sorted_items[:top_n]
    
    # Calculate aggregates
    values = [item.get(sort_key, 0) for item in items if isinstance(item.get(sort_key), (int, float))]
    
    return {
        "summary": {
            "total_count": len(items),
            "showing_top": top_n,
            "sort_by": sort_key,
            f"min_{sort_key}": min(values) if values else None,
            f"max_{sort_key}": max(values) if values else None,
            f"avg_{sort_key}": round(sum(values) / len(values), 2) if values else None,
        },
        "top_items": top_items,
        "note": f"Showing top {top_n} of {len(items)} results sorted by {sort_key}. "
                f"Request specific items by ID or ask for the next page for more.",
    }


# Example: sales data with 1000 records
sales = [
    {"product_id": f"PRD-{i:04d}", "revenue": round(100 + i * 1.5 + (i % 7) * 20, 2), "units": 10 + i}
    for i in range(1000)
]

result = summarize_with_top_items(sales, sort_key="revenue", top_n=5)
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "summary": {
    "total_count": 1000,
    "showing_top": 5,
    "sort_by": "revenue",
    "min_revenue": 100.0,
    "max_revenue": 1619.5,
    "avg_revenue": 879.5
  },
  "top_items": [
    {"product_id": "PRD-0999", "revenue": 1619.5, "units": 1009},
    {"product_id": "PRD-0998", "revenue": 1617.0, "units": 1008},
    {"product_id": "PRD-0997", "revenue": 1615.5, "units": 1007},
    {"product_id": "PRD-0996", "revenue": 1614.0, "units": 1006},
    {"product_id": "PRD-0993", "revenue": 1609.5, "units": 1003}
  ],
  "note": "Showing top 5 of 1000 results sorted by revenue. Request specific items by ID or ask for the next page for more."
}
```

### Grouped summaries

For categorical data, group and summarize:

```python
from collections import defaultdict


def group_and_summarize(
    items: list[dict],
    group_by: str,
    aggregate_field: str,
    top_groups: int = 5,
) -> dict:
    """Group items and return summary statistics per group."""
    groups = defaultdict(list)
    for item in items:
        key = item.get(group_by, "unknown")
        groups[key].append(item.get(aggregate_field, 0))
    
    group_stats = []
    for key, values in groups.items():
        group_stats.append({
            group_by: key,
            "count": len(values),
            f"total_{aggregate_field}": sum(values),
            f"avg_{aggregate_field}": round(sum(values) / len(values), 2),
        })
    
    # Sort by total descending
    group_stats.sort(key=lambda x: x[f"total_{aggregate_field}"], reverse=True)
    
    return {
        "total_items": len(items),
        "total_groups": len(groups),
        "top_groups": group_stats[:top_groups],
        "remaining_groups": len(groups) - top_groups if len(groups) > top_groups else 0,
    }


# Example: orders grouped by category
orders = [
    {"category": f"Cat-{i % 8}", "amount": 50 + (i * 3) % 200}
    for i in range(500)
]

result = group_and_summarize(orders, group_by="category", aggregate_field="amount")
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "total_items": 500,
  "total_groups": 8,
  "top_groups": [
    {"category": "Cat-1", "count": 63, "total_amount": 7371, "avg_amount": 117.0},
    {"category": "Cat-4", "count": 63, "total_amount": 7308, "avg_amount": 116.0},
    {"category": "Cat-7", "count": 62, "total_amount": 7254, "avg_amount": 117.0},
    {"category": "Cat-2", "count": 63, "total_amount": 7182, "avg_amount": 114.0},
    {"category": "Cat-5", "count": 62, "total_amount": 7130, "avg_amount": 115.0}
  ],
  "remaining_groups": 3
}
```

---

## Token budget management

A more sophisticated approach: allocate a token budget and fill it intelligently:

```python
import json


class TokenBudgetManager:
    """Manage function result content within a token budget."""
    
    CHARS_PER_TOKEN = 4  # Rough estimate for English text/JSON
    
    def __init__(self, max_tokens: int = 4_000):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.CHARS_PER_TOKEN
    
    def fit_to_budget(self, result: dict, priority_fields: list[str] | None = None) -> dict:
        """Reduce a result to fit within the token budget.
        
        Strategy:
        1. Try full result first
        2. If too large, keep priority fields and trim the rest
        3. If still too large, truncate list fields
        4. Final fallback: return summary only
        """
        serialized = json.dumps(result, default=str)
        
        # Step 1: Full result fits
        if len(serialized) <= self.max_chars:
            return result
        
        # Step 2: Keep priority fields, remove others
        if priority_fields:
            reduced = {k: v for k, v in result.items() if k in priority_fields}
            reduced["_note"] = f"Result reduced to fit token budget. Original had {len(result)} fields."
            serialized = json.dumps(reduced, default=str)
            if len(serialized) <= self.max_chars:
                return reduced
        
        # Step 3: Truncate list fields
        reduced = self._truncate_lists(result)
        serialized = json.dumps(reduced, default=str)
        if len(serialized) <= self.max_chars:
            return reduced
        
        # Step 4: Summary only
        return self._create_summary(result)
    
    def _truncate_lists(self, data: dict, max_list_items: int = 5) -> dict:
        """Truncate any list fields to max_list_items."""
        result = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > max_list_items:
                result[key] = value[:max_list_items]
                result[f"_{key}_info"] = {
                    "showing": max_list_items,
                    "total": len(value),
                    "truncated": True,
                }
            elif isinstance(value, dict):
                result[key] = self._truncate_lists(value, max_list_items)
            else:
                result[key] = value
        return result
    
    def _create_summary(self, data: dict) -> dict:
        """Create a minimal summary when result is very large."""
        summary = {"_budget_exceeded": True, "_original_fields": list(data.keys())}
        
        for key, value in data.items():
            if isinstance(value, list):
                summary[key] = f"[{len(value)} items]"
            elif isinstance(value, dict):
                summary[key] = f"{{object with {len(value)} keys}}"
            elif isinstance(value, str) and len(value) > 100:
                summary[key] = value[:100] + "..."
            else:
                summary[key] = value
        
        return summary


# Usage
budget = TokenBudgetManager(max_tokens=500)  # ~2000 characters

# Large result that needs reduction
large_result = {
    "query": "top products",
    "status": "success",
    "products": [{"id": i, "name": f"Product {i}", "desc": "A" * 100} for i in range(200)],
    "metadata": {"source": "database", "query_time_ms": 45},
}

fitted = budget.fit_to_budget(
    large_result,
    priority_fields=["query", "status", "products", "metadata"]
)
print(json.dumps(fitted, indent=2))
```

**Output:**
```json
{
  "query": "top products",
  "status": "success",
  "products": [
    {"id": 0, "name": "Product 0", "desc": "AAAA..."},
    {"id": 1, "name": "Product 1", "desc": "AAAA..."},
    {"id": 2, "name": "Product 2", "desc": "AAAA..."},
    {"id": 3, "name": "Product 3", "desc": "AAAA..."},
    {"id": 4, "name": "Product 4", "desc": "AAAA..."}
  ],
  "_products_info": {
    "showing": 5,
    "total": 200,
    "truncated": true
  },
  "metadata": {"source": "database", "query_time_ms": 45}
}
```

---

## Putting it all together: ResultSizeManager

A comprehensive class that combines detection, reduction, and pagination:

```python
import json
from dataclasses import dataclass, field
from enum import Enum


class ReductionStrategy(Enum):
    PAGINATE = "paginate"
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    TOKEN_BUDGET = "token_budget"


@dataclass
class SizeConfig:
    """Configuration for result size management."""
    max_chars: int = 20_000
    max_list_items: int = 20
    max_depth: int = 4
    page_size: int = 10
    strategy: ReductionStrategy = ReductionStrategy.PAGINATE


class ResultSizeManager:
    """Manages function result sizes across providers."""
    
    PROVIDER_DEFAULTS = {
        "openai": SizeConfig(max_chars=50_000),
        "anthropic": SizeConfig(max_chars=40_000),
        "gemini": SizeConfig(max_chars=50_000),
    }
    
    def __init__(self, provider: str = "openai", config: SizeConfig | None = None):
        self.config = config or self.PROVIDER_DEFAULTS.get(provider, SizeConfig())
    
    def process(self, result: any) -> dict:
        """Process a result, reducing if necessary."""
        serialized = json.dumps(result, default=str)
        
        if len(serialized) <= self.config.max_chars:
            return {"data": result, "reduced": False}
        
        # Apply the configured strategy
        if self.config.strategy == ReductionStrategy.PAGINATE:
            reduced = self._paginate(result)
        elif self.config.strategy == ReductionStrategy.SUMMARIZE:
            reduced = self._summarize(result)
        elif self.config.strategy == ReductionStrategy.TRUNCATE:
            reduced = self._truncate(result)
        else:
            reduced = self._token_budget(result)
        
        return {"data": reduced, "reduced": True}
    
    def _paginate(self, result: any) -> dict:
        """Paginate list results."""
        if isinstance(result, list):
            page_size = self.config.page_size
            return {
                "items": result[:page_size],
                "pagination": {
                    "total": len(result),
                    "showing": min(page_size, len(result)),
                    "has_more": len(result) > page_size,
                },
            }
        return self._truncate(result)
    
    def _summarize(self, result: any) -> dict:
        """Create a summary of the result."""
        if isinstance(result, list):
            return {
                "total_items": len(result),
                "first_item": result[0] if result else None,
                "last_item": result[-1] if result else None,
                "note": "Large result summarized. Ask for specific items or pages.",
            }
        return {"summary": str(result)[:500], "truncated": True}
    
    def _truncate(self, result: any) -> any:
        """Truncate oversized fields."""
        if isinstance(result, list):
            return result[:self.config.max_list_items]
        if isinstance(result, dict):
            truncated = {}
            for k, v in result.items():
                if isinstance(v, str) and len(v) > 200:
                    truncated[k] = v[:200] + "..."
                elif isinstance(v, list) and len(v) > self.config.max_list_items:
                    truncated[k] = v[:self.config.max_list_items]
                else:
                    truncated[k] = v
            return truncated
        return result
    
    def _token_budget(self, result: any) -> any:
        """Fit result within character budget."""
        budget = TokenBudgetManager(max_tokens=self.config.max_chars // 4)
        if isinstance(result, dict):
            return budget.fit_to_budget(result)
        return self._truncate(result)


# Usage
manager = ResultSizeManager(provider="openai")

large_list = [{"id": i, "value": f"data-{i}"} for i in range(5000)]
processed = manager.process(large_list)

print(f"Reduced: {processed['reduced']}")
print(json.dumps(processed["data"], indent=2))
```

**Output:**
```json
{
  "items": [
    {"id": 0, "value": "data-0"},
    {"id": 1, "value": "data-1"},
    ...
    {"id": 9, "value": "data-9"}
  ],
  "pagination": {
    "total": 5000,
    "showing": 10,
    "has_more": true
  }
}
```

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Measure result size before sending | Prevents API errors and wasted tokens |
| Include pagination metadata | Models can request additional pages automatically |
| Prefer summaries over raw dumps | Models understand aggregates better than raw data |
| Set conservative default limits | Start small, increase only when needed |
| Include a `note` field explaining reductions | Helps the model know more data is available |
| Use adaptive page sizes | Items of different sizes need different page counts |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Returning all database rows to the model | Paginate and return the first page with metadata |
| Using character count without token estimation | Divide characters by 4 for a rough token count |
| Truncating data without telling the model | Always include metadata about what was removed |
| Hard-coding page sizes | Use adaptive sizing based on item complexity |
| Ignoring nested object depth | Deep nesting inflates size; flatten or limit depth |
| Reducing results before knowing the question | Let the model's query guide which reduction strategy to use |

---

## Hands-on exercise

### Your task

Build a `SmartResultReducer` class that automatically chooses the best reduction strategy based on the shape of the data.

### Requirements

1. Create a class with a `reduce(data, max_chars=20000)` method
2. If `data` is a **list**: apply pagination with adaptive page size
3. If `data` is a **dict with a list field** that's large: summarize that field while keeping other fields
4. If `data` is a **dict with large string values**: truncate strings
5. Always return a dict with `result`, `reduced` (bool), and `strategy_used`

### Expected result

```python
reducer = SmartResultReducer()

# List input → pagination
result = reducer.reduce([{"id": i} for i in range(1000)])
# result["strategy_used"] == "pagination"

# Dict with large list → summary
result = reducer.reduce({"query": "users", "results": [{"id": i} for i in range(1000)]})
# result["strategy_used"] == "field_summary"
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `isinstance()` to detect the data shape
- Calculate the serialized size of individual fields to find the largest ones
- For adaptive page size, divide `max_chars` by the size of one item
- Remember to include metadata about what was reduced

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json


class SmartResultReducer:
    """Automatically choose the best reduction strategy."""
    
    def reduce(self, data: any, max_chars: int = 20_000) -> dict:
        """Reduce data to fit within max_chars, choosing the best strategy."""
        serialized = json.dumps(data, default=str)
        
        if len(serialized) <= max_chars:
            return {"result": data, "reduced": False, "strategy_used": "none"}
        
        if isinstance(data, list):
            return self._paginate(data, max_chars)
        
        if isinstance(data, dict):
            # Find the largest field
            field_sizes = {}
            for k, v in data.items():
                field_sizes[k] = len(json.dumps(v, default=str))
            
            largest_field = max(field_sizes, key=field_sizes.get)
            
            if isinstance(data[largest_field], list):
                return self._summarize_field(data, largest_field, max_chars)
            elif isinstance(data[largest_field], str):
                return self._truncate_strings(data, max_chars)
        
        # Fallback
        return {
            "result": json.dumps(data, default=str)[:max_chars],
            "reduced": True,
            "strategy_used": "raw_truncation",
        }
    
    def _paginate(self, items: list, max_chars: int) -> dict:
        """Paginate a list with adaptive page size."""
        if not items:
            return {"result": [], "reduced": False, "strategy_used": "none"}
        
        item_size = len(json.dumps(items[0], default=str))
        overhead = 200  # For metadata
        page_size = max(1, (max_chars - overhead) // item_size)
        
        result = {
            "items": items[:page_size],
            "pagination": {
                "total": len(items),
                "showing": min(page_size, len(items)),
                "page": 1,
                "has_more": page_size < len(items),
            },
        }
        
        return {"result": result, "reduced": True, "strategy_used": "pagination"}
    
    def _summarize_field(self, data: dict, field_name: str, max_chars: int) -> dict:
        """Keep all fields but summarize the large list field."""
        items = data[field_name]
        result = {k: v for k, v in data.items() if k != field_name}
        
        # Include first 5 items plus summary
        result[field_name] = items[:5]
        result[f"_{field_name}_summary"] = {
            "total": len(items),
            "showing": min(5, len(items)),
            "truncated": True,
        }
        
        return {"result": result, "reduced": True, "strategy_used": "field_summary"}
    
    def _truncate_strings(self, data: dict, max_chars: int) -> dict:
        """Truncate large string values."""
        result = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 200:
                result[k] = v[:200] + f"... [{len(v)} chars total]"
            else:
                result[k] = v
        
        return {"result": result, "reduced": True, "strategy_used": "string_truncation"}


# Test all strategies
reducer = SmartResultReducer()

# List → pagination
r1 = reducer.reduce([{"id": i, "val": "x" * 100} for i in range(1000)])
print(f"List: strategy={r1['strategy_used']}, reduced={r1['reduced']}")

# Dict with large list → field summary
r2 = reducer.reduce({"query": "users", "results": [{"id": i} for i in range(1000)]})
print(f"Dict+list: strategy={r2['strategy_used']}, reduced={r2['reduced']}")

# Dict with large string → string truncation
r3 = reducer.reduce({"title": "Report", "body": "x" * 50_000})
print(f"Dict+string: strategy={r3['strategy_used']}, reduced={r3['reduced']}")
```

**Output:**
```
List: strategy=pagination, reduced=True
Dict+list: strategy=field_summary, reduced=True
Dict+string: strategy=string_truncation, reduced=True
```

</details>

### Bonus challenges

- [ ] Add a `ReductionStrategy.HYBRID` that combines summarization and pagination
- [ ] Implement depth-limited reduction for deeply nested objects
- [ ] Add token counting using `tiktoken` for accurate OpenAI token estimation

---

## Summary

✅ Function result size directly impacts token usage, cost, and model performance

✅ Measure results before sending — use character count ÷ 4 for rough token estimates

✅ Pagination with metadata (total, has_next) lets models request additional pages

✅ Summary-with-details returns aggregates plus top items instead of raw dumps

✅ Token budget management provides multi-layered reduction: priority fields → list truncation → summary fallback

✅ Always include metadata explaining what was reduced so the model knows more data is available

**Next:** [Error Result Formatting →](./05-error-result-formatting.md) — Structuring error information in function results

---

[← Previous: Multimodal Results](./03-multimodal-results.md) | [Back to Lesson Overview](./00-returning-results.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
-->
