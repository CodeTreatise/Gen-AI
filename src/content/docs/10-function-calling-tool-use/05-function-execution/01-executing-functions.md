---
title: "Executing Functions"
---

# Executing Functions

## Introduction

You've detected the function call, parsed the arguments, validated them against your schema, and dispatched to the right handler. Now it's time to actually **invoke** the function. This step seems straightforward — call the function, get a result — but production systems need to handle parameter passing, inject execution context, manage return values, and deal with functions that weren't designed to be called by an AI model.

This lesson covers the mechanics of function invocation: how to pass parsed arguments to your handlers, how to inject context that the model doesn't know about (like user permissions or database connections), and how to structure your execution layer across all three major providers.

### What we'll cover

- Invoking handler functions with parsed arguments
- Parameter passing strategies (spread vs. explicit)
- Injecting execution context (user info, request metadata)
- Execution context objects for production systems
- Provider-specific execution patterns (OpenAI, Anthropic, Gemini)
- Building a unified execution layer

### Prerequisites

- Parsing arguments ([Lesson 04: Parsing Arguments](../04-handling-function-calls/03-parsing-arguments.md))
- Function dispatch ([Lesson 04: Function Dispatch](../04-handling-function-calls/07-function-dispatch.md))

---

## Basic function invocation

At its simplest, executing a function means calling it with the arguments the model provided. Here's the minimal pattern after you've parsed and validated everything:

```python
import json

# Simulated parsed function call from the model
function_name = "get_weather"
arguments_str = '{"location": "Paris, France", "units": "celsius"}'

# Parse arguments (OpenAI sends JSON string, others send objects)
args = json.loads(arguments_str)

# Your function registry
def get_weather(location: str, units: str = "celsius") -> dict:
    """Look up weather for a location."""
    # In production: call a real weather API
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "condition": "partly cloudy"
    }

registry = {
    "get_weather": get_weather,
}

# Execute
func = registry[function_name]
result = func(**args)
print(result)
```

**Output:**
```
{'location': 'Paris, France', 'temperature': 22, 'units': 'celsius', 'condition': 'partly cloudy'}
```

The `**args` syntax unpacks the dictionary into keyword arguments, matching each key to a function parameter. This works because the model's arguments (after validation) should match your function's signature exactly.

---

## Parameter passing strategies

### Spread (kwargs unpacking)

The most common approach — unpack the parsed dictionary directly:

```python
def search_products(query: str, category: str = "all", max_results: int = 10) -> list:
    """Search product catalog."""
    # Implementation here
    return [{"name": f"Result for '{query}' in {category}", "id": 1}]

# Model provides: {"query": "red shoes", "category": "footwear"}
args = {"query": "red shoes", "category": "footwear"}
result = search_products(**args)
```

**Output:**
```
[{'name': "Result for 'red shoes' in footwear", 'id': 1}]
```

> **Note:** This works well when your function signature matches the JSON Schema you defined for the model. If the model sends extra keys not in your function signature, `**args` will raise a `TypeError`. Use strict mode or filter arguments first.

### Explicit parameter extraction

When you need more control — for example, to transform arguments before passing them:

```python
def execute_with_extraction(func_name: str, args: dict) -> dict:
    """Execute function with explicit parameter handling."""
    if func_name == "search_products":
        # Transform or validate specific parameters
        query = args["query"].strip().lower()
        category = args.get("category", "all")
        max_results = min(args.get("max_results", 10), 50)  # Cap at 50
        
        return search_products(
            query=query,
            category=category,
            max_results=max_results
        )
    
    raise ValueError(f"Unknown function: {func_name}")
```

### Filtering unexpected arguments

If the model might send extra fields (especially without strict mode), filter them:

```python
import inspect

def safe_invoke(func, args: dict) -> any:
    """Invoke function with only the parameters it accepts."""
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    
    # Filter to only accepted parameters
    filtered_args = {k: v for k, v in args.items() if k in valid_params}
    
    # Check for missing required parameters
    for param_name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and param_name not in filtered_args:
            raise ValueError(f"Missing required parameter: {param_name}")
    
    return func(**filtered_args)

# Even if the model sends extra fields, this won't crash
args_with_extra = {"query": "red shoes", "category": "footwear", "unknown_field": True}
result = safe_invoke(search_products, args_with_extra)
print(result)
```

**Output:**
```
[{'name': "Result for 'red shoes' in footwear", 'id': 1}]
```

---

## Context injection

The model only sees the parameters defined in your function's JSON Schema. But real functions often need additional context: who's making the request, what database connection to use, what permissions apply. Context injection passes this information alongside the model's arguments.

### The context object pattern

```python
from dataclasses import dataclass, field
from typing import Optional
import uuid
import time

@dataclass
class ExecutionContext:
    """Context passed to every function execution."""
    user_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    permissions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "admin" in self.permissions

# Functions accept context as first argument (not exposed to model)
def get_user_orders(ctx: ExecutionContext, status: str = "all", limit: int = 10) -> dict:
    """Get orders for the current user."""
    if not ctx.has_permission("read:orders"):
        return {"error": "Permission denied: read:orders required"}
    
    # Use ctx.user_id to scope the query
    return {
        "user_id": ctx.user_id,
        "orders": [
            {"id": "ord_123", "status": "shipped", "total": 49.99},
            {"id": "ord_456", "status": "delivered", "total": 29.99},
        ],
        "filtered_by": status,
        "request_id": ctx.request_id
    }

# Execution with context injection
ctx = ExecutionContext(
    user_id="user_789",
    permissions=["read:orders", "read:profile"]
)

# Model provides: {"status": "shipped", "limit": 5}
model_args = {"status": "shipped", "limit": 5}

# Inject context — model args go alongside
result = get_user_orders(ctx, **model_args)
print(result)
```

**Output:**
```
{'user_id': 'user_789', 'orders': [{'id': 'ord_123', 'status': 'shipped', 'total': 49.99}, {'id': 'ord_456', 'status': 'delivered', 'total': 29.99}], 'filtered_by': 'shipped', 'request_id': 'a1b2c3d4-...'}
```

> **🤖 AI Context:** The model never sees the `ExecutionContext` parameter — it's not in the JSON Schema. Your dispatch layer injects it before calling the function. This separates the model's interface (what arguments it can provide) from your application's requirements (who is making the request).

### Separating model args from context args

A cleaner pattern uses a wrapper that handles injection automatically:

```python
from functools import wraps
from typing import Callable

def with_context(func: Callable) -> Callable:
    """Decorator marking a function as requiring execution context."""
    func._requires_context = True
    return func

@with_context
def cancel_order(ctx: ExecutionContext, order_id: str, reason: str = "") -> dict:
    """Cancel an order for the current user."""
    if not ctx.has_permission("write:orders"):
        return {"error": "Permission denied: write:orders required"}
    
    return {
        "cancelled": True,
        "order_id": order_id,
        "cancelled_by": ctx.user_id,
        "reason": reason
    }

def execute_function(
    func: Callable,
    model_args: dict,
    ctx: ExecutionContext
) -> dict:
    """Execute a function, injecting context if required."""
    if getattr(func, '_requires_context', False):
        return func(ctx, **model_args)
    else:
        return func(**model_args)

# Usage
result = execute_function(
    cancel_order,
    {"order_id": "ord_123", "reason": "Changed my mind"},
    ctx
)
print(result)
```

**Output:**
```
{'cancelled': True, 'order_id': 'ord_123', 'cancelled_by': 'user_789', 'reason': 'Changed my mind'}
```

---

## Provider-specific execution patterns

Each provider delivers function calls differently, so the execution entry point varies. Here's how to execute functions from each provider's response format.

### OpenAI (Responses API)

```python
from openai import OpenAI
import json

client = OpenAI()

def execute_openai_function_calls(response, registry: dict, ctx: ExecutionContext) -> list:
    """Execute all function calls from an OpenAI Responses API response."""
    results = []
    
    for item in response.output:
        if item.type != "function_call":
            continue
        
        func_name = item.name
        call_id = item.call_id
        
        # OpenAI sends arguments as JSON string
        args = json.loads(item.arguments)
        
        func = registry.get(func_name)
        if func is None:
            results.append({
                "call_id": call_id,
                "output": json.dumps({"error": f"Unknown function: {func_name}"})
            })
            continue
        
        try:
            result = execute_function(func, args, ctx)
            results.append({
                "call_id": call_id,
                "output": json.dumps(result)
            })
        except Exception as e:
            results.append({
                "call_id": call_id,
                "output": json.dumps({"error": str(e)})
            })
    
    return results

# Each result becomes a function_call_output item in the next request:
# {
#     "type": "function_call_output",
#     "call_id": "call_abc123",
#     "output": '{"cancelled": true, ...}'
# }
```

### Anthropic

```python
import anthropic
import json

client = anthropic.Anthropic()

def execute_anthropic_tool_calls(response, registry: dict, ctx: ExecutionContext) -> list:
    """Execute all tool calls from an Anthropic response."""
    results = []
    
    for block in response.content:
        if block.type != "tool_use":
            continue
        
        func_name = block.name
        tool_use_id = block.id
        
        # Anthropic sends arguments as Python dict (already parsed)
        args = block.input  # No json.loads() needed
        
        func = registry.get(func_name)
        if func is None:
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": f"Unknown function: {func_name}"})
            })
            continue
        
        try:
            result = execute_function(func, args, ctx)
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result)
            })
        except Exception as e:
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": str(e)}),
                "is_error": True
            })
    
    return results

# Results are sent as user message content blocks:
# {"role": "user", "content": [
#     {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "..."}
# ]}
```

### Google Gemini

```python
from google import genai
from google.genai import types
import json

client = genai.Client()

def execute_gemini_function_calls(response, registry: dict, ctx: ExecutionContext) -> list:
    """Execute all function calls from a Gemini response."""
    results = []
    
    for part in response.candidates[0].content.parts:
        if not part.function_call:
            continue
        
        func_name = part.function_call.name
        
        # Gemini sends arguments as a dict-like object
        args = dict(part.function_call.args)
        
        func = registry.get(func_name)
        if func is None:
            results.append(
                types.Part.from_function_response(
                    name=func_name,
                    response={"error": f"Unknown function: {func_name}"}
                )
            )
            continue
        
        try:
            result = execute_function(func, args, ctx)
            results.append(
                types.Part.from_function_response(
                    name=func_name,
                    response=result
                )
            )
        except Exception as e:
            results.append(
                types.Part.from_function_response(
                    name=func_name,
                    response={"error": str(e)}
                )
            )
    
    return results

# Results are sent as function response parts in user content:
# types.Content(role="user", parts=[...function_response_parts...])
```

---

## Unified execution layer

For applications that support multiple providers, wrap execution in a unified layer:

```python
import json
import logging
import time
from typing import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FunctionCallRequest:
    """Provider-agnostic function call representation."""
    name: str
    arguments: dict
    call_id: str  # Provider-specific ID (or generated for Gemini)
    provider: str  # "openai", "anthropic", "gemini"

@dataclass
class FunctionCallResult:
    """Provider-agnostic function result."""
    call_id: str
    output: str  # JSON-stringified result
    is_error: bool = False
    execution_time_ms: float = 0

def execute_call(
    call: FunctionCallRequest,
    registry: dict[str, Callable],
    ctx: ExecutionContext
) -> FunctionCallResult:
    """Execute a single function call, provider-agnostic."""
    start = time.monotonic()
    
    func = registry.get(call.name)
    if func is None:
        return FunctionCallResult(
            call_id=call.call_id,
            output=json.dumps({"error": f"Unknown function: {call.name}"}),
            is_error=True
        )
    
    try:
        result = execute_function(func, call.arguments, ctx)
        elapsed = (time.monotonic() - start) * 1000
        
        logger.info(
            "Function executed",
            extra={
                "function": call.name,
                "call_id": call.call_id,
                "execution_time_ms": elapsed,
                "user_id": ctx.user_id
            }
        )
        
        return FunctionCallResult(
            call_id=call.call_id,
            output=json.dumps(result),
            execution_time_ms=elapsed
        )
    
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        
        logger.error(
            "Function execution failed",
            extra={
                "function": call.name,
                "call_id": call.call_id,
                "error": str(e),
                "execution_time_ms": elapsed
            }
        )
        
        return FunctionCallResult(
            call_id=call.call_id,
            output=json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "function": call.name
            }),
            is_error=True,
            execution_time_ms=elapsed
        )
```

This unified layer gives you:

- **Consistent logging** across all providers
- **Timing metrics** for every execution
- **Error formatting** that works for any provider's result format
- **Single entry point** for adding timeout, sandboxing, or authorization checks

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use `**kwargs` unpacking for clean invocation | Keeps function signatures readable and matches schema definitions |
| Inject context as first argument, not in kwargs | Separates model-controlled args from app-controlled context |
| Log every execution with timing | Essential for debugging slow tools and monitoring costs |
| Always catch exceptions during execution | Never let a function crash the calling loop |
| Return structured error objects, not strings | The model can parse structured errors and self-correct |
| Use `inspect.signature` to filter unexpected args | Protects against non-strict mode sending extra fields |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Passing JSON string directly to function | Parse with `json.loads()` first (OpenAI sends strings) |
| Letting execution exceptions propagate | Wrap every call in `try`/`except`, return error as result |
| Exposing internal context to the model | Keep `ExecutionContext` out of JSON Schema definitions |
| Not logging which user triggered execution | Always include `user_id` and `request_id` in execution logs |
| Using the same execution path for all providers | Account for format differences (JSON string vs. dict) in a normalization layer |
| Trusting model arguments without validation | Always validate before executing, even after dispatch |

---

## Hands-on exercise

### Your task

Build a function execution layer that handles multi-function execution from a simulated model response. The layer should support context injection, parameter filtering, execution timing, and error handling.

### Requirements

1. Create an `ExecutionContext` with `user_id`, `request_id`, and `permissions`
2. Define three functions: `get_balance` (requires `read:finance`), `transfer_funds` (requires `write:finance`), and `get_exchange_rate` (no permissions needed)
3. Build an `execute_call` function that injects context and checks permissions
4. Process a batch of three simulated function calls, including one that should fail due to missing permissions
5. Return structured results with timing information

### Expected result

Three results: two successful, one permission-denied error — all with execution timing and request IDs.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `@dataclass` for both `ExecutionContext` and result types
- Check `ctx.has_permission()` inside each function, or build a permissions decorator
- Use `time.monotonic()` for accurate timing (not `time.time()`)
- Return results as a list of dictionaries with `call_id`, `output`, `is_error`, and `execution_time_ms`
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass, field
import json
import time
import uuid
from typing import Callable

@dataclass
class ExecutionContext:
    user_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    permissions: list[str] = field(default_factory=list)
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "admin" in self.permissions

@dataclass
class CallResult:
    call_id: str
    output: str
    is_error: bool = False
    execution_time_ms: float = 0

# Define functions with context
def get_balance(ctx: ExecutionContext, account_id: str = "default") -> dict:
    if not ctx.has_permission("read:finance"):
        return {"error": "Permission denied: read:finance required"}
    return {"account_id": account_id, "balance": 1250.00, "currency": "USD"}

def transfer_funds(ctx: ExecutionContext, from_account: str, to_account: str, amount: float) -> dict:
    if not ctx.has_permission("write:finance"):
        return {"error": "Permission denied: write:finance required"}
    return {"transferred": amount, "from": from_account, "to": to_account, "status": "completed"}

def get_exchange_rate(ctx: ExecutionContext, base: str, target: str) -> dict:
    rates = {"USD_EUR": 0.92, "USD_GBP": 0.79, "EUR_USD": 1.09}
    key = f"{base}_{target}"
    rate = rates.get(key, 1.0)
    return {"base": base, "target": target, "rate": rate}

registry = {
    "get_balance": get_balance,
    "transfer_funds": transfer_funds,
    "get_exchange_rate": get_exchange_rate,
}

def execute_call(name: str, args: dict, call_id: str, ctx: ExecutionContext) -> CallResult:
    start = time.monotonic()
    func = registry.get(name)
    
    if func is None:
        return CallResult(
            call_id=call_id,
            output=json.dumps({"error": f"Unknown function: {name}"}),
            is_error=True
        )
    
    try:
        result = func(ctx, **args)
        elapsed = (time.monotonic() - start) * 1000
        is_err = "error" in result
        return CallResult(
            call_id=call_id,
            output=json.dumps(result),
            is_error=is_err,
            execution_time_ms=round(elapsed, 2)
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return CallResult(
            call_id=call_id,
            output=json.dumps({"error": str(e)}),
            is_error=True,
            execution_time_ms=round(elapsed, 2)
        )

# Simulate execution
ctx = ExecutionContext(
    user_id="user_42",
    permissions=["read:finance"]  # Note: no write:finance
)

calls = [
    ("get_balance", {"account_id": "checking"}, "call_001"),
    ("transfer_funds", {"from_account": "checking", "to_account": "savings", "amount": 100.0}, "call_002"),
    ("get_exchange_rate", {"base": "USD", "target": "EUR"}, "call_003"),
]

results = [execute_call(name, args, cid, ctx) for name, args, cid in calls]

for r in results:
    print(f"[{'ERROR' if r.is_error else 'OK'}] {r.call_id}: {r.output} ({r.execution_time_ms}ms)")
```

**Output:**
```
[OK] call_001: {"account_id": "checking", "balance": 1250.0, "currency": "USD"} (0.01ms)
[ERROR] call_002: {"error": "Permission denied: write:finance required"} (0.01ms)
[OK] call_003: {"base": "USD", "target": "EUR", "rate": 0.92} (0.01ms)
```
</details>

### Bonus challenges

- [ ] Add a `@requires_permission("write:finance")` decorator that checks permissions before the function body runs
- [ ] Implement argument filtering using `inspect.signature` so extra model arguments don't crash functions
- [ ] Add structured logging with `request_id` correlation across all executions

---

## Summary

✅ Use `**args` dictionary unpacking to pass parsed model arguments to handler functions

✅ Inject execution context (user info, permissions, request IDs) as the first argument, separate from model-controlled parameters

✅ Each provider delivers arguments differently: OpenAI sends JSON strings, Anthropic and Gemini send Python objects

✅ Always wrap function execution in `try`/`except` — return errors as structured results, never let them crash the calling loop

✅ Build a unified execution layer with logging and timing for multi-provider support

**Next:** [Automatic Function Calling →](./02-automatic-function-calling.md) — SDK-managed execution that skips the manual dispatch loop

---

[← Previous: Lesson Overview](./00-function-execution.md) | [Next: Automatic Function Calling →](./02-automatic-function-calling.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Python inspect module: https://docs.python.org/3/library/inspect.html
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
-->
