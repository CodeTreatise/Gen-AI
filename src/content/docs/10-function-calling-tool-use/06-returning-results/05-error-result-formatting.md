---
title: "Error Result Formatting"
---

# Error Result Formatting

## Introduction

Functions fail. APIs return 404s, databases time out, calculations divide by zero. When a function encounters an error, you need to communicate that failure back to the model in a structured way that helps it **understand what went wrong** and **decide what to do next**. A well-formatted error result lets the model retry with different parameters, try an alternative approach, or explain the issue to the user.

Each provider handles error results differently. Anthropic has a dedicated `is_error` flag. OpenAI and Gemini expect you to encode errors within the result string or dictionary. This lesson covers how to format error results across all three providers and build reusable error formatting utilities.

### What we'll cover

- Provider-specific error result mechanisms
- Structuring error information for model comprehension
- Error categories and severity levels
- Actionable error messages with recovery hints
- Building a unified error formatter
- Error context and stack trace handling

### Prerequisites

- Result format structure ([Lesson 06-01](./01-result-format-structure.md))
- Error handling basics ([Unit 02 - Lesson 13](../../02-python-for-ai-development/13-error-handling-debugging/))

---

## Provider-specific error handling

### Anthropic: The `is_error` flag

Anthropic is the only provider with a **dedicated error flag** in the tool result format:

```python
import anthropic

client = anthropic.Anthropic()

# When a function succeeds
success_result = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_abc123",
            "content": '{"temperature": 72, "unit": "fahrenheit"}'
            # No is_error field → defaults to False
        }
    ]
}

# When a function fails
error_result = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_abc123",
            "is_error": True,  # Explicit error flag
            "content": "City 'Atlantis' not found. Available cities include major world capitals. Please try a valid city name."
        }
    ]
}
```

> **Important:** When `is_error` is `True`, Claude knows the function failed and adjusts its behavior accordingly — it won't treat the error message as successful data. This is the cleanest error signaling among the three providers.

### OpenAI: Error in the output string

OpenAI has no dedicated error flag. You communicate errors through the output content:

```python
from openai import OpenAI

client = OpenAI()

# Error result — encode the error information in the output
error_result = {
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": '{"error": true, "error_type": "not_found", "message": "City \'Atlantis\' not found. Try a valid city name like London, Tokyo, or New York."}'
}

# The model sees "error": true in the JSON and understands the function failed
```

> **Tip:** Use a consistent error structure in your output JSON (`"error": true`, `"error_type"`, `"message"`) so the model can reliably detect failures.

### Gemini: Error in the response dictionary

Gemini function responses accept a Python dictionary, so you can include error information directly:

```python
from google.genai import types

# Error result — include error info in the response dict
error_response = types.Part.from_function_response(
    name="get_weather",
    response={
        "error": True,
        "error_type": "not_found",
        "message": "City 'Atlantis' not found in weather database.",
        "suggestion": "Try a major city name like 'London' or 'Tokyo'.",
    }
)
```

### Provider comparison for error handling

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| **Dedicated error flag** | ❌ Encode in output | ✅ `is_error: true` | ❌ Encode in response |
| **Error content type** | String (JSON) | String or content blocks | Python dict |
| **Model awareness** | Infers from content | Explicit signal | Infers from content |
| **Structured errors** | JSON in string | Text or JSON | Dict fields |

---

## Structuring error information

A good error result gives the model three things: **what failed**, **why it failed**, and **what to try instead**.

```python
import json
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ErrorCategory(Enum):
    """Categories of function errors for model comprehension."""
    VALIDATION = "validation_error"       # Bad input parameters
    NOT_FOUND = "not_found"               # Resource doesn't exist
    PERMISSION = "permission_denied"      # Auth/access issue
    RATE_LIMIT = "rate_limited"           # Too many requests
    TIMEOUT = "timeout"                   # Operation took too long
    UNAVAILABLE = "service_unavailable"   # External service down
    INTERNAL = "internal_error"           # Unexpected bug
    CONFLICT = "conflict"                 # Resource state conflict


class ErrorSeverity(Enum):
    """How severe the error is — guides model behavior."""
    RETRYABLE = "retryable"         # Same call might work if retried
    FIXABLE = "fixable"             # Different parameters would work
    PERMANENT = "permanent"         # This will never work
    DEGRADED = "degraded"           # Partial results available


@dataclass
class FunctionError:
    """A structured function error with context for the model."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    suggestion: str | None = None
    details: dict | None = None
    
    def to_dict(self) -> dict:
        result = {
            "error": True,
            "error_type": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.details:
            result["details"] = self.details
        return result
    
    def to_string(self) -> str:
        """For providers that need string output."""
        return json.dumps(self.to_dict())


# Usage examples
errors = [
    FunctionError(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.FIXABLE,
        message="Parameter 'date' must be in YYYY-MM-DD format.",
        suggestion="Received '02/15/2025'. Try '2025-02-15' instead.",
    ),
    FunctionError(
        category=ErrorCategory.RATE_LIMIT,
        severity=ErrorSeverity.RETRYABLE,
        message="Rate limit exceeded: 60 requests per minute.",
        suggestion="Wait 15 seconds before retrying.",
        details={"retry_after_seconds": 15, "limit": 60, "window": "1 minute"},
    ),
    FunctionError(
        category=ErrorCategory.NOT_FOUND,
        severity=ErrorSeverity.FIXABLE,
        message="User with ID 'usr_999' not found.",
        suggestion="Check the user ID. Use search_users() to find valid IDs.",
    ),
    FunctionError(
        category=ErrorCategory.UNAVAILABLE,
        severity=ErrorSeverity.RETRYABLE,
        message="Weather API is currently unavailable.",
        suggestion="The service may be temporarily down. Try again in a few minutes.",
    ),
]

for error in errors:
    print(f"{error.category.value}: {error.message}")
    print(f"  Severity: {error.severity.value}")
    print(f"  Suggestion: {error.suggestion}\n")
```

**Output:**
```
validation_error: Parameter 'date' must be in YYYY-MM-DD format.
  Severity: fixable
  Suggestion: Received '02/15/2025'. Try '2025-02-15' instead.

rate_limited: Rate limit exceeded: 60 requests per minute.
  Severity: retryable
  Suggestion: Wait 15 seconds before retrying.

not_found: User with ID 'usr_999' not found.
  Severity: fixable
  Suggestion: Check the user ID. Use search_users() to find valid IDs.

service_unavailable: Weather API is currently unavailable.
  Severity: retryable
  Suggestion: The service may be temporarily down. Try again in a few minutes.
```

---

## Actionable error messages

The difference between a helpful and unhelpful error result:

```python
# ❌ Unhelpful — model doesn't know what to do
bad_error = {"error": True, "message": "Failed"}

# ❌ Still unhelpful — what dates ARE valid?
mediocre_error = {"error": True, "message": "Invalid date format"}

# ✅ Actionable — model knows exactly how to fix it
good_error = {
    "error": True,
    "error_type": "validation_error",
    "message": "Invalid date format for parameter 'start_date'.",
    "received": "Feb 15 2025",
    "expected_format": "YYYY-MM-DD",
    "example": "2025-02-15",
    "suggestion": "Reformat the date as '2025-02-15' and try again."
}
```

### Writing effective error messages

Follow this template:

```
[What failed] + [What was received] + [What was expected] + [How to fix it]
```

```python
def format_validation_error(
    parameter: str,
    received_value: any,
    expected: str,
    example: str | None = None,
) -> FunctionError:
    """Create an actionable validation error."""
    message = f"Invalid value for parameter '{parameter}'."
    suggestion_parts = [f"Received: {repr(received_value)}.", f"Expected: {expected}."]
    if example:
        suggestion_parts.append(f"Example: {example}.")
    
    return FunctionError(
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.FIXABLE,
        message=message,
        suggestion=" ".join(suggestion_parts),
        details={
            "parameter": parameter,
            "received": str(received_value),
            "expected": expected,
        }
    )


# Usage
error = format_validation_error(
    parameter="limit",
    received_value=-5,
    expected="A positive integer between 1 and 100",
    example="25"
)
print(json.dumps(error.to_dict(), indent=2))
```

**Output:**
```json
{
  "error": true,
  "error_type": "validation_error",
  "severity": "fixable",
  "message": "Invalid value for parameter 'limit'.",
  "suggestion": "Received: -5. Expected: A positive integer between 1 and 100. Example: 25.",
  "details": {
    "parameter": "limit",
    "received": "-5",
    "expected": "A positive integer between 1 and 100"
  }
}
```

---

## Partial results with errors

Sometimes a function partially succeeds — for example, fetching data for 3 of 5 requested items. Report both the successes and failures:

```python
def fetch_multiple_items(item_ids: list[str]) -> dict:
    """Fetch multiple items, reporting partial failures."""
    results = []
    errors = []
    
    for item_id in item_ids:
        try:
            # Simulate: some items exist, some don't
            if item_id.startswith("VALID"):
                results.append({"id": item_id, "name": f"Item {item_id}", "price": 29.99})
            else:
                raise ValueError(f"Item '{item_id}' not found")
        except Exception as e:
            errors.append({"id": item_id, "error": str(e)})
    
    response = {
        "results": results,
        "total_requested": len(item_ids),
        "successful": len(results),
        "failed": len(errors),
    }
    
    if errors:
        response["errors"] = errors
        response["partial_failure"] = True
        response["note"] = (
            f"Retrieved {len(results)} of {len(item_ids)} items. "
            f"{len(errors)} items could not be found."
        )
    
    return response


# Example
result = fetch_multiple_items(["VALID-001", "INVALID-002", "VALID-003", "INVALID-004", "VALID-005"])
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "results": [
    {"id": "VALID-001", "name": "Item VALID-001", "price": 29.99},
    {"id": "VALID-003", "name": "Item VALID-003", "price": 29.99},
    {"id": "VALID-005", "name": "Item VALID-005", "price": 29.99}
  ],
  "total_requested": 5,
  "successful": 3,
  "failed": 2,
  "errors": [
    {"id": "INVALID-002", "error": "Item 'INVALID-002' not found"},
    {"id": "INVALID-004", "error": "Item 'INVALID-004' not found"}
  ],
  "partial_failure": true,
  "note": "Retrieved 3 of 5 items. 2 items could not be found."
}
```

> **Note:** This is a `"severity": "degraded"` scenario. The model can work with the partial results and decide whether to report the failures or try to recover the missing items.

---

## Formatting errors for each provider

A unified formatter that produces the correct error format for any provider:

```python
import json
import base64


class ErrorResultFormatter:
    """Format function errors for any provider."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
    
    def format(
        self,
        error: FunctionError,
        call_id: str = "",
        function_name: str = "",
    ) -> dict:
        """Format an error result for the configured provider."""
        if self.provider == "anthropic":
            return self._format_anthropic(error, call_id)
        elif self.provider == "gemini":
            return self._format_gemini(error, function_name)
        else:
            return self._format_openai(error, call_id)
    
    def _format_openai(self, error: FunctionError, call_id: str) -> dict:
        """OpenAI: error info encoded in the output string."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": error.to_string(),
        }
    
    def _format_anthropic(self, error: FunctionError, tool_use_id: str) -> dict:
        """Anthropic: uses the is_error flag."""
        # Build a human-readable error message
        content_parts = [error.message]
        if error.suggestion:
            content_parts.append(f"Suggestion: {error.suggestion}")
        
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": " ".join(content_parts),
                }
            ]
        }
    
    def _format_gemini(self, error: FunctionError, function_name: str) -> dict:
        """Gemini: error info in the response dictionary."""
        # Returns the dict (not Part) for flexibility
        return {
            "function_name": function_name,
            "response": error.to_dict(),
        }


# Usage
error = FunctionError(
    category=ErrorCategory.NOT_FOUND,
    severity=ErrorSeverity.FIXABLE,
    message="Product 'SKU-INVALID' not found in catalog.",
    suggestion="Use search_products() to find valid product SKUs.",
)

for provider in ["openai", "anthropic", "gemini"]:
    formatter = ErrorResultFormatter(provider=provider)
    result = formatter.format(
        error,
        call_id="call_abc123" if provider != "gemini" else "",
        function_name="get_product" if provider == "gemini" else "",
    )
    print(f"\n--- {provider.upper()} ---")
    print(json.dumps(result, indent=2))
```

**Output:**
```
--- OPENAI ---
{
  "type": "function_call_output",
  "call_id": "call_abc123",
  "output": "{\"error\": true, \"error_type\": \"not_found\", \"severity\": \"fixable\", \"message\": \"Product 'SKU-INVALID' not found in catalog.\", \"suggestion\": \"Use search_products() to find valid product SKUs.\"}"
}

--- ANTHROPIC ---
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_abc123",
      "is_error": true,
      "content": "Product 'SKU-INVALID' not found in catalog. Suggestion: Use search_products() to find valid product SKUs."
    }
  ]
}

--- GEMINI ---
{
  "function_name": "get_product",
  "response": {
    "error": true,
    "error_type": "not_found",
    "severity": "fixable",
    "message": "Product 'SKU-INVALID' not found in catalog.",
    "suggestion": "Use search_products() to find valid product SKUs."
  }
}
```

---

## Stack traces and sensitive information

Never send raw stack traces to the model. They waste tokens and may leak sensitive information:

```python
import traceback


def safe_error_from_exception(
    exception: Exception,
    function_name: str,
    include_type: bool = True,
    include_trace: bool = False,  # Only for debugging
) -> FunctionError:
    """Convert a Python exception to a safe, model-friendly error."""
    # Map common exception types to categories
    category_map = {
        ValueError: ErrorCategory.VALIDATION,
        KeyError: ErrorCategory.NOT_FOUND,
        FileNotFoundError: ErrorCategory.NOT_FOUND,
        PermissionError: ErrorCategory.PERMISSION,
        TimeoutError: ErrorCategory.TIMEOUT,
        ConnectionError: ErrorCategory.UNAVAILABLE,
    }
    
    category = ErrorCategory.INTERNAL
    for exc_type, cat in category_map.items():
        if isinstance(exception, exc_type):
            category = cat
            break
    
    # Determine severity
    severity_map = {
        ErrorCategory.VALIDATION: ErrorSeverity.FIXABLE,
        ErrorCategory.NOT_FOUND: ErrorSeverity.FIXABLE,
        ErrorCategory.PERMISSION: ErrorSeverity.PERMANENT,
        ErrorCategory.TIMEOUT: ErrorSeverity.RETRYABLE,
        ErrorCategory.UNAVAILABLE: ErrorSeverity.RETRYABLE,
        ErrorCategory.INTERNAL: ErrorSeverity.PERMANENT,
    }
    severity = severity_map.get(category, ErrorSeverity.PERMANENT)
    
    # Build safe message
    message = f"Function '{function_name}' failed"
    if include_type:
        message += f" with {type(exception).__name__}"
    message += f": {str(exception)}"
    
    details = None
    if include_trace:
        # Only include in development/debugging
        details = {"traceback": traceback.format_exc()}
    
    return FunctionError(
        category=category,
        severity=severity,
        message=message,
        suggestion=_generate_suggestion(category, function_name),
        details=details,
    )


def _generate_suggestion(category: ErrorCategory, function_name: str) -> str:
    """Generate a helpful suggestion based on error category."""
    suggestions = {
        ErrorCategory.VALIDATION: f"Check the parameters passed to {function_name}() and ensure they match the expected types and ranges.",
        ErrorCategory.NOT_FOUND: f"The requested resource does not exist. Verify the identifier or use a search function first.",
        ErrorCategory.PERMISSION: "This operation requires additional permissions that are not available.",
        ErrorCategory.TIMEOUT: "The operation timed out. Try again or use a simpler query.",
        ErrorCategory.UNAVAILABLE: "The external service is temporarily unavailable. Try again in a few moments.",
        ErrorCategory.INTERNAL: f"An unexpected error occurred in {function_name}(). Try different parameters.",
    }
    return suggestions.get(category, "Try again with different parameters.")


# Example usage in a function executor
def execute_with_error_handling(func_name: str, func, args: dict) -> dict:
    """Execute a function and format any errors."""
    try:
        result = func(**args)
        return {"error": False, "result": result}
    except Exception as e:
        error = safe_error_from_exception(e, func_name)
        return error.to_dict()


# Simulate various failures
def get_user(user_id: int):
    if user_id < 0:
        raise ValueError("user_id must be positive")
    raise KeyError(f"User {user_id} not found in database")

result = execute_with_error_handling("get_user", get_user, {"user_id": 999})
print(json.dumps(result, indent=2))
```

**Output:**
```json
{
  "error": true,
  "error_type": "not_found",
  "severity": "fixable",
  "message": "Function 'get_user' failed with KeyError: 'User 999 not found in database'",
  "suggestion": "The requested resource does not exist. Verify the identifier or use a search function first."
}
```

> **Warning:** Never include database connection strings, API keys, file system paths, or internal IP addresses in error messages sent to the model. Sanitize all error output.

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Use Anthropic's `is_error` flag when available | Gives the model an explicit signal that the result is an error |
| Include a `suggestion` field | Guides the model toward a fix instead of generic retries |
| Categorize errors consistently | Models learn patterns from consistent error structures |
| Report partial successes separately | Don't throw away valid results because some items failed |
| Sanitize exception messages | Raw tracebacks waste tokens and may leak sensitive data |
| Include received vs. expected values | For validation errors, show what was wrong and what's right |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Returning raw tracebacks as function results | Convert exceptions to structured error objects with `safe_error_from_exception()` |
| Using generic "Error occurred" messages | Be specific: what failed, why, and how to fix it |
| Not setting `is_error: true` for Anthropic | Without this flag, Claude may treat error text as valid data |
| Including sensitive data in error messages | Sanitize connection strings, keys, and internal paths |
| Treating all errors the same | Use severity levels so the model knows whether to retry or give up |
| Swallowing errors silently | Always return something — even `{"error": true, "message": "Unknown error"}` |

---

## Hands-on exercise

### Your task

Build an `ErrorHandler` class that wraps function execution with comprehensive error handling and formatting for all three providers.

### Requirements

1. Create an `ErrorHandler` class that accepts a `provider` parameter
2. Implement an `execute(func, args, call_id, func_name)` method
3. On success: return the provider-formatted success result
4. On failure: catch the exception, create a `FunctionError`, and return the provider-formatted error result
5. Support at least 3 exception types mapping to different error categories

### Expected result

```python
handler = ErrorHandler(provider="anthropic")

# Success case
result = handler.execute(
    func=lambda city: {"temp": 72},
    args={"city": "London"},
    call_id="toolu_123",
    func_name="get_weather"
)
# Returns: {"role": "user", "content": [{"type": "tool_result", ...}]}

# Failure case
result = handler.execute(
    func=lambda city: (_ for _ in ()).throw(ValueError("Invalid city")),
    args={"city": "Atlantis"},
    call_id="toolu_456",
    func_name="get_weather"
)
# Returns: {"role": "user", "content": [{"type": "tool_result", "is_error": True, ...}]}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Reuse the `FunctionError` and `ErrorCategory` classes from this lesson
- Use a try/except block inside `execute()`
- For the success case, use the result formatting from [Lesson 06-01](./01-result-format-structure.md)
- Map `ValueError` → VALIDATION, `KeyError` → NOT_FOUND, `TimeoutError` → TIMEOUT

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json


class ErrorHandler:
    """Wraps function execution with error handling for any provider."""
    
    EXCEPTION_MAP = {
        ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.FIXABLE),
        KeyError: (ErrorCategory.NOT_FOUND, ErrorSeverity.FIXABLE),
        FileNotFoundError: (ErrorCategory.NOT_FOUND, ErrorSeverity.FIXABLE),
        PermissionError: (ErrorCategory.PERMISSION, ErrorSeverity.PERMANENT),
        TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.RETRYABLE),
        ConnectionError: (ErrorCategory.UNAVAILABLE, ErrorSeverity.RETRYABLE),
    }
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
    
    def execute(
        self,
        func: callable,
        args: dict,
        call_id: str,
        func_name: str,
    ) -> dict:
        """Execute a function and return a provider-formatted result."""
        try:
            result = func(**args)
            return self._format_success(result, call_id, func_name)
        except Exception as e:
            error = self._exception_to_error(e, func_name)
            return self._format_error(error, call_id, func_name)
    
    def _exception_to_error(self, exc: Exception, func_name: str) -> FunctionError:
        """Convert an exception to a FunctionError."""
        category, severity = self.EXCEPTION_MAP.get(
            type(exc),
            (ErrorCategory.INTERNAL, ErrorSeverity.PERMANENT)
        )
        return FunctionError(
            category=category,
            severity=severity,
            message=f"{func_name}() failed: {str(exc)}",
            suggestion=_generate_suggestion(category, func_name),
        )
    
    def _format_success(self, result: any, call_id: str, func_name: str) -> dict:
        """Format a successful result for the provider."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result, default=str),
                }]
            }
        elif self.provider == "gemini":
            return {"function_name": func_name, "response": result}
        else:
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result, default=str),
            }
    
    def _format_error(self, error: FunctionError, call_id: str, func_name: str) -> dict:
        """Format an error result for the provider."""
        formatter = ErrorResultFormatter(self.provider)
        return formatter.format(error, call_id=call_id, function_name=func_name)


# Test it
handler = ErrorHandler(provider="anthropic")

# Success
result = handler.execute(
    func=lambda city: {"temp": 72, "unit": "F"},
    args={"city": "London"},
    call_id="toolu_123",
    func_name="get_weather",
)
print("SUCCESS:")
print(json.dumps(result, indent=2))

# Failure
def bad_weather(city):
    raise ValueError(f"Unknown city: {city}")

result = handler.execute(
    func=bad_weather,
    args={"city": "Atlantis"},
    call_id="toolu_456",
    func_name="get_weather",
)
print("\nERROR:")
print(json.dumps(result, indent=2))
```

**Output:**
```
SUCCESS:
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_123",
      "content": "{\"temp\": 72, \"unit\": \"F\"}"
    }
  ]
}

ERROR:
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_456",
      "is_error": true,
      "content": "get_weather() failed: Unknown city: Atlantis Suggestion: Check the parameters passed to get_weather() and ensure they match the expected types and ranges."
    }
  ]
}
```

</details>

### Bonus challenges

- [ ] Add retry logic for `RETRYABLE` severity errors (max 3 attempts with exponential backoff)
- [ ] Implement error rate tracking that switches to a fallback function after too many failures
- [ ] Add a `sanitize` option that strips file paths and connection strings from error messages

---

## Summary

✅ Anthropic provides `is_error: true` — the cleanest error signal; always use it

✅ OpenAI and Gemini require encoding error info in the output string or dictionary

✅ Categorize errors (validation, not_found, timeout, etc.) and include severity levels

✅ Write actionable error messages: what failed + what was received + what was expected + how to fix

✅ Report partial successes separately — don't discard valid results because of partial failures

✅ Never send raw stack traces or sensitive information in error results

**Next:** [No-Return-Value Handling →](./06-no-return-value-handling.md) — Handling functions that perform actions without returning data

---

[← Previous: Handling Large Results](./04-handling-large-results.md) | [Back to Lesson Overview](./00-returning-results.md)

<!-- 
Sources Consulted:
- Anthropic Tool Use (is_error flag): https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Gemini Function Calling Tutorial: https://ai.google.dev/gemini-api/docs/function-calling
- OpenAI Responses API Reference: https://platform.openai.com/docs/api-reference/responses
-->
