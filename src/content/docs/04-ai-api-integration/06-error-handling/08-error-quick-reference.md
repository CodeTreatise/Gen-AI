---
title: "Error Quick Reference"
---

# Error Quick Reference

## Introduction

This reference provides a fast lookup for API error codes, their meanings, and recommended handling strategies. Use this as a cheat sheet when debugging production issues.

---

## HTTP Status Codes

### Client Errors (4xx)

| Code | Name | Cause | Action |
|------|------|-------|--------|
| **400** | Bad Request | Malformed request, invalid JSON, missing required fields | Fix request format, validate input |
| **401** | Unauthorized | Invalid, expired, or missing API key | Refresh credentials, check key validity |
| **403** | Forbidden | Valid key but insufficient permissions | Check organization access, upgrade plan |
| **404** | Not Found | Invalid endpoint or model not available | Verify model name, check API version |
| **409** | Conflict | Request conflicts with current state | Retry with different parameters |
| **413** | Payload Too Large | Request body exceeds limits | Reduce input size, chunk content |
| **422** | Unprocessable Entity | Valid JSON but semantically invalid | Check parameter values and types |
| **429** | Too Many Requests | Rate limit exceeded | Implement backoff, wait for `retry-after` |

### Server Errors (5xx)

| Code | Name | Cause | Action |
|------|------|-------|--------|
| **500** | Internal Server Error | Server-side failure | Retry with exponential backoff |
| **502** | Bad Gateway | Upstream server error | Retry after brief delay |
| **503** | Service Unavailable | Server overloaded or maintenance | Retry with backoff, check status page |
| **504** | Gateway Timeout | Request took too long | Retry with smaller request |

---

## OpenAI Error Types

### Error Response Structure

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "context_length_exceeded",
    "message": "This model's maximum context length is 128000 tokens.",
    "param": "messages"
  }
}
```

### Error Type Reference

| Type | Description | Retryable |
|------|-------------|-----------|
| `invalid_request_error` | Request malformed or missing parameters | ❌ No |
| `authentication_error` | API key invalid or missing | ❌ No |
| `permission_error` | Key valid but lacks permissions | ❌ No |
| `not_found_error` | Resource (model, file) doesn't exist | ❌ No |
| `rate_limit_error` | Too many requests or tokens | ✅ Yes |
| `server_error` | OpenAI server issue | ✅ Yes |
| `api_error` | Generic API error | ✅ Maybe |
| `service_unavailable_error` | Temporary capacity issues | ✅ Yes |

### Common Error Codes

| Code | Type | Description |
|------|------|-------------|
| `invalid_api_key` | authentication | API key is malformed |
| `model_not_found` | not_found | Requested model doesn't exist |
| `context_length_exceeded` | invalid_request | Input too long for model |
| `content_policy_violation` | invalid_request | Content blocked by safety filters |
| `rate_limit_exceeded` | rate_limit | Request rate too high |
| `insufficient_quota` | rate_limit | Billing limit reached |
| `server_error` | server | Internal server failure |
| `timeout` | api | Request took too long |

---

## Anthropic Error Types

### Error Response Structure

```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded. Please retry after 30 seconds."
  }
}
```

### Error Type Reference

| Type | HTTP Code | Retryable |
|------|-----------|-----------|
| `invalid_request_error` | 400 | ❌ No |
| `authentication_error` | 401 | ❌ No |
| `permission_error` | 403 | ❌ No |
| `not_found_error` | 404 | ❌ No |
| `rate_limit_error` | 429 | ✅ Yes |
| `api_error` | 500 | ✅ Yes |
| `overloaded_error` | 529 | ✅ Yes |

---

## Python SDK Exceptions

### OpenAI

```python
from openai import (
    OpenAIError,          # Base exception
    APIError,             # API returned error
    APIConnectionError,   # Network issues
    APITimeoutError,      # Request timed out
    AuthenticationError,  # 401
    PermissionDeniedError,# 403
    NotFoundError,        # 404
    UnprocessableEntityError, # 422
    RateLimitError,       # 429
    InternalServerError,  # 500+
    BadRequestError,      # 400
)
```

### Anthropic

```python
from anthropic import (
    AnthropicError,       # Base exception
    APIError,             # API returned error
    APIConnectionError,   # Network issues
    APIStatusError,       # HTTP error status
    AuthenticationError,  # 401
    PermissionDeniedError,# 403
    NotFoundError,        # 404
    RateLimitError,       # 429
    InternalServerError,  # 500+
    BadRequestError,      # 400
)
```

---

## JavaScript SDK Errors

### OpenAI

```javascript
import {
    OpenAIError,
    APIError,
    APIConnectionError,
    APIUserAbortError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from 'openai';
```

### Error Properties

```javascript
catch (error) {
    if (error instanceof OpenAI.APIError) {
        console.log(error.status);    // HTTP status code
        console.log(error.message);   // Error message
        console.log(error.code);      // Error code
        console.log(error.type);      // Error type
        console.log(error.headers);   // Response headers
    }
}
```

---

## Quick Decision Tree

```
Error occurred?
├── Is it a 4xx error?
│   ├── 400 → Fix request format
│   ├── 401 → Check API key
│   ├── 403 → Check permissions/plan
│   ├── 404 → Check model/endpoint
│   ├── 429 → Wait and retry with backoff
│   └── Other → Check documentation
│
├── Is it a 5xx error?
│   ├── 500/502/503 → Retry with exponential backoff
│   ├── 504 → Reduce request size, retry
│   └── All → Check status page
│
└── Is it a network error?
    ├── Timeout → Increase timeout, retry
    ├── Connection refused → Check URL, retry
    └── SSL error → Check certificates
```

---

## Retry Strategy Quick Reference

| Error Type | Retry? | Backoff | Max Retries |
|------------|--------|---------|-------------|
| Rate limit (429) | ✅ Yes | Respect `retry-after` or exponential | 5-10 |
| Server error (5xx) | ✅ Yes | Exponential with jitter | 3-5 |
| Timeout | ✅ Yes | Linear or exponential | 2-3 |
| Auth error (401) | ❌ No | — | — |
| Bad request (400) | ❌ No | — | — |
| Not found (404) | ❌ No | — | — |
| Connection error | ✅ Yes | Exponential | 3-5 |

### Backoff Formula

```python
delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
jitter = random.uniform(0, delay * jitter_factor)
wait_time = delay + jitter
```

**Recommended defaults:**
- `initial_delay`: 1 second
- `backoff_factor`: 2
- `max_delay`: 60 seconds
- `jitter_factor`: 0.5

---

## User-Friendly Message Mapping

| Error Code | User Message |
|------------|--------------|
| `rate_limit_error` | "We're experiencing high demand. Please wait a moment." |
| `authentication_error` | "Session expired. Please log in again." |
| `context_length_exceeded` | "Your message is too long. Please shorten it." |
| `content_policy_violation` | "This request couldn't be processed. Please rephrase." |
| `server_error` | "We're having technical difficulties. Please try again." |
| `timeout` | "The request took too long. Please try a simpler query." |
| `model_not_found` | "This feature is temporarily unavailable." |

---

## Headers to Check

| Header | Purpose | Example |
|--------|---------|---------|
| `retry-after` | Seconds to wait before retry | `30` |
| `x-ratelimit-limit-requests` | Request limit per period | `10000` |
| `x-ratelimit-remaining-requests` | Requests left in period | `9500` |
| `x-ratelimit-limit-tokens` | Token limit per period | `1000000` |
| `x-ratelimit-remaining-tokens` | Tokens left in period | `950000` |
| `x-ratelimit-reset-requests` | Time until request limit resets | `1s` |
| `x-ratelimit-reset-tokens` | Time until token limit resets | `6m` |
| `x-request-id` | Unique request ID for support | `req_abc123` |

---

## Logging Template

```python
import logging
import json

def log_api_error(error, context: dict = None):
    """Structured error logging for production."""
    
    log_data = {
        "error_type": type(error).__name__,
        "error_code": getattr(error, "code", None),
        "status_code": getattr(error, "status_code", None),
        "message": str(error),
        "retryable": is_retryable(error),
    }
    
    # Add response details if available
    if hasattr(error, "response"):
        log_data["request_id"] = error.response.headers.get("x-request-id")
        log_data["retry_after"] = error.response.headers.get("retry-after")
    
    # Add custom context
    if context:
        log_data["context"] = context
    
    logging.error(json.dumps(log_data))
    
    return log_data


def is_retryable(error) -> bool:
    """Check if error is worth retrying."""
    
    retryable_types = {
        "RateLimitError",
        "InternalServerError", 
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError"
    }
    
    return type(error).__name__ in retryable_types
```

---

## Circuit Breaker Thresholds

| Scenario | Failure Threshold | Recovery Timeout | Success Threshold |
|----------|-------------------|------------------|-------------------|
| Production API | 5 | 60s | 2 |
| Development | 3 | 30s | 1 |
| High availability | 10 | 30s | 3 |
| Conservative | 3 | 120s | 3 |

---

## Model Context Limits

| Model | Max Context | Max Output |
|-------|-------------|------------|
| `gpt-4.1` | 1,047,576 | 32,768 |
| `gpt-4.1-mini` | 1,047,576 | 32,768 |
| `gpt-4.1-nano` | 1,047,576 | 32,768 |
| `o3` | 200,000 | 100,000 |
| `o4-mini` | 200,000 | 100,000 |
| `claude-sonnet-4` | 200,000 | 64,000 |
| `claude-opus-4` | 200,000 | 32,000 |

> **Note:** Token limits change. Always check official documentation.

---

## Status Page Links

| Provider | Status Page |
|----------|-------------|
| OpenAI | [status.openai.com](https://status.openai.com) |
| Anthropic | [status.anthropic.com](https://status.anthropic.com) |
| Azure OpenAI | [status.azure.com](https://status.azure.com) |
| Google AI | [status.cloud.google.com](https://status.cloud.google.com) |

---

## Summary

This reference covers:

✅ HTTP status codes and their meanings  
✅ Provider-specific error types  
✅ SDK exception classes  
✅ Retry strategies and backoff formulas  
✅ User-friendly message mappings  
✅ Rate limit headers  
✅ Logging best practices

**Back to:** [Error Handling Overview](./00-error-handling.md)

---

## Further Reading

- [OpenAI Error Codes](https://platform.openai.com/docs/guides/error-codes) — Official reference
- [Anthropic Errors](https://docs.anthropic.com/en/api/errors) — Error handling guide
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) — MDN reference

<!-- 
Sources Consulted:
- OpenAI Error Codes: https://platform.openai.com/docs/guides/error-codes
- Anthropic Errors: https://docs.anthropic.com/en/api/errors
- MDN HTTP Status: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
-->
