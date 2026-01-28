---
title: "Error Response Parsing"
---

# Error Response Parsing

## Introduction

When API requests fail, the response body contains detailed error information. Parsing this data correctly helps you understand what went wrong and how to fix it.

### What We'll Cover

- Error response structure
- Extracting error message and type
- Error codes and their meanings
- Provider-specific formats
- Building universal error parsers

### Prerequisites

- Common API errors knowledge
- JSON parsing fundamentals

---

## Error Response Structure

### OpenAI Error Format

```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

| Field | Description |
|-------|-------------|
| `message` | Human-readable error description |
| `type` | Error category (e.g., `invalid_request_error`) |
| `param` | Which parameter caused the error (if applicable) |
| `code` | Machine-readable error code |

### Anthropic Error Format

```json
{
  "type": "error",
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key"
  }
}
```

---

## Extracting Error Details

### From SDK Exceptions

```python
from openai import OpenAI, APIStatusError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "invalid", "content": "Hello"}]
    )
except APIStatusError as e:
    # Extract structured error details
    print(f"Status Code: {e.status_code}")
    print(f"Message: {e.message}")
    
    # Access response body
    if e.body:
        error_body = e.body
        if isinstance(error_body, dict):
            error_info = error_body.get("error", {})
            print(f"Type: {error_info.get('type')}")
            print(f"Code: {error_info.get('code')}")
            print(f"Param: {error_info.get('param')}")
```

### From Raw HTTP Response

```python
import httpx

def make_raw_request(api_key: str, payload: dict) -> dict:
    """Make raw HTTP request and parse error."""
    
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30.0
    )
    
    if response.status_code >= 400:
        error_data = response.json()
        
        return {
            "success": False,
            "status_code": response.status_code,
            "error": parse_error_response(error_data)
        }
    
    return {
        "success": True,
        "data": response.json()
    }


def parse_error_response(data: dict) -> dict:
    """Parse error response body."""
    
    error = data.get("error", {})
    
    return {
        "message": error.get("message", "Unknown error"),
        "type": error.get("type", "unknown"),
        "code": error.get("code"),
        "param": error.get("param")
    }
```

---

## Error Types

### OpenAI Error Types

| Type | Description |
|------|-------------|
| `invalid_request_error` | Request is malformed or invalid |
| `authentication_error` | API key issues |
| `permission_error` | Access denied |
| `rate_limit_error` | Too many requests |
| `server_error` | OpenAI server problem |
| `insufficient_quota` | Out of credits |
| `model_not_found` | Model doesn't exist |
| `context_length_exceeded` | Too many tokens |

### Error Codes

| Code | Meaning |
|------|---------|
| `invalid_api_key` | API key is wrong |
| `model_not_found` | Model doesn't exist |
| `context_length_exceeded` | Input too long |
| `rate_limit_exceeded` | Too many requests |
| `insufficient_quota` | No credits |
| `content_policy_violation` | Content filtered |
| `invalid_prompt` | Prompt has issues |

---

## Universal Error Parser

```python
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum

class ErrorType(Enum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    MODEL_NOT_FOUND = "model_not_found"
    SERVER = "server"
    NETWORK = "network"
    UNKNOWN = "unknown"

@dataclass
class ParsedError:
    type: ErrorType
    message: str
    code: Optional[str] = None
    param: Optional[str] = None
    status_code: Optional[int] = None
    raw: Optional[Any] = None
    retryable: bool = False


class ErrorParser:
    """Parse errors from any AI provider."""
    
    # Map error types/codes to our enum
    TYPE_MAP = {
        # OpenAI types
        "authentication_error": ErrorType.AUTHENTICATION,
        "invalid_request_error": ErrorType.VALIDATION,
        "rate_limit_error": ErrorType.RATE_LIMIT,
        "server_error": ErrorType.SERVER,
        "permission_error": ErrorType.PERMISSION,
        # OpenAI codes
        "invalid_api_key": ErrorType.AUTHENTICATION,
        "context_length_exceeded": ErrorType.CONTEXT_LENGTH,
        "rate_limit_exceeded": ErrorType.RATE_LIMIT,
        "insufficient_quota": ErrorType.QUOTA,
        "content_policy_violation": ErrorType.CONTENT_FILTER,
        "model_not_found": ErrorType.MODEL_NOT_FOUND,
        # Anthropic types
        "authentication_error": ErrorType.AUTHENTICATION,
        "permission_error": ErrorType.PERMISSION,
        "rate_limit_error": ErrorType.RATE_LIMIT,
        "overloaded_error": ErrorType.SERVER,
    }
    
    RETRYABLE_TYPES = {
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER,
        ErrorType.NETWORK
    }
    
    @classmethod
    def parse(cls, error: Exception) -> ParsedError:
        """Parse any API error into structured format."""
        
        # Try to get status code
        status_code = getattr(error, "status_code", None)
        
        # Try to get error body
        body = getattr(error, "body", None)
        if body is None:
            body = getattr(error, "response", None)
            if body and hasattr(body, "json"):
                try:
                    body = body.json()
                except:
                    body = None
        
        # Extract error info from body
        error_info = {}
        if isinstance(body, dict):
            error_info = body.get("error", body)
        
        # Determine error type
        raw_type = error_info.get("type", "")
        code = error_info.get("code", "")
        
        error_type = cls.TYPE_MAP.get(raw_type) or cls.TYPE_MAP.get(code)
        
        # Fall back to status code mapping
        if not error_type and status_code:
            error_type = {
                400: ErrorType.VALIDATION,
                401: ErrorType.AUTHENTICATION,
                403: ErrorType.PERMISSION,
                404: ErrorType.MODEL_NOT_FOUND,
                429: ErrorType.RATE_LIMIT,
                500: ErrorType.SERVER,
                503: ErrorType.SERVER,
            }.get(status_code, ErrorType.UNKNOWN)
        
        if not error_type:
            error_type = ErrorType.UNKNOWN
        
        return ParsedError(
            type=error_type,
            message=error_info.get("message", str(error)),
            code=code or None,
            param=error_info.get("param"),
            status_code=status_code,
            raw=body,
            retryable=error_type in cls.RETRYABLE_TYPES
        )
    
    @classmethod
    def from_response(cls, response_data: dict, status_code: int) -> ParsedError:
        """Parse error from raw response data."""
        error_info = response_data.get("error", response_data)
        
        raw_type = error_info.get("type", "")
        code = error_info.get("code", "")
        
        error_type = cls.TYPE_MAP.get(raw_type) or cls.TYPE_MAP.get(code)
        if not error_type:
            error_type = ErrorType.UNKNOWN
        
        return ParsedError(
            type=error_type,
            message=error_info.get("message", "Unknown error"),
            code=code or None,
            param=error_info.get("param"),
            status_code=status_code,
            raw=response_data,
            retryable=error_type in cls.RETRYABLE_TYPES
        )


# Usage
try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    parsed = ErrorParser.parse(e)
    
    print(f"Type: {parsed.type.value}")
    print(f"Message: {parsed.message}")
    print(f"Retryable: {parsed.retryable}")
    
    if parsed.param:
        print(f"Parameter: {parsed.param}")
```

---

## JavaScript Implementation

```javascript
const ErrorType = {
    AUTHENTICATION: 'authentication',
    PERMISSION: 'permission',
    VALIDATION: 'validation',
    RATE_LIMIT: 'rate_limit',
    QUOTA: 'quota',
    CONTENT_FILTER: 'content_filter',
    CONTEXT_LENGTH: 'context_length',
    MODEL_NOT_FOUND: 'model_not_found',
    SERVER: 'server',
    NETWORK: 'network',
    UNKNOWN: 'unknown'
};

class ErrorParser {
    static TYPE_MAP = {
        'authentication_error': ErrorType.AUTHENTICATION,
        'invalid_request_error': ErrorType.VALIDATION,
        'rate_limit_error': ErrorType.RATE_LIMIT,
        'server_error': ErrorType.SERVER,
        'invalid_api_key': ErrorType.AUTHENTICATION,
        'context_length_exceeded': ErrorType.CONTEXT_LENGTH,
        'insufficient_quota': ErrorType.QUOTA,
        'content_policy_violation': ErrorType.CONTENT_FILTER,
        'model_not_found': ErrorType.MODEL_NOT_FOUND
    };
    
    static RETRYABLE = new Set([
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER,
        ErrorType.NETWORK
    ]);
    
    static parse(error) {
        const status = error.status;
        const body = error.error || {};
        
        const rawType = body.type || '';
        const code = body.code || '';
        
        let errorType = this.TYPE_MAP[rawType] || this.TYPE_MAP[code];
        
        if (!errorType && status) {
            const statusMap = {
                400: ErrorType.VALIDATION,
                401: ErrorType.AUTHENTICATION,
                403: ErrorType.PERMISSION,
                404: ErrorType.MODEL_NOT_FOUND,
                429: ErrorType.RATE_LIMIT,
                500: ErrorType.SERVER,
                503: ErrorType.SERVER
            };
            errorType = statusMap[status] || ErrorType.UNKNOWN;
        }
        
        return {
            type: errorType || ErrorType.UNKNOWN,
            message: body.message || error.message || 'Unknown error',
            code: code || null,
            param: body.param || null,
            statusCode: status,
            retryable: this.RETRYABLE.has(errorType),
            raw: body
        };
    }
}

// Usage
try {
    const response = await openai.chat.completions.create({
        model: 'gpt-4.1',
        messages: [{ role: 'user', content: 'Hello' }]
    });
} catch (error) {
    const parsed = ErrorParser.parse(error);
    
    console.log(`Type: ${parsed.type}`);
    console.log(`Message: ${parsed.message}`);
    console.log(`Retryable: ${parsed.retryable}`);
    
    if (parsed.retryable) {
        // Implement retry logic
    }
}
```

---

## Extracting Retry Information

```python
def extract_retry_info(error: Exception) -> dict:
    """Extract retry-related information from error."""
    
    result = {
        "should_retry": False,
        "retry_after": None,
        "reset_at": None
    }
    
    # Check for response headers
    response = getattr(error, "response", None)
    if response and hasattr(response, "headers"):
        headers = response.headers
        
        # Retry-After header (seconds)
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                result["retry_after"] = int(retry_after)
                result["should_retry"] = True
            except ValueError:
                pass
        
        # X-RateLimit-Reset header (Unix timestamp)
        reset_at = headers.get("x-ratelimit-reset")
        if reset_at:
            try:
                result["reset_at"] = int(reset_at)
                result["should_retry"] = True
            except ValueError:
                pass
        
        # X-RateLimit-Remaining
        remaining = headers.get("x-ratelimit-remaining")
        if remaining:
            result["remaining_requests"] = int(remaining)
    
    # Check error type for retryability
    parsed = ErrorParser.parse(error)
    result["should_retry"] = result["should_retry"] or parsed.retryable
    
    return result


# Usage
try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    retry_info = extract_retry_info(e)
    
    if retry_info["should_retry"]:
        wait_time = retry_info["retry_after"] or 60
        print(f"Waiting {wait_time} seconds before retry...")
        time.sleep(wait_time)
```

---

## Error Logging

```python
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_api_error(
    error: Exception,
    request_context: dict = None
) -> str:
    """Log API error with context for debugging."""
    
    parsed = ErrorParser.parse(error)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": parsed.type.value,
        "error_code": parsed.code,
        "message": parsed.message,
        "status_code": parsed.status_code,
        "param": parsed.param,
        "retryable": parsed.retryable
    }
    
    if request_context:
        log_entry["context"] = {
            "model": request_context.get("model"),
            "endpoint": request_context.get("endpoint"),
            "user_id": request_context.get("user_id")
        }
    
    # Log at appropriate level
    if parsed.retryable:
        logger.warning(f"Retryable API error: {json.dumps(log_entry)}")
    else:
        logger.error(f"API error: {json.dumps(log_entry)}")
    
    return log_entry.get("error_type")


# Usage
try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    error_type = log_api_error(e, {
        "model": "gpt-4.1",
        "endpoint": "chat.completions",
        "user_id": "user_123"
    })
```

---

## Hands-on Exercise

### Your Task

Build an error parser that extracts all relevant information from API errors.

### Requirements

1. Parse error message and type
2. Extract error code and param
3. Determine if retryable
4. Check for retry-after header

### Expected Result

```python
parsed = parse_api_error(error)
# {
#   "type": "rate_limit",
#   "message": "Rate limit exceeded",
#   "code": "rate_limit_exceeded",
#   "status_code": 429,
#   "retryable": True,
#   "retry_after": 60
# }
```

<details>
<summary>💡 Hints</summary>

- Check error.body for error details
- Look at error.response.headers for retry info
- Map types to retryable status
</details>

<details>
<summary>✅ Solution</summary>

```python
def parse_api_error(error: Exception) -> dict:
    """Parse API error into structured dict."""
    
    result = {
        "type": "unknown",
        "message": str(error),
        "code": None,
        "param": None,
        "status_code": None,
        "retryable": False,
        "retry_after": None
    }
    
    # Get status code
    result["status_code"] = getattr(error, "status_code", None)
    
    # Parse error body
    body = getattr(error, "body", {})
    if isinstance(body, dict):
        error_info = body.get("error", body)
        result["message"] = error_info.get("message", result["message"])
        result["code"] = error_info.get("code")
        result["param"] = error_info.get("param")
        
        # Map type
        raw_type = error_info.get("type", "")
        type_map = {
            "rate_limit_error": "rate_limit",
            "authentication_error": "authentication",
            "invalid_request_error": "validation",
            "server_error": "server"
        }
        result["type"] = type_map.get(raw_type, result["type"])
    
    # Check retryability
    retryable_types = {"rate_limit", "server"}
    retryable_codes = {429, 500, 503}
    
    result["retryable"] = (
        result["type"] in retryable_types or
        result["status_code"] in retryable_codes
    )
    
    # Extract retry-after
    response = getattr(error, "response", None)
    if response and hasattr(response, "headers"):
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                result["retry_after"] = int(retry_after)
            except ValueError:
                pass
    
    return result


# Test
try:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    parsed = parse_api_error(e)
    print(json.dumps(parsed, indent=2))
```

</details>

---

## Summary

✅ Error responses contain type, message, code, and param fields  
✅ Parse error.body or error.response for detailed information  
✅ Map error types to determine retryability  
✅ Check response headers for retry-after guidance  
✅ Log errors with full context for debugging

**Next:** [Retry Strategies](./03-retry-strategies.md)

---

## Further Reading

- [OpenAI Error Types](https://platform.openai.com/docs/guides/error-codes/api-errors) — Official reference
- [HTTP Response Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers) — MDN
- [Retry-After Header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After) — MDN

<!-- 
Sources Consulted:
- OpenAI Error Codes: https://platform.openai.com/docs/guides/error-codes
- MDN HTTP Headers: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
-->
