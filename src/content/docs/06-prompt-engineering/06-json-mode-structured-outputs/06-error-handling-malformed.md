---
title: "Error Handling for Malformed JSON"
---

# Error Handling for Malformed JSON

## Introduction

Even with Structured Outputs, things can go wrong. Network issues, model refusals, validation failures—your code needs to handle them gracefully. We'll build robust error handling for every failure mode.

### What We'll Cover

- Common JSON generation failures
- Parse error detection and recovery
- API error handling
- Graceful degradation strategies
- Production-ready error patterns

### Prerequisites

- [Ensuring Valid JSON](./05-ensuring-valid-json.md)
- [Response Format Parameter](./03-response-format-parameter.md)

---

## Failure Modes

### Types of Failures

| Failure Type | Cause | Recovery |
|--------------|-------|----------|
| `JSONDecodeError` | Invalid syntax | Retry or extract |
| `ValidationError` | Schema mismatch | Retry with context |
| API Error | Network/rate limit | Retry with backoff |
| Refusal | Safety filters | Rephrase request |
| Truncation | max_tokens too low | Increase limit |
| Empty response | Edge cases | Provide defaults |

---

## Handling JSON Parse Errors

### Basic Error Handling

```python
import json
from typing import Any

def parse_json_response(content: str) -> tuple[dict | None, str | None]:
    """
    Parse JSON with detailed error information.
    
    Returns:
        Tuple of (parsed_data, error_message)
    """
    try:
        return json.loads(content), None
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse error at position {e.pos}: {e.msg}"
        return None, error_msg

# Usage
content = '{"name": "test", "value": }'
data, error = parse_json_response(content)

if error:
    print(f"Failed: {error}")
    # Failed: JSON parse error at position 27: Expecting value
```

### Extracting JSON from Mixed Content

```python
import re
import json

def extract_json(content: str) -> dict | None:
    """Extract JSON from mixed text/markdown content."""
    
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)```',
        r'```\s*([\s\S]*?)```',
        r'\{[\s\S]*\}',  # Match any {...}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            candidate = match.group(1) if '```' in pattern else match.group(0)
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue
    
    return None

# Test cases
test_inputs = [
    '{"valid": true}',
    'Here is the data: ```json\n{"valid": true}\n```',
    'The answer is {"valid": true} as shown.',
    'This has no JSON at all.'
]

for content in test_inputs:
    result = extract_json(content)
    print(f"Extracted: {result}")
```

---

## Handling Pydantic Validation Errors

### Detailed Validation Error Handling

```python
from pydantic import BaseModel, ValidationError, Field
from typing import List

class UserData(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str
    tags: List[str]

def validate_with_details(data: dict, model: type[BaseModel]) -> tuple[BaseModel | None, list[str]]:
    """
    Validate data against Pydantic model with detailed errors.
    
    Returns:
        Tuple of (validated_model, list_of_errors)
    """
    try:
        return model(**data), []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            msg = error['msg']
            errors.append(f"{field}: {msg}")
        return None, errors

# Test with invalid data
bad_data = {
    "name": "",
    "age": -5,
    "email": "test@example.com",
    "tags": "not-a-list"
}

result, errors = validate_with_details(bad_data, UserData)

if errors:
    print("Validation errors:")
    for err in errors:
        print(f"  - {err}")
    # Output:
    # - name: String should have at least 1 character
    # - age: Input should be greater than or equal to 0
    # - tags: Input should be a valid list
```

### Auto-Fixing Common Issues

```python
def attempt_auto_fix(data: dict, errors: list[str]) -> dict:
    """Attempt to auto-fix common validation issues."""
    
    fixed = data.copy()
    
    for error in errors:
        field, msg = error.split(': ', 1)
        
        if 'should be a valid list' in msg:
            # Convert single value to list
            value = fixed.get(field)
            if value is not None and not isinstance(value, list):
                fixed[field] = [value]
        
        elif 'should be a valid integer' in msg:
            # Try to convert string to int
            value = fixed.get(field)
            if isinstance(value, str):
                try:
                    fixed[field] = int(float(value))
                except ValueError:
                    pass
        
        elif 'should have at least' in msg:
            # Can't auto-fix empty required strings
            pass
    
    return fixed
```

---

## API Error Handling

### OpenAI API Errors

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import time

def call_with_retry(
    client: OpenAI,
    messages: list,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str | None:
    """Call OpenAI API with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            delay = base_delay * (2 ** attempt)
            print(f"Rate limited, waiting {delay}s...")
            time.sleep(delay)
            
        except APIConnectionError:
            delay = base_delay * (2 ** attempt)
            print(f"Connection error, retrying in {delay}s...")
            time.sleep(delay)
            
        except APIError as e:
            print(f"API error: {e}")
            if e.status_code >= 500:
                # Server error - retry
                time.sleep(base_delay)
            else:
                # Client error - don't retry
                return None
    
    return None
```

### Comprehensive Error Types

```python
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    InternalServerError
)

def handle_api_call(client: OpenAI, messages: list) -> dict:
    """Handle all API error types with appropriate responses."""
    
    result = {
        "success": False,
        "data": None,
        "error": None,
        "retryable": False
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result["success"] = True
        result["data"] = response.choices[0].message.content
        
    except AuthenticationError:
        result["error"] = "Invalid API key"
        result["retryable"] = False
        
    except BadRequestError as e:
        result["error"] = f"Invalid request: {e.message}"
        result["retryable"] = False
        
    except NotFoundError:
        result["error"] = "Model not found"
        result["retryable"] = False
        
    except RateLimitError:
        result["error"] = "Rate limit exceeded"
        result["retryable"] = True
        
    except APIConnectionError:
        result["error"] = "Network connection failed"
        result["retryable"] = True
        
    except InternalServerError:
        result["error"] = "OpenAI server error"
        result["retryable"] = True
        
    except APIError as e:
        result["error"] = f"Unknown API error: {e}"
        result["retryable"] = e.status_code >= 500
    
    return result
```

---

## Handling Refusals

### Detecting Model Refusals

```python
from openai import OpenAI
from pydantic import BaseModel

class ExtractedData(BaseModel):
    content: str
    category: str

def extract_with_refusal_handling(
    client: OpenAI,
    text: str
) -> ExtractedData | str | None:
    """
    Extract data, handling refusals appropriately.
    
    Returns:
        ExtractedData on success
        String message on refusal
        None on other failures
    """
    
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "user", "content": f"Extract info from: {text}"}
        ],
        text_format=ExtractedData
    )
    
    # Check for refusal
    if hasattr(response, 'refusal') and response.refusal:
        return f"Model refused: {response.refusal}"
    
    return response.output_parsed

# Usage
result = extract_with_refusal_handling(client, "Some text...")

if isinstance(result, str):
    print(f"Refusal: {result}")
elif result is None:
    print("Extraction failed")
else:
    print(f"Extracted: {result.content}")
```

### Handling Refusals in Chat Completions

```python
def parse_with_refusal_check(client: OpenAI, messages: list, schema: type) -> dict:
    """Parse response with refusal detection."""
    
    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=schema
    )
    
    message = response.choices[0].message
    
    return {
        "parsed": message.parsed,
        "refusal": message.refusal,
        "finish_reason": response.choices[0].finish_reason
    }
```

---

## Graceful Degradation

### Fallback Chain

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class ExtractionResult:
    data: Any
    method: str
    confidence: float

def extract_with_fallbacks(
    client: OpenAI,
    text: str,
    schema: type
) -> ExtractionResult | None:
    """Try multiple extraction methods in order of reliability."""
    
    # Method 1: Structured Outputs (highest reliability)
    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[{"role": "user", "content": text}],
            text_format=schema
        )
        if not (hasattr(response, 'refusal') and response.refusal):
            return ExtractionResult(
                data=response.output_parsed,
                method="structured_outputs",
                confidence=0.99
            )
    except Exception:
        pass
    
    # Method 2: JSON Mode with validation
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Extract as JSON: {schema.model_json_schema()}"},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        validated = schema(**data)
        return ExtractionResult(
            data=validated,
            method="json_mode",
            confidence=0.90
        )
    except Exception:
        pass
    
    # Method 3: Plain text with JSON extraction
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Extract as JSON from: {text}"}
            ]
        )
        data = extract_json(response.choices[0].message.content)
        if data:
            validated = schema(**data)
            return ExtractionResult(
                data=validated,
                method="text_extraction",
                confidence=0.70
            )
    except Exception:
        pass
    
    return None
```

### Partial Results

```python
from pydantic import BaseModel
from typing import Optional

class PartialExtraction(BaseModel):
    """Schema that accepts partial results."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    
    @property
    def completeness(self) -> float:
        """Calculate how complete the extraction is."""
        fields = [self.name, self.email, self.phone, self.company]
        filled = sum(1 for f in fields if f is not None)
        return filled / len(fields)

def extract_with_partial_results(
    client: OpenAI,
    text: str,
    min_completeness: float = 0.5
) -> PartialExtraction | None:
    """Extract data, accepting partial results above threshold."""
    
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "Extract contact info. Use null for missing fields."},
            {"role": "user", "content": text}
        ],
        text_format=PartialExtraction
    )
    
    result = response.output_parsed
    
    if result.completeness >= min_completeness:
        return result
    
    print(f"Extraction only {result.completeness:.0%} complete")
    return None
```

---

## Production Error Patterns

### Comprehensive Error Handler

```python
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Generic

class ErrorType(Enum):
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"
    REFUSAL = "refusal"
    TRUNCATION = "truncation"
    UNKNOWN = "unknown"

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Result type with error handling."""
    success: bool
    data: T | None
    error_type: ErrorType | None
    error_message: str | None
    
    @classmethod
    def ok(cls, data: T) -> "Result[T]":
        return cls(success=True, data=data, error_type=None, error_message=None)
    
    @classmethod
    def fail(cls, error_type: ErrorType, message: str) -> "Result[T]":
        return cls(success=False, data=None, error_type=error_type, error_message=message)

def safe_extract(
    client: OpenAI,
    text: str,
    schema: type[T]
) -> Result[T]:
    """Extract with comprehensive error handling."""
    
    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[{"role": "user", "content": text}],
            text_format=schema
        )
        
        # Check refusal
        if hasattr(response, 'refusal') and response.refusal:
            return Result.fail(ErrorType.REFUSAL, response.refusal)
        
        return Result.ok(response.output_parsed)
        
    except json.JSONDecodeError as e:
        return Result.fail(ErrorType.PARSE_ERROR, str(e))
        
    except ValidationError as e:
        return Result.fail(ErrorType.VALIDATION_ERROR, str(e))
        
    except APIError as e:
        return Result.fail(ErrorType.API_ERROR, str(e))
        
    except Exception as e:
        return Result.fail(ErrorType.UNKNOWN, str(e))

# Usage
result = safe_extract(client, "some text", MySchema)

if result.success:
    process_data(result.data)
elif result.error_type == ErrorType.REFUSAL:
    log_refusal(result.error_message)
elif result.error_type in [ErrorType.API_ERROR]:
    schedule_retry()
else:
    log_error(result.error_type, result.error_message)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use typed result objects | Clear success/failure handling |
| Categorize error types | Different errors need different responses |
| Implement retry with backoff | Transient errors self-resolve |
| Log all failures | Debug and improve over time |
| Accept partial results | Better than nothing |
| Have fallback methods | Graceful degradation |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Catching generic `Exception` | Handle specific error types |
| No retry for transient errors | Implement exponential backoff |
| Ignoring refusals | Check and handle appropriately |
| Silent failures | Log with context for debugging |
| All-or-nothing validation | Accept partial results when appropriate |

---

## Hands-on Exercise

### Your Task

Build a robust extraction function with comprehensive error handling.

### Requirements

1. Handle JSON parse errors
2. Handle validation errors with specific messages
3. Detect and report refusals
4. Implement retry for transient API errors
5. Return a typed Result object

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the Result dataclass pattern
- Separate retryable from non-retryable errors
- Include the original error message for debugging

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError, Field
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Generic, Optional
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Error types
class ErrorType(Enum):
    SUCCESS = "success"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    REFUSAL = "refusal"
    MAX_RETRIES = "max_retries_exceeded"

# Generic result type
T = TypeVar('T')

@dataclass
class ExtractionResult(Generic[T]):
    success: bool
    data: Optional[T]
    error_type: ErrorType
    error_message: Optional[str]
    attempts: int

# Schema
class ArticleExtraction(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=10)
    topics: list[str] = Field(min_length=1)
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")

def robust_extract(
    client: OpenAI,
    text: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> ExtractionResult[ArticleExtraction]:
    """
    Extract article info with comprehensive error handling.
    """
    
    last_error_type = ErrorType.SUCCESS
    last_error_msg = None
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}")
        
        try:
            response = client.responses.parse(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": "Extract article information. Use 'positive', 'negative', or 'neutral' for sentiment."
                    },
                    {"role": "user", "content": text}
                ],
                text_format=ArticleExtraction
            )
            
            # Check for refusal
            if hasattr(response, 'refusal') and response.refusal:
                logger.warning(f"Model refused: {response.refusal}")
                return ExtractionResult(
                    success=False,
                    data=None,
                    error_type=ErrorType.REFUSAL,
                    error_message=response.refusal,
                    attempts=attempt
                )
            
            # Success!
            logger.info("Extraction successful")
            return ExtractionResult(
                success=True,
                data=response.output_parsed,
                error_type=ErrorType.SUCCESS,
                error_message=None,
                attempts=attempt
            )
            
        except RateLimitError as e:
            last_error_type = ErrorType.RATE_LIMIT
            last_error_msg = str(e)
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Rate limited, waiting {delay}s")
            time.sleep(delay)
            continue
            
        except APIConnectionError as e:
            last_error_type = ErrorType.CONNECTION_ERROR
            last_error_msg = str(e)
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Connection error, retrying in {delay}s")
            time.sleep(delay)
            continue
            
        except ValidationError as e:
            # Validation errors won't improve with retry
            errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
            logger.error(f"Validation failed: {errors}")
            return ExtractionResult(
                success=False,
                data=None,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message="; ".join(errors),
                attempts=attempt
            )
            
        except json.JSONDecodeError as e:
            last_error_type = ErrorType.PARSE_ERROR
            last_error_msg = f"Position {e.pos}: {e.msg}"
            logger.warning(f"JSON parse error: {last_error_msg}")
            # Might work on retry
            continue
            
        except APIError as e:
            if e.status_code >= 500:
                # Server error - retry
                last_error_type = ErrorType.API_ERROR
                last_error_msg = str(e)
                delay = base_delay
                logger.warning(f"Server error, retrying in {delay}s")
                time.sleep(delay)
                continue
            else:
                # Client error - don't retry
                logger.error(f"Client error: {e}")
                return ExtractionResult(
                    success=False,
                    data=None,
                    error_type=ErrorType.API_ERROR,
                    error_message=str(e),
                    attempts=attempt
                )
    
    # All retries exhausted
    logger.error("Max retries exceeded")
    return ExtractionResult(
        success=False,
        data=None,
        error_type=ErrorType.MAX_RETRIES,
        error_message=f"Failed after {max_retries} attempts. Last error: {last_error_msg}",
        attempts=max_retries
    )

# Usage
if __name__ == "__main__":
    client = OpenAI()
    
    article = """
    New AI Breakthrough Announced
    
    Researchers have developed a new method for training language models
    that reduces computational costs by 40%. The technique shows promise
    for making AI more accessible to smaller organizations.
    
    Topics covered: AI, machine learning, efficiency, research
    """
    
    result = robust_extract(client, article)
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Error Type: {result.error_type.value}")
    
    if result.success:
        print(f"  Title: {result.data.title}")
        print(f"  Summary: {result.data.summary[:50]}...")
        print(f"  Topics: {result.data.topics}")
        print(f"  Sentiment: {result.data.sentiment}")
    else:
        print(f"  Error: {result.error_message}")
```

</details>

### Bonus Challenge

- [ ] Add circuit breaker pattern for repeated failures
- [ ] Implement dead letter queue for failed extractions

---

## Summary

✅ **Categorize error types** for appropriate handling

✅ **Retry transient errors** with exponential backoff

✅ **Handle refusals** as a special case

✅ **Use typed result objects** for clear error propagation

✅ **Accept partial results** when full extraction fails

**Next:** [JSON Mode vs Prompting Comparison](./07-json-mode-vs-prompting.md)

---

## Further Reading

- [OpenAI Error Handling](https://platform.openai.com/docs/guides/error-codes)
- [Pydantic Error Handling](https://docs.pydantic.dev/latest/concepts/models/#error-handling)
- [Python Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)

---

<!-- 
Sources Consulted:
- OpenAI API documentation: https://platform.openai.com/docs/guides/error-codes
- Pydantic documentation: https://docs.pydantic.dev/
-->
