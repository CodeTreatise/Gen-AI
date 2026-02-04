---
title: "Ensuring Valid JSON"
---

# Ensuring Valid JSON

## Introduction

Getting valid JSON from an LLM requires a multi-layered approach. Even with API-level guarantees, edge cases exist. We'll explore the guarantee layers and how to build robust JSON generation pipelines.

### What We'll Cover

- API-level JSON guarantees
- Prompt techniques for JSON compliance
- Validation layers
- Handling edge cases
- Defense-in-depth strategies

### Prerequisites

- [JSON Mode in API Calls](./01-json-mode-api.md)
- [Response Format Parameter](./03-response-format-parameter.md)

---

## Guarantee Layers

### The JSON Reliability Pyramid

```
┌─────────────────────────────────┐
│   Structured Outputs (Strict)   │  ← Strongest: Schema guaranteed
├─────────────────────────────────┤
│        JSON Mode               │  ← Valid JSON, no schema
├─────────────────────────────────┤
│     Prompt Engineering         │  ← Best effort, may fail
├─────────────────────────────────┤
│    Post-processing / Retry     │  ← Fallback handling
└─────────────────────────────────┘
```

| Layer | Reliability | Use Case |
|-------|-------------|----------|
| Structured Outputs | 99.9%+ | Production systems |
| JSON Mode | ~99% | When schema flexibility needed |
| Prompt-only | ~80-95% | Older models, basic tasks |
| Retry/fallback | Catch remaining | All systems |

---

## API-Level Guarantees

### OpenAI Structured Outputs

```python
from openai import OpenAI
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

client = OpenAI()

response = client.responses.parse(
    model="gpt-4o",
    input=[{"role": "user", "content": "What is 2+2?"}],
    text_format=Response
)

# Guaranteed: valid JSON matching schema
# Exception: refusal (safety) or truncation (length)
result = response.output_parsed
```

### What's Guaranteed

| Feature | JSON Mode | Structured Outputs |
|---------|-----------|-------------------|
| Valid JSON syntax | ✅ | ✅ |
| All fields present | ❌ | ✅ |
| Correct field types | ❌ | ✅ |
| Enum value compliance | ❌ | ✅ |
| No extra fields | ❌ | ✅ (with additionalProperties: false) |

### What's NOT Guaranteed

| Risk | Cause | Mitigation |
|------|-------|------------|
| Truncated output | `max_tokens` too low | Set appropriate limit |
| Empty objects | Edge cases | Validate content, not just syntax |
| Refusal | Safety filters | Check for refusal field |
| Semantic errors | Model misunderstanding | Better prompts, examples |

---

## Prompt Techniques

When API-level guarantees aren't available or sufficient, use prompt engineering:

### Clear Format Instructions

```python
system_prompt = """You are a data extraction assistant.

IMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown.

Output format:
{
    "name": "extracted name",
    "value": 123
}"""
```

### Schema in System Message

```python
system_prompt = """Extract information and respond in this exact JSON format:

{
    "title": "string - the main title",
    "items": ["string array - list of items"],
    "count": "integer - number of items"
}

Rules:
- Output ONLY the JSON object
- No markdown code blocks
- No explanatory text
- All fields are required"""
```

### Few-Shot Examples

```python
messages = [
    {
        "role": "system",
        "content": "Extract product info as JSON."
    },
    {
        "role": "user",
        "content": "Blue widget, $29.99, electronics"
    },
    {
        "role": "assistant",
        "content": '{"name": "Blue widget", "price": 29.99, "category": "electronics"}'
    },
    {
        "role": "user",
        "content": "Red gadget, $15.50, toys"
    }
    # Model learns the exact format from the example
]
```

---

## Validation Layers

### Layer 1: JSON Parsing

```python
import json

def safe_parse_json(content: str) -> dict | None:
    """Attempt to parse JSON, return None on failure."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None

response_text = '{"name": "test", "value": 42}'
data = safe_parse_json(response_text)
```

### Layer 2: Schema Validation

```python
from pydantic import BaseModel, ValidationError

class ExpectedSchema(BaseModel):
    name: str
    value: int
    tags: list[str]

def validate_response(data: dict) -> ExpectedSchema | None:
    """Validate dict against schema."""
    try:
        return ExpectedSchema(**data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None

# Usage
raw_data = {"name": "test", "value": 42, "tags": ["a", "b"]}
validated = validate_response(raw_data)
```

### Layer 3: Business Logic Validation

```python
def validate_business_rules(data: ExpectedSchema) -> bool:
    """Apply business-specific validation."""
    
    # Check required values
    if not data.name.strip():
        return False
    
    # Check reasonable ranges
    if data.value < 0 or data.value > 1000000:
        return False
    
    # Check tag count
    if len(data.tags) > 20:
        return False
    
    return True
```

### Combined Validation Pipeline

```python
def process_llm_response(content: str) -> ExpectedSchema | None:
    """Complete validation pipeline."""
    
    # Layer 1: Parse JSON
    data = safe_parse_json(content)
    if data is None:
        return None
    
    # Layer 2: Validate schema
    validated = validate_response(data)
    if validated is None:
        return None
    
    # Layer 3: Business rules
    if not validate_business_rules(validated):
        return None
    
    return validated
```

---

## Handling Edge Cases

### Truncated JSON

```python
def handle_truncated_json(content: str) -> dict | None:
    """Attempt to salvage truncated JSON."""
    
    # Try direct parse first
    data = safe_parse_json(content)
    if data:
        return data
    
    # Count brackets to detect truncation
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')
    
    # Simple fix: add missing closing brackets
    fixed = content
    fixed += ']' * open_brackets
    fixed += '}' * open_braces
    
    return safe_parse_json(fixed)

# Example
truncated = '{"items": ["a", "b", "c"'
result = handle_truncated_json(truncated)
# Returns: {"items": ["a", "b", "c"]}
```

> **Warning:** This is a heuristic. In production, prefer retrying with higher `max_tokens`.

### Markdown Code Blocks

Models sometimes wrap JSON in markdown:

```python
import re

def extract_json_from_markdown(content: str) -> str:
    """Extract JSON from markdown code blocks."""
    
    # Pattern for ```json ... ``` blocks
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    
    if match:
        return match.group(1).strip()
    
    return content.strip()

# Usage
response = """Here's the data:

```json
{"name": "test", "value": 42}
```
"""

clean_json = extract_json_from_markdown(response)
# Returns: '{"name": "test", "value": 42}'
```

### Empty or Default Values

```python
def validate_non_empty(data: dict, required_fields: list[str]) -> bool:
    """Check that required fields have meaningful values."""
    
    for field in required_fields:
        value = data.get(field)
        
        # Check for None or empty
        if value is None:
            return False
        
        # Check for empty strings
        if isinstance(value, str) and not value.strip():
            return False
        
        # Check for empty lists
        if isinstance(value, list) and len(value) == 0:
            return False
    
    return True

# Usage
data = {"name": "Product", "description": "", "tags": []}
is_valid = validate_non_empty(data, ["name", "description"])  # False
```

---

## Retry Strategies

### Simple Retry with Clarification

```python
from openai import OpenAI
import json

client = OpenAI()

def get_json_with_retry(prompt: str, max_retries: int = 3) -> dict | None:
    """Retry JSON generation with error feedback."""
    
    messages = [
        {"role": "system", "content": "Respond only in valid JSON format."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Add error context for next attempt
            messages.append({
                "role": "assistant",
                "content": content
            })
            messages.append({
                "role": "user",
                "content": f"That was not valid JSON. Error: {e}. Please try again with valid JSON only."
            })
    
    return None
```

### Retry with Schema Reminder

```python
def get_structured_with_retry(
    prompt: str,
    schema: type[BaseModel],
    max_retries: int = 2
) -> BaseModel | None:
    """Retry with schema validation feedback."""
    
    for attempt in range(max_retries):
        response = client.responses.parse(
            model="gpt-4o",
            input=[{"role": "user", "content": prompt}],
            text_format=schema
        )
        
        # Check for refusal
        if hasattr(response, 'refusal') and response.refusal:
            print(f"Model refused: {response.refusal}")
            return None
        
        result = response.output_parsed
        
        # Validate with business rules
        if validate_business_rules(result):
            return result
        
        # Enhance prompt for retry
        prompt = f"""The previous response didn't meet requirements.
Original request: {prompt}

Please ensure:
- All fields have meaningful values
- Values are within expected ranges
- Response is complete"""
    
    return None
```

---

## Defense-in-Depth Strategy

### Complete Pipeline

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Type
import json

T = TypeVar('T', bound=BaseModel)

class JSONPipeline:
    """Robust JSON generation pipeline."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate(
        self,
        prompt: str,
        schema: Type[T],
        max_retries: int = 2
    ) -> T | None:
        """Generate and validate structured output."""
        
        # Layer 1: Use Structured Outputs
        for attempt in range(max_retries):
            try:
                response = self.client.responses.parse(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "system",
                            "content": "Extract information accurately and completely."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    text_format=schema
                )
                
                # Layer 2: Check for refusal
                if hasattr(response, 'refusal') and response.refusal:
                    print(f"Refusal: {response.refusal}")
                    return None
                
                result = response.output_parsed
                
                # Layer 3: Additional validation
                if self._validate(result):
                    return result
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        return None
    
    def _validate(self, result: BaseModel) -> bool:
        """Apply custom validation rules."""
        # Override in subclass for specific rules
        return True

# Usage
pipeline = JSONPipeline(OpenAI())

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str

result = pipeline.generate(
    "Blue widget, $29.99, electronics category",
    ProductInfo
)

if result:
    print(f"Got: {result.name} - ${result.price}")
else:
    print("Failed to generate valid response")
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Structured Outputs | Strongest guarantee |
| Set adequate max_tokens | Prevent truncation |
| Always check for refusals | Safety filter bypass |
| Validate beyond syntax | Empty values are valid JSON |
| Have fallback strategies | Handle edge cases gracefully |
| Log failures | Improve prompts over time |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Trusting JSON mode for schema | Use Structured Outputs |
| Low max_tokens | Calculate expected output size |
| No retry mechanism | Implement retries with feedback |
| Ignoring empty values | Validate content, not just syntax |
| No logging | Track failure patterns |

---

## Hands-on Exercise

### Your Task

Build a robust JSON extraction function with all validation layers.

### Requirements

1. Use Structured Outputs with a Pydantic model
2. Handle refusals appropriately
3. Validate that extracted values are non-empty
4. Implement one retry on validation failure
5. Return `None` with appropriate logging on failure

<details>
<summary>💡 Hints (click to expand)</summary>

- Check `response.refusal` before accessing `output_parsed`
- Use a validation function that checks string length
- Add context to the retry prompt

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContactExtraction(BaseModel):
    """Contact information extraction schema."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    company: Optional[str] = Field(default=None, description="Company name if mentioned")
    role: Optional[str] = Field(default=None, description="Job title if mentioned")

def validate_contact(contact: ContactExtraction) -> bool:
    """Validate extracted contact has meaningful data."""
    
    # Name must be non-empty
    if not contact.name or len(contact.name.strip()) < 2:
        logger.warning("Invalid name: too short or empty")
        return False
    
    # Email must look valid
    if not contact.email or '@' not in contact.email:
        logger.warning("Invalid email format")
        return False
    
    return True

def extract_contact(
    client: OpenAI,
    text: str,
    max_retries: int = 2
) -> Optional[ContactExtraction]:
    """
    Extract contact information with full validation pipeline.
    
    Returns:
        ContactExtraction if successful, None if extraction failed.
    """
    
    prompt = text
    
    for attempt in range(max_retries):
        logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")
        
        try:
            response = client.responses.parse(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": "Extract contact information from the provided text. If information is not present, use null for optional fields."
                    },
                    {"role": "user", "content": prompt}
                ],
                text_format=ContactExtraction
            )
            
            # Check for refusal
            if hasattr(response, 'refusal') and response.refusal:
                logger.error(f"Model refused: {response.refusal}")
                return None
            
            contact = response.output_parsed
            
            # Validate extraction
            if validate_contact(contact):
                logger.info(f"Successfully extracted: {contact.name}")
                return contact
            
            # Prepare enhanced prompt for retry
            prompt = f"""Please extract contact info more carefully from:
{text}

Requirements:
- Name must be a real person's name (at least 2 characters)
- Email must be a valid email address
- Extract exactly what's in the text"""
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            continue
    
    logger.error("All extraction attempts failed")
    return None

# Usage example
if __name__ == "__main__":
    client = OpenAI()
    
    # Test cases
    test_texts = [
        "Contact John Smith at john.smith@example.com, he's the CTO at TechCorp",
        "Email me: jane@company.io",
        "Call 555-1234",  # No email - should fail
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        result = extract_contact(client, text)
        if result:
            print(f"  Name: {result.name}")
            print(f"  Email: {result.email}")
            print(f"  Company: {result.company}")
            print(f"  Role: {result.role}")
        else:
            print("  Extraction failed")
```

**Expected output:**
```
Input: Contact John Smith at john.smith@example.com, he's the CTO at TechCorp
  Name: John Smith
  Email: john.smith@example.com
  Company: TechCorp
  Role: CTO

Input: Email me: jane@company.io
  Name: Jane
  Email: jane@company.io
  Company: None
  Role: None

Input: Call 555-1234
  Extraction failed
```

</details>

### Bonus Challenge

- [ ] Add exponential backoff between retries
- [ ] Track success/failure metrics

---

## Summary

✅ **Structured Outputs** provide the strongest JSON guarantees

✅ **Multiple validation layers** catch different failure modes

✅ **Prompt techniques** improve reliability when needed

✅ **Retry strategies** handle transient failures

✅ **Log and monitor** to continuously improve

**Next:** [Error Handling for Malformed JSON](./06-error-handling-malformed.md)

---

## Further Reading

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Validation](https://docs.pydantic.dev/latest/concepts/validators/)
- [JSON Schema Validation](https://json-schema.org/understanding-json-schema/reference/index.html)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Pydantic documentation: https://docs.pydantic.dev/
-->
