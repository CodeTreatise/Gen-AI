---
title: "Handling Format Compliance"
---

# Handling Format Compliance

## Introduction

Even with the best prompts and schemas, models sometimes produce malformed outputs. Production applications need strategies to detect, handle, and recover from format errors. This lesson covers validation, retry strategies, partial parsing, and fallback approaches.

> **🤖 AI Context:** API-level Structured Outputs (like OpenAI's `response_format`) guarantee valid JSON, but schema compliance and semantic correctness still need verification. Prompt-based formatting always needs validation.

### What We'll Cover

- Validating model outputs
- Retry strategies for malformed responses
- Partial parsing and recovery
- Fallback strategies
- Building robust output pipelines

### Prerequisites

- [Schema Definition in Prompts](./06-schema-definition.md)

---

## Why Validation Matters

| Issue | Impact |
|-------|--------|
| Malformed JSON | Application crash |
| Missing fields | NullPointerException |
| Wrong types | Type errors downstream |
| Invalid values | Business logic failures |
| Extra content | Parsing failures |

Even small format errors can break production systems.

---

## Validation Strategies

### JSON Syntax Validation

```python
import json

def validate_json(response_text):
    """Check if response is valid JSON"""
    try:
        data = json.loads(response_text)
        return True, data
    except json.JSONDecodeError as e:
        return False, str(e)
```

### Schema Validation with Pydantic

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List

class TicketClassification(BaseModel):
    category: str = Field(..., pattern="^(billing|technical|account|general)$")
    priority: str = Field(..., pattern="^(low|medium|high|critical)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, max_length=500)

def validate_response(response_text):
    """Validate response against schema"""
    try:
        data = json.loads(response_text)
        result = TicketClassification(**data)
        return True, result
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except ValidationError as e:
        return False, f"Schema validation failed: {e}"
```

### TypeScript/Zod Validation

```typescript
import { z } from 'zod';

const TicketSchema = z.object({
  category: z.enum(['billing', 'technical', 'account', 'general']),
  priority: z.enum(['low', 'medium', 'high', 'critical']),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().min(10).max(500)
});

function validateResponse(responseText: string) {
  try {
    const data = JSON.parse(responseText);
    const result = TicketSchema.parse(data);
    return { success: true, data: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}
```

---

## Retry Strategies

### Simple Retry

```python
def get_structured_response(prompt, max_retries=3):
    """Retry on format errors"""
    for attempt in range(max_retries):
        response = call_llm(prompt)
        
        valid, result = validate_response(response)
        if valid:
            return result
            
        print(f"Attempt {attempt + 1} failed: {result}")
    
    raise ValueError(f"Failed after {max_retries} attempts")
```

### Retry with Feedback

Include the error in the retry prompt:

```python
def get_structured_response_with_feedback(prompt, max_retries=3):
    """Retry with error feedback"""
    last_response = None
    last_error = None
    
    for attempt in range(max_retries):
        if attempt == 0:
            current_prompt = prompt
        else:
            # Include error feedback
            current_prompt = f"""{prompt}

IMPORTANT: Your previous response had a format error:
{last_error}

Your previous response was:
{last_response[:500]}...

Please correct the format and try again."""
        
        last_response = call_llm(current_prompt)
        valid, result = validate_response(last_response)
        
        if valid:
            return result
        last_error = result
    
    raise ValueError(f"Failed after {max_retries} attempts")
```

### Exponential Backoff

```python
import time

def get_with_backoff(prompt, max_retries=3, base_delay=1):
    """Retry with exponential backoff"""
    for attempt in range(max_retries):
        response = call_llm(prompt)
        valid, result = validate_response(response)
        
        if valid:
            return result
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"Retrying in {delay}s...")
            time.sleep(delay)
    
    raise ValueError("Max retries exceeded")
```

---

## Partial Parsing

When full parsing fails, try to extract what you can:

### Extract JSON from Mixed Content

```python
import re

def extract_json_from_response(response_text):
    """Extract JSON from response that may contain other text"""
    
    # Try to find JSON in code blocks
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object directly
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array
    array_match = re.search(r'\[[\s\S]*\]', response_text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None
```

### Fill Missing Fields with Defaults

```python
def fill_defaults(data, schema_defaults):
    """Fill missing fields with defaults"""
    result = schema_defaults.copy()
    result.update(data)
    return result

DEFAULTS = {
    "category": "general",
    "priority": "medium",
    "confidence": 0.5,
    "reasoning": "Unable to determine"
}

def parse_with_defaults(response_text):
    data = extract_json_from_response(response_text)
    if data:
        return fill_defaults(data, DEFAULTS)
    return DEFAULTS
```

### Lenient Parsing

```python
def parse_lenient(response_text):
    """Parse with type coercion and cleanup"""
    data = extract_json_from_response(response_text)
    if not data:
        return None
    
    result = {}
    
    # Category - normalize case
    if 'category' in data:
        result['category'] = str(data['category']).lower().strip()
    
    # Confidence - handle string numbers
    if 'confidence' in data:
        try:
            conf = float(data['confidence'])
            result['confidence'] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            result['confidence'] = 0.5
    
    # Priority - map variations
    if 'priority' in data:
        priority_map = {
            'high': 'high', 'hi': 'high', '1': 'high',
            'medium': 'medium', 'med': 'medium', '2': 'medium',
            'low': 'low', 'lo': 'low', '3': 'low'
        }
        p = str(data['priority']).lower().strip()
        result['priority'] = priority_map.get(p, 'medium')
    
    return result
```

---

## Fallback Strategies

### Cascading Fallbacks

```python
def get_classification(text):
    """Try multiple approaches, fall back as needed"""
    
    # Level 1: Structured output with strict schema
    try:
        result = call_with_structured_output(text)
        if validate_strict(result):
            return result
    except Exception:
        pass
    
    # Level 2: JSON output with lenient parsing
    try:
        response = call_with_json_prompt(text)
        result = parse_lenient(response)
        if result:
            return result
    except Exception:
        pass
    
    # Level 3: Simple text classification
    try:
        response = call_simple_prompt(text)
        result = parse_simple_response(response)
        if result:
            return result
    except Exception:
        pass
    
    # Level 4: Default response
    return {
        "category": "general",
        "priority": "medium",
        "confidence": 0.0,
        "reasoning": "Classification failed - using defaults"
    }
```

### Model Fallback

```python
MODELS = [
    "gpt-4o",           # Primary
    "gpt-4o-mini",      # Fallback 1
    "gpt-3.5-turbo"     # Fallback 2
]

def get_with_model_fallback(prompt):
    """Try multiple models"""
    for model in MODELS:
        try:
            response = call_llm(prompt, model=model)
            valid, result = validate_response(response)
            if valid:
                return result
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    
    raise ValueError("All models failed")
```

---

## Error Classification

Categorize errors to handle them appropriately:

```python
from enum import Enum

class ErrorType(Enum):
    SYNTAX_ERROR = "syntax"         # Invalid JSON
    MISSING_FIELD = "missing"       # Required field absent
    WRONG_TYPE = "type"             # Wrong data type
    INVALID_VALUE = "value"         # Value out of range
    EXTRA_CONTENT = "extra"         # Text outside structure
    UNKNOWN = "unknown"

def classify_error(response_text, validation_error):
    """Classify the error type for appropriate handling"""
    
    if "JSONDecodeError" in str(validation_error):
        return ErrorType.SYNTAX_ERROR
    
    if "required" in str(validation_error).lower():
        return ErrorType.MISSING_FIELD
    
    if "type" in str(validation_error).lower():
        return ErrorType.WRONG_TYPE
    
    if "value" in str(validation_error).lower():
        return ErrorType.INVALID_VALUE
    
    if response_text and not response_text.strip().startswith('{'):
        return ErrorType.EXTRA_CONTENT
    
    return ErrorType.UNKNOWN

def handle_error(error_type, response, prompt):
    """Handle different error types differently"""
    
    handlers = {
        ErrorType.SYNTAX_ERROR: retry_with_json_emphasis,
        ErrorType.MISSING_FIELD: retry_with_field_reminder,
        ErrorType.WRONG_TYPE: retry_with_type_examples,
        ErrorType.INVALID_VALUE: retry_with_value_constraints,
        ErrorType.EXTRA_CONTENT: extract_and_parse,
        ErrorType.UNKNOWN: retry_simple
    }
    
    handler = handlers.get(error_type, retry_simple)
    return handler(response, prompt)
```

---

## Production Pipeline

### Complete Validation Pipeline

```python
class OutputPipeline:
    def __init__(self, schema_class, max_retries=3):
        self.schema = schema_class
        self.max_retries = max_retries
        self.stats = {"success": 0, "retry": 0, "fallback": 0, "fail": 0}
    
    def process(self, prompt):
        """Full processing pipeline with validation"""
        
        for attempt in range(self.max_retries):
            try:
                # Get response
                response = self.call_model(prompt)
                
                # Extract JSON
                data = self.extract_json(response)
                if not data:
                    continue
                
                # Validate schema
                result = self.schema(**data)
                
                # Semantic validation
                if not self.semantic_check(result):
                    continue
                
                # Success
                if attempt > 0:
                    self.stats["retry"] += 1
                else:
                    self.stats["success"] += 1
                return result
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    break
                prompt = self.add_error_feedback(prompt, e)
        
        # All retries failed - use fallback
        self.stats["fallback"] += 1
        return self.get_fallback()
    
    def semantic_check(self, result):
        """Domain-specific validation"""
        # Override in subclass
        return True
    
    def get_fallback(self):
        """Return safe default"""
        # Override in subclass
        return None
```

### Monitoring and Alerting

```python
class MonitoredPipeline(OutputPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_log = []
    
    def process(self, prompt):
        try:
            return super().process(prompt)
        except Exception as e:
            self.error_log.append({
                "timestamp": datetime.now(),
                "prompt_hash": hash(prompt),
                "error": str(e)
            })
            
            # Alert if error rate too high
            if self.get_error_rate() > 0.1:  # 10%
                self.send_alert()
            
            self.stats["fail"] += 1
            return self.get_fallback()
    
    def get_error_rate(self):
        total = sum(self.stats.values())
        if total == 0:
            return 0
        return self.stats["fail"] / total
    
    def get_metrics(self):
        return {
            "stats": self.stats,
            "error_rate": self.get_error_rate(),
            "recent_errors": self.error_log[-10:]
        }
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always validate | Never trust model output |
| Classify errors | Handle each type appropriately |
| Limit retries | Prevent infinite loops |
| Log failures | Debug and improve prompts |
| Use defaults | Graceful degradation |
| Monitor rates | Catch prompt regressions |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No validation | Always parse and validate |
| Infinite retries | Set max retry limit |
| Silent failures | Log all errors |
| Crash on failure | Use fallback values |
| Same retry prompt | Add error feedback |
| No monitoring | Track success/failure rates |

---

## Hands-on Exercise

### Your Task

Build a robust validation pipeline for a sentiment analysis API.

### Requirements

1. Schema: sentiment (positive/negative/neutral), confidence (0-1), keywords (array)
2. Handle JSON extraction from mixed responses
3. Implement retry with feedback (max 2 retries)
4. Provide sensible defaults on failure
5. Track success/retry/failure statistics

<details>
<summary>💡 Hints (click to expand)</summary>

- What should the default sentiment be?
- How do you handle a confidence of "high" instead of a number?
- When should you stop retrying?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json
import re

# Schema definition
class SentimentResult(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list, max_items=10)
    
    @validator('sentiment', pre=True)
    def normalize_sentiment(cls, v):
        return str(v).lower().strip()
    
    @validator('confidence', pre=True)
    def normalize_confidence(cls, v):
        if isinstance(v, str):
            mapping = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
            return mapping.get(v.lower(), 0.5)
        return float(v)

# Default response
DEFAULT_RESPONSE = SentimentResult(
    sentiment="neutral",
    confidence=0.0,
    keywords=[]
)

class SentimentPipeline:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries
        self.stats = {"success": 0, "retry": 0, "failure": 0}
    
    def extract_json(self, text):
        """Extract JSON from possibly mixed response"""
        # Try code block first
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def validate(self, data):
        """Validate against schema with lenient parsing"""
        return SentimentResult(**data)
    
    def analyze(self, text, prompt_template):
        """Full analysis with retry logic"""
        prompt = prompt_template.format(text=text)
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add error feedback on retry
                if attempt > 0 and last_error:
                    prompt += f"\n\nPrevious error: {last_error}. Please fix."
                
                # Call model (mock for example)
                response = self.call_model(prompt)
                
                # Extract and validate
                data = self.extract_json(response)
                if data:
                    result = self.validate(data)
                    
                    # Track stats
                    if attempt > 0:
                        self.stats["retry"] += 1
                    else:
                        self.stats["success"] += 1
                    
                    return result
                else:
                    last_error = "Could not find JSON in response"
                    
            except Exception as e:
                last_error = str(e)
        
        # All attempts failed
        self.stats["failure"] += 1
        return DEFAULT_RESPONSE
    
    def call_model(self, prompt):
        """Mock LLM call - replace with actual API"""
        # In production: return openai.chat.completions.create(...)
        pass
    
    def get_stats(self):
        total = sum(self.stats.values())
        return {
            **self.stats,
            "total": total,
            "success_rate": self.stats["success"] / total if total > 0 else 0
        }

# Usage
pipeline = SentimentPipeline(max_retries=2)

PROMPT = """Analyze the sentiment of this text:

{text}

Return JSON with:
- sentiment: "positive", "negative", or "neutral"
- confidence: number 0.0 to 1.0
- keywords: array of key sentiment words

Return ONLY the JSON object."""

# result = pipeline.analyze("I love this product!", PROMPT)
# print(result)
# print(pipeline.get_stats())
```

**Key features:**
1. Pydantic schema with validators for normalization
2. JSON extraction from mixed content
3. Retry with error feedback
4. Default response on complete failure
5. Statistics tracking

</details>

### Bonus Challenge

- [ ] Add semantic validation (confidence should be low if sentiment is neutral)
- [ ] Implement different retry strategies based on error type

---

## Summary

✅ **Always validate** model outputs before use

✅ **Retry with feedback** improves success rate

✅ **Partial parsing** recovers from minor format issues

✅ **Fallback values** ensure graceful degradation

✅ **Error classification** enables targeted fixes

**Next:** [JSON Mode & Structured Outputs](../06-json-mode-structured-outputs.md)

---

## Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Zod (TypeScript Validation)](https://zod.dev/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

---

<!-- 
Sources Consulted:
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Pydantic Documentation: https://docs.pydantic.dev/
-->
