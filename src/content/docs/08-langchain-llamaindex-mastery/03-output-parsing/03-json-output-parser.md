---
title: "JSON Output Parser"
---

# JSON Output Parser

## Introduction

JsonOutputParser offers flexible JSON parsing with optional schema validation. Unlike PydanticOutputParser, it can work with or without a schema, making it ideal for dynamic or loosely structured data. Its streaming support handles partial JSON during generation, enabling real-time updates.

### What We'll Cover

- JsonOutputParser fundamentals
- Schema-based and schema-free parsing
- Streaming JSON with partial updates
- SimpleJsonOutputParser for basic cases
- Integration patterns and best practices

### Prerequisites

- Parser Basics (Lesson 8.3.1)
- Understanding of JSON structure
- LCEL chains

---

## JsonOutputParser Fundamentals

### Basic Usage (Without Schema)

```python
from langchain_core.output_parsers import JsonOutputParser

# Create parser without schema - returns dict
parser = JsonOutputParser()

# Parse any valid JSON
json_text = '{"name": "Alice", "age": 30, "skills": ["Python", "AI"]}'
result = parser.parse(json_text)

print(result)
# {'name': 'Alice', 'age': 30, 'skills': ['Python', 'AI']}

print(type(result))  # <class 'dict'>
```

### With Pydantic Schema

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Age in years")
    skills: list[str] = Field(description="List of skills")

# Create parser with schema
parser = JsonOutputParser(pydantic_object=Person)

# Parse returns Pydantic model instance
json_text = '{"name": "Alice", "age": 30, "skills": ["Python", "AI"]}'
person = parser.parse(json_text)

print(person.name)   # Alice
print(type(person))  # <class '__main__.Person'>
```

### Format Instructions

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Availability status")

parser = JsonOutputParser(pydantic_object=Product)

print(parser.get_format_instructions())
```

**Output:**
```
Return a JSON object matching this schema:
{"name": "Product name", "price": "Price in USD", "in_stock": "Availability status"}
```

---

## Complete Chain Example

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class WeatherData(BaseModel):
    """Weather information."""
    location: str = Field(description="City and country")
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")

parser = JsonOutputParser(pydantic_object=WeatherData)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a weather information assistant.
Always respond with valid JSON.
{format_instructions}"""),
    ("human", "What's the weather like in {city}?")
])

model = init_chat_model("gpt-4o")

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

weather = chain.invoke({"city": "Tokyo"})
print(f"Location: {weather.location}")
print(f"Temperature: {weather.temperature}°C")
print(f"Conditions: {weather.conditions}")
print(f"Humidity: {weather.humidity}%")
```

---

## Streaming JSON

### Real-time Partial Updates

JsonOutputParser supports streaming, returning partial valid JSON as it's generated:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Story(BaseModel):
    title: str = Field(description="Story title")
    content: str = Field(description="The story content")
    moral: str = Field(description="The moral of the story")

parser = JsonOutputParser(pydantic_object=Story)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Write stories in JSON format. {format_instructions}"),
    ("human", "Write a short story about {topic}")
])

model = init_chat_model("gpt-4o")

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

# Stream partial JSON updates
for partial in chain.stream({"topic": "a brave robot"}):
    print(partial)
    print("---")
```

**Output (progressive updates):**
```python
{}
---
{'title': ''}
---
{'title': 'The Brave Robot'}
---
{'title': 'The Brave Robot', 'content': ''}
---
{'title': 'The Brave Robot', 'content': 'In a factory far away...'}
---
{'title': 'The Brave Robot', 'content': 'In a factory far away, there lived...', 'moral': ''}
---
{'title': 'The Brave Robot', 'content': 'In a factory far away, there lived a small robot...', 'moral': 'Courage is not about size...'}
---
```

### Streaming with UI Updates

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    summary: str = Field(description="Summary text")
    score: float = Field(description="Score 0-100")
    tags: list[str] = Field(description="Relevant tags")

parser = JsonOutputParser(pydantic_object=Analysis)

model = init_chat_model("gpt-4o")
chain = model | parser

# Track what fields are available
previous = {}
for partial in chain.stream("Analyze: 'Great product, fast shipping!'"):
    # Detect new fields
    new_fields = set(partial.keys()) - set(previous.keys())
    
    if new_fields:
        for field in new_fields:
            print(f"✅ Field '{field}' available: {partial[field]}")
    
    previous = partial

# Final result
print(f"\nFinal: {previous}")
```

---

## SimpleJsonOutputParser

For basic JSON parsing without Pydantic validation:

```python
from langchain_core.output_parsers.json import SimpleJsonOutputParser

parser = SimpleJsonOutputParser()

# Parses to dict
result = parser.parse('{"key": "value", "number": 42}')
print(result)  # {'key': 'value', 'number': 42}

# Also handles JSON arrays
array_result = parser.parse('[1, 2, 3, "four"]')
print(array_result)  # [1, 2, 3, 'four']
```

### When to Use SimpleJsonOutputParser

| Use Case | Parser Choice |
|----------|---------------|
| Need validated structured output | `JsonOutputParser(pydantic_object=...)` |
| Need flexible dict output | `JsonOutputParser()` |
| Simple parsing, minimal overhead | `SimpleJsonOutputParser()` |
| Streaming with validation | `JsonOutputParser(pydantic_object=...)` |

---

## Handling JSON in Text

Sometimes LLMs wrap JSON in markdown code blocks:

### Extracting JSON from Markdown

```python
from langchain_core.output_parsers import JsonOutputParser
import re

def extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks or raw text."""
    # Try to find JSON in code block
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()

parser = JsonOutputParser()

# Text with JSON in code block
response = '''Here's the data:
```json
{"name": "Alice", "score": 95}
```
That's all!'''

# Extract and parse
json_str = extract_json(response)
result = parser.parse(json_str)
print(result)  # {'name': 'Alice', 'score': 95}
```

### Using JsonOutputKeyToolsParser

For extracting specific keys:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# Parse and extract specific key
full_json = '{"status": "success", "data": {"items": [1, 2, 3]}, "meta": {}}'
result = parser.parse(full_json)

# Access nested data
items = result.get("data", {}).get("items", [])
print(items)  # [1, 2, 3]
```

---

## Schema-Free Dynamic Parsing

### When Structure Varies

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# No schema - flexible output
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract key information as JSON.
Include whatever fields are relevant to the content.
Always return valid JSON."""),
    ("human", "{text}")
])

model = init_chat_model("gpt-4o")
chain = prompt | model | parser

# Different inputs produce different structures
result1 = chain.invoke({"text": "John is 30 years old and lives in NYC"})
print(result1)
# {'name': 'John', 'age': 30, 'city': 'NYC'}

result2 = chain.invoke({"text": "Product: Widget, Price: $99, Rating: 4.5 stars"})
print(result2)
# {'product': 'Widget', 'price': 99, 'rating': 4.5}
```

### Combining with Runtime Validation

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class FlexibleData(BaseModel):
    """Flexible model for validation."""
    
    class Config:
        extra = "allow"  # Allow additional fields

# Parse without schema
parser = JsonOutputParser()
data = parser.parse('{"name": "Test", "unknown_field": 123}')

# Validate after parsing
try:
    validated = FlexibleData(**data)
    print(f"Validated: {validated}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

## Error Handling

### Malformed JSON

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

parser = JsonOutputParser()

# Invalid JSON
try:
    result = parser.parse("This is not JSON at all")
except OutputParserException as e:
    print(f"Parse error: {e}")

# Partial JSON
try:
    result = parser.parse('{"name": "incomplete')
except OutputParserException as e:
    print(f"Incomplete JSON: {e}")
```

### Safe Parsing with Defaults

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from typing import Optional

parser = JsonOutputParser()

def safe_parse(text: str, default: dict = None) -> dict:
    """Parse JSON with fallback to default."""
    try:
        return parser.parse(text)
    except OutputParserException:
        return default or {}

# Usage
result = safe_parse("invalid json", {"error": True})
print(result)  # {'error': True}
```

---

## Advanced Patterns

### Nested JSON Extraction

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    source: str
    timestamp: str

class DataPoint(BaseModel):
    value: float
    unit: str

class Report(BaseModel):
    title: str
    metadata: Metadata
    data: list[DataPoint]

parser = JsonOutputParser(pydantic_object=Report)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate reports in JSON. {format_instructions}"),
    ("human", "Create a report about {topic}")
])

model = init_chat_model("gpt-4o")
chain = prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser

report = chain.invoke({"topic": "temperature readings"})
print(f"Title: {report.title}")
print(f"Source: {report.metadata.source}")
for point in report.data:
    print(f"  {point.value} {point.unit}")
```

### Multiple JSON Objects

```python
from langchain_core.output_parsers import JsonOutputParser
import json

def parse_json_lines(text: str) -> list[dict]:
    """Parse newline-delimited JSON."""
    results = []
    for line in text.strip().split('\n'):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results

# Parse JSONL format
jsonl_text = """{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}"""

items = parse_json_lines(jsonl_text)
print(items)
# [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}, {'id': 3, 'name': 'Charlie'}]
```

### JSON with Streaming Aggregation

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Items(BaseModel):
    items: list[str] = Field(description="List of items")

parser = JsonOutputParser(pydantic_object=Items)
model = init_chat_model("gpt-4o")
chain = model | parser

# Collect streaming items
all_items = []
for partial in chain.stream("List 5 programming languages as JSON"):
    if 'items' in partial:
        current = partial['items']
        new_items = current[len(all_items):]
        for item in new_items:
            print(f"New item: {item}")
            all_items.append(item)

print(f"\nAll items: {all_items}")
```

---

## Comparing JSON Parsers

| Feature | JsonOutputParser | PydanticOutputParser | SimpleJsonOutputParser |
|---------|------------------|---------------------|------------------------|
| Schema validation | Optional | Required | No |
| Returns | dict or Model | Model | dict |
| Streaming | ✅ Partial JSON | ❌ Full only | ❌ |
| Format instructions | Basic | Detailed | None |
| Type coercion | Via Pydantic | Yes | No |
| Use case | Flexible | Strict | Simple |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use schema for critical data | Ensures correct types |
| Enable streaming for long outputs | Better UX |
| Handle parse errors gracefully | Robust applications |
| Extract JSON from markdown | Handle LLM variations |
| Validate after parsing | Catch edge cases |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Assuming valid JSON always | Add error handling |
| Not checking for code blocks | Extract JSON first |
| Ignoring partial streaming | Track progressive updates |
| Schema-free for critical data | Use Pydantic schema |
| Not testing edge cases | Test with various inputs |

---

## Hands-on Exercise

### Your Task

Build a real-time data extraction pipeline that:
1. Parses JSON with streaming updates
2. Displays progress as fields become available
3. Handles both valid and invalid responses

### Requirements

1. Create a model with 4+ fields
2. Implement streaming with field detection
3. Add error handling with retry
4. Show progressive UI updates

### Expected Result

```
⏳ Starting extraction...
✅ title: "AI in Healthcare"
✅ category: "Technology"
✅ summary: "An overview of..."
⏳ Waiting for more fields...
✅ tags: ["AI", "Healthcare", "ML"]
✅ confidence: 0.95
🎉 Complete! 5 fields extracted
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Track previous keys vs current keys
- Use set operations to find new fields
- Add sleep/delay for visual effect
- Handle OutputParserException in stream

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
import time

class Article(BaseModel):
    """Extracted article data."""
    title: str = Field(description="Article title")
    category: str = Field(description="Category like Technology, Health, etc.")
    summary: str = Field(description="Brief 2-3 sentence summary")
    tags: list[str] = Field(description="3-5 relevant tags")
    confidence: float = Field(description="Extraction confidence 0.0-1.0")

parser = JsonOutputParser(pydantic_object=Article)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract article information as JSON.
{format_instructions}
Be thorough and accurate."""),
    ("human", "Extract from: {text}")
]).partial(format_instructions=parser.get_format_instructions())

model = init_chat_model("gpt-4o")
chain = prompt | model | parser

def stream_with_progress(text: str):
    """Stream extraction with visual progress."""
    print("⏳ Starting extraction...\n")
    
    previous_keys = set()
    final_result = {}
    
    try:
        for partial in chain.stream({"text": text}):
            current_keys = set(partial.keys())
            final_result = partial
            
            # Find new fields
            new_keys = current_keys - previous_keys
            
            for key in new_keys:
                value = partial[key]
                # Truncate long values for display
                display = str(value)[:50] + "..." if len(str(value)) > 50 else value
                print(f"✅ {key}: {display}")
                time.sleep(0.1)  # Visual effect
            
            previous_keys = current_keys
        
        print(f"\n🎉 Complete! {len(final_result)} fields extracted")
        return final_result
        
    except OutputParserException as e:
        print(f"\n❌ Parse error: {e}")
        print("🔄 Would retry here...")
        return None

# Test
article_text = """
Breaking: New AI System Revolutionizes Medical Diagnosis

A groundbreaking artificial intelligence system developed by researchers 
at Stanford University can now diagnose certain cancers with 95% accuracy,
outperforming human radiologists in blind tests. The system, called 
MedAI-Vision, was trained on over 10 million medical images.

Experts say this could transform healthcare delivery in underserved areas.
"""

result = stream_with_progress(article_text)

if result:
    print("\n--- Final Structured Data ---")
    print(f"Title: {result.get('title')}")
    print(f"Category: {result.get('category')}")
    print(f"Tags: {result.get('tags')}")
    print(f"Confidence: {result.get('confidence')}")
```

</details>

### Bonus Challenges

- [ ] Add retry logic on parse failures
- [ ] Support batch processing multiple articles
- [ ] Implement timeout for slow responses
- [ ] Create a progress bar for streaming

---

## Summary

✅ `JsonOutputParser` parses JSON with optional Pydantic schema  
✅ Schema-free mode returns flexible dict output  
✅ Streaming provides real-time partial JSON updates  
✅ `SimpleJsonOutputParser` offers lightweight parsing  
✅ Handle JSON wrapped in markdown code blocks  
✅ Progressive field detection enables real-time UIs  

**Next:** [Structured Output](./04-structured-output.md) — Native LLM structured output with `with_structured_output()`

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Pydantic Parser](./02-pydantic-output-parser.md) | [Output Parsing](./00-output-parsing.md) | [Structured Output](./04-structured-output.md) |

<!-- 
Sources Consulted:
- LangChain JsonOutputParser: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/output_parsers/json.py
- LangChain streaming documentation: https://python.langchain.com/docs/concepts/streaming/
- JSON parsing in Python: https://docs.python.org/3/library/json.html
-->
