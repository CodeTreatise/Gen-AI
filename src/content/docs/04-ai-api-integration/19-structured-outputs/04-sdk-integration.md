---
title: "SDK Integration"
---

# SDK Integration

## Introduction

The OpenAI SDKs provide native support for Structured Outputs through Pydantic (Python) and Zod (JavaScript). These integrations automatically convert your type definitions into JSON schemas and parse responses back into typed objects.

### What We'll Cover

- Pydantic models in Python
- Zod schemas in JavaScript
- The `parse()` method
- Accessing `output_parsed` in responses

### Prerequisites

- Python or JavaScript experience
- OpenAI SDK installed
- Basic type system knowledge

---

## Pydantic Models (Python)

### Basic Model Definition

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum
from datetime import datetime
import json


# Simple model
class Person(BaseModel):
    """Person extracted from text."""
    
    name: str
    age: int
    occupation: str


# Model with descriptions (used in schema)
class PersonWithDescriptions(BaseModel):
    """Person with field descriptions."""
    
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years", ge=0, le=150)
    occupation: str = Field(description="Current job or profession")


# Model with optional fields
class PersonOptional(BaseModel):
    """Person with optional fields."""
    
    name: str
    age: int
    occupation: Optional[str] = Field(
        default=None,
        description="Job, if mentioned"
    )
    email: Optional[str] = Field(
        default=None,
        description="Email address, if provided"
    )


# Show generated schemas
print("Pydantic to JSON Schema")
print("=" * 60)

print("\nBasic Model:")
print(json.dumps(Person.model_json_schema(), indent=2))

print("\n\nWith Descriptions:")
print(json.dumps(PersonWithDescriptions.model_json_schema(), indent=2))
```

### Complex Nested Models

```python
from typing import List, Dict, Any


class Address(BaseModel):
    """Physical address."""
    
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province")
    zip_code: str = Field(description="ZIP or postal code")
    country: str = Field(default="USA", description="Country")


class ContactInfo(BaseModel):
    """Contact information."""
    
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    address: Optional[Address] = Field(default=None, description="Physical address")


class Employee(BaseModel):
    """Employee information."""
    
    id: str = Field(description="Employee ID")
    name: str = Field(description="Full name")
    title: str = Field(description="Job title")
    department: str = Field(description="Department name")
    contact: ContactInfo = Field(description="Contact information")
    skills: List[str] = Field(description="List of skills")
    years_experience: int = Field(description="Years of experience", ge=0)
    is_manager: bool = Field(default=False, description="Whether manages others")


print("\n\nNested Model (Employee):")
print(json.dumps(Employee.model_json_schema(), indent=2))
```

### Models with Enums

```python
class Priority(str, Enum):
    """Task priority levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    """Task status values."""
    
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"


class Task(BaseModel):
    """Task with enum fields."""
    
    id: str = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: Priority = Field(description="Priority level")
    status: Status = Field(description="Current status")
    assignee: Optional[str] = Field(default=None, description="Assigned person")
    tags: List[str] = Field(default=[], description="Task tags")


# Using Literal for inline enums
class QuickTask(BaseModel):
    """Task with inline enum using Literal."""
    
    title: str
    priority: Literal["low", "medium", "high", "critical"]
    status: Literal["todo", "in_progress", "done"]


print("\n\nModel with Enums:")
print(json.dumps(Task.model_json_schema(), indent=2))
```

---

## Python SDK Usage

### The parse() Method

```python
# Simulated SDK response for demonstration
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MockParsedResponse:
    """Simulates SDK response structure."""
    
    output_parsed: Any
    output: List[dict]
    usage: dict


def simulate_responses_parse(
    model: str,
    input: List[dict],
    text_format: type
) -> MockParsedResponse:
    """Simulate the responses.parse() method."""
    
    # In real SDK:
    # response = client.responses.parse(
    #     model=model,
    #     input=input,
    #     text_format=text_format
    # )
    
    # Simulated successful response
    return MockParsedResponse(
        output_parsed=text_format(
            name="Alice Johnson",
            age=28,
            occupation="Software Engineer"
        ),
        output=[{
            "type": "message",
            "content": [{
                "type": "text",
                "text": '{"name": "Alice Johnson", "age": 28, "occupation": "Software Engineer"}'
            }]
        }],
        usage={"input_tokens": 50, "output_tokens": 25}
    )


# Usage example
response = simulate_responses_parse(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Extract person information."},
        {"role": "user", "content": "Alice Johnson is a 28-year-old software engineer."}
    ],
    text_format=Person
)

print("\n\nPython SDK parse() Result")
print("=" * 60)
print(f"Parsed type: {type(response.output_parsed).__name__}")
print(f"Name: {response.output_parsed.name}")
print(f"Age: {response.output_parsed.age}")
print(f"Occupation: {response.output_parsed.occupation}")
print(f"\nToken usage: {response.usage}")
```

### Accessing output_parsed

```python
class ResponseHandler:
    """Handles Structured Outputs responses."""
    
    def __init__(self, response):
        self.response = response
    
    @property
    def parsed(self) -> Any:
        """Get the parsed output."""
        return self.response.output_parsed
    
    @property
    def raw_json(self) -> str:
        """Get raw JSON string."""
        return self.response.output[0]["content"][0]["text"]
    
    @property
    def has_refusal(self) -> bool:
        """Check if response was refused."""
        content = self.response.output[0]["content"][0]
        return content.get("type") == "refusal"
    
    @property
    def refusal_message(self) -> Optional[str]:
        """Get refusal message if present."""
        if self.has_refusal:
            return self.response.output[0]["content"][0].get("refusal")
        return None
    
    def to_dict(self) -> dict:
        """Convert parsed output to dictionary."""
        if hasattr(self.parsed, "model_dump"):
            return self.parsed.model_dump()
        return dict(self.parsed)


# Usage
handler = ResponseHandler(response)

print("\n\nResponse Handler Usage")
print("=" * 60)
print(f"Parsed object: {handler.parsed}")
print(f"As dictionary: {handler.to_dict()}")
print(f"Has refusal: {handler.has_refusal}")
```

---

## Zod Schemas (JavaScript)

### Basic Schema Definition

```javascript
// JavaScript/TypeScript examples shown as code blocks

const ZOD_BASIC_EXAMPLE = `
import { z } from 'zod';
import { zodResponseFormat } from 'openai/helpers/zod';

// Simple schema
const PersonSchema = z.object({
  name: z.string().describe("The person's full name"),
  age: z.number().int().describe("Age in years"),
  occupation: z.string().describe("Current job")
});

// Type inference
type Person = z.infer<typeof PersonSchema>;

// Convert to response format
const responseFormat = zodResponseFormat(PersonSchema, "person");
`;

const ZOD_NESTED_EXAMPLE = `
// Nested schemas
const AddressSchema = z.object({
  street: z.string(),
  city: z.string(),
  state: z.string(),
  zip: z.string()
});

const ContactSchema = z.object({
  email: z.string().email(),
  phone: z.string().optional(),
  address: AddressSchema.optional()
});

const EmployeeSchema = z.object({
  id: z.string(),
  name: z.string(),
  title: z.string(),
  contact: ContactSchema,
  skills: z.array(z.string()),
  yearsExperience: z.number().int().min(0)
});
`;

const ZOD_ENUM_EXAMPLE = `
// Enums in Zod
const PrioritySchema = z.enum(["low", "medium", "high", "critical"]);
const StatusSchema = z.enum(["todo", "in_progress", "review", "done"]);

const TaskSchema = z.object({
  id: z.string(),
  title: z.string(),
  priority: PrioritySchema,
  status: StatusSchema,
  assignee: z.string().nullable(),
  tags: z.array(z.string())
});
`;


print("Zod Schema Examples (JavaScript/TypeScript)")
print("=" * 60)

print("\n📋 Basic Schema:")
print(ZOD_BASIC_EXAMPLE)

print("\n📋 Nested Schema:")
print(ZOD_NESTED_EXAMPLE)

print("\n📋 With Enums:")
print(ZOD_ENUM_EXAMPLE)
```

### JavaScript SDK Usage

```javascript
const JS_SDK_USAGE = `
import OpenAI from 'openai';
import { z } from 'zod';
import { zodResponseFormat } from 'openai/helpers/zod';

const client = new OpenAI();

// Define schema
const PersonSchema = z.object({
  name: z.string(),
  age: z.number().int(),
  occupation: z.string()
});

// Use with parse()
async function extractPerson(text: string) {
  const response = await client.responses.parse({
    model: 'gpt-4o',
    input: [
      { role: 'system', content: 'Extract person information.' },
      { role: 'user', content: text }
    ],
    text_format: zodResponseFormat(PersonSchema, 'person')
  });
  
  // Type-safe access
  const person = response.output_parsed;
  console.log(person.name);  // TypeScript knows this is string
  console.log(person.age);   // TypeScript knows this is number
  
  return person;
}

// With Chat Completions
async function extractWithChat(text: string) {
  const response = await client.chat.completions.parse({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: 'Extract person information.' },
      { role: 'user', content: text }
    ],
    response_format: zodResponseFormat(PersonSchema, 'person')
  });
  
  const person = response.choices[0].message.parsed;
  return person;
}
`;


print("\n\nJavaScript SDK Usage")
print("=" * 60)
print(JS_SDK_USAGE)
```

---

## Complete Python Example

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum
from dataclasses import dataclass


# Define comprehensive models
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Entity(BaseModel):
    """Named entity from text."""
    
    name: str = Field(description="Entity name")
    type: Literal["person", "organization", "location", "product", "other"]
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)


class TextAnalysis(BaseModel):
    """Complete text analysis result."""
    
    summary: str = Field(description="Brief summary of the text")
    sentiment: Sentiment = Field(description="Overall sentiment")
    entities: List[Entity] = Field(description="Named entities found")
    topics: List[str] = Field(description="Main topics discussed")
    word_count: int = Field(description="Approximate word count")
    language: str = Field(description="Detected language code")
    key_phrases: List[str] = Field(description="Key phrases extracted")


# Simulated extraction
@dataclass 
class ExtractorResult:
    success: bool
    data: Optional[TextAnalysis]
    error: Optional[str]


class TextAnalyzer:
    """Structured text analysis using OpenAI."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # self.client = OpenAI(api_key=api_key)
    
    def analyze(self, text: str) -> ExtractorResult:
        """Analyze text and return structured result."""
        
        try:
            # Real SDK call would be:
            # response = self.client.responses.parse(
            #     model="gpt-4o",
            #     input=[
            #         {
            #             "role": "system",
            #             "content": "Analyze the following text and extract structured information."
            #         },
            #         {"role": "user", "content": text}
            #     ],
            #     text_format=TextAnalysis
            # )
            # return ExtractorResult(
            #     success=True,
            #     data=response.output_parsed,
            #     error=None
            # )
            
            # Simulated response
            return ExtractorResult(
                success=True,
                data=TextAnalysis(
                    summary="A technology company announced a new product launch.",
                    sentiment=Sentiment.POSITIVE,
                    entities=[
                        Entity(name="TechCorp", type="organization", confidence=0.95),
                        Entity(name="San Francisco", type="location", confidence=0.88)
                    ],
                    topics=["technology", "product launch", "innovation"],
                    word_count=150,
                    language="en",
                    key_phrases=["new product", "innovation", "market leader"]
                ),
                error=None
            )
            
        except Exception as e:
            return ExtractorResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def analyze_batch(self, texts: List[str]) -> List[ExtractorResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]


# Usage
analyzer = TextAnalyzer("demo-api-key")
result = analyzer.analyze("TechCorp announced a new product in San Francisco today...")

print("\n\nComplete Analysis Example")
print("=" * 60)

if result.success:
    analysis = result.data
    print(f"Summary: {analysis.summary}")
    print(f"Sentiment: {analysis.sentiment.value}")
    print(f"Language: {analysis.language}")
    print(f"Word count: {analysis.word_count}")
    
    print(f"\nEntities ({len(analysis.entities)}):")
    for entity in analysis.entities:
        print(f"  - {entity.name} ({entity.type}): {entity.confidence:.0%}")
    
    print(f"\nTopics: {', '.join(analysis.topics)}")
    print(f"Key phrases: {', '.join(analysis.key_phrases)}")
```

---

## Error Handling

```python
class StructuredOutputError(Exception):
    """Base error for Structured Outputs."""
    pass


class SchemaValidationError(StructuredOutputError):
    """Schema doesn't meet requirements."""
    pass


class ParseError(StructuredOutputError):
    """Failed to parse response."""
    pass


class RefusalError(StructuredOutputError):
    """Model refused the request."""
    
    def __init__(self, message: str, refusal_reason: str):
        super().__init__(message)
        self.refusal_reason = refusal_reason


class SafeExtractor:
    """Extractor with proper error handling."""
    
    def __init__(self, client):
        self.client = client
    
    def extract(
        self,
        text: str,
        schema: type
    ) -> Any:
        """Extract with error handling."""
        
        try:
            # Validate schema first
            if not hasattr(schema, 'model_json_schema'):
                raise SchemaValidationError(
                    "Schema must be a Pydantic BaseModel"
                )
            
            # Make request
            # response = self.client.responses.parse(...)
            
            # Simulated response with refusal check
            response = self._simulate_response(schema)
            
            # Check for refusal
            if self._is_refusal(response):
                refusal_msg = self._get_refusal_message(response)
                raise RefusalError(
                    "Model refused request",
                    refusal_msg
                )
            
            # Get parsed result
            if response.output_parsed is None:
                raise ParseError("No parsed output in response")
            
            return response.output_parsed
            
        except RefusalError:
            raise
        except SchemaValidationError:
            raise
        except Exception as e:
            raise ParseError(f"Extraction failed: {e}")
    
    def _simulate_response(self, schema):
        """Simulate response for demo."""
        return MockParsedResponse(
            output_parsed=schema(
                name="Test",
                age=25,
                occupation="Tester"
            ),
            output=[{
                "content": [{"type": "text", "text": "{}"}]
            }],
            usage={}
        )
    
    def _is_refusal(self, response) -> bool:
        """Check if response is a refusal."""
        try:
            content = response.output[0]["content"][0]
            return content.get("type") == "refusal"
        except (IndexError, KeyError):
            return False
    
    def _get_refusal_message(self, response) -> str:
        """Get refusal message."""
        try:
            return response.output[0]["content"][0].get("refusal", "Unknown reason")
        except (IndexError, KeyError):
            return "Unknown reason"


# Usage with error handling
print("\n\nError Handling Example")
print("=" * 60)

try:
    extractor = SafeExtractor(None)
    result = extractor.extract("Some text", Person)
    print(f"✅ Extracted: {result}")
except RefusalError as e:
    print(f"❌ Refused: {e.refusal_reason}")
except ParseError as e:
    print(f"❌ Parse error: {e}")
except SchemaValidationError as e:
    print(f"❌ Schema error: {e}")
```

---

## Hands-on Exercise

### Your Task

Build a type-safe API client wrapper that uses Pydantic models for Structured Outputs.

### Requirements

1. Define models for different extraction types
2. Create a reusable client class
3. Handle errors and refusals
4. Support multiple output types

<details>
<summary>💡 Hints</summary>

- Use generics for type safety
- Create a base model class
- Implement a registry pattern for schemas
</details>

<details>
<summary>✅ Solution</summary>

```python
from pydantic import BaseModel, Field
from typing import TypeVar, Generic, Type, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json


# Generic type for output models
T = TypeVar('T', bound=BaseModel)


# Output models
class PersonOutput(BaseModel):
    """Person extraction output."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title")


class ContactOutput(BaseModel):
    """Contact extraction output."""
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)


class MeetingOutput(BaseModel):
    """Meeting extraction output."""
    title: str = Field(description="Meeting title")
    date: str = Field(description="Date in ISO format")
    attendees: List[str] = Field(description="List of attendees")
    agenda: List[str] = Field(description="Agenda items")
    duration_minutes: int = Field(description="Meeting duration")


class SentimentOutput(BaseModel):
    """Sentiment analysis output."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence 0-1")
    keywords: List[str] = Field(description="Keywords affecting sentiment")


# Result wrapper
@dataclass
class ExtractionResult(Generic[T]):
    """Result of extraction."""
    success: bool
    data: Optional[T]
    error: Optional[str]
    usage: Dict[str, int]


# Type-safe client
class StructuredClient:
    """Type-safe Structured Outputs client."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # self.client = OpenAI(api_key=api_key)
        
        # Schema registry
        self._schemas: Dict[str, Type[BaseModel]] = {
            "person": PersonOutput,
            "contact": ContactOutput,
            "meeting": MeetingOutput,
            "sentiment": SentimentOutput
        }
    
    def register_schema(
        self,
        name: str,
        schema: Type[BaseModel]
    ):
        """Register a new schema."""
        self._schemas[name] = schema
    
    def extract(
        self,
        text: str,
        output_type: Type[T],
        system_prompt: str = None
    ) -> ExtractionResult[T]:
        """Extract structured data."""
        
        prompt = system_prompt or f"Extract {output_type.__name__} from the text."
        
        try:
            # Simulated API call
            # response = self.client.responses.parse(
            #     model="gpt-4o",
            #     input=[
            #         {"role": "system", "content": prompt},
            #         {"role": "user", "content": text}
            #     ],
            #     text_format=output_type
            # )
            
            # Simulated success
            if output_type == PersonOutput:
                data = PersonOutput(
                    name="John Smith",
                    age=35,
                    occupation="Engineer"
                )
            elif output_type == SentimentOutput:
                data = SentimentOutput(
                    sentiment="positive",
                    confidence=0.92,
                    keywords=["great", "excellent", "happy"]
                )
            else:
                data = output_type.model_validate({
                    field: "test" if info.annotation == str else 0
                    for field, info in output_type.model_fields.items()
                })
            
            return ExtractionResult(
                success=True,
                data=data,
                error=None,
                usage={"input_tokens": 50, "output_tokens": 25}
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                data=None,
                error=str(e),
                usage={}
            )
    
    def extract_by_name(
        self,
        text: str,
        schema_name: str
    ) -> ExtractionResult:
        """Extract using registered schema name."""
        
        if schema_name not in self._schemas:
            return ExtractionResult(
                success=False,
                data=None,
                error=f"Unknown schema: {schema_name}",
                usage={}
            )
        
        schema = self._schemas[schema_name]
        return self.extract(text, schema)
    
    def batch_extract(
        self,
        texts: List[str],
        output_type: Type[T]
    ) -> List[ExtractionResult[T]]:
        """Extract from multiple texts."""
        return [self.extract(text, output_type) for text in texts]


# Custom model registration
class ProductReview(BaseModel):
    """Product review extraction."""
    product_name: str
    rating: float = Field(ge=1, le=5)
    pros: List[str]
    cons: List[str]
    recommendation: bool


# Usage
client = StructuredClient("demo-key")

# Register custom schema
client.register_schema("review", ProductReview)

# Type-safe extraction
print("Type-Safe Structured Client")
print("=" * 60)

# Extract person (type-safe)
person_result = client.extract(
    "John Smith is a 35-year-old engineer at TechCorp.",
    PersonOutput
)

if person_result.success:
    person = person_result.data  # Type: PersonOutput
    print(f"\n👤 Person: {person.name}, {person.age}, {person.occupation}")

# Extract sentiment
sentiment_result = client.extract(
    "This product is absolutely amazing! Best purchase ever!",
    SentimentOutput
)

if sentiment_result.success:
    sentiment = sentiment_result.data  # Type: SentimentOutput
    print(f"\n😊 Sentiment: {sentiment.sentiment} ({sentiment.confidence:.0%})")
    print(f"   Keywords: {', '.join(sentiment.keywords)}")

# Extract by name (dynamic)
contact_result = client.extract_by_name(
    "Contact me at john@example.com",
    "contact"
)

print(f"\n📧 Contact result success: {contact_result.success}")

# Show available schemas
print("\n\n📋 Registered Schemas:")
for name, schema in client._schemas.items():
    fields = list(schema.model_fields.keys())
    print(f"  - {name}: {', '.join(fields)}")
```

</details>

---

## Summary

✅ Pydantic provides Python type definitions for schemas  
✅ Zod provides JavaScript/TypeScript schema definitions  
✅ `parse()` method auto-converts schemas and parses responses  
✅ `output_parsed` contains the typed result object  
✅ Error handling should check for refusals and parse failures

**Next:** [Schema Requirements](./05-schema-requirements.md)

---

## Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/) — Python models
- [Zod Documentation](https://zod.dev/) — JavaScript schemas
- [OpenAI Python SDK](https://github.com/openai/openai-python/blob/main/helpers.md) — SDK helpers
- [OpenAI Node SDK](https://github.com/openai/openai-node/blob/master/helpers.md) — SDK helpers
