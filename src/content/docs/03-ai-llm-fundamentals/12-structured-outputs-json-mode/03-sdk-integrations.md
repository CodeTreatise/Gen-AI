---
title: "SDK Integrations"
---

# SDK Integrations

## Introduction

Modern SDKs integrate schema validation libraries to make structured outputs type-safe and developer-friendly. Python uses Pydantic, while JavaScript/TypeScript uses Zod.

### What We'll Cover

- Pydantic integration (Python)
- Zod integration (TypeScript/JavaScript)
- Automatic schema generation
- Validation and parsing utilities

---

## Pydantic Integration (Python)

### Basic Usage

```python
from openai import OpenAI
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    pros: list[str]
    cons: list[str]

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Review the movie 'Inception'"}
    ],
    response_format=MovieReview
)

# Automatically parsed to Pydantic model
review = response.choices[0].message.parsed
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
```

### Complex Models

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import date

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")

class Person(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: Optional[str] = None
    address: Address
    role: Literal["admin", "user", "guest"]

# Use with structured outputs
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Create a sample user profile"}],
    response_format=Person
)
```

### Field Descriptions

```python
class TaskExtraction(BaseModel):
    """Extract task information from natural language"""
    
    title: str = Field(
        description="Short title for the task (3-10 words)"
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="Priority level based on urgency indicators"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Due date in YYYY-MM-DD format if mentioned"
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Person assigned to the task if mentioned"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Relevant tags or categories"
    )
```

### Nested Models

```python
class Ingredient(BaseModel):
    name: str
    amount: str
    unit: str

class Step(BaseModel):
    number: int
    instruction: str
    duration_minutes: Optional[int] = None

class Recipe(BaseModel):
    title: str
    description: str
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    ingredients: list[Ingredient]
    steps: list[Step]
    tags: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Create a recipe for chocolate chip cookies"}
    ],
    response_format=Recipe
)
```

---

## Zod Integration (TypeScript)

### Basic Usage

```typescript
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod";

const client = new OpenAI();

const MovieReview = z.object({
  title: z.string(),
  rating: z.number(),
  summary: z.string(),
  pros: z.array(z.string()),
  cons: z.array(z.string()),
});

type MovieReview = z.infer<typeof MovieReview>;

async function getReview(): Promise<MovieReview> {
  const response = await client.beta.chat.completions.parse({
    model: "gpt-4o",
    messages: [
      { role: "user", content: "Review the movie 'Inception'" }
    ],
    response_format: zodResponseFormat(MovieReview, "movie_review"),
  });

  return response.choices[0].message.parsed!;
}
```

### Complex Schemas

```typescript
const Address = z.object({
  street: z.string(),
  city: z.string(),
  state: z.string(),
  zipCode: z.string().regex(/^\d{5}(-\d{4})?$/),
});

const Person = z.object({
  name: z.string().min(1).max(100),
  age: z.number().int().min(0).max(150),
  email: z.string().email().optional(),
  address: Address,
  role: z.enum(["admin", "user", "guest"]),
});

const response = await client.beta.chat.completions.parse({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Create a sample user" }],
  response_format: zodResponseFormat(Person, "person"),
});
```

### Descriptions in Zod

```typescript
const TaskExtraction = z.object({
  title: z.string().describe("Short title for the task (3-10 words)"),
  priority: z.enum(["low", "medium", "high"]).describe("Priority based on urgency"),
  dueDate: z.string().nullable().describe("Due date in YYYY-MM-DD format"),
  assignee: z.string().nullable().describe("Person assigned if mentioned"),
  tags: z.array(z.string()).describe("Relevant categories"),
});
```

---

## Automatic Schema Generation

### From Pydantic

```python
from pydantic import BaseModel
import json

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Generate JSON Schema
schema = Product.model_json_schema()
print(json.dumps(schema, indent=2))
```

**Output:**
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "price": {"type": "number"},
    "in_stock": {"type": "boolean"}
  },
  "required": ["name", "price", "in_stock"]
}
```

### From Zod

```typescript
import { zodToJsonSchema } from "zod-to-json-schema";

const Product = z.object({
  name: z.string(),
  price: z.number(),
  inStock: z.boolean(),
});

const jsonSchema = zodToJsonSchema(Product);
console.log(JSON.stringify(jsonSchema, null, 2));
```

---

## Validation Utilities

### Pydantic Validation

```python
from pydantic import BaseModel, ValidationError, field_validator

class ValidatedResponse(BaseModel):
    score: float
    reasoning: str
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v

# Validation happens automatically
try:
    response = ValidatedResponse(score=1.5, reasoning="test")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Post-Response Validation

```python
def parse_with_validation(response, model_class):
    """Parse and validate response"""
    
    parsed = response.choices[0].message.parsed
    
    if parsed is None:
        # Model refused to answer
        refusal = response.choices[0].message.refusal
        raise ValueError(f"Model refused: {refusal}")
    
    # Additional business logic validation
    if hasattr(parsed, 'score') and parsed.score < 0:
        raise ValueError("Negative scores not allowed")
    
    return parsed
```

### Zod Validation

```typescript
const ValidatedResponse = z.object({
  score: z.number().min(0).max(1),
  reasoning: z.string().min(10),
});

function parseWithValidation(response: any) {
  const result = ValidatedResponse.safeParse(response);
  
  if (!result.success) {
    console.error("Validation errors:", result.error.issues);
    throw new Error("Invalid response structure");
  }
  
  return result.data;
}
```

---

## Handling Refusals

### Python

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}],
    response_format=MyModel
)

message = response.choices[0].message

if message.refusal:
    # Model refused to answer
    print(f"Refused: {message.refusal}")
elif message.parsed:
    # Successfully parsed
    result = message.parsed
else:
    # Unexpected state
    print("No parsed content or refusal")
```

### TypeScript

```typescript
const response = await client.beta.chat.completions.parse({
  model: "gpt-4o",
  messages: [{ role: "user", content: "..." }],
  response_format: zodResponseFormat(MySchema, "my_schema"),
});

const message = response.choices[0].message;

if (message.refusal) {
  console.log(`Refused: ${message.refusal}`);
} else if (message.parsed) {
  const result = message.parsed;
}
```

---

## Summary

✅ **Pydantic**: Python's go-to for structured outputs

✅ **Zod**: TypeScript's type-safe validation

✅ **Auto-generation**: Convert models to JSON Schema

✅ **Validation**: Built-in constraints and custom validators

✅ **Refusals**: Handle when model can't comply

**Next:** [JSON Schema Constraints](./04-json-schema-constraints.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Response Format](./02-response-format.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [JSON Schema Constraints](./04-json-schema-constraints.md) |
