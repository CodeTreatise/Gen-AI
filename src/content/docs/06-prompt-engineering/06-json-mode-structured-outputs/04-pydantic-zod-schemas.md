---
title: "Pydantic and Zod Schema Definitions"
---

# Pydantic and Zod Schema Definitions

## Introduction

Writing JSON schemas by hand is tedious and error-prone. Pydantic (Python) and Zod (TypeScript) let you define schemas using native type systems. The SDKs automatically convert these to JSON Schema for Structured Outputs.

> **🤖 AI Context:** Schema-first development ensures your AI outputs are type-safe from generation through consumption. Your IDE catches errors before they reach production.

### What We'll Cover

- Pydantic schema definitions for Python
- Zod schema definitions for TypeScript
- Field constraints and validation
- Nested and complex structures
- Best practices for schema design

### Prerequisites

- [Structured Outputs with Schemas](./02-structured-outputs-schemas.md)
- [Response Format Parameter](./03-response-format-parameter.md)

---

## Pydantic for Python

### Basic Schema Definition

```python
from pydantic import BaseModel
from typing import Optional, List

class UserProfile(BaseModel):
    username: str
    email: str
    age: int
    is_active: bool

# Usage with OpenAI
from openai import OpenAI
client = OpenAI()

response = client.responses.parse(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "Create a profile for john_doe, john@example.com, 28 years old, active user"}
    ],
    text_format=UserProfile
)

user = response.output_parsed
print(f"Username: {user.username}")  # "john_doe"
print(f"Email: {user.email}")        # "john@example.com"
```

### Type Mapping

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[T]` / `List[T]` | `"array"` with items |
| `dict[str, T]` | `"object"` |
| `Optional[T]` | union with null |
| `Literal["a", "b"]` | `"enum"` |

---

## Field Constraints with Pydantic

### Using `Field()` for Constraints

```python
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str = Field(
        description="Product name",
        min_length=1,
        max_length=100
    )
    price: float = Field(
        description="Price in USD",
        ge=0,  # greater than or equal
        le=10000  # less than or equal
    )
    rating: float = Field(
        description="Rating from 0 to 5",
        ge=0,
        le=5
    )
    tags: List[str] = Field(
        description="Product tags",
        min_length=1,  # At least one tag
        max_length=10  # Max 10 tags
    )
```

### Common Field Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `description` | Help model understand field | `"User's email address"` |
| `min_length` / `max_length` | String/list length | `min_length=1` |
| `ge` / `le` | Number range (inclusive) | `ge=0, le=100` |
| `gt` / `lt` | Number range (exclusive) | `gt=0` |
| `pattern` | Regex pattern | `pattern=r"^\d{5}$"` |
| `default` | Default value | `default="Unknown"` |

### Pattern Matching

```python
from pydantic import BaseModel, Field

class PhoneNumber(BaseModel):
    country_code: str = Field(
        pattern=r"^\+\d{1,3}$",
        description="Country code starting with +"
    )
    number: str = Field(
        pattern=r"^\d{10}$",
        description="10-digit phone number"
    )
```

---

## Enums and Literals

### Using Literal for Fixed Values

```python
from pydantic import BaseModel
from typing import Literal

class SupportTicket(BaseModel):
    priority: Literal["low", "medium", "high", "critical"]
    category: Literal["billing", "technical", "account", "other"]
    status: Literal["open", "in_progress", "resolved", "closed"]
    summary: str
```

### Using Python Enum

```python
from pydantic import BaseModel
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"

class Ticket(BaseModel):
    priority: Priority
    category: Category
    description: str

# Access enum values
ticket = Ticket(priority=Priority.HIGH, category=Category.TECHNICAL, description="...")
print(ticket.priority.value)  # "high"
```

---

## Optional Fields

### Handling Optional Data

In Structured Outputs, all fields must be "required" at the schema level. Use `Optional` with `None` to indicate missing values:

```python
from pydantic import BaseModel
from typing import Optional

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None  # Can be null
    company: Optional[str] = None
    website: Optional[str] = None

# Generated JSON Schema includes null in type union
# "phone": {"type": ["string", "null"]}
```

### With Descriptions

```python
from pydantic import BaseModel, Field
from typing import Optional

class PersonExtraction(BaseModel):
    full_name: str = Field(description="Person's full name")
    email: Optional[str] = Field(
        default=None,
        description="Email address if mentioned, otherwise null"
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number if mentioned, otherwise null"
    )
```

---

## Nested Objects

### Simple Nesting

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class Customer(BaseModel):
    name: str
    email: str
    billing_address: Address
    shipping_address: Address
```

### Lists of Nested Objects

```python
class LineItem(BaseModel):
    product_name: str
    quantity: int
    unit_price: float

class Order(BaseModel):
    order_id: str
    customer_name: str
    items: List[LineItem]
    total: float

# Usage
response = client.responses.parse(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "Order #123: 2x Widget ($10 each), 1x Gadget ($25)"}
    ],
    text_format=Order
)

order = response.output_parsed
for item in order.items:
    print(f"{item.quantity}x {item.product_name}: ${item.unit_price}")
```

---

## Zod for TypeScript

### Basic Schema Definition

```typescript
import OpenAI from "openai";
import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

const client = new OpenAI();

// Define schema with Zod
const UserProfile = z.object({
    username: z.string(),
    email: z.string().email(),
    age: z.number().int().min(0),
    is_active: z.boolean()
});

const response = await client.chat.completions.parse({
    model: "gpt-4o-2024-08-06",
    messages: [
        { role: "user", content: "Create profile for john_doe, john@example.com, 28, active" }
    ],
    response_format: zodResponseFormat(UserProfile, "user_profile")
});

const user = response.choices[0].message.parsed;
console.log(user.username);  // "john_doe"
```

### Zod Type Mapping

| Zod Type | JSON Schema |
|----------|-------------|
| `z.string()` | `"string"` |
| `z.number()` | `"number"` |
| `z.number().int()` | `"integer"` |
| `z.boolean()` | `"boolean"` |
| `z.array(T)` | `"array"` |
| `z.object({...})` | `"object"` |
| `z.enum([...])` | `"enum"` |
| `z.nullable(T)` | union with null |

### Zod Constraints

```typescript
const Product = z.object({
    name: z.string()
        .min(1)
        .max(100)
        .describe("Product name"),
    
    price: z.number()
        .min(0)
        .max(10000)
        .describe("Price in USD"),
    
    category: z.enum(["electronics", "clothing", "food", "other"])
        .describe("Product category"),
    
    tags: z.array(z.string())
        .min(1)
        .max(10)
        .describe("Product tags")
});
```

### Nested Objects in Zod

```typescript
const Address = z.object({
    street: z.string(),
    city: z.string(),
    country: z.string()
});

const Person = z.object({
    name: z.string(),
    email: z.string().email(),
    addresses: z.array(Address),
    primary_address: Address.nullable()  // Optional field
});
```

---

## Advanced Patterns

### Union Types with Discriminator

```python
from pydantic import BaseModel, Field
from typing import Literal, Union

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageContent(BaseModel):
    type: Literal["image"]
    url: str
    alt_text: str

class Message(BaseModel):
    role: str
    content: Union[TextContent, ImageContent] = Field(
        discriminator="type"
    )
```

### Self-Referential Structures

```python
from pydantic import BaseModel
from typing import List, Optional

class Comment(BaseModel):
    author: str
    text: str
    replies: List["Comment"] = []

# Required for self-reference
Comment.model_rebuild()
```

### Recursive Trees

```python
from pydantic import BaseModel
from typing import List, Optional

class FileNode(BaseModel):
    name: str
    is_directory: bool
    children: Optional[List["FileNode"]] = None
    size_bytes: Optional[int] = None

FileNode.model_rebuild()

# Can represent:
# {
#     "name": "src",
#     "is_directory": true,
#     "children": [
#         {"name": "main.py", "is_directory": false, "size_bytes": 1024},
#         {"name": "utils", "is_directory": true, "children": [...]}
#     ]
# }
```

---

## Schema Design Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use descriptive field names | Model understands intent better |
| Add descriptions to fields | Improves extraction accuracy |
| Keep schemas focused | One schema per task |
| Use enums for categories | Constrains output to valid values |
| Validate at boundaries | Check after generation, before use |
| Version your schemas | Track changes over time |

### Example: Well-Designed Schema

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime

class AnalysisResult(BaseModel):
    """Sentiment analysis result for a piece of text."""
    
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment classification"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    
    key_phrases: List[str] = Field(
        description="Important phrases that influenced the sentiment",
        max_length=5
    )
    
    summary: str = Field(
        description="One-sentence summary of the sentiment reasoning",
        max_length=200
    )
    
    topics: Optional[List[str]] = Field(
        default=None,
        description="Main topics discussed, if identifiable"
    )
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Missing `model_rebuild()` for self-reference | Call after class definition |
| No descriptions on fields | Add `Field(description=...)` |
| Too complex schemas | Split into smaller pieces |
| Not using enums for categories | Use `Literal` or `Enum` |
| Ignoring validation errors | Handle Pydantic `ValidationError` |

---

## Hands-on Exercise

### Your Task

Create a Pydantic schema for extracting recipe information.

### Requirements

1. Recipe name, servings, prep_time_minutes, cook_time_minutes
2. List of ingredients with: name, amount, unit (enum)
3. List of instruction steps (strings)
4. Tags (optional list of dietary tags)
5. Difficulty level (enum: easy, medium, hard)

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Literal` for difficulty and unit enums
- Ingredients should be a list of nested objects
- Make tags optional with `Optional[List[str]]`
- Add descriptions to help the model

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class Ingredient(BaseModel):
    """Single ingredient with measurement."""
    name: str = Field(description="Ingredient name")
    amount: float = Field(
        ge=0,
        description="Quantity needed"
    )
    unit: Literal["cup", "tbsp", "tsp", "oz", "lb", "g", "ml", "piece", "clove", "pinch"] = Field(
        description="Unit of measurement"
    )

class Recipe(BaseModel):
    """Complete recipe extraction schema."""
    name: str = Field(
        description="Recipe name or title"
    )
    servings: int = Field(
        ge=1,
        description="Number of servings this recipe makes"
    )
    prep_time_minutes: int = Field(
        ge=0,
        description="Preparation time in minutes"
    )
    cook_time_minutes: int = Field(
        ge=0,
        description="Cooking time in minutes"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Recipe difficulty level"
    )
    ingredients: List[Ingredient] = Field(
        description="List of required ingredients"
    )
    instructions: List[str] = Field(
        description="Step-by-step cooking instructions"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Dietary tags like vegetarian, vegan, gluten-free"
    )

# Usage
from openai import OpenAI
client = OpenAI()

recipe_text = """
Simple Garlic Pasta (4 servings)
Easy, 10 min prep, 15 min cook
Vegetarian

Ingredients:
- 1 lb spaghetti
- 4 cloves garlic, minced
- 1/4 cup olive oil
- Pinch of red pepper flakes

Instructions:
1. Cook pasta according to package directions
2. Sauté garlic in olive oil until fragrant
3. Toss pasta with garlic oil
4. Add red pepper flakes and serve
"""

response = client.responses.parse(
    model="gpt-4o",
    input=[
        {
            "role": "system",
            "content": "Extract recipe information from the provided text."
        },
        {"role": "user", "content": recipe_text}
    ],
    text_format=Recipe
)

recipe = response.output_parsed
print(f"Recipe: {recipe.name}")
print(f"Servings: {recipe.servings}, Difficulty: {recipe.difficulty}")
print(f"Total time: {recipe.prep_time_minutes + recipe.cook_time_minutes} minutes")
print("\nIngredients:")
for ing in recipe.ingredients:
    print(f"  - {ing.amount} {ing.unit} {ing.name}")
print("\nInstructions:")
for i, step in enumerate(recipe.instructions, 1):
    print(f"  {i}. {step}")
if recipe.tags:
    print(f"\nTags: {', '.join(recipe.tags)}")
```

**Expected output:**
```
Recipe: Simple Garlic Pasta
Servings: 4, Difficulty: easy
Total time: 25 minutes

Ingredients:
  - 1.0 lb spaghetti
  - 4.0 clove garlic
  - 0.25 cup olive oil
  - 1.0 pinch red pepper flakes

Instructions:
  1. Cook pasta according to package directions
  2. Sauté garlic in olive oil until fragrant
  3. Toss pasta with garlic oil
  4. Add red pepper flakes and serve

Tags: vegetarian
```

</details>

### Bonus Challenge

- [ ] Add nutritional info as an optional nested object
- [ ] Include source URL as optional field

---

## Summary

✅ **Pydantic** provides type-safe schema definitions for Python

✅ **Zod** provides type-safe schema definitions for TypeScript

✅ **Field constraints** map to JSON Schema validation

✅ **Nested objects** and lists work naturally

✅ **Descriptions** improve model understanding

**Next:** [Ensuring Valid JSON](./05-ensuring-valid-json.md)

---

## Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Field Types](https://docs.pydantic.dev/latest/concepts/fields/)
- [Zod Documentation](https://zod.dev/)
- [OpenAI Structured Outputs with Pydantic](https://platform.openai.com/docs/guides/structured-outputs)

---

<!-- 
Sources Consulted:
- Pydantic documentation: https://docs.pydantic.dev/
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
- Zod documentation: https://zod.dev/
-->
