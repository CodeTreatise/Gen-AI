---
title: "Pydantic Output Parser"
---

# Pydantic Output Parser

## Introduction

PydanticOutputParser is the gold standard for structured LLM output. It combines Python's most popular validation library with automatic format instructions, ensuring LLM responses match your exact data schemas. This lesson covers everything from basic usage to complex nested models.

### What We'll Cover

- PydanticOutputParser fundamentals
- Defining Pydantic models for LLM output
- Nested and complex structures
- Field validation and constraints
- Automatic format instructions
- Error handling and validation

### Prerequisites

- Parser Basics (Lesson 8.3.1)
- Pydantic models (basic understanding)
- LCEL chains

---

## PydanticOutputParser Fundamentals

### Basic Usage

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    genre: str

# Create parser from Pydantic model
parser = PydanticOutputParser(pydantic_object=Movie)

# Parse JSON text into Movie object
json_text = '{"title": "Inception", "year": 2010, "genre": "Sci-Fi"}'
movie = parser.parse(json_text)

print(movie)        # title='Inception' year=2010 genre='Sci-Fi'
print(movie.title)  # Inception
print(type(movie))  # <class '__main__.Movie'>
```

### Format Instructions

PydanticOutputParser automatically generates format instructions from your model:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")

parser = PydanticOutputParser(pydantic_object=Movie)

print(parser.get_format_instructions())
```

**Output:**
```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"title": {"description": "The movie title", "title": "Title", "type": "string"}, "year": {"description": "Release year", "title": "Year", "type": "integer"}, "genre": {"description": "Primary genre", "title": "Genre", "type": "string"}}, "required": ["title", "year", "genre"]}
```
```

---

## Complete Chain Example

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class BookRecommendation(BaseModel):
    """A book recommendation with details."""
    title: str = Field(description="The book title")
    author: str = Field(description="The author name")
    summary: str = Field(description="Brief summary in 2-3 sentences")
    why_read: str = Field(description="Why someone should read this book")

# Create parser
parser = PydanticOutputParser(pydantic_object=BookRecommendation)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a book recommendation expert.
Always respond with valid JSON matching this format:
{format_instructions}"""),
    ("human", "Recommend a book about {topic}")
])

# Initialize model
model = init_chat_model("gpt-4o")

# Create chain
chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

# Invoke
book = chain.invoke({"topic": "machine learning"})
print(f"Title: {book.title}")
print(f"Author: {book.author}")
print(f"Summary: {book.summary}")
print(f"Why read: {book.why_read}")
```

---

## Defining Effective Pydantic Models

### Using Field Descriptions

Field descriptions are included in format instructions, guiding the LLM:

```python
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    """Analysis of a product review."""
    
    sentiment: str = Field(
        description="Overall sentiment: 'positive', 'negative', or 'neutral'"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    key_points: list[str] = Field(
        description="List of 3-5 main points from the review"
    )
    product_aspects: dict[str, str] = Field(
        description="Aspects mentioned (e.g., 'quality': 'excellent')"
    )
```

### Optional Fields

```python
from pydantic import BaseModel, Field
from typing import Optional

class UserProfile(BaseModel):
    """User profile extracted from text."""
    
    name: str = Field(description="Full name")
    email: Optional[str] = Field(
        default=None,
        description="Email address if mentioned"
    )
    age: Optional[int] = Field(
        default=None,
        description="Age if mentioned"
    )
    interests: list[str] = Field(
        default_factory=list,
        description="List of interests or hobbies"
    )
```

### Enum Fields

```python
from pydantic import BaseModel, Field
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskExtraction(BaseModel):
    """Extracted task from text."""
    
    title: str = Field(description="Task title")
    priority: Priority = Field(description="Task priority level")
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline if mentioned (YYYY-MM-DD format)"
    )
```

---

## Nested and Complex Structures

### Nested Models

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import list

class Address(BaseModel):
    """Physical address."""
    street: str
    city: str
    country: str
    postal_code: str

class ContactInfo(BaseModel):
    """Contact information."""
    email: str
    phone: Optional[str] = None

class Person(BaseModel):
    """Person with nested information."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    address: Address = Field(description="Home address")
    contact: ContactInfo = Field(description="Contact details")

# Parser handles nested structure
parser = PydanticOutputParser(pydantic_object=Person)

json_text = '''
{
    "name": "Alice Johnson",
    "age": 32,
    "address": {
        "street": "123 Main St",
        "city": "Boston",
        "country": "USA",
        "postal_code": "02101"
    },
    "contact": {
        "email": "alice@example.com",
        "phone": "555-0123"
    }
}
'''

person = parser.parse(json_text)
print(person.address.city)  # Boston
print(person.contact.email)  # alice@example.com
```

### Lists of Objects

```python
from pydantic import BaseModel, Field

class Step(BaseModel):
    """A single step in a process."""
    number: int = Field(description="Step number")
    action: str = Field(description="What to do")
    duration: Optional[str] = Field(default=None, description="Time estimate")

class Recipe(BaseModel):
    """A recipe with steps."""
    name: str = Field(description="Recipe name")
    servings: int = Field(description="Number of servings")
    ingredients: list[str] = Field(description="List of ingredients")
    steps: list[Step] = Field(description="Ordered list of steps")

parser = PydanticOutputParser(pydantic_object=Recipe)
```

### Union Types

```python
from pydantic import BaseModel, Field
from typing import Union

class TextResponse(BaseModel):
    """Simple text response."""
    type: str = "text"
    content: str

class DataResponse(BaseModel):
    """Structured data response."""
    type: str = "data"
    data: dict
    source: str

class APIResponse(BaseModel):
    """API response wrapper."""
    success: bool
    response: Union[TextResponse, DataResponse]
```

---

## Validation and Constraints

### Built-in Validators

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class Survey(BaseModel):
    """Survey response with validation."""
    
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    feedback: str = Field(min_length=10, max_length=500)
    tags: list[str] = Field(min_length=1, max_length=5)
    
    @field_validator('feedback')
    @classmethod
    def feedback_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Feedback cannot be empty')
        return v.strip()
```

### Custom Validation

```python
from pydantic import BaseModel, Field, model_validator

class DateRange(BaseModel):
    """Date range with validation."""
    
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    
    @model_validator(mode='after')
    def validate_date_range(self):
        if self.start_date > self.end_date:
            raise ValueError('start_date must be before end_date')
        return self
```

### Validation Error Handling

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="Price must be positive")
    quantity: int = Field(ge=0, description="Quantity cannot be negative")

parser = PydanticOutputParser(pydantic_object=Product)

# Invalid data - negative price
invalid_json = '{"name": "Widget", "price": -10, "quantity": 5}'

try:
    product = parser.parse(invalid_json)
except OutputParserException as e:
    print(f"Validation failed: {e}")
    # Validation failed: ... Input should be greater than 0
```

---

## Advanced Patterns

### Multiple Models with Discriminated Unions

```python
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated

class EmailAction(BaseModel):
    """Send email action."""
    action_type: Literal["email"] = "email"
    recipient: str
    subject: str
    body: str

class SlackAction(BaseModel):
    """Send Slack message action."""
    action_type: Literal["slack"] = "slack"
    channel: str
    message: str

class APICallAction(BaseModel):
    """Make API call action."""
    action_type: Literal["api_call"] = "api_call"
    endpoint: str
    method: str
    payload: dict

# Use discriminated union
Action = Annotated[
    Union[EmailAction, SlackAction, APICallAction],
    Field(discriminator="action_type")
]

class Workflow(BaseModel):
    """Workflow with multiple action types."""
    name: str
    actions: list[Action]
```

### Generic Models

```python
from pydantic import BaseModel, Field
from typing import Generic, TypeVar

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: list[T]
    total: int
    page: int
    per_page: int
    
    @property
    def has_more(self) -> bool:
        return self.page * self.per_page < self.total
```

### Model with Computed Fields

```python
from pydantic import BaseModel, Field, computed_field

class Invoice(BaseModel):
    """Invoice with computed total."""
    
    items: list[dict] = Field(description="List of {name, price, quantity}")
    tax_rate: float = Field(default=0.1, description="Tax rate (0.1 = 10%)")
    
    @computed_field
    @property
    def subtotal(self) -> float:
        return sum(item['price'] * item['quantity'] for item in self.items)
    
    @computed_field
    @property
    def tax(self) -> float:
        return self.subtotal * self.tax_rate
    
    @computed_field
    @property
    def total(self) -> float:
        return self.subtotal + self.tax
```

---

## Integration Patterns

### With Prompt Templates

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    summary: str = Field(description="One paragraph summary")
    key_insights: list[str] = Field(description="3-5 key insights")
    sentiment: str = Field(description="Overall sentiment")
    recommendations: list[str] = Field(description="Action recommendations")

parser = PydanticOutputParser(pydantic_object=Analysis)

# Template with partial for format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the text and respond in JSON. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

model = init_chat_model("gpt-4o")
chain = prompt | model | parser

result = chain.invoke({"text": "Long document text here..."})
```

### Async Usage

```python
import asyncio
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

class Response(BaseModel):
    answer: str

parser = PydanticOutputParser(pydantic_object=Response)

async def async_parse():
    # Async parsing
    result = await parser.aparse('{"answer": "Hello async!"}')
    return result

# Run
result = asyncio.run(async_parse())
print(result.answer)  # Hello async!
```

### Batch Processing

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Summary(BaseModel):
    title: str
    summary: str = Field(max_length=100)

parser = PydanticOutputParser(pydantic_object=Summary)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize briefly. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

model = init_chat_model("gpt-4o")
chain = prompt | model | parser

# Batch invoke
texts = [
    {"text": "Article 1 content..."},
    {"text": "Article 2 content..."},
    {"text": "Article 3 content..."}
]

results = chain.batch(texts)
for r in results:
    print(f"{r.title}: {r.summary}")
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always use Field descriptions | Guides LLM output format |
| Set appropriate constraints | Validates output automatically |
| Use Optional for nullable fields | Handles missing data gracefully |
| Include examples in docstrings | Improves LLM understanding |
| Use discriminated unions | Clean handling of multiple types |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No field descriptions | Add descriptions to all fields |
| Too complex nested structures | Flatten or simplify models |
| Missing validation | Add Field constraints |
| Not handling validation errors | Wrap in try/except |
| Ignoring Optional fields | Use default values |

---

## Hands-on Exercise

### Your Task

Build a job posting analyzer that extracts structured data:

1. Create a Pydantic model for job postings
2. Include nested models for requirements and benefits
3. Add field validation
4. Build a complete chain

### Requirements

Model should include:
- Job title, company, location
- Salary range (min/max)
- Required skills (list)
- Experience level (enum)
- Benefits (list of objects)

### Expected Result

```python
result = chain.invoke({"job_text": """
    Senior Python Developer at TechCorp
    San Francisco, CA - $150,000 - $200,000
    Requirements: 5+ years Python, Django, PostgreSQL
    Benefits: Health insurance, 401k match, Remote work
"""})

print(result.title)  # "Senior Python Developer"
print(result.salary.min_amount)  # 150000
print(result.skills)  # ["Python", "Django", "PostgreSQL"]
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Create SalaryRange model with min_amount and max_amount
- Use Enum for experience level
- Create Benefit model with name and description
- Add Field constraints for validation

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"

class SalaryRange(BaseModel):
    """Salary range for the position."""
    min_amount: int = Field(ge=0, description="Minimum salary")
    max_amount: int = Field(ge=0, description="Maximum salary")
    currency: str = Field(default="USD", description="Currency code")

class Benefit(BaseModel):
    """A company benefit."""
    name: str = Field(description="Benefit name")
    description: Optional[str] = Field(default=None, description="Details")

class JobPosting(BaseModel):
    """Structured job posting data."""
    
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    remote: bool = Field(description="Is remote work available")
    
    salary: Optional[SalaryRange] = Field(
        default=None,
        description="Salary range if mentioned"
    )
    
    experience_level: ExperienceLevel = Field(
        description="Required experience level"
    )
    
    skills: list[str] = Field(
        min_length=1,
        description="Required skills and technologies"
    )
    
    years_experience: Optional[int] = Field(
        default=None,
        ge=0,
        description="Years of experience required"
    )
    
    benefits: list[Benefit] = Field(
        default_factory=list,
        description="Company benefits"
    )

# Create parser
parser = PydanticOutputParser(pydantic_object=JobPosting)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract structured data from job postings.
Respond with valid JSON matching this schema:
{format_instructions}"""),
    ("human", "{job_text}")
]).partial(format_instructions=parser.get_format_instructions())

# Create chain
model = init_chat_model("gpt-4o")
chain = prompt | model | parser

# Test
job_text = """
    Senior Python Developer at TechCorp
    San Francisco, CA - $150,000 - $200,000/year
    
    Requirements:
    - 5+ years Python experience
    - Strong Django and PostgreSQL skills
    - Experience with AWS
    
    Benefits:
    - Comprehensive health insurance
    - 401k with 4% match
    - Flexible remote work policy
    - Unlimited PTO
"""

result = chain.invoke({"job_text": job_text})

print(f"Title: {result.title}")
print(f"Company: {result.company}")
print(f"Location: {result.location}")
print(f"Remote: {result.remote}")
print(f"Salary: ${result.salary.min_amount:,} - ${result.salary.max_amount:,}")
print(f"Level: {result.experience_level.value}")
print(f"Skills: {', '.join(result.skills)}")
print(f"Years: {result.years_experience}+")
print(f"Benefits: {[b.name for b in result.benefits]}")
```

</details>

### Bonus Challenges

- [ ] Add validation that max_salary > min_salary
- [ ] Extract contact information if present
- [ ] Compare multiple job postings
- [ ] Handle international salary formats

---

## Summary

✅ `PydanticOutputParser` creates parsers from Pydantic models  
✅ Field descriptions guide LLM output through format instructions  
✅ Nested models handle complex hierarchical data  
✅ Validation constraints ensure data quality  
✅ Enum fields restrict values to valid options  
✅ Use with LCEL chains for seamless integration  

**Next:** [JSON Parser](./03-json-output-parser.md) — Flexible JSON parsing with streaming support

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Parser Basics](./01-parser-basics.md) | [Output Parsing](./00-output-parsing.md) | [JSON Parser](./03-json-output-parser.md) |

<!-- 
Sources Consulted:
- LangChain PydanticOutputParser: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/output_parsers/pydantic.py
- Pydantic Field documentation: https://docs.pydantic.dev/latest/concepts/fields/
- LangChain output parsers: https://python.langchain.com/docs/concepts/output_parsers/
-->
