---
title: "Use Cases"
---

# Use Cases

## Introduction

Structured outputs enable reliable machine-readable responses for a variety of applications. This lesson covers practical use cases with implementation examples.

### What We'll Cover

- Data extraction from unstructured text
- UI component generation
- Form validation responses
- API response formatting
- Classification with confidence scores

---

## Data Extraction

### Extract Structured Data from Text

```python
from pydantic import BaseModel
from openai import OpenAI

class ContactInfo(BaseModel):
    name: str
    email: str | None
    phone: str | None
    company: str | None
    title: str | None

client = OpenAI()

def extract_contact(text: str) -> ContactInfo:
    """Extract contact info from unstructured text"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract contact information from the text. Use null for missing fields."
            },
            {"role": "user", "content": text}
        ],
        response_format=ContactInfo
    )
    
    return response.choices[0].message.parsed

# Usage
text = """
Hi, I'm Sarah Chen from TechCorp. 
You can reach me at sarah.chen@techcorp.com or call 555-123-4567.
I'm the VP of Engineering.
"""

contact = extract_contact(text)
print(f"Name: {contact.name}")
print(f"Email: {contact.email}")
print(f"Company: {contact.company}")
```

### Invoice Data Extraction

```python
class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float

def extract_invoice(text: str) -> Invoice:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract invoice data from the text."
            },
            {"role": "user", "content": text}
        ],
        response_format=Invoice
    )
    return response.choices[0].message.parsed
```

---

## UI Component Generation

### Dynamic Form Generation

```python
from typing import Literal

class FormField(BaseModel):
    name: str
    label: str
    type: Literal["text", "email", "number", "select", "checkbox", "textarea"]
    required: bool
    placeholder: str | None
    options: list[str] | None  # For select fields
    validation: str | None  # Regex pattern

class GeneratedForm(BaseModel):
    title: str
    description: str
    fields: list[FormField]
    submit_button_text: str

def generate_form(description: str) -> GeneratedForm:
    """Generate form schema from natural language"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Generate a form schema based on the description."
            },
            {"role": "user", "content": description}
        ],
        response_format=GeneratedForm
    )
    return response.choices[0].message.parsed

# Usage
form = generate_form("A user registration form with name, email, password, and age verification")

# Convert to React JSX, HTML, etc.
for field in form.fields:
    print(f'<Input name="{field.name}" type="{field.type}" required={field.required} />')
```

### Card Component Generation

```python
class CardComponent(BaseModel):
    title: str
    subtitle: str | None
    body: str
    image_description: str | None
    cta_text: str | None
    cta_url: str | None
    tags: list[str]

class CardGrid(BaseModel):
    cards: list[CardComponent]
    layout: Literal["2-column", "3-column", "masonry"]

def generate_cards(content: str, count: int = 3) -> CardGrid:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Generate {count} card components from the content."
            },
            {"role": "user", "content": content}
        ],
        response_format=CardGrid
    )
    return response.choices[0].message.parsed
```

---

## Form Validation Responses

### Validation Result Schema

```python
class FieldError(BaseModel):
    field: str
    message: str
    suggestion: str | None

class ValidationResult(BaseModel):
    valid: bool
    errors: list[FieldError]
    warnings: list[str]

def validate_form_data(form_data: dict, rules: str) -> ValidationResult:
    """Validate form data with AI"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Validate the form data against these rules: {rules}"
            },
            {"role": "user", "content": json.dumps(form_data)}
        ],
        response_format=ValidationResult
    )
    return response.choices[0].message.parsed

# Usage
result = validate_form_data(
    {"email": "invalid", "age": -5},
    "Email must be valid format. Age must be positive."
)

for error in result.errors:
    print(f"{error.field}: {error.message}")
```

---

## API Response Formatting

### Standardized API Responses

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None
    error: str | None
    metadata: dict | None

class ProductData(BaseModel):
    id: str
    name: str
    price: float
    in_stock: bool

def format_product_response(raw_text: str) -> APIResponse[ProductData]:
    """Format unstructured data as API response"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract product data and format as API response."
            },
            {"role": "user", "content": raw_text}
        ],
        response_format=APIResponse[ProductData]
    )
    return response.choices[0].message.parsed
```

### Error Response Formatting

```python
class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None
    docs_url: str | None

class ErrorResponse(BaseModel):
    status: Literal["error"]
    errors: list[ErrorDetail]
    request_id: str
    timestamp: str

def format_error(exception: Exception, context: str) -> ErrorResponse:
    """Generate user-friendly error response"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Format this error as a user-friendly API error response."
            },
            {"role": "user", "content": f"Error: {exception}\nContext: {context}"}
        ],
        response_format=ErrorResponse
    )
    return response.choices[0].message.parsed
```

---

## Classification with Confidence

### Multi-Label Classification

```python
class Classification(BaseModel):
    label: str
    confidence: float
    reasoning: str

class ClassificationResult(BaseModel):
    primary_category: Classification
    secondary_categories: list[Classification]
    sentiment: Literal["positive", "negative", "neutral"]
    sentiment_confidence: float

def classify_text(text: str, categories: list[str]) -> ClassificationResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Classify the text into these categories: {categories}. Provide confidence scores."
            },
            {"role": "user", "content": text}
        ],
        response_format=ClassificationResult
    )
    return response.choices[0].message.parsed

# Usage
result = classify_text(
    "The new iPhone camera is amazing but battery life could be better",
    ["Technology", "Review", "Complaint", "Praise"]
)
print(f"Primary: {result.primary_category.label} ({result.primary_category.confidence:.0%})")
```

### Intent Detection

```python
class Intent(BaseModel):
    name: str
    confidence: float
    parameters: dict

class IntentResult(BaseModel):
    detected_intent: Intent
    alternative_intents: list[Intent]
    requires_clarification: bool
    clarification_question: str | None

def detect_intent(utterance: str, possible_intents: list[str]) -> IntentResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Detect intent from: {possible_intents}"
            },
            {"role": "user", "content": utterance}
        ],
        response_format=IntentResult
    )
    return response.choices[0].message.parsed
```

---

## Content Generation

### Blog Post Structure

```python
class Section(BaseModel):
    heading: str
    content: str
    subsections: list[str]

class BlogPost(BaseModel):
    title: str
    meta_description: str
    introduction: str
    sections: list[Section]
    conclusion: str
    tags: list[str]
    estimated_read_time: int

def generate_blog_structure(topic: str) -> BlogPost:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Generate a complete blog post structure."
            },
            {"role": "user", "content": f"Topic: {topic}"}
        ],
        response_format=BlogPost
    )
    return response.choices[0].message.parsed
```

---

## Summary

✅ **Data extraction**: Contacts, invoices, receipts

✅ **UI generation**: Forms, cards, layouts

✅ **Validation**: Structured error feedback

✅ **API formatting**: Consistent response structures

✅ **Classification**: Labels with confidence scores

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Function Schemas](./06-function-schemas.md) | [Structured Outputs](./00-structured-outputs-json-mode.md) | [Model Benchmarks](../13-model-benchmarks-evaluation/00-model-benchmarks-evaluation.md) |
