---
title: "Use Cases & Patterns"
---

# Use Cases & Patterns

## Introduction

Structured Outputs excels in scenarios requiring reliable, typed data extraction. This lesson explores common patterns and production-ready implementations for data extraction, classification, reasoning, and UI generation.

### What We'll Cover

- Data extraction patterns
- Classification with enums
- Chain-of-thought reasoning
- UI component generation
- API response formatting

### Prerequisites

- Structured Outputs fundamentals
- Schema design knowledge
- Application architecture patterns

---

## Data Extraction Patterns

### Entity Extraction

```python
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """Types of entities to extract."""
    
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"


class Entity(BaseModel):
    """An extracted entity."""
    
    text: str = Field(description="The entity text as it appears")
    entity_type: EntityType = Field(description="Category of entity")
    normalized: Optional[str] = Field(
        default=None,
        description="Standardized form (e.g., date format)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Extraction confidence score"
    )


class ExtractedEntities(BaseModel):
    """Collection of extracted entities."""
    
    entities: List[Entity]
    source_text_length: int
    extraction_notes: Optional[str] = None


# Example schema for API
ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": ["person", "organization", "location", 
                                 "date", "money", "product"]
                    },
                    "normalized": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "confidence": {"type": "number"}
                },
                "required": ["text", "entity_type", "normalized", "confidence"],
                "additionalProperties": False
            }
        },
        "source_text_length": {"type": "integer"},
        "extraction_notes": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
    },
    "required": ["entities", "source_text_length", "extraction_notes"],
    "additionalProperties": False
}


print("Entity Extraction Schema")
print("=" * 60)

# Simulate extracted entities
sample_entities = ExtractedEntities(
    entities=[
        Entity(
            text="John Smith",
            entity_type=EntityType.PERSON,
            normalized="Smith, John",
            confidence=0.95
        ),
        Entity(
            text="Acme Corporation",
            entity_type=EntityType.ORGANIZATION,
            normalized=None,
            confidence=0.88
        ),
        Entity(
            text="January 15, 2025",
            entity_type=EntityType.DATE,
            normalized="2025-01-15",
            confidence=0.99
        )
    ],
    source_text_length=150,
    extraction_notes="All entities extracted with high confidence"
)

print("\n📋 Extracted Entities:")
for entity in sample_entities.entities:
    print(f"   {entity.entity_type.value}: {entity.text}")
    if entity.normalized:
        print(f"      → Normalized: {entity.normalized}")
    print(f"      Confidence: {entity.confidence:.0%}")
```

### Document Parsing

```python
class DocumentSection(BaseModel):
    """A section of a document."""
    
    title: str
    content: str
    section_type: str
    page_number: Optional[int] = None


class ParsedDocument(BaseModel):
    """Structured document representation."""
    
    title: str
    author: Optional[str] = None
    date: Optional[str] = None
    document_type: str
    sections: List[DocumentSection]
    key_points: List[str]
    summary: str


# Document parsing example
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "document_type": {
            "type": "string",
            "enum": ["report", "article", "memo", "contract", "email", "other"]
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "section_type": {"type": "string"},
                    "page_number": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                },
                "required": ["title", "content", "section_type", "page_number"],
                "additionalProperties": False
            }
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {"type": "string"}
    },
    "required": ["title", "author", "date", "document_type", 
                 "sections", "key_points", "summary"],
    "additionalProperties": False
}


print("\n\nDocument Parsing Pattern")
print("=" * 60)

sample_doc = ParsedDocument(
    title="Q4 Financial Report",
    author="Finance Team",
    date="2024-12-31",
    document_type="report",
    sections=[
        DocumentSection(
            title="Executive Summary",
            content="Q4 showed strong growth...",
            section_type="summary",
            page_number=1
        ),
        DocumentSection(
            title="Revenue Analysis",
            content="Revenue increased 15% YoY...",
            section_type="analysis",
            page_number=3
        )
    ],
    key_points=[
        "Revenue up 15% YoY",
        "Operating margin improved to 22%",
        "Customer acquisition cost reduced 8%"
    ],
    summary="Q4 2024 demonstrated strong financial performance..."
)

print(f"\n📄 {sample_doc.title}")
print(f"   Type: {sample_doc.document_type}")
print(f"   Sections: {len(sample_doc.sections)}")
print(f"   Key Points: {len(sample_doc.key_points)}")
```

---

## Classification Patterns

### Multi-Label Classification

```python
class SentimentLevel(str, Enum):
    """Sentiment intensity levels."""
    
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ContentCategory(str, Enum):
    """Content categories."""
    
    TECHNICAL = "technical"
    BUSINESS = "business"
    PERSONAL = "personal"
    NEWS = "news"
    ENTERTAINMENT = "entertainment"
    EDUCATIONAL = "educational"


class TextClassification(BaseModel):
    """Multi-label text classification."""
    
    primary_category: ContentCategory
    secondary_categories: List[ContentCategory]
    sentiment: SentimentLevel
    sentiment_confidence: float
    topics: List[str]
    language: str
    is_question: bool
    requires_response: bool


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_category": {
            "type": "string",
            "enum": ["technical", "business", "personal", 
                     "news", "entertainment", "educational"]
        },
        "secondary_categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["technical", "business", "personal", 
                         "news", "entertainment", "educational"]
            }
        },
        "sentiment": {
            "type": "string",
            "enum": ["very_negative", "negative", "neutral", 
                     "positive", "very_positive"]
        },
        "sentiment_confidence": {"type": "number"},
        "topics": {
            "type": "array",
            "items": {"type": "string"}
        },
        "language": {"type": "string"},
        "is_question": {"type": "boolean"},
        "requires_response": {"type": "boolean"}
    },
    "required": [
        "primary_category", "secondary_categories", "sentiment",
        "sentiment_confidence", "topics", "language",
        "is_question", "requires_response"
    ],
    "additionalProperties": False
}


print("\n\nMulti-Label Classification")
print("=" * 60)

sample_classification = TextClassification(
    primary_category=ContentCategory.TECHNICAL,
    secondary_categories=[ContentCategory.EDUCATIONAL],
    sentiment=SentimentLevel.POSITIVE,
    sentiment_confidence=0.87,
    topics=["machine learning", "python", "data science"],
    language="en",
    is_question=False,
    requires_response=False
)

print(f"\n🏷️ Classification Result:")
print(f"   Primary: {sample_classification.primary_category.value}")
print(f"   Secondary: {[c.value for c in sample_classification.secondary_categories]}")
print(f"   Sentiment: {sample_classification.sentiment.value} ({sample_classification.sentiment_confidence:.0%})")
print(f"   Topics: {sample_classification.topics}")
```

### Priority and Urgency Classification

```python
class Priority(str, Enum):
    """Task priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketClassification(BaseModel):
    """Support ticket classification."""
    
    category: str
    subcategory: str
    priority: Priority
    estimated_resolution_hours: int
    requires_escalation: bool
    suggested_assignee_team: str
    auto_response_appropriate: bool
    keywords: List[str]


print("\n\nTicket Classification Pattern")
print("=" * 60)

sample_ticket = TicketClassification(
    category="Technical Support",
    subcategory="Login Issues",
    priority=Priority.HIGH,
    estimated_resolution_hours=4,
    requires_escalation=False,
    suggested_assignee_team="Authentication Team",
    auto_response_appropriate=True,
    keywords=["login", "password", "authentication", "error"]
)

print(f"\n🎫 Ticket Classification:")
print(f"   Category: {sample_ticket.category} > {sample_ticket.subcategory}")
print(f"   Priority: {sample_ticket.priority.value}")
print(f"   Est. Resolution: {sample_ticket.estimated_resolution_hours}h")
print(f"   Escalation: {'Yes' if sample_ticket.requires_escalation else 'No'}")
```

---

## Chain-of-Thought Reasoning

### Structured Reasoning

```python
class ReasoningStep(BaseModel):
    """A single reasoning step."""
    
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None


class ReasonedAnswer(BaseModel):
    """Answer with explicit reasoning chain."""
    
    question: str
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    alternative_answers: List[str]
    assumptions_made: List[str]


REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reasoning_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "thought": {"type": "string"},
                    "action": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "observation": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
                "required": ["step_number", "thought", "action", "observation"],
                "additionalProperties": False
            }
        },
        "final_answer": {"type": "string"},
        "confidence": {"type": "number"},
        "alternative_answers": {
            "type": "array",
            "items": {"type": "string"}
        },
        "assumptions_made": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "question", "reasoning_steps", "final_answer",
        "confidence", "alternative_answers", "assumptions_made"
    ],
    "additionalProperties": False
}


print("\n\nChain-of-Thought Reasoning")
print("=" * 60)

sample_reasoning = ReasonedAnswer(
    question="What is the best database for a high-write workload?",
    reasoning_steps=[
        ReasoningStep(
            step_number=1,
            thought="High-write workloads need databases optimized for write performance",
            action="Consider write-optimized databases",
            observation="Cassandra, ScyllaDB, and TimescaleDB are known for high write throughput"
        ),
        ReasoningStep(
            step_number=2,
            thought="Need to consider data model requirements",
            action="Evaluate data structure needs",
            observation="Time-series data favors TimescaleDB; wide-column suits Cassandra"
        ),
        ReasoningStep(
            step_number=3,
            thought="Consider operational complexity and team expertise",
            action=None,
            observation="Managed services reduce operational burden"
        )
    ],
    final_answer="For a high-write workload, Cassandra or ScyllaDB are excellent choices for general use cases, while TimescaleDB is ideal for time-series data.",
    confidence=0.85,
    alternative_answers=[
        "MongoDB with appropriate write concern settings",
        "ClickHouse for analytics-heavy write workloads"
    ],
    assumptions_made=[
        "Workload is distributed across multiple nodes",
        "Eventual consistency is acceptable",
        "Data is primarily append-only"
    ]
)

print(f"\n❓ {sample_reasoning.question}")
print(f"\n🧠 Reasoning Chain:")
for step in sample_reasoning.reasoning_steps:
    print(f"   Step {step.step_number}: {step.thought}")

print(f"\n✅ Answer: {sample_reasoning.final_answer}")
print(f"   Confidence: {sample_reasoning.confidence:.0%}")
```

---

## UI Component Generation

### Dynamic Form Generation

```python
class FieldType(str, Enum):
    """Form field types."""
    
    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    NUMBER = "number"
    SELECT = "select"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    DATE = "date"


class FormField(BaseModel):
    """A form field definition."""
    
    name: str
    label: str
    field_type: FieldType
    placeholder: Optional[str] = None
    required: bool = True
    options: Optional[List[str]] = None  # For select fields
    validation_pattern: Optional[str] = None
    help_text: Optional[str] = None


class GeneratedForm(BaseModel):
    """A dynamically generated form."""
    
    form_id: str
    title: str
    description: str
    fields: List[FormField]
    submit_button_text: str
    cancel_button_text: str


FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "form_id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "field_type": {
                        "type": "string",
                        "enum": ["text", "email", "password", "number",
                                 "select", "checkbox", "textarea", "date"]
                    },
                    "placeholder": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "required": {"type": "boolean"},
                    "options": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"}
                        ]
                    },
                    "validation_pattern": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "help_text": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
                "required": ["name", "label", "field_type", "placeholder",
                            "required", "options", "validation_pattern", "help_text"],
                "additionalProperties": False
            }
        },
        "submit_button_text": {"type": "string"},
        "cancel_button_text": {"type": "string"}
    },
    "required": ["form_id", "title", "description", "fields",
                 "submit_button_text", "cancel_button_text"],
    "additionalProperties": False
}


print("\n\nUI Form Generation")
print("=" * 60)

sample_form = GeneratedForm(
    form_id="contact_form",
    title="Contact Us",
    description="Fill out this form and we'll get back to you within 24 hours.",
    fields=[
        FormField(
            name="name",
            label="Full Name",
            field_type=FieldType.TEXT,
            placeholder="John Doe",
            required=True
        ),
        FormField(
            name="email",
            label="Email Address",
            field_type=FieldType.EMAIL,
            placeholder="john@example.com",
            required=True,
            validation_pattern=r"^[^@]+@[^@]+\.[^@]+$"
        ),
        FormField(
            name="department",
            label="Department",
            field_type=FieldType.SELECT,
            required=True,
            options=["Sales", "Support", "Engineering", "Other"]
        ),
        FormField(
            name="message",
            label="Message",
            field_type=FieldType.TEXTAREA,
            placeholder="How can we help?",
            required=True,
            help_text="Please provide as much detail as possible"
        )
    ],
    submit_button_text="Send Message",
    cancel_button_text="Cancel"
)

print(f"\n📝 Form: {sample_form.title}")
print(f"   Fields: {len(sample_form.fields)}")
for field in sample_form.fields:
    req = "✓" if field.required else "○"
    print(f"   {req} {field.label} ({field.field_type.value})")
```

### Card/Widget Generation

```python
class CardAction(BaseModel):
    """An action button on a card."""
    
    label: str
    action_type: str  # "link", "button", "modal"
    target: str
    style: str  # "primary", "secondary", "danger"


class GeneratedCard(BaseModel):
    """A UI card component."""
    
    card_id: str
    title: str
    subtitle: Optional[str] = None
    image_url: Optional[str] = None
    body_text: str
    tags: List[str]
    actions: List[CardAction]
    metadata: dict


print("\n\nCard Generation Pattern")
print("=" * 60)

sample_card = GeneratedCard(
    card_id="product_001",
    title="Premium Plan",
    subtitle="Best for growing teams",
    image_url=None,
    body_text="Unlock all features including advanced analytics, priority support, and unlimited integrations.",
    tags=["Popular", "Best Value"],
    actions=[
        CardAction(
            label="Start Free Trial",
            action_type="button",
            target="/signup?plan=premium",
            style="primary"
        ),
        CardAction(
            label="Compare Plans",
            action_type="link",
            target="/pricing",
            style="secondary"
        )
    ],
    metadata={"price": 99, "currency": "USD", "billing": "monthly"}
)

print(f"\n🃏 Card: {sample_card.title}")
print(f"   {sample_card.body_text[:50]}...")
print(f"   Tags: {sample_card.tags}")
print(f"   Actions: {len(sample_card.actions)}")
```

---

## API Response Formatting

### Standardized API Responses

```python
class APIError(BaseModel):
    """API error details."""
    
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


class PaginationInfo(BaseModel):
    """Pagination metadata."""
    
    page: int
    per_page: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class APIResponse(BaseModel):
    """Standardized API response wrapper."""
    
    success: bool
    data: Optional[dict] = None
    errors: Optional[List[APIError]] = None
    pagination: Optional[PaginationInfo] = None
    request_id: str
    timestamp: str


API_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "data": {
            "anyOf": [{"type": "object"}, {"type": "null"}]
        },
        "errors": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                            "field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "suggestion": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                        },
                        "required": ["code", "message", "field", "suggestion"],
                        "additionalProperties": False
                    }
                },
                {"type": "null"}
            ]
        },
        "pagination": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "per_page": {"type": "integer"},
                        "total_items": {"type": "integer"},
                        "total_pages": {"type": "integer"},
                        "has_next": {"type": "boolean"},
                        "has_previous": {"type": "boolean"}
                    },
                    "required": ["page", "per_page", "total_items", 
                                "total_pages", "has_next", "has_previous"],
                    "additionalProperties": False
                },
                {"type": "null"}
            ]
        },
        "request_id": {"type": "string"},
        "timestamp": {"type": "string"}
    },
    "required": ["success", "data", "errors", "pagination", 
                 "request_id", "timestamp"],
    "additionalProperties": False
}


print("\n\nAPI Response Formatting")
print("=" * 60)

# Success response
success_response = APIResponse(
    success=True,
    data={"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]},
    errors=None,
    pagination=PaginationInfo(
        page=1,
        per_page=10,
        total_items=42,
        total_pages=5,
        has_next=True,
        has_previous=False
    ),
    request_id="req_abc123",
    timestamp="2025-01-15T10:30:00Z"
)

print(f"\n✅ Success Response:")
print(f"   Request ID: {success_response.request_id}")
print(f"   Data keys: {list(success_response.data.keys()) if success_response.data else None}")
if success_response.pagination:
    print(f"   Page {success_response.pagination.page} of {success_response.pagination.total_pages}")

# Error response
error_response = APIResponse(
    success=False,
    data=None,
    errors=[
        APIError(
            code="VALIDATION_ERROR",
            message="Email format is invalid",
            field="email",
            suggestion="Use format: user@domain.com"
        ),
        APIError(
            code="REQUIRED_FIELD",
            message="Name is required",
            field="name",
            suggestion="Provide a non-empty name"
        )
    ],
    pagination=None,
    request_id="req_def456",
    timestamp="2025-01-15T10:31:00Z"
)

print(f"\n❌ Error Response:")
print(f"   Request ID: {error_response.request_id}")
if error_response.errors:
    for err in error_response.errors:
        print(f"   • {err.code}: {err.message}")
```

---

## Production Pattern: Schema Registry

```python
from typing import Dict, Type


class SchemaRegistry:
    """Central registry for structured output schemas."""
    
    def __init__(self):
        self._schemas: Dict[str, dict] = {}
        self._models: Dict[str, Type[BaseModel]] = {}
    
    def register(
        self,
        name: str,
        model: Type[BaseModel],
        description: str = ""
    ):
        """Register a schema by name."""
        
        self._models[name] = model
        self._schemas[name] = {
            "name": name,
            "description": description,
            "schema": model.model_json_schema(),
            "strict": True
        }
    
    def get_schema(self, name: str) -> dict:
        """Get schema for API call."""
        
        if name not in self._schemas:
            raise KeyError(f"Schema '{name}' not registered")
        
        return self._schemas[name]
    
    def get_model(self, name: str) -> Type[BaseModel]:
        """Get Pydantic model for parsing."""
        
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")
        
        return self._models[name]
    
    def list_schemas(self) -> List[str]:
        """List all registered schema names."""
        
        return list(self._schemas.keys())


# Create and populate registry
print("\n\nSchema Registry Pattern")
print("=" * 60)

registry = SchemaRegistry()

# Register schemas
registry.register(
    "entity_extraction",
    ExtractedEntities,
    "Extract entities from text"
)

registry.register(
    "text_classification",
    TextClassification,
    "Classify text by category and sentiment"
)

registry.register(
    "form_generation",
    GeneratedForm,
    "Generate dynamic form definitions"
)

registry.register(
    "api_response",
    APIResponse,
    "Standardized API response format"
)

print("\n📋 Registered Schemas:")
for name in registry.list_schemas():
    schema = registry.get_schema(name)
    print(f"   • {name}: {schema['description']}")

# Usage example
print("\n\n🔧 Using Registry:")
schema = registry.get_schema("entity_extraction")
print(f"   Schema name: {schema['name']}")
print(f"   Strict mode: {schema['strict']}")
```

---

## Hands-on Exercise

### Your Task

Build a multi-purpose extraction system that handles different content types with appropriate schemas.

### Requirements

1. Create schemas for 3+ content types
2. Auto-detect content type
3. Apply appropriate schema
4. Return typed results

<details>
<summary>💡 Hints</summary>

- Use a registry pattern
- Create a router based on content hints
- Return union types or tagged results
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type
from pydantic import BaseModel, Field
from enum import Enum
import re


# Content type definitions
class ContentType(str, Enum):
    """Detectable content types."""
    
    EMAIL = "email"
    INVOICE = "invoice"
    RESUME = "resume"
    ARTICLE = "article"
    UNKNOWN = "unknown"


# Schemas for each content type
class ExtractedEmail(BaseModel):
    """Email extraction schema."""
    
    sender: str
    recipients: List[str]
    subject: str
    body_summary: str
    sentiment: str
    action_items: List[str]
    is_urgent: bool


class ExtractedInvoice(BaseModel):
    """Invoice extraction schema."""
    
    invoice_number: str
    vendor_name: str
    total_amount: float
    currency: str
    due_date: Optional[str] = None
    line_items: List[dict]
    payment_status: str


class ExtractedResume(BaseModel):
    """Resume extraction schema."""
    
    candidate_name: str
    contact_email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str]
    experience_years: int
    education: List[dict]
    work_history: List[dict]


class ExtractedArticle(BaseModel):
    """Article extraction schema."""
    
    title: str
    author: Optional[str] = None
    publication_date: Optional[str] = None
    main_topics: List[str]
    summary: str
    key_quotes: List[str]


# Content detector
class ContentDetector:
    """Detect content type from text."""
    
    PATTERNS = {
        ContentType.EMAIL: [
            r"from:", r"to:", r"subject:",
            r"@[\w\.-]+\.\w+", r"dear\s+\w+"
        ],
        ContentType.INVOICE: [
            r"invoice\s*(#|number|no\.?)", r"total\s*:",
            r"\$[\d,]+\.?\d*", r"due\s*date", r"bill\s*to"
        ],
        ContentType.RESUME: [
            r"experience", r"education", r"skills",
            r"references", r"objective", r"curriculum\s*vitae"
        ],
        ContentType.ARTICLE: [
            r"by\s+[A-Z][\w\s]+", r"published",
            r"introduction", r"conclusion", r"abstract"
        ]
    }
    
    def detect(self, text: str) -> ContentType:
        """Detect content type from text."""
        
        lower_text = text.lower()
        scores = {ct: 0 for ct in ContentType if ct != ContentType.UNKNOWN}
        
        for content_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, lower_text, re.IGNORECASE))
                scores[content_type] += matches
        
        if max(scores.values()) == 0:
            return ContentType.UNKNOWN
        
        return max(scores, key=scores.get)


# Multi-purpose extractor
class MultiPurposeExtractor:
    """Extract structured data based on content type."""
    
    SCHEMA_MAP: Dict[ContentType, Type[BaseModel]] = {
        ContentType.EMAIL: ExtractedEmail,
        ContentType.INVOICE: ExtractedInvoice,
        ContentType.RESUME: ExtractedResume,
        ContentType.ARTICLE: ExtractedArticle
    }
    
    def __init__(self):
        self.detector = ContentDetector()
        self.registry = SchemaRegistry()
        
        # Register all schemas
        for content_type, model in self.SCHEMA_MAP.items():
            self.registry.register(
                content_type.value,
                model,
                f"Extract {content_type.value} data"
            )
    
    def extract(
        self,
        text: str,
        content_type: Optional[ContentType] = None
    ) -> dict:
        """Extract structured data from text."""
        
        # Auto-detect if not specified
        if content_type is None:
            content_type = self.detector.detect(text)
        
        if content_type == ContentType.UNKNOWN:
            return {
                "success": False,
                "error": "Could not determine content type",
                "detected_type": None
            }
        
        # Get schema
        schema = self.registry.get_schema(content_type.value)
        model = self.registry.get_model(content_type.value)
        
        # Simulate extraction (in reality, call API here)
        extracted = self._simulate_extraction(text, content_type)
        
        return {
            "success": True,
            "detected_type": content_type.value,
            "data": extracted,
            "schema_used": schema["name"]
        }
    
    def _simulate_extraction(
        self,
        text: str,
        content_type: ContentType
    ) -> dict:
        """Simulate extraction for demo."""
        
        if content_type == ContentType.EMAIL:
            return ExtractedEmail(
                sender="sender@example.com",
                recipients=["recipient@example.com"],
                subject="Meeting Tomorrow",
                body_summary="Discussion about project updates",
                sentiment="neutral",
                action_items=["Review slides", "Prepare questions"],
                is_urgent=False
            ).model_dump()
        
        elif content_type == ContentType.INVOICE:
            return ExtractedInvoice(
                invoice_number="INV-001",
                vendor_name="Acme Corp",
                total_amount=1500.00,
                currency="USD",
                due_date="2025-02-15",
                line_items=[{"item": "Service", "amount": 1500}],
                payment_status="pending"
            ).model_dump()
        
        elif content_type == ContentType.RESUME:
            return ExtractedResume(
                candidate_name="Jane Doe",
                contact_email="jane@example.com",
                phone=None,
                skills=["Python", "Machine Learning", "Data Analysis"],
                experience_years=5,
                education=[{"degree": "MS", "field": "Computer Science"}],
                work_history=[{"company": "Tech Corp", "role": "Developer"}]
            ).model_dump()
        
        else:  # ARTICLE
            return ExtractedArticle(
                title="The Future of AI",
                author="John Writer",
                publication_date="2025-01-15",
                main_topics=["AI", "Technology", "Future"],
                summary="An exploration of AI trends...",
                key_quotes=["AI will transform every industry"]
            ).model_dump()
    
    def get_supported_types(self) -> List[str]:
        """List supported content types."""
        
        return [ct.value for ct in ContentType if ct != ContentType.UNKNOWN]


# Test the system
print("\nMulti-Purpose Extraction System")
print("=" * 60)

extractor = MultiPurposeExtractor()

print(f"\n📋 Supported Types: {extractor.get_supported_types()}")

# Test with different content types
test_texts = [
    "From: boss@company.com\nTo: team@company.com\nSubject: Q4 Review\nDear Team...",
    "INVOICE #12345\nBill To: Customer Inc\nTotal: $2,500.00\nDue Date: Feb 1, 2025",
    "RESUME\nJohn Smith\nSkills: Python, JavaScript\nExperience: 5 years\nEducation: BS CS",
    "Published by Jane Writer\nIntroduction\nThis article explores the latest in AI..."
]

for text in test_texts:
    result = extractor.extract(text)
    print(f"\n🔍 Detected: {result['detected_type']}")
    print(f"   Success: {result['success']}")
    if result['success']:
        data = result['data']
        # Show first few fields
        preview = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        print(f"   Data: {preview}")
```

</details>

---

## Summary

✅ Entity extraction provides typed, structured data from unstructured text  
✅ Classification with enums ensures valid category assignments  
✅ Chain-of-thought schemas capture reasoning alongside answers  
✅ UI generation creates typed component definitions  
✅ Schema registries centralize and manage multiple extraction types

**Back to:** [Structured Outputs Overview](./00-structured-outputs.md)

---

## Further Reading

- [OpenAI Structured Outputs Examples](https://platform.openai.com/docs/guides/structured-outputs#examples) — Official examples
- [Pydantic Documentation](https://docs.pydantic.dev/) — Model definitions
- [JSON Schema Patterns](https://json-schema.org/learn/getting-started-step-by-step) — Schema design
