---
title: "Schema Definition in Prompts"
---

# Schema Definition in Prompts

## Introduction

Schema definitions tell the model exactly what structure to produce. By specifying fields, types, constraints, and relationships in a formal way, you dramatically reduce format variations and parsing errors. This lesson covers practical schema notation styles that work well in prompts.

> **🤖 AI Context:** While APIs like OpenAI's Structured Outputs can enforce schemas programmatically, you'll often need to specify schemas in the prompt itself—especially for models or endpoints that don't support formal schema enforcement.

### What We'll Cover

- TypeScript-style interface definitions
- Field descriptions and constraints
- Optional vs required fields
- Validation hints in schemas
- Practical schema patterns

### Prerequisites

- [XML & Structured Text Outputs](./05-xml-structured-outputs.md)

---

## Why Define Schemas in Prompts

| Benefit | Description |
|---------|-------------|
| **Precision** | Exact field names and types |
| **Consistency** | Same structure every time |
| **Documentation** | Schema serves as output spec |
| **Validation** | Clear rules for checking output |
| **Developer familiarity** | Uses known notation |

---

## TypeScript-Style Schemas

TypeScript interfaces are intuitive and widely understood:

### Basic Interface

```markdown
# Output Schema

```typescript
interface ProductAnalysis {
  name: string;
  price: number;
  rating: number;         // 1.0 to 5.0
  inStock: boolean;
  categories: string[];   // 1-5 categories
}
```
```

### With Descriptions

```markdown
# Output Schema

```typescript
interface CustomerTicket {
  // Unique identifier for the ticket
  id: string;
  
  // Brief description of the issue (max 100 chars)
  title: string;
  
  // Detailed description of the problem
  description: string;
  
  // Priority level based on impact
  priority: "critical" | "high" | "medium" | "low";
  
  // Department to route the ticket to
  category: "billing" | "technical" | "account" | "general";
  
  // Confidence in the classification (0.0 to 1.0)
  confidence: number;
}
```
```

### Nested Objects

```markdown
# Output Schema

```typescript
interface OrderSummary {
  order: {
    id: string;
    date: string;           // ISO 8601 format
    status: "pending" | "shipped" | "delivered";
  };
  customer: {
    name: string;
    email: string;
    address: Address | null;  // null if not provided
  };
  items: OrderItem[];         // at least 1 item
  totals: {
    subtotal: number;
    tax: number;
    shipping: number;
    total: number;
  };
}

interface Address {
  street: string;
  city: string;
  state: string;
  zip: string;
  country: string;
}

interface OrderItem {
  sku: string;
  name: string;
  quantity: number;   // integer, 1 or more
  unitPrice: number;
  total: number;
}
```
```

---

## Optional vs Required Fields

### Optional Notation

```typescript
interface UserProfile {
  // Required fields
  id: string;
  email: string;
  
  // Optional fields (use ? suffix)
  phone?: string;
  company?: string;
  bio?: string;
  
  // Optional with default
  role?: string;  // defaults to "user"
}
```

### Nullable vs Optional

```markdown
# Field Handling

There's a difference between optional and nullable:

```typescript
interface Contact {
  name: string;           // Required, must have value
  email: string | null;   // Required, but can be null
  phone?: string;         // Optional, may not exist
  fax?: string | null;    // Optional, if present can be null
}
```

Rules:
- Required fields: Always present in output
- Nullable fields: Present but value is null
- Optional fields: May be omitted entirely

For consistency, prefer: **always include all fields, use null for missing values**
```

### Explicit Required/Optional

```markdown
# Schema

| Field | Type | Required | Default |
|-------|------|----------|---------|
| id | string | ✅ Yes | - |
| name | string | ✅ Yes | - |
| email | string | ✅ Yes | - |
| phone | string | ❌ No | null |
| role | string | ❌ No | "user" |
| active | boolean | ❌ No | true |
```

---

## Type Constraints

### Primitive Constraints

```typescript
interface Metrics {
  // String constraints
  id: string;           // UUID format
  name: string;         // 1-100 characters
  code: string;         // exactly 6 alphanumeric chars
  
  // Number constraints  
  count: number;        // integer, >= 0
  percentage: number;   // 0.0 to 100.0
  score: number;        // -1.0 to 1.0, 2 decimal places
  
  // Boolean
  active: boolean;      // true or false only
}
```

### Enum Constraints

```typescript
interface Task {
  status: "pending" | "in-progress" | "completed" | "cancelled";
  priority: 1 | 2 | 3 | 4 | 5;  // 1 = highest
  type: TaskType;
}

type TaskType = 
  | "bug"
  | "feature"
  | "documentation"
  | "maintenance";
```

### Array Constraints

```typescript
interface Analysis {
  // Array with length constraints
  tags: string[];           // 1-10 items
  scores: number[];         // 3-5 items, each 0.0-1.0
  
  // Array of objects
  findings: Finding[];      // at least 1
  recommendations: Action[]; // 0-5 items
}

interface Finding {
  issue: string;
  severity: "low" | "medium" | "high";
}
```

---

## JSON Schema Style

For complex validation, use JSON Schema notation:

### Inline JSON Schema

```markdown
# Output Schema (JSON Schema)

```json
{
  "type": "object",
  "required": ["name", "email", "status"],
  "properties": {
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100,
      "description": "User's full name"
    },
    "email": {
      "type": "string",
      "format": "email",
      "description": "Valid email address"
    },
    "age": {
      "type": "integer",
      "minimum": 0,
      "maximum": 150,
      "description": "Age in years"
    },
    "status": {
      "type": "string",
      "enum": ["active", "inactive", "pending"],
      "description": "Account status"
    },
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 0,
      "maxItems": 10,
      "description": "Optional tags"
    }
  },
  "additionalProperties": false
}
```
```

### Simplified Schema Table

```markdown
# Schema

| Property | Type | Constraints | Description |
|----------|------|-------------|-------------|
| name | string | 1-100 chars, required | User's full name |
| email | string | email format, required | Contact email |
| age | integer | 0-150, optional | Age in years |
| status | enum | active/inactive/pending, required | Account status |
| tags | string[] | 0-10 items, optional | User tags |
```

---

## Validation Hints

Include validation rules in your schema:

### Explicit Validation Rules

```markdown
# Schema with Validation

```typescript
interface Invoice {
  // ID: UUID v4 format (xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx)
  id: string;
  
  // Date: ISO 8601 format (YYYY-MM-DDTHH:mm:ssZ)
  createdAt: string;
  
  // Amount: positive number, 2 decimal places max
  amount: number;
  
  // Currency: ISO 4217 code (USD, EUR, GBP, etc.)
  currency: string;
  
  // Items: at least 1 item required
  items: LineItem[];
  
  // Total must equal sum of item amounts
  total: number;
}
```

# Validation Rules

1. `id` must be valid UUID v4
2. `createdAt` must be valid ISO 8601 timestamp
3. `amount` and `total` must be positive
4. `currency` must be valid ISO 4217 code
5. `items` array must have at least 1 element
6. `total` must equal sum of all `items[].amount`
```

### Example-Based Validation

```markdown
# Schema Examples

Valid:
```json
{
  "date": "2025-01-15",
  "amount": 99.99,
  "status": "pending"
}
```

Invalid (with errors):
```json
{
  "date": "Jan 15, 2025",    // Wrong format: use YYYY-MM-DD
  "amount": -50,              // Must be positive
  "status": "waiting"         // Invalid: use pending/approved/rejected
}
```
```

---

## Practical Patterns

### Entity Extraction Schema

```typescript
interface ExtractedEntities {
  persons: Person[];
  organizations: Organization[];
  locations: Location[];
  dates: DateMention[];
  amounts: Amount[];
}

interface Person {
  name: string;
  role?: string;        // if mentioned
  confidence: number;   // 0.0-1.0
}

interface Organization {
  name: string;
  type?: "company" | "government" | "nonprofit" | "other";
  confidence: number;
}

interface Location {
  name: string;
  type: "city" | "state" | "country" | "address" | "other";
  confidence: number;
}

interface DateMention {
  text: string;         // original text
  normalized?: string;  // ISO 8601 if parseable
  confidence: number;
}

interface Amount {
  text: string;         // original text
  value?: number;       // numeric value if clear
  currency?: string;    // if specified
  confidence: number;
}
```

### Classification Schema

```typescript
interface Classification {
  // Primary classification
  primary: {
    category: Category;
    confidence: number;
  };
  
  // Secondary classifications (if applicable)
  secondary: {
    category: Category;
    confidence: number;
  }[];
  
  // Reasoning for classification
  reasoning: string;
}

type Category = 
  | "technical"
  | "billing"
  | "account"
  | "general"
  | "feedback";
```

### Analysis Result Schema

```typescript
interface AnalysisResult {
  summary: string;          // 2-3 sentences
  
  sentiment: {
    label: "positive" | "negative" | "neutral" | "mixed";
    score: number;          // -1.0 to 1.0
    confidence: number;     // 0.0 to 1.0
  };
  
  keyPoints: {
    point: string;
    importance: "high" | "medium" | "low";
  }[];
  
  actionItems: {
    action: string;
    priority: 1 | 2 | 3;    // 1 = highest
    assignee?: string;
  }[];
  
  metadata: {
    processedAt: string;    // ISO 8601
    modelConfidence: number;
    wordCount: number;
  };
}
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use familiar notation | TypeScript/JSON Schema are widely known |
| Document all fields | Comments explain purpose |
| Specify all constraints | Types, ranges, formats |
| Show valid examples | Models learn from examples |
| Define error cases | What to do with invalid input |
| Keep schemas focused | One purpose per schema |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Vague types like "data" | Use specific types |
| No length constraints | Specify min/max |
| Missing enum values | List all valid options |
| Unclear optionality | Mark required vs optional |
| No format specification | Specify date, email, etc. formats |

---

## Hands-on Exercise

### Your Task

Design a schema for a meeting transcription analysis system.

### Requirements

1. Extract meeting metadata (title, date, attendees, duration)
2. Identify action items with assignees and deadlines
3. Summarize key decisions made
4. Classify the meeting type
5. Include confidence scores where appropriate

<details>
<summary>💡 Hints (click to expand)</summary>

- How do you handle attendees with unclear names?
- What if no deadline is mentioned for an action item?
- Should decisions include who made them?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```typescript
interface MeetingAnalysis {
  // Meeting metadata
  metadata: {
    title: string;                  // Meeting title/topic (inferred if not stated)
    date: string | null;            // ISO 8601 date, null if unclear
    duration: number | null;        // Duration in minutes, null if unknown
    type: MeetingType;
    confidence: number;             // Overall extraction confidence
  };
  
  // Attendees mentioned
  attendees: Attendee[];            // At least 1
  
  // Summary of the meeting
  summary: string;                  // 3-5 sentence overview
  
  // Key decisions made
  decisions: Decision[];            // 0 or more
  
  // Action items identified
  actionItems: ActionItem[];        // 0 or more
  
  // Topics discussed
  topics: Topic[];                  // 1-10 topics
}

type MeetingType = 
  | "standup"
  | "planning"
  | "review"
  | "brainstorm"
  | "decision"
  | "informational"
  | "one-on-one"
  | "other";

interface Attendee {
  name: string;                     // Name as mentioned
  role?: string;                    // Role if mentioned (e.g., "PM", "engineer")
  confidence: number;               // 0.0-1.0, confidence in extraction
}

interface Decision {
  decision: string;                 // What was decided
  madeBy?: string;                  // Who made/announced it, if clear
  context?: string;                 // Brief context for the decision
  confidence: number;
}

interface ActionItem {
  task: string;                     // Description of the task
  assignee: string | null;          // Who is responsible, null if unclear
  deadline: string | null;          // ISO 8601 date or null
  priority: "high" | "medium" | "low";  // Inferred from context
  confidence: number;
}

interface Topic {
  name: string;                     // Topic name/title
  duration?: number;                // Approximate minutes spent, if estimable
  relevance: number;                // 0.0-1.0, relevance to meeting purpose
}
```

**Usage in prompt:**

```markdown
# Instructions

Analyze the meeting transcript and extract structured information.

# Output Schema

[Include the TypeScript schema above]

# Rules

1. Always include at least 1 attendee
2. Use null for unknown dates/deadlines, not empty strings
3. Infer priority from urgency words ("ASAP" = high, "when you can" = low)
4. confidence values between 0.0 and 1.0
5. Summarize in 3-5 sentences, not bullet points
6. Topics limited to 10 most relevant

# Example Output

```json
{
  "metadata": {
    "title": "Q1 Planning Review",
    "date": "2025-01-15",
    "duration": 45,
    "type": "planning",
    "confidence": 0.92
  },
  "attendees": [
    {"name": "Sarah", "role": "PM", "confidence": 0.95},
    {"name": "Mike", "role": "engineer", "confidence": 0.88}
  ],
  "summary": "The team reviewed Q1 progress and adjusted timelines...",
  "decisions": [
    {
      "decision": "Push feature X to Q2",
      "madeBy": "Sarah",
      "context": "Due to resource constraints",
      "confidence": 0.90
    }
  ],
  "actionItems": [
    {
      "task": "Update project timeline in Jira",
      "assignee": "Mike",
      "deadline": "2025-01-17",
      "priority": "high",
      "confidence": 0.85
    }
  ],
  "topics": [
    {"name": "Q1 milestone review", "duration": 15, "relevance": 0.95}
  ]
}
```
```

</details>

### Bonus Challenge

- [ ] Add a `risks` array for identified risks or concerns
- [ ] Include `followUp` for scheduling the next meeting

---

## Summary

✅ **TypeScript interfaces** are intuitive and widely understood

✅ **Field descriptions** document expected content

✅ **Type constraints** prevent invalid values

✅ **Optional notation** clarifies required vs optional

✅ **Validation hints** guide correct output format

**Next:** [Handling Format Compliance](./07-handling-format-compliance.md)

---

## Further Reading

- [TypeScript Handbook - Interfaces](https://www.typescriptlang.org/docs/handbook/interfaces.html)
- [JSON Schema](https://json-schema.org/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

---

<!-- 
Sources Consulted:
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
- JSON Schema: https://json-schema.org/
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
