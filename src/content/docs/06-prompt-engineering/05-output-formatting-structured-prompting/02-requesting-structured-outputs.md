---
title: "Requesting Structured Outputs"
---

# Requesting Structured Outputs

## Introduction

Requesting structured outputs means explicitly telling the model the format you expect. Without clear format specifications, models produce varied output styles that are difficult to parse or display consistently. This lesson covers techniques to get reliable, predictable output formats.

> **🤖 AI Context:** Simply asking for "JSON" isn't enough. You need to specify the exact structure, field names, and types. The more explicit your format specification, the more reliable the output.

### What We'll Cover

- Format specification in prompts
- Template-based requests
- Field definitions and types
- Nested structures
- Enforcing format compliance

### Prerequisites

- [XML & Markdown Structure](./01-xml-markdown-structure.md)

---

## The Problem with Unspecified Formats

When you don't specify a format, the model chooses one:

**Prompt:**
```
List the top 3 programming languages for AI development.
```

**Possible outputs:**
```
Python, JavaScript, and Julia are great for AI.
```
```
1. Python
2. JavaScript  
3. Julia
```
```
- Python: Best for ML
- JavaScript: Good for web AI
- Julia: Fast numerical computing
```

All valid, but unpredictable. For programmatic use, you need consistency.

---

## Format Specification Techniques

### Explicit Format Instructions

State exactly what you want:

```markdown
# Instructions

List the top 3 programming languages for AI development.

# Output Format

Return a JSON array of objects with these fields:
- name: string (language name)
- rank: number (1-3)
- reason: string (one sentence why it's good for AI)

Example structure:
[
  {"name": "...", "rank": 1, "reason": "..."}
]
```

### Format Placement

Place format instructions where they have most impact:

| Position | Effectiveness | Use Case |
|----------|--------------|----------|
| End of prompt | ✅ High | Most cases |
| Before examples | ✅ High | Few-shot learning |
| In system message | ✅ High | Consistent format across requests |
| Start of prompt | ⚠️ Medium | May be forgotten for long prompts |

---

## Template-Based Requests

Provide a template the model should fill in:

### Simple Template

```markdown
# Instructions

Analyze the sentiment of this review.

# Template (fill in the values)

Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
Confidence: [HIGH/MEDIUM/LOW]
Key phrases: [comma-separated list]

# Review

"The product arrived quickly but the packaging was damaged."
```

**Expected output:**
```
Sentiment: NEUTRAL
Confidence: MEDIUM
Key phrases: arrived quickly, packaging was damaged
```

### Structured Template

```markdown
# Instructions

Extract contact information from the text.

# Output Template

```json
{
  "name": "<extracted name or null>",
  "email": "<extracted email or null>",
  "phone": "<extracted phone or null>",
  "company": "<extracted company or null>"
}
```

Only include fields that are explicitly mentioned.

# Text

"Hi, I'm Sarah Chen from TechCorp. Reach me at sarah@techcorp.com."
```

---

## Field Definitions

Define each field explicitly for complex outputs:

### Comprehensive Field Specification

```markdown
# Output Format

Return a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | yes | Unique identifier (UUID format) |
| title | string | yes | Article title, max 100 chars |
| summary | string | yes | 2-3 sentence summary |
| tags | array[string] | yes | 3-5 relevant tags |
| sentiment | enum | yes | One of: positive, negative, neutral |
| confidence | number | yes | 0.0 to 1.0 confidence score |
| entities | array[object] | no | Extracted named entities |

## Entity Object Schema

| Field | Type | Description |
|-------|------|-------------|
| name | string | Entity name |
| type | enum | person, organization, location, date |
```

### TypeScript-Style Definitions

Many developers find TypeScript interfaces intuitive:

```markdown
# Output Format

Return data matching this TypeScript interface:

```typescript
interface AnalysisResult {
  sentiment: "positive" | "negative" | "neutral";
  confidence: number;  // 0.0 to 1.0
  topics: string[];    // 1-5 main topics
  summary: string;     // Max 200 characters
  actionItems?: {      // Optional
    task: string;
    priority: "high" | "medium" | "low";
  }[];
}
```
```

---

## Nested Structures

For complex data, define the hierarchy clearly:

### Hierarchical Schema Definition

```markdown
# Output Format

Return a structured analysis with this hierarchy:

```json
{
  "document": {
    "title": "string",
    "author": "string | null",
    "date": "ISO date string | null"
  },
  "analysis": {
    "summary": "2-3 sentence summary",
    "keyPoints": [
      {
        "point": "string",
        "importance": "high | medium | low",
        "evidence": "quote from document"
      }
    ],
    "sentiment": {
      "overall": "positive | negative | neutral",
      "score": "number -1.0 to 1.0"
    }
  },
  "metadata": {
    "wordCount": "number",
    "processingTime": "ISO timestamp"
  }
}
```
```

### Flattening Complex Structures

Sometimes flattening is easier to parse:

```markdown
# Flat Output Format

Instead of nested objects, use dot notation:

document.title: string
document.author: string
analysis.summary: string
analysis.sentiment.overall: positive/negative/neutral
analysis.sentiment.score: number
```

---

## Constraining Values

Specify acceptable values explicitly:

### Enum Constraints

```markdown
# Category Field

The category must be exactly one of:
- "billing"
- "technical" 
- "account"
- "general"

Do not use variations like "Billing" or "BILLING".
```

### Range Constraints

```markdown
# Numeric Fields

- confidence: number between 0 and 1 (e.g., 0.85)
- priority: integer 1 to 5 (1 = highest)
- rating: number 1.0 to 5.0, one decimal place
```

### Length Constraints

```markdown
# Text Length Requirements

- title: 10-100 characters
- summary: 100-300 characters
- description: maximum 1000 characters
```

---

## Output Wrappers

Use markers to isolate structured output from reasoning:

### Code Block Wrapper

```markdown
# Instructions

Analyze the data and provide insights.

# Output Format

After your analysis, provide the structured result in a JSON code block:

```json
{
  "insights": [...],
  "recommendations": [...]
}
```

You may explain your reasoning before the JSON block.
```

### XML Wrapper

```markdown
# Output Format

Wrap your structured response in XML tags:

<analysis_result>
{your JSON here}
</analysis_result>

Any explanation should come before the tags.
```

This makes extraction reliable with simple parsing:

```python
import re

def extract_json(response):
    match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return None
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Show the exact format | Models imitate what they see |
| Specify all field types | Prevents type mismatches |
| Use consistent casing | `camelCase` or `snake_case`, not mixed |
| Include edge cases | What to output for missing data |
| Define acceptable values | Enums prevent hallucinated categories |
| Limit string lengths | Prevents overly verbose outputs |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Vague format: "return JSON" | Specify exact schema with field names |
| No type information | Define types: string, number, boolean |
| Missing null handling | State: "use null if not found" |
| Assuming format persistence | Repeat format requirements in long chats |
| Over-complicated schemas | Flatten or simplify where possible |

---

## Hands-on Exercise

### Your Task

Create a prompt that extracts product information from descriptions.

### Requirements

1. Define a schema for: name, price, features, category
2. Handle missing information with null
3. Constrain category to: electronics, clothing, home, other
4. Features should be an array of strings (max 5)

### Sample Input

```
"Apple AirPods Pro 2nd Gen - Premium wireless earbuds with 
active noise cancellation, spatial audio, and MagSafe charging. 
Perfect for music and calls. $249"
```

<details>
<summary>💡 Hints (click to expand)</summary>

- How will you handle price parsing (with or without $)?
- Should features be short phrases or sentences?
- What if no category is obvious?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```markdown
# Instructions

Extract product information from the description.

# Output Schema

```typescript
interface Product {
  name: string;           // Product name/title
  price: number | null;   // Price in USD (number only, no $)
  features: string[];     // Up to 5 key features, short phrases
  category: "electronics" | "clothing" | "home" | "other";
}
```

# Rules

- If price isn't stated, use null
- Extract at most 5 features, each under 50 characters
- Choose the best fitting category; default to "other" if unclear
- Return only the JSON object, no explanation

# Example

Input: "Cozy Cotton Blanket - Soft throw blanket for your couch. 50x60 inches. Machine washable. $35"

Output:
```json
{
  "name": "Cozy Cotton Blanket",
  "price": 35,
  "features": ["Soft throw blanket", "50x60 inches", "Machine washable"],
  "category": "home"
}
```

# Product Description

"Apple AirPods Pro 2nd Gen - Premium wireless earbuds with 
active noise cancellation, spatial audio, and MagSafe charging. 
Perfect for music and calls. $249"
```

**Expected output:**
```json
{
  "name": "Apple AirPods Pro 2nd Gen",
  "price": 249,
  "features": [
    "Active noise cancellation",
    "Spatial audio",
    "MagSafe charging",
    "Wireless earbuds"
  ],
  "category": "electronics"
}
```

</details>

### Bonus Challenge

- [ ] Add a `specs` object for technical specifications (key-value pairs)
- [ ] Include a `confidence` score for each extracted field

---

## Summary

✅ **Explicit format** specifications produce predictable outputs

✅ **Templates** guide the model to fill in exact structures

✅ **Field definitions** prevent type and value mismatches

✅ **Constraints** on values, lengths, and ranges improve reliability

✅ **Wrappers** make extraction from mixed content easier

**Next:** [JSON Output Specification](./03-json-output-specification.md)

---

## Further Reading

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [JSON Schema Specification](https://json-schema.org/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
