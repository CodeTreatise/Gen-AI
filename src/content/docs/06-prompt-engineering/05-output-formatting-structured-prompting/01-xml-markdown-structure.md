---
title: "XML & Markdown Structure for Prompt Clarity"
---

# XML & Markdown Structure for Prompt Clarity

## Introduction

Structured prompts produce structured outputs. By using XML tags and markdown formatting in your prompts, you help the model understand the logical boundaries between different sections—instructions, context, examples, and user input. This is especially important for complex prompts and reasoning models.

> **🤖 AI Context:** OpenAI recommends using markdown headers and XML tags to "help the model understand logical boundaries of your prompt and context data." This technique works across all major model providers.

### What We'll Cover

- Using XML tags to delineate prompt sections
- Markdown headers for organizing complex prompts
- Combining XML and markdown effectively
- Best practices for reasoning models

### Prerequisites

- [Few-Shot Prompting](../04-few-shot-prompting/)

---

## Why Structure Matters

Unstructured prompts force the model to guess where one section ends and another begins. Structured prompts eliminate ambiguity.

### Unstructured vs Structured

**❌ Unstructured:**
```
You are a helpful assistant. Classify the sentiment of product reviews as 
Positive, Negative, or Neutral. Here's an example: "I love this product!" 
should be Positive. Now classify: "The battery died after one day."
```

**✅ Structured:**
```markdown
# Identity
You are a sentiment classification assistant.

# Instructions
Classify product reviews as Positive, Negative, or Neutral.
Output only the classification word.

# Examples
<review>I love this product!</review>
<classification>Positive</classification>

# Input
<review>The battery died after one day.</review>
```

The structured version clearly separates:
- **Identity**: Who the assistant is
- **Instructions**: What to do
- **Examples**: How to do it
- **Input**: Current task

---

## XML Tags for Section Delineation

XML tags create unambiguous boundaries. The model recognizes where content starts and ends.

### Common XML Tag Patterns

| Tag Pattern | Use Case |
|-------------|----------|
| `<instructions>...</instructions>` | Task instructions |
| `<context>...</context>` | Background information |
| `<user_input>...</user_input>` | User-provided content |
| `<example>...</example>` | Few-shot examples |
| `<document>...</document>` | Reference documents |
| `<output>...</output>` | Expected output format |

### Basic XML Structure

```xml
<instructions>
Summarize the document in 2-3 sentences.
Focus on the main argument and key evidence.
</instructions>

<document>
[Long document text here...]
</document>

<output_format>
Provide the summary as plain text, no bullet points.
</output_format>
```

### XML Attributes for Metadata

Use attributes to add context about content:

```xml
<document id="research-paper-1" source="Nature" date="2024-06">
Climate change is accelerating faster than predicted...
</document>

<document id="research-paper-2" source="Science" date="2024-08">
New carbon capture technologies show promise...
</document>

<instructions>
Compare the findings from document id="research-paper-1" with 
document id="research-paper-2".
</instructions>
```

The `id` attribute lets you reference specific documents in instructions.

---

## Markdown Headers for Organization

Markdown headers create visual hierarchy and logical sections. Models parse headers to understand prompt structure.

### Standard Prompt Sections

```markdown
# Identity

You are a coding assistant that helps developers write clean, 
maintainable JavaScript code.

# Instructions

* Review the provided code for potential issues
* Suggest improvements for readability
* Identify any security vulnerabilities
* Format your response as a numbered list

# Context

The code is part of an Express.js web application handling 
user authentication.

# Code to Review

```javascript
app.post('/login', (req, res) => {
  const query = `SELECT * FROM users WHERE email='${req.body.email}'`;
  // ... rest of code
});
```
```

### Header Hierarchy

| Level | Use For |
|-------|---------|
| `#` H1 | Major sections (Identity, Instructions, Context) |
| `##` H2 | Subsections within major sections |
| `###` H3 | Specific items or categories |

```markdown
# Instructions

## General Guidelines
- Be concise
- Use professional tone

## Formatting Requirements
- Use bullet points for lists
- Include code examples when relevant

### Code Block Format
- Always specify language
- Include comments
```

---

## Combining XML and Markdown

Use markdown for high-level structure and XML for data delineation:

```markdown
# Identity

You are a code review assistant specialized in Python security.

# Instructions

Review the code for security vulnerabilities. For each issue found:
1. Identify the vulnerability type
2. Explain the risk
3. Provide a secure alternative

# Examples

<example id="1">
<user_query>Review this code for security issues</user_query>
<code_snippet>
password = input("Enter password: ")
if password == "admin123":
    grant_access()
</code_snippet>
<assistant_response>
**Vulnerability**: Hardcoded credentials
**Risk**: Anyone with source code access can authenticate
**Fix**: Use environment variables or secure secret management
</assistant_response>
</example>

# Current Task

<code_snippet>
import pickle

def load_user_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
</code_snippet>
```

---

## Reasoning Models and Structure

Reasoning models (o3, GPT-5 with reasoning, Claude with extended thinking) parse structured prompts more effectively.

### Why Structure Helps Reasoning Models

| Benefit | Explanation |
|---------|-------------|
| Clearer task decomposition | Model can identify subtasks from sections |
| Better context isolation | Keeps reference data separate from instructions |
| Improved example matching | Clear example boundaries help pattern recognition |
| Reduced confusion | Unambiguous section transitions |

### Recommended Structure for Reasoning Tasks

```markdown
# Goal

Determine the optimal database schema for a social media application.

# Constraints

- Must support 10M+ users
- Real-time feed updates required
- GDPR compliance necessary

# Available Options

<option id="1" name="Relational">
PostgreSQL with sharding
</option>

<option id="2" name="Document">
MongoDB with replica sets
</option>

<option id="3" name="Hybrid">
PostgreSQL for users, Redis for feeds
</option>

# Evaluation Criteria

1. Scalability (40% weight)
2. Query performance (30% weight)
3. Data consistency (20% weight)
4. Operational complexity (10% weight)

# Output Format

Provide your recommendation with:
- Selected option and rationale
- Trade-offs acknowledged
- Implementation considerations
```

---

## Fenced Code Blocks

When including code in prompts, use fenced code blocks with language specification:

```markdown
# Instructions

Convert this Python code to JavaScript:

```python
def calculate_total(items):
    return sum(item['price'] * item['quantity'] for item in items)
```

# Output Format

Provide the JavaScript equivalent using ES6+ syntax:

```javascript
// Your code here
```
```

### Code Block Best Practices

| Practice | Example |
|----------|---------|
| Always specify language | ` ```python ` not just ` ``` ` |
| Include context comments | `// Convert this to async/await` |
| Show expected output | `// Expected: [1, 2, 3]` |
| Use consistent indentation | 2 or 4 spaces throughout |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Consistent tag naming | `<user_input>` everywhere, not mixed |
| Close all XML tags | Unclosed tags confuse parsing |
| Use markdown for structure | Headers organize the prompt |
| Use XML for data | Tags isolate content from instructions |
| Keep sections focused | One purpose per section |
| Order sections logically | Identity → Instructions → Examples → Context → Input |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mixing instructions with context | Separate into distinct sections |
| Inconsistent tag names | Standardize: `<document>` not sometimes `<doc>` |
| No clear output format | Always specify expected format |
| Overly deep nesting | Max 2-3 levels of XML nesting |
| Forgetting closing tags | Validate XML structure |

---

## Hands-on Exercise

### Your Task

Create a structured prompt for a customer support ticket classifier.

### Requirements

1. Define the assistant's identity
2. Provide classification categories (Billing, Technical, Account, General)
3. Include 2 examples with XML tags
4. Specify output format
5. Include a placeholder for user input

<details>
<summary>💡 Hints (click to expand)</summary>

- Use markdown headers for major sections
- Use XML tags for examples and user input
- Include `id` attributes on examples
- Specify exactly what the output should look like

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```markdown
# Identity

You are a customer support ticket classifier for a SaaS company.
Your role is to categorize incoming tickets for routing.

# Instructions

Classify each ticket into exactly one category:
- **Billing**: Payment issues, invoices, subscriptions, refunds
- **Technical**: Bugs, errors, feature questions, integrations
- **Account**: Login issues, password resets, profile changes
- **General**: Feedback, general questions, other

Output only the category name, nothing else.

# Examples

<ticket id="example-1">
I was charged twice for my monthly subscription. Please refund.
</ticket>
<classification id="example-1">Billing</classification>

<ticket id="example-2">
The export button gives an error when I click it.
</ticket>
<classification id="example-2">Technical</classification>

# Current Ticket

<ticket>
{{user_ticket_content}}
</ticket>
```

**Why this works:**
- Clear identity and purpose
- Explicit category definitions
- Examples with matching ID attributes
- Simple, parseable output format
- Placeholder for dynamic content

</details>

### Bonus Challenge

- [ ] Add a confidence level to the output (High, Medium, Low)
- [ ] Handle multi-category tickets (output primary + secondary)

---

## Summary

✅ **XML tags** create unambiguous section boundaries

✅ **Markdown headers** organize prompt structure

✅ **Combine both** for complex prompts

✅ **Attributes** add metadata to content

✅ **Reasoning models** benefit especially from structure

**Next:** [Requesting Structured Outputs](./02-requesting-structured-outputs.md)

---

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [XML Basics (W3Schools)](https://www.w3schools.com/xml/)
- [Markdown Guide](https://www.markdownguide.org/)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
-->
