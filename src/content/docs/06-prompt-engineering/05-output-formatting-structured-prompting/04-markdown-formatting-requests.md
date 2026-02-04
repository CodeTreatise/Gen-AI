---
title: "Markdown Formatting Requests"
---

# Markdown Formatting Requests

## Introduction

When AI outputs are meant for human reading—documentation, chat responses, reports—markdown is the ideal format. Unlike JSON, markdown balances structure with readability. This lesson covers how to request consistent markdown formatting from language models.

> **🤖 AI Context:** Markdown is the default output format for most chat interfaces. Understanding how to control markdown structure helps you build better user-facing AI applications.

### What We'll Cover

- Heading structure and hierarchy
- Lists and bullet formatting
- Code blocks with syntax highlighting
- Tables for structured data
- Combining markdown elements

### Prerequisites

- [JSON Output Specification](./03-json-output-specification.md)

---

## When to Use Markdown

| Use Case | Why Markdown |
|----------|--------------|
| Chat responses | Native rendering in most UIs |
| Documentation | Headings, lists, code blocks |
| Reports | Structure with readability |
| Tutorials | Step-by-step with code examples |
| Mixed content | Text + code + data together |

---

## Heading Structure

### Specifying Heading Levels

```markdown
# Instructions

Write a tutorial on Python functions.

# Formatting Requirements

Use this heading structure:
- H1 (#): Tutorial title (one only)
- H2 (##): Major sections
- H3 (###): Subsections within sections

Example structure:
# Python Functions Tutorial
## What Are Functions?
## Defining Functions
### Basic Syntax
### Parameters and Arguments
## Return Values
```

### Heading Consistency

```markdown
# Output Format

For each section:
- Use H2 for section title
- Use H3 for subsections
- Do not skip heading levels (no H2 → H4)
- Use sentence case: "Getting started" not "Getting Started"
```

---

## Lists and Bullets

### Unordered Lists

```markdown
# List Formatting

Use bullet points (- or *) for:
- Items with no specific order
- Feature lists
- Requirements

Example:
- First item
- Second item
  - Nested item
  - Another nested item
- Third item
```

### Ordered Lists

```markdown
# Numbered Lists

Use numbered lists (1. 2. 3.) for:
- Sequential steps
- Ranked items
- Procedures

Example:
1. First step
2. Second step
   1. Sub-step A
   2. Sub-step B
3. Third step
```

### Task Lists

```markdown
# Task List Format

Use task list syntax for actionable items:

- [ ] Uncompleted task
- [x] Completed task

Example output:
- [x] Install dependencies
- [x] Configure environment
- [ ] Run tests
- [ ] Deploy to production
```

---

## Code Blocks

### Fenced Code Blocks

```markdown
# Code Formatting

Always use fenced code blocks with language specification:

```python
def example():
    return "Hello"
```

Supported languages: python, javascript, typescript, bash, sql, json, yaml
```

### Inline Code

```markdown
# Inline Code

Use backticks for:
- Function names: `calculate_total()`
- Variable names: `user_data`
- File names: `config.json`
- Commands: `npm install`
- Values: `true`, `null`, `200`
```

### Code with Output

```markdown
# Code Examples

Show code with expected output:

```python
numbers = [1, 2, 3, 4, 5]
print(sum(numbers))
```

**Output:**
```
15
```
```

---

## Tables

### Basic Tables

```markdown
# Table Format

Use markdown tables for structured comparisons:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

Rules:
- Always include header row
- Separate headers with |---|
- Align columns consistently
```

### Alignment

```markdown
# Table Alignment

| Left-aligned | Center-aligned | Right-aligned |
|:-------------|:--------------:|--------------:|
| Text         |     Text       |          Text |
| More text    |   More text    |     More text |

Use:
- :--- for left align
- :---: for center align
- ---: for right align
```

### Complex Tables

```markdown
# Comparison Table Format

When comparing options, use this structure:

| Feature | Option A | Option B | Notes |
|---------|----------|----------|-------|
| Speed   | ✅ Fast  | ⚠️ Medium | A is 2x faster |
| Cost    | ❌ High  | ✅ Low    | B is 50% cheaper |
| Ease    | ✅ Easy  | ❌ Complex | A has better docs |
```

---

## Emphasis and Formatting

### Text Emphasis

```markdown
# Emphasis Guidelines

- **Bold** for important terms or emphasis
- *Italic* for introducing new terms or titles
- `code` for technical terms, values, or commands
- ~~Strikethrough~~ for deprecated or incorrect info

Examples:
- The **primary key** uniquely identifies each row
- This pattern is called *dependency injection*
- Set `DEBUG=true` in your environment
- ~~Don't use var~~ Use const or let instead
```

### Callout Boxes

```markdown
# Callout Format

Use blockquotes with labels for callouts:

> **Note:** General helpful information

> **Warning:** Something that could cause problems

> **Tip:** Optional optimization or shortcut

Example in output:

> **Warning:** This operation cannot be undone. Make sure to backup your data first.
```

---

## Links and References

### Link Formatting

```markdown
# Links

Use markdown link syntax:
- Inline: [Link text](https://example.com)
- Reference: [Link text][ref] then [ref]: url

For documentation, prefer descriptive link text:
✅ [OpenAI API documentation](https://...)
❌ [Click here](https://...)
```

### Internal References

```markdown
# Cross-References

When referencing other sections:
- "See [Installation](#installation) for setup steps"
- "As discussed in [Chapter 2](./chapter-2.md)"
```

---

## Combining Elements

### Tutorial Format

```markdown
# Tutorial Output Format

Structure each tutorial section like this:

## Section Title

Brief introduction explaining what we'll cover.

### Step 1: Description

Explanation of the step.

```python
# Code for this step
```

**Expected result:**
```
Output shown here
```

> **Note:** Any important callouts

### Step 2: Description

Continue the pattern...

---

## Summary

Key takeaways as a bullet list.
```

### Documentation Format

```markdown
# Documentation Format

For API or feature documentation:

## Feature Name

Brief description of what it does.

### Usage

```python
# Basic usage example
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `param1`  | string | Yes | Description |
| `param2`  | number | No | Description |

### Returns

Description of return value.

### Example

```python
# Complete working example
```

### See Also

- [Related Feature](./related.md)
- [API Reference](./api.md)
```

---

## Requesting Specific Formats

### Explicit Format Specification

```markdown
# Output Requirements

Format your response as follows:
1. Start with a brief summary (2-3 sentences)
2. Use H2 headers for main sections
3. Include a code example with Python
4. End with a bullet list of key points

Do not include:
- H1 headers (I'll add the title)
- Introductory phrases like "Here's..."
- Closing phrases like "Let me know if..."
```

### Template-Based Request

```markdown
# Response Template

Use exactly this structure:

## Summary

[2-3 sentence overview]

## How It Works

[Explanation with technical details]

## Example

```python
[Working code example]
```

## Key Points

- [Point 1]
- [Point 2]
- [Point 3]
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Consistent heading levels | Clear hierarchy |
| Code blocks with language | Syntax highlighting |
| Tables for comparisons | Easy scanning |
| Bullet lists for features | Quick reading |
| Numbered lists for steps | Clear sequence |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Inconsistent heading levels | Specify exact structure |
| Code without language tag | Always require language |
| Missing table headers | Show example with headers |
| Mixed list styles | Specify bullets vs numbers |
| Overly nested lists | Limit to 2 levels |

---

## Hands-on Exercise

### Your Task

Create a prompt that generates consistent API documentation.

### Requirements

1. Include: description, parameters table, code example, return value
2. Use proper heading hierarchy
3. Parameters table with type, required, description columns
4. Code example with expected output

### API to Document

```
Function: search_users(query, limit=10, include_inactive=False)
Purpose: Search for users by name or email
Returns: List of user objects matching the query
```

<details>
<summary>💡 Hints (click to expand)</summary>

- What heading level should each section use?
- How do you represent optional parameters?
- Should the code example show all parameters or just required ones?

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```markdown
# Instructions

Generate API documentation for the provided function.

# Output Format

Use this exact structure:

## [Function Name]

[One-paragraph description of purpose and use case]

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| [name] | [type] | [Yes/No] | [value or -] | [description] |

### Returns

[Description of return value with type]

### Example

```python
# Example usage showing common use case
[code]
```

**Output:**
```python
[expected output]
```

### Notes

- [Any important notes or caveats]

---

# Function to Document

Function: search_users(query, limit=10, include_inactive=False)
Purpose: Search for users by name or email
Returns: List of user objects matching the query
```

**Expected output:**

```markdown
## search_users

Searches for users in the database by matching their name or email address against the provided query string. Returns a list of matching user objects, useful for implementing search functionality in user management interfaces.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search term to match against user names and emails |
| limit | integer | No | 10 | Maximum number of results to return |
| include_inactive | boolean | No | False | Whether to include deactivated user accounts |

### Returns

`List[User]` - A list of user objects matching the search query. Returns an empty list if no matches are found.

### Example

```python
# Search for users named "John"
results = search_users("john", limit=5)
```

**Output:**
```python
[
    {"id": 1, "name": "John Smith", "email": "john.smith@example.com"},
    {"id": 2, "name": "Johnny Appleseed", "email": "johnny@example.com"}
]
```

### Notes

- Search is case-insensitive
- Partial matches are supported (e.g., "joh" matches "John")
- Results are sorted by relevance
```

</details>

### Bonus Challenge

- [ ] Add a "Related Functions" section with links
- [ ] Include error handling examples

---

## Summary

✅ **Heading hierarchy** creates scannable structure

✅ **Code blocks** with language tags enable highlighting

✅ **Tables** present comparisons clearly

✅ **Lists** organize items and steps

✅ **Explicit templates** ensure consistent formatting

**Next:** [XML & Structured Text Outputs](./05-xml-structured-outputs.md)

---

## Further Reading

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [CommonMark Specification](https://commonmark.org/)

---

<!-- 
Sources Consulted:
- Markdown Guide: https://www.markdownguide.org/
- CommonMark: https://commonmark.org/
-->
