---
title: "Advanced Features"
---

# Advanced Features

## Introduction

Beyond the basics, LangChain's prompt templates support advanced templating with Jinja2, conditional content, loops, custom formatters, and validation. These features enable dynamic prompts that adapt based on input complexity, user preferences, or domain-specific requirements.

This lesson covers Jinja2 templates, conditional rendering, iteration, custom formatters, and template validation.

### What We'll Cover

- Jinja2 templating syntax
- Conditional content rendering
- Loops and iteration in templates
- Custom formatting functions
- Template validation and debugging
- Pipeline templates with preprocessing

### Prerequisites

- PromptTemplate and ChatPromptTemplate fundamentals (Lessons 8.2.1-2)
- Basic understanding of Jinja2 syntax (helpful but not required)

---

## Jinja2 Templates

LangChain supports Jinja2 syntax for complex templating:

### Enabling Jinja2

```python
from langchain.prompts import PromptTemplate

# F-string format (default)
fstring_template = PromptTemplate.from_template(
    "Hello {name}!",
    template_format="f-string"  # Default
)

# Jinja2 format
jinja_template = PromptTemplate.from_template(
    "Hello {{ name }}!",
    template_format="jinja2"
)

# Both work the same way
print(fstring_template.format(name="Alice"))  # Hello Alice!
print(jinja_template.format(name="Alice"))    # Hello Alice!
```

> **Note:** Use double braces `{{ }}` for Jinja2 variables. Single braces `{ }` are for f-string templates.

### When to Use Jinja2

| Use Case | f-string | Jinja2 |
|----------|----------|--------|
| Simple variable substitution | ✅ | ✅ |
| Conditional content | ❌ | ✅ |
| Loops/iteration | ❌ | ✅ |
| Filters (uppercase, etc.) | ❌ | ✅ |
| Complex logic | ❌ | ✅ |

---

## Conditional Content

Jinja2 enables if/else logic in templates:

### Basic Conditionals

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """You are a {{ role }} assistant.

{% if include_examples %}
Here are some examples:
- Example 1: ...
- Example 2: ...
{% endif %}

{% if verbose %}
Please provide a detailed explanation with step-by-step reasoning.
{% else %}
Be concise and direct.
{% endif %}

Question: {{ question }}""",
    template_format="jinja2"
)

# Verbose mode with examples
result = template.format(
    role="helpful",
    include_examples=True,
    verbose=True,
    question="What is Python?"
)
print(result)
```

**Output:**
```
You are a helpful assistant.

Here are some examples:
- Example 1: ...
- Example 2: ...

Please provide a detailed explanation with step-by-step reasoning.

Question: What is Python?
```

### Conditional with Else-If

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """{% if level == 'beginner' %}
Explain like I'm new to programming.
{% elif level == 'intermediate' %}
You can assume basic programming knowledge.
{% elif level == 'expert' %}
Feel free to use technical jargon and advanced concepts.
{% else %}
Provide a balanced explanation.
{% endif %}

Topic: {{ topic }}""",
    template_format="jinja2"
)

print(template.format(level="beginner", topic="APIs"))
```

### Existence Checks

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """{% if context is defined and context %}
Use this context: {{ context }}
{% endif %}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

Question: {{ question }}""",
    template_format="jinja2"
)

# With context and constraints
result = template.format(
    context="User is a Python developer",
    constraints=["Keep response under 100 words", "Use code examples"],
    question="How do I read a file?"
)
```

---

## Loops and Iteration

Process lists and dictionaries dynamically:

### Iterating Over Lists

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """You have access to the following tools:

{% for tool in tools %}
{{ loop.index }}. {{ tool.name }}: {{ tool.description }}
{% endfor %}

Use the appropriate tool to answer: {{ question }}""",
    template_format="jinja2"
)

tools = [
    {"name": "search", "description": "Search the web"},
    {"name": "calculator", "description": "Perform math"},
    {"name": "weather", "description": "Get weather info"}
]

result = template.format(tools=tools, question="What's 5 + 3?")
print(result)
```

**Output:**
```
You have access to the following tools:

1. search: Search the web
2. calculator: Perform math
3. weather: Get weather info

Use the appropriate tool to answer: What's 5 + 3?
```

### Loop Variables

Jinja2 provides special loop variables:

| Variable | Description |
|----------|-------------|
| `loop.index` | Current iteration (1-indexed) |
| `loop.index0` | Current iteration (0-indexed) |
| `loop.first` | True if first iteration |
| `loop.last` | True if last iteration |
| `loop.length` | Total number of items |

```python
template = PromptTemplate.from_template(
    """{% for item in items %}
{{ item }}{% if not loop.last %}, {% endif %}
{% endfor %}""",
    template_format="jinja2"
)

result = template.format(items=["apple", "banana", "cherry"])
# Output: apple, banana, cherry
```

### Iterating Over Dictionaries

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """User Profile:
{% for key, value in profile.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Personalize the response for this user.""",
    template_format="jinja2"
)

profile = {
    "name": "Alice",
    "role": "Developer",
    "experience": "5 years",
    "interests": "AI, Web Development"
}

result = template.format(profile=profile)
```

---

## Jinja2 Filters

Transform values inline:

### Common Filters

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """Name: {{ name | upper }}
Description: {{ text | truncate(50) }}
Tags: {{ tags | join(', ') }}
Count: {{ items | length }}""",
    template_format="jinja2"
)

result = template.format(
    name="alice",
    text="This is a very long description that should be truncated for display purposes.",
    tags=["python", "ai", "langchain"],
    items=[1, 2, 3, 4, 5]
)
print(result)
```

**Output:**
```
Name: ALICE
Description: This is a very long description that should be...
Tags: python, ai, langchain
Count: 5
```

### Useful Filters

| Filter | Purpose | Example |
|--------|---------|---------|
| `upper` | Uppercase | `{{ name \| upper }}` |
| `lower` | Lowercase | `{{ name \| lower }}` |
| `capitalize` | Capitalize first letter | `{{ name \| capitalize }}` |
| `title` | Title Case | `{{ name \| title }}` |
| `trim` | Remove whitespace | `{{ text \| trim }}` |
| `truncate(n)` | Limit length | `{{ text \| truncate(50) }}` |
| `join(sep)` | Join list | `{{ items \| join(', ') }}` |
| `length` | Get count | `{{ items \| length }}` |
| `default(val)` | Fallback value | `{{ name \| default('Guest') }}` |
| `replace(a, b)` | Replace text | `{{ text \| replace('old', 'new') }}` |

---

## Custom Formatters

Create reusable formatting logic:

### String Formatters

```python
from langchain.prompts import PromptTemplate

def format_list_as_bullets(items: list) -> str:
    """Format list as bullet points."""
    return "\n".join(f"• {item}" for item in items)

def format_code_block(code: str, language: str = "python") -> str:
    """Wrap code in markdown code block."""
    return f"```{language}\n{code}\n```"

# Use in prompt construction
template = PromptTemplate.from_template(
    """{system_context}

Available tools:
{tools}

Code to analyze:
{code}

Provide your analysis."""
)

result = template.format(
    system_context="You are a code reviewer.",
    tools=format_list_as_bullets(["linter", "type-checker", "formatter"]),
    code=format_code_block("def add(a, b): return a + b")
)
print(result)
```

### Callable Partial Variables

```python
from langchain.prompts import PromptTemplate
from datetime import datetime

def get_current_time():
    """Return formatted current time."""
    return datetime.now().strftime("%I:%M %p")

def get_user_context(user_id: str):
    """Fetch user context (simulate)."""
    users = {
        "u1": "Premium user, prefers detailed answers",
        "u2": "New user, prefers simple explanations"
    }
    return users.get(user_id, "Standard user")

template = PromptTemplate(
    input_variables=["question", "user_id"],
    partial_variables={
        "time": get_current_time  # Called at format time
    },
    template="""Current time: {time}
    
Question: {question}"""
)

# Time is evaluated when format() is called
result = template.format(question="What time is it?", user_id="u1")
```

---

## Template Validation

Ensure templates are correct before runtime:

### Validating Input Variables

```python
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

template = PromptTemplate(
    input_variables=["name", "task"],
    template="Hello {name}, please complete: {task}"
)

# Check required variables
print(f"Required: {template.input_variables}")

# Validate by formatting with test values
try:
    result = template.format(name="Alice", task="Review code")
    print("Valid template!")
except KeyError as e:
    print(f"Missing variable: {e}")
```

### Schema Validation

```python
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator

class PromptInputs(BaseModel):
    """Validate prompt inputs."""
    name: str = Field(..., min_length=1, max_length=100)
    task: str = Field(..., min_length=5)
    priority: str = Field(default="normal")
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed = ['low', 'normal', 'high']
        if v not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v

# Validate before formatting
inputs = PromptInputs(
    name="Alice",
    task="Review the PR",
    priority="high"
)

template = PromptTemplate.from_template(
    "Hi {name}, please {task}. Priority: {priority}"
)
result = template.format(**inputs.model_dump())
```

### Debugging Templates

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """System: {system}
Context: {context}
Question: {question}"""
)

# Inspect template
print(f"Template format: {template.template_format}")
print(f"Input variables: {template.input_variables}")
print(f"Partial variables: {template.partial_variables}")

# Get schema
schema = template.input_schema.schema()
print(f"Schema: {schema}")

# Pretty print formatted output
result = template.format(
    system="You are helpful.",
    context="Python development",
    question="How do I read files?"
)
print(f"\n--- Formatted Output ---\n{result}")
```

---

## Pipeline Templates

Preprocess inputs before formatting:

### Input Preprocessing

```python
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

def preprocess_inputs(inputs: dict) -> dict:
    """Clean and preprocess inputs."""
    processed = inputs.copy()
    
    # Normalize text
    if "query" in processed:
        processed["query"] = processed["query"].strip().lower()
    
    # Add defaults
    processed.setdefault("language", "English")
    
    # Truncate long inputs
    if "context" in processed and len(processed["context"]) > 1000:
        processed["context"] = processed["context"][:1000] + "..."
    
    return processed

template = PromptTemplate.from_template(
    """Language: {language}
Query: {query}
Context: {context}

Provide a helpful response."""
)

# Create preprocessing pipeline
preprocess = RunnableLambda(preprocess_inputs)
pipeline = preprocess | template

# Use pipeline
result = pipeline.invoke({
    "query": "  HOW DO I START?  ",
    "context": "Python basics..."
})
```

### Multi-Stage Templates

```python
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def extract_entities(inputs: dict) -> dict:
    """Extract entities from query (simulated)."""
    query = inputs.get("query", "")
    entities = []
    
    if "python" in query.lower():
        entities.append("Python")
    if "api" in query.lower():
        entities.append("API")
    
    return {**inputs, "entities": entities}

def format_entities(inputs: dict) -> dict:
    """Format entities for display."""
    entities = inputs.get("entities", [])
    if entities:
        inputs["entity_text"] = f"Detected topics: {', '.join(entities)}"
    else:
        inputs["entity_text"] = "No specific topics detected."
    return inputs

template = PromptTemplate.from_template(
    """{entity_text}

Query: {query}

Please provide relevant information."""
)

# Multi-stage pipeline
pipeline = (
    RunnablePassthrough()
    | RunnableLambda(extract_entities)
    | RunnableLambda(format_entities)
    | template
)

result = pipeline.invoke({"query": "How do I use Python APIs?"})
print(result)
```

---

## Advanced Jinja2 Patterns

### Macros (Reusable Snippets)

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """{% macro format_tool(name, desc) %}
**{{ name }}**: {{ desc }}
{% endmacro %}

Available Tools:
{{ format_tool("search", "Search the web") }}
{{ format_tool("calculate", "Perform math operations") }}
{{ format_tool("weather", "Get weather information") }}

Use these tools to answer: {{ question }}""",
    template_format="jinja2"
)

result = template.format(question="What's the weather in NYC?")
```

### Whitespace Control

```python
from langchain.prompts import PromptTemplate

# Use {%- and -%} to strip whitespace
template = PromptTemplate.from_template(
    """Items:
{%- for item in items %}
- {{ item }}
{%- endfor %}

Total: {{ items | length }}""",
    template_format="jinja2"
)

result = template.format(items=["apple", "banana", "cherry"])
# Clean output without extra blank lines
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Jinja2 for complex logic | Keeps Python code clean |
| Validate templates early | Catch errors before runtime |
| Document template variables | Makes templates maintainable |
| Use filters for transforms | Cleaner than inline Python |
| Test with edge cases | Empty lists, None values, etc. |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mixing f-string and Jinja2 syntax | Pick one format per template |
| Not handling None values | Use `default()` filter |
| Complex logic in templates | Move to preprocessing functions |
| Forgetting to specify format | Set `template_format="jinja2"` |
| Not escaping special chars | Use `{{ '{' }}` for literal braces |

---

## Hands-on Exercise

### Your Task

Create a dynamic email template system that:
1. Uses Jinja2 for conditional content
2. Handles different email types (welcome, reset, notification)
3. Includes loops for dynamic content (items, links, etc.)

### Requirements

1. Create a template that adapts based on `email_type`
2. Include conditional greeting based on time of day
3. Use loops to render list content
4. Add proper validation for required fields

### Expected Result

A flexible template that generates appropriate emails for each type.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use elif for multiple email types
- Include common footer with macro
- Validate email_type is one of allowed values
- Test with empty lists to ensure graceful handling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain.prompts import PromptTemplate
from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Optional

# Input validation
class EmailInputs(BaseModel):
    email_type: str = Field(..., description="Type of email")
    recipient_name: str = Field(..., min_length=1)
    items: list = Field(default_factory=list)
    action_url: Optional[str] = None
    
    @validator('email_type')
    def validate_type(cls, v):
        allowed = ['welcome', 'password_reset', 'notification']
        if v not in allowed:
            raise ValueError(f"email_type must be one of {allowed}")
        return v

# Email template
email_template = PromptTemplate.from_template(
    """{% macro footer() %}
---
Best regards,
The Team
support@example.com
{% endmacro %}

{# Determine greeting based on hour #}
{% if hour < 12 %}
Good morning, {{ recipient_name }}!
{% elif hour < 17 %}
Good afternoon, {{ recipient_name }}!
{% else %}
Good evening, {{ recipient_name }}!
{% endif %}

{% if email_type == 'welcome' %}
Welcome to our platform! We're excited to have you.

Here's what you can do:
{% for item in items %}
{{ loop.index }}. {{ item }}
{% endfor %}

Get started: {{ action_url | default('#', true) }}

{% elif email_type == 'password_reset' %}
We received a request to reset your password.

Click here to reset: {{ action_url }}

If you didn't request this, please ignore this email.

{% elif email_type == 'notification' %}
You have new updates:

{% if items %}
{% for item in items %}
• {{ item }}
{% endfor %}
{% else %}
No new items at this time.
{% endif %}

{% endif %}

{{ footer() }}""",
    template_format="jinja2"
)

def send_email(email_type: str, recipient_name: str, 
               items: list = None, action_url: str = None):
    """Generate and send email."""
    # Validate inputs
    inputs = EmailInputs(
        email_type=email_type,
        recipient_name=recipient_name,
        items=items or [],
        action_url=action_url
    )
    
    # Get current hour for greeting
    hour = datetime.now().hour
    
    # Format email
    return email_template.format(
        hour=hour,
        **inputs.model_dump()
    )

# Test different email types
print("=== Welcome Email ===")
print(send_email(
    email_type="welcome",
    recipient_name="Alice",
    items=["Create your profile", "Explore features", "Join community"],
    action_url="https://example.com/start"
))

print("\n=== Password Reset Email ===")
print(send_email(
    email_type="password_reset",
    recipient_name="Bob",
    action_url="https://example.com/reset?token=abc123"
))

print("\n=== Notification Email ===")
print(send_email(
    email_type="notification",
    recipient_name="Charlie",
    items=["New message from Alice", "Your order shipped", "Weekly report ready"]
))
```

</details>

### Bonus Challenges

- [ ] Add localization support (templates in multiple languages)
- [ ] Implement template inheritance for shared layouts
- [ ] Create a template testing framework
- [ ] Add HTML rendering with escape filters

---

## Summary

✅ Jinja2 enables conditionals, loops, and filters in templates  
✅ Use `{% if %}` for conditional content based on inputs  
✅ Use `{% for %}` to iterate over lists and dictionaries  
✅ Filters transform values inline (`upper`, `truncate`, `join`)  
✅ Validate inputs with Pydantic before formatting  
✅ Pipeline templates preprocess inputs before rendering  

**Next:** [Prompt Hub](./06-prompt-hub.md) — LangSmith integration for prompt management

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Few-Shot Prompting](./04-few-shot-prompting.md) | [Prompt Templates](./00-prompt-templates.md) | [Prompt Hub](./06-prompt-hub.md) |

<!-- 
Sources Consulted:
- Jinja2 Template Designer Documentation: https://jinja.palletsprojects.com/en/3.1.x/templates/
- LangChain GitHub prompts/prompt.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/prompts/prompt.py
- LangChain prompts documentation concepts: https://python.langchain.com/docs/concepts/prompts/
-->
