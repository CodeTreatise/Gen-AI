---
title: "Template Design Patterns"
---

# Template Design Patterns

## Introduction

Template design patterns establish consistent structures for prompt reuse. From simple placeholder substitution to complex conditional logic and template inheritance, these patterns make prompts maintainable, testable, and scalable. We'll explore patterns that work across templating systems.

> **🤖 AI Context:** The patterns here mirror web templating (Jinja2, Handlebars) because the same separation-of-concerns benefits apply—keep logic separate from content.

### What We'll Cover

- Placeholder syntax conventions
- Conditional sections
- Loops for dynamic content
- Template inheritance and composition
- Best practices for maintainability

### Prerequisites

- [Prompt Templates Overview](./00-prompt-templates-overview.md)
- Basic Python programming

---

## Placeholder Syntax

### Simple Placeholders

The most basic pattern—insert a value at a marked position:

```python
# Python string formatting
template = "Analyze this {document_type} for {analysis_focus}."

# Usage
prompt = template.format(
    document_type="contract",
    analysis_focus="liability risks"
)
# Output: "Analyze this contract for liability risks."
```

### Common Syntax Conventions

| System | Syntax | Example |
|--------|--------|---------|
| Python f-strings | `{variable}` | `f"Hello {name}"` |
| Python `.format()` | `{variable}` | `"Hello {name}".format(name="World")` |
| Python Template | `$variable` or `${variable}` | `Template("Hello $name")` |
| Jinja2 | `{{ variable }}` | `{{ user_name }}` |
| OpenAI Dashboard | `{{variable}}` | `{{customer_name}}` |
| Handlebars | `{{variable}}` | `{{productName}}` |

### Safe Placeholder Pattern

Avoid `KeyError` when variables are missing:

```python
from string import Template

class SafeTemplate(Template):
    """Template that leaves missing variables as-is."""
    
    def safe_render(self, **kwargs):
        # safe_substitute doesn't raise on missing keys
        return self.safe_substitute(**kwargs)

template = SafeTemplate("Hello $name, your order $order_id is ready.")
result = template.safe_render(name="Alice")
# Output: "Hello Alice, your order $order_id is ready."
```

### Named vs Positional

```python
# ❌ Positional - hard to maintain
template = "The {} analyzed the {} and found {}."

# ✅ Named - self-documenting
template = "The {role} analyzed the {document} and found {findings}."
```

---

## Conditional Sections

### Jinja2 Conditionals

```python
from jinja2 import Template

template = Template("""
You are a {{ role }} assistant.

{% if include_examples %}
## Examples
Here are some examples of good responses:
{% for example in examples %}
- {{ example }}
{% endfor %}
{% endif %}

{% if strict_mode %}
IMPORTANT: Only use information from the provided context.
Do not use external knowledge.
{% else %}
You may use your general knowledge when the context is insufficient.
{% endif %}

## Task
{{ task_description }}
""")

# Render with conditions
prompt = template.render(
    role="customer support",
    include_examples=True,
    examples=["I understand your concern...", "Let me help you with that..."],
    strict_mode=False,
    task_description="Help the customer with their billing question."
)
```

### Python-Based Conditionals

For simpler needs, build conditionally in Python:

```python
def build_prompt(task: str, examples: list = None, strict: bool = False) -> str:
    sections = [f"# Task\n{task}"]
    
    if examples:
        example_text = "\n".join(f"- {ex}" for ex in examples)
        sections.insert(0, f"# Examples\n{example_text}")
    
    if strict:
        sections.append("\n⚠️ Use ONLY the provided context. No external knowledge.")
    
    return "\n\n".join(sections)

# Usage
prompt = build_prompt(
    task="Summarize this document",
    examples=["Good summary example..."],
    strict=True
)
```

### Conditional Blocks Pattern

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PromptConfig:
    task: str
    role: Optional[str] = None
    examples: Optional[List[str]] = None
    context: Optional[str] = None
    output_format: Optional[str] = None
    constraints: Optional[List[str]] = None

def render_prompt(config: PromptConfig) -> str:
    """Render prompt with conditional sections."""
    
    blocks = []
    
    # Role (optional)
    if config.role:
        blocks.append(f"You are a {config.role}.")
    
    # Context (optional)
    if config.context:
        blocks.append(f"## Context\n{config.context}")
    
    # Examples (optional)
    if config.examples:
        examples_text = "\n".join(f"- {ex}" for ex in config.examples)
        blocks.append(f"## Examples\n{examples_text}")
    
    # Task (required)
    blocks.append(f"## Task\n{config.task}")
    
    # Output format (optional)
    if config.output_format:
        blocks.append(f"## Output Format\n{config.output_format}")
    
    # Constraints (optional)
    if config.constraints:
        constraints_text = "\n".join(f"- {c}" for c in config.constraints)
        blocks.append(f"## Constraints\n{constraints_text}")
    
    return "\n\n".join(blocks)
```

---

## Loops for Dynamic Content

### Iterating Over Examples

```python
from jinja2 import Template

few_shot_template = Template("""
# Classification Task

Classify the sentiment of text as: positive, negative, or neutral.

## Examples
{% for example in examples %}
Text: "{{ example.text }}"
Sentiment: {{ example.label }}
{% endfor %}

## Classify This
Text: "{{ input_text }}"
Sentiment:""")

prompt = few_shot_template.render(
    examples=[
        {"text": "I love this product!", "label": "positive"},
        {"text": "Terrible experience.", "label": "negative"},
        {"text": "It arrived on time.", "label": "neutral"}
    ],
    input_text="Best purchase I've ever made!"
)
```

### Generating Tool Descriptions

```python
from jinja2 import Template

tools_template = Template("""
You have access to the following tools:

{% for tool in tools %}
## {{ tool.name }}
{{ tool.description }}
Parameters:
{% for param in tool.parameters %}
- `{{ param.name }}` ({{ param.type }}{% if param.required %}, required{% endif %}): {{ param.description }}
{% endfor %}

{% endfor %}

When you need to use a tool, respond with:
```json
{"tool": "tool_name", "parameters": {...}}
```
""")

prompt = tools_template.render(tools=[
    {
        "name": "search_documents",
        "description": "Search the document database",
        "parameters": [
            {"name": "query", "type": "string", "required": True, "description": "Search query"},
            {"name": "limit", "type": "integer", "required": False, "description": "Max results"}
        ]
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": [
            {"name": "city", "type": "string", "required": True, "description": "City name"}
        ]
    }
])
```

### Loop Controls

```python
from jinja2 import Template

template = Template("""
{% for item in items %}
{{ loop.index }}. {{ item }}{% if not loop.last %}, {% endif %}
{% endfor %}

Total items: {{ items | length }}
""")

# loop.index = 1-based index
# loop.index0 = 0-based index
# loop.first = True if first iteration
# loop.last = True if last iteration
# loop.length = total iterations
```

---

## Template Inheritance

### Base Template Pattern

Create a base template that child templates extend:

```python
# base_prompt.py
BASE_TEMPLATE = """
# Identity
You are {{ role }}.

# Core Instructions
{{ core_instructions }}

{% block additional_instructions %}{% endblock %}

# Output Format
{{ output_format }}

{% block task %}{% endblock %}
"""

# specialized_prompt.py
from jinja2 import Environment, BaseLoader

env = Environment(loader=BaseLoader())

child_template = env.from_string("""
{% extends base %}

{% block additional_instructions %}
## Domain-Specific Rules
- Always cite sources
- Use formal language
- Maximum 500 words
{% endblock %}

{% block task %}
## Your Task
Analyze the following document:
{{ document }}
{% endblock %}
""")

# Render with parent context
prompt = child_template.render(
    base=BASE_TEMPLATE,
    role="legal analyst",
    core_instructions="Provide accurate, unbiased analysis.",
    output_format="Structured markdown with headers",
    document=document_text
)
```

### Composition Pattern

Compose prompts from reusable components:

```python
class PromptComponents:
    """Reusable prompt building blocks."""
    
    ROLES = {
        "analyst": "You are a data analyst specializing in business intelligence.",
        "writer": "You are a professional technical writer.",
        "coder": "You are an expert software engineer."
    }
    
    OUTPUT_FORMATS = {
        "json": "Respond with valid JSON only. No markdown, no explanation.",
        "markdown": "Format your response using markdown headers and lists.",
        "plain": "Respond in plain text without any formatting."
    }
    
    CONSTRAINTS = {
        "concise": "Keep your response under 200 words.",
        "detailed": "Provide comprehensive analysis with examples.",
        "factual": "Only state facts you are certain about."
    }

class PromptBuilder:
    """Build prompts by composing components."""
    
    def __init__(self):
        self.sections = []
    
    def with_role(self, role_key: str):
        role = PromptComponents.ROLES.get(role_key, role_key)
        self.sections.append(f"# Role\n{role}")
        return self
    
    def with_output_format(self, format_key: str):
        fmt = PromptComponents.OUTPUT_FORMATS.get(format_key, format_key)
        self.sections.append(f"# Output Format\n{fmt}")
        return self
    
    def with_constraints(self, *constraint_keys):
        constraints = [
            PromptComponents.CONSTRAINTS.get(k, k) 
            for k in constraint_keys
        ]
        self.sections.append("# Constraints\n" + "\n".join(f"- {c}" for c in constraints))
        return self
    
    def with_task(self, task: str):
        self.sections.append(f"# Task\n{task}")
        return self
    
    def build(self) -> str:
        return "\n\n".join(self.sections)

# Usage
prompt = (PromptBuilder()
    .with_role("analyst")
    .with_output_format("json")
    .with_constraints("concise", "factual")
    .with_task("Analyze this sales data and identify trends.")
    .build())
```

---

## Template Includes

### Modular Template Files

```python
# prompts/
#   base.jinja2
#   components/
#     examples.jinja2
#     constraints.jinja2
#   tasks/
#     classification.jinja2
#     summarization.jinja2

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('prompts'))

# classification.jinja2
"""
{% include 'components/examples.jinja2' %}

# Classification Task
{{ task_description }}

{% include 'components/constraints.jinja2' %}
"""

# components/examples.jinja2
"""
# Examples
{% for ex in examples %}
Input: {{ ex.input }}
Output: {{ ex.output }}
{% endfor %}
"""

# Load and render
template = env.get_template('tasks/classification.jinja2')
prompt = template.render(
    examples=[...],
    task_description="...",
    constraints=[...]
)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use named placeholders | Self-documenting, order-independent |
| Validate required variables | Fail fast on missing data |
| Keep templates readable | Others need to understand them |
| Document expected variables | Include type hints and examples |
| Use consistent naming | `snake_case` or `camelCase`, pick one |
| Separate logic from content | Templates shouldn't contain business logic |

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Unescaped user input | Prompt injection risk | Sanitize or escape variables |
| Missing variable crashes | Runtime errors | Use safe substitution |
| Over-complex templates | Hard to maintain | Extract logic to Python |
| No default values | Empty sections look broken | Provide sensible defaults |
| Inconsistent formatting | Confusing prompts | Use linters/formatters |

---

## Hands-on Exercise

### Your Task

Create a reusable template system for customer support responses that:
1. Supports multiple languages (English, Spanish)
2. Has conditional sections for different issue types
3. Includes personalization placeholders

<details>
<summary>💡 Hints</summary>

1. Use a dictionary for language-specific strings
2. Use Jinja2 `{% if %}` for issue type branching
3. Include `{{customer_name}}`, `{{issue_summary}}`, `{{resolution}}`

</details>

<details>
<summary>✅ Solution</summary>

```python
from jinja2 import Template

SUPPORT_TEMPLATE = Template("""
{% if language == 'es' %}
Hola {{ customer_name }},

Gracias por contactarnos sobre: {{ issue_summary }}

{% if issue_type == 'billing' %}
Hemos revisado tu cuenta y {{ resolution }}
{% elif issue_type == 'technical' %}
Nuestro equipo técnico ha investigado y {{ resolution }}
{% else %}
{{ resolution }}
{% endif %}

¿Hay algo más en lo que podamos ayudarte?

Saludos,
Equipo de Soporte
{% else %}
Hi {{ customer_name }},

Thank you for reaching out about: {{ issue_summary }}

{% if issue_type == 'billing' %}
We've reviewed your account and {{ resolution }}
{% elif issue_type == 'technical' %}
Our technical team has investigated and {{ resolution }}
{% else %}
{{ resolution }}
{% endif %}

Is there anything else we can help you with?

Best regards,
Support Team
{% endif %}
""")

# Usage
response = SUPPORT_TEMPLATE.render(
    language="en",
    customer_name="Alice",
    issue_summary="payment not processing",
    issue_type="billing",
    resolution="we've updated your payment method and your next charge will process normally."
)
```

</details>

---

## Summary

- Placeholder syntax varies by system—pick one and be consistent
- Conditionals enable flexible, context-aware prompts
- Loops handle dynamic lists like examples and tool definitions
- Template inheritance reduces duplication across similar prompts
- Composition pattern builds prompts from reusable components
- Always validate required variables and provide defaults

**Next:** [OpenAI Reusable Prompts](./02-openai-reusable-prompts.md)

---

<!-- Sources: Jinja2 documentation patterns, OpenAI Prompt Engineering Guide -->
