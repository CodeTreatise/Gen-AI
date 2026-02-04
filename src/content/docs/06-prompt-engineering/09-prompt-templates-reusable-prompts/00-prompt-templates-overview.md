---
title: "Prompt Templates & Reusable Prompts"
---

# Prompt Templates & Reusable Prompts

## Introduction

As AI applications scale, managing prompts becomes a critical engineering challenge. Hardcoded prompts scattered across codebases lead to inconsistency, difficult maintenance, and missed optimization opportunities. Prompt templates solve this by separating prompt logic from application code—making prompts reusable, testable, and version-controlled.

> **🔑 Key Insight:** Treat prompts like code. They deserve version control, testing, documentation, and proper software engineering practices.

### What You'll Learn

This lesson covers the complete lifecycle of production prompt management:

1. **Template Design Patterns** — Placeholder syntax, conditionals, loops, inheritance
2. **OpenAI Reusable Prompts** — Dashboard-based prompt management and A/B testing
3. **Prompt Libraries** — Organizing and managing prompts at scale
4. **Variable Substitution** — Runtime value injection and type handling
5. **Dynamic Prompt Construction** — Building prompts programmatically
6. **Template Versioning** — Change tracking, rollback, and A/B testing
7. **Template Testing** — Unit tests, validation, and regression testing

### Prerequisites

- [Fundamentals of Effective Prompts](../01-fundamentals-of-effective-prompts/00-fundamentals-overview.md)
- [System Prompts & Developer Messages](../02-system-prompts-developer-messages/00-system-prompts-overview.md)
- Basic programming experience (Python examples throughout)

---

## Why Templates Matter

### The Problem with Hardcoded Prompts

```python
# ❌ Anti-pattern: Prompts scattered in code
def analyze_document(doc):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": "You are a document analyst. Extract key points..."
        }]
    )
```

**Problems:**
- Prompts hidden in application logic
- No version history
- Can't test prompts independently
- Changes require code deployments
- No visibility for non-engineers

### The Template Solution

```python
# ✅ Better: Templates separate concerns
from prompt_library import load_prompt

def analyze_document(doc):
    prompt = load_prompt("document-analysis", version="2.1")
    response = client.chat.completions.create(
        model=prompt.model,
        messages=prompt.render(document=doc)
    )
```

**Benefits:**
- Prompts managed separately from code
- Version controlled with rollback capability
- Testable in isolation
- Non-engineers can iterate on prompts
- A/B testing without code changes

---

## Template Approaches Comparison

| Approach | Best For | Complexity | Features |
|----------|----------|------------|----------|
| **String Templates** | Simple variable insertion | Low | Basic `{variable}` substitution |
| **Jinja2 Templates** | Conditionals, loops, inheritance | Medium | Full templating language |
| **OpenAI Dashboard** | Team collaboration, A/B testing | Low | Built-in versioning, no code |
| **Prompt Libraries** | Large-scale management | High | Metadata, search, analytics |
| **LangChain Templates** | AI application frameworks | Medium | Integration with chains |

---

## Choosing the Right Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPLATE SELECTION GUIDE                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ How many prompts do you have? │
              └───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      1-5 prompts        5-20 prompts        20+ prompts
           │                  │                  │
           ▼                  ▼                  ▼
    String Templates    Jinja2/LangChain    Prompt Library
    or OpenAI Dash      + Version Control   with Database
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   Need team collaboration?    │
              └───────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                   Yes                  No
                    │                   │
                    ▼                   ▼
           OpenAI Dashboard       File-based
           or Prompt DB           templates OK
```

---

## Quick Start Examples

### Python String Templates

```python
from string import Template

prompt_template = Template("""
You are a $role specialist.

Analyze the following $document_type:
$content

Focus on: $focus_areas
""")

rendered = prompt_template.substitute(
    role="legal",
    document_type="contract",
    content=contract_text,
    focus_areas="liability clauses, termination terms"
)
```

### OpenAI Reusable Prompts

```javascript
// Use prompt from OpenAI Dashboard
const response = await client.responses.create({
    model: "gpt-5",
    prompt: {
        id: "pmpt_abc123",
        version: "2",
        variables: {
            customer_name: "Jane Doe",
            product: "Enterprise Plan"
        }
    }
});
```

### Jinja2 Templates

```python
from jinja2 import Template

prompt = Template("""
# Task: {{ task_type | title }}

{% if examples %}
## Examples
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}
{% endfor %}
{% endif %}

## Your Task
{{ user_input }}
""")

rendered = prompt.render(
    task_type="classification",
    examples=[{"input": "Great!", "output": "positive"}],
    user_input="This is terrible."
)
```

---

## Prompt Caching Optimization

Templates work hand-in-hand with prompt caching. Structure templates with static content first:

```
┌────────────────────────────────────────────┐
│  STATIC CONTENT (Cacheable)                │
│  ─────────────────────────────             │
│  • System instructions                     │
│  • Role definitions                        │
│  • Examples (few-shot)                     │
│  • Tool definitions                        │
│  • Background context                      │
├────────────────────────────────────────────┤
│  DYNAMIC CONTENT (Variable)                │
│  ─────────────────────────────             │
│  • User-specific data                      │
│  • Current request details                 │
│  • Session context                         │
└────────────────────────────────────────────┘
```

> **⏱️ Performance:** OpenAI caches prompts ≥1024 tokens automatically. Anthropic requires explicit `cache_control` markers. Static prefixes can reduce latency by up to 80%.

---

## Lesson Navigation

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-template-design-patterns.md) | Template Design Patterns | Placeholders, conditionals, loops, inheritance |
| [02](./02-openai-reusable-prompts.md) | OpenAI Reusable Prompts | Dashboard, variables, versioning, A/B testing |
| [03](./03-prompt-libraries.md) | Prompt Libraries | Organization, metadata, centralized management |
| [04](./04-variable-substitution.md) | Variable Substitution | Runtime injection, types, escaping, defaults |
| [05](./05-dynamic-prompt-construction.md) | Dynamic Construction | Programmatic building, conditionals, context |
| [06](./06-template-versioning.md) | Template Versioning | Version control, rollback, A/B testing |
| [07](./07-template-testing-frameworks.md) | Template Testing | Unit tests, validation, regression testing |

---

## Summary

- Prompt templates separate prompt logic from application code
- Multiple approaches exist: string templates → full prompt libraries
- OpenAI Dashboard provides no-code template management
- Structure templates with static content first for caching benefits
- Version control and testing are essential for production prompts

**Next:** [Template Design Patterns](./01-template-design-patterns.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide (Reusable Prompts), OpenAI Prompt Caching Guide, Anthropic Prompt Caching -->
