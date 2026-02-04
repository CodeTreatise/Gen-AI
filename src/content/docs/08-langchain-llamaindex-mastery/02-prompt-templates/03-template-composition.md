---
title: "Template Composition"
---

# Template Composition

## Introduction

As your LangChain applications grow, you'll need to combine templates, share common components, and dynamically select prompts based on context. Template composition enables **reusable**, **maintainable**, and **flexible** prompt architectures.

This lesson covers techniques for combining templates, creating partial templates, implementing inheritance patterns, and selecting prompts dynamically at runtime.

### What We'll Cover

- Combining templates with the `+` operator
- Partial templates for pre-filled values
- Nested template composition
- Template inheritance patterns
- Dynamic template selection with RunnableBranch
- Building prompt libraries

### Prerequisites

- ChatPromptTemplate mastery (Lesson 8.2.2)
- Understanding of Runnables and LCEL

---

## Combining Templates

### The `+` Operator

Templates can be combined using the `+` operator:

```python
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Create component templates
system = SystemMessagePromptTemplate.from_template(
    "You are an expert {role}."
)

human = HumanMessagePromptTemplate.from_template(
    "Question: {question}"
)

# Combine with +
combined = system + human

print(type(combined))  # ChatPromptTemplate
print(combined.input_variables)  # ['role', 'question']
```

### Building Templates Incrementally

```python
from langchain.prompts import ChatPromptTemplate

# Start with a base
base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.")
])

# Add context
with_context = base + [("human", "Context: {context}")]

# Add the question
full_template = with_context + [("human", "Question: {question}")]

print(full_template.input_variables)  # ['context', 'question']

messages = full_template.format_messages(
    context="Python is a programming language.",
    question="What is Python used for?"
)
```

### Combining Multiple Message Templates

```python
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate
)

# Define reusable components
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a {persona} assistant."
)

example_exchange = (
    HumanMessagePromptTemplate.from_template("Example: {example_input}")
    + AIMessagePromptTemplate.from_template("{example_output}")
)

user_query = HumanMessagePromptTemplate.from_template("{user_input}")

# Combine all components
full_prompt = system_prompt + example_exchange + user_query

print(full_prompt.input_variables)
# ['persona', 'example_input', 'example_output', 'user_input']
```

---

## Partial Templates

Partial templates pre-fill some variables while leaving others for runtime:

### Using `partial()`

```python
from langchain.prompts import ChatPromptTemplate

# Original template with many variables
template = ChatPromptTemplate.from_messages([
    ("system", """You are {persona} for {company}.
Respond in {language}.
Today's date: {date}."""),
    ("human", "{question}")
])

print(f"Original variables: {template.input_variables}")
# ['persona', 'company', 'language', 'date', 'question']

# Create a partial with fixed values
support_template = template.partial(
    persona="a customer support agent",
    company="TechCorp",
    language="English"
)

print(f"Partial variables: {support_template.input_variables}")
# ['date', 'question']
```

### Callable Partials

Use functions for dynamic values:

```python
from datetime import datetime
from langchain.prompts import ChatPromptTemplate

def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

template = ChatPromptTemplate.from_messages([
    ("system", "{greeting}! Today is {date}. You are a helpful assistant."),
    ("human", "{question}")
])

# Functions are called at format time
dynamic_template = template.partial(
    greeting=get_greeting,
    date=get_current_date
)

messages = dynamic_template.format_messages(question="What time is it?")
print(messages[0].content)
# "Good afternoon! Today is February 03, 2026. You are a helpful assistant."
```

### Partial at Construction

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You work for {company}. Respond in {tone} tone."),
        ("human", "{input}")
    ],
    partial_variables={
        "company": "Acme Inc",
        "tone": "professional"
    }
)

print(template.input_variables)  # ['input']
```

---

## Nested Template Composition

### Composing Complex Prompts

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Inner template for formatting context
context_template = PromptTemplate.from_template("""
Relevant Information:
- Source: {source}
- Content: {content}
- Relevance Score: {score}
""")

# Format context items
contexts = [
    {"source": "doc1.pdf", "content": "Python is versatile", "score": 0.95},
    {"source": "doc2.pdf", "content": "Python supports OOP", "score": 0.87}
]

formatted_contexts = "\n".join([
    context_template.format(**ctx) for ctx in contexts
])

# Use in chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use the provided context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

messages = chat_template.format_messages(
    context=formatted_contexts,
    question="What are Python's key features?"
)
```

### Template Factories

Create functions that generate templates:

```python
from langchain.prompts import ChatPromptTemplate

def create_persona_template(persona: str, expertise: list[str]) -> ChatPromptTemplate:
    """Factory function for persona-specific templates."""
    expertise_str = ", ".join(expertise)
    
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are {persona}.
Your areas of expertise: {expertise_str}.
Provide detailed, accurate responses in your area of expertise.
If asked about something outside your expertise, acknowledge the limitation."""),
        ("human", "{question}")
    ])

# Create specialized templates
python_expert = create_persona_template(
    "a Python programming expert",
    ["Python", "Django", "FastAPI", "data structures"]
)

ml_expert = create_persona_template(
    "a machine learning specialist", 
    ["neural networks", "transformers", "scikit-learn", "PyTorch"]
)

# Use the templates
response = python_expert.format_messages(
    question="Explain Python decorators"
)
```

---

## Template Inheritance Patterns

### Base Template Pattern

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptLibrary:
    """Centralized prompt template management."""
    
    @staticmethod
    def base_chat() -> ChatPromptTemplate:
        """Base template with common structure."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant for {company}."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}")
        ])
    
    @staticmethod
    def support_chat() -> ChatPromptTemplate:
        """Extended template for customer support."""
        base = PromptLibrary.base_chat()
        return base.partial(company="TechCorp")
    
    @staticmethod
    def sales_chat() -> ChatPromptTemplate:
        """Extended template for sales assistance."""
        base = PromptLibrary.base_chat()
        return base.partial(company="TechCorp Sales Team")

# Usage
support_template = PromptLibrary.support_chat()
sales_template = PromptLibrary.sales_chat()
```

### Extending Templates

```python
from langchain.prompts import ChatPromptTemplate

def extend_with_context(
    base_template: ChatPromptTemplate,
    context_instructions: str
) -> ChatPromptTemplate:
    """Extend a base template with additional context."""
    
    # Get existing messages
    messages = list(base_template.messages)
    
    # Insert context after system message
    context_msg = ("system", context_instructions)
    messages.insert(1, context_msg)
    
    return ChatPromptTemplate.from_messages(messages)

# Base template
base = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}")
])

# Extended with RAG context
rag_template = extend_with_context(
    base,
    "Use this context to answer: {context}"
)

print(rag_template.input_variables)  # ['context', 'input']
```

---

## Dynamic Template Selection

### Using RunnableBranch

Select templates based on input conditions:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Define specialized templates
coding_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert programmer. Provide code examples."),
    ("human", "{question}")
])

writing_template = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer. Be eloquent and engaging."),
    ("human", "{question}")
])

general_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Create routing logic
def is_coding_question(x: dict) -> bool:
    keywords = ["code", "programming", "function", "debug", "error"]
    return any(kw in x["question"].lower() for kw in keywords)

def is_writing_question(x: dict) -> bool:
    keywords = ["write", "story", "essay", "poem", "creative"]
    return any(kw in x["question"].lower() for kw in keywords)

# Create branch
template_router = RunnableBranch(
    (is_coding_question, coding_template),
    (is_writing_question, writing_template),
    general_template  # default
)

# Test routing
coding_result = template_router.invoke({"question": "Write a Python function"})
writing_result = template_router.invoke({"question": "Write a poem about AI"})
general_result = template_router.invoke({"question": "What is the weather?"})
```

### Category-Based Selection

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Template registry
TEMPLATES = {
    "technical": ChatPromptTemplate.from_messages([
        ("system", "You are a technical expert. Be precise and detailed."),
        ("human", "{question}")
    ]),
    "casual": ChatPromptTemplate.from_messages([
        ("system", "You're a friendly assistant. Keep it light and fun!"),
        ("human", "{question}")
    ]),
    "formal": ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant. Maintain formal tone."),
        ("human", "{question}")
    ])
}

def select_template(inputs: dict):
    """Select template based on 'style' key."""
    style = inputs.get("style", "casual")
    template = TEMPLATES.get(style, TEMPLATES["casual"])
    return template.invoke({"question": inputs["question"]})

template_selector = RunnableLambda(select_template)

# Usage
result = template_selector.invoke({
    "style": "technical",
    "question": "Explain neural networks"
})
```

---

## Building Prompt Libraries

### Structured Prompt Organization

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class PromptConfig:
    """Configuration for a prompt template."""
    name: str
    system_prompt: str
    include_history: bool = False
    partial_vars: Optional[dict] = None

class PromptRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self):
        self._templates: dict[str, ChatPromptTemplate] = {}
    
    def register(self, config: PromptConfig) -> None:
        """Register a new prompt template."""
        messages = [("system", config.system_prompt)]
        
        if config.include_history:
            messages.append(MessagesPlaceholder("history", optional=True))
        
        messages.append(("human", "{input}"))
        
        template = ChatPromptTemplate.from_messages(messages)
        
        if config.partial_vars:
            template = template.partial(**config.partial_vars)
        
        self._templates[config.name] = template
    
    def get(self, name: str) -> ChatPromptTemplate:
        """Retrieve a template by name."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        return self._templates[name]
    
    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return list(self._templates.keys())

# Usage
registry = PromptRegistry()

registry.register(PromptConfig(
    name="support",
    system_prompt="You are a customer support agent. Be helpful and patient.",
    include_history=True
))

registry.register(PromptConfig(
    name="code_review",
    system_prompt="You are a senior developer reviewing code. Be constructive.",
    include_history=False
))

# Get and use templates
support = registry.get("support")
code_review = registry.get("code_review")
```

### File-Based Template Management

```python
import yaml
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def load_templates_from_yaml(file_path: str) -> dict[str, ChatPromptTemplate]:
    """Load multiple templates from a YAML file."""
    with open(file_path) as f:
        config = yaml.safe_load(f)
    
    templates = {}
    for name, template_config in config.items():
        messages = []
        
        for msg in template_config["messages"]:
            if msg["type"] == "placeholder":
                messages.append(MessagesPlaceholder(
                    msg["name"],
                    optional=msg.get("optional", False)
                ))
            else:
                messages.append((msg["type"], msg["content"]))
        
        templates[name] = ChatPromptTemplate.from_messages(messages)
    
    return templates

# prompts.yaml:
# support:
#   messages:
#     - type: system
#       content: "You are a support agent."
#     - type: placeholder
#       name: history
#       optional: true
#     - type: human
#       content: "{input}"
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use partials for constants | Reduces repetition, easier updates |
| Create template factories | Consistent patterns, less duplication |
| Centralize in a registry | Single source of truth |
| Version control templates | Track changes, rollback if needed |
| Document input variables | Clear expectations for callers |
| Test templates in isolation | Catch format errors early |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Duplicating template code | Use factories and composition |
| Hardcoding values | Use partials for configurable defaults |
| Complex nested strings | Break into smaller, composable parts |
| No template validation | Test templates before deployment |
| Inconsistent variable names | Establish naming conventions |

---

## Hands-on Exercise

### Your Task

Build a prompt library with:
1. A base template with common structure
2. Three specialized variants (support, sales, technical)
3. A router that selects based on input category
4. Dynamic date/time injection

### Requirements

1. Create a `PromptLibrary` class with factory methods
2. Use `partial()` to customize base template
3. Implement `RunnableBranch` for routing
4. Include callable partials for date

### Expected Result

```python
library = PromptLibrary()
router = library.create_router()

# Routes to support template
result1 = router.invoke({"category": "support", "input": "Help!"})

# Routes to sales template  
result2 = router.invoke({"category": "sales", "input": "Pricing?"})
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with a base template that has all common elements
- Use `partial()` to create specialized versions
- `RunnableBranch` takes `(condition, runnable)` tuples
- Callable partials are invoked at format time

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from datetime import datetime
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

class PromptLibrary:
    """Library of composable prompt templates."""
    
    def __init__(self, company: str = "TechCorp"):
        self.company = company
    
    def _base_template(self) -> ChatPromptTemplate:
        """Base template with common structure."""
        return ChatPromptTemplate.from_messages([
            ("system", """Company: {company}
Timestamp: {timestamp}
Department: {department}

{instructions}"""),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}")
        ])
    
    def support_template(self) -> ChatPromptTemplate:
        """Customer support template."""
        return self._base_template().partial(
            company=self.company,
            timestamp=get_timestamp,
            department="Customer Support",
            instructions="Be helpful, patient, and empathetic. Resolve issues efficiently."
        )
    
    def sales_template(self) -> ChatPromptTemplate:
        """Sales assistance template."""
        return self._base_template().partial(
            company=self.company,
            timestamp=get_timestamp,
            department="Sales",
            instructions="Be enthusiastic and helpful. Focus on customer needs and value."
        )
    
    def technical_template(self) -> ChatPromptTemplate:
        """Technical support template."""
        return self._base_template().partial(
            company=self.company,
            timestamp=get_timestamp,
            department="Technical Support",
            instructions="Be precise and technical. Provide detailed solutions with examples."
        )
    
    def create_router(self):
        """Create a router that selects template based on category."""
        return RunnableBranch(
            (lambda x: x.get("category") == "support", self.support_template()),
            (lambda x: x.get("category") == "sales", self.sales_template()),
            (lambda x: x.get("category") == "technical", self.technical_template()),
            self.support_template()  # default
        )

# Usage
library = PromptLibrary(company="Acme Corp")
router = library.create_router()

# Test routing
support_result = router.invoke({"category": "support", "input": "I need help!"})
sales_result = router.invoke({"category": "sales", "input": "What's the pricing?"})
technical_result = router.invoke({"category": "technical", "input": "API error code"})

print("Support template messages:")
for msg in support_result.to_messages():
    print(f"  {msg.type}: {msg.content[:80]}...")
```

</details>

### Bonus Challenges

- [ ] Add template versioning with metadata
- [ ] Implement A/B testing between template variants
- [ ] Create a CLI tool for managing templates
- [ ] Add template validation on registration

---

## Summary

✅ Use `+` operator to combine templates incrementally  
✅ **Partial templates** pre-fill values for reuse  
✅ **Callable partials** enable dynamic values (dates, greetings)  
✅ Template factories create consistent, specialized prompts  
✅ **RunnableBranch** enables dynamic template selection  
✅ Prompt registries centralize template management  

**Next:** [Few-Shot Prompting](./04-few-shot-prompting.md) — Add examples for in-context learning

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [ChatPromptTemplate](./02-chatprompttemplate.md) | [Prompt Templates](./00-prompt-templates.md) | [Few-Shot Prompting](./04-few-shot-prompting.md) |

<!-- 
Sources Consulted:
- LangChain GitHub prompts/chat.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/prompts/chat.py
- LangChain GitHub runnables: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/runnables/
-->
