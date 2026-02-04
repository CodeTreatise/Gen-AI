---
title: "PromptTemplate Basics"
---

# PromptTemplate Basics

## Introduction

`PromptTemplate` is LangChain's foundational class for creating reusable text templates with variable placeholders. While `ChatPromptTemplate` is preferred for modern chat models, understanding `PromptTemplate` is essential—it's the building block upon which all other templates are built.

In this lesson, we'll master creating templates from strings, validating input variables, loading templates from files, and understanding template formatting options.

### What We'll Cover

- Creating templates with `from_template()`
- Variable placeholders and validation
- Input variables vs partial variables
- Loading templates from files
- Template formatting options (f-string, Jinja2, mustache)
- The Runnable interface for templates

### Prerequisites

- Python string formatting basics
- LangChain installation complete

---

## Creating Simple Templates

### The `from_template()` Method

The most common way to create a `PromptTemplate`:

```python
from langchain.prompts import PromptTemplate

# Create from a template string
template = PromptTemplate.from_template(
    "Write a {length} poem about {topic}."
)

# Check input variables
print(template.input_variables)  # ['length', 'topic']

# Format the template
result = template.format(length="short", topic="nature")
print(result)
```

**Output:**
```
Write a short poem about nature.
```

### Direct Instantiation

You can also create templates by specifying input variables explicitly:

```python
from langchain.prompts import PromptTemplate

# Explicit variable declaration
template = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write marketing copy for {product} targeting {audience}."
)

result = template.format(product="headphones", audience="gamers")
print(result)
```

**Output:**
```
Write marketing copy for headphones targeting gamers.
```

> **Note:** Using `from_template()` is preferred—it automatically extracts input variables from the template string, reducing potential errors.

---

## Variable Placeholders

### F-String Style (Default)

LangChain uses Python's f-string syntax by default:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Hello {name}! You are {age} years old."
)

print(template.format(name="Alice", age=30))
```

**Output:**
```
Hello Alice! You are 30 years old.
```

### Escaping Curly Braces

To include literal curly braces in your template:

```python
from langchain.prompts import PromptTemplate

# Double braces escape to single braces
template = PromptTemplate.from_template(
    "Format JSON like this: {{'key': '{value}'}}"
)

print(template.format(value="example"))
```

**Output:**
```
Format JSON like this: {'key': 'example'}
```

### Multiline Templates

Use triple quotes for longer prompts:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("""
You are an expert {role}.

Context:
{context}

Question: {question}

Please provide a detailed answer:
""")

result = template.format(
    role="data scientist",
    context="We have a dataset with 1M rows of customer transactions.",
    question="How should we approach churn prediction?"
)

print(result)
```

---

## Input Validation

### Automatic Validation

Templates validate that all required variables are provided:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Summarize {text} in {style} style."
)

# Missing 'style' raises an error
try:
    template.format(text="Hello world")
except KeyError as e:
    print(f"Missing variable: {e}")
```

**Output:**
```
Missing variable: 'style'
```

### Checking Required Variables

Inspect what variables a template needs:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Create a {format} about {topic} for {audience}."
)

print(f"Required variables: {template.input_variables}")
print(f"Template string: {template.template}")
```

**Output:**
```
Required variables: ['format', 'topic', 'audience']
Template string: Create a {format} about {topic} for {audience}.
```

### Extra Variables Warning

By default, extra variables are ignored:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("Hello {name}!")

# Extra variable 'age' is ignored
result = template.format(name="Bob", age=25)
print(result)  # "Hello Bob!"
```

---

## Partial Variables

Partial variables let you pre-fill some template values:

### Using `partial()`

```python
from langchain.prompts import PromptTemplate

# Original template with 3 variables
template = PromptTemplate.from_template(
    "You are a {role}. Respond in {language}. Question: {question}"
)

# Create a partial with 'role' pre-filled
expert_template = template.partial(role="Python expert")

# Now only need 2 variables
print(expert_template.input_variables)  # ['language', 'question']

result = expert_template.format(
    language="English",
    question="What are decorators?"
)
print(result)
```

**Output:**
```
['language', 'question']
You are a Python expert. Respond in English. Question: What are decorators?
```

### Callable Partial Variables

Use functions for dynamic values:

```python
from datetime import datetime
from langchain.prompts import PromptTemplate

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

template = PromptTemplate(
    input_variables=["question"],
    partial_variables={"date": get_current_date},
    template="Today is {date}. Question: {question}"
)

# Date is computed at format time
result = template.format(question="What day is it?")
print(result)
```

**Output:**
```
Today is 2026-02-03. Question: What day is it?
```

### Partial at Instantiation

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Language: {language}. Code: {code}",
    partial_variables={"language": "Python"}
)

print(template.input_variables)  # ['code']
print(template.format(code="print('hello')"))
```

---

## Loading Templates from Files

### Template Files

Store templates in separate files for easier management:

**prompts/summarize.txt:**
```
Summarize the following text in {style} style.

Text:
{text}

Summary:
```

### Loading from File

```python
from langchain.prompts import PromptTemplate
from pathlib import Path

# Read template from file
template_path = Path("prompts/summarize.txt")
template_content = template_path.read_text()

template = PromptTemplate.from_template(template_content)
print(template.input_variables)  # ['style', 'text']
```

### Using `load_prompt()`

For YAML/JSON template files:

**prompts/qa_prompt.yaml:**
```yaml
_type: prompt
input_variables:
  - context
  - question
template: |
  Answer the question based on the context below.
  
  Context: {context}
  
  Question: {question}
  
  Answer:
```

```python
from langchain.prompts import load_prompt

template = load_prompt("prompts/qa_prompt.yaml")
print(template.input_variables)  # ['context', 'question']
```

### JSON Template Files

**prompts/code_review.json:**
```json
{
  "_type": "prompt",
  "input_variables": ["code", "language"],
  "template": "Review this {language} code:\n\n{code}\n\nProvide feedback:"
}
```

```python
from langchain.prompts import load_prompt

template = load_prompt("prompts/code_review.json")
result = template.format(language="Python", code="def add(a,b): return a+b")
print(result)
```

---

## Template Format Options

### F-String (Default)

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Hello {name}!",
    template_format="f-string"  # default
)
```

### Jinja2 Templates

For more complex logic:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "{% if formal %}Dear {{ name }},{% else %}Hey {{ name }}!{% endif %}",
    template_format="jinja2"
)

# Jinja2 uses double braces and supports logic
formal_result = template.format(name="Dr. Smith", formal=True)
casual_result = template.format(name="Bob", formal=False)

print(formal_result)  # "Dear Dr. Smith,"
print(casual_result)  # "Hey Bob!"
```

### Mustache Templates

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Hello {{name}}! Welcome to {{place}}.",
    template_format="mustache"
)

result = template.format(name="Alice", place="LangChain")
print(result)
```

> **Note:** Jinja2 requires the `jinja2` package: `pip install jinja2`

---

## The Runnable Interface

Templates implement the Runnable protocol, enabling seamless chaining:

### Using `invoke()`

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("Explain {concept} simply.")

# invoke() is the standard Runnable method
result = template.invoke({"concept": "recursion"})
print(type(result))  # StringPromptValue
print(result.to_string())  # "Explain recursion simply."
```

### Chaining with Models

```python
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

template = PromptTemplate.from_template("Write a haiku about {topic}.")
model = init_chat_model("gpt-4o")
parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic": "coding"})
print(result)
```

### Batch Processing

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("Translate '{text}' to French.")

results = template.batch([
    {"text": "hello"},
    {"text": "goodbye"},
    {"text": "thank you"}
])

for r in results:
    print(r.to_string())
```

**Output:**
```
Translate 'hello' to French.
Translate 'goodbye' to French.
Translate 'thank you' to French.
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `from_template()` | Auto-extracts variables, reduces errors |
| Store templates in files | Easier version control and reuse |
| Use partial variables for constants | Reduces repetition, cleaner code |
| Validate templates early | Catch missing variables before runtime |
| Prefer f-string format | Simple, fast, widely understood |
| Use Jinja2 for complex logic | Conditionals, loops when needed |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Hardcoding prompts as strings | Use templates for reusability |
| Forgetting to escape `{}` | Use `{{` and `}}` for literals |
| Inconsistent variable naming | Use snake_case consistently |
| Not checking `input_variables` | Always verify required variables |
| Using wrong template format | Match format to your syntax |

---

## Hands-on Exercise

### Your Task

Create a flexible code review template that:
1. Takes `language`, `code`, and `review_type` as variables
2. Has a partial variable for `current_date` using a function
3. Produces a well-structured prompt for code review

### Requirements

1. Create a multiline template with clear sections
2. Use partial variables for the date
3. Format and print the result
4. Verify input_variables are correct

### Expected Result

```
Code Review Request
Date: 2026-02-03
Language: Python
Review Type: security

Code:
def login(user, password):
    query = f"SELECT * FROM users WHERE user='{user}'"
    return db.execute(query)

Please review this code focusing on security issues.
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a function that returns the current date
- Use `partial_variables` in the template constructor
- Triple-quoted strings work well for multiline templates
- Remember to check `input_variables` after creating

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from datetime import datetime
from langchain.prompts import PromptTemplate

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

template = PromptTemplate(
    input_variables=["language", "code", "review_type"],
    partial_variables={"current_date": get_date},
    template="""Code Review Request
Date: {current_date}
Language: {language}
Review Type: {review_type}

Code:
{code}

Please review this code focusing on {review_type} issues.
"""
)

# Verify input variables (should not include current_date)
print(f"Required variables: {template.input_variables}")

# Format the template
result = template.format(
    language="Python",
    review_type="security",
    code="""def login(user, password):
    query = f"SELECT * FROM users WHERE user='{user}'"
    return db.execute(query)"""
)

print(result)
```

</details>

### Bonus Challenges

- [ ] Create a version that loads the template from a YAML file
- [ ] Add validation to ensure `review_type` is one of: security, performance, style
- [ ] Create multiple partial templates for different review types

---

## Summary

✅ `PromptTemplate.from_template()` is the preferred way to create templates  
✅ Templates auto-extract and validate input variables  
✅ **Partial variables** pre-fill values (can be functions for dynamic values)  
✅ Load templates from files for better organization  
✅ Templates are **Runnables**—use `invoke()`, `batch()`, chain with `|`  
✅ Choose format: **f-string** (simple), **Jinja2** (complex logic)  

**Next:** [ChatPromptTemplate](./02-chatprompttemplate.md) — Create message-based prompts for chat models

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [Prompt Templates Overview](./00-prompt-templates.md) | [Prompt Templates](./00-prompt-templates.md) | [ChatPromptTemplate](./02-chatprompttemplate.md) |

<!-- 
Sources Consulted:
- LangChain GitHub prompts/prompt.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/prompts/prompt.py
- LangChain GitHub prompts/string.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/prompts/string.py
-->
