---
title: "Stop Sequences"
---

# Stop Sequences

## Introduction

Stop sequences tell the model when to stop generating text. They're custom termination triggers that give you fine-grained control over output format and length.

### What We'll Cover

- What stop sequences are
- Common patterns
- Multiple stop sequences
- Preventing runaway generation

---

## What Are Stop Sequences?

Stop sequences are strings that, when generated, cause the model to stop producing more text.

### Basic Example

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List 3 fruits:"}],
    stop=["\n\n"]  # Stop at double newline
)

# Without stop: Might generate more content after the list
# With stop: Stops as soon as it generates "\n\n"
```

### How It Works

```
Model generates: "1. Apple\n2. Banana\n3. Orange\n\nNow let me tell you..."
                                              ▲
                                              │
                                        Stop sequence "\n\n" detected
                                              │
                                              ▼
Returned output: "1. Apple\n2. Banana\n3. Orange"
```

The stop sequence itself is NOT included in the output.

---

## Common Stop Patterns

### Double Newline

```python
# Stop at paragraph breaks
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarize this in one paragraph:..."}],
    stop=["\n\n"]
)

# Ensures single paragraph output
```

### Custom Delimiters

```python
# Stop at custom markers
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user", 
        "content": "Extract the name from this text. Respond with just the name followed by END_NAME"
    }],
    stop=["END_NAME"]
)

# Model: "John Smith"  (stops before outputting "END_NAME")
```

### Structured Output Markers

```python
# For Q&A format
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user", 
        "content": """Answer this question:
Q: What is the capital of France?
A:"""
    }],
    stop=["Q:", "\n\n"]  # Stop before next question or double newline
)

# Output: " Paris" (clean, single answer)
```

### Code Block Termination

```python
# Stop at end of code block
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user", 
        "content": "Write a Python function to calculate factorial:\n```python"
    }],
    stop=["```"]
)

# Output: Function code without closing backticks
# Add them back in post-processing if needed
```

---

## Multiple Stop Sequences

You can specify multiple stop sequences—generation stops at whichever comes first.

### Syntax

```python
# Up to 4 stop sequences (OpenAI limit)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    stop=["END", "\n\n", "---", "STOP"]
)

# Stops at whichever appears first in generated text
```

### Use Cases

```python
# Chat application: Stop at user turn markers
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    stop=[
        "User:",      # Stop before next user message
        "Human:",     # Alternative format
        "\n\nUser:",  # With spacing
        "<|endoftext|>"  # End of text marker
    ]
)

# Data extraction: Multiple possible endings
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract the price: 'The item costs $19.99'"}],
    stop=[
        "\n",         # Stop at newline
        ".",          # Stop at period
        " -",         # Stop before explanation
    ]
)
```

---

## Preventing Runaway Generation

Stop sequences help prevent models from generating too much text.

### The Problem

```python
# Without stop sequences, model might:
# 1. Answer the question
# 2. Then provide examples
# 3. Then add disclaimers
# 4. Then suggest follow-ups
# ... continuing until max_tokens

prompt = "What is 2+2?"
# Might generate: "2+2 equals 4. Let me explain why: In mathematics, 
# addition is... Furthermore, you might be interested in knowing that..."
```

### The Solution

```python
# Force concise answers with stop sequences
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user", 
        "content": "Answer in one sentence: What is 2+2?"
    }],
    stop=["\n", "."],  # Stop at first newline or period
    max_tokens=50
)

# Output: "2+2 equals 4"  (just the answer)
```

### Combined with max_tokens

```python
# Belt and suspenders approach
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Define 'algorithm'"}],
    stop=["\n\n", "---"],  # Content delimiter
    max_tokens=100         # Hard limit backup
)

# Stops at whichever comes first:
# - Stop sequence
# - max_tokens limit
# - Natural completion
```

---

## Practical Patterns

### Chat Completion Without Continuation

```python
def single_turn_response(user_message: str) -> str:
    """Get a single response without model continuing the conversation"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Give concise answers."},
            {"role": "user", "content": user_message}
        ],
        stop=[
            "\nUser:",
            "\nHuman:", 
            "\n\nQuestion:",
            "---"
        ]
    )
    
    return response.choices[0].message.content.strip()
```

### JSON Extraction

```python
def extract_json(prompt: str) -> dict:
    """Extract JSON from model output"""
    import json
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"{prompt}\n\nRespond with valid JSON only:"
        }],
        stop=["\n\n", "```"],  # Stop after JSON block
    )
    
    content = response.choices[0].message.content.strip()
    
    # Clean up potential formatting
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    return json.loads(content)
```

### Fill-in-the-Blank

```python
def complete_template(template: str, blank_marker: str = "___") -> str:
    """Fill in blanks in a template"""
    
    parts = template.split(blank_marker)
    if len(parts) < 2:
        return template
    
    # Complete the blank
    prompt = parts[0]  # Text before blank
    after_blank = parts[1] if len(parts) > 1 else ""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Complete this: {prompt}"
        }],
        stop=[after_blank[:10]] if after_blank else ["\n"],  # Stop before next part
        max_tokens=50
    )
    
    completion = response.choices[0].message.content.strip()
    return template.replace(blank_marker, completion, 1)

# Usage
result = complete_template("The capital of France is ___ and it's famous for the Eiffel Tower.")
# "The capital of France is Paris and it's famous for the Eiffel Tower."
```

---

## Stop Sequence Limitations

### What Doesn't Work

```python
# Stop sequences are exact matches, not patterns

# ❌ This won't work as regex
stop=["\\d+"]  # Doesn't match numbers

# ❌ This won't work as partial match
stop=["end"]  # Won't stop at "ending" or "ended"

# ✅ Must be exact string
stop=["end"]  # Only stops at exactly "end"
```

### Maximum Count

```python
# OpenAI: Maximum 4 stop sequences
# Anthropic: Maximum 4 stop sequences
# Google: Varies

# If you need more, prioritize the most important ones
```

### Tokens vs Characters

```python
# Stop sequences are matched character-by-character
# But generation happens token-by-token

# This means stop might occur mid-token conceptually
# But practically, the model generates full tokens

# Example: stop=["ing"]
# If model generates token "running" (as single token)
# It won't stop at "runn" before "ing"
# It will generate "running" then check for "ing"
```

---

## Hands-on Exercise

### Your Task

Experiment with stop sequences for different formats:

```python
from openai import OpenAI

client = OpenAI()

# Task 1: Single sentence answer
def get_single_sentence(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}],
        stop=["\n", ".", "!", "?"],  # Stop at sentence end
        max_tokens=100
    )
    return response.choices[0].message.content + "."  # Add period back

# Task 2: List extraction
def get_list_items(topic: str, count: int = 3) -> list:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"List exactly {count} {topic}, one per line:"
        }],
        stop=["\n\n", f"{count+1}.", f"{count+1})"],  # Stop before extra items
        max_tokens=200
    )
    items = response.choices[0].message.content.strip().split("\n")
    return [item.strip().lstrip("0123456789.-) ") for item in items if item.strip()]

# Task 3: Q&A format
def qa_format(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Q: {question}\nA:"
        }],
        stop=["Q:", "\n\n", "\nNote:"],  # Stop before follow-ups
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Test each
print("Single sentence:", get_single_sentence("What is Python?"))
print("\nList items:", get_list_items("programming languages"))
print("\nQ&A:", qa_format("What is machine learning?"))
```

---

## Summary

✅ **Stop sequences** are strings that trigger generation termination

✅ The **stop sequence itself** is not included in output

✅ **Multiple sequences** (up to 4) can be specified—first match wins

✅ **Common patterns**: "\n\n", "END", role markers, code block endings

✅ **Combine with max_tokens** for robust output control

✅ Stop sequences are **exact matches**, not patterns

**Next:** [Advanced Parameters](./07-advanced-parameters.md)

---

## Further Reading

- [OpenAI Stop Parameter](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop) — Official documentation
- [Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) — Using stop sequences effectively

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Max Tokens](./05-max-tokens.md) | [Model Parameters](./00-model-parameters-settings.md) | [Advanced Parameters](./07-advanced-parameters.md) |

