---
title: "Anthropic"
---

# Anthropic

## Introduction

Anthropic, founded by former OpenAI researchers, focuses on AI safety and has developed the Claude family of models. Known for thoughtful responses, strong coding abilities, and features like computer use.

### What We'll Cover

- Claude model family
- Constitutional AI approach
- Extended context windows
- Computer use capabilities

---

## Model Lineup

### Current Models (2025-2026)

| Model | Context | Best For |
|-------|---------|----------|
| Claude 3.5 Sonnet | 200K | Coding, analysis, general |
| Claude 3.5 Haiku | 200K | Fast, cost-effective |
| Claude 3 Opus | 200K | Complex reasoning |
| Claude 4 | 300K+ | Latest flagship |

### Model Selection

```python
from anthropic import Anthropic

client = Anthropic()

def select_claude_model(task: dict) -> str:
    """Select appropriate Claude model"""
    
    if task.get("needs_speed"):
        return "claude-3-5-haiku-20241022"
    
    if task.get("needs_maximum_quality"):
        return "claude-4"  # When available
    
    if task.get("context_tokens", 0) > 200000:
        return "claude-4"  # Larger context
    
    # Default: Best balance
    return "claude-3-5-sonnet-20241022"
```

---

## Claude 3.5 Sonnet

### Capabilities

```python
def claude_chat(prompt: str) -> str:
    """Basic Claude chat"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# With system prompt
def claude_with_system(system: str, prompt: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

### Vision

```python
import base64

def claude_vision(image_path: str, question: str) -> str:
    """Analyze image with Claude"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {"type": "text", "text": question}
            ]
        }]
    )
    
    return response.content[0].text
```

---

## Constitutional AI

### Approach

```python
constitutional_ai_principles = {
    "harmlessness": "Avoid outputs that could cause harm",
    "helpfulness": "Be genuinely useful to users",
    "honesty": "Be truthful, acknowledge uncertainty",
    "training": "Model critiques and revises its own outputs",
    "result": "More aligned behavior without extensive RLHF"
}

# Claude's training includes:
# 1. Generate response
# 2. Self-critique against principles
# 3. Revise response
# 4. Learn from revision process
```

### Safety Features

```python
# Claude has built-in safety features
# - Refuses harmful requests
# - Acknowledges limitations
# - Provides balanced perspectives
# - Warns about sensitive topics

# Example of Claude's careful approach
response = claude_chat(
    "What are the pros and cons of controversial topic X?"
)
# Claude will provide balanced analysis, acknowledge complexity
```

---

## Extended Context

### 200K Token Context

```python
def analyze_long_document(document: str, question: str) -> str:
    """Analyze very long documents"""
    
    # Claude 3.5 Sonnet: 200K tokens (~500 pages)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Document:
{document}

Question: {question}"""
        }]
    )
    
    return response.content[0].text
```

### PDF Analysis

```python
def analyze_pdf(pdf_path: str, prompt: str) -> str:
    """Native PDF analysis"""
    
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    return response.content[0].text
```

---

## Tool Use

### Function Calling

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    }
}]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}]
)

# Check for tool use
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")
```

---

## Computer Use

### Desktop Automation

```python
def computer_use_session(task: str):
    """Use Claude to control a computer"""
    
    tools = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
            "display_number": 1
        },
        {
            "type": "text_editor_20241022",
            "name": "str_replace_editor"
        },
        {
            "type": "bash_20241022",
            "name": "bash"
        }
    ]
    
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=tools,
        messages=[{"role": "user", "content": task}],
        betas=["computer-use-2024-10-22"]
    )
    
    return response
```

---

## Pricing

### Current Pricing

| Model | Input/1M | Output/1M |
|-------|----------|-----------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3.5 Haiku | $0.25 | $1.25 |
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 4 | TBD | TBD |

### Prompt Caching

```python
# Anthropic supports prompt caching
# Cache long system prompts or few-shot examples
# Significant savings for repeated context
```

---

## API Features

### Streaming

```python
def stream_claude(prompt: str):
    """Stream Claude response"""
    
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
```

### Message Batches

```python
# Process many messages at 50% cost
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        }
        # ... more requests
    ]
)
```

---

## Summary

✅ **Claude 3.5 Sonnet**: Excellent for coding and analysis

✅ **200K context**: Long document processing

✅ **Computer use**: Unique GUI automation capability

✅ **Safety-focused**: Constitutional AI approach

✅ **Native PDF**: Document analysis without preprocessing

**Next:** [Google](./03-google.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [OpenAI](./01-openai.md) | [AI Providers](./00-ai-providers-landscape.md) | [Google](./03-google.md) |

