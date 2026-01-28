---
title: "Parameter Combinations by Use Case"
---

# Parameter Combinations by Use Case

## Introduction

Different tasks require different parameter combinations. This lesson provides ready-to-use configurations for common use cases, helping you quickly configure models for specific applications.

### What We'll Cover

- Recommended defaults
- Customer service settings
- Code generation settings
- Creative writing settings
- Analysis and extraction settings

---

## Recommended Defaults

When in doubt, start with these balanced defaults:

```python
default_config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 1000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None,
}

# This configuration:
# - Produces natural-sounding text
# - Allows some creativity
# - Avoids extreme behavior
# - Works for most general purposes
```

---

## Customer Service Configuration

For professional, helpful, and consistent responses:

```python
customer_service_config = {
    "temperature": 0.5,          # Consistent but not robotic
    "top_p": 0.9,                # Avoid unusual responses
    "max_tokens": 300,           # Concise responses
    "frequency_penalty": 0.3,    # Reduce word repetition
    "presence_penalty": 0.3,     # Encourage topic coverage
    "stop": ["\n\nCustomer:", "\n\nUser:"],  # Stop before next turn
}

def customer_service_response(query: str, context: str = None) -> str:
    """Generate customer service response"""
    
    system_prompt = """You are a helpful customer service representative. 
    Be professional, empathetic, and concise. 
    Always offer to help further if the issue isn't fully resolved."""
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if context:
        messages.append({
            "role": "system", 
            "content": f"Customer context: {context}"
        })
    
    messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        **customer_service_config
    )
    
    return response.choices[0].message.content
```

### Why These Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| temperature=0.5 | Moderate | Consistent but natural |
| max_tokens=300 | Limited | Concise support responses |
| frequency_penalty=0.3 | Slight | Avoid repetitive phrasing |
| presence_penalty=0.3 | Slight | Cover multiple aspects |

---

## Code Generation Configuration

For accurate, working code:

```python
code_generation_config = {
    "temperature": 0,            # Deterministic for code
    "top_p": 1.0,                # Not needed with temp=0
    "max_tokens": 2000,          # Allow complete implementations
    "frequency_penalty": 0.0,    # Code often repeats patterns
    "presence_penalty": 0.0,     # Variables should repeat
    "stop": ["```\n\n", "---"],  # Stop after code block
}

def generate_code(task: str, language: str = "python") -> str:
    """Generate code for a given task"""
    
    system_prompt = f"""You are an expert {language} programmer.
    Write clean, well-documented, production-ready code.
    Include error handling and follow best practices.
    Only output code, no explanations unless in comments."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ],
        **code_generation_config
    )
    
    return response.choices[0].message.content

# Usage
code = generate_code(
    "Write a function to validate email addresses using regex",
    language="python"
)
```

### Why These Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| temperature=0 | Zero | Exact, reproducible code |
| max_tokens=2000 | High | Complete implementations |
| frequency_penalty=0 | None | Code patterns repeat |
| presence_penalty=0 | None | Variables must repeat |

---

## Creative Writing Configuration

For stories, poetry, and creative content:

```python
creative_writing_config = {
    "temperature": 1.2,          # More creative choices
    "top_p": 0.95,               # Allow unusual but not crazy
    "max_tokens": 1500,          # Room for storytelling
    "frequency_penalty": 0.5,    # Varied vocabulary
    "presence_penalty": 0.5,     # Explore different themes
    "stop": None,                # Let creativity flow
}

def generate_creative_content(
    prompt: str, 
    style: str = "narrative"
) -> str:
    """Generate creative content"""
    
    style_prompts = {
        "narrative": "Write in an engaging narrative style with vivid descriptions.",
        "poetic": "Write with poetic language, using metaphors and imagery.",
        "humorous": "Write with wit and humor, keeping the tone light.",
        "dramatic": "Write with tension and emotional depth.",
    }
    
    system_prompt = f"""You are a creative writer. 
    {style_prompts.get(style, style_prompts['narrative'])}
    Focus on originality and engaging the reader."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        **creative_writing_config
    )
    
    return response.choices[0].message.content

# Usage
story = generate_creative_content(
    "Write a short story about a robot discovering emotions",
    style="dramatic"
)
```

### Why These Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| temperature=1.2 | High | Creative, unexpected choices |
| top_p=0.95 | Near full | Allow unusual words |
| frequency_penalty=0.5 | Moderate | Varied vocabulary |
| presence_penalty=0.5 | Moderate | Explore themes |

---

## Analysis and Extraction Configuration

For data extraction, summarization, and analysis:

```python
analysis_config = {
    "temperature": 0.2,          # Low for accuracy
    "top_p": 0.9,                # Focused vocabulary
    "max_tokens": 500,           # Depends on task
    "frequency_penalty": 0.0,    # Consistent terminology
    "presence_penalty": 0.0,     # May need to repeat facts
    "response_format": {"type": "json_object"},  # Structured output
}

def extract_entities(text: str) -> dict:
    """Extract named entities from text"""
    import json
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": "Extract named entities and return as JSON."
            },
            {
                "role": "user", 
                "content": f"Extract entities from: {text}\n\nReturn JSON with keys: people, places, organizations, dates"
            }
        ],
        **analysis_config
    )
    
    return json.loads(response.choices[0].message.content)

def summarize(text: str, max_sentences: int = 3) -> str:
    """Summarize text to key points"""
    
    summary_config = {**analysis_config}
    summary_config["response_format"] = None  # Plain text
    summary_config["max_tokens"] = 200
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"Summarize in exactly {max_sentences} sentences. Be concise and accurate."
            },
            {"role": "user", "content": text}
        ],
        **summary_config
    )
    
    return response.choices[0].message.content
```

---

## Configuration Factory

A unified approach to managing configurations:

```python
class LLMConfigFactory:
    """
    Factory for creating task-specific LLM configurations.
    """
    
    CONFIGS = {
        "default": {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1000,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "customer_service": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 300,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3,
        },
        "code": {
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 2000,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "creative": {
            "temperature": 1.2,
            "top_p": 0.95,
            "max_tokens": 1500,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        },
        "analysis": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 500,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "brainstorm": {
            "temperature": 1.5,
            "top_p": 0.95,
            "max_tokens": 1000,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.8,
        },
        "translation": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": None,  # Match input length
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "factual_qa": {
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 300,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    }
    
    @classmethod
    def get_config(cls, task_type: str, **overrides) -> dict:
        """
        Get configuration for a task type with optional overrides.
        """
        base = cls.CONFIGS.get(task_type, cls.CONFIGS["default"]).copy()
        base.update(overrides)
        return base
    
    @classmethod
    def list_available(cls) -> list:
        """List available configuration types"""
        return list(cls.CONFIGS.keys())
    
    @classmethod
    def describe(cls, task_type: str) -> str:
        """Describe a configuration's settings and rationale"""
        config = cls.CONFIGS.get(task_type)
        if not config:
            return f"Unknown task type: {task_type}"
        
        descriptions = {
            "default": "Balanced settings for general use",
            "customer_service": "Professional, consistent support responses",
            "code": "Deterministic, accurate code generation",
            "creative": "High creativity with vocabulary variety",
            "analysis": "Accurate extraction and analysis",
            "brainstorm": "Maximum divergent thinking",
            "translation": "Accurate, fluent translations",
            "factual_qa": "Precise factual answers",
        }
        
        return f"{task_type}: {descriptions.get(task_type, 'Custom configuration')}\n{config}"

# Usage
config = LLMConfigFactory.get_config("code", max_tokens=3000)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    **config
)
```

---

## Quick Reference Table

| Use Case | Temp | Top-P | Max Tokens | Freq Pen | Pres Pen |
|----------|------|-------|------------|----------|----------|
| **Default** | 0.7 | 1.0 | 1000 | 0.0 | 0.0 |
| **Customer Service** | 0.5 | 0.9 | 300 | 0.3 | 0.3 |
| **Code Generation** | 0 | 1.0 | 2000 | 0.0 | 0.0 |
| **Creative Writing** | 1.2 | 0.95 | 1500 | 0.5 | 0.5 |
| **Analysis/Extraction** | 0.2 | 0.9 | 500 | 0.0 | 0.0 |
| **Brainstorming** | 1.5 | 0.95 | 1000 | 0.3 | 0.8 |
| **Translation** | 0.3 | 0.9 | auto | 0.0 | 0.0 |
| **Factual Q&A** | 0 | 1.0 | 300 | 0.0 | 0.0 |

---

## Hands-on Exercise

### Your Task

Create a multi-purpose chatbot that adjusts parameters based on detected intent:

```python
from openai import OpenAI

client = OpenAI()

def detect_intent(message: str) -> str:
    """Detect user intent to choose configuration"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["code", "function", "program", "script"]):
        return "code"
    elif any(word in message_lower for word in ["story", "poem", "creative", "write"]):
        return "creative"
    elif any(word in message_lower for word in ["summarize", "extract", "analyze"]):
        return "analysis"
    elif any(word in message_lower for word in ["ideas", "brainstorm", "suggest"]):
        return "brainstorm"
    elif "?" in message and len(message.split()) < 15:
        return "factual_qa"
    else:
        return "default"

def smart_chat(message: str) -> dict:
    """Chat with automatically selected configuration"""
    
    intent = detect_intent(message)
    config = LLMConfigFactory.get_config(intent)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        **config
    )
    
    return {
        "intent": intent,
        "config_used": config,
        "response": response.choices[0].message.content,
        "tokens_used": response.usage.completion_tokens
    }

# Test different intents
test_messages = [
    "Write a Python function to reverse a string",
    "Tell me a short story about a magical forest",
    "What is the capital of Japan?",
    "Give me 10 ideas for a mobile app",
    "Summarize the key points of machine learning",
]

for message in test_messages:
    result = smart_chat(message)
    print(f"\nMessage: {message}")
    print(f"Detected intent: {result['intent']}")
    print(f"Temperature used: {result['config_used']['temperature']}")
    print(f"Response preview: {result['response'][:100]}...")
```

---

## Summary

✅ **Match parameters to use case** for optimal results

✅ **Code generation**: temperature=0, no penalties

✅ **Creative writing**: temperature>1, moderate penalties

✅ **Analysis**: low temperature, structured output

✅ **Customer service**: balanced settings with slight penalties

✅ **Use a config factory** for consistent, maintainable settings

**Next Lesson:** [Streaming and Response Modes](../06-streaming-response-modes/00-streaming-response-modes.md)

---

## Further Reading

- [OpenAI Best Practices](https://platform.openai.com/docs/guides/text-generation/best-practices) — Official recommendations
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Parameter tuning tips

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Advanced Parameters](./07-advanced-parameters.md) | [Model Parameters](./00-model-parameters-settings.md) | [Streaming Response Modes](../06-streaming-response-modes/00-streaming-response-modes.md) |

<!-- 
Sources Consulted:
- OpenAI API Reference: https://platform.openai.com/docs/api-reference/chat/create
- OpenAI Best Practices: https://platform.openai.com/docs/guides/text-generation
-->

