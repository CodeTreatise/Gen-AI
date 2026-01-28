---
title: "Alibaba Qwen"
---

# Alibaba Qwen

## Introduction

Alibaba's Qwen (通义千问 - Tongyi Qianwen) is one of the most powerful open-source model families available, with exceptional multilingual support—particularly excelling in Chinese and English. The Qwen2.5 series competes directly with GPT-4 and Claude while being fully open-weight under the Apache 2.0 license.

### What We'll Cover

- Qwen model family and variants
- Multilingual and coding capabilities
- API access options (DashScope, third-party)
- Self-hosting with Ollama, vLLM, and Transformers
- QwQ reasoning model
- Best practices and use cases

### Prerequisites

- Python 3.8+
- For self-hosting: GPU with 16GB+ VRAM (for smaller models)
- API access via Alibaba Cloud DashScope or third-party providers

---

## Model Lineup

### Qwen2.5 Family (Current)

```
┌─────────────────────────────────────────────────────────────────┐
│                    QWEN MODEL FAMILY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base Models (Qwen2.5)                                          │
│  ├── Qwen2.5-72B-Instruct  ──── Flagship, best quality         │
│  ├── Qwen2.5-32B-Instruct  ──── Balanced size/performance      │
│  ├── Qwen2.5-14B-Instruct  ──── Good for mid-range GPUs        │
│  ├── Qwen2.5-7B-Instruct   ──── Popular for self-hosting       │
│  ├── Qwen2.5-3B-Instruct   ──── Edge devices                   │
│  ├── Qwen2.5-1.5B-Instruct ──── Mobile/embedded                │
│  └── Qwen2.5-0.5B-Instruct ──── Smallest, fastest              │
│                                                                  │
│  Specialized Models                                              │
│  ├── Qwen2.5-Coder-32B     ──── Code generation specialist     │
│  ├── Qwen2.5-Math-72B      ──── Mathematical reasoning         │
│  └── QwQ-32B-Preview       ──── Chain-of-thought reasoning     │
│                                                                  │
│  Multimodal                                                      │
│  └── Qwen2-VL-72B          ──── Vision-language model          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Model | Parameters | Context | VRAM (FP16) | Best For |
|-------|------------|---------|-------------|----------|
| Qwen2.5-72B | 72B | 128K | 144GB | Maximum quality |
| Qwen2.5-32B | 32B | 128K | 64GB | High quality, easier hosting |
| Qwen2.5-14B | 14B | 128K | 28GB | Single A100/H100 |
| Qwen2.5-7B | 7B | 128K | 14GB | Consumer GPUs (RTX 4090) |
| Qwen2.5-3B | 3B | 32K | 6GB | RTX 3060/4060 |

### Qwen2.5 Key Features

```python
qwen25_features = {
    "parameters": "0.5B to 72B range",
    "context": "Up to 128K tokens",
    "languages": "29+ languages supported",
    "code_languages": "92+ programming languages",
    "math": "Strong mathematical reasoning",
    "license": "Apache 2.0 (fully open, commercial OK)",
    "training_data": "18 trillion tokens"
}
```

---

## API Usage

### Via Third-Party Providers (Easiest)

```python
from together import Together

client = Together()

def qwen_chat(prompt: str, model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo") -> str:
    """Use Qwen via Together AI"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Also available on: Fireworks, Groq, Replicate, etc.
```

### Via Alibaba Cloud DashScope

```python
from openai import OpenAI

# Alibaba's DashScope API (OpenAI-compatible)
client = OpenAI(
    api_key="YOUR_DASHSCOPE_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def qwen_alibaba(prompt: str, model: str = "qwen-max") -> str:
    """Direct Alibaba Cloud access"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Models available:
# qwen-max: Best quality
# qwen-plus: Balanced
# qwen-turbo: Fast
```

### Streaming

```python
def qwen_stream(prompt: str):
    """Stream responses in real-time"""
    stream = client.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
```

---

## Multilingual Strength

### Language Capabilities

```python
qwen_languages = {
    "top_tier": ["Chinese", "English"],
    "excellent": ["Japanese", "Korean", "Vietnamese", "Thai", "Indonesian"],
    "strong": ["French", "German", "Spanish", "Portuguese", "Italian", "Russian"],
    "supported": "29+ languages total",
    "unique": "Best Chinese capability of any open-source model"
}
```

### Multilingual Example

```python
def translate_and_respond(text: str, target_lang: str) -> str:
    """Qwen excels at multilingual tasks"""
    
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[{
            "role": "user",
            "content": f"Translate this to {target_lang} and provide a thoughtful response: {text}"
        }]
    )
    return response.choices[0].message.content

# Example: Chinese business communication
result = translate_and_respond(
    "Can we schedule a meeting next week to discuss the partnership?",
    "Chinese"
)
```

---

## Self-Hosting

### With Ollama (Easiest)

```bash
# Pull and run Qwen models
ollama pull qwen2.5:7b
ollama run qwen2.5:7b "Hello!"

# Larger models
ollama pull qwen2.5:32b
ollama pull qwen2.5:72b  # Requires significant VRAM

# Coder variant
ollama pull qwen2.5-coder:32b
```

### With vLLM (Production)

```python
from vllm import LLM, SamplingParams

# High-performance serving
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,  # Distribute across 4 GPUs
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2000
)

outputs = llm.generate(["Explain machine learning"], sampling_params)
print(outputs[0].outputs[0].text)
```

### With Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [{"role": "user", "content": "Write a Python function to calculate Fibonacci numbers"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Quantized Models for Consumer Hardware

```bash
# 4-bit quantization for lower VRAM
ollama pull qwen2.5:7b-instruct-q4_0  # ~4GB VRAM
ollama pull qwen2.5:14b-instruct-q4_0 # ~8GB VRAM
ollama pull qwen2.5:32b-instruct-q4_0 # ~18GB VRAM
```

---

## Qwen Coder

### Code Specialization

```python
def qwen_code(task: str) -> str:
    """Use Qwen2.5-Coder for programming tasks"""
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are an expert programmer. Write clean, efficient, well-documented code."
            },
            {"role": "user", "content": task}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# Supports 92+ programming languages including:
# Python, JavaScript, TypeScript, Java, C++, Rust, Go, Ruby, PHP,
# Swift, Kotlin, Scala, R, Julia, Lua, Bash, SQL, and many more
```

### Code Review and Explanation

```python
def explain_code(code: str, language: str = "python") -> str:
    """Get detailed code explanation"""
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{
            "role": "user",
            "content": f"""Explain this {language} code in detail:
            
```{language}
{code}
```

Include:
1. What the code does
2. How it works step by step
3. Time and space complexity
4. Potential improvements"""
        }]
    )
    return response.choices[0].message.content
```

---

## QwQ: Reasoning Model

### Chain-of-Thought Reasoning

```python
def qwq_reason(problem: str) -> str:
    """Use QwQ for complex reasoning tasks
    
    QwQ shows its thinking process, similar to OpenAI o1 or DeepSeek R1.
    """
    
    response = client.chat.completions.create(
        model="Qwen/QwQ-32B-Preview",
        messages=[{"role": "user", "content": problem}]
    )
    
    # QwQ outputs detailed reasoning followed by answer
    return response.choices[0].message.content

# Best for:
# - Mathematical proofs
# - Logic puzzles
# - Complex word problems
# - Multi-step reasoning
```

### Example Use Case

```python
result = qwq_reason("""
Three friends split a dinner bill. Alice paid 40% of the total.
Bob paid $12 more than Charlie. Charlie paid half of what Alice paid.
What was the total bill?
""")
# QwQ will show step-by-step mathematical reasoning
```

---

## Pricing

### Via Third-Party Providers

| Provider | Qwen2.5-72B Input | Qwen2.5-72B Output |
|----------|-------------------|-------------------|
| Together AI | $0.90/M | $0.90/M |
| Fireworks | $0.90/M | $0.90/M |
| Groq | $0.59/M | $0.79/M |

### Via Alibaba Cloud DashScope

| Model | Input (per 1M) | Output (per 1M) |
|-------|----------------|-----------------|
| qwen-max | ¥0.02 (~$0.003) | ¥0.06 (~$0.008) |
| qwen-plus | ¥0.004 | ¥0.012 |
| qwen-turbo | ¥0.002 | ¥0.006 |

> **Note:** DashScope offers extremely low pricing, but requires Alibaba Cloud account.

---

## When to Choose Qwen

```python
choose_qwen_when = [
    "Need Chinese language support",
    "Want fully open-source (Apache 2.0)",
    "Self-hosting is a priority",
    "Need range of model sizes",
    "Building multilingual applications",
    "Want strong coding capabilities (Coder)",
    "Need reasoning (QwQ)"
]

consider_alternatives_when = [
    "Need maximum English quality (GPT-4, Claude)",
    "Require vision (limited Qwen-VL availability)",
    "Want managed API with strong SLAs"
]
```

---

## Summary

✅ **Fully open**: Apache 2.0 license, commercial use OK

✅ **Size range**: 0.5B to 72B for any use case

✅ **Multilingual**: 29+ languages, best-in-class Chinese

✅ **Coding**: Qwen2.5-Coder for 92+ languages

✅ **Reasoning**: QwQ for chain-of-thought

✅ **128K context**: Long document support

✅ **Easy hosting**: Works with Ollama, vLLM, Transformers

**Next:** [Other Providers](./14-other-providers.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [xAI](./12-xai.md) | [AI Providers](./00-ai-providers-landscape.md) | [Other Providers](./14-other-providers.md) |

