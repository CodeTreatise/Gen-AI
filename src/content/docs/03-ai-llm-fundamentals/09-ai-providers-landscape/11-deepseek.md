---
title: "DeepSeek"
---

# DeepSeek

## Introduction

DeepSeek is a Chinese AI research company that has disrupted the industry with remarkably capable open-source models at a fraction of competitors' costs. DeepSeek V3 and R1 offer GPT-4/Claude-level performance while being 10x cheaper to use. Their models are fully open-weight with permissive licenses, enabling self-hosting.

### What We'll Cover

- DeepSeek V3 and R1 model capabilities
- API usage with OpenAI-compatible interface
- Reasoning mode (chain-of-thought)
- Code generation excellence
- Self-hosting options
- Cost analysis and pricing

### Prerequisites

- Python 3.8+
- DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com/)
- Basic understanding of LLM APIs

---

## Model Lineup

### Current Models (2025-2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                  DEEPSEEK MODEL FAMILY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DeepSeek V3.2 (Latest)                                         │
│  ├── Architecture: MoE (671B total, 37B active)                │
│  ├── Context: 128K tokens                                       │
│  ├── Training: 14.8 trillion tokens                            │
│  ├── API Names:                                                 │
│  │   ├── deepseek-chat (non-thinking mode)                     │
│  │   └── deepseek-reasoner (thinking mode)                     │
│  └── Performance: GPT-4o / Claude 3.5 Sonnet level             │
│                                                                  │
│  DeepSeek R1                                                     │
│  ├── Reasoning-focused model                                    │
│  ├── Shows chain-of-thought process                            │
│  └── Excels at math, coding, complex logic                     │
│                                                                  │
│  DeepSeek Coder V2                                              │
│  ├── 236B MoE, specialized for code                            │
│  ├── 128K context                                               │
│  └── Top-tier code generation                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Model | Parameters | Active | Context | Specialty |
|-------|------------|--------|---------|-----------|
| DeepSeek V3.2 | 671B MoE | 37B | 128K | General/Code |
| DeepSeek R1 | 671B MoE | 37B | 128K | Reasoning |
| DeepSeek Coder V2 | 236B MoE | 21B | 128K | Code |
| DeepSeek R1 Distill | 7B-70B | 7B-70B | 128K | Efficient reasoning |

### Key Innovation: MoE Architecture

```python
moe_explanation = {
    "total_params": "671 billion parameters",
    "active_params": "Only 37B active per token",
    "benefit": "Quality of 671B, cost of 37B",
    "efficiency": "10x more efficient than dense models"
}
```

---

## API Usage

### Setup

```python
from openai import OpenAI

# DeepSeek uses OpenAI-compatible API
client = OpenAI(
    api_key="YOUR_DEEPSEEK_KEY",
    base_url="https://api.deepseek.com"  # Note: /v1 optional
)
```

### Basic Chat (Non-Thinking Mode)

```python
def deepseek_chat(prompt: str) -> str:
    """Standard chat completion"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek V3.2 non-thinking
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

result = deepseek_chat("Explain how transformers work in neural networks")
print(result)
```

### Reasoning Mode (R1 / Thinking)

```python
def deepseek_reason(problem: str) -> tuple[str, str]:
    """Use reasoning mode for complex problems
    
    Returns both the thinking process and final answer.
    """
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",  # V3.2 thinking mode
        messages=[{"role": "user", "content": problem}]
    )
    
    message = response.choices[0].message
    
    # R1 includes reasoning_content (thinking) and content (answer)
    thinking = getattr(message, 'reasoning_content', '')
    answer = message.content
    
    return thinking, answer

# Example: Complex math problem
thinking, answer = deepseek_reason("""
A ball is thrown upward with initial velocity 20 m/s from a height of 5 meters.
When does it hit the ground? (g = 10 m/s²)
""")

print("Reasoning:", thinking[:500], "...")
print("\nFinal Answer:", answer)
```

### Streaming

```python
def deepseek_stream(prompt: str):
    """Stream responses in real-time"""
    
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
```

---

## Code Generation

### Expert-Level Coding

```python
def deepseek_code(task: str, language: str = "python") -> str:
    """Generate high-quality code"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert {language} programmer. 
Write clean, efficient, well-documented code.
Include error handling and follow best practices."""
            },
            {"role": "user", "content": task}
        ],
        temperature=0.2  # Lower for more precise code
    )
    return response.choices[0].message.content

# Example: Generate a complete module
code = deepseek_code("""
Create a Python class for rate limiting API calls.
Features:
- Token bucket algorithm
- Configurable rate and capacity  
- Thread-safe
- Async support
""")
```

### Code Review

```python
def code_review(code: str) -> str:
    """Get comprehensive code review"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"""Review this code for:
1. Bugs and potential issues
2. Performance problems
3. Security vulnerabilities
4. Best practice violations
5. Suggestions for improvement

```python
{code}
```"""
        }]
    )
    return response.choices[0].message.content
```

---

## Self-Hosting

### With Ollama (Easiest)

```bash
# Pull and run DeepSeek models
ollama pull deepseek-v3
ollama run deepseek-v3 "Hello, how are you?"

# Or the coder variant
ollama pull deepseek-coder:33b
ollama run deepseek-coder:33b "Write a binary search in Rust"
```

### With vLLM (Production)

```python
from vllm import LLM, SamplingParams

# Requires significant GPU memory (8x A100 80GB recommended)
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2000
)

outputs = llm.generate(["Explain quantum computing"], sampling_params)
print(outputs[0].outputs[0].text)
```

### With Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# For smaller distilled models
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [{"role": "user", "content": "Solve: 2x + 5 = 17"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Pricing

### API Pricing (Per Million Tokens)

| Model | Input | Output |
|-------|-------|--------|
| DeepSeek V3.2 Chat | $0.27 | $1.10 |
| DeepSeek Reasoner (R1) | $0.55 | $2.19 |
| DeepSeek Coder | ~$0.14 | ~$0.28 |

### Cost Comparison (Output per $1)

```python
cost_comparison = {
    "deepseek_chat": "~900K tokens",
    "gpt_4o": "~100K tokens",
    "claude_35_sonnet": "~66K tokens",
    
    "savings": "DeepSeek is 9-14x cheaper for similar quality"
}
```

### Why So Cheap?

```python
deepseek_efficiency = {
    "moe_architecture": "Only 37B of 671B params active",
    "training_cost": "$5.5M vs $100M+ for competitors",
    "chinese_compute": "Lower infrastructure costs",
    "business_model": "Open source, API subsidized"
}
```

---

## Strengths and Considerations

### Strengths

```python
deepseek_strengths = [
    "GPT-4 level quality at 10% of the cost",
    "Exceptional code generation and review",
    "Strong mathematical reasoning (R1)",
    "Fully open weights - self-host anywhere",
    "128K context window",
    "MoE efficiency - fast inference",
    "Apache 2.0 license - commercial use OK"
]
```

### Considerations

```python
deepseek_considerations = [
    "Chinese company - potential data concerns for some",
    "May have content restrictions on sensitive topics",
    "Less extensive documentation than OpenAI/Anthropic",
    "Smaller ecosystem of tools and integrations"
]
```

---

## When to Choose DeepSeek

```python
choose_deepseek_when = [
    "Cost is a primary concern",
    "Need excellent code generation",
    "Want open weights for self-hosting",
    "Building cost-sensitive applications",
    "Need strong reasoning (R1)"
]

consider_alternatives_when = [
    "Need maximum safety/alignment (Claude)",
    "Require enterprise support/SLAs",
    "Have data residency requirements (US/EU)",
    "Need multimodal (vision) - limited support"
]
```

---

## Summary

✅ **GPT-4 level quality**: Competitive with frontier models

✅ **10x cheaper**: Dramatically lower costs

✅ **Open weights**: Full self-hosting option

✅ **R1 reasoning**: Strong chain-of-thought capabilities

✅ **Code excellence**: Top-tier code generation and review

✅ **MoE efficiency**: Fast, efficient architecture

**Next:** [xAI (Grok)](./12-xai.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Replicate](./10-replicate.md) | [AI Providers](./00-ai-providers-landscape.md) | [xAI](./12-xai.md) |

