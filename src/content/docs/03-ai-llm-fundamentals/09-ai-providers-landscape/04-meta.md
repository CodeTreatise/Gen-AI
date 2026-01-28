---
title: "Meta (Llama)"
---

# Meta (Llama)

## Introduction

Meta's Llama models are the leading open-weights models, enabling self-hosting, customization, and avoiding vendor lock-in. The Llama family has become the foundation of the open-source AI ecosystem.

### What We'll Cover

- Llama model family
- Self-hosting options
- Commercial license
- Community ecosystem

---

## Model Lineup

### Current Models (2025-2026)

| Model | Parameters | Context | Notes |
|-------|------------|---------|-------|
| Llama 3.3 70B | 70B | 128K | Latest 70B |
| Llama 3.1 405B | 405B | 128K | Largest open |
| Llama 3.1 70B | 70B | 128K | Popular choice |
| Llama 3.1 8B | 8B | 128K | Fast, efficient |
| Llama 4 Scout | ~100B | 256K+ | Latest flagship |
| Llama 4 Maverick | ~400B | 256K+ | Maximum capability |

### Model Selection

```python
def select_llama_model(requirements: dict) -> str:
    """Select appropriate Llama model"""
    
    if requirements.get("max_quality"):
        return "llama-4-maverick"  # or 3.1-405b
    
    if requirements.get("on_device"):
        return "llama-3.1-8b"  # Fits on consumer GPU
    
    if requirements.get("balanced"):
        return "llama-3.3-70b"  # Best quality/speed
    
    if requirements.get("context_tokens", 0) > 128000:
        return "llama-4-scout"  # Larger context
    
    return "llama-3.1-70b"
```

---

## Self-Hosting Options

### With vLLM

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,  # Number of GPUs
    gpu_memory_utilization=0.9
)

# Generate
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9
)

def generate(prompts: list) -> list:
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# Usage
responses = generate(["Hello, how are you?"])
```

### With Ollama

```python
import ollama

# Pull model (first time)
# ollama pull llama3.3

# Generate
response = ollama.generate(
    model="llama3.3",
    prompt="Write a haiku about coding"
)
print(response['response'])

# Chat
response = ollama.chat(
    model="llama3.3",
    messages=[{
        "role": "user",
        "content": "Hello!"
    }]
)
print(response['message']['content'])
```

### With Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.7
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Via API Providers

### Together AI

```python
from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Groq

```python
from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Fireworks

```python
import fireworks.client as fireworks

fireworks.client.api_key = "YOUR_KEY"

response = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

## Commercial License

### Llama License Terms

```python
llama_license = {
    "type": "Custom Meta license",
    "commercial_use": True,
    "requirements": [
        "Accept license agreement",
        "Don't use outputs to train competing models",
        "Include attribution in products",
        "Follow acceptable use policy"
    ],
    "restrictions": [
        "Monthly active users > 700M require special license",
        "Cannot use for illegal activities",
        "Cannot remove safety features"
    ]
}
```

### Getting Access

```python
access_steps = [
    "1. Go to huggingface.co/meta-llama",
    "2. Accept Meta's license agreement",
    "3. Get approval (usually instant)",
    "4. Download weights or use via API"
]
```

---

## Hardware Requirements

### GPU Memory Needs

| Model | FP16 VRAM | INT8 VRAM | INT4 VRAM |
|-------|-----------|-----------|-----------|
| 8B | 16GB | 8GB | 5GB |
| 70B | 140GB | 70GB | 35GB |
| 405B | 810GB | 405GB | 200GB |

### Practical Configurations

```python
deployment_configs = {
    "llama-8b": {
        "single_gpu": "RTX 4090 (24GB)",
        "cloud": "1x A100 40GB"
    },
    "llama-70b": {
        "consumer": "4x RTX 4090 (96GB total)",
        "cloud": "4x A100 80GB or 8x A100 40GB"
    },
    "llama-405b": {
        "cloud_only": "8x H100 80GB or 16x A100 80GB"
    }
}
```

---

## Community Ecosystem

### Fine-Tuned Variants

```python
popular_llama_finetunes = {
    "code": [
        "CodeLlama",
        "Phind-CodeLlama",
        "WizardCoder"
    ],
    "chat": [
        "Vicuna",
        "Nous-Hermes",
        "OpenChat"
    ],
    "reasoning": [
        "WizardLM",
        "Orca"
    ],
    "small": [
        "TinyLlama",
        "Llama-3.2-1B"
    ]
}
```

### Tools and Frameworks

```python
llama_ecosystem = {
    "inference": ["vLLM", "TGI", "llama.cpp", "Ollama"],
    "fine_tuning": ["LoRA", "QLoRA", "PEFT", "Axolotl"],
    "quantization": ["GPTQ", "AWQ", "GGUF", "bitsandbytes"],
    "deployment": ["TensorRT-LLM", "ONNX", "OpenVINO"]
}
```

---

## Quantization

### Running on Consumer Hardware

```python
# Using llama.cpp for GGUF quantized models
# Much smaller memory footprint

# In terminal:
# ./main -m llama-3.3-70b-instruct.Q4_K_M.gguf -p "Hello!" -n 256

# Quantization levels:
quantization_options = {
    "Q8_0": "8-bit, minimal quality loss",
    "Q6_K": "6-bit, good balance",
    "Q4_K_M": "4-bit medium, popular choice",
    "Q4_K_S": "4-bit small, more compression",
    "Q3_K_M": "3-bit, significant compression"
}
```

---

## Summary

✅ **Open weights**: Full control, no API dependency

✅ **Strong models**: Competitive with proprietary options

✅ **Flexible hosting**: Self-host or use API providers

✅ **Rich ecosystem**: Tools, fine-tunes, community

✅ **Commercial friendly**: Permissive license for most uses

**Next:** [Mistral AI](./05-mistral.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Google](./03-google.md) | [AI Providers](./00-ai-providers-landscape.md) | [Mistral AI](./05-mistral.md) |

