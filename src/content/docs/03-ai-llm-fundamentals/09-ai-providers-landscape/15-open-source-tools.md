---
title: "Open Source Tools"
---

# Open Source Tools

## Introduction

Open-source tools enable running LLMs locally without API costs or data privacy concerns. This lesson covers Hugging Face, Ollama, LM Studio, and related tools.

### What We'll Cover

- Hugging Face ecosystem
- Ollama for local deployment
- LM Studio GUI
- llama.cpp and friends

---

## Hugging Face

### The AI Hub

Hugging Face is the central hub for open-source AI models, datasets, and tools.

```python
huggingface_offerings = {
    "hub": "Host for 500K+ models",
    "transformers": "Python library for running models",
    "inference_api": "Hosted inference endpoints",
    "spaces": "Demo hosting",
    "datasets": "Dataset repository"
}
```

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
    
    outputs = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Inference Endpoints

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="YOUR_HF_TOKEN")

def hf_inference(prompt: str) -> str:
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content
```

---

## Ollama

### Simple Local LLMs

Ollama makes running LLMs locally as easy as Docker containers.

### Installation and Usage

```bash
# Install (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.3

# Run interactively
ollama run llama3.3

# List models
ollama list

# Remove model
ollama rm llama3.3
```

### Python Integration

```python
import ollama

# Simple generation
response = ollama.generate(
    model="llama3.3",
    prompt="Write a haiku about coding"
)
print(response['response'])

# Chat interface
response = ollama.chat(
    model="llama3.3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model="llama3.3",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### OpenAI-Compatible API

```python
from openai import OpenAI

# Ollama exposes OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not actually used
)

response = client.chat.completions.create(
    model="llama3.3",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Available Models

```python
popular_ollama_models = [
    "llama3.3",           # Latest Llama
    "mistral",            # Mistral 7B
    "mixtral",            # Mixtral 8x7B
    "codellama",          # Code-specialized
    "qwen2.5",            # Alibaba's model
    "deepseek-coder",     # Code generation
    "phi3",               # Microsoft's small model
    "gemma2",             # Google's open model
]
```

---

## LM Studio

### GUI for Local LLMs

LM Studio provides a user-friendly desktop application for running LLMs.

```python
lm_studio_features = {
    "gui": "Visual interface for model management",
    "model_search": "Browse and download from HuggingFace",
    "chat_interface": "Built-in chat UI",
    "local_server": "OpenAI-compatible API",
    "quantization": "Automatic GGUF quantization"
}
```

### API Integration

```python
from openai import OpenAI

# LM Studio local server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # Not actually used
)

response = client.chat.completions.create(
    model="local-model",  # Whatever is loaded
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## llama.cpp

### Efficient CPU/GPU Inference

llama.cpp is a C++ implementation for efficient LLM inference.

```python
llama_cpp_features = {
    "efficiency": "Optimized for CPU and GPU",
    "quantization": "GGUF format, 2-8 bit",
    "platforms": "Mac, Linux, Windows",
    "memory": "Runs 70B on 32GB RAM (quantized)",
    "integrations": "Ollama, LM Studio use this"
}
```

### Python Bindings

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35  # Offload to GPU
)

output = llm(
    "Write a haiku:",
    max_tokens=50,
    temperature=0.7
)
print(output['choices'][0]['text'])
```

---

## vLLM

### Production Serving

vLLM is a high-performance inference engine for serving LLMs.

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256
)

# Generate
outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Server Mode

```bash
# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

---

## Comparison

| Tool | Use Case | Ease | Performance |
|------|----------|------|-------------|
| Ollama | Local dev | ★★★ | ★★ |
| LM Studio | Desktop GUI | ★★★ | ★★ |
| llama.cpp | Edge/CPU | ★★ | ★★★ |
| vLLM | Production | ★ | ★★★ |
| Transformers | Research | ★★ | ★★ |

---

## Summary

✅ **Hugging Face**: Central hub for models and tools

✅ **Ollama**: Easiest local deployment

✅ **LM Studio**: Best GUI experience

✅ **llama.cpp**: Most efficient for edge

✅ **vLLM**: Production serving

**Next:** [Specialized Providers](./16-specialized-providers.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Other Providers](./14-other-providers.md) | [AI Providers](./00-ai-providers-landscape.md) | [Specialized Providers](./16-specialized-providers.md) |

