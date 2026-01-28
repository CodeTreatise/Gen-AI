---
title: "Cloud AI Services"
---

# Cloud AI Services

## Introduction

Major cloud providers offer AI services integrated with their infrastructure. AWS Bedrock, Azure OpenAI, and Google Vertex AI provide enterprise features, compliance, and unified billing.

### What We'll Cover

- AWS Bedrock
- Azure OpenAI Service
- Google Vertex AI
- Choosing between providers

---

## AWS Bedrock

### Multi-Model Access

AWS Bedrock provides access to multiple foundation models through a unified API.

```python
import boto3
import json

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def bedrock_claude(prompt: str) -> str:
    """Use Claude via Bedrock"""
    
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        })
    )
    
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def bedrock_llama(prompt: str) -> str:
    """Use Llama via Bedrock"""
    
    response = bedrock.invoke_model(
        modelId='meta.llama3-1-70b-instruct-v1:0',
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 512
        })
    )
    
    result = json.loads(response['body'].read())
    return result['generation']
```

### Available Models

```python
bedrock_models = {
    "anthropic": ["Claude 3.5 Sonnet", "Claude 3 Haiku", "Claude 3 Opus"],
    "meta": ["Llama 3.1 8B", "Llama 3.1 70B", "Llama 3.1 405B"],
    "mistral": ["Mistral 7B", "Mixtral 8x7B", "Mistral Large"],
    "amazon": ["Titan Text", "Titan Embeddings"],
    "cohere": ["Command R", "Command R+", "Embed"]
}
```

### Bedrock Features

```python
bedrock_features = {
    "knowledge_bases": "Built-in RAG",
    "agents": "Agentic workflows",
    "guardrails": "Content filtering",
    "fine_tuning": "Custom model training",
    "batch_inference": "Async processing",
    "model_evaluation": "Compare models"
}
```

---

## Azure OpenAI

### Enterprise OpenAI Access

Azure OpenAI provides OpenAI models with enterprise features and Azure integration.

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="YOUR_AZURE_KEY",
    api_version="2024-02-15-preview",
    azure_endpoint="https://your-resource.openai.azure.com"
)

def azure_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",  # Your deployment name
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Key Differences from OpenAI

```python
azure_openai_differences = {
    "deployment_required": "Must deploy models before use",
    "regional": "Choose deployment region",
    "quota": "Request quota for each model",
    "private_endpoints": "VNet integration",
    "managed_identity": "Azure AD authentication",
    "content_filtering": "Built-in, configurable"
}
```

### Available Models

```python
azure_openai_models = [
    "GPT-4o",
    "GPT-4o-mini", 
    "GPT-4 Turbo",
    "GPT-3.5 Turbo",
    "DALL-E 3",
    "Whisper",
    "Text Embeddings"
]
```

### Enterprise Features

```python
azure_enterprise = {
    "compliance": "SOC 2, HIPAA, FedRAMP, GDPR",
    "private_link": "No public internet exposure",
    "managed_identity": "Passwordless auth",
    "rbac": "Fine-grained access control",
    "monitoring": "Azure Monitor integration",
    "sla": "99.9% uptime"
}
```

---

## Google Vertex AI

### Unified AI Platform

Vertex AI combines Google's AI services, including Gemini models, under one platform.

```python
from vertexai.generative_models import GenerativeModel
import vertexai

# Initialize
vertexai.init(project="your-project", location="us-central1")

# Create model
model = GenerativeModel("gemini-2.0-flash")

def vertex_chat(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text
```

### Available Models

```python
vertex_models = {
    "gemini": ["Gemini 2.0 Pro", "Gemini 2.0 Flash", "Gemini 1.5 Pro"],
    "claude": ["Claude 3.5 Sonnet", "Claude 3 Haiku"],  # Via Model Garden
    "llama": ["Llama 3.1 405B", "Llama 3.1 70B"],  # Via Model Garden
    "palm": ["PaLM 2"],  # Legacy
    "imagen": ["Imagen 2", "Imagen 3"],  # Image generation
}
```

### Vertex Features

```python
vertex_features = {
    "model_garden": "Access to many models",
    "grounding": "Ground responses in Google Search",
    "tuning": "Fine-tune Gemini models",
    "evaluation": "Model comparison tools",
    "mlops": "Full ML pipeline support",
    "reasoning_engine": "Agent frameworks"
}
```

---

## Comparison

| Feature | AWS Bedrock | Azure OpenAI | Vertex AI |
|---------|-------------|--------------|-----------|
| Model Variety | ★★★ | ★★ | ★★★ |
| OpenAI Models | ❌ | ✅ | ❌ |
| Gemini Models | ❌ | ❌ | ✅ |
| Claude Models | ✅ | ❌ | ✅ |
| Llama Models | ✅ | ❌ | ✅ |
| Built-in RAG | ✅ | ✅ | ✅ |
| Global Regions | ★★★ | ★★★ | ★★ |

---

## When to Choose Each

```python
cloud_provider_selection = {
    "aws_bedrock": [
        "Already on AWS",
        "Want model variety",
        "Need Claude + Llama",
        "Want Knowledge Bases"
    ],
    "azure_openai": [
        "Already on Azure",
        "Need OpenAI models specifically",
        "Enterprise Microsoft ecosystem",
        "Government/regulated industries"
    ],
    "vertex_ai": [
        "Already on GCP",
        "Want Gemini models",
        "Need massive context (2M tokens)",
        "Google Search grounding"
    ]
}
```

---

## Summary

✅ **AWS Bedrock**: Best model variety, knowledge bases

✅ **Azure OpenAI**: Enterprise OpenAI with compliance

✅ **Vertex AI**: Gemini access, Google ecosystem

✅ **All three**: Enterprise-ready, compliance, unified billing

---

## Lesson Complete

You've now explored the entire AI provider landscape! Key takeaways:

| Category | Top Choices |
|----------|-------------|
| Best Quality | OpenAI GPT-4o, Claude 3.5 Sonnet |
| Fastest | Groq |
| Best Value | DeepSeek V3 |
| Best Open | Meta Llama 3.3 70B, Qwen 2.5 |
| Best Local | Ollama, LM Studio |
| Enterprise | Azure OpenAI, AWS Bedrock |

---

## Navigation

| Previous | Up | Next Lesson |
|----------|-------|-------------|
| [Specialized Providers](./16-specialized-providers.md) | [AI Providers](./00-ai-providers-landscape.md) | [Reasoning Models](../10-reasoning-thinking-models/00-reasoning-thinking-models.md) |

