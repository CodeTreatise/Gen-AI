---
title: "AI Providers Landscape"
---

# AI Providers Landscape

## Overview

The AI provider ecosystem has grown dramatically, with options ranging from frontier labs to specialized inference providers. Understanding the landscape helps you choose the right provider for your needs.

### What You'll Learn

| Provider Category | Topics Covered |
|-------------------|----------------|
| [OpenAI](./01-openai.md) | GPT-4o, o-series, GPT-5, API features |
| [Anthropic](./02-anthropic.md) | Claude 3.5/4, constitutional AI, computer use |
| [Google](./03-google.md) | Gemini 2.5, Vertex AI, long context |
| [Meta](./04-meta.md) | Llama 3/4, open source, self-hosting |
| [Mistral AI](./05-mistral.md) | European AI, MoE architecture |
| [Cohere](./06-cohere.md) | Enterprise RAG, embeddings |
| [Groq](./07-groq.md) | Ultra-low latency inference |
| [Together AI](./08-together-ai.md) | Open model hosting |
| [Fireworks AI](./09-fireworks.md) | Fast inference |
| [Replicate](./10-replicate.md) | Model marketplace |
| [DeepSeek](./11-deepseek.md) | Code and reasoning models |
| [xAI](./12-xai.md) | Grok models |
| [Alibaba Qwen](./13-alibaba-qwen.md) | Multilingual open source |
| [Other Providers](./14-other-providers.md) | AI21, MiniMax, NVIDIA, Zhipu, Kimi |
| [Open Source Tools](./15-open-source-tools.md) | Hugging Face, Ollama, LM Studio |
| [Specialized Providers](./16-specialized-providers.md) | Perplexity, Cursor |
| [Cloud AI Services](./17-cloud-ai-services.md) | AWS Bedrock, Azure OpenAI, Vertex |

### Prerequisites

Before this lesson, understand:
- Types of AI models (Lesson 07)
- Model selection criteria (Lesson 08)

---

## Landscape Overview

### Provider Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI PROVIDER LANDSCAPE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FRONTIER LABS           │  INFERENCE PROVIDERS                 │
│  (Train & Serve)         │  (Serve Open Models)                 │
│  ──────────────────      │  ─────────────────────               │
│  • OpenAI                │  • Groq (ultra-fast)                 │
│  • Anthropic             │  • Together AI                       │
│  • Google DeepMind       │  • Fireworks AI                      │
│  • Meta AI               │  • Replicate                         │
│                          │                                      │
│  CLOUD PROVIDERS         │  SPECIALIZED                         │
│  (Enterprise)            │  (Domain Focus)                      │
│  ──────────────────      │  ─────────────────────               │
│  • Azure OpenAI          │  • Perplexity (search)               │
│  • AWS Bedrock           │  • Cursor (code)                     │
│  • Google Vertex         │  • Cohere (RAG/embeddings)           │
│                          │                                      │
│  OPEN SOURCE             │  EMERGING PLAYERS                    │
│  ──────────────────      │  ─────────────────────               │
│  • Hugging Face          │  • DeepSeek                          │
│  • Ollama                │  • xAI (Grok)                        │
│  • LM Studio             │  • Mistral AI                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Comparison

| Provider | Best For | Models | Pricing Tier |
|----------|----------|--------|--------------|
| OpenAI | General purpose, latest capabilities | GPT-4o, o1, GPT-5 | $$$ |
| Anthropic | Safety, long context, coding | Claude 3.5/4 | $$$ |
| Google | Multimodal, massive context | Gemini 2.5 | $$ |
| Meta | Self-hosting, open source | Llama 4 | Free/$ |
| Groq | Real-time, lowest latency | Llama, Mixtral | $ |
| Together | Open models, variety | Many | $ |
| AWS Bedrock | Enterprise, multiple models | Various | $$$ |

---

## Choosing a Provider

### Decision Factors

```python
provider_selection_factors = {
    "capability": "What tasks do you need?",
    "cost": "What's your budget?",
    "latency": "How fast must responses be?",
    "privacy": "Where can data go?",
    "compliance": "What certifications needed?",
    "scale": "How much volume?",
    "support": "What SLA do you need?",
}
```

### Provider Selection Matrix

```python
def recommend_provider(requirements: dict) -> list:
    """Recommend providers based on requirements"""
    
    recommendations = []
    
    # Check each requirement
    if requirements.get("need_best_quality"):
        recommendations.append(("OpenAI GPT-4o or Claude 3.5", "Top quality"))
    
    if requirements.get("need_lowest_latency"):
        recommendations.append(("Groq", "Fastest inference"))
    
    if requirements.get("need_lowest_cost"):
        recommendations.append(("Self-hosted Llama or DeepSeek", "Minimal cost"))
    
    if requirements.get("need_compliance"):
        recommendations.append(("Azure OpenAI or AWS Bedrock", "Enterprise compliance"))
    
    if requirements.get("need_long_context"):
        recommendations.append(("Google Gemini 1.5 Pro", "1M+ context"))
    
    if requirements.get("need_open_source"):
        recommendations.append(("Meta Llama via Together/Groq", "Open weights"))
    
    return recommendations

# Example
providers = recommend_provider({
    "need_best_quality": True,
    "need_compliance": True
})
```

---

## Pricing Overview

### Cost Comparison (Per 1M Tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 |
| Google | Gemini 1.5 Pro | $1.25 | $5.00 |
| Google | Gemini 1.5 Flash | $0.075 | $0.30 |
| Groq | Llama 3 70B | ~$0.59 | ~$0.79 |
| Together | Llama 3 70B | ~$0.90 | ~$0.90 |
| DeepSeek | DeepSeek V3 | $0.27 | $1.10 |

---

## Lesson Structure

Each provider topic covers:

1. **Company Overview** - Background and focus
2. **Model Lineup** - Available models and capabilities
3. **API Features** - Unique features and integrations
4. **Pricing** - Cost structure and tiers
5. **Best Use Cases** - When to choose this provider
6. **Code Examples** - How to integrate

---

## Navigation

| Previous Lesson | Unit Home | Next Topic |
|-----------------|-----------|------------|
| [Model Selection Criteria](../08-model-selection-criteria/00-model-selection-criteria.md) | [AI/LLM Fundamentals](../00-overview.md) | [OpenAI](./01-openai.md) |

