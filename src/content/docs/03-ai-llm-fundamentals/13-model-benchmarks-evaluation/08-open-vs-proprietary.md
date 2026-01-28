---
title: "Open Weights vs Proprietary Models"
---

# Open Weights vs Proprietary Models

## Introduction

The AI landscape includes both proprietary models (OpenAI, Anthropic) and open-weight models (Meta's Llama, Mistral). Understanding the trade-offs helps you choose the right approach for your needs.

### What We'll Cover

- Performance parity trends
- Cost at scale differences
- Customization trade-offs
- When to use each

---

## Performance Comparison

### The Narrowing Gap

```
┌─────────────────────────────────────────────────────────────┐
│              PERFORMANCE GAP OVER TIME                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Capability ↑                                                │
│            │                                                 │
│  100% ─────│─────────────────────── ● GPT-4o (Proprietary)  │
│            │                    ● Claude 3.5                 │
│   95% ─────│                                                │
│            │              ● Llama 3.1 405B                   │
│   90% ─────│                    ● Qwen 2.5 72B              │
│            │                                                 │
│   85% ─────│           ● Mistral Large                      │
│            │                                                 │
│   80% ─────│      ● Llama 3.1 70B                           │
│            │                                                 │
│            │                                                 │
│            └────────────────────────────────────────→ Time  │
│             2023      2024      2025                        │
│                                                              │
│  Trend: Open models closing gap, ~6-12 months behind        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Current Comparison (Example Benchmarks)

| Model | Type | MMLU | HumanEval | Cost |
|-------|------|------|-----------|------|
| GPT-4o | Proprietary | 88.7 | 90.2 | $$$ |
| Claude 3.5 Sonnet | Proprietary | 88.3 | 92.0 | $$$ |
| Llama 3.1 405B | Open | 85.2 | 89.0 | $$ (self-host) |
| Llama 3.1 70B | Open | 82.0 | 80.5 | $ (self-host) |
| Qwen 2.5 72B | Open | 83.1 | 86.4 | $ (self-host) |
| Mistral Large 2 | Open* | 84.0 | 84.2 | $$ |

*Open weights with commercial restrictions

---

## Cost at Scale

### API Pricing vs Self-Hosting

```python
cost_comparison = {
    "api_pricing": {
        "advantages": [
            "No infrastructure management",
            "Pay per token used",
            "Scales automatically",
            "Latest models immediately"
        ],
        "costs": {
            "gpt-4o": "$2.50-$10 per 1M tokens",
            "claude-3.5-sonnet": "$3-$15 per 1M tokens"
        }
    },
    "self_hosting": {
        "advantages": [
            "Fixed cost at high volume",
            "Data stays on-premise",
            "Full customization control",
            "No rate limits"
        ],
        "costs": {
            "70b_model": "$1-2/hour per GPU (need 2-4 GPUs)",
            "8b_model": "$0.50-1/hour per GPU (1 GPU)"
        }
    }
}
```

### Break-Even Analysis

```python
def calculate_breakeven(
    monthly_tokens: int,
    api_cost_per_million: float,
    hosting_cost_per_hour: float,
    tokens_per_second: int,
    utilization: float = 0.7
) -> dict:
    """Calculate break-even point for self-hosting"""
    
    # API cost
    api_monthly_cost = (monthly_tokens / 1_000_000) * api_cost_per_million
    
    # Self-hosting cost (24/7)
    hours_per_month = 24 * 30
    hosting_monthly_cost = hosting_cost_per_hour * hours_per_month
    
    # Self-hosting capacity (tokens it can process)
    seconds_per_month = hours_per_month * 3600 * utilization
    hosting_capacity = tokens_per_second * seconds_per_month
    
    # Break-even
    if hosting_monthly_cost < api_monthly_cost:
        savings = api_monthly_cost - hosting_monthly_cost
        savings_percent = (savings / api_monthly_cost) * 100
        recommendation = "Self-hosting saves money"
    else:
        savings = hosting_monthly_cost - api_monthly_cost
        savings_percent = 0
        recommendation = "API is more cost-effective"
    
    return {
        "api_monthly_cost": api_monthly_cost,
        "hosting_monthly_cost": hosting_monthly_cost,
        "recommendation": recommendation,
        "monthly_savings": savings if hosting_monthly_cost < api_monthly_cost else 0,
        "savings_percent": savings_percent
    }

# Example: 100M tokens/month with GPT-4o vs Llama 70B
result = calculate_breakeven(
    monthly_tokens=100_000_000,
    api_cost_per_million=5.0,  # GPT-4o average
    hosting_cost_per_hour=4.0,  # 4x A100 GPUs
    tokens_per_second=100
)
# API: $500/month, Hosting: $2880/month → API wins at this volume
```

### When Self-Hosting Wins

```python
self_hosting_advantages = {
    "high_volume": "100M+ tokens/month may break even",
    "consistent_traffic": "Predictable, steady usage",
    "latency_requirements": "Need lowest possible latency",
    "data_privacy": "Cannot send data to third party",
    "customization": "Need fine-tuned models",
    "offline_requirements": "Air-gapped environments"
}
```

---

## Customization Trade-offs

### Proprietary Models

```python
proprietary_customization = {
    "fine_tuning": {
        "availability": "Limited (GPT-4 fine-tuning, Claude custom models)",
        "control": "Provider manages training",
        "cost": "Premium pricing for custom models",
        "data": "Data sent to provider"
    },
    "system_prompts": {
        "flexibility": "High",
        "persistence": "Per-request configuration"
    },
    "weights_access": "No access to model weights"
}
```

### Open Models

```python
open_customization = {
    "fine_tuning": {
        "availability": "Full control",
        "techniques": ["LoRA", "QLoRA", "Full fine-tune"],
        "cost": "GPU time only",
        "data": "Stays on your infrastructure"
    },
    "modifications": {
        "quantization": "Reduce model size (4-bit, 8-bit)",
        "distillation": "Create smaller specialized models",
        "merging": "Combine multiple LoRA adapters"
    },
    "deployment": {
        "options": ["vLLM", "TGI", "llama.cpp", "Ollama"],
        "flexibility": "Full control over serving infrastructure"
    }
}
```

### Fine-Tuning Comparison

| Aspect | Proprietary | Open |
|--------|-------------|------|
| Data privacy | Sent to provider | Stays local |
| Training control | Limited | Full |
| Cost | Per-token premium | GPU time only |
| Turnaround | Days-weeks | Hours-days |
| Techniques | Provider's method | Any approach |
| Model access | API only | Full weights |

---

## When to Use Each

### Choose Proprietary When

```python
proprietary_use_cases = {
    "highest_quality_needed": "Critical applications requiring best models",
    "quick_start": "MVP, prototyping, getting started fast",
    "variable_traffic": "Unpredictable usage patterns",
    "no_ml_team": "Don't have ML engineering resources",
    "latest_features": "Need newest capabilities immediately",
    "low_volume": "Thousands to low millions of tokens/month"
}
```

### Choose Open When

```python
open_use_cases = {
    "data_sensitivity": "Cannot send data to third parties",
    "customization": "Need fine-tuned models for domain",
    "cost_at_scale": "Very high token volumes",
    "latency_critical": "Need lowest possible latency",
    "offline": "Air-gapped or edge deployments",
    "regulatory": "Compliance requires on-premise",
    "ml_expertise": "Have team to manage infrastructure"
}
```

### Decision Framework

```python
def recommend_model_type(requirements: dict) -> str:
    """Recommend proprietary vs open based on requirements"""
    
    open_factors = 0
    proprietary_factors = 0
    
    # Data privacy
    if requirements.get("data_must_stay_private"):
        open_factors += 3
    
    # Volume
    monthly_tokens = requirements.get("monthly_tokens", 0)
    if monthly_tokens > 500_000_000:  # 500M
        open_factors += 2
    elif monthly_tokens < 10_000_000:  # 10M
        proprietary_factors += 1
    
    # Team capabilities
    if requirements.get("has_ml_team"):
        open_factors += 1
    else:
        proprietary_factors += 2
    
    # Quality requirements
    if requirements.get("needs_best_quality"):
        proprietary_factors += 2
    
    # Customization
    if requirements.get("needs_fine_tuning"):
        open_factors += 2
    
    # Time to market
    if requirements.get("needs_fast_launch"):
        proprietary_factors += 2
    
    if open_factors > proprietary_factors:
        return "Consider open-weight models"
    else:
        return "Consider proprietary APIs"
```

---

## Hybrid Approaches

### Best of Both Worlds

```python
hybrid_strategies = {
    "tiered_routing": {
        "description": "Use open for simple, proprietary for complex",
        "example": "Llama for FAQ, GPT-4o for analysis"
    },
    "development_vs_production": {
        "description": "Open for dev/test, proprietary for prod",
        "benefit": "Lower dev costs, best quality in prod"
    },
    "fallback_architecture": {
        "description": "Self-hosted primary, API fallback",
        "benefit": "Cost savings with reliability backup"
    },
    "specialized_routing": {
        "description": "Best model for each task type",
        "example": "Claude for writing, GPT-4o for code, Llama for translation"
    }
}
```

---

## Summary

✅ **Gap narrowing**: Open models 6-12 months behind proprietary

✅ **Cost**: API wins at low volume, self-hosting at scale

✅ **Customization**: Open offers full control, proprietary limited

✅ **Use case dependent**: No universal right answer

✅ **Hybrid**: Often best approach for production systems

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Interpreting Scores](./07-interpreting-scores.md) | [Benchmarks](./00-model-benchmarks-evaluation.md) | [Unit 3 Overview](../00-overview.md) |
