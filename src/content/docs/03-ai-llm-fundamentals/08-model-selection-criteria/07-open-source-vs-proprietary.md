---
title: "Open Source vs Proprietary Models"
---

# Open Source vs Proprietary Models

## Introduction

The choice between open-source and proprietary models involves trade-offs between control, cost, capability, and convenience. Understanding these trade-offs helps you make the right choice for your use case.

### What We'll Cover

- Hosted API convenience
- Self-hosting benefits
- Cost comparison at scale
- Customization options

---

## The Landscape

### Model Categories

```python
model_categories = {
    "proprietary_api": {
        "examples": ["GPT-4o", "Claude 3.5", "Gemini"],
        "access": "API only",
        "control": "Low",
        "setup_effort": "Minimal",
        "cost_model": "Pay per token"
    },
    "open_weights": {
        "examples": ["Llama 3", "Mistral", "Qwen"],
        "access": "Weights available",
        "control": "High",
        "setup_effort": "Medium to High",
        "cost_model": "Infrastructure costs"
    },
    "open_source": {
        "examples": ["Bloom", "Falcon", "OLMo"],
        "access": "Weights + training code",
        "control": "Full",
        "setup_effort": "High",
        "cost_model": "Infrastructure costs"
    }
}
```

### Current Leaders

| Category | Top Models | Strengths |
|----------|-----------|-----------|
| Proprietary | GPT-4o, Claude 3.5 | Best overall capability |
| Open Weights | Llama 3 70B, Mistral Large | Strong, deployable |
| Open Source | Qwen 72B, DeepSeek | Competitive quality |

---

## Proprietary API Advantages

### Convenience

```python
# Proprietary: 5 lines to production
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Why Choose Proprietary APIs

```python
proprietary_advantages = {
    "zero_infrastructure": "No GPUs, no DevOps, no maintenance",
    "instant_scaling": "Handles any traffic automatically",
    "latest_models": "Access to newest capabilities first",
    "enterprise_support": "SLAs, support contracts available",
    "compliance": "SOC 2, GDPR compliance handled",
    "updates": "Security and capability updates automatic",
}
```

### Cost Predictability

```python
def estimate_api_costs(
    daily_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """Estimate monthly API costs"""
    
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    
    prices = pricing.get(model, pricing["gpt-4o-mini"])
    
    daily_cost = (
        (daily_requests * avg_input_tokens / 1_000_000) * prices["input"] +
        (daily_requests * avg_output_tokens / 1_000_000) * prices["output"]
    )
    
    return {
        "daily": round(daily_cost, 2),
        "monthly": round(daily_cost * 30, 2),
        "yearly": round(daily_cost * 365, 2),
        "no_infrastructure_costs": True
    }
```

---

## Self-Hosting Advantages

### Full Control

```python
self_hosting_advantages = {
    "data_privacy": "Data never leaves your infrastructure",
    "no_rate_limits": "Limited only by your hardware",
    "customization": "Fine-tune for your use case",
    "cost_at_scale": "Cheaper above certain volume",
    "offline_capable": "Works without internet",
    "regulatory": "Meet any compliance requirement",
}
```

### Self-Hosted Architecture

```python
# Example: vLLM deployment
from vllm import LLM, SamplingParams

class SelfHostedLLM:
    """Self-hosted LLM with vLLM"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf"):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=4,  # Number of GPUs
            gpu_memory_utilization=0.9
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2048
        )
    
    def generate(self, prompts: list) -> list:
        """Generate completions"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

# Usage
# llm = SelfHostedLLM()
# responses = llm.generate(["Hello, how are you?"])
```

### Infrastructure Options

| Option | Cost/Month | Throughput | Setup |
|--------|-----------|------------|-------|
| Single A100 (80GB) | ~$2,000 | ~50 req/s | Medium |
| 4x A100 Cluster | ~$8,000 | ~200 req/s | High |
| 8x H100 Cluster | ~$30,000 | ~1000 req/s | High |
| Serverless (Modal, Replicate) | Variable | Scalable | Low |

---

## Cost Comparison

### Break-Even Analysis

```python
def calculate_breakeven(
    monthly_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int
) -> dict:
    """Calculate break-even point for self-hosting"""
    
    # API costs (GPT-4o-mini)
    api_cost = (
        (monthly_requests * avg_input_tokens / 1_000_000) * 0.15 +
        (monthly_requests * avg_output_tokens / 1_000_000) * 0.60
    )
    
    # Self-hosting costs (Llama 3 70B on 4x A100)
    infrastructure_cost = 8000  # Monthly
    setup_cost_amortized = 500  # One-time spread over months
    ops_cost = 1000  # DevOps time
    self_hosted_cost = infrastructure_cost + setup_cost_amortized + ops_cost
    
    # Break-even
    if self_hosted_cost < api_cost:
        savings = api_cost - self_hosted_cost
        recommendation = "Self-host"
    else:
        savings = self_hosted_cost - api_cost
        recommendation = "Use API"
    
    return {
        "api_monthly_cost": round(api_cost, 2),
        "self_hosted_monthly_cost": round(self_hosted_cost, 2),
        "savings": round(savings, 2),
        "recommendation": recommendation,
        "break_even_requests": round(self_hosted_cost / (
            (avg_input_tokens / 1_000_000) * 0.15 +
            (avg_output_tokens / 1_000_000) * 0.60
        ))
    }

# Example: 1M requests/month
result = calculate_breakeven(
    monthly_requests=1_000_000,
    avg_input_tokens=500,
    avg_output_tokens=200
)
print(f"Recommendation: {result['recommendation']}")
print(f"Monthly savings: ${result['savings']}")
```

### Total Cost of Ownership

```python
tco_factors = {
    "api": {
        "direct_costs": "Token usage",
        "hidden_costs": "Rate limit delays, vendor lock-in",
        "savings": "No infrastructure, no ops team"
    },
    "self_hosted": {
        "direct_costs": "GPU rental/purchase, electricity",
        "hidden_costs": "DevOps time, monitoring, security",
        "savings": "No per-token costs, full control"
    }
}
```

---

## Customization

### Fine-Tuning Options

```python
fine_tuning_comparison = {
    "openai_fine_tuning": {
        "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "method": "Supervised fine-tuning",
        "data_requirements": "~50-100 examples minimum",
        "cost": "Training + higher inference cost",
        "control": "Limited (can't access weights)"
    },
    "open_source_fine_tuning": {
        "models": ["Llama 3", "Mistral", "Qwen"],
        "methods": ["Full fine-tune", "LoRA", "QLoRA"],
        "data_requirements": "Flexible",
        "cost": "GPU time only",
        "control": "Full (own the weights)"
    }
}
```

### LoRA Fine-Tuning Example

```python
# Fine-tuning open source with LoRA
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_lora_training(base_model: str):
    """Setup LoRA fine-tuning"""
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=16,                     # LoRA rank
        lora_alpha=32,            # Scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Trainable params: ~0.1% of total
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model
```

---

## Decision Framework

### When to Use Proprietary APIs

```python
use_proprietary_when = [
    "Getting started or prototyping",
    "Low to medium volume (<100K requests/month)",
    "Need latest model capabilities",
    "No dedicated ML infrastructure team",
    "Compliance handled by provider is acceptable",
    "Rapid scaling requirements",
]
```

### When to Self-Host

```python
use_self_hosted_when = [
    "High volume (>500K requests/month)",
    "Strict data privacy requirements",
    "Need full customization/fine-tuning",
    "Have ML infrastructure team",
    "Predictable costs required",
    "Offline operation needed",
]
```

### Decision Flowchart

```
                    START
                      │
         ┌────────────┴────────────┐
         │ Do you have strict      │
         │ data privacy needs?     │
         └────────────┬────────────┘
                 YES  │  NO
                  │   └──────────────────┐
                  │                      │
         ┌────────┴────────┐    ┌────────┴────────┐
         │ Self-host or    │    │ Volume > 500K   │
         │ private cloud   │    │ requests/month? │
         └─────────────────┘    └────────┬────────┘
                                    YES  │  NO
                                     │   └─────────┐
                                     │             │
                            ┌────────┴──────┐      │
                            │ Calculate     │      │
                            │ break-even    │      │
                            └───────────────┘      │
                                                   │
                                          ┌────────┴────────┐
                                          │ Use Proprietary │
                                          │ API             │
                                          └─────────────────┘
```

---

## Hybrid Approaches

### Best of Both Worlds

```python
class HybridAIClient:
    """Use both proprietary and self-hosted models"""
    
    def __init__(self):
        self.api_client = OpenAI()
        self.local_llm = None  # Lazy load
    
    def query(
        self,
        prompt: str,
        use_case: str = "general"
    ) -> str:
        """Route to appropriate model"""
        
        routing = {
            "simple_qa": "local",       # Fast, cheap
            "complex_reasoning": "api",  # Best quality
            "code_generation": "api",    # Latest capabilities
            "bulk_processing": "local",  # Cost effective
            "customer_facing": "api",    # Reliability
            "internal_tools": "local",   # Privacy
        }
        
        target = routing.get(use_case, "api")
        
        if target == "local":
            return self._local_query(prompt)
        else:
            return self._api_query(prompt)
    
    def _api_query(self, prompt: str) -> str:
        response = self.api_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _local_query(self, prompt: str) -> str:
        if self.local_llm is None:
            self._init_local_llm()
        return self.local_llm.generate([prompt])[0]
```

---

## Summary

✅ **Proprietary APIs**: Convenience, latest capabilities, no infrastructure

✅ **Self-hosted**: Control, privacy, cost at scale

✅ **Break-even**: ~500K+ requests/month typically favors self-hosting

✅ **Hybrid**: Use APIs for quality, self-hosted for volume

✅ **Consider TCO**: Include ops costs, not just compute

**Next:** [Benchmark-Based Selection](./08-benchmark-based-selection.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Compliance & Privacy](./06-compliance-data-privacy.md) | [Model Selection](./00-model-selection-criteria.md) | [Benchmark-Based Selection](./08-benchmark-based-selection.md) |

