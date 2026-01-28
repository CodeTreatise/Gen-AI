---
title: "Benchmark-Based Selection"
---

# Benchmark-Based Selection

## Introduction

Benchmarks provide quantitative comparisons between models. However, understanding their limitations and how to interpret them is crucial for making good decisions.

### What We'll Cover

- Key benchmarks (MMLU, HumanEval, GPQA)
- Throughput and latency metrics
- Model versioning considerations
- Benchmark limitations

---

## Key Benchmarks

### Popular Evaluation Benchmarks

| Benchmark | Measures | Range | Good Score |
|-----------|----------|-------|------------|
| MMLU | General knowledge (57 subjects) | 0-100% | >80% |
| HumanEval | Python coding | 0-100% | >70% |
| GPQA | Graduate-level science | 0-100% | >50% |
| GSM8K | Grade school math | 0-100% | >90% |
| HellaSwag | Common sense reasoning | 0-100% | >95% |
| ARC-C | Science reasoning | 0-100% | >90% |
| TruthfulQA | Factual accuracy | 0-100% | >60% |

### Current Model Scores (2025-2026)

```python
benchmark_scores = {
    "gpt-4o": {
        "mmlu": 88.7,
        "humaneval": 90.2,
        "gpqa": 53.6,
        "gsm8k": 95.3,
        "arc_c": 96.4,
    },
    "claude-3-5-sonnet": {
        "mmlu": 88.3,
        "humaneval": 92.0,
        "gpqa": 59.4,
        "gsm8k": 96.4,
        "arc_c": 96.7,
    },
    "gemini-1.5-pro": {
        "mmlu": 85.9,
        "humaneval": 84.1,
        "gpqa": 46.2,
        "gsm8k": 91.7,
        "arc_c": 94.4,
    },
    "llama-3-70b": {
        "mmlu": 82.0,
        "humaneval": 81.7,
        "gpqa": 39.5,
        "gsm8k": 93.0,
        "arc_c": 93.0,
    },
    "gpt-4o-mini": {
        "mmlu": 82.0,
        "humaneval": 87.2,
        "gpqa": 40.2,
        "gsm8k": 93.2,
        "arc_c": 90.2,
    }
}
```

### Benchmark Visualization

```python
def compare_models_on_benchmarks(
    models: list,
    benchmarks: list
) -> dict:
    """Compare models across benchmarks"""
    
    comparison = {}
    
    for benchmark in benchmarks:
        scores = {}
        for model in models:
            if model in benchmark_scores:
                scores[model] = benchmark_scores[model].get(benchmark, None)
        
        # Rank models
        ranked = sorted(
            [(m, s) for m, s in scores.items() if s],
            key=lambda x: x[1],
            reverse=True
        )
        
        comparison[benchmark] = {
            "scores": scores,
            "best": ranked[0] if ranked else None,
            "ranking": [m for m, s in ranked]
        }
    
    return comparison

# Compare on coding
result = compare_models_on_benchmarks(
    ["gpt-4o", "claude-3-5-sonnet", "llama-3-70b"],
    ["humaneval", "mmlu"]
)
print(f"Best for coding: {result['humaneval']['best']}")
```

---

## Throughput Metrics

### Measuring Performance

```python
import time
from concurrent.futures import ThreadPoolExecutor

def measure_throughput(
    client,
    model: str,
    num_requests: int = 100,
    concurrent: int = 10
) -> dict:
    """Measure model throughput"""
    
    prompt = "Write a haiku about programming."
    results = []
    
    def make_request():
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        end = time.time()
        tokens = response.usage.completion_tokens
        return {
            "latency": end - start,
            "tokens": tokens
        }
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        results = list(executor.map(lambda _: make_request(), range(num_requests)))
    
    total_time = time.time() - start_time
    total_tokens = sum(r["tokens"] for r in results)
    
    return {
        "total_requests": num_requests,
        "total_time_seconds": round(total_time, 2),
        "requests_per_second": round(num_requests / total_time, 2),
        "tokens_per_second": round(total_tokens / total_time, 2),
        "avg_latency_ms": round(sum(r["latency"] for r in results) / num_requests * 1000, 2),
        "p99_latency_ms": round(sorted([r["latency"] for r in results])[int(num_requests * 0.99)] * 1000, 2)
    }
```

### Throughput Comparison

| Model | Tokens/sec (est.) | TTFT (ms) | Best For |
|-------|-------------------|-----------|----------|
| GPT-4o | ~50 | 400-600 | Quality |
| GPT-4o-mini | ~100 | 200-300 | Speed + Quality |
| Claude Haiku | ~100 | 200-300 | Speed |
| Gemini Flash | ~150 | 150-250 | Maximum speed |
| Llama 3 70B (self-hosted) | ~30-100 | 100-500 | Control |

---

## Model Versioning

### Version Management

```python
model_versions = {
    "gpt-4o": {
        "latest": "gpt-4o-2024-11-20",
        "stable": "gpt-4o-2024-08-06",
        "deprecation": {
            "gpt-4o-2024-05-13": "2025-06-01"
        }
    },
    "claude-3-5-sonnet": {
        "latest": "claude-3-5-sonnet-20241022",
        "previous": "claude-3-5-sonnet-20240620"
    }
}

def get_model_version(
    model_family: str,
    preference: str = "stable"
) -> str:
    """Get appropriate model version"""
    
    versions = model_versions.get(model_family, {})
    
    if preference == "latest":
        return versions.get("latest", model_family)
    elif preference == "stable":
        return versions.get("stable", versions.get("latest", model_family))
    
    return model_family
```

### Handling Updates

```python
class VersionedModelClient:
    """Client with version pinning and migration support"""
    
    def __init__(self, model_family: str, pin_version: bool = True):
        self.model_family = model_family
        self.pinned_version = None
        
        if pin_version:
            self.pinned_version = get_model_version(model_family, "stable")
    
    def get_model(self) -> str:
        """Get model to use"""
        return self.pinned_version or self.model_family
    
    def check_for_updates(self) -> dict:
        """Check for model updates"""
        latest = get_model_version(self.model_family, "latest")
        current = self.get_model()
        
        return {
            "current": current,
            "latest": latest,
            "update_available": current != latest,
            "deprecation_date": model_versions.get(
                self.model_family, {}
            ).get("deprecation", {}).get(current)
        }
    
    def migrate_to_latest(self):
        """Migrate to latest version"""
        self.pinned_version = get_model_version(self.model_family, "latest")
```

---

## Benchmark Limitations

### Known Issues

```python
benchmark_limitations = {
    "contamination": {
        "issue": "Training data may include benchmark questions",
        "impact": "Inflated scores don't reflect real capability",
        "mitigation": "Use private evaluation sets"
    },
    "gaming": {
        "issue": "Models optimized specifically for benchmarks",
        "impact": "Benchmark scores don't transfer to real tasks",
        "mitigation": "Evaluate on your actual use cases"
    },
    "narrow_scope": {
        "issue": "Benchmarks test limited capabilities",
        "impact": "Miss important real-world skills",
        "mitigation": "Use multiple diverse benchmarks"
    },
    "static": {
        "issue": "Benchmarks don't evolve with capabilities",
        "impact": "Ceiling effects as models improve",
        "mitigation": "Use newer, harder benchmarks"
    },
    "format_sensitivity": {
        "issue": "Small prompt changes affect scores",
        "impact": "Scores aren't fully comparable",
        "mitigation": "Standardize evaluation methodology"
    }
}
```

### Better Evaluation Approaches

```python
class RealWorldEvaluator:
    """Evaluate on your actual use case"""
    
    def __init__(self, test_cases: list):
        self.test_cases = test_cases
        self.results = {}
    
    def evaluate_model(self, model: str, client) -> dict:
        """Run evaluation on test cases"""
        
        correct = 0
        total = len(self.test_cases)
        latencies = []
        
        for case in self.test_cases:
            start = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": case["input"]}]
            )
            
            latency = time.time() - start
            latencies.append(latency)
            
            output = response.choices[0].message.content
            
            # Check against expected output
            if self._check_output(output, case["expected"]):
                correct += 1
        
        self.results[model] = {
            "accuracy": correct / total,
            "avg_latency": sum(latencies) / len(latencies),
            "total_cost": self._estimate_cost(model)
        }
        
        return self.results[model]
    
    def _check_output(self, output: str, expected: str) -> bool:
        """Check if output matches expected"""
        # Implement your comparison logic
        return expected.lower() in output.lower()
    
    def compare_models(self) -> dict:
        """Compare all evaluated models"""
        if not self.results:
            return {}
        
        best_accuracy = max(self.results.items(), key=lambda x: x[1]["accuracy"])
        best_latency = min(self.results.items(), key=lambda x: x[1]["avg_latency"])
        
        return {
            "best_accuracy": best_accuracy[0],
            "best_latency": best_latency[0],
            "all_results": self.results
        }
```

---

## Creating Your Own Benchmarks

### Custom Evaluation Set

```python
def create_custom_benchmark(
    use_case: str,
    num_examples: int = 50
) -> list:
    """Create custom benchmark for your use case"""
    
    templates = {
        "customer_support": {
            "categories": ["billing", "technical", "general"],
            "format": {
                "input": "Customer query",
                "expected": "Appropriate response category and solution"
            }
        },
        "code_review": {
            "categories": ["bugs", "style", "security"],
            "format": {
                "input": "Code snippet",
                "expected": "Identified issues and fixes"
            }
        },
        "summarization": {
            "categories": ["short", "medium", "long"],
            "format": {
                "input": "Document text",
                "expected": "Key points covered"
            }
        }
    }
    
    template = templates.get(use_case, templates["customer_support"])
    
    # Generate test cases
    test_cases = []
    for i in range(num_examples):
        category = template["categories"][i % len(template["categories"])]
        test_cases.append({
            "id": i,
            "category": category,
            "input": f"Example {i} for {category}",  # Replace with real data
            "expected": f"Expected output {i}",  # Replace with real expectations
            "metadata": {"use_case": use_case}
        })
    
    return test_cases
```

---

## Decision Matrix

### Selecting by Benchmark + Needs

```python
def recommend_model(
    primary_need: str,
    secondary_needs: list = None
) -> dict:
    """Recommend model based on benchmarks and needs"""
    
    recommendations = {
        "general_knowledge": {
            "best": "gpt-4o",
            "benchmark": "mmlu",
            "alternatives": ["claude-3-5-sonnet", "gemini-1.5-pro"]
        },
        "coding": {
            "best": "claude-3-5-sonnet",
            "benchmark": "humaneval",
            "alternatives": ["gpt-4o", "deepseek-coder"]
        },
        "math": {
            "best": "claude-3-5-sonnet",
            "benchmark": "gsm8k",
            "alternatives": ["gpt-4o", "gemini-1.5-pro"]
        },
        "science": {
            "best": "claude-3-5-sonnet",
            "benchmark": "gpqa",
            "alternatives": ["gpt-4o"]
        },
        "speed": {
            "best": "gemini-1.5-flash",
            "benchmark": "throughput",
            "alternatives": ["gpt-4o-mini", "claude-haiku"]
        },
        "cost": {
            "best": "gpt-4o-mini",
            "benchmark": "cost_per_token",
            "alternatives": ["gemini-flash", "llama-3-70b"]
        }
    }
    
    primary = recommendations.get(primary_need, recommendations["general_knowledge"])
    
    return {
        "recommended": primary["best"],
        "based_on": primary["benchmark"],
        "alternatives": primary["alternatives"],
        "note": "Always validate with your own evaluation"
    }

# Example
print(recommend_model("coding"))
```

---

## Summary

✅ **Use benchmarks as starting point** - Not final decision

✅ **MMLU**: General knowledge, HumanEval: Coding, GPQA: Science

✅ **Measure real throughput** - Actual performance varies

✅ **Pin model versions** - Avoid surprise changes

✅ **Create your own benchmarks** - Test on your actual tasks

**Lesson Complete!** Return to [Model Selection Overview](./00-model-selection-criteria.md)

---

## Navigation

| Previous | Up | Next Lesson |
|----------|-------|------|
| [Open Source vs Proprietary](./07-open-source-vs-proprietary.md) | [Model Selection](./00-model-selection-criteria.md) | [AI Providers Landscape](../09-ai-providers-landscape/00-ai-providers-landscape.md) |

