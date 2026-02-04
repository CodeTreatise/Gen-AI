---
title: "A/B Testing and Debugging"
---

# A/B Testing and Debugging

## Introduction

A/B testing lets you compare prompt variants with statistical rigor. Debugging helps you understand why prompts fail and systematically fix them. Together, they complete the eval-driven development workflow.

This lesson covers practical techniques for comparing prompts and diagnosing failures.

### What We'll Cover

- A/B testing prompt variants
- Multi-model comparison
- Statistical significance
- Production traffic splitting
- Failure mode analysis
- Debugging techniques
- Change impact tracking

### Prerequisites

- [Eval-Driven Development](./04-eval-driven-development.md)
- Basic statistics (mean, variance, confidence intervals)

---

## A/B Testing Prompts

### When to A/B Test

| Scenario | A/B Test? |
|----------|-----------|
| Major prompt rewrite | ✅ Yes |
| Adding few-shot examples | ✅ Yes |
| Model upgrade (4.1 → 5) | ✅ Yes |
| Temperature change | ✅ Yes |
| Minor wording tweak | Maybe (run eval first) |
| Bug fix for specific case | No (use regression test) |

### Basic A/B Test Structure

```python
import random

def ab_test_prompts(
    prompt_a: str,
    prompt_b: str,
    test_cases: list,
    model: str = "gpt-4.1"
) -> dict:
    """Run A/B test on two prompt variants."""
    
    results = {"A": [], "B": []}
    
    for case in test_cases:
        # Randomize order to eliminate position effects
        variants = [("A", prompt_a), ("B", prompt_b)]
        random.shuffle(variants)
        
        for name, prompt in variants:
            output = call_model(prompt, case["input"], model)
            correct = evaluate(output, case["expected"])
            results[name].append({
                "id": case["id"],
                "output": output,
                "correct": correct
            })
    
    # Calculate metrics
    accuracy_a = sum(r["correct"] for r in results["A"]) / len(results["A"])
    accuracy_b = sum(r["correct"] for r in results["B"]) / len(results["B"])
    
    return {
        "prompt_a_accuracy": accuracy_a,
        "prompt_b_accuracy": accuracy_b,
        "difference": accuracy_b - accuracy_a,
        "results": results
    }
```

### Sample Size Considerations

| Accuracy Difference | Needed Sample Size |
|--------------------|--------------------|
| 10%+ | ~100 samples |
| 5% | ~400 samples |
| 2% | ~2,000 samples |
| 1% | ~8,000 samples |

> **Rule of thumb:** More samples = more confident. Start with 100-200 for quick comparison, scale up for production decisions.

---

## Statistical Significance

### Why It Matters

```
Prompt A: 85% accuracy (100 samples)
Prompt B: 88% accuracy (100 samples)

Is B actually better, or just random variation?
```

### Chi-Square Test

For comparing pass/fail rates:

```python
from scipy.stats import chi2_contingency

def is_significant(results_a, results_b, alpha=0.05):
    """Test if difference is statistically significant."""
    
    pass_a = sum(1 for r in results_a if r["correct"])
    fail_a = len(results_a) - pass_a
    pass_b = sum(1 for r in results_b if r["correct"])
    fail_b = len(results_b) - fail_b
    
    # Create contingency table
    table = [[pass_a, fail_a], [pass_b, fail_b]]
    
    chi2, p_value, dof, expected = chi2_contingency(table)
    
    return {
        "p_value": p_value,
        "significant": p_value < alpha,
        "interpretation": "Significant difference" if p_value < alpha else "No significant difference"
    }

# Usage
result = is_significant(results["A"], results["B"])
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
```

### Confidence Intervals

```python
import numpy as np
from scipy import stats

def confidence_interval(results, confidence=0.95):
    """Calculate confidence interval for accuracy."""
    
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n
    
    # Wilson score interval (better for proportions)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / n
    center = (accuracy + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((accuracy * (1 - accuracy) + z**2 / (4*n)) / n) / denominator
    
    return {
        "accuracy": accuracy,
        "lower": center - margin,
        "upper": center + margin,
        "confidence": confidence
    }

# Usage
ci_a = confidence_interval(results["A"])
ci_b = confidence_interval(results["B"])

print(f"A: {ci_a['accuracy']:.1%} [{ci_a['lower']:.1%}, {ci_a['upper']:.1%}]")
print(f"B: {ci_b['accuracy']:.1%} [{ci_b['lower']:.1%}, {ci_b['upper']:.1%}]")
```

**Interpretation:**
```
A: 85.0% [77.3%, 90.6%]
B: 88.0% [80.7%, 92.8%]

Overlapping intervals → Not clearly different
Non-overlapping → Likely real difference
```

---

## Multi-Model Comparison

### Comparing Models

```python
def compare_models(
    prompt: str,
    models: list,
    test_cases: list
) -> dict:
    """Compare the same prompt across multiple models."""
    
    results = {model: [] for model in models}
    
    for case in test_cases:
        for model in models:
            output = call_model(prompt, case["input"], model)
            correct = evaluate(output, case["expected"])
            results[model].append({
                "id": case["id"],
                "output": output,
                "correct": correct,
                "latency_ms": measure_latency()
            })
    
    # Summary
    summary = {}
    for model, model_results in results.items():
        accuracy = sum(r["correct"] for r in model_results) / len(model_results)
        avg_latency = sum(r["latency_ms"] for r in model_results) / len(model_results)
        summary[model] = {
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency
        }
    
    return summary

# Usage
models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
comparison = compare_models(prompt, models, test_cases)

for model, metrics in comparison.items():
    print(f"{model}: {metrics['accuracy']:.1%} accuracy, {metrics['avg_latency_ms']:.0f}ms")
```

**Output:**
```
gpt-4.1:      94.0% accuracy, 1250ms
gpt-4.1-mini: 91.0% accuracy, 450ms
gpt-4.1-nano: 86.0% accuracy, 120ms
```

### Cost-Performance Trade-offs

```python
def cost_performance_analysis(model_results: dict, cost_per_1k: dict):
    """Analyze cost vs performance trade-offs."""
    
    analysis = []
    for model, metrics in model_results.items():
        cost = cost_per_1k.get(model, 0)
        
        analysis.append({
            "model": model,
            "accuracy": metrics["accuracy"],
            "latency_ms": metrics["avg_latency_ms"],
            "cost_per_1k": cost,
            "accuracy_per_dollar": metrics["accuracy"] / (cost + 0.001)  # Avoid div/0
        })
    
    # Sort by accuracy/dollar ratio
    analysis.sort(key=lambda x: -x["accuracy_per_dollar"])
    
    return analysis

# Usage
cost_per_1k = {"gpt-4.1": 2.50, "gpt-4.1-mini": 0.40, "gpt-4.1-nano": 0.10}
analysis = cost_performance_analysis(comparison, cost_per_1k)
```

---

## Production Traffic Splitting

### Gradual Rollout

```
Day 1: 100% A, 0% B           (Baseline)
Day 2: 95% A, 5% B            (Canary)
Day 3: 90% A, 10% B           (Expand if no issues)
Day 4: 50% A, 50% B           (Full comparison)
Day 5: 10% A, 90% B           (Prepare for switch)
Day 6: 0% A, 100% B           (Complete migration)
```

### Traffic Splitting Implementation

```python
import hashlib

def get_variant(user_id: str, experiment: str, split: float = 0.5) -> str:
    """Deterministic A/B assignment based on user ID."""
    
    # Hash user_id + experiment for consistent assignment
    key = f"{user_id}:{experiment}"
    hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    # Use hash to determine bucket
    bucket = (hash_value % 100) / 100
    
    return "B" if bucket < split else "A"

# Usage
def handle_request(user_id: str, input_text: str):
    variant = get_variant(user_id, "prompt_v2_test", split=0.1)  # 10% to B
    
    if variant == "B":
        prompt = prompt_b
    else:
        prompt = prompt_a
    
    output = call_model(prompt, input_text)
    
    # Log for analysis
    log_experiment(user_id, variant, input_text, output)
    
    return output
```

### Monitoring During Rollout

| Metric | Action if Threshold Exceeded |
|--------|------------------------------|
| Error rate > 2x baseline | Auto-rollback to 100% A |
| Latency p99 > 3x | Alert + manual review |
| User complaints spike | Pause rollout |
| Accuracy drop > 5% | Rollback |

---

## Failure Mode Analysis

### Categorizing Failures

```python
def analyze_failures(results: list) -> dict:
    """Group failures by pattern."""
    
    failures = [r for r in results if not r["correct"]]
    
    categories = {}
    for f in failures:
        # Determine failure type
        category = classify_failure(f["output"], f["expected"])
        
        if category not in categories:
            categories[category] = []
        categories[category].append(f)
    
    # Sort by frequency
    sorted_categories = sorted(
        categories.items(),
        key=lambda x: -len(x[1])
    )
    
    return dict(sorted_categories)

def classify_failure(output: str, expected: str) -> str:
    """Classify the type of failure."""
    
    if not output or output.strip() == "":
        return "Empty output"
    
    if "I cannot" in output or "I'm sorry" in output:
        return "Refusal"
    
    if expected.lower() in output.lower():
        return "Format issue (correct content, wrong format)"
    
    # Check for common issues
    if len(output) > 500:
        return "Too verbose"
    
    if len(output) < 10:
        return "Too brief"
    
    return "Wrong answer"
```

### Failure Dashboard

```
Failure Analysis Report
=======================
Total: 100 tests, 15 failures (85% accuracy)

By Type:
├── Wrong answer: 7 (46.7%)
│   └── Examples: Case #23, #45, #67
├── Format issue: 4 (26.7%)
│   └── Examples: Case #12, #34
├── Too verbose: 2 (13.3%)
│   └── Examples: Case #56, #78
├── Refusal: 1 (6.7%)
│   └── Examples: Case #89
└── Empty output: 1 (6.7%)
    └── Examples: Case #90

Recommended Actions:
1. [Wrong answer] Add more examples for confused categories
2. [Format issue] Add explicit format instructions
3. [Too verbose] Add word limit
```

---

## Debugging Techniques

### Prompt Debugging Checklist

| Issue | Diagnostic | Fix |
|-------|-----------|-----|
| Model ignores instructions | Check if instruction is clear and prominent | Move to top, make bold |
| Wrong format | Model may not understand format spec | Add example output |
| Hallucinations | No grounding information | Add "only use provided info" |
| Too long | No length constraint | Add word/sentence limit |
| Too short | Insufficient detail requested | Ask for "detailed" response |
| Wrong tone | Tone not specified | Add tone instructions |

### Step-by-Step Debugging

```python
def debug_case(prompt: str, input_text: str, expected: str, model: str):
    """Debug a specific failing case."""
    
    print("=" * 60)
    print("DEBUGGING CASE")
    print("=" * 60)
    
    # 1. Show the full prompt
    full_prompt = f"{prompt}\n\nInput: {input_text}"
    print("\n📝 FULL PROMPT:")
    print(full_prompt)
    
    # 2. Get model output
    output = call_model(prompt, input_text, model)
    print(f"\n🤖 MODEL OUTPUT:")
    print(output)
    
    # 3. Show expected
    print(f"\n✅ EXPECTED:")
    print(expected)
    
    # 4. Analyze mismatch
    print(f"\n🔍 ANALYSIS:")
    if output.strip() == expected.strip():
        print("  ✓ Exact match")
    elif expected.lower() in output.lower():
        print("  ⚠ Contains expected, but format differs")
    else:
        print("  ✗ Content mismatch")
        
    # 5. Check for common issues
    if len(output) > 2 * len(expected):
        print("  ⚠ Output much longer than expected")
    if "I " in output[:20]:
        print("  ⚠ Output may be too conversational")
        
    # 6. Suggest fixes
    print("\n💡 SUGGESTIONS:")
    if len(output) > len(expected) * 2:
        print("  - Add word limit to prompt")
    if expected.lower() not in output.lower():
        print("  - Add explicit instruction for this case type")
    
    return output
```

### Ablation Studies

Remove parts of the prompt to see what's essential:

```python
def ablation_study(base_prompt: str, sections: dict, test_cases: list):
    """Test which prompt sections are essential."""
    
    results = {}
    
    # Test full prompt
    full_accuracy = run_eval(base_prompt, test_cases)
    results["full"] = full_accuracy
    print(f"Full prompt: {full_accuracy:.1%}")
    
    # Test with each section removed
    for section_name, section_text in sections.items():
        ablated_prompt = base_prompt.replace(section_text, "")
        accuracy = run_eval(ablated_prompt, test_cases)
        results[f"without_{section_name}"] = accuracy
        
        impact = full_accuracy - accuracy
        print(f"Without {section_name}: {accuracy:.1%} (impact: {impact:+.1%})")
    
    return results

# Usage
sections = {
    "examples": "Examples:\n- Input: X → Output: Y\n- Input: A → Output: B",
    "constraints": "Constraints:\n- Keep response under 100 words",
    "formatting": "Format your response as JSON",
}

ablation_study(full_prompt, sections, test_cases)
```

**Output:**
```
Full prompt: 94.0%
Without examples: 82.0% (impact: -12.0%)      ← Critical!
Without constraints: 91.0% (impact: -3.0%)    ← Helpful
Without formatting: 94.0% (impact: +0.0%)     ← Not needed
```

---

## Change Impact Tracking

### Version History

```python
# prompts/versions.yaml

versions:
  - version: "1.0"
    date: "2024-01-01"
    accuracy: 0.75
    changes: "Initial prompt"
    
  - version: "1.1"
    date: "2024-01-05"
    accuracy: 0.82
    changes: "Added 3 few-shot examples"
    
  - version: "1.2"
    date: "2024-01-10"
    accuracy: 0.88
    changes: "Added edge case handling for multi-topic emails"
    regressions: ["Case #45 now fails"]
    
  - version: "2.0"
    date: "2024-01-15"
    accuracy: 0.94
    changes: "Major rewrite with structured output"
    ab_test: "5% improvement vs 1.2, p < 0.01"
```

### Change Impact Report

```python
def generate_impact_report(old_results: list, new_results: list, change_description: str):
    """Generate a change impact report."""
    
    old_pass = {r["id"]: r["correct"] for r in old_results}
    new_pass = {r["id"]: r["correct"] for r in new_results}
    
    improved = [id for id in new_pass if new_pass[id] and not old_pass.get(id)]
    regressed = [id for id in old_pass if old_pass[id] and not new_pass.get(id)]
    
    old_accuracy = sum(old_pass.values()) / len(old_pass)
    new_accuracy = sum(new_pass.values()) / len(new_pass)
    
    report = f"""
    Change Impact Report
    ====================
    Change: {change_description}
    
    Accuracy: {old_accuracy:.1%} → {new_accuracy:.1%} ({new_accuracy - old_accuracy:+.1%})
    
    Improvements: {len(improved)} cases
    {chr(10).join(f'  - {id}' for id in improved[:5])}
    {'  ...' if len(improved) > 5 else ''}
    
    Regressions: {len(regressed)} cases
    {chr(10).join(f'  - {id}' for id in regressed[:5])}
    {'  ...' if len(regressed) > 5 else ''}
    
    Net change: {len(improved) - len(regressed):+d} cases
    
    Recommendation: {"SHIP" if len(regressed) == 0 or len(improved) > len(regressed) * 2 else "REVIEW"}
    """
    
    return report.strip()
```

---

## Hands-on Exercise

### Your Task

Design a complete A/B testing plan for upgrading a customer service prompt:

1. Define your test criteria
2. Calculate required sample size
3. Plan your rollout schedule
4. Create a monitoring checklist

<details>
<summary>✅ Solution (click to expand)</summary>

**1. Test Criteria:**
```yaml
Primary Metrics:
  - Resolution rate (customer satisfied)
  - Accuracy (correct information provided)
  
Secondary Metrics:
  - Response length (words)
  - Latency (ms)
  - Cost per query

Success Threshold:
  - New prompt must match or exceed current on resolution rate
  - No more than 5% regression on any metric
  - p-value < 0.05 for primary metrics
```

**2. Sample Size Calculation:**
```python
# Current accuracy: 88%
# Expected improvement: 3% (to 91%)
# Desired power: 80%
# Significance level: 0.05

from statsmodels.stats.power import NormalIndPower

analysis = NormalIndPower()
sample_size = analysis.solve_power(
    effect_size=0.03 / 0.5,  # Cohen's h for proportions
    power=0.80,
    alpha=0.05,
    ratio=1.0
)

print(f"Required per variant: {int(sample_size)} samples")
# Result: ~800 samples per variant
```

**3. Rollout Schedule:**
```yaml
Week 1:
  - Day 1-2: 0% B (final prep, verify monitoring)
  - Day 3-4: 2% B (canary, ~20 requests)
  - Day 5-7: 5% B (early signal, ~100 requests)
  
Week 2:
  - Day 8-10: 10% B (~300 requests)
  - Day 11-14: 25% B (~700 requests)
  
Week 3:
  - Day 15-18: 50% B (~1500 requests)
  - Day 19-21: Statistical analysis, decision

Week 4 (if approved):
  - Day 22-24: 75% B (prepare for full switch)
  - Day 25-28: 100% B (complete migration)
  
Rollback Triggers:
  - Error rate > 2x baseline: Immediate rollback
  - Resolution rate drops > 10%: Pause at current level
  - Any critical bug: Immediate rollback
```

**4. Monitoring Checklist:**
```yaml
Hourly:
  □ Error rate (should be < 1%)
  □ p99 latency (should be < 3s)
  □ Request volume (ensure traffic split is correct)

Daily:
  □ Resolution rate by variant
  □ Accuracy by variant
  □ User feedback / complaints
  □ Sample 10 responses from each variant for manual review

Weekly:
  □ Statistical significance check
  □ Regression analysis on edge cases
  □ Cost analysis
  □ Stakeholder update

Alerts (auto):
  - Error rate > 2x: PagerDuty
  - Latency p99 > 5s: Slack
  - Resolution rate < 80%: Email
```

</details>

---

## Lesson 19 Complete ✅

You've completed the Prompt Evaluation & Optimization lesson. You now understand:

1. **Evaluation fundamentals** and success criteria
2. **OpenAI Evals system** with data_source_config and testing_criteria
3. **Graders** for automated testing (string, similarity, model, python)
4. **Prompt optimizer** workflow with annotations
5. **Eval-driven development** methodology
6. **A/B testing** and debugging techniques

---

## Summary

✅ **A/B test significant changes** with proper sample sizes
✅ **Check statistical significance** before drawing conclusions
✅ **Compare models** for cost-performance trade-offs
✅ **Roll out gradually** with monitoring and rollback triggers
✅ **Analyze failures systematically** by category
✅ **Track changes** with version history and impact reports

**Next Lesson:** [Prompt Security & Injection Defense](../20-prompt-security-injection-defense.md)

---

## Further Reading

- [OpenAI Cookbook: Bulk Model Experimentation](https://cookbook.openai.com/)
- [OpenAI Cookbook: Completion Monitoring](https://cookbook.openai.com/)
- [Statistical Methods for A/B Testing](https://www.exp-platform.com/Documents/GuideControlledExperiments.pdf)

---

<!-- 
Sources Consulted:
- OpenAI Evaluation Best Practices: A/B testing, continuous evaluation
- OpenAI Cookbook: Bulk experimentation, regression detection
- Statistical analysis: Standard proportions testing methods
-->
