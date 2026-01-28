---
title: "Artificial Analysis Intelligence Index"
---

# Artificial Analysis Intelligence Index

## Introduction

Artificial Analysis provides independent, comprehensive model rankings using their Intelligence Index v4.0 methodology. This index evaluates models across 10 different evaluations for a balanced assessment.

### What We'll Cover

- The 10 evaluations methodology
- Understanding the composite score
- Current model rankings
- How to use the index

---

## The 10 Evaluations

### Methodology Overview

```
┌─────────────────────────────────────────────────────────────┐
│          ARTIFICIAL ANALYSIS INTELLIGENCE INDEX              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  REASONING & KNOWLEDGE (4 evaluations)                       │
│  ├── GPQA Diamond - Graduate-level science QA               │
│  ├── CritPt - Critical reasoning assessment                 │
│  ├── Humanity's Last Exam - Expert knowledge test           │
│  └── AA-Omniscience - Hallucination measurement             │
│                                                              │
│  CODING (3 evaluations)                                      │
│  ├── SciCode - Scientific coding challenges                 │
│  ├── Terminal-Bench Hard - Terminal/system tasks            │
│  └── τ²-Bench (Tau-bench) - Agent coding tasks              │
│                                                              │
│  INSTRUCTION FOLLOWING (2 evaluations)                       │
│  ├── IFBench - Instruction following accuracy               │
│  └── AA-LCR - Long context retrieval                        │
│                                                              │
│  AGENT TASKS (1 evaluation)                                  │
│  └── GDPval-AA - Complex multi-step agent tasks             │
│                                                              │
│  ────────────────────────────────────────────────────────   │
│  COMPOSITE: Weighted average → Intelligence Index Score     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Individual Evaluations

| Evaluation | Focus | Description |
|------------|-------|-------------|
| **GPQA Diamond** | Science | PhD-level questions in physics, chemistry, biology |
| **CritPt** | Critical thinking | Logical reasoning and analysis |
| **Humanity's Last Exam** | Expert knowledge | Questions experts would ask |
| **AA-Omniscience** | Hallucination | Factual accuracy measurement |
| **SciCode** | Scientific coding | Code for scientific computing |
| **Terminal-Bench Hard** | System ops | Shell commands and system tasks |
| **τ²-Bench** | Agentic coding | Multi-step coding with tools |
| **IFBench** | Instruction following | Following complex instructions |
| **AA-LCR** | Long context | Needle-in-haystack retrieval |
| **GDPval-AA** | Agent tasks | Complex multi-step workflows |

---

## Understanding the Score

### Score Interpretation

```python
intelligence_index_interpretation = {
    "90+": "Frontier capability - Best available models",
    "80-89": "Strong performance - Production-ready for most tasks",
    "70-79": "Good performance - Suitable for many applications",
    "60-69": "Moderate - May struggle with complex tasks",
    "below_60": "Limited - Best for simple, specific use cases"
}
```

### Composite Calculation

```python
# Simplified representation of scoring
def calculate_intelligence_index(model_scores: dict) -> float:
    """
    Each evaluation is weighted and normalized.
    The composite balances breadth of capabilities.
    """
    
    weights = {
        "gpqa_diamond": 0.12,
        "critpt": 0.10,
        "humanitys_last_exam": 0.10,
        "aa_omniscience": 0.10,
        "scicode": 0.10,
        "terminal_bench_hard": 0.10,
        "tau_bench": 0.10,
        "ifbench": 0.10,
        "aa_lcr": 0.08,
        "gdpval_aa": 0.10
    }
    
    weighted_sum = sum(
        model_scores[eval_name] * weight 
        for eval_name, weight in weights.items()
    )
    
    return weighted_sum
```

---

## Current Rankings (Example)

### Top Models by Intelligence Index

| Rank | Model | Index Score | Provider |
|------|-------|-------------|----------|
| 1 | Claude 3.5 Sonnet | 85.2 | Anthropic |
| 2 | GPT-4o | 83.7 | OpenAI |
| 3 | Gemini 1.5 Pro | 81.4 | Google |
| 4 | GPT-4o-mini | 75.3 | OpenAI |
| 5 | Claude 3 Haiku | 72.1 | Anthropic |
| 6 | Llama 3.1 70B | 71.8 | Meta |
| 7 | Mistral Large | 70.5 | Mistral |
| 8 | Qwen 2.5 72B | 69.2 | Alibaba |

> **Note:** Rankings change frequently as new models release. Check artificialanalysis.ai for current data.

### By Category

```python
top_models_by_category = {
    "reasoning": {
        "1": "Claude 3.5 Sonnet",
        "2": "GPT-4o",
        "why": "Best at GPQA Diamond, CritPt"
    },
    "coding": {
        "1": "Claude 3.5 Sonnet",
        "2": "GPT-4o",
        "why": "Leads on SWE-bench, Terminal-Bench"
    },
    "instruction_following": {
        "1": "GPT-4o",
        "2": "Claude 3.5 Sonnet",
        "why": "Highest IFBench scores"
    },
    "long_context": {
        "1": "Gemini 1.5 Pro",
        "2": "Claude 3.5 Sonnet",
        "why": "Best retrieval at 100K+ context"
    }
}
```

---

## Using the Index

### Model Selection Workflow

```python
def select_model_with_aa_index(
    task_type: str,
    budget: str,
    minimum_score: float = 70.0
) -> list[str]:
    """Select models using Artificial Analysis data"""
    
    # Define task to evaluation mapping
    task_evaluations = {
        "coding": ["scicode", "terminal_bench", "tau_bench"],
        "reasoning": ["gpqa_diamond", "critpt", "humanitys_last_exam"],
        "qa_accuracy": ["aa_omniscience"],
        "long_documents": ["aa_lcr"],
        "agents": ["gdpval_aa", "tau_bench"]
    }
    
    relevant_evals = task_evaluations.get(task_type, [])
    
    # Fetch and filter models (pseudo-code)
    models = fetch_aa_rankings()
    
    filtered = [
        m for m in models
        if m.intelligence_index >= minimum_score
        and m.fits_budget(budget)
        and m.strong_in(relevant_evals)
    ]
    
    return sorted(filtered, key=lambda m: m.intelligence_index, reverse=True)
```

### Comparison Example

```python
# Compare two models for a specific task
def compare_for_task(model_a: str, model_b: str, task: str) -> str:
    """Compare models for specific task type"""
    
    evaluations_for_task = {
        "coding": ["SciCode", "Terminal-Bench Hard", "τ²-Bench"],
        "research": ["GPQA Diamond", "Humanity's Last Exam", "AA-Omniscience"]
    }
    
    evals = evaluations_for_task.get(task, [])
    
    print(f"Comparing {model_a} vs {model_b} for {task}:")
    print("-" * 50)
    
    for eval_name in evals:
        score_a = get_score(model_a, eval_name)
        score_b = get_score(model_b, eval_name)
        winner = model_a if score_a > score_b else model_b
        print(f"{eval_name}: {model_a}={score_a:.1f}, {model_b}={score_b:.1f} → {winner}")
```

---

## Data Access

### Artificial Analysis Resources

```python
aa_resources = {
    "leaderboard": "https://artificialanalysis.ai/leaderboard",
    "methodology": "https://artificialanalysis.ai/methodology",
    "api": "https://artificialanalysis.ai/api",
    "updates": "Rankings updated as new models release"
}
```

### Key Features

- **Independent testing**: Not provider-submitted scores
- **Consistent methodology**: Same tests across all models
- **Regular updates**: New models added quickly
- **Performance data**: Speed and cost alongside quality

---

## Summary

✅ **10 evaluations**: Comprehensive coverage of capabilities

✅ **Composite score**: Single number for quick comparison

✅ **Category breakdowns**: Task-specific performance

✅ **Independent**: Not relying on provider claims

**Next:** [Common Benchmarks](./03-common-benchmarks.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Why Benchmarks Matter](./01-why-benchmarks-matter.md) | [Benchmarks](./00-model-benchmarks-evaluation.md) | [Common Benchmarks](./03-common-benchmarks.md) |
