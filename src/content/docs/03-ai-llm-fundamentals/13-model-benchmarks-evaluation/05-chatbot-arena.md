---
title: "Chatbot Arena & LMSYS Leaderboard"
---

# Chatbot Arena & LMSYS Leaderboard

## Introduction

Chatbot Arena provides human preference rankings through blind pairwise comparisons. Unlike automated benchmarks, it measures what humans actually prefer in model responses.

### What We'll Cover

- How Chatbot Arena works
- Understanding ELO ratings
- Category-specific rankings
- Limitations and considerations

---

## How Chatbot Arena Works

### The Process

```
┌─────────────────────────────────────────────────────────────┐
│                    CHATBOT ARENA FLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. USER SUBMITS PROMPT                                      │
│     └── Any question or task                                │
│                                                              │
│  2. TWO RANDOM MODELS RESPOND                                │
│     ├── Model A (identity hidden)                           │
│     └── Model B (identity hidden)                           │
│                                                              │
│  3. USER VOTES                                               │
│     ├── "A is better"                                       │
│     ├── "B is better"                                       │
│     ├── "Tie"                                               │
│     └── "Both are bad"                                      │
│                                                              │
│  4. IDENTITIES REVEALED                                      │
│     └── User sees which models they compared                │
│                                                              │
│  5. ELO RATINGS UPDATED                                      │
│     └── Winner gains points, loser loses points             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why Blind Comparison Matters

```python
blind_comparison_benefits = {
    "no_brand_bias": "Users can't favor OpenAI or Anthropic by name",
    "actual_preference": "Measures real human preference, not specs",
    "diverse_testing": "Users bring their own varied use cases",
    "continuous_update": "Rankings update as new votes come in"
}
```

---

## Understanding ELO Ratings

### What is ELO?

```python
elo_explanation = {
    "origin": "Chess rating system invented by Arpad Elo",
    "concept": "Zero-sum: winner gains what loser loses",
    "starting_point": 1000-1500 for new models",
    "interpretation": {
        "100_point_difference": "Expected 64% win rate for higher-rated",
        "200_point_difference": "Expected 76% win rate for higher-rated",
        "400_point_difference": "Expected 91% win rate for higher-rated"
    }
}
```

### ELO Calculation

```python
def calculate_new_elo(
    rating_a: float, 
    rating_b: float, 
    result: str,  # "A", "B", or "tie"
    k: float = 32  # K-factor
) -> tuple[float, float]:
    """Calculate new ELO ratings after a match"""
    
    # Expected scores
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    # Actual scores
    if result == "A":
        actual_a, actual_b = 1, 0
    elif result == "B":
        actual_a, actual_b = 0, 1
    else:  # tie
        actual_a, actual_b = 0.5, 0.5
    
    # New ratings
    new_a = rating_a + k * (actual_a - expected_a)
    new_b = rating_b + k * (actual_b - expected_b)
    
    return new_a, new_b
```

### Current Rankings Example

| Rank | Model | ELO | Provider |
|------|-------|-----|----------|
| 1 | GPT-4o | 1287 | OpenAI |
| 2 | Claude 3.5 Sonnet | 1271 | Anthropic |
| 3 | Gemini 1.5 Pro | 1260 | Google |
| 4 | GPT-4 Turbo | 1252 | OpenAI |
| 5 | Llama 3.1 405B | 1241 | Meta |
| 6 | Claude 3 Opus | 1233 | Anthropic |
| 7 | GPT-4o-mini | 1219 | OpenAI |
| 8 | Mistral Large | 1205 | Mistral |

> **Note:** Rankings change daily. Check lmsys.org for current data.

---

## Category-Specific Rankings

### Available Categories

```python
arena_categories = {
    "overall": "All prompts combined",
    "coding": "Programming and code-related tasks",
    "hard_prompts": "Complex, challenging requests",
    "creative": "Creative writing and artistic tasks",
    "math": "Mathematical problems",
    "instruction_following": "Following specific instructions",
    "longer_query": "Prompts with more context"
}
```

### Why Categories Matter

```python
# Models may rank differently by category
category_variations = {
    "GPT-4o": {
        "overall": 1287,
        "coding": 1295,  # Better at coding
        "creative": 1275  # Slightly lower for creative
    },
    "Claude-3.5-Sonnet": {
        "overall": 1271,
        "coding": 1290,  # Strong at coding
        "creative": 1285  # Better at creative than GPT-4o
    }
}

# Choose model based on your primary use case
def recommend_for_category(category: str) -> str:
    # Coding: Both GPT-4o and Claude excellent
    # Creative: Slight edge to Claude
    # Math: Edge to GPT-4o
    pass
```

---

## Accessing Arena Data

### LMSYS Resources

```python
lmsys_resources = {
    "arena_hard": "https://lmarena.ai/?arena",
    "leaderboard": "https://lmarena.ai/leaderboard",
    "blog": "https://blog.lmarena.ai/",
    "dataset": "Anonymized votes available for research"
}
```

### Using the API

```python
# Leaderboard data is available via the website
# For research, datasets can be requested

def get_model_ranking(model_name: str) -> dict:
    """Get model's current Arena ranking"""
    
    # Pseudo-code - actual implementation would scrape or use API
    rankings = fetch_arena_leaderboard()
    
    for rank, data in enumerate(rankings, 1):
        if data["model"] == model_name:
            return {
                "rank": rank,
                "elo": data["elo"],
                "votes": data["total_votes"],
                "confidence_interval": data["ci"]
            }
    
    return None
```

---

## Limitations

### What Arena Doesn't Capture

```python
arena_limitations = {
    "user_bias": "Users may have unstated preferences",
    "prompt_distribution": "Not all use cases represented equally",
    "surface_quality": "May favor verbose or polished responses",
    "correctness": "Users can't always verify factual accuracy",
    "specialized_tasks": "Fewer votes for niche domains",
    "recency_bias": "New models may be over/under-sampled initially"
}
```

### Arena vs Automated Benchmarks

| Aspect | Chatbot Arena | Automated Benchmarks |
|--------|--------------|---------------------|
| Measures | Human preference | Task accuracy |
| Subjectivity | High | Low |
| Coverage | User-defined prompts | Fixed test sets |
| Correctness | Not verified | Ground truth available |
| Sample size | ~1M+ votes | Fixed dataset |
| Update frequency | Continuous | Per evaluation |

### Best Practice: Use Both

```python
evaluation_strategy = {
    "automated_benchmarks": "Measure objective capabilities",
    "chatbot_arena": "Measure human preference",
    "your_own_testing": "Test on your specific use cases",
    "production_metrics": "Measure actual user satisfaction"
}
```

---

## Summary

✅ **Blind comparison**: No brand bias in voting

✅ **ELO ratings**: Chess-style ranking system

✅ **Categories**: Different rankings for different tasks

✅ **Limitations**: Surface preference, not verified accuracy

✅ **Best practice**: Combine with automated benchmarks

**Next:** [Performance Metrics](./06-performance-metrics.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Hallucination Metrics](./04-hallucination-metrics.md) | [Benchmarks](./00-model-benchmarks-evaluation.md) | [Performance Metrics](./06-performance-metrics.md) |
