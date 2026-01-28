---
title: "Common Evaluation Benchmarks"
---

# Common Evaluation Benchmarks

## Introduction

The AI community has developed numerous benchmarks to evaluate different aspects of language model performance. Understanding the major benchmarks helps you interpret model comparisons.

### What We'll Cover

- Knowledge benchmarks (MMLU, ARC)
- Coding benchmarks (HumanEval, MBPP)
- Reasoning benchmarks (GSM8K, MATH)
- Conversation benchmarks (MT-Bench)

---

## Knowledge Benchmarks

### MMLU (Massive Multitask Language Understanding)

```python
mmlu_overview = {
    "full_name": "Massive Multitask Language Understanding",
    "questions": 15908,
    "subjects": 57,
    "format": "Multiple choice (4 options)",
    "domains": [
        "STEM (physics, math, chemistry, biology, CS)",
        "Humanities (history, philosophy, law)",
        "Social Sciences (psychology, economics, politics)",
        "Other (business, health, misc)"
    ]
}
```

**Example MMLU Question:**
```
Subject: High School Physics
Question: A ball is thrown vertically upward with an initial 
velocity of 20 m/s. Ignoring air resistance, what is its 
velocity at the highest point?
A) 20 m/s
B) 10 m/s
C) 0 m/s
D) -20 m/s
Answer: C
```

### ARC (AI2 Reasoning Challenge)

```python
arc_overview = {
    "description": "Grade school science questions",
    "splits": {
        "ARC-Easy": "Easier questions, higher baseline",
        "ARC-Challenge": "Harder questions, requires reasoning"
    },
    "format": "Multiple choice",
    "focus": "Scientific reasoning, not just recall"
}
```

### HellaSwag

```python
hellaswag_overview = {
    "description": "Commonsense reasoning about situations",
    "task": "Choose correct continuation of a scenario",
    "focus": "Real-world commonsense knowledge",
    "note": "Humans score ~95%, good test of understanding"
}
```

---

## Coding Benchmarks

### HumanEval

```python
humaneval_overview = {
    "description": "Code generation from docstrings",
    "problems": 164,
    "languages": "Python (original), MultiPL-E extends to others",
    "metric": "pass@k (% problems solved in k attempts)",
    "evaluation": "Functional correctness via unit tests"
}
```

**Example HumanEval Problem:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers 
    closer to each other than given threshold.
    
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Model must implement this
```

### MBPP (Mostly Basic Programming Problems)

```python
mbpp_overview = {
    "description": "Python programming problems",
    "problems": 974,
    "difficulty": "Entry-level to intermediate",
    "includes": "Task description, example I/O, test cases"
}
```

### SWE-bench

```python
swe_bench_overview = {
    "description": "Real-world GitHub issue resolution",
    "source": "Actual issues from popular Python repos",
    "task": "Given issue, produce pull request diff",
    "repos": "Django, Flask, pytest, scikit-learn, etc.",
    "difficulty": "Very challenging - real codebase understanding"
}
```

**Current SWE-bench Scores (Example):**
| Model | SWE-bench % |
|-------|-------------|
| Claude 3.5 Sonnet | 49.0 |
| GPT-4o | 33.2 |
| Gemini 1.5 Pro | 28.4 |

---

## Reasoning Benchmarks

### GSM8K (Grade School Math)

```python
gsm8k_overview = {
    "description": "Grade school math word problems",
    "problems": 8500,
    "difficulty": "Elementary to middle school level",
    "focus": "Multi-step arithmetic reasoning",
    "format": "Free-form answer (number)"
}
```

**Example GSM8K Problem:**
```
Janet's ducks lay 16 eggs per day. She eats three for 
breakfast every morning and bakes muffins for her friends 
every day with four. She sells the remainder at the farmers' 
market daily for $2 per fresh duck egg. How much in dollars 
does she make every day at the farmers' market?

Answer: 18
(16 - 3 - 4 = 9 eggs, 9 × $2 = $18)
```

### MATH

```python
math_overview = {
    "description": "Competition mathematics problems",
    "problems": 12500,
    "levels": ["Level 1 (easy)", "Level 2", "Level 3", "Level 4", "Level 5 (hard)"],
    "topics": [
        "Algebra", "Counting & Probability", "Geometry",
        "Intermediate Algebra", "Number Theory", 
        "Prealgebra", "Precalculus"
    ],
    "difficulty": "AMC, AIME, Olympiad level"
}
```

### GPQA Diamond

```python
gpqa_diamond_overview = {
    "description": "Graduate-level science questions",
    "difficulty": "PhD-level experts score ~65%",
    "domains": "Physics, Chemistry, Biology",
    "format": "Multiple choice",
    "note": "Very challenging, good discriminator for top models"
}
```

---

## Conversation Benchmarks

### MT-Bench

```python
mt_bench_overview = {
    "description": "Multi-turn conversation quality",
    "turns": 2,
    "categories": [
        "Writing", "Roleplay", "Extraction",
        "Reasoning", "Math", "Coding",
        "Knowledge", "Common sense"
    ],
    "scoring": "GPT-4 as judge, 1-10 scale",
    "focus": "Quality over multiple conversation turns"
}
```

**Example MT-Bench Dialog:**
```
Turn 1: Write a persuasive email to convince your introverted 
friend to join a group vacation.

Turn 2: Now rewrite the email, but use only formal English, 
no contractions, and make it exactly 150 words.
```

---

## Benchmark Comparison Table

| Benchmark | Focus | # Problems | Format | Difficulty |
|-----------|-------|------------|--------|------------|
| MMLU | Knowledge | 15,908 | Multiple choice | Mixed |
| ARC-Challenge | Reasoning | 1,172 | Multiple choice | Medium |
| HumanEval | Coding | 164 | Code generation | Medium |
| MBPP | Coding | 974 | Code generation | Easy-Medium |
| SWE-bench | Real coding | 2,294 | Code diff | Hard |
| GSM8K | Math | 8,500 | Free answer | Medium |
| MATH | Math | 12,500 | Free answer | Hard |
| GPQA Diamond | Science | 198 | Multiple choice | Very Hard |
| MT-Bench | Conversation | 80 | Open-ended | Medium |

---

## How to Use These Benchmarks

### Matching Task to Benchmark

```python
def relevant_benchmarks(your_task: str) -> list[str]:
    """Get benchmarks most relevant to your use case"""
    
    task_benchmark_map = {
        "customer_support": ["MT-Bench", "HellaSwag"],
        "code_assistant": ["HumanEval", "SWE-bench", "MBPP"],
        "research_helper": ["MMLU", "GPQA Diamond"],
        "math_tutor": ["GSM8K", "MATH"],
        "general_qa": ["MMLU", "ARC-Challenge", "TriviaQA"],
        "reasoning_tasks": ["ARC-Challenge", "Big-Bench Hard"]
    }
    
    return task_benchmark_map.get(your_task, ["MMLU"])
```

---

## Summary

✅ **MMLU**: Broad knowledge across 57 subjects

✅ **HumanEval/SWE-bench**: Code generation ability

✅ **GSM8K/MATH**: Mathematical reasoning

✅ **MT-Bench**: Multi-turn conversation quality

✅ **GPQA Diamond**: Expert-level science knowledge

**Next:** [Hallucination Metrics](./04-hallucination-metrics.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Artificial Analysis](./02-artificial-analysis-index.md) | [Benchmarks](./00-model-benchmarks-evaluation.md) | [Hallucination Metrics](./04-hallucination-metrics.md) |
