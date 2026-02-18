---
title: "Evaluation Frameworks"
---

# Evaluation Frameworks

## Introduction

Tracing tells you **what** your agent did. Evaluation tells you **how well** it did it. Without systematic evaluation, you're relying on gut feelings and anecdotal user feedback to assess quality. Evaluation frameworks let you define measurable quality criteria, run agents against benchmark datasets, detect regressions, and compare different agent versions — all automatically.

This lesson covers evaluation from simple metrics (keyword matching) through sophisticated approaches (LLM-as-a-judge, human annotation) and shows how to integrate evaluation into your development and CI/CD workflows.

### What we'll cover

- Evaluation fundamentals: what to measure and how
- LangSmith evaluators and evaluation workflows
- Custom evaluation functions
- LLM-as-a-judge patterns
- Benchmark datasets and dataset management
- Regression testing in CI/CD
- Cross-platform evaluation strategies

### Prerequisites

- LangSmith or Langfuse account (Lessons 18-01 or 18-03)
- Python 3.10+ with `langsmith` installed
- Understanding of agent tracing (Lessons 18-01 through 18-05)
- Basic testing concepts (Unit 2, Lesson 15)

---

## What to evaluate

Agent evaluation covers multiple dimensions. A complete evaluation strategy measures all of them.

| Dimension | What It Measures | Example Metric |
|-----------|-----------------|----------------|
| **Correctness** | Is the answer factually right? | Keyword overlap, LLM-judge score |
| **Relevance** | Does the answer address the question? | Cosine similarity with expected |
| **Helpfulness** | Is the response useful and actionable? | LLM-judge rating 1–5 |
| **Safety** | Does it avoid harmful content? | Content filter pass rate |
| **Latency** | How fast does the agent respond? | P50/P95 in seconds |
| **Cost** | How many tokens does it use? | USD per request |
| **Tool usage** | Does it call the right tools? | Tool selection accuracy |
| **Groundedness** | Is the response supported by sources? | Citation accuracy |

---

## Simple evaluators

Start with straightforward evaluators that don't require an LLM.

### Keyword overlap evaluator

```python
def keyword_overlap(outputs: dict, reference_outputs: dict) -> dict:
    """Measure word overlap between predicted and expected answers."""
    predicted = set(outputs.get("answer", "").lower().split())
    expected = set(reference_outputs.get("answer", "").lower().split())
    
    if not expected:
        return {"key": "keyword_overlap", "score": 0.0}
    
    overlap = len(predicted & expected) / len(expected)
    return {"key": "keyword_overlap", "score": round(overlap, 3)}
```

### Exact match evaluator

```python
def exact_match(outputs: dict, reference_outputs: dict) -> dict:
    """Check if the answer exactly matches the expected output."""
    predicted = outputs.get("answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()
    
    return {
        "key": "exact_match",
        "score": 1.0 if predicted == expected else 0.0,
    }
```

### Contains evaluator

```python
def contains_keywords(outputs: dict, reference_outputs: dict) -> dict:
    """Check if the answer contains required keywords."""
    answer = outputs.get("answer", "").lower()
    keywords = reference_outputs.get("required_keywords", [])
    
    found = sum(1 for kw in keywords if kw.lower() in answer)
    score = found / max(len(keywords), 1)
    
    return {"key": "contains_keywords", "score": round(score, 3)}
```

---

## LLM-as-a-judge

For nuanced evaluation, use an LLM to judge agent outputs. This is the most common pattern for evaluating open-ended text generation.

### Basic LLM judge

```python
from openai import OpenAI

judge_client = OpenAI()

def llm_judge_correctness(outputs: dict, reference_outputs: dict) -> dict:
    """Use GPT-4o to judge if the answer is correct."""
    predicted = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")
    question = reference_outputs.get("question", "")
    
    response = judge_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Score the predicted answer 
against the expected answer on a scale of 0.0 to 1.0.

Score 1.0: The predicted answer is factually correct and covers all key points.
Score 0.5: The answer is partially correct or misses important details.
Score 0.0: The answer is incorrect or irrelevant.

Respond with ONLY a JSON object: {"score": 0.0, "reasoning": "..."}""",
            },
            {
                "role": "user",
                "content": f"""Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}""",
            },
        ],
        temperature=0,
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    
    return {
        "key": "llm_correctness",
        "score": result["score"],
        "comment": result["reasoning"],
    }
```

### Multi-criteria judge

```python
def llm_judge_multi(outputs: dict, reference_outputs: dict) -> list[dict]:
    """Evaluate on multiple criteria simultaneously."""
    predicted = outputs.get("answer", "")
    question = reference_outputs.get("question", "")
    
    response = judge_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Evaluate the answer on three criteria (0.0 to 1.0 each):
1. correctness: Is it factually accurate?
2. helpfulness: Is it useful and actionable?
3. conciseness: Is it appropriately brief?

Respond with JSON: {"correctness": 0.0, "helpfulness": 0.0, "conciseness": 0.0}""",
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {predicted}",
            },
        ],
        temperature=0,
    )
    
    import json
    scores = json.loads(response.choices[0].message.content)
    
    return [
        {"key": "correctness", "score": scores["correctness"]},
        {"key": "helpfulness", "score": scores["helpfulness"]},
        {"key": "conciseness", "score": scores["conciseness"]},
    ]
```

> **⚠️ Warning:** LLM judges are not deterministic. Run each evaluation 3–5 times and average the scores for more reliable results.

---

## Benchmark datasets

Systematic evaluation requires curated datasets. Here's how to build and manage them.

### Dataset structure

```python
# A good evaluation dataset
evaluation_data = [
    {
        "input": {"question": "What is RAG?"},
        "expected": {
            "answer": "RAG (Retrieval Augmented Generation) combines document retrieval with LLM generation.",
            "required_keywords": ["retrieval", "generation", "documents"],
        },
    },
    {
        "input": {"question": "How do agents use tools?"},
        "expected": {
            "answer": "Agents use tools by generating structured function calls that are executed and returned as context.",
            "required_keywords": ["function", "calls", "context"],
        },
    },
    # ... 20+ examples for meaningful evaluation
]
```

### Creating datasets in LangSmith

```python
from langsmith import Client

ls = Client()

# Create dataset
dataset = ls.create_dataset(
    "Agent QA v2",
    description="Core questions for RAG agent evaluation",
)

# Add examples
for item in evaluation_data:
    ls.create_example(
        inputs=item["input"],
        outputs=item["expected"],
        dataset_id=dataset.id,
    )

print(f"Created dataset with {len(evaluation_data)} examples")
```

### Dataset guidelines

| Guideline | Recommendation |
|-----------|---------------|
| **Minimum size** | 20–50 examples for basic evaluation |
| **Coverage** | Include edge cases, not just happy paths |
| **Diversity** | Vary question types, lengths, and topics |
| **Ground truth** | Have human-verified expected outputs |
| **Updates** | Add new examples when you find failure modes |
| **Versioning** | Create new datasets for major agent changes |

---

## Running evaluations with LangSmith

LangSmith's `evaluate()` function runs your agent against a dataset and applies evaluators:

```python
from langsmith import evaluate

def my_agent(inputs: dict) -> dict:
    """The agent we want to evaluate."""
    question = inputs["question"]
    answer = rag_pipeline(question)  # Your agent function
    return {"answer": answer}

# Run evaluation
results = evaluate(
    my_agent,
    data="Agent QA v2",          # Dataset name
    evaluators=[
        keyword_overlap,
        contains_keywords,
        llm_judge_correctness,
    ],
    experiment_prefix="rag-v2",   # Groups results in the UI
    max_concurrency=4,            # Parallel evaluation
)

# Print aggregate results
for metric_name, score in results.aggregate_metrics.items():
    print(f"{metric_name}: {score:.2%}")
```

**Output:**
```
keyword_overlap: 72.50%
contains_keywords: 85.00%
llm_correctness: 90.00%
```

In the LangSmith UI, navigate to your dataset → Experiments tab. You'll see all experiment runs compared side-by-side with per-example scores.

---

## Regression testing in CI/CD

The real value of evaluation is catching regressions **before** they reach production. Integrate evaluations into your CI/CD pipeline.

### GitHub Actions example

```yaml
# .github/workflows/agent-evaluation.yml
name: Agent Evaluation

on:
  pull_request:
    paths:
      - 'agent/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run evaluation
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGSMITH_TRACING: true
        run: python evaluate.py
      
      - name: Check quality gate
        run: python check_quality.py
```

### Quality gate script

```python
# check_quality.py
import sys
from langsmith import Client

ls = Client()

# Get the latest experiment
experiments = ls.list_experiments(dataset_name="Agent QA v2", limit=1)
latest = list(experiments)[0]

# Check aggregate scores
results = ls.get_experiment_results(latest.id)

thresholds = {
    "keyword_overlap": 0.70,
    "llm_correctness": 0.85,
}

all_passed = True
for metric, threshold in thresholds.items():
    score = results.aggregate_metrics.get(metric, 0)
    status = "✅" if score >= threshold else "❌"
    print(f"{status} {metric}: {score:.2%} (threshold: {threshold:.0%})")
    if score < threshold:
        all_passed = False

if not all_passed:
    print("\n❌ Quality gate FAILED — agent quality below threshold")
    sys.exit(1)
else:
    print("\n✅ Quality gate PASSED")
```

**Output (passing):**
```
✅ keyword_overlap: 72.50% (threshold: 70%)
✅ llm_correctness: 90.00% (threshold: 85%)

✅ Quality gate PASSED
```

**Output (failing):**
```
✅ keyword_overlap: 72.50% (threshold: 70%)
❌ llm_correctness: 78.00% (threshold: 85%)

❌ Quality gate FAILED — agent quality below threshold
```

---

## Evaluation comparison patterns

### A/B testing agents

```python
from langsmith import evaluate

# Evaluate version A
results_a = evaluate(
    agent_v1,
    data="Agent QA v2",
    evaluators=[llm_judge_correctness],
    experiment_prefix="agent-v1",
)

# Evaluate version B
results_b = evaluate(
    agent_v2,
    data="Agent QA v2",
    evaluators=[llm_judge_correctness],
    experiment_prefix="agent-v2",
)

# Compare
score_a = results_a.aggregate_metrics["llm_correctness"]
score_b = results_b.aggregate_metrics["llm_correctness"]

print(f"Agent v1: {score_a:.2%}")
print(f"Agent v2: {score_b:.2%}")
print(f"Delta: {score_b - score_a:+.2%}")
```

**Output:**
```
Agent v1: 85.00%
Agent v2: 91.00%
Delta: +6.00%
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Start with simple evaluators | Keyword overlap catches obvious failures fast |
| Add LLM-as-a-judge for nuance | Human-like evaluation for open-ended responses |
| Maintain 20+ examples minimum | Small datasets give unstable scores |
| Run evaluations on every PR | Catch regressions before they reach production |
| Average LLM judge scores | Single runs are noisy; average 3–5 runs |
| Version your datasets | Track how evaluation criteria evolve |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Only using exact match | LLM outputs vary; use semantic similarity or LLM-judge |
| Too few evaluation examples | Minimum 20; aim for 50+ for reliable metrics |
| Not evaluating tool usage | Check whether the agent called the right tools, not just the final answer |
| Expensive evaluation runs | Use `gpt-4o-mini` for the judge when `gpt-4o` precision isn't needed |
| No quality gate in CI/CD | Set minimum thresholds and fail the build when they're not met |
| Evaluating only happy paths | Include edge cases, adversarial inputs, and out-of-scope questions |

---

## Hands-on exercise

### Your task

Build a complete evaluation pipeline with a dataset, multiple evaluators, and a quality gate.

### Requirements

1. Create a dataset with at least 5 question/answer pairs
2. Implement a keyword overlap evaluator
3. Implement an LLM-as-a-judge evaluator
4. Run the agent against the dataset using `evaluate()`
5. Write a quality gate that fails if correctness drops below 80%

### Expected result

Console output showing per-evaluator aggregate scores and a pass/fail quality gate decision.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `langsmith.evaluate()` to run the pipeline
- The LLM judge should return `{"key": "...", "score": 0.0-1.0}`
- Use `results.aggregate_metrics` for the quality gate check
- `sys.exit(1)` fails a CI/CD pipeline

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import sys
from openai import OpenAI
from langsmith import Client, evaluate
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())
ls = Client()
judge = OpenAI()

# 1. Create dataset
dataset = ls.create_dataset("Eval Exercise")
for q, a in [
    ("What is Python?", "Python is a programming language."),
    ("What is an API?", "An API is an application programming interface."),
    ("What is JSON?", "JSON is a data interchange format."),
    ("What is REST?", "REST is an architectural style for web APIs."),
    ("What is HTTP?", "HTTP is a protocol for transferring web data."),
]:
    ls.create_example(
        inputs={"question": q}, outputs={"answer": a},
        dataset_id=dataset.id,
    )

# 2. Agent
def my_agent(inputs):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": r.choices[0].message.content}

# 3. Evaluators
def keyword_overlap(outputs, reference_outputs):
    pred = set(outputs["answer"].lower().split())
    ref = set(reference_outputs["answer"].lower().split())
    score = len(pred & ref) / max(len(ref), 1)
    return {"key": "overlap", "score": round(score, 3)}

def llm_judge(outputs, reference_outputs):
    import json
    r = judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Score 0.0-1.0. Respond JSON: {\"score\": 0.0}"},
            {"role": "user", "content": f"Expected: {reference_outputs['answer']}\nGot: {outputs['answer']}"},
        ],
        temperature=0,
    )
    score = json.loads(r.choices[0].message.content)["score"]
    return {"key": "llm_judge", "score": score}

# 4. Run
results = evaluate(my_agent, data="Eval Exercise", evaluators=[keyword_overlap, llm_judge])

# 5. Quality gate
for name, score in results.aggregate_metrics.items():
    status = "✅" if score >= 0.80 else "❌"
    print(f"{status} {name}: {score:.2%}")

if results.aggregate_metrics.get("llm_judge", 0) < 0.80:
    sys.exit(1)
```

</details>

### Bonus challenges

- [ ] Add a multi-criteria LLM judge (correctness + helpfulness + conciseness)
- [ ] Implement an evaluator that checks tool selection accuracy
- [ ] Set up the quality gate in a GitHub Actions workflow

---

## Summary

✅ **Simple evaluators** (keyword overlap, contains) catch obvious quality issues fast  
✅ **LLM-as-a-judge** provides nuanced evaluation for open-ended responses  
✅ **Benchmark datasets** with 20+ examples enable systematic, reproducible testing  
✅ **`evaluate()`** runs agents against datasets with multiple evaluators simultaneously  
✅ **Quality gates** in CI/CD catch regressions before they reach production  

**Previous:** [Custom Observability Setup](./05-custom-observability-setup.md)  
**Next:** [Production Monitoring](./07-production-monitoring.md)

---

## Further Reading

- [LangSmith Evaluation Quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart) — Official evaluation guide
- [Langfuse Evaluation Overview](https://langfuse.com/docs/evaluation/overview) — Open-source evaluation
- [LLM-as-a-Judge Patterns](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) — Judge evaluator design
- [Braintrust Evaluation](https://braintrust.dev/docs/guides/evals/) — Alternative evaluation platform

<!--
Sources Consulted:
- LangSmith evaluation quickstart: https://docs.langchain.com/langsmith/evaluation-quickstart
- Langfuse evaluation overview: https://langfuse.com/docs/evaluation/overview
- LangSmith docs: https://docs.langchain.com/langsmith
-->
