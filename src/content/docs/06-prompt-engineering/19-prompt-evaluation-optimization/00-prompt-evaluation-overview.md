---
title: "Prompt Evaluation & Optimization"
---

# Prompt Evaluation & Optimization

## Introduction

"It seems like it's working" is not an evaluation strategy. LLMs are nondeterministic—the same prompt can produce different outputs, and changes that improve one case may break another. Evaluations (evals) provide structured testing to measure model performance, catch regressions, and systematically improve prompts.

This lesson introduces evaluation fundamentals and how to think about measuring prompt quality.

### What We'll Cover

- Why traditional testing fails for LLMs
- Types of evaluations
- Defining success criteria
- Evaluation metrics and approaches
- The eval-driven development mindset

### Prerequisites

- Understanding of prompt engineering basics (Lessons 1-11)
- Experience with OpenAI/Anthropic APIs

---

## Why Evals Matter

### The Nondeterminism Problem

Unlike traditional software, LLMs introduce variability at multiple points:

| Source | Example |
|--------|---------|
| **User inputs** | Typos, phrasing variations, unexpected formats |
| **Model outputs** | Different wording, structure, or content each run |
| **Tool selection** (agents) | Model chooses different tools or sequences |
| **Agent handoffs** | Routing decisions vary between runs |

Traditional unit tests expect deterministic outputs. LLM applications need **probabilistic evaluation**.

### What Evals Help You Do

```
Without Evals:                    With Evals:
─────────────                     ───────────
"I think it works better"   →    "Accuracy improved from 78% to 91%"
"Let me try this change"    →    "No regression on 500 test cases"
"Users are complaining"     →    "Edge case failure rate: 12%"
"Which model is better?"    →    "GPT-4.1: 94%, Claude: 91%, Gemini: 89%"
```

---

## Types of Evaluations

### Industry Benchmarks

Standardized tests for comparing models in isolation:

- **MMLU** (Massive Multitask Language Understanding)
- **HumanEval** (code generation)
- **GSM8K** (grade school math)
- **HellaSwag** (commonsense reasoning)

> **🤖 AI Context:** Benchmark scores help choose base models, but they don't tell you how a model performs on *your specific task*.

### Standard Metrics

Numerical scores for specific output types:

| Metric | Use Case |
|--------|----------|
| **ROUGE** | Summarization (overlap with reference) |
| **BLEU** | Translation quality |
| **BERTScore** | Semantic similarity |
| **Exact match** | Classification, factual extraction |

### Task-Specific Evals

**This is what you build.** Custom evaluations measuring your application's performance on your data:

```python
# Example: Customer support ticket classification
eval = {
    "name": "Ticket Classification Accuracy",
    "dataset": "500 labeled support tickets",
    "metric": "Exact match on category",
    "target": ">= 95% accuracy"
}
```

---

## Defining Success Criteria

### Start with the End Goal

Before writing any eval, answer:

1. **What does "correct" look like?**
   - Exact match required? (category label)
   - Semantic equivalence? (paraphrase is okay)
   - Subset match? (must include key facts)

2. **What does "good enough" look like?**
   - Accuracy threshold (e.g., 90%+)
   - Latency requirements (e.g., < 2 seconds)
   - Cost constraints (e.g., < $0.01 per query)

3. **What are the failure modes?**
   - Hallucinations
   - Off-topic responses
   - Format violations
   - Safety issues

### Success Criteria Examples

| Task | Success Criteria |
|------|-----------------|
| **Classification** | Exact match with ground truth label ≥ 95% |
| **Summarization** | ROUGE-L ≥ 0.40 AND human coherence score ≥ 80% |
| **Q&A** | Context recall ≥ 0.85, user satisfaction ≥ 70% |
| **Code generation** | Passes test suite AND no security vulnerabilities |
| **Data extraction** | All required fields extracted with ≥ 98% accuracy |

---

## Evaluation Approaches

### Metric-Based Evals

Quantitative scores for automated regression testing:

```python
# Exact match for classification
def exact_match(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0

# String containment
def contains_keyword(output: str, keyword: str) -> float:
    return 1.0 if keyword.lower() in output.lower() else 0.0

# ROUGE for summarization
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
rouge_l = scores['rougeL'].fmeasure
```

**Strengths:** Fast, reproducible, scalable
**Weaknesses:** May miss nuance, not suitable for open-ended tasks

### Human Evals

Expert judgment for quality assessment:

| Approach | When to Use |
|----------|-------------|
| **Quick review** | Skim outputs to sense check |
| **Randomized blind test** | Compare variants without bias |
| **Expert annotation** | Ground truth for training/calibration |

**Best practices:**
- Multiple reviewers for consensus
- Clear rubrics with examples (show "1 star" vs "5 star")
- Include pass/fail threshold, not just scores

**Strengths:** Highest quality signal
**Weaknesses:** Slow, expensive, subjective

### LLM-as-a-Judge

Use a capable model to evaluate outputs:

```python
judge_prompt = """
You are evaluating a customer service response.

Original question: {question}
Agent response: {response}
Reference answer: {reference}

Rate the response on:
1. Accuracy (0-10): Does it answer the question correctly?
2. Helpfulness (0-10): Is it actionable and clear?
3. Tone (0-10): Is it professional and empathetic?

Provide your rating as JSON:
{{"accuracy": X, "helpfulness": X, "tone": X, "explanation": "..."}}
"""
```

**Best practices:**
- Use the most capable model (o3, GPT-4.1) as judge
- Prefer pairwise comparison or pass/fail over absolute scores
- Control for length bias (LLMs prefer longer responses)
- Add reasoning before scoring (improves reliability)

**Strengths:** Scalable, cheaper than humans, handles nuance
**Weaknesses:** Position bias, may not match human preferences perfectly

---

## The Eval Development Mindset

### Behavior-Driven Development for LLMs

Evals are like tests in BDD—specify behavior *before* implementation:

```
1. Define the task       → "Categorize support tickets"
2. Write success criteria → "Match human labels 95%+"
3. Create test dataset   → 500 labeled examples
4. Implement prompt      → System prompt + few-shot examples
5. Run eval              → Current: 82%
6. Iterate               → Improve prompt, rerun eval
7. Repeat until target   → Achieved: 96%
```

### Evals Tips (Do's)

| Practice | Why |
|----------|-----|
| **Evaluate early and often** | Catch issues before they compound |
| **Design task-specific evals** | Generic benchmarks miss your use case |
| **Log everything** | Mine logs for good eval cases |
| **Automate when possible** | Fast feedback loop |
| **Calibrate with humans** | Ensure automated scores match human judgment |

### Evals Anti-Patterns (Don'ts)

| Anti-Pattern | Problem |
|--------------|---------|
| **"Vibes-based" evals** | "It seems better" isn't measurable |
| **Overly generic metrics** | Perplexity doesn't measure task success |
| **Biased datasets** | Overrepresenting one class skews results |
| **Ignoring human feedback** | Automated scores drift from user reality |
| **One-time evals** | No regression detection |

---

## Where to Eval in Your Architecture

### Single-Turn Interactions

```
User Input → Model → Output
     ↓                  ↓
Eval: Instruction following?  Eval: Correct output?
```

### Workflow Architectures

```
Step 1 → Step 2 → Step 3 → Final Output
  ↓        ↓        ↓          ↓
Eval     Eval     Eval       Eval
```

Each step in a chain can be evaluated independently.

### Agent Architectures

| What to Eval | Example Question |
|--------------|------------------|
| **Instruction following** | Does the agent stay on task? |
| **Tool selection** | Did it pick the right tool? |
| **Argument extraction** | Did it pass correct arguments? |
| **Agent handoff** | Did it route to the right specialist? |
| **Final response** | Is the answer correct and complete? |

---

## Building Your First Eval

### Step-by-Step Process

```
1. Define objective     → "Classify tickets into Hardware/Software/Other"
2. Collect dataset      → 100+ labeled examples (diverse, representative)
3. Define metrics       → Exact match accuracy >= 95%
4. Run baseline         → Current prompt achieves 82%
5. Iterate              → Improve prompt, add examples
6. Monitor continuously → Run eval on every prompt change
```

### Dataset Requirements

| Requirement | Guideline |
|-------------|-----------|
| **Size** | Minimum 50-100 examples, ideally 500+ |
| **Diversity** | Cover all categories, edge cases |
| **Balance** | Avoid overrepresenting one class |
| **Quality** | Expert-labeled ground truth |
| **Sources** | Mix of production data + synthetic edge cases |

---

## Hands-on Exercise

### Your Task

Design an eval for a sentiment classification task.

**Scenario:** You're building a product review analyzer that classifies reviews as `positive`, `negative`, or `neutral`.

1. Define 3 success criteria
2. List 5 edge cases to include in your dataset
3. Choose which evaluation approach(es) to use

<details>
<summary>💡 Hints (click to expand)</summary>

- Success criteria: Think accuracy, latency, and edge case handling
- Edge cases: Sarcasm, mixed sentiment, non-English, very short reviews
- Evaluation approach: Start with exact match, add LLM judge for ambiguous cases

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

**Success Criteria:**
1. Exact match accuracy ≥ 90% on labeled test set
2. Latency < 500ms per classification
3. Correct handling of sarcasm in ≥ 80% of cases

**Edge Cases:**
1. Sarcastic reviews ("Great, another broken product!")
2. Mixed sentiment ("Love the features, hate the price")
3. Very short reviews ("Meh", "Ok", "!!!")
4. Non-English reviews or mixed language
5. Reviews with negations ("Not bad", "Couldn't be happier")

**Evaluation Approach:**
- Primary: **Exact match** against human labels (automated, fast)
- Secondary: **LLM-as-judge** for edge cases and ambiguous examples
- Calibration: Periodic **human eval** to ensure automated scores align

</details>

---

## Summary

✅ **Evals are essential** for reliable LLM applications
✅ **Define success criteria** before implementing
✅ **Combine approaches**: metrics + LLM judges + human review
✅ **Eval continuously**, not just once
✅ **Build task-specific evals** that reflect your use case

**Next:** [OpenAI Evals System](./01-openai-evals-system.md)

---

## Further Reading

- [OpenAI Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices)
- [OpenAI Evals Guide](https://platform.openai.com/docs/guides/evals)
- [OpenAI Cookbook: Evals](https://cookbook.openai.com/)

---

<!-- 
Sources Consulted:
- OpenAI Evaluation Best Practices: Types of evals, design process, architecture patterns
- OpenAI Evals Guide: BDD approach, success criteria
- OpenAI Graders: Evaluation approaches
-->
