---
title: "Testing & Iteration Overview"
---

# Testing & Iteration Overview

## Introduction

LLMs are nondeterministic—the same prompt can produce different outputs. This makes traditional software testing insufficient for AI applications. Evaluations (evals) are structured tests that measure model performance despite this variability. This lesson teaches you to build evaluation systems that ensure your prompts work reliably in production.

> **🔑 Key Insight:** Prompt engineering without testing is guessing. Systematic evaluation transforms prompt development from art into engineering.

### What We'll Cover

- Eval-driven development methodology
- Types of evaluators and graders
- The complete evaluation workflow
- When to use different testing strategies

### Prerequisites

- [Common Pitfalls & Solutions](../10-common-pitfalls-solutions/00-common-pitfalls-overview.md)
- Basic understanding of prompt engineering concepts
- Experience writing prompts for production use

---

## Why Prompt Testing Matters

### The Nondeterminism Problem

```
┌──────────────────────────────────────────────────────────────┐
│              SOURCES OF NONDETERMINISM                        │
└──────────────────────────────────────────────────────────────┘

    INPUTS                    MODEL                    OUTPUTS
    ──────                    ─────                    ───────
    
    User prompts              Sampling                 Generated
    vary widely               (temperature)            responses
    
    Developer                 Model updates            Tool calls
    instructions              and changes              selected
    
    Context data              Token selection          Agent
    changes                   randomness               handoffs
```

| Architecture | Nondeterminism Sources | What to Evaluate |
|--------------|------------------------|------------------|
| **Single-turn** | User input, model sampling | Instruction following, output correctness |
| **Workflows** | Multiple model calls | Each step independently + end-to-end |
| **Single-agent** | Tool selection, execution | Tool choice, argument extraction |
| **Multi-agent** | Agent handoffs | Handoff accuracy, coordination |

---

## Eval-Driven Development

Like test-driven development (TDD) for code, eval-driven development starts with specifying expected behavior before implementing prompts.

### The Eval-Driven Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              EVAL-DRIVEN DEVELOPMENT CYCLE                   │
└─────────────────────────────────────────────────────────────┘

    1. DEFINE              2. COLLECT            3. SPECIFY
    ────────               ───────────           ─────────
    Eval objective         Test dataset          Eval metrics
    "What should           "Representative       "How do we
    the model do?"         inputs/outputs"       measure success?"
    
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  4. RUN     │
                         │  ─────────  │
                         │  Execute    │
                         │  evals      │
                         └─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
         5. ANALYZE        6. ITERATE        7. CONTINUOUS
         ──────────        ──────────        ──────────────
         Review            Improve           Monitor
         results           prompt            production
```

### Best Practices

| Practice | Description |
|----------|-------------|
| **Evaluate early and often** | Write scoped tests at every stage |
| **Design task-specific evals** | Tests should reflect real-world distributions |
| **Log everything** | Mine logs for good eval cases |
| **Automate when possible** | Structure evals for automated scoring |
| **Maintain human agreement** | Calibrate automated scoring against human evals |

### Anti-Patterns to Avoid

| Anti-Pattern | Problem |
|--------------|---------|
| Overly generic metrics | Relying solely on perplexity or BLEU score |
| Biased eval design | Datasets that don't match production traffic |
| Vibe-based evals | "It seems like it's working" as evaluation |
| Ignoring human feedback | Not calibrating against human judgment |
| Waiting until production | No evals until after shipping |

---

## Types of Evaluators

### 1. Metric-Based Evaluators

Quantitative scores for automated regression testing:

| Metric Type | Use Case | Example |
|-------------|----------|---------|
| **Exact match** | Classification tasks | Label matches expected |
| **String check** | Contains/equals | Output includes required text |
| **ROUGE/BLEU** | Text similarity | Summarization quality |
| **Fuzzy match** | Approximate matching | Names, entities |
| **Cosine similarity** | Semantic similarity | Embedding comparison |

```json
{
  "type": "string_check",
  "name": "Match output to human label",
  "input": "{{ sample.output_text }}",
  "operation": "eq",
  "reference": "{{ item.correct_label }}"
}
```

### 2. LLM-as-Judge Evaluators

Use models to grade outputs—more scalable than human evaluation:

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Pairwise comparison** | Compare two responses | A/B testing |
| **Single answer grading** | Score response in isolation | Quality assessment |
| **Reference-guided** | Compare against gold standard | Accuracy testing |

```json
{
  "type": "score_model",
  "name": "Quality grader",
  "model": "gpt-4o",
  "input": [
    {
      "role": "system",
      "content": "Grade the response quality 0-1. Consider accuracy, completeness, and clarity."
    },
    {
      "role": "user", 
      "content": "Reference: {{ item.reference_answer }}. Response: {{ sample.output_text }}"
    }
  ],
  "range": [0, 1],
  "pass_threshold": 0.7
}
```

> **Warning:** LLM judges have biases—position bias (response order) and verbosity bias (preferring longer responses). Control for these in eval design.

### 3. Human Evaluators

Highest quality but expensive and slow:

| Method | Description | Use When |
|--------|-------------|----------|
| **Quick review** | Skim outputs for obvious issues | Early development |
| **Randomized blind test** | Rate outputs without knowing source | Comparing prompts |
| **Detailed scoring** | Multi-criteria evaluation | Critical applications |

**Tips for human evaluation:**
- Multiple rounds of detailed review to refine scorecard
- "Show rather than tell" with examples of different score levels
- Include pass/fail threshold in addition to numerical score
- Aggregate multiple reviewers using consensus votes

### 4. Python/Custom Evaluators

Programmatic evaluation for complex logic:

```python
def grade(sample: dict, item: dict) -> float:
    """
    Custom grader for structured output validation.
    Returns score 0-1 based on JSON schema compliance.
    """
    output_json = sample.get("output_json", {})
    required_fields = item.get("required_fields", [])
    
    if not output_json:
        return 0.0
    
    present = sum(1 for f in required_fields if f in output_json)
    return present / len(required_fields) if required_fields else 1.0
```

---

## Grader Types Reference

| Grader Type | Output | Use Case |
|-------------|--------|----------|
| **string_check** | 0 or 1 | Exact match, contains, starts with |
| **text_similarity** | 0 to 1 | ROUGE, BLEU, cosine similarity |
| **score_model** | Numeric range | LLM-as-judge with custom criteria |
| **python** | Float | Custom logic, schema validation |
| **multi** | Combined score | Multiple criteria weighted |

### Multigrader Example

Combine multiple graders for comprehensive evaluation:

```json
{
  "type": "multi",
  "graders": {
    "accuracy": {
      "type": "string_check",
      "input": "{{ sample.output_json.category }}",
      "reference": "{{ item.expected_category }}",
      "operation": "eq"
    },
    "completeness": {
      "type": "text_similarity",
      "input": "{{ sample.output_text }}",
      "reference": "{{ item.reference_response }}",
      "evaluation_metric": "rouge_l",
      "pass_threshold": 0.6
    }
  },
  "calculate_output": "(accuracy * 0.7) + (completeness * 0.3)"
}
```

---

## The Evaluation Workflow

### Step 1: Define Eval Objective

Be specific about success criteria:

| Task | Good Objective | Poor Objective |
|------|----------------|----------------|
| Classification | "Match human labels with 95% accuracy" | "Classify correctly" |
| Summarization | "ROUGE-L > 0.40, coherence > 80%" | "Good summaries" |
| Q&A | "Answer accuracy > 90%, citation recall > 85%" | "Answer questions well" |

### Step 2: Collect Dataset

| Source | When to Use | Considerations |
|--------|-------------|----------------|
| **Production data** | Real traffic patterns | Privacy, labeling effort |
| **Synthetic data** | Edge cases, coverage | May miss real patterns |
| **Expert-created** | Ground truth | Expensive, limited scale |
| **Historical logs** | Regression testing | Needs cleaning |

```jsonl
{"item": {"input": "What's 2+2?", "expected": "4"}}
{"item": {"input": "Capital of France?", "expected": "Paris"}}
{"item": {"input": "Explain quantum computing", "expected": "..."}}
```

### Step 3: Define Metrics

Match metrics to task requirements:

| Task | Recommended Metrics |
|------|---------------------|
| **Classification** | Accuracy, precision, recall, F1 |
| **Generation** | ROUGE, BLEU, human rating |
| **Extraction** | Exact match, F1 on extracted entities |
| **Q&A** | Answer accuracy, context recall |
| **Agents** | Tool selection accuracy, task completion |

### Step 4: Run Evals

```python
import openai

# Create eval definition
eval_config = {
    "name": "Ticket Classification",
    "data_source_config": {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "ticket_text": {"type": "string"},
                "correct_label": {"type": "string"}
            }
        }
    },
    "testing_criteria": [{
        "type": "string_check",
        "name": "Label accuracy",
        "input": "{{ sample.output_text }}",
        "operation": "eq",
        "reference": "{{ item.correct_label }}"
    }]
}

# Run eval with prompt template
run_config = {
    "data_source": {
        "type": "responses",
        "model": "gpt-4o",
        "input_messages": {
            "type": "template",
            "template": [
                {"role": "developer", "content": "Classify ticket as: Hardware, Software, Other"},
                {"role": "user", "content": "{{ item.ticket_text }}"}
            ]
        },
        "source": {"type": "file_id", "id": "file-xxx"}
    }
}
```

### Step 5: Analyze Results

```json
{
  "status": "completed",
  "result_counts": {
    "total": 100,
    "passed": 93,
    "failed": 5,
    "errored": 2
  },
  "per_testing_criteria_results": [
    {
      "testing_criteria": "Label accuracy",
      "passed": 93,
      "failed": 7
    }
  ]
}
```

**Questions to ask:**
- Which test cases failed and why?
- Are failures systematic or random?
- What edge cases need addressing?
- Is the eval itself accurate?

---

## Lesson Navigation

This lesson is organized into focused topics:

| Lesson | Topic | Key Skills |
|--------|-------|------------|
| [01-evaluation-metrics.md](./01-evaluation-metrics.md) | Evaluation Metrics | Task-specific metrics, automated scoring, LLM-as-judge |
| [02-ab-testing-prompts.md](./02-ab-testing-prompts.md) | A/B Testing Prompts | Test design, statistical significance, winner determination |
| [03-edge-case-testing.md](./03-edge-case-testing.md) | Edge Case Testing | Adversarial inputs, boundary conditions, failure modes |
| [04-regression-testing.md](./04-regression-testing.md) | Regression Testing | Golden datasets, CI/CD integration, change detection |
| [05-documentation-versioning.md](./05-documentation-versioning.md) | Documentation & Versioning | Prompt documentation, semantic versioning, migration |

---

## Quick Reference: When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| New prompt development | Start with few manual tests, add automated as patterns emerge |
| Comparing two prompts | A/B testing with statistical significance |
| Before production deploy | Full regression suite + edge case testing |
| After production issues | Root cause analysis + add failing case to test suite |
| Model upgrade | Full regression + comparison testing |
| Continuous monitoring | Automated regression + sample human review |

---

## Summary

✅ Evals are structured tests for measuring model performance despite nondeterminism
✅ Eval-driven development: specify behavior before implementing prompts
✅ Four evaluator types: metric-based, LLM-as-judge, human, and custom Python
✅ Match metrics to task requirements (accuracy for classification, ROUGE for summarization)
✅ The workflow: define objective → collect data → specify metrics → run → analyze → iterate

**Next:** [Evaluation Metrics](./01-evaluation-metrics.md)

---

## Further Reading

- [OpenAI Evals Guide](https://platform.openai.com/docs/guides/evals) - Creating and running evals
- [OpenAI Graders](https://platform.openai.com/docs/guides/graders) - Grader types and configuration
- [OpenAI Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices) - Design principles

---

<!-- 
Sources Consulted:
- OpenAI Evals Guide: Eval creation workflow, grader types
- OpenAI Graders: String check, text similarity, model graders, Python graders
- OpenAI Evaluation Best Practices: Eval-driven development, anti-patterns
- Google Gemini Prompt Strategies: Iteration strategies
-->
