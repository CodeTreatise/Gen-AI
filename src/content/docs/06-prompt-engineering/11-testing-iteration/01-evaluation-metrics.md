---
title: "Evaluation Metrics"
---

# Evaluation Metrics

## Introduction

Choosing the right metrics is crucial for meaningful prompt evaluation. Different tasks require different measurements—what works for classification won't work for creative writing. This lesson teaches you to select, implement, and interpret evaluation metrics for any prompt engineering task.

> **🔑 Key Insight:** A metric is only useful if it aligns with what you actually care about. High scores on the wrong metric can mislead you into shipping bad prompts.

### What We'll Cover

- Task-specific metric selection
- Automated scoring methods
- LLM-as-judge implementation
- Human evaluation strategies
- Combining metrics for comprehensive evaluation

### Prerequisites

- [Testing & Iteration Overview](./00-testing-iteration-overview.md)
- Understanding of basic statistics (mean, variance)

---

## Task-Specific Metrics

### Classification Tasks

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | Correct / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | False positives are costly |
| **Recall** | TP / (TP + FN) | False negatives are costly |
| **F1 Score** | 2 × (P × R) / (P + R) | Need balance of precision and recall |

```python
def evaluate_classification(predictions: list[str], labels: list[str]) -> dict:
    """Calculate classification metrics."""
    from collections import Counter
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    
    # Per-class metrics
    classes = set(labels)
    per_class = {}
    
    for cls in classes:
        tp = sum(p == l == cls for p, l in zip(predictions, labels))
        fp = sum(p == cls and l != cls for p, l in zip(predictions, labels))
        fn = sum(p != cls and l == cls for p, l in zip(predictions, labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
    
    return {
        "accuracy": correct / total,
        "per_class": per_class
    }
```

**When to use each:**

| Scenario | Metric | Reason |
|----------|--------|--------|
| Spam detection | Precision | False positives (blocking real email) are costly |
| Medical diagnosis | Recall | False negatives (missing disease) are dangerous |
| Multi-label tasks | F1 per class | Need balanced performance across categories |
| Ticket routing | Accuracy | All misroutes have similar cost |

### Text Generation Tasks

| Metric | What It Measures | Use When |
|--------|------------------|----------|
| **ROUGE-L** | Longest common subsequence | Summarization, extraction |
| **ROUGE-1/2** | Unigram/bigram overlap | Content coverage |
| **BLEU** | N-gram precision | Translation quality |
| **METEOR** | Alignment with synonyms | More flexible matching |
| **BERTScore** | Semantic similarity | Meaning over exact words |

```python
from rouge_score import rouge_scorer

def evaluate_summary(generated: str, reference: str) -> dict:
    """Calculate summarization metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(reference, generated)
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }
```

**Interpreting scores:**

| Score Range | Interpretation |
|-------------|----------------|
| ROUGE-L > 0.50 | Strong overlap with reference |
| ROUGE-L 0.30-0.50 | Moderate similarity |
| ROUGE-L < 0.30 | Significant divergence |

> **Note:** ROUGE measures surface overlap, not semantic quality. A paraphrase might score low but be excellent.

### Extraction Tasks

| Metric | What It Measures | Use When |
|--------|------------------|----------|
| **Exact Match** | Perfect string equality | Structured fields (IDs, dates) |
| **Fuzzy Match** | Approximate string similarity | Names, addresses |
| **Entity F1** | Precision/recall on entities | Named entity extraction |
| **Schema Compliance** | Valid structure | JSON/structured output |

```python
from rapidfuzz import fuzz

def evaluate_extraction(extracted: dict, expected: dict) -> dict:
    """Evaluate entity extraction quality."""
    results = {}
    
    for field, expected_value in expected.items():
        extracted_value = extracted.get(field)
        
        if extracted_value is None:
            results[field] = {"match": "missing", "score": 0}
        elif extracted_value == expected_value:
            results[field] = {"match": "exact", "score": 1.0}
        else:
            fuzzy_score = fuzz.ratio(str(extracted_value), str(expected_value)) / 100
            results[field] = {"match": "fuzzy", "score": fuzzy_score}
    
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    return {"fields": results, "overall": avg_score}
```

### Q&A and RAG Tasks

| Metric | What It Measures | Use When |
|--------|------------------|----------|
| **Answer Accuracy** | Correctness of answer | Factual Q&A |
| **Context Recall** | Relevant docs retrieved | RAG retrieval quality |
| **Context Precision** | Retrieved docs relevance | Retrieval efficiency |
| **Faithfulness** | Answer grounded in context | Hallucination detection |

```python
def evaluate_rag_response(
    answer: str,
    expected_answer: str,
    retrieved_contexts: list[str],
    relevant_contexts: list[str]
) -> dict:
    """Evaluate RAG system response."""
    
    # Answer accuracy (using LLM or similarity)
    answer_correct = judge_answer_equivalence(answer, expected_answer)
    
    # Context recall: what % of relevant docs were retrieved
    relevant_retrieved = len(set(retrieved_contexts) & set(relevant_contexts))
    context_recall = relevant_retrieved / len(relevant_contexts) if relevant_contexts else 1.0
    
    # Context precision: what % of retrieved docs were relevant
    context_precision = relevant_retrieved / len(retrieved_contexts) if retrieved_contexts else 0.0
    
    return {
        "answer_accuracy": answer_correct,
        "context_recall": context_recall,
        "context_precision": context_precision
    }
```

### Agent Tasks

| Metric | What It Measures | Use When |
|--------|------------------|----------|
| **Tool Selection Accuracy** | Correct tool chosen | Agent behavior |
| **Argument Precision** | Correct tool arguments | Data extraction |
| **Task Completion Rate** | End goal achieved | Overall success |
| **Handoff Accuracy** | Correct agent routing | Multi-agent systems |

---

## Automated Scoring Methods

### String Check Graders

For exact matching and containment:

```json
{
  "type": "string_check",
  "name": "Contains required keyword",
  "input": "{{ sample.output_text }}",
  "operation": "ilike",
  "reference": "safety warning"
}
```

| Operation | Description | Case Sensitive |
|-----------|-------------|----------------|
| `eq` | Exact match | Yes |
| `neq` | Not equal | Yes |
| `like` | Contains substring | Yes |
| `ilike` | Contains substring | No |

### Text Similarity Graders

For fuzzy and semantic matching:

```json
{
  "type": "text_similarity",
  "name": "Summary quality",
  "input": "{{ sample.output_text }}",
  "reference": "{{ item.reference_summary }}",
  "evaluation_metric": "rouge_l",
  "pass_threshold": 0.4
}
```

| Metric | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `fuzzy_match` | RapidFuzz string similarity | Fast | Character-level |
| `bleu` | N-gram precision | Fast | Surface form |
| `rouge_l` | Longest common subsequence | Fast | Recall-oriented |
| `meteor` | Alignment with synonyms | Medium | Better semantics |
| `cosine` | Embedding similarity | Slower | Deep semantics |

### Python Graders

For complex, custom logic:

```python
def grade(sample: dict, item: dict) -> float:
    """
    Custom grader for multi-field JSON validation.
    
    Args:
        sample: Model output containing output_text, output_json
        item: Test case with expected values
    
    Returns:
        Score between 0 and 1
    """
    import json
    
    output = sample.get("output_json", {})
    expected = item.get("expected_output", {})
    
    if not output:
        # Try parsing output_text as JSON
        try:
            output = json.loads(sample.get("output_text", "{}"))
        except:
            return 0.0
    
    # Check required fields
    required_fields = ["name", "email", "category"]
    field_scores = []
    
    for field in required_fields:
        if field in output and output[field] == expected.get(field):
            field_scores.append(1.0)
        elif field in output:
            field_scores.append(0.5)  # Partial credit for presence
        else:
            field_scores.append(0.0)
    
    return sum(field_scores) / len(field_scores)
```

**Available packages in Python graders:**

```
numpy, scipy, sympy, pandas, rapidfuzz, scikit-learn,
rouge-score, deepdiff, jsonschema, pydantic, pyyaml,
nltk, sqlparse, ast-grep-py
```

---

## LLM-as-Judge Implementation

### When to Use LLM Judges

| Situation | Use LLM Judge | Use Traditional Metrics |
|-----------|---------------|------------------------|
| Open-ended generation | ✅ | ❌ |
| Subjective quality | ✅ | ❌ |
| Exact match needed | ❌ | ✅ |
| High volume, low cost | ❌ | ✅ |
| Complex reasoning | ✅ | ❌ |
| Explainable scores | ✅ | Varies |

### Score Model Grader

```json
{
  "type": "score_model",
  "name": "Response quality judge",
  "model": "gpt-4o",
  "input": [
    {
      "role": "system",
      "content": "You are an expert grader evaluating response quality.\n\nScore 0-1 based on:\n- Accuracy (does it answer correctly?)\n- Completeness (does it cover all aspects?)\n- Clarity (is it well-written?)\n\nProvide reasoning before your score."
    },
    {
      "role": "user",
      "content": "Question: {{ item.question }}\n\nReference answer: {{ item.reference_answer }}\n\nModel response: {{ sample.output_text }}\n\nEvaluate the model response."
    }
  ],
  "range": [0, 1],
  "pass_threshold": 0.7,
  "sampling_params": {
    "temperature": 0,
    "max_completions_tokens": 1024
  }
}
```

### Writing Effective Judge Prompts

**Best practices:**

1. **Define clear criteria** with specific descriptions
2. **Provide scoring examples** at different quality levels
3. **Request reasoning** before the numeric score
4. **Specify the output format** explicitly

```markdown
## Evaluation Criteria

Score the response on a scale of 0-10:

**10 (Excellent):** Perfectly accurate, complete, well-organized, professional tone
**7-9 (Good):** Mostly accurate with minor gaps, clear writing
**4-6 (Adequate):** Partially correct, missing key information, or unclear
**1-3 (Poor):** Significant errors, misleading information, unprofessional
**0 (Fail):** Completely wrong, harmful, or doesn't address the question

## Examples

### Score: 9
Response: [example of excellent response]
Reasoning: Accurate, complete, well-structured...

### Score: 5
Response: [example of adequate response]
Reasoning: Partially correct but missing X, Y, Z...

### Score: 2
Response: [example of poor response]
Reasoning: Contains factual errors about...
```

### Controlling Judge Biases

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers first/last option | Randomize order |
| **Verbosity bias** | Prefers longer responses | Normalize for length |
| **Self-preference** | Prefers own model's outputs | Use different judge model |
| **Anchoring** | Influenced by reference | Blind evaluation |

```python
def debias_pairwise_comparison(
    response_a: str,
    response_b: str,
    judge_prompt: str,
    judge_model: str
) -> dict:
    """Run pairwise comparison with position debiasing."""
    
    # Run comparison in both orders
    result_ab = run_judge(judge_prompt, response_a, response_b, judge_model)
    result_ba = run_judge(judge_prompt, response_b, response_a, judge_model)
    
    # Count wins accounting for position
    a_wins = 0
    b_wins = 0
    
    if result_ab["winner"] == "first":
        a_wins += 1
    elif result_ab["winner"] == "second":
        b_wins += 1
    
    if result_ba["winner"] == "second":
        a_wins += 1
    elif result_ba["winner"] == "first":
        b_wins += 1
    
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "verdict": "A" if a_wins > b_wins else "B" if b_wins > a_wins else "tie"
    }
```

---

## Human Evaluation Strategies

### When Human Evaluation is Essential

- **Subjective quality** (tone, style, user experience)
- **Safety-critical** applications
- **Calibrating** automated metrics
- **Novel tasks** without established benchmarks
- **Final validation** before production

### Evaluation Interface Design

```python
class HumanEvalTask:
    """Structure for human evaluation tasks."""
    
    def __init__(
        self,
        task_id: str,
        prompt: str,
        response: str,
        criteria: list[dict]
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.response = response
        self.criteria = criteria  # List of {"name": str, "description": str, "scale": (min, max)}
    
    def to_interface(self) -> dict:
        """Generate evaluation interface structure."""
        return {
            "task_id": self.task_id,
            "context": {
                "prompt": self.prompt,
                "response": self.response
            },
            "ratings": [
                {
                    "criterion": c["name"],
                    "description": c["description"],
                    "scale": c["scale"]
                }
                for c in self.criteria
            ],
            "free_text": {
                "label": "Additional comments",
                "required": False
            }
        }
```

### Aggregating Human Ratings

| Method | When to Use | Formula |
|--------|-------------|---------|
| **Mean** | Normal distribution | Sum / Count |
| **Median** | Outlier resistance | Middle value |
| **Majority vote** | Classification | Most common |
| **Weighted** | Expertise-based | Expert × Weight |
| **Inter-rater agreement** | Reliability check | Cohen's Kappa |

```python
from scipy import stats

def calculate_inter_rater_reliability(ratings: list[list[int]]) -> dict:
    """
    Calculate agreement between raters.
    
    Args:
        ratings: List of ratings per item [[rater1, rater2, ...], ...]
    """
    # Fleiss' Kappa for multiple raters
    # Simplified: calculate pairwise agreement
    n_items = len(ratings)
    n_raters = len(ratings[0])
    
    agreements = 0
    comparisons = 0
    
    for item_ratings in ratings:
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                comparisons += 1
                if item_ratings[i] == item_ratings[j]:
                    agreements += 1
    
    agreement_rate = agreements / comparisons if comparisons > 0 else 0
    
    return {
        "agreement_rate": agreement_rate,
        "interpretation": interpret_kappa(agreement_rate)
    }

def interpret_kappa(kappa: float) -> str:
    if kappa > 0.8:
        return "Almost perfect agreement"
    elif kappa > 0.6:
        return "Substantial agreement"
    elif kappa > 0.4:
        return "Moderate agreement"
    elif kappa > 0.2:
        return "Fair agreement"
    else:
        return "Poor agreement"
```

---

## Combining Metrics

### Multigrader Configuration

```json
{
  "type": "multi",
  "graders": {
    "accuracy": {
      "type": "string_check",
      "input": "{{ sample.output_json.answer }}",
      "reference": "{{ item.correct_answer }}",
      "operation": "eq"
    },
    "format": {
      "type": "python",
      "source": "def grade(sample, item):\n    import json\n    try:\n        json.loads(sample['output_text'])\n        return 1.0\n    except:\n        return 0.0"
    },
    "quality": {
      "type": "score_model",
      "model": "gpt-4o",
      "input": [{"role": "user", "content": "Rate 0-1: {{ sample.output_text }}"}],
      "range": [0, 1]
    }
  },
  "calculate_output": "(accuracy * 0.5) + (format * 0.2) + (quality * 0.3)"
}
```

### Weighting Strategies

| Strategy | Formula | Use When |
|----------|---------|----------|
| **Equal weight** | (A + B + C) / 3 | All metrics equally important |
| **Priority weight** | A × 0.6 + B × 0.3 + C × 0.1 | Clear priority order |
| **Pass/fail gate** | A × B × C (binary) | All must pass |
| **Best of** | max(A, B, C) | Any success counts |

---

## Hands-on Exercise

### Your Task

Design an evaluation suite for a customer service chatbot that:
1. Classifies customer intent
2. Generates helpful responses
3. Follows company guidelines

### Requirements

1. Select appropriate metrics for each capability
2. Define at least one automated grader
3. Design an LLM-as-judge prompt
4. Create human evaluation criteria
5. Specify a multigrader combining all metrics

<details>
<summary>💡 Hints (click to expand)</summary>

- Classification: accuracy, per-class F1
- Response quality: LLM judge for helpfulness, tone
- Guidelines: string_check for required/forbidden content
- Combine with priority weighting (safety > accuracy > quality)

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```json
{
  "type": "multi",
  "graders": {
    "intent_accuracy": {
      "type": "string_check",
      "name": "Intent classification",
      "input": "{{ sample.output_json.intent }}",
      "reference": "{{ item.expected_intent }}",
      "operation": "eq"
    },
    "forbidden_content": {
      "type": "python",
      "source": "def grade(sample, item):\n    forbidden = ['competitor_name', 'profanity', 'promise_refund']\n    text = sample.get('output_text', '').lower()\n    for word in forbidden:\n        if word in text:\n            return 0.0\n    return 1.0"
    },
    "response_quality": {
      "type": "score_model",
      "model": "gpt-4o",
      "input": [
        {
          "role": "system",
          "content": "Evaluate customer service response quality.\n\nCriteria:\n- Helpful (addresses the issue)\n- Empathetic (acknowledges customer feelings)\n- Professional (appropriate tone)\n- Actionable (provides clear next steps)\n\nScore 0-1 where 1 is excellent on all criteria."
        },
        {
          "role": "user",
          "content": "Customer message: {{ item.customer_message }}\n\nAgent response: {{ sample.output_text }}"
        }
      ],
      "range": [0, 1],
      "pass_threshold": 0.7
    }
  },
  "calculate_output": "min(forbidden_content, (intent_accuracy * 0.3) + (response_quality * 0.7))"
}
```

**Human evaluation criteria:**

| Criterion | Description | Scale |
|-----------|-------------|-------|
| Helpfulness | Does the response solve the customer's problem? | 1-5 |
| Empathy | Does the response acknowledge customer emotions? | 1-5 |
| Professionalism | Is the tone appropriate for our brand? | 1-5 |
| Completeness | Does it provide all necessary information? | 1-5 |

**Multigrader logic:**
- `forbidden_content` acts as a gate (0 = auto-fail)
- Intent and quality weighted (30/70)
- Overall pass threshold: 0.7

</details>

---

## Summary

✅ Choose metrics that align with what you actually care about
✅ Classification: accuracy, precision, recall, F1 based on error costs
✅ Generation: ROUGE, BLEU for overlap; LLM judge for quality
✅ LLM judges: control for position/verbosity bias, request reasoning
✅ Human eval: essential for subjective quality, safety, calibration
✅ Combine metrics with weighted multigraders for comprehensive evaluation

**Next:** [A/B Testing Prompts](./02-ab-testing-prompts.md)

---

## Further Reading

- [OpenAI Graders Documentation](https://platform.openai.com/docs/guides/graders) - All grader types
- [ROUGE Score Explained](https://aclanthology.org/W04-1013/) - Original paper
- [G-Eval](https://arxiv.org/abs/2303.16634) - LLM-as-judge methodology

---

<!-- 
Sources Consulted:
- OpenAI Graders: string_check, text_similarity, score_model, python graders
- OpenAI Evaluation Best Practices: LLM-as-judge recommendations
- ROUGE, BLEU scoring documentation
-->
