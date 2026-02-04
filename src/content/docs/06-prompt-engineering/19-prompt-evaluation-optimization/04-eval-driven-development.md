---
title: "Eval-Driven Development"
---

# Eval-Driven Development

## Introduction

Eval-driven development (EDD) treats prompts like code: specify behavior first, then implement. It's BDD (Behavior-Driven Development) for LLM applications. You define what success looks like, create tests to measure it, then iterate until you pass.

This lesson covers the methodology for systematic prompt development.

### What We'll Cover

- The EDD mindset
- Writing criteria before prompts
- Building test datasets
- Iteration workflows
- Regression testing
- Continuous evaluation

### Prerequisites

- [Prompt Optimizer](./03-prompt-optimizer.md)
- Understanding of test-driven development concepts

---

## The EDD Mindset

### Traditional Prompt Development

```
1. Write prompt
2. Try a few examples
3. "Seems to work"
4. Deploy
5. Users complain
6. Panic fix
```

### Eval-Driven Development

```
1. Define success criteria
2. Create test dataset
3. Write initial prompt
4. Run eval → 65%
5. Analyze failures
6. Improve prompt
7. Run eval → 82%
8. Repeat until target
9. Deploy with confidence
10. Monitor with continuous eval
```

### Why EDD Works

| Benefit | How It Helps |
|---------|--------------|
| **Measurable progress** | "85% → 92%" vs "seems better" |
| **Catch regressions** | Changes that break existing cases |
| **Document requirements** | Test cases = specification |
| **Faster iteration** | Immediate feedback on changes |
| **Confidence to deploy** | Known performance level |

---

## Write Criteria First

### Before Any Prompt

Answer these questions:

1. **What does correct output look like?**
   - Exact match? Semantic equivalence? Contains key info?
   
2. **What are the failure modes?**
   - Hallucinations, wrong format, missing info, wrong tone
   
3. **What's the acceptance threshold?**
   - 90%? 95%? 99%?
   
4. **How will you measure it?**
   - Which graders? Human review? Both?

### Example: Email Classifier

**Before writing the prompt:**

```yaml
Task: Classify support emails into categories

Success Criteria:
  - Accuracy: >= 95% exact match with human labels
  - Latency: < 500ms per classification
  - Categories: Billing, Technical, General, Spam

Failure Modes:
  - Wrong category assignment
  - Refusing to classify
  - Multiple categories when one required

Measurement:
  - string_check grader for exact match
  - 200+ labeled test emails
  - Include edge cases: sarcastic emails, mixed topics
```

### Criteria Categories

| Category | Examples |
|----------|----------|
| **Functional** | Correct output, complete information |
| **Format** | JSON structure, word count, sections |
| **Safety** | No harmful content, no PII leakage |
| **Quality** | Tone, clarity, helpfulness |
| **Performance** | Latency, cost per query |

---

## Building Test Datasets

### Dataset Sources

| Source | Pros | Cons |
|--------|------|------|
| **Production logs** | Real distribution | May lack labels |
| **Human-created** | High quality labels | Time-consuming |
| **Synthetic (LLM)** | Scalable | May not match real data |
| **Purchased** | Domain expertise | Expensive |

### Dataset Composition

```
Total: 200 examples

By class:
├── Billing: 50 (25%)
├── Technical: 60 (30%)
├── General: 50 (25%)
└── Spam: 40 (20%)

By difficulty:
├── Easy (obvious): 100 (50%)
├── Medium (typical): 60 (30%)
└── Hard (edge cases): 40 (20%)

Split:
├── Dev set: 50 (for iteration)
├── Test set: 150 (for final eval)
```

### Creating Edge Cases

Use an LLM to generate challenging examples:

```python
edge_case_prompt = """
Generate 10 challenging support emails that would be hard to classify.

Categories: Billing, Technical, General, Spam

Include:
- Emails that mention multiple categories but have one primary intent
- Sarcastic or frustrated customers
- Very short emails (1-2 sentences)
- Emails in broken English
- Emails that could be spam but might be legitimate

Format as JSONL with "email_text" and "correct_category" fields.
"""
```

### Labeling Guidelines

Create clear rules for labelers:

```markdown
## Classification Rules

**Billing**: Primary intent is about payments, invoices, subscriptions, pricing
- "Why was I charged $50?" → Billing
- "Cancel my subscription" → Billing

**Technical**: Primary intent is about product functionality, bugs, how-to
- "App keeps crashing" → Technical
- "How do I export data?" → Technical

**General**: Inquiries, feedback, or requests not fitting other categories
- "Do you have a referral program?" → General
- "Great product!" → General

**Spam**: Promotional, phishing, or irrelevant
- "URGENT: Claim your prize" → Spam

**Edge Case Rules**:
- If billing AND technical: Choose based on what resolution they need
- If unclear: Choose General
```

---

## The Iteration Workflow

### Step-by-Step Process

```
┌──────────────────────────────────────────────────────────┐
│  1. BASELINE                                              │
│     Write initial prompt                                  │
│     Run eval on dev set                                   │
│     Record: Accuracy = 72%                                │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  2. ANALYZE FAILURES                                      │
│     Group failures by type                                │
│     Most common: "Mixed topic emails" (15 failures)       │
│     Second: "Sarcastic tone" (8 failures)                 │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  3. TARGETED IMPROVEMENT                                  │
│     Add to prompt: "For emails with multiple topics,      │
│     classify by the primary intent..."                    │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  4. RERUN EVAL                                            │
│     Accuracy = 84%                                        │
│     Check: Did any previously passing cases now fail?     │
│     If yes: Address regression                            │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  5. REPEAT until target (95%)                             │
│     Round 1: 72% → 84%                                    │
│     Round 2: 84% → 89%                                    │
│     Round 3: 89% → 93%                                    │
│     Round 4: 93% → 96% ✅                                 │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  6. FINAL VALIDATION                                      │
│     Run on held-out test set                              │
│     Test set accuracy: 94% (close to dev, no overfit)     │
└──────────────────────────────────────────────────────────┘
```

### Failure Analysis

Categorize failures to prioritize fixes:

```python
# After running eval
failures = [r for r in results if not r["passed"]]

# Group by failure type
failure_types = {}
for f in failures:
    predicted = f["output"]
    expected = f["expected"]
    key = f"{expected} → {predicted}"
    failure_types[key] = failure_types.get(key, 0) + 1

# Sort by frequency
sorted_failures = sorted(failure_types.items(), key=lambda x: -x[1])

print("Top failure patterns:")
for pattern, count in sorted_failures[:5]:
    print(f"  {pattern}: {count} cases")
```

**Output:**
```
Top failure patterns:
  Technical → Billing: 8 cases
  General → Technical: 5 cases
  Billing → General: 4 cases
  Spam → General: 3 cases
  Technical → General: 2 cases
```

### Prompt Improvement Strategies

| Failure Pattern | Improvement Strategy |
|-----------------|---------------------|
| Confusion between classes | Add explicit differentiation rules |
| Edge cases failing | Add specific instructions for edge cases |
| Format issues | Add output format examples |
| Missing information | Add "always include X" instructions |
| Hallucinations | Add "only use information provided" |

---

## Regression Testing

### What Is Regression?

A change that fixes new cases but breaks previously passing cases.

```
Before change:
  Case A: ✅ Pass
  Case B: ❌ Fail
  Case C: ✅ Pass

After "fix" for Case B:
  Case A: ❌ REGRESSION!
  Case B: ✅ Pass
  Case C: ✅ Pass
```

### Detecting Regressions

```python
def check_regressions(old_results, new_results):
    """Compare eval runs to detect regressions."""
    
    old_pass = {r["id"] for r in old_results if r["passed"]}
    new_pass = {r["id"] for r in new_results if r["passed"]}
    
    regressions = old_pass - new_pass  # Was passing, now failing
    improvements = new_pass - old_pass  # Was failing, now passing
    
    return {
        "regressions": list(regressions),
        "improvements": list(improvements),
        "net_change": len(improvements) - len(regressions)
    }

# Usage
comparison = check_regressions(baseline_results, new_results)
print(f"Improvements: {len(comparison['improvements'])}")
print(f"Regressions: {len(comparison['regressions'])}")
print(f"Net change: {comparison['net_change']}")
```

### Handling Regressions

| Scenario | Action |
|----------|--------|
| Improvements > Regressions | Accept if regressions are minor |
| Regressions are critical | Revise change to preserve critical cases |
| Trade-off unavoidable | Document and make conscious choice |
| Many regressions | Rollback and try different approach |

### Golden Set

Maintain a "golden set" of critical cases that must always pass:

```python
GOLDEN_SET = [
    {"id": "critical_1", "input": "...", "expected": "..."},
    {"id": "critical_2", "input": "...", "expected": "..."},
    # Cases that absolutely must work
]

def validate_golden_set(prompt):
    """Run golden set and fail if any case regresses."""
    for case in GOLDEN_SET:
        result = run_prompt(prompt, case["input"])
        if result != case["expected"]:
            raise Exception(f"Golden set failure: {case['id']}")
    return True
```

---

## Continuous Evaluation

### Beyond One-Time Testing

```
Development      Production        Continuous
───────────      ──────────        ──────────
Dev set eval  →  Deploy      →    Monitor live traffic
                              →    Log outputs
                              →    Sample for review
                              →    Detect drift
                              →    Alert on failures
```

### Continuous Eval Strategies

| Strategy | How It Works |
|----------|--------------|
| **Stored completions** | Log all production outputs |
| **Sampling** | Evaluate random sample daily/weekly |
| **User feedback** | Track thumbs up/down |
| **Automated monitoring** | Run graders on live outputs |
| **A/B testing** | Compare prompt variants in production |

### Monitoring Dashboard Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy over time | Detect performance drift |
| Failure rate by type | Identify emerging failure patterns |
| Latency distribution | Performance monitoring |
| User satisfaction | Thumbs up/down ratio |
| Cost per query | Budget tracking |

### Alerting

Set up alerts for:

```yaml
Alerts:
  - name: "Accuracy drop"
    condition: "accuracy < 90% over 1 hour"
    action: "Page on-call"
    
  - name: "New failure type"
    condition: "Unknown failure category detected"
    action: "Slack notification"
    
  - name: "Cost spike"
    condition: "Cost > 2x average"
    action: "Email team"
```

---

## EDD in Practice

### Project Structure

```
project/
├── prompts/
│   ├── classifier_v1.txt
│   ├── classifier_v2.txt      # Current
│   └── classifier_v3.txt      # Testing
├── evals/
│   ├── config.yaml
│   ├── graders/
│   │   ├── exact_match.py
│   │   └── format_check.py
│   └── datasets/
│       ├── dev_set.jsonl
│       ├── test_set.jsonl
│       └── golden_set.jsonl
├── results/
│   ├── 2024-01-15_v2_baseline.json
│   └── 2024-01-16_v3_experiment.json
└── scripts/
    ├── run_eval.py
    ├── compare_results.py
    └── check_regressions.py
```

### Version Control for Prompts

Track prompts like code:

```python
# prompts/classifier_v2.txt

# Version: 2.0
# Date: 2024-01-15
# Author: team
# Accuracy: 94% on test set
# Changes: Added multi-topic handling, improved spam detection

You are an email classifier...
```

### Eval Automation

```python
# scripts/run_eval.py

import json
from datetime import datetime

def run_eval(prompt_file: str, dataset_file: str):
    """Run evaluation and save results."""
    
    # Load prompt and dataset
    prompt = open(prompt_file).read()
    dataset = [json.loads(l) for l in open(dataset_file)]
    
    # Run eval
    results = []
    for case in dataset:
        output = call_model(prompt, case["input"])
        passed = output == case["expected"]
        results.append({
            "id": case["id"],
            "input": case["input"],
            "expected": case["expected"],
            "output": output,
            "passed": passed
        })
    
    # Calculate metrics
    accuracy = sum(r["passed"] for r in results) / len(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = f"results/{timestamp}_{prompt_file.stem}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "prompt_file": prompt_file,
            "dataset_file": dataset_file,
            "timestamp": timestamp,
            "accuracy": accuracy,
            "total": len(results),
            "passed": sum(r["passed"] for r in results),
            "failed": sum(not r["passed"] for r in results),
            "results": results
        }, f, indent=2)
    
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Results saved to: {output_file}")
    
    return accuracy, results
```

---

## Hands-on Exercise

### Your Task

Design an EDD workflow for a content moderation classifier:

1. Define success criteria (3+ metrics)
2. Plan your test dataset composition
3. Create a regression testing strategy
4. Outline your iteration workflow

<details>
<summary>✅ Solution (click to expand)</summary>

**1. Success Criteria:**
```yaml
Metrics:
  - Accuracy: >= 98% (high stakes - must catch harmful content)
  - False positive rate: < 2% (avoid over-moderation)
  - False negative rate: < 0.5% (CRITICAL - must catch harmful)
  - Latency: < 200ms (real-time moderation)
  
Categories:
  - Safe
  - Hate speech
  - Violence
  - Sexual content
  - Spam
```

**2. Dataset Composition:**
```yaml
Total: 500 examples

By class:
  Safe: 250 (50%)           # Most content is safe
  Hate speech: 75 (15%)
  Violence: 50 (10%)
  Sexual: 50 (10%)
  Spam: 75 (15%)

By difficulty:
  Clear violations: 200 (40%)
  Borderline cases: 200 (40%)   # Critical for false positives
  Adversarial: 100 (20%)        # Attempts to evade detection

Sources:
  - Synthetic (LLM-generated): 200
  - Real production samples: 200
  - Purchased adversarial set: 100
```

**3. Regression Testing Strategy:**
```python
# Golden set: Cases that MUST always pass
GOLDEN_SET = [
    # Clear harmful content (must catch)
    {"id": "hate_1", "input": "...", "expected": "Hate speech"},
    {"id": "violence_1", "input": "...", "expected": "Violence"},
    
    # Clear safe content (must not flag)
    {"id": "safe_1", "input": "News article about election", "expected": "Safe"},
    {"id": "safe_2", "input": "Recipe discussion", "expected": "Safe"},
    
    # Historical false positives (must stay fixed)
    {"id": "fp_fix_1", "input": "Health discussion", "expected": "Safe"},
]

# Regression rules
REGRESSION_RULES = {
    "false_negative_regression": "BLOCK - Cannot ship",
    "false_positive_regression": "REVIEW - May accept if improvements > 3x",
    "golden_set_failure": "BLOCK - Must pass 100%"
}
```

**4. Iteration Workflow:**
```
Week 1: Foundation
  - Create initial prompt
  - Build dataset (200 examples)
  - Run baseline eval: 89%
  - Identify top 3 failure patterns

Week 2: First Improvement Cycle
  - Address: Borderline hate speech (top failure)
  - Add: Explicit examples in prompt
  - Run eval: 93%
  - Check regressions: 2 new FPs → fix

Week 3: Edge Case Hardening
  - Address: Adversarial evasion attempts
  - Add: Common evasion pattern detection
  - Run eval: 96%
  - Check regressions: None

Week 4: Final Validation
  - Expand dataset to 500
  - Run on held-out test set: 97%
  - Run golden set: 100%
  - Document and deploy

Ongoing: Continuous Eval
  - Sample 1% of production traffic daily
  - Human review weekly
  - Alert on false negative rate > 0.3%
```

</details>

---

## Summary

✅ **Define criteria before writing prompts**
✅ **Build diverse test datasets** with edge cases
✅ **Iterate systematically**: eval → analyze → improve → repeat
✅ **Watch for regressions** on every change
✅ **Maintain a golden set** of critical cases
✅ **Continue evaluating** in production

**Next:** [A/B Testing and Debugging](./05-ab-testing-debugging.md)

---

## Further Reading

- [OpenAI Cookbook: Detecting Prompt Regressions](https://cookbook.openai.com/)
- [OpenAI Cookbook: Bulk Experimentation](https://cookbook.openai.com/)
- [OpenAI Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices)

---

<!-- 
Sources Consulted:
- OpenAI Evaluation Best Practices: EDD workflow, architecture patterns
- OpenAI Evals Guide: BDD approach, continuous evaluation
- OpenAI Cookbook: Regression detection, monitoring
-->
