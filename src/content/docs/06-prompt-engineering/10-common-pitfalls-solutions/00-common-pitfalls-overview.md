---
title: "Common Pitfalls & Solutions Overview"
---

# Common Pitfalls & Solutions Overview

## Introduction

Even experienced prompt engineers make mistakes. The difference between beginners and experts isn't avoiding mistakes—it's recognizing them quickly and knowing how to fix them. This lesson covers the most common prompt engineering pitfalls, why they happen, and systematic approaches to diagnose and resolve them.

> **🔑 Key Insight:** Most prompt failures fall into predictable categories. Learning to recognize these patterns accelerates your debugging and improves prompt quality from the start.

### What We'll Cover

- The six major pitfall categories
- Diagnostic framework for identifying issues
- Quick reference for common symptoms and fixes
- When to refine vs. restructure prompts

### Prerequisites

- [Fundamentals of Effective Prompts](../01-fundamentals-of-effective-prompts/00-fundamentals-overview.md)
- [System Prompts & Developer Messages](../02-system-prompts-developer-messages/00-system-prompts-overview.md)
- Experience writing prompts (even imperfect ones)

---

## The Six Major Pitfall Categories

```
┌─────────────────────────────────────────────────────────────┐
│                 PROMPT PITFALL TAXONOMY                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  INSTRUCTION  │   │   RESOURCE    │   │    OUTPUT     │
│    ISSUES     │   │    ISSUES     │   │    ISSUES     │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ • Ambiguous   │   │ • Token waste │   │ • Hallucina-  │
│ • Conflicting │   │ • Under-      │   │   tions       │
│ • Over/under  │   │   specifying  │   │ • Format      │
│   constrained │   │   context     │   │   drift       │
└───────────────┘   └───────────────┘   └───────────────┘
```

| Category | Description | Typical Symptoms |
|----------|-------------|------------------|
| **Ambiguous Instructions** | Unclear or multi-interpretable directions | Inconsistent outputs, unexpected interpretations |
| **Conflicting Requirements** | Instructions that contradict each other | Partial compliance, random priority selection |
| **Over-Constraining** | Too many restrictions limiting model capability | Confused outputs, refusals, poor quality |
| **Under-Specifying** | Missing details the model needs | Wrong assumptions, incomplete outputs |
| **Token Waste** | Unnecessary verbosity consuming tokens | High costs, potential context overflow |
| **Hallucination Triggers** | Prompts that encourage fabrication | False facts, invented citations, fictional details |

---

## Diagnostic Framework

### Step 1: Identify the Symptom

| Symptom | Likely Pitfall | First Check |
|---------|----------------|-------------|
| Different outputs each time | Ambiguous instructions | Add specificity to vague terms |
| Partial task completion | Conflicting requirements | Check for contradicting instructions |
| Generic/safe outputs | Over-constraining | Reduce restrictions |
| Wrong assumptions made | Under-specifying | Add missing context |
| Outputs too long/verbose | Token waste (or under-specified length) | Add length constraints |
| Made-up information | Hallucination triggers | Add grounding, verify sources |
| Model refuses task | Over-constraining OR conflicting | Simplify prompt, remove conflicts |
| Inconsistent formatting | Ambiguous format spec | Add explicit format examples |

### Step 2: Isolate the Problem

```python
def diagnose_prompt_issue(prompt: str, outputs: list[str]) -> dict:
    """Framework for diagnosing prompt issues."""
    diagnosis = {
        "consistency": check_output_consistency(outputs),
        "completeness": check_task_completion(outputs),
        "accuracy": check_factual_accuracy(outputs),
        "format": check_format_compliance(outputs),
        "length": check_length_appropriateness(outputs),
    }
    
    issues = []
    
    if diagnosis["consistency"] < 0.7:
        issues.append("AMBIGUOUS: Outputs vary significantly")
    
    if diagnosis["completeness"] < 0.8:
        issues.append("CONFLICT or UNDER-SPECIFIED: Tasks incomplete")
    
    if diagnosis["accuracy"] < 0.9:
        issues.append("HALLUCINATION: Factual errors detected")
    
    if diagnosis["format"] < 0.9:
        issues.append("AMBIGUOUS FORMAT: Inconsistent structure")
    
    return {
        "scores": diagnosis,
        "likely_issues": issues,
        "recommendation": prioritize_fix(issues)
    }
```

### Step 3: Apply Targeted Fix

| Pitfall | Primary Fix | Secondary Fix |
|---------|-------------|---------------|
| Ambiguous | Add specific examples | Define terms explicitly |
| Conflicting | Prioritize requirements | Remove lower-priority constraints |
| Over-constraining | Remove unnecessary rules | Soften "must" to "prefer" |
| Under-specifying | Add context/examples | Specify assumptions |
| Token waste | Remove redundancy | Compress instructions |
| Hallucinations | Add grounding sources | Request citations |

---

## Quick Reference: Symptoms → Solutions

### "The model doesn't do what I want"

```
1. Is my instruction clear?
   → If no: Rephrase, add examples
   
2. Are there conflicting requirements?
   → If yes: Prioritize, remove conflicts
   
3. Did I provide enough context?
   → If no: Add relevant details
   
4. Am I over-constraining?
   → If yes: Remove unnecessary restrictions
```

### "The model makes stuff up"

```
1. Am I asking about facts beyond training data?
   → If yes: Provide the information or use RAG
   
2. Am I asking for specific details I didn't provide?
   → If yes: Supply the details or say "use only provided info"
   
3. Does my prompt encourage invention?
   → If yes: Add "If you don't know, say so"
```

### "Outputs are too long/expensive"

```
1. Did I specify length?
   → If no: Add explicit length limits
   
2. Is my prompt redundant?
   → If yes: Remove repeated instructions
   
3. Am I providing unnecessary examples?
   → If yes: Reduce to minimum needed
```

---

## Pitfall Severity Assessment

| Pitfall | Impact on Quality | Impact on Cost | Fix Difficulty |
|---------|-------------------|----------------|----------------|
| Ambiguous | High | Low | Medium |
| Conflicting | High | Low | Medium |
| Over-constraining | High | Low | Easy |
| Under-specifying | High | Low | Easy |
| Token waste | Low | High | Easy |
| Hallucinations | Critical | Low | Hard |

> **🤖 AI Context:** Hallucinations are the highest-severity issue because they can cause real harm when users trust fabricated information. Always prioritize fixing hallucination triggers first.

---

## When to Refine vs. Restructure

### Refine the Existing Prompt When:

- Output is close but not quite right
- Only one aspect needs adjustment
- The overall structure works
- Adding 1-2 clarifications would fix it

### Restructure from Scratch When:

- Multiple interacting issues
- Fundamental misunderstanding of task
- Prompt has grown through patches
- Starting fresh is faster than debugging

```
┌─────────────────────────────────────────────────────────────┐
│              REFINE vs RESTRUCTURE DECISION                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
              │ How many issues identified? │
              └─────────────────────────────┘
                     │              │
                   1-2             3+
                     │              │
                     ▼              ▼
              ┌───────────┐  ┌─────────────────────┐
              │  REFINE   │  │ Are issues related? │
              └───────────┘  └─────────────────────┘
                                   │          │
                                  Yes         No
                                   │          │
                                   ▼          ▼
                            ┌───────────┐  ┌───────────┐
                            │  REFINE   │  │RESTRUCTURE│
                            └───────────┘  └───────────┘
```

---

## Lesson Navigation

This lesson is organized into focused topics for each pitfall category:

| Lesson | Topic | Key Skills |
|--------|-------|------------|
| [01-ambiguous-instructions.md](./01-ambiguous-instructions.md) | Ambiguous Instructions | Identification, clarification, edge case testing |
| [02-conflicting-requirements.md](./02-conflicting-requirements.md) | Conflicting Requirements | Detection, prioritization, resolution |
| [03-over-under-constraining.md](./03-over-under-constraining.md) | Over/Under-Constraining | Balance, essential-only constraints |
| [04-token-waste.md](./04-token-waste.md) | Token Waste | Compression, cost optimization |
| [05-hallucination-triggers.md](./05-hallucination-triggers.md) | Hallucination Triggers | Grounding, verification, prevention |

---

## Best Practices Summary

| Practice | Why It Matters |
|----------|----------------|
| Test with edge cases early | Reveals ambiguity before production |
| Review for conflicts | Conflicting rules cause unpredictable behavior |
| Start minimal, add constraints | Easier to add than remove |
| Provide context, not just instructions | Models need information to work with |
| Measure token usage | Cost scales with verbosity |
| Ground claims in provided data | Reduces hallucination risk |

---

## Summary

- Six major pitfall categories cover most prompt issues
- Use systematic diagnosis: symptom → pitfall → targeted fix
- Hallucinations are highest severity—prioritize fixing them
- Refine for 1-2 issues, restructure for complex problems
- Each pitfall has specific solutions covered in subsequent lessons

**Next:** [Ambiguous Instructions](./01-ambiguous-instructions.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide, Anthropic Prompt Engineering Best Practices, Google Gemini Prompting Strategies -->
