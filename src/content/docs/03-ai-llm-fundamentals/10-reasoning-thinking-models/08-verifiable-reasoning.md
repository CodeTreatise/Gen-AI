---
title: "Verifiable Reasoning Outputs"
---

# Verifiable Reasoning Outputs

## Introduction

Visible reasoning enables verification - you can inspect the model's logic, debug failures, and audit decisions. This lesson covers techniques for verifying and validating AI reasoning.

### What We'll Cover

- Inspecting reasoning steps
- Debugging through thinking
- Consistency checking
- Reasoning audits

---

## Inspecting Reasoning Steps

### Extracting and Analyzing

```python
from anthropic import Anthropic
import re

client = Anthropic()

def get_reasoning_steps(problem: str) -> dict:
    """Extract structured reasoning from thinking"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": problem}]
    )
    
    thinking = ""
    answer = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text
    
    # Extract reasoning steps
    steps = extract_steps(thinking)
    
    return {
        "raw_thinking": thinking,
        "steps": steps,
        "answer": answer,        "step_count": len(steps)
    }

def extract_steps(thinking: str) -> list[dict]:
    """Parse thinking into discrete steps"""
    
    # Common reasoning patterns
    patterns = [
        r"(?:Step \d+|First|Second|Third|Next|Then|Finally)[:\s]+(.+?)(?=Step \d+|First|Second|Third|Next|Then|Finally|$)",
        r"(?:\d+\.\s+)(.+?)(?=\d+\.|$)",
        r"(?:Let me|I should|I need to)\s+(.+?)(?=Let me|I should|I need to|$)"
    ]
    
    steps = []
    for pattern in patterns:
        matches = re.findall(pattern, thinking, re.DOTALL | re.IGNORECASE)
        if matches:
            for i, match in enumerate(matches):
                steps.append({
                    "number": i + 1,
                    "content": match.strip()[:500],
                    "pattern": pattern[:20]
                })
            break
    
    return steps
```

---

## Debugging Through Thinking

### Identifying Failure Points

```python
def debug_reasoning_failure(problem: str, expected: str) -> dict:
    """Analyze where reasoning went wrong"""
    
    result = get_reasoning_steps(problem)
    
    # Check if answer matches expected
    is_correct = expected.lower() in result["answer"].lower()
    
    if not is_correct:
        thinking = result["raw_thinking"].lower()
        
        issues = []
        
        # Check for common failure patterns
        if "i'm not sure" in thinking or "uncertain" in thinking:
            issues.append("Model expressed uncertainty")
        
        if "assuming" in thinking:
            issues.append("Model made assumptions - may need clarification")
        
        if len(result["steps"]) < 2:
            issues.append("Insufficient reasoning depth")
        
        if "however" in thinking and "but" in thinking:
            issues.append("Possible internal contradiction in reasoning")
        
        return {
            "correct": False,
            "answer": result["answer"],
            "expected": expected,
            "issues": issues,
            "steps": result["steps"],
            "recommendation": "Review thinking steps for logical gaps"
        }
    
    return {
        "correct": True,
        "answer": result["answer"],
        "confidence": "high" if len(result["steps"]) >= 3 else "medium"
    }
```

### Visualizing Reasoning Flow

```
Problem: "What's 15% of 240?"

┌─────────────────────────────────────────────┐
│ THINKING TRACE                              │
├─────────────────────────────────────────────┤
│ Step 1: Understand the problem              │
│   └── Need to find 15% of 240               │
│                                             │
│ Step 2: Convert percentage                  │
│   └── 15% = 15/100 = 0.15                   │
│                                             │
│ Step 3: Calculate                           │
│   └── 240 × 0.15 = 36                       │
│                                             │
│ Step 4: Verify                              │
│   └── 36 / 240 = 0.15 ✓                     │
├─────────────────────────────────────────────┤
│ ANSWER: 36                                  │
│ CONFIDENCE: High (4 steps, verified)        │
└─────────────────────────────────────────────┘
```

---

## Consistency Checking

### Cross-Validation Approach

```python
def verify_with_multiple_runs(problem: str, runs: int = 3) -> dict:
    """Run same problem multiple times and check consistency"""
    
    answers = []
    reasoning_summaries = []
    
    for i in range(runs):
        result = get_reasoning_steps(problem)
        answers.append(result["answer"])
        reasoning_summaries.append({
            "run": i + 1,
            "step_count": result["step_count"],
            "first_step": result["steps"][0]["content"][:100] if result["steps"] else ""
        })
    
    # Check answer consistency
    unique_answers = set(answers)
    consistency_score = 1 - (len(unique_answers) - 1) / runs
    
    return {
        "answers": answers,
        "unique_answers": list(unique_answers),
        "consistency_score": consistency_score,
        "reasoning_summaries": reasoning_summaries,
        "is_reliable": consistency_score >= 0.67,
        "recommendation": (
            "High confidence" if consistency_score == 1.0
            else "Consider rephrasing question" if consistency_score < 0.67
            else "Moderate confidence - verify critical results"
        )
    }
```

### Self-Consistency Prompting

```python
def self_consistency_check(problem: str) -> dict:
    """Ask model to verify its own reasoning"""
    
    # First pass: Get initial answer with reasoning
    initial = get_reasoning_steps(problem)
    
    # Second pass: Ask model to critique its reasoning
    critique_prompt = f"""
    Problem: {problem}
    
    Your previous reasoning:
    {initial['raw_thinking'][:2000]}
    
    Your answer: {initial['answer']}
    
    Please critique this reasoning:
    1. Are there any logical errors?
    2. Were any assumptions made that should be stated?
    3. Is the conclusion properly supported?
    4. Rate confidence: LOW / MEDIUM / HIGH
    """
    
    critique = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{"role": "user", "content": critique_prompt}]
    )
    
    return {
        "initial_answer": initial["answer"],
        "critique": critique.content[-1].text,
        "self_validated": "no logical errors" in critique.content[-1].text.lower()
    }
```

---

## Reasoning Audits for Production

### Audit Log Structure

```python
from datetime import datetime
import json

def create_audit_log(
    problem: str,
    result: dict,
    user_id: str,
    metadata: dict = None
) -> dict:
    """Create comprehensive audit log for reasoning"""
    
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "problem_hash": hash(problem) % 10**8,
        "problem_preview": problem[:100] + "..." if len(problem) > 100 else problem,
        
        "reasoning": {
            "step_count": result.get("step_count", 0),
            "thinking_length": len(result.get("raw_thinking", "")),
            "answer_length": len(result.get("answer", "")),
        },
        
        "quality_signals": {
            "has_uncertainty": "not sure" in result.get("raw_thinking", "").lower(),
            "has_verification": "verify" in result.get("raw_thinking", "").lower(),
            "has_assumptions": "assuming" in result.get("raw_thinking", "").lower(),
        },
        
        "metadata": metadata or {}
    }
    
    return audit_entry

def log_for_review(audit_entry: dict, threshold: float = 0.7):
    """Flag entries needing human review"""
    
    signals = audit_entry["quality_signals"]
    
    needs_review = (
        signals["has_uncertainty"] or
        signals["has_assumptions"] or
        audit_entry["reasoning"]["step_count"] < 2
    )
    
    if needs_review:
        audit_entry["flagged_for_review"] = True
        audit_entry["flag_reason"] = [
            k for k, v in signals.items() if v
        ]
    
    return audit_entry
```

---

## Summary

✅ Visible reasoning enables debugging and verification  
✅ Extract structured steps from thinking blocks  
✅ Use consistency checking across multiple runs  
✅ Implement audit logging for production systems  
✅ Self-consistency prompting validates model's own logic  

**Next:** [Lesson Overview](./00-reasoning-thinking-models.md)

---

## Further Reading

- [Chain-of-Thought Prompting Research](https://arxiv.org/abs/2201.11903)
- [Self-Consistency in LLMs](https://arxiv.org/abs/2203.11171)
- [Anthropic Extended Thinking Guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
