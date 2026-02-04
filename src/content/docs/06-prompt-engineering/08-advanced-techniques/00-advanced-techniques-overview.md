---
title: "Advanced Prompting Techniques"
---

# Advanced Prompting Techniques

## Overview

Basic prompting gets you started. Advanced techniques transform how you solve complex problems. This lesson covers strategies that go beyond simple instructions—role-playing, constraint systems, meta-prompting, decomposition, and sophisticated reasoning frameworks like Tree of Thoughts and Self-Consistency.

> **🔑 Key Insight:** Advanced techniques trade simplicity for power. Each adds complexity but unlocks capabilities that basic prompting cannot match.

## When to Use Advanced Techniques

| Technique | Use When | Token Cost | Complexity |
|-----------|----------|------------|------------|
| Role-playing | Need domain expertise, consistent persona | Low | Low |
| Constraints | Multiple requirements must be balanced | Low | Medium |
| Meta-prompting | Generating or optimizing prompts at scale | Medium | Medium |
| Decomposition | Complex multi-step tasks | Medium | Medium |
| Tree of Thoughts | Strategic lookahead needed | High | High |
| Self-Consistency | High-stakes decisions, need confidence | High | High |
| Self-Critique | Quality assurance, iterative refinement | Medium | Medium |
| Explicit Planning | Complex tasks with dependencies | Low | Medium |

---

## Quick Reference

### Technique Selection Matrix

```
Is the task simple?
├── YES → Basic prompting is sufficient
│
└── NO → What's the main challenge?
         │
         ├── Need domain expertise → Role-playing prompts
         ├── Multiple requirements → Constraint-based prompting
         ├── Need to generate prompts → Meta-prompting
         ├── Too complex for one step → Decomposition
         ├── Need strategic exploration → Tree of Thoughts
         ├── Need high confidence → Self-Consistency
         ├── Quality verification needed → Self-Critique
         └── Complex dependencies → Explicit Planning
```

### Model Considerations

| Model Type | Recommended Techniques |
|------------|------------------------|
| **GPT models** (GPT-4o, GPT-4.1) | All techniques beneficial |
| **Reasoning models** (GPT-5, o3) | Role-playing, constraints, decomposition (skip CoT-based) |
| **Smaller models** | Role-playing, constraints, few-shot variants |
| **Gemini 3** | Explicit planning especially effective |

---

## Lessons in This Section

| # | Lesson | Focus |
|---|--------|-------|
| 01 | [Role-Playing Prompts](./01-role-playing-prompts.md) | Expert personas, character consistency, domain activation |
| 02 | [Constraint-Based Prompting](./02-constraint-based-prompting.md) | Hard/soft constraints, prioritization, conflict handling |
| 03 | [Meta-Prompting](./03-meta-prompting.md) | Generating prompts with AI, optimization cycles |
| 04 | [Decomposition Strategies](./04-decomposition-strategies.md) | Task breakdown, parallel execution, synthesis |
| 05 | [Tree of Thoughts](./05-tree-of-thought-prompting.md) | Multiple reasoning paths, evaluation, backtracking |
| 06 | [Self-Consistency](./06-self-consistency-checking.md) | Multiple generations, voting, confidence estimation |
| 07 | [Self-Critique & Reflexion](./07-self-critique-reflexion.md) | Review own output, iterative improvement |
| 08 | [Explicit Planning](./08-explicit-planning-prompts.md) | Structured planning before execution |

---

## Prerequisites

Before diving into advanced techniques, ensure you understand:

- [Fundamentals of Effective Prompts](../01-fundamentals-of-effective-prompts/00-fundamentals-overview.md)
- [System Prompts & Developer Messages](../02-system-prompts-developer-messages/00-system-prompts-overview.md)
- [Chain-of-Thought Prompting](../07-chain-of-thought-prompting/00-chain-of-thought-overview.md)

---

## Combining Techniques

Advanced techniques work best in combination:

```
Role + Constraints + Decomposition
= Expert persona following rules while breaking down complex task

Self-Consistency + Self-Critique
= Multiple attempts with quality verification

Tree of Thoughts + Explicit Planning
= Strategic exploration with structured approach
```

---

**Next:** [Role-Playing Prompts](./01-role-playing-prompts.md)

---

<!-- Sources: OpenAI Prompt Engineering, Anthropic Prompt Engineering, Google Gemini Prompting Strategies, Prompt Engineering Guide -->
