---
title: "AI Safety & Security Fundamentals"
---

# AI Safety & Security Fundamentals

## Introduction

As AI systems become more powerful and widely deployed, understanding their security vulnerabilities and safety considerations becomes critical. This lesson covers the essential threats, defenses, and ethical considerations every AI developer must understand.

### What We'll Cover

1. [Prompt Injection Attacks](./01-prompt-injection.md) - Direct and indirect injection threats
2. [Jailbreaking Techniques](./02-jailbreaking.md) - Common exploits and defenses
3. [OWASP Top 10 for LLMs](./03-owasp-llm-top-10.md) - Industry security standards
4. [Bias and Ethics](./04-bias-fairness-ethics.md) - Fairness considerations
5. [Content Moderation](./05-content-moderation.md) - Safety filtering APIs
6. [Security Tools](./06-security-tools.md) - Defense frameworks and scanners
7. [Input/Output Sanitization](./07-sanitization-strategies.md) - Practical protection patterns

### Prerequisites

- Understanding of LLM APIs and prompting
- Basic security concepts
- Familiarity with Python

---

## The Security Landscape

```
┌─────────────────────────────────────────────────────────────────┐
│                AI SECURITY THREAT MODEL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Input                 External Data                        │
│      ↓                           ↓                               │
│  ┌──────────────────────────────────────────────┐               │
│  │           PROMPT INJECTION VECTORS            │               │
│  │  • Direct injection (user crafted)           │               │
│  │  • Indirect injection (retrieved docs)       │               │
│  │  • Jailbreaks (safety bypass)                │               │
│  └──────────────────────────────────────────────┘               │
│      ↓                                                           │
│  ┌──────────────────────────────────────────────┐               │
│  │              LLM PROCESSING                   │               │
│  │  • System prompt manipulation                │               │
│  │  • Instruction hijacking                     │               │
│  │  • Context confusion                         │               │
│  └──────────────────────────────────────────────┘               │
│      ↓                                                           │
│  ┌──────────────────────────────────────────────┐               │
│  │              OUTPUT RISKS                     │               │
│  │  • Sensitive data exposure                   │               │
│  │  • Malicious code generation                 │               │
│  │  • Harmful content creation                  │               │
│  │  • Unauthorized actions (agents)             │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Security Matters

### Real-World Incidents

```python
notable_incidents = {
    "chevrolet_chatbot": {
        "year": 2023,
        "issue": "Convinced to sell car for $1",
        "cause": "Prompt injection via user input"
    },
    "bing_chat_sydney": {
        "year": 2023,
        "issue": "Revealed internal codename and instructions",
        "cause": "Jailbreak prompts exposed system prompt"
    },
    "air_canada_chatbot": {
        "year": 2024,
        "issue": "Promised refund policy that didn't exist",
        "cause": "Hallucination + lack of guardrails"
    },
    "dpd_chatbot": {
        "year": 2024,
        "issue": "Wrote poem criticizing the company",
        "cause": "Jailbreak exploit"
    }
}
```

### Security Mindset

> **Important:** LLMs are fundamentally different from traditional software. They interpret natural language instructions, making them vulnerable to manipulation through carefully crafted text.

```python
security_principles = {
    "never_trust_input": "All user input is potentially adversarial",
    "defense_in_depth": "Multiple layers of protection",
    "least_privilege": "Minimize what the LLM can access or do",
    "fail_safe": "Default to rejection when uncertain",
    "monitoring": "Log and analyze all interactions"
}
```

---

## Quick Reference: Common Threats

| Threat | Description | Severity |
|--------|-------------|----------|
| Prompt Injection | Malicious instructions bypass safety | 🔴 Critical |
| Jailbreaking | Bypassing content restrictions | 🟠 High |
| Data Leakage | Exposing training or context data | 🔴 Critical |
| Hallucinations | Confident but false outputs | 🟡 Medium |
| Denial of Service | Resource exhaustion attacks | 🟠 High |
| Excessive Agency | Agents taking harmful actions | 🔴 Critical |

---

## Summary

This lesson provides the foundation for building secure AI applications:

✅ Understanding injection and jailbreak attacks
✅ Following OWASP LLM security standards
✅ Addressing bias and ethical concerns
✅ Implementing content moderation
✅ Using security tools and frameworks
✅ Applying input/output sanitization

**Next:** [Prompt Injection Attacks](./01-prompt-injection.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Reasoning Models](../10-reasoning-thinking-models/00-reasoning-thinking-models.md) | [Unit Overview](../00-overview.md) | [Prompt Injection](./01-prompt-injection.md) |
