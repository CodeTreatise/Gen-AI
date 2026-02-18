---
title: "Human-in-the-Loop"
---

# Human-in-the-Loop

## Introduction

The most reliable AI agents aren't fully autonomous — they know when to ask for help. Human-in-the-loop (HITL) patterns let agents pause execution, request human input, and resume with better information. This creates a partnership where agents handle routine work while humans make critical decisions.

HITL isn't a sign of agent weakness. It's a design pattern that combines the speed and scalability of AI with the judgment and accountability of humans. Production agents that skip human oversight on high-stakes decisions eventually cause incidents. Agents that ask for input too often become slower than doing the work manually. The art is finding the right balance.

### What we'll cover

This lesson covers six interconnected aspects of human-in-the-loop design:

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 01 | [Confirmation Workflows](./01-confirmation-workflows.md) | Preview actions, approve/reject UIs, modification options, batch approval |
| 02 | [Approval Gates](./02-approval-gates.md) | Critical action gates, threshold-based gates, role-based approval, audit trails |
| 03 | [Feedback Incorporation](./03-feedback-incorporation.md) | User corrections, preference learning, behavior adjustment over time |
| 04 | [Override Mechanisms](./04-override-mechanisms.md) | Manual overrides, emergency stops, direction changes, undo capabilities |
| 05 | [Collaborative Execution](./05-collaborative-execution.md) | Human + agent partnerships, task sharing, complementary strengths |
| 06 | [Trust Calibration](./06-trust-calibration.md) | Building trust, transparency measures, confidence communication, recovery |

### Prerequisites

- [Error Handling and Recovery](../08-error-handling-recovery/00-error-handling-recovery.md) — escalation triggers that initiate human involvement
- [Agent Fundamentals](../01-agent-fundamentals/) — basic agent architecture and execution loops
- [State Management](../07-state-management/) — persisting state across interruptions

---

## The HITL spectrum

Human-in-the-loop isn't binary. Agents operate on a spectrum from full autonomy to full human control, and the right position depends on the task's risk, the agent's confidence, and the user's trust level:

```mermaid
flowchart LR
    A["🤖 Full Autonomy<br/>Low risk, high confidence"] --> B["📋 Notify<br/>Inform after action"]
    B --> C["✋ Confirm<br/>Ask before action"]
    C --> D["👥 Collaborate<br/>Work together"]
    D --> E["👤 Human Leads<br/>Agent assists"]
    
    style A fill:#4CAF50,color:#fff
    style B fill:#8BC34A,color:#fff
    style C fill:#FFC107,color:#000
    style D fill:#FF9800,color:#fff
    style E fill:#F44336,color:#fff
```

| Level | When to Use | Example |
|-------|-------------|---------|
| **Full autonomy** | Routine, reversible, low-risk actions | Formatting text, fetching public data |
| **Notify** | Low-risk actions the user should know about | Sending a calendar invite, creating a draft |
| **Confirm** | Irreversible or sensitive actions | Deleting records, sending emails, payments |
| **Collaborate** | Complex tasks benefiting from human insight | Research synthesis, creative writing |
| **Human leads** | High-stakes or novel situations | Medical recommendations, legal decisions |

### Framework support

| Framework | HITL Mechanism | Key Feature |
|-----------|---------------|-------------|
| **LangGraph** | `interrupt()` + `Command(resume=)` | Pauses at any point, resumes with human input |
| **OpenAI Agents SDK** | Handoffs + external orchestration | Agent-to-agent transfer for escalation |
| **CrewAI** | `human_input=True` on tasks | Built-in human feedback collection |
| **AutoGen** | `UserProxyAgent` | Dedicated human-in-the-loop agent type |

---

## Summary

✅ HITL is a **spectrum** from full autonomy to full human control — the right level depends on risk, confidence, and trust

✅ **LangGraph's `interrupt()`** is the most flexible HITL mechanism — it pauses anywhere and resumes with any JSON-serializable input

✅ Production agents need HITL for **irreversible actions**, **low-confidence outputs**, and **novel situations** the agent wasn't designed for

✅ The goal is a **partnership**: agents handle speed and scale, humans provide judgment and accountability

**Next:** [Confirmation Workflows](./01-confirmation-workflows.md)

---

## Further reading

- [LangGraph — Interrupts](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — complete interrupt documentation
- [Google PAIR Guidebook — Feedback + Control](https://pair.withgoogle.com/chapter/feedback-controls/) — designing human-AI feedback loops
- [Google PAIR Guidebook — Trust + Explanations](https://pair.withgoogle.com/chapter/trust/) — calibrating user trust in AI

*[Back to AI Agents unit overview](../00-overview.md)*

<!-- 
Sources Consulted:
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- Google PAIR Guidebook: https://pair.withgoogle.com/guidebook/
- Google PAIR Feedback + Control: https://pair.withgoogle.com/chapter/feedback-controls/
-->
