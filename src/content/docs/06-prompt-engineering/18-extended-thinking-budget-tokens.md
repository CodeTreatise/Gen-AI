---
title: "Extended Thinking & Budget Tokens"
---

# Extended Thinking & Budget Tokens

- **What is Extended Thinking?**
  - Claude's extended thinking mode
  - Longer reasoning before response
  - Budget tokens allocation
  - Higher quality for complex tasks
- **Thinking Budget Configuration**
  - `thinking: { type: "enabled", budget_tokens: 10000 }`
  - Higher budget = more reasoning
  - Token cost considerations
  - Task-dependent optimization
- **When to Enable Extended Thinking**
  - Complex analysis tasks
  - Multi-step problem solving
  - Creative tasks requiring exploration
  - When accuracy trumps speed
- **Prompting for Extended Thinking**
  - Provide rich context
  - Ask open-ended analytical questions
  - Don't constrain the approach
  - Allow model to explore
- **Gemini Thinking Mode**
  - Similar to Claude extended thinking
  - Gemini 2.5 Pro/Flash thinking
  - Thinking budget parameter
  - Disable for specific tasks: `thinking_budget: 0`
