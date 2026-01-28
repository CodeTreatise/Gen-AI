---
title: "Temperature & Sampling Interaction"
---

# Temperature & Sampling Interaction

- **Temperature Effects**
  - 0.0: Deterministic, most probable tokens
  - 0.7: Balanced creativity/consistency
  - 1.0: Default for Gemini 3 (recommended)
  - Higher: More random/creative
- **Prompt Design for Temperature**
  - Low temp + explicit prompts = consistent
  - High temp + open prompts = creative
  - Match temperature to task type
- **Top-P and Top-K**
  - top_p: Nucleus sampling threshold
  - top_k: Limit candidate tokens
  - Interaction with temperature
  - Provider-specific defaults
- **When to Adjust Sampling**
  - Factual tasks: lower temperature
  - Creative tasks: higher temperature
  - Code generation: low to medium
  - Brainstorming: high
