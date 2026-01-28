---
title: "Context Window Management"
---

# Context Window Management

- **Understanding Context Windows**
  - Maximum tokens model can process
  - Includes input + output tokens
  - Varies by model (128K to 1M+)
  - Check model docs for limits
- **Token Budgeting Strategies**
  - Reserve space for output
  - Reserve for reasoning (25K+ for reasoning models)
  - Calculate: input + expected output < limit
  - Monitor actual usage
- **Context Overflow Handling**
  - Truncation strategies (oldest first, summary)
  - Sliding window for conversations
  - Summarization of older context
  - Important info pinning
- **Long Context Best Practices**
  - Place context first, instructions last
  - Use clear section markers
  - "Based on the information above..."
  - Needle-in-haystack positioning awareness
