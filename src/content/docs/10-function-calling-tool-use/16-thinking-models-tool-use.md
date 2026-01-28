---
title: "Thinking Models & Tool Use"
---

# Thinking Models & Tool Use

- How thinking models differ
  - Internal reasoning before tool calls
  - Better parameter selection
  - Improved function choice
  - GPT-5 (GA), o3, o4-mini reasoning models
  - Gemini 3, Claude Opus 4.5
- OpenAI reasoning models (UPDATED 2026)
  - GPT-5: Flagship model with thinking built-in
  - o3: Most powerful reasoning model (April 2025)
  - o4-mini: Fast, cost-efficient reasoning
  - o3-pro: Extended thinking version (June 2025)
  - Responses API for reasoning models
- Reasoning items (OpenAI)
  - Returned in response output
  - Must pass back with tool results
  - Maintains reasoning context
  - SDK may handle automatically
- Thought signatures (Gemini)
  - `thought_signature` field in parts
  - Mandatory for function calling
  - Don't merge parts with signatures
  - Don't combine two signed parts
  - SDK handles automatically
- Extended thinking (Anthropic)
  - Claude extended thinking mode
  - Thinking before tool selection
  - Budget token configuration
  - Tool use with thinking
- Best practices for thinking models
  - Let model reason through complex requests
  - Don't over-constrain tool choice
  - Temperature recommendations
  - Prompt design differences
