---
title: "Reasoning & Thinking Display"
---

# Reasoning & Thinking Display

- Rendering reasoning tokens
  - Detecting `part.type === 'reasoning'`
  - `part.text` for reasoning content
  - Styling thinking vs response content
  - Progressive reveal of thinking
- Thinking section UI patterns
  - Collapsible/expandable sections
  - "Show thinking" toggle button
  - Thinking duration display
  - Token count visualization
- Streaming reasoning display
  - Real-time thinking visualization
  - Animated thinking indicators
  - Partial reasoning updates
  - Completion detection
- Server-side configuration
  - `sendReasoning: true` in toUIMessageStreamResponse
  - DeepSeek R1 reasoning tokens
  - Claude thinking tokens
  - OpenAI o-series reasoning
- Cost implications UI
  - Reasoning token counts
  - Cost breakdown display
  - User preference for hiding/showing
  - Reasoning summary option
- Tool invocation display
  - Rendering `part.type === 'tool-invocation'`
  - Tool name and status indicator
  - Arguments display (collapsible JSON)
  - Loading spinner during execution
  - `part.type === 'tool-result'` rendering
  - Success/error state styling
  - Tool output formatting
