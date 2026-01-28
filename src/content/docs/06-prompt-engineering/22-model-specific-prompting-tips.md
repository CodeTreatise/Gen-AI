---
title: "Model-Specific Prompting Tips"
---

# Model-Specific Prompting Tips

- **OpenAI GPT-4.1 / GPT-5**
  - Strong instruction following
  - Excellent at structured outputs
  - Use developer role in Responses API
  - Supports predicted outputs for speed
  - Context windows: 128K-1M tokens
- **OpenAI o3 / o4-mini (Reasoning)**
  - Don't use chain-of-thought prompts
  - Give goals, not step-by-step instructions
  - Use reasoning effort parameter
  - Reserve tokens for reasoning
- **Anthropic Claude 4**
  - Excellent at nuanced instructions
  - System prompt as separate parameter
  - Extended thinking for complex tasks
  - Strong at refusing harmful requests
- **Google Gemini 3**
  - Keep temperature at 1.0 (recommended - lower may cause loops)
  - Native multimodal (text, image, video, audio)
  - Use systemInstruction parameter
  - Thinking mode for complex reasoning
  - Object detection/segmentation capabilities
  - Add "Remember it is 2025" for time-sensitive queries
  - Specify knowledge cutoff: "Your knowledge cutoff is January 2025"
- **Cross-Provider Tips**
  - Test same prompt across providers
  - Adjust for provider-specific syntax
  - Account for different tokenization
  - Monitor quality variations
