---
title: "Prompt Security & Injection Defense"
---

# Prompt Security & Injection Defense

- **Understanding Prompt Injection**
  - What is prompt injection?
  - Direct injection (user input manipulation)
  - Indirect injection (malicious content in data)
  - Goal: override system instructions
- **Attack Vectors**
  - "Ignore previous instructions" patterns
  - Role-playing escape ("pretend you're...")
  - Encoded instructions (base64, unicode)
  - Multi-language injection
  - Markdown/HTML injection
- **Defense Strategies**
  - Input sanitization and validation
  - Clear delimiters between instructions and data
  - XML tags to separate trusted/untrusted content
  - Output validation and filtering
  - Instruction hierarchy (system > developer > user)
- **Structural Defenses**
  - Sandwich defense (instructions before AND after user input)
  - Random delimiters that attackers can't predict
  - Separate LLM calls for validation
  - Capability restrictions in system prompt
- **Detection Techniques**
  - Anomaly detection on outputs
  - Semantic similarity to expected outputs
  - Classifier for injection attempts
  - Logging and monitoring suspicious patterns
- **Provider Security Features**
  - OpenAI moderation API integration
  - Anthropic constitutional AI principles
  - Content filtering layers
  - Rate limiting for abuse prevention
