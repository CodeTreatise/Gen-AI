---
title: "RAG Security & Guardrails"
---

# RAG Security & Guardrails

- Prompt injection in RAG
  - Malicious content in retrieved documents
  - Instructions hidden in user-uploaded content
  - Defense: separate user input from context
  - Defense: content scanning before indexing
- Context poisoning
  - Adversarial document injection
  - SEO-style manipulation
  - Defense: source verification
  - Defense: content validation
- PII handling in RAG
  - PII detection in retrieved content
  - Redaction before generation
  - Access control by document sensitivity
  - Audit logging for compliance
- Output validation
  - Factual consistency checking
  - Harmful content filtering
  - Source verification for claims
  - Confidence thresholds for responses
- Access control patterns
  - User-level document permissions
  - Role-based retrieval filtering
  - Metadata-based access control
  - Query-time permission checks
- Guardrails implementation
  - Input validation
  - Retrieval result filtering
  - Output content moderation
  - Rate limiting and abuse prevention
