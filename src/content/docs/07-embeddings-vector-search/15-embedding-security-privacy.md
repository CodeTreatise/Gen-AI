---
title: "Embedding Security & Privacy"
---

# Embedding Security & Privacy

- PII in embeddings
  - Embeddings encode semantic content
  - PII can be partially reconstructed
  - Privacy risks in shared embeddings
  - Regulatory considerations (GDPR, CCPA)
- Embedding inversion attacks
  - Research shows partial text recovery possible
  - Sensitive data exposure risks
  - Defense: access control, not obfuscation
  - Monitor for embedding exfiltration
- Secure storage practices
  - Encryption at rest
  - Encryption in transit
  - Access control and audit logs
  - Separate embedding and source storage
  - Key management for encrypted stores
- Multi-tenant isolation
  - Namespace/collection per tenant
  - Query-level tenant filtering
  - Prevent cross-tenant leakage
  - Audit tenant access patterns
- Compliance considerations
  - Data residency requirements
  - Right to deletion (embedding deletion)
  - Data processing agreements
  - Vendor security certifications
- Model and data poisoning
  - Adversarial inputs affecting retrieval
  - Backdoor attacks on fine-tuned models
  - Input validation and sanitization
  - Monitoring for anomalous queries
