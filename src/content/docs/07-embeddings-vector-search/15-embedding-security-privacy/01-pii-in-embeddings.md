---
title: "PII in Embeddings"
---

# PII in Embeddings

## Introduction

Embeddings are designed to capture semantic meaning—which is precisely why they're a privacy risk. When you embed text containing personally identifiable information (PII), that information gets encoded into the vector representation. The embedding "remembers" the sensitive content.

This lesson explores how PII becomes embedded, the privacy risks this creates, and strategies for mitigation.

### What We'll Cover

- How embeddings encode semantic content
- PII that can be partially reconstructed
- Privacy risks in shared embeddings
- Regulatory implications

### Prerequisites

- Basic understanding of embeddings
- Familiarity with data privacy concepts

---

## How Embeddings Capture PII

### Semantic Encoding

Embeddings map text to vectors where similar meanings produce similar vectors. This means:

```
┌─────────────────────────────────────────────────────────────────┐
│              Semantic Encoding of PII                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Text:                                                    │
│  "John Smith's SSN is 123-45-6789 and he lives in Seattle"     │
│                                                                 │
│  What the embedding encodes:                                    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ • Person named "John Smith"                            │    │
│  │ • SSN number pattern                                   │    │
│  │ • Geographic location (Seattle)                        │    │
│  │ • Relationship between person and identifiers         │    │
│  │ • Semantic context of personal information            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  The embedding doesn't just store "noise"—it stores            │
│  meaningful representations of the PII.                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Types of PII at Risk

| PII Category | Examples | Encoding Risk |
|--------------|----------|---------------|
| **Direct Identifiers** | Names, SSN, email | High - directly encoded |
| **Quasi-Identifiers** | Age, ZIP code, job title | Medium - combinable |
| **Sensitive Attributes** | Health conditions, finances | High - semantically rich |
| **Behavioral Data** | Preferences, opinions | Medium - contextual |
| **Biometric Data** | Voice transcripts, writing style | High - unique patterns |

---

## The Myth of Embedding Anonymization

### Common Misconception

> "Converting text to embeddings anonymizes the data because you can't read the original text."

**This is dangerously false.**

### Why Embeddings Aren't Anonymous

```python
# This does NOT anonymize the data
def dangerous_assumption(pii_text: str) -> list:
    """Incorrectly assuming embedding = anonymization."""
    embedding = embed_model.encode(pii_text)
    
    # The embedding still contains:
    # - Semantic information about the PII
    # - Patterns that can be inverted
    # - Signals for attribute inference
    
    return embedding

# The embedding is NOT safe to share publicly
# It should be treated with same security as source text
```

### What Research Shows

Studies demonstrate that embeddings leak significant information:

| Attack Type | Information Recovered | Source |
|-------------|----------------------|--------|
| Inversion Attack | 50-70% of words (F1 score 0.5-0.7) | Song & Raghunathan, 2020 |
| Attribute Inference | Author identity, demographics | Multiple studies |
| Membership Inference | Whether text was in training data | Shokri et al., 2017 |

---

## Privacy Risks by Use Case

### Enterprise Document Search

```
┌─────────────────────────────────────────────────────────────────┐
│              Risk: Document Search                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario: Company embeds internal documents for search         │
│                                                                 │
│  Documents contain:                                             │
│  • Employee performance reviews (names, ratings)               │
│  • Customer contracts (names, addresses, payment terms)         │
│  • Legal documents (sensitive case details)                     │
│  • HR records (salaries, personal issues)                       │
│                                                                 │
│  Risks:                                                         │
│  1. Embeddings stored in third-party vector database           │
│  2. API calls contain sensitive content                         │
│  3. Query results may leak information across access levels    │
│  4. Embedding theft exposes document semantics                  │
│                                                                 │
│  Mitigation:                                                    │
│  • Pre-process documents to remove/redact PII                  │
│  • Use self-hosted embedding models                            │
│  • Implement strict access controls on vector database         │
│  • Audit query patterns                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Customer Support RAG

```
┌─────────────────────────────────────────────────────────────────┐
│              Risk: Customer Support                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario: RAG system for customer support tickets             │
│                                                                 │
│  Tickets contain:                                               │
│  • Customer names and contact info                             │
│  • Account numbers and order IDs                               │
│  • Complaints and personal situations                          │
│  • Payment issues and financial details                        │
│                                                                 │
│  Risks:                                                         │
│  1. Cross-customer information leakage                         │
│  2. Support agents seeing unrelated customer data              │
│  3. Embedding model provider access to customer data           │
│  4. Historical tickets persisting after deletion requests      │
│                                                                 │
│  Mitigation:                                                    │
│  • Tenant isolation in vector database                         │
│  • Query-time customer filtering                               │
│  • On-premise or private embedding APIs                        │
│  • Implement deletion workflows for GDPR/CCPA                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Healthcare Applications

```
┌─────────────────────────────────────────────────────────────────┐
│              Risk: Healthcare                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario: Medical record search or clinical decision support   │
│                                                                 │
│  Records contain:                                               │
│  • Patient names and identifiers                               │
│  • Diagnoses and conditions (PHI)                              │
│  • Treatment histories                                          │
│  • Genetic information                                          │
│                                                                 │
│  Regulatory requirements:                                       │
│  • HIPAA (US): PHI must be protected                           │
│  • GDPR (EU): Health data is "special category"                │
│  • De-identification not achieved by embedding                 │
│                                                                 │
│  Mitigation:                                                    │
│  • HIPAA-compliant infrastructure only                         │
│  • Business Associate Agreements with all vendors              │
│  • Consider: Do you need PII in embeddings at all?             │
│  • De-identify before embedding (not just after)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mitigation Strategies

### Strategy 1: PII Removal Before Embedding

The most effective approach: remove PII before creating embeddings.

```python
import re
from typing import Tuple

def remove_pii(text: str) -> Tuple[str, dict]:
    """
    Remove PII before embedding.
    Returns sanitized text and extracted PII for separate secure storage.
    """
    pii_extracted = {}
    sanitized = text
    
    # Email addresses
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', sanitized)
    for i, email in enumerate(emails):
        placeholder = f"[EMAIL_{i}]"
        pii_extracted[placeholder] = email
        sanitized = sanitized.replace(email, placeholder)
    
    # Phone numbers (simple pattern)
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', sanitized)
    for i, phone in enumerate(phones):
        placeholder = f"[PHONE_{i}]"
        pii_extracted[placeholder] = phone
        sanitized = sanitized.replace(phone, placeholder)
    
    # SSN pattern
    ssns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', sanitized)
    for i, ssn in enumerate(ssns):
        placeholder = f"[SSN_{i}]"
        pii_extracted[placeholder] = ssn
        sanitized = sanitized.replace(ssn, placeholder)
    
    # Names (requires NER - simplified example)
    # In production, use spaCy or similar NER model
    
    return sanitized, pii_extracted

# Usage
original = "Contact John Smith at john@example.com or 555-123-4567"
sanitized, pii = remove_pii(original)

print(f"Sanitized: {sanitized}")
# Output: Contact John Smith at [EMAIL_0] or [PHONE_0]

# Store PII separately with strong encryption
# Only store sanitized text in vector database
```

### Strategy 2: Named Entity Recognition (NER)

Use NER to identify and mask named entities:

```python
import spacy

# Load NER model
nlp = spacy.load("en_core_web_sm")

def mask_entities(text: str) -> str:
    """Replace named entities with type placeholders."""
    doc = nlp(text)
    
    masked = text
    # Process in reverse order to preserve indices
    for ent in reversed(doc.ents):
        placeholder = f"[{ent.label_}]"
        masked = masked[:ent.start_char] + placeholder + masked[ent.end_char:]
    
    return masked

# Example
text = "Dr. Sarah Johnson prescribed medication for patient at 123 Main St, Boston"
masked = mask_entities(text)
print(masked)
# Output: [PERSON] prescribed medication for patient at [CARDINAL] [FAC], [GPE]
```

### Strategy 3: Differential Privacy (Advanced)

Add noise to embeddings to provide privacy guarantees:

```python
import numpy as np

def add_differential_privacy(
    embedding: np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0
) -> np.ndarray:
    """
    Add Laplacian noise for differential privacy.
    
    Lower epsilon = more privacy, less utility
    Higher epsilon = less privacy, more utility
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, embedding.shape)
    
    noisy_embedding = embedding + noise
    
    # Re-normalize if using cosine similarity
    noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
    
    return noisy_embedding

# Example
original_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
original_embedding = original_embedding / np.linalg.norm(original_embedding)

private_embedding = add_differential_privacy(original_embedding, epsilon=0.5)

print(f"Original: {original_embedding[:3]}")
print(f"Private:  {private_embedding[:3]}")
```

> **Warning:** Differential privacy reduces embedding utility. Benchmark carefully for your use case.

### Strategy 4: Separate Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Secure Architecture Pattern                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ENCRYPTED PII STORE                   │  │
│  │  • Original documents with PII                          │  │
│  │  • Strong encryption (AES-256)                          │  │
│  │  • Strict access controls                               │  │
│  │  • Full audit logging                                   │  │
│  │  • Separate key management                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ document_id                      │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    VECTOR DATABASE                       │  │
│  │  • Embeddings from sanitized text                       │  │
│  │  • Metadata (document_id, sanitized_preview)            │  │
│  │  • NO PII in embeddings or metadata                     │  │
│  │  • Standard encryption                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ search results                   │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    APPLICATION                           │  │
│  │  1. Query vector database (sanitized)                   │  │
│  │  2. Get document_ids from results                       │  │
│  │  3. Fetch full documents from PII store (if authorized) │  │
│  │  4. Apply access control at fetch time                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## What PII Categories Are Most Risky?

### High Risk - Avoid Embedding

| Category | Why High Risk | Recommendation |
|----------|---------------|----------------|
| SSN/Tax IDs | Unique identifiers, fraud risk | Never embed |
| Credit card numbers | Direct financial risk | Never embed |
| Passwords/secrets | Security breach | Never embed |
| Medical record numbers | HIPAA/protected | Never embed |

### Medium Risk - Mask Before Embedding

| Category | Why Medium Risk | Recommendation |
|----------|-----------------|----------------|
| Names | Personally identifying | Mask with NER |
| Addresses | Location identification | Mask or generalize |
| Phone numbers | Contact identification | Mask |
| Email addresses | Identity + contact | Mask |

### Lower Risk - Consider Context

| Category | Consideration | Decision Factor |
|----------|---------------|-----------------|
| Job titles | Semi-public | Depends on context |
| Company names | Often public | Usually OK |
| Dates | Contextual | May need masking |
| General topics | Low risk | Usually OK |

---

## Summary

✅ **Embeddings encode semantic meaning** including PII—they are NOT anonymization  
✅ **Research shows 50-70% word recovery** is possible from embeddings  
✅ **Remove PII before embedding** using regex, NER, or specialized tools  
✅ **Store PII separately** from vector database with stricter controls  
✅ **Treat embeddings like source data** for access control and compliance

---

**Next:** [Embedding Inversion Attacks →](./02-embedding-inversion-attacks.md)

---

<!-- 
Sources Consulted:
- Song & Raghunathan, "Information Leakage in Embedding Models" (2020): https://arxiv.org/abs/2004.00053
- NIST Privacy Framework
- GDPR Article 4 (Personal Data Definition)
-->
