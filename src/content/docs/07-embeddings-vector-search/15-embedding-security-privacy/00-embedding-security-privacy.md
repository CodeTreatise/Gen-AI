---
title: "Embedding Security & Privacy"
---

# Embedding Security & Privacy

## Overview

Embeddings encode semantic meaning—but that semantic richness creates privacy risks. This lesson explores the security considerations unique to vector embeddings, from PII leakage to multi-tenant isolation to regulatory compliance.

Understanding these risks is essential for building production systems that handle sensitive data responsibly.

### Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [PII in Embeddings](./01-pii-in-embeddings.md) | How embeddings encode sensitive content |
| 02 | [Embedding Inversion Attacks](./02-embedding-inversion-attacks.md) | Research on text recovery from vectors |
| 03 | [Secure Storage Practices](./03-secure-storage-practices.md) | Encryption, access control, key management |
| 04 | [Multi-Tenant Isolation](./04-multi-tenant-isolation.md) | Preventing cross-tenant data leakage |
| 05 | [Compliance Considerations](./05-compliance-considerations.md) | GDPR, CCPA, data residency, right to deletion |
| 06 | [Model & Data Poisoning](./06-model-data-poisoning.md) | Adversarial attacks and defenses |

---

## The Security Mindset for Embeddings

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding Security Threat Model                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA LIFECYCLE THREATS:                                        │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Input   │───▶│ Embedding│───▶│ Storage  │───▶│  Query   │  │
│  │  Text    │    │ Creation │    │  & Index │    │ Results  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ PII in   │    │Inversion │    │ Data at  │    │ Cross-   │  │
│  │ Source   │    │ Attacks  │    │ Rest     │    │ Tenant   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
│  ATTACK VECTORS:                                                │
│  • Embedding exfiltration → partial text reconstruction        │
│  • Unauthorized vector access → semantic information leak      │
│  • Tenant confusion → cross-tenant query results               │
│  • Adversarial inputs → poisoned retrieval results             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insight: Embeddings Are Not Anonymization

A common misconception: "Converting text to embeddings anonymizes the data."

**This is false.** Research shows:

| Attack Type | What's Leaked | Severity |
|-------------|---------------|----------|
| Embedding Inversion | 50-70% of words recoverable | High |
| Attribute Inference | Authorship, sentiment, demographics | Medium |
| Membership Inference | Whether specific text was in training | Medium |

> **Critical:** Treat embeddings with the same security controls as the source text.

---

## Defense in Depth

```
┌─────────────────────────────────────────────────────────────────┐
│              Security Controls by Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: ACCESS CONTROL                                        │
│  ├── API key authentication                                    │
│  ├── Role-based access (RBAC)                                  │
│  ├── JWT with tenant scoping                                   │
│  └── Audit logging                                             │
│                                                                 │
│  LAYER 2: DATA PROTECTION                                       │
│  ├── Encryption at rest (AES-256)                              │
│  ├── Encryption in transit (TLS 1.2+)                          │
│  ├── Customer-managed encryption keys (CMEK)                   │
│  └── Separate storage for embeddings and source                │
│                                                                 │
│  LAYER 3: TENANT ISOLATION                                      │
│  ├── Database-level separation                                 │
│  ├── Collection/namespace per tenant                           │
│  ├── Query-level tenant filtering                              │
│  └── Network isolation (VPC, private endpoints)                │
│                                                                 │
│  LAYER 4: MONITORING & RESPONSE                                 │
│  ├── Anomaly detection on query patterns                       │
│  ├── Embedding exfiltration monitoring                         │
│  ├── Input validation and sanitization                         │
│  └── Incident response procedures                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Security Checklist

### Before Production

| Category | Check | Priority |
|----------|-------|----------|
| **Access Control** | API keys configured | Critical |
| | RBAC enabled | High |
| | Audit logging enabled | High |
| **Encryption** | TLS for all connections | Critical |
| | Encryption at rest enabled | Critical |
| | Key rotation scheduled | Medium |
| **Tenant Isolation** | Tenant separation strategy chosen | Critical |
| | Query filtering by tenant ID | Critical |
| | Cross-tenant access tested | High |
| **Compliance** | Data residency requirements met | Varies |
| | Deletion procedures documented | High |
| | Data processing agreements signed | High |
| **Monitoring** | Query anomaly alerts | Medium |
| | Access pattern monitoring | Medium |
| | Incident response plan | High |

---

## Vector Database Security Features

### Quick Comparison

| Feature | Pinecone | Qdrant | Milvus | Weaviate |
|---------|----------|--------|--------|----------|
| API Key Auth | ✅ | ✅ | ✅ | ✅ |
| RBAC | ✅ | ✅ (JWT) | ✅ | ✅ |
| Encryption at Rest | ✅ (AES-256) | ✅ | ✅ | ✅ |
| Encryption in Transit | ✅ (TLS 1.2) | ✅ | ✅ | ✅ |
| CMEK | ✅ (AWS KMS) | ❌ | ✅ | ✅ |
| Multi-Tenancy | Projects | Collections | Database/Collection/Partition | Tenants |
| Audit Logs | ✅ (Enterprise) | ❌ | ✅ | ✅ |
| SOC 2 | ✅ | ✅ | ✅ | ✅ |
| GDPR | ✅ | ✅ | ✅ | ✅ |

---

## Learning Path

**Start here if you're:**

| Scenario | Start With |
|----------|------------|
| Handling customer PII | [PII in Embeddings](./01-pii-in-embeddings.md) |
| Building multi-tenant SaaS | [Multi-Tenant Isolation](./04-multi-tenant-isolation.md) |
| Subject to GDPR/CCPA | [Compliance Considerations](./05-compliance-considerations.md) |
| Concerned about attacks | [Embedding Inversion Attacks](./02-embedding-inversion-attacks.md) |
| Setting up infrastructure | [Secure Storage Practices](./03-secure-storage-practices.md) |

---

**Next:** [PII in Embeddings →](./01-pii-in-embeddings.md)

---

<!-- 
Sources Consulted:
- Song & Raghunathan, "Information Leakage in Embedding Models" (2020): https://arxiv.org/abs/2004.00053
- Pinecone Security Overview: https://docs.pinecone.io/docs/security
- Qdrant Security Guide: https://qdrant.tech/documentation/guides/security/
- Milvus Multi-tenancy: https://milvus.io/docs/multi_tenancy.md
-->
