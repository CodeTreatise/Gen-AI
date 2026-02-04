---
title: "Compliance Considerations"
---

# Compliance Considerations

## Introduction

Privacy regulations like GDPR and CCPA create specific requirements for systems that process personal data. Vector databases storing embeddings of user content fall under these regulations, requiring careful attention to data residency, deletion rights, and processing agreements.

This lesson covers the compliance landscape for embedding systems and practical implementation strategies.

### What We'll Cover

- GDPR requirements for vector databases
- CCPA compliance considerations
- Right to deletion implementation
- Data residency and sovereignty
- Vendor security certifications
- Documentation and audit readiness

### Prerequisites

- Understanding of [PII in embeddings](./01-pii-in-embeddings.md)
- Basic knowledge of data privacy concepts
- Familiarity with [secure storage practices](./03-secure-storage-practices.md)

---

## Regulatory Landscape

### Key Regulations Affecting Embedding Systems

```
┌─────────────────────────────────────────────────────────────────┐
│              Privacy Regulations Overview                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GDPR (EU)                     CCPA/CPRA (California)          │
│  ───────────                   ────────────────────             │
│  • Applies to EU residents     • California residents           │
│  • Extraterritorial reach      • Revenue/data thresholds       │
│  • Consent-based processing    • Opt-out focused               │
│  • Right to erasure (Art. 17)  • Right to delete               │
│  • Data portability            • Right to know                 │
│  • 72-hour breach notice       • 30-day response window        │
│  • Fines: €20M or 4% revenue   • Fines: $7,500/violation       │
│                                                                 │
│  HIPAA (US Healthcare)         SOC 2 (Industry Standard)       │
│  ──────────────────────        ─────────────────────           │
│  • PHI protection required     • Trust service criteria        │
│  • BAA with vendors            • Annual audit                  │
│  • Encryption mandates         • Security, availability        │
│  • Audit trail requirements    • Confidentiality, privacy      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## GDPR Requirements

### Lawful Basis for Processing

Before creating embeddings from personal data, establish lawful basis:

```python
class GDPRProcessingBasis:
    """
    Document lawful basis for embedding personal data.
    """
    CONSENT = "consent"              # User explicitly agreed
    CONTRACT = "contract"            # Necessary for service delivery
    LEGITIMATE_INTEREST = "legitimate_interest"  # Business need, balanced
    LEGAL_OBLIGATION = "legal_obligation"        # Required by law
    
    @staticmethod
    def validate_basis(data_type: str, purpose: str) -> dict:
        """Validate and document processing basis."""
        
        # Customer support embeddings - typically contract/legitimate interest
        if purpose == "customer_support_search":
            return {
                "basis": GDPRProcessingBasis.CONTRACT,
                "justification": "Processing necessary to provide support services",
                "data_minimization": "Only support-relevant content embedded",
                "retention": "Deleted after support relationship ends"
            }
        
        # Analytics embeddings - typically legitimate interest
        if purpose == "content_analytics":
            return {
                "basis": GDPRProcessingBasis.LEGITIMATE_INTEREST,
                "justification": "Business need to understand content patterns",
                "balancing_test": "Low privacy impact, anonymized where possible",
                "opt_out": "Users can opt out via settings"
            }
        
        # Marketing embeddings - typically consent
        if purpose == "personalized_recommendations":
            return {
                "basis": GDPRProcessingBasis.CONSENT,
                "justification": "User consented to personalized experience",
                "consent_record": "Stored in consent management platform",
                "withdrawal": "Embeddings deleted on consent withdrawal"
            }
```

### Data Subject Rights

| Right | Vector Database Implication | Implementation |
|-------|----------------------------|----------------|
| Access (Art. 15) | Provide copy of stored embeddings | Export with metadata |
| Rectification (Art. 16) | Update embeddings when source changes | Re-embed on update |
| Erasure (Art. 17) | Delete embeddings on request | Implement deletion API |
| Portability (Art. 20) | Export in machine-readable format | Export as JSON/vectors |
| Objection (Art. 21) | Stop processing for certain purposes | Remove from specific indexes |

### Data Processing Agreement (DPA)

When using hosted vector databases:

```
┌─────────────────────────────────────────────────────────────────┐
│              DPA Checklist for Vector DB Vendors                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ Vendor is processor, you are controller                     │
│  □ Processing purposes clearly defined                         │
│  □ Sub-processor list available                                │
│  □ Data deletion obligations specified                         │
│  □ Breach notification procedures (72 hours)                   │
│  □ Audit rights included                                       │
│  □ Data transfer mechanisms (SCCs, adequacy)                   │
│  □ Technical and organizational measures documented            │
│  □ Return/deletion of data on termination                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Right to Deletion Implementation

### The Challenge with Embeddings

```
┌─────────────────────────────────────────────────────────────────┐
│              Deletion Complexity                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STRAIGHTFORWARD:                                               │
│  • Delete vector by ID from database                           │
│  • Delete source document                                       │
│  • Delete from backups (within retention period)               │
│                                                                 │
│  COMPLEX:                                                       │
│  • Aggregate embeddings (multiple sources combined)            │
│  • Fine-tuned models trained on user data                      │
│  • Cached search results containing user data                  │
│  • Audit logs mentioning user identifiers                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Deletion Implementation

```python
from datetime import datetime
from typing import Optional, List
import logging

class GDPRDeletionService:
    """
    Handle GDPR Article 17 deletion requests for embedding systems.
    """
    def __init__(self, vector_db, source_db, audit_logger):
        self.vector_db = vector_db
        self.source_db = source_db
        self.audit = audit_logger
    
    def process_deletion_request(
        self, 
        user_id: str, 
        request_id: str,
        verification_token: str
    ) -> dict:
        """
        Process a complete deletion request.
        GDPR requires completion within 30 days.
        """
        # Step 1: Verify request authenticity
        if not self._verify_request(user_id, verification_token):
            raise ValueError("Could not verify deletion request")
        
        deletion_record = {
            "request_id": request_id,
            "user_id": user_id,
            "requested_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "components": {}
        }
        
        try:
            # Step 2: Find all user data across systems
            user_data = self._find_all_user_data(user_id)
            
            # Step 3: Delete from vector database
            vector_result = self._delete_vectors(user_id, user_data["vector_ids"])
            deletion_record["components"]["vectors"] = vector_result
            
            # Step 4: Delete source documents
            source_result = self._delete_sources(user_id, user_data["source_ids"])
            deletion_record["components"]["sources"] = source_result
            
            # Step 5: Invalidate caches
            cache_result = self._invalidate_caches(user_id)
            deletion_record["components"]["caches"] = cache_result
            
            # Step 6: Schedule backup deletion
            backup_result = self._schedule_backup_deletion(user_id)
            deletion_record["components"]["backups"] = backup_result
            
            deletion_record["status"] = "completed"
            deletion_record["completed_at"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            deletion_record["status"] = "failed"
            deletion_record["error"] = str(e)
            raise
        
        finally:
            # Always audit the deletion attempt
            self.audit.log_deletion(deletion_record)
        
        return deletion_record
    
    def _delete_vectors(self, user_id: str, vector_ids: List[str]) -> dict:
        """Delete all vectors associated with user."""
        deleted_count = 0
        
        for collection in self.vector_db.list_collections():
            # Delete by user_id metadata filter
            result = self.vector_db.delete(
                collection_name=collection,
                filter={"user_id": user_id}
            )
            deleted_count += result.deleted_count
            
            # Also delete by explicit IDs if available
            if vector_ids:
                result = self.vector_db.delete(
                    collection_name=collection,
                    ids=vector_ids
                )
                deleted_count += result.deleted_count
        
        return {
            "deleted_count": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _find_all_user_data(self, user_id: str) -> dict:
        """Find all data associated with user across systems."""
        return {
            "vector_ids": self._find_user_vectors(user_id),
            "source_ids": self._find_user_sources(user_id),
            "cache_keys": self._find_user_caches(user_id)
        }
    
    def generate_deletion_certificate(self, request_id: str) -> dict:
        """
        Generate certificate of deletion for compliance records.
        """
        deletion_record = self.audit.get_deletion_record(request_id)
        
        return {
            "certificate_id": f"DEL-{request_id}",
            "request_id": request_id,
            "user_id_hash": hash(deletion_record["user_id"]),  # Don't store actual ID
            "requested_at": deletion_record["requested_at"],
            "completed_at": deletion_record.get("completed_at"),
            "systems_cleared": list(deletion_record["components"].keys()),
            "certification": "All personal data has been deleted per GDPR Article 17",
            "generated_at": datetime.utcnow().isoformat()
        }
```

### Deletion Verification

```python
class DeletionVerifier:
    """
    Verify that deletion was complete.
    """
    def verify_deletion(self, user_id: str) -> dict:
        """Verify no user data remains in system."""
        verification = {
            "user_id": user_id,
            "verified_at": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check vector database
        for collection in self.vector_db.list_collections():
            results = self.vector_db.search(
                collection_name=collection,
                filter={"user_id": user_id},
                limit=1
            )
            verification["checks"][f"vectors_{collection}"] = len(results) == 0
        
        # Check source database
        source_count = self.source_db.count({"user_id": user_id})
        verification["checks"]["source_documents"] = source_count == 0
        
        # Overall verification
        verification["complete"] = all(verification["checks"].values())
        
        return verification
```

---

## Data Residency and Sovereignty

### Region Selection

```python
class DataResidencyManager:
    """
    Manage vector database deployments for data residency compliance.
    """
    # Region mapping for major regulations
    REGION_REQUIREMENTS = {
        "gdpr_eu": ["eu-west-1", "eu-central-1", "eu-north-1"],
        "gdpr_uk": ["eu-west-2"],  # UK region
        "ccpa": ["us-west-1", "us-west-2", "us-east-1"],
        "china_pipl": ["cn-north-1", "cn-northwest-1"],
        "australia_privacy": ["ap-southeast-2"],
        "canada_pipeda": ["ca-central-1"]
    }
    
    def get_allowed_regions(self, user_country: str) -> list:
        """Get allowed regions based on user's country."""
        if user_country in ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL"]:
            return self.REGION_REQUIREMENTS["gdpr_eu"]
        elif user_country == "GB":
            return self.REGION_REQUIREMENTS["gdpr_uk"]
        elif user_country == "US":
            return self.REGION_REQUIREMENTS["ccpa"]
        elif user_country == "CN":
            return self.REGION_REQUIREMENTS["china_pipl"]
        else:
            # Default to EU regulations as most restrictive
            return self.REGION_REQUIREMENTS["gdpr_eu"]
    
    def route_to_region(self, user_country: str) -> str:
        """Route user to appropriate regional deployment."""
        allowed = self.get_allowed_regions(user_country)
        # Return primary region for country
        return allowed[0] if allowed else "eu-west-1"
```

### Multi-Region Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Region Deployment for Compliance              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   EU Region     │    │   US Region     │                    │
│  │  (eu-west-1)    │    │  (us-west-2)    │                    │
│  │                 │    │                 │                    │
│  │  ┌───────────┐  │    │  ┌───────────┐  │                    │
│  │  │ Vector DB │  │    │  │ Vector DB │  │                    │
│  │  │ (EU data) │  │    │  │ (US data) │  │                    │
│  │  └───────────┘  │    │  └───────────┘  │                    │
│  │                 │    │                 │                    │
│  │  ┌───────────┐  │    │  ┌───────────┐  │                    │
│  │  │ Embedding │  │    │  │ Embedding │  │                    │
│  │  │  Service  │  │    │  │  Service  │  │                    │
│  │  └───────────┘  │    │  └───────────┘  │                    │
│  └────────▲────────┘    └────────▲────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │  Geo-Router   │                                  │
│              │ (Edge Layer)  │                                  │
│              └───────────────┘                                  │
│                                                                 │
│  Key Points:                                                    │
│  • Data never crosses regional boundaries                      │
│  • Embedding model deployed in each region                     │
│  • User routed based on location/preference                    │
│  • Audit logs kept in respective regions                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## CCPA Compliance

### Key Differences from GDPR

| Aspect | GDPR | CCPA/CPRA |
|--------|------|-----------|
| Consent model | Opt-in | Opt-out |
| Right to delete | Yes | Yes (with exceptions) |
| Sale of data | Requires consent | Requires opt-out option |
| Private right of action | Limited | Yes (for breaches) |
| Covered entities | Any processing EU data | Revenue/data thresholds |

### CCPA Implementation

```python
class CCPAComplianceService:
    """
    Handle CCPA-specific requirements.
    """
    def __init__(self, vector_db, preference_store):
        self.vector_db = vector_db
        self.preferences = preference_store
    
    def handle_do_not_sell(self, user_id: str):
        """
        Honor "Do Not Sell My Personal Information" request.
        """
        # Record preference
        self.preferences.set(user_id, "do_not_sell", True)
        
        # Remove from any "sale" related indexes
        # (e.g., advertising, partner sharing)
        sale_related_collections = [
            "advertising_embeddings",
            "partner_recommendations",
            "audience_segments"
        ]
        
        for collection in sale_related_collections:
            self.vector_db.delete(
                collection_name=collection,
                filter={"user_id": user_id}
            )
    
    def get_disclosure(self, user_id: str) -> dict:
        """
        Provide disclosure of data collected (Right to Know).
        Must respond within 45 days.
        """
        disclosure = {
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "categories_collected": [],
            "purposes": [],
            "third_parties": [],
            "data_samples": {}
        }
        
        # Find all collections containing user data
        for collection in self.vector_db.list_collections():
            results = self.vector_db.query(
                collection_name=collection,
                filter={"user_id": user_id},
                limit=5,
                include_metadata=True
            )
            
            if results:
                disclosure["categories_collected"].append(collection)
                disclosure["data_samples"][collection] = [
                    r.metadata for r in results
                ]
        
        return disclosure
```

---

## Vendor Security Certifications

### What to Look For

```
┌─────────────────────────────────────────────────────────────────┐
│              Security Certifications Comparison                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CERTIFICATION    COVERAGE                RENEWAL              │
│  ─────────────    ────────                ───────              │
│  SOC 2 Type II    Security, availability  Annual audit         │
│  ISO 27001        InfoSec management      3-year cycle         │
│  HIPAA            Healthcare data         Ongoing compliance   │
│  PCI DSS          Payment data            Annual assessment    │
│  FedRAMP          US government           Annual audit         │
│  GDPR compliance  EU data protection      Continuous           │
│                                                                 │
│  VECTOR DB VENDOR CERTIFICATIONS (as of 2024):                 │
│                                                                 │
│  Pinecone:  SOC 2 Type II, GDPR, HIPAA (Enterprise)           │
│  Qdrant:    Self-hosted (your responsibility), Cloud (SOC 2)  │
│  Weaviate:  SOC 2 Type II (Cloud)                              │
│  Milvus:    Self-hosted (your responsibility)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Due Diligence Checklist

```python
class VendorComplianceChecker:
    """
    Evaluate vector database vendor compliance.
    """
    REQUIRED_CERTIFICATIONS = {
        "healthcare": ["HIPAA", "SOC 2 Type II"],
        "fintech": ["SOC 2 Type II", "PCI DSS"],
        "government": ["FedRAMP", "SOC 2 Type II"],
        "general": ["SOC 2 Type II"],
        "eu_operations": ["GDPR compliance", "SOC 2 Type II"]
    }
    
    def evaluate_vendor(self, vendor_name: str, vendor_certs: list, use_case: str) -> dict:
        """Evaluate if vendor meets requirements."""
        required = self.REQUIRED_CERTIFICATIONS.get(use_case, ["SOC 2 Type II"])
        
        evaluation = {
            "vendor": vendor_name,
            "use_case": use_case,
            "required_certifications": required,
            "vendor_certifications": vendor_certs,
            "missing": [],
            "approved": True
        }
        
        for cert in required:
            if cert not in vendor_certs:
                evaluation["missing"].append(cert)
                evaluation["approved"] = False
        
        return evaluation
```

---

## Documentation and Audit Readiness

### Records to Maintain

```python
class ComplianceDocumentation:
    """
    Maintain required compliance documentation.
    """
    def generate_processing_record(self) -> dict:
        """
        GDPR Article 30: Records of Processing Activities.
        """
        return {
            "controller": {
                "name": "Your Company Name",
                "contact": "dpo@yourcompany.com"
            },
            "processing_activities": [
                {
                    "purpose": "Semantic search for customer support",
                    "data_categories": ["Support tickets", "Customer messages"],
                    "data_subjects": ["Customers"],
                    "recipients": ["Support staff"],
                    "transfers": "No international transfers",
                    "retention": "2 years after last interaction",
                    "security_measures": [
                        "Encryption at rest (AES-256)",
                        "Encryption in transit (TLS 1.2)",
                        "Access control (RBAC)",
                        "Audit logging"
                    ]
                },
                {
                    "purpose": "Document embedding for knowledge base",
                    "data_categories": ["Internal documents"],
                    "data_subjects": ["Employees"],
                    "recipients": ["All employees"],
                    "transfers": "EU-US (Standard Contractual Clauses)",
                    "retention": "Document lifecycle + 1 year",
                    "security_measures": [
                        "Encryption at rest",
                        "Department-level access control"
                    ]
                }
            ],
            "last_updated": datetime.utcnow().isoformat(),
            "dpo_approved": True
        }
    
    def generate_dpia(self, processing_activity: str) -> dict:
        """
        Data Protection Impact Assessment template.
        Required for high-risk processing.
        """
        return {
            "activity": processing_activity,
            "assessment_date": datetime.utcnow().isoformat(),
            "necessity": "Processing is necessary for stated purpose",
            "risks": [
                {
                    "risk": "Embedding inversion revealing PII",
                    "likelihood": "Low",
                    "impact": "Medium",
                    "mitigation": "PII removed before embedding"
                },
                {
                    "risk": "Unauthorized access to embeddings",
                    "likelihood": "Low",
                    "impact": "High",
                    "mitigation": "RBAC, encryption, audit logging"
                }
            ],
            "residual_risk": "Acceptable with mitigations in place",
            "dpo_consultation": True,
            "approved": True
        }
```

---

## Summary

✅ **Establish lawful basis** before processing personal data into embeddings  
✅ **Implement right to deletion** with verification and certification  
✅ **Choose regions carefully** to meet data residency requirements  
✅ **Verify vendor certifications** match your compliance needs (SOC 2, HIPAA, etc.)  
✅ **Maintain processing records** for GDPR Article 30 compliance  
✅ **Document security measures** and conduct DPIAs for high-risk processing

---

**Next:** [Model and Data Poisoning →](./06-model-data-poisoning.md)

---

<!-- 
Sources Consulted:
- GDPR Text: https://gdpr-info.eu/
- CCPA Text: https://oag.ca.gov/privacy/ccpa
- ICO GDPR Guidance: https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/
- Pinecone Security Docs: https://docs.pinecone.io/guides/security/overview
-->
