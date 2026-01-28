---
title: "Compliance and Data Privacy"
---

# Compliance and Data Privacy

## Introduction

Regulatory compliance and data privacy requirements significantly impact model selection. Using the wrong provider or model can expose your organization to legal and reputational risks.

### What We'll Cover

- Data processing agreements
- GDPR considerations
- HIPAA compliance
- Data residency requirements
- Audit logging

---

## Key Regulations

### Regulatory Landscape

| Regulation | Region | Key Requirements |
|------------|--------|------------------|
| GDPR | EU/EEA | Data minimization, consent, right to deletion |
| CCPA | California | Consumer privacy rights, opt-out |
| HIPAA | USA | Protected health information security |
| SOC 2 | Global | Security, availability, confidentiality |
| PCI DSS | Global | Payment card data protection |
| LGPD | Brazil | Data protection (similar to GDPR) |

### Compliance Matrix

```python
compliance_by_provider = {
    "openai_api": {
        "gdpr": "Partial (DPA available)",
        "hipaa": "No BAA available",
        "soc2": "Type II certified",
        "data_retention": "30 days (can request 0)",
        "training_on_data": "Opt-out available"
    },
    "azure_openai": {
        "gdpr": "Yes (Microsoft DPA)",
        "hipaa": "BAA available",
        "soc2": "Type II certified",
        "data_retention": "Configurable",
        "training_on_data": "No"
    },
    "anthropic_api": {
        "gdpr": "Partial (DPA available)",
        "hipaa": "No BAA",
        "soc2": "Type II certified",
        "data_retention": "30 days",
        "training_on_data": "Opt-out available"
    },
    "aws_bedrock": {
        "gdpr": "Yes (AWS DPA)",
        "hipaa": "BAA available",
        "soc2": "Type II certified",
        "data_retention": "None (pass-through)",
        "training_on_data": "No"
    },
    "google_vertex": {
        "gdpr": "Yes (Google DPA)",
        "hipaa": "BAA available",
        "soc2": "Type II certified",
        "data_retention": "Configurable",
        "training_on_data": "No (by default)"
    },
    "self_hosted": {
        "gdpr": "Your responsibility",
        "hipaa": "Your responsibility",
        "soc2": "Your responsibility",
        "data_retention": "Full control",
        "training_on_data": "No"
    }
}
```

---

## Data Processing Agreements

### What to Look For

```python
dpa_checklist = {
    "data_handling": [
        "How is data transmitted (encrypted in transit)?",
        "How is data stored (encrypted at rest)?",
        "Who has access to the data?",
        "What subprocessors are used?",
    ],
    "retention": [
        "How long is data retained?",
        "Can retention be customized?",
        "How is data deleted?",
    ],
    "training": [
        "Is data used to train models?",
        "Can training use be disabled?",
        "Is this opt-in or opt-out?",
    ],
    "rights": [
        "Can data be exported?",
        "Can data be deleted on request?",
        "What happens at contract termination?",
    ]
}
```

### Implementation

```python
class ComplianceChecker:
    """Verify compliance requirements before API call"""
    
    def __init__(self, requirements: list):
        self.requirements = requirements
    
    def check_provider(self, provider: str) -> dict:
        """Check if provider meets requirements"""
        
        provider_compliance = compliance_by_provider.get(provider, {})
        
        results = {}
        for req in self.requirements:
            status = provider_compliance.get(req.lower(), "unknown")
            results[req] = {
                "status": status,
                "compliant": self._is_compliant(status)
            }
        
        return {
            "provider": provider,
            "requirements": results,
            "all_compliant": all(r["compliant"] for r in results.values())
        }
    
    def _is_compliant(self, status: str) -> bool:
        compliant_statuses = ["yes", "available", "type ii certified", "full control"]
        return any(s in status.lower() for s in compliant_statuses)

# Usage
checker = ComplianceChecker(["GDPR", "HIPAA", "SOC2"])
result = checker.check_provider("azure_openai")
print(f"Azure OpenAI compliant: {result['all_compliant']}")
```

---

## GDPR Considerations

### Key Requirements

```python
gdpr_requirements = {
    "lawful_basis": "Need legal basis for processing (consent, contract, etc.)",
    "purpose_limitation": "Only use data for stated purposes",
    "data_minimization": "Collect only necessary data",
    "storage_limitation": "Don't keep data longer than needed",
    "integrity_confidentiality": "Keep data secure",
    "accountability": "Demonstrate compliance",
    "data_subject_rights": [
        "Right of access",
        "Right to rectification",
        "Right to erasure",
        "Right to data portability",
        "Right to object",
    ]
}
```

### GDPR-Compliant AI Implementation

```python
class GDPRCompliantAI:
    """AI client with GDPR compliance features"""
    
    def __init__(self, client):
        self.client = client
        self.processing_log = []
    
    def process_with_consent(
        self,
        data: str,
        user_id: str,
        consent_record: dict
    ) -> str:
        """Process data only with valid consent"""
        
        # Verify consent
        if not self._verify_consent(consent_record):
            raise ValueError("Valid consent required for processing")
        
        # Minimize data
        minimized_data = self._minimize_data(data)
        
        # Log processing
        self._log_processing(user_id, "ai_inference")
        
        # Process
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": minimized_data}]
        )
        
        return response.choices[0].message.content
    
    def _verify_consent(self, consent: dict) -> bool:
        """Verify consent is valid and current"""
        required_fields = ["user_id", "timestamp", "purposes", "signature"]
        return all(f in consent for f in required_fields)
    
    def _minimize_data(self, data: str) -> str:
        """Remove unnecessary personal data"""
        # Implement PII detection and removal
        return data
    
    def _log_processing(self, user_id: str, purpose: str):
        """Log processing for accountability"""
        self.processing_log.append({
            "user_id": user_id,
            "purpose": purpose,
            "timestamp": time.time(),
            "processor": "ai_service"
        })
    
    def handle_deletion_request(self, user_id: str):
        """Handle GDPR deletion request"""
        # Remove from logs
        self.processing_log = [
            log for log in self.processing_log
            if log["user_id"] != user_id
        ]
        
        # Request deletion from provider (if supported)
        # self.client.delete_user_data(user_id)
```

---

## HIPAA Compliance

### Protected Health Information (PHI)

```python
phi_identifiers = [
    "names",
    "geographic_data",
    "dates",  # except year
    "phone_numbers",
    "fax_numbers",
    "email_addresses",
    "ssn",
    "medical_record_numbers",
    "health_plan_numbers",
    "account_numbers",
    "certificate_numbers",
    "vehicle_identifiers",
    "device_identifiers",
    "urls",
    "ip_addresses",
    "biometric_identifiers",
    "photos",
    "any_unique_identifier"
]
```

### HIPAA-Compliant Providers

```python
hipaa_compliant_options = {
    "azure_openai": {
        "baa_available": True,
        "phi_allowed": True,
        "notes": "Requires BAA execution with Microsoft"
    },
    "aws_bedrock": {
        "baa_available": True,
        "phi_allowed": True,
        "notes": "Part of AWS BAA (if healthcare workload)"
    },
    "google_vertex_healthcare": {
        "baa_available": True,
        "phi_allowed": True,
        "notes": "Healthcare AI specific offering"
    },
    "self_hosted": {
        "baa_available": "N/A",
        "phi_allowed": True,
        "notes": "Full control, your responsibility"
    }
}

def get_hipaa_provider() -> str:
    """Get recommended provider for HIPAA workloads"""
    return "azure_openai"  # Most established BAA process
```

### PHI De-identification

```python
import re

class PHIDeidentifier:
    """Remove PHI before sending to non-HIPAA API"""
    
    def __init__(self):
        self.patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            "date": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "mrn": r"\bMRN:?\s*\d+\b",
        }
    
    def deidentify(self, text: str) -> str:
        """Remove PHI from text"""
        
        for phi_type, pattern in self.patterns.items():
            text = re.sub(pattern, f"[{phi_type.upper()}_REDACTED]", text)
        
        return text
    
    def is_safe(self, text: str) -> bool:
        """Check if text contains PHI"""
        for pattern in self.patterns.values():
            if re.search(pattern, text):
                return False
        return True

# Usage
deidentifier = PHIDeidentifier()
safe_text = deidentifier.deidentify("Patient John Doe, SSN 123-45-6789")
# Result: "Patient John Doe, SSN [SSN_REDACTED]"
```

---

## Data Residency

### Geographic Requirements

```python
data_residency_options = {
    "eu_only": {
        "azure": ["westeurope", "northeurope", "germanywestcentral"],
        "gcp": ["europe-west1", "europe-west4", "europe-north1"],
        "aws": ["eu-west-1", "eu-central-1", "eu-north-1"]
    },
    "us_only": {
        "azure": ["eastus", "westus", "centralus"],
        "gcp": ["us-central1", "us-east1", "us-west1"],
        "aws": ["us-east-1", "us-west-2"]
    },
    "apac": {
        "azure": ["eastasia", "southeastasia", "australiaeast"],
        "gcp": ["asia-northeast1", "australia-southeast1"],
        "aws": ["ap-northeast-1", "ap-southeast-1"]
    }
}

def select_region_for_residency(
    residency_requirement: str,
    cloud_provider: str
) -> list:
    """Get regions that meet residency requirements"""
    
    regions = data_residency_options.get(residency_requirement, {})
    return regions.get(cloud_provider, [])
```

---

## Audit Logging

### Comprehensive Logging

```python
import json
from datetime import datetime

class AuditLogger:
    """Log all AI interactions for compliance"""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def log_request(
        self,
        request_id: str,
        user_id: str,
        model: str,
        prompt: str,
        response: str,
        metadata: dict = None
    ):
        """Log AI request for audit"""
        
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "model": model,
            "prompt_hash": self._hash(prompt),  # Don't store raw prompt
            "response_hash": self._hash(response),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {}
        }
        
        self.storage.write(log_entry)
    
    def _hash(self, text: str) -> str:
        """Hash text for logging without storing content"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Generate audit report for compliance"""
        
        logs = self.storage.query(start_date, end_date)
        
        return {
            "period": f"{start_date} to {end_date}",
            "total_requests": len(logs),
            "unique_users": len(set(l["user_id"] for l in logs)),
            "models_used": list(set(l["model"] for l in logs)),
            "request_breakdown": self._breakdown_by_day(logs)
        }
```

---

## Summary

✅ **Know your regulations** - GDPR, HIPAA, SOC 2, etc.

✅ **Check provider compliance** - BAAs, DPAs, certifications

✅ **Minimize data** - Only send what's necessary

✅ **Consider data residency** - Use appropriate regions

✅ **Log everything** - Maintain audit trails

**Next:** [Open Source vs Proprietary](./07-open-source-vs-proprietary.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [API Availability](./05-api-availability-reliability.md) | [Model Selection](./00-model-selection-criteria.md) | [Open Source vs Proprietary](./07-open-source-vs-proprietary.md) |

