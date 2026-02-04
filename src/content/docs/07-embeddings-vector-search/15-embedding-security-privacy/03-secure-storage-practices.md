---
title: "Secure Storage Practices"
---

# Secure Storage Practices

## Introduction

Embeddings are assets. They encode your organization's knowledge, customer interactions, and potentially sensitive information. Securing vector storage requires encryption, access controls, and network isolation—the same principles protecting any sensitive database.

This lesson covers practical security configurations for production vector databases.

### What We'll Cover

- Encryption at rest and in transit
- API key management and rotation
- Access control and audit logging
- Network security and isolation
- Key management for encrypted vectors

### Prerequisites

- Understanding of [PII in embeddings](./01-pii-in-embeddings.md)
- Familiarity with database security concepts
- Basic knowledge of TLS/encryption

---

## Encryption at Rest

### Why It Matters

```
┌─────────────────────────────────────────────────────────────────┐
│              Encryption at Rest Threat Model                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  THREATS MITIGATED:                                             │
│  • Stolen/decommissioned storage devices                       │
│  • Unauthorized filesystem access                               │
│  • Snapshot/backup exposure                                     │
│  • Cloud provider insider threats                               │
│                                                                 │
│  THREATS NOT MITIGATED:                                         │
│  • Application-level attacks (SQL injection equivalent)        │
│  • Authorized user misuse                                       │
│  • Memory-based attacks                                         │
│  • API-level exfiltration                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Vector Database Encryption Support

| Database | Encryption at Rest | Algorithm | Key Management |
|----------|-------------------|-----------|----------------|
| Pinecone | ✅ Default | AES-256 | Pinecone-managed or CMEK |
| Qdrant | ✅ Via filesystem | OS-level | User-managed |
| Weaviate | ✅ Optional | AES-256 | User-managed |
| Milvus | ✅ Via filesystem | OS-level | User-managed |
| Chroma | ❌ Default | N/A | N/A |

### Pinecone: Encryption with CMEK

Customer-Managed Encryption Keys (CMEK) provide control over your encryption keys via AWS KMS:

```python
# Pinecone CMEK Configuration (via Console/API)
# Enterprise plan required

# Step 1: Create AWS KMS key with proper policy
"""
KMS Key Policy must include:
{
    "Sid": "Allow Pinecone",
    "Effect": "Allow",
    "Principal": {
        "AWS": "arn:aws:iam::PINECONE_ACCOUNT_ID:root"
    },
    "Action": [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:GenerateDataKey*"
    ],
    "Resource": "*"
}
"""

# Step 2: Create index with CMEK
import pinecone

# Index creation specifies KMS key ARN
# (configured in Pinecone console for Enterprise)

# Result: Your data encrypted with YOUR key
# Revoke key access = Pinecone can't decrypt your data
```

### Qdrant: Filesystem-Level Encryption

Qdrant relies on the underlying filesystem for encryption:

```bash
# Option 1: LUKS encryption for Linux
# Encrypt the volume where Qdrant stores data

# Create encrypted volume
cryptsetup luksFormat /dev/sdb1
cryptsetup luksOpen /dev/sdb1 qdrant_data
mkfs.ext4 /dev/mapper/qdrant_data
mount /dev/mapper/qdrant_data /var/lib/qdrant

# Option 2: Cloud provider encryption
# AWS EBS: Enable encryption on volume creation
# GCP Persistent Disk: Encryption by default
# Azure Managed Disks: Enable Azure Disk Encryption
```

```yaml
# Qdrant config.yaml - ensure data directory is on encrypted volume
storage:
  storage_path: /var/lib/qdrant/storage  # Mount encrypted volume here
```

---

## Encryption in Transit

### TLS Configuration

All production vector databases should use TLS 1.2+:

```yaml
# Qdrant TLS Configuration (config.yaml)
service:
  enable_tls: true
  
tls:
  cert: /path/to/server.crt    # Server certificate
  key: /path/to/server.key     # Private key
  ca_cert: /path/to/ca.crt     # CA certificate (for client verification)
  verify_https_client: false   # Set true for mTLS
```

```python
# Qdrant Client with TLS
from qdrant_client import QdrantClient

# Connect with TLS verification
client = QdrantClient(
    url="https://your-qdrant-server:6333",
    api_key="your-api-key",
    https=True,
    verify=True  # Verify server certificate
)

# For self-signed certificates in development
client = QdrantClient(
    url="https://localhost:6333",
    api_key="your-api-key",
    https=True,
    verify="/path/to/ca.crt"  # Custom CA certificate
)
```

### Pinecone: TLS Enforced

```python
import pinecone

# Pinecone enforces TLS 1.2 by default
# All connections use HTTPS

pc = pinecone.Pinecone(api_key="your-api-key")

# The SDK handles TLS automatically
# API endpoint: https://controller.{environment}.pinecone.io
# Data plane: https://{index-name}-{project}.svc.{environment}.pinecone.io
```

---

## API Key Management

### Key Types and Permissions

```
┌─────────────────────────────────────────────────────────────────┐
│              API Key Permission Matrix                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PINECONE KEY TYPES:                                            │
│                                                                 │
│  ┌─────────────────┬────────────┬────────────┬───────────────┐ │
│  │  Permission     │ ReadWrite  │ ReadOnly   │ None          │ │
│  ├─────────────────┼────────────┼────────────┼───────────────┤ │
│  │ Create Index    │ ✅         │ ❌         │ ❌           │ │
│  │ Delete Index    │ ✅         │ ❌         │ ❌           │ │
│  │ Query Vectors   │ ✅         │ ✅         │ ❌           │ │
│  │ Upsert Vectors  │ ✅         │ ❌         │ ❌           │ │
│  │ Delete Vectors  │ ✅         │ ❌         │ ❌           │ │
│  └─────────────────┴────────────┴────────────┴───────────────┘ │
│                                                                 │
│  QDRANT KEY TYPES (v1.7.0+):                                    │
│                                                                 │
│  ┌─────────────────┬────────────┬────────────┐                 │
│  │  Operation      │ api_key    │ read_only  │                 │
│  ├─────────────────┼────────────┼────────────┤                 │
│  │ Search          │ ✅         │ ✅         │                 │
│  │ Read Points     │ ✅         │ ✅         │                 │
│  │ Write Points    │ ✅         │ ❌         │                 │
│  │ Manage Collections │ ✅      │ ❌         │                 │
│  └─────────────────┴────────────┴────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Qdrant API Key Configuration

```yaml
# config.yaml - API Key Setup
service:
  api_key: "${QDRANT_API_KEY}"  # Full access key
  read_only_api_key: "${QDRANT_READ_ONLY_KEY}"  # Read-only access
```

```bash
# Environment variables (production)
export QDRANT_API_KEY=$(openssl rand -base64 32)
export QDRANT_READ_ONLY_KEY=$(openssl rand -base64 32)
```

```python
# Application usage - use read-only for query services
from qdrant_client import QdrantClient
import os

# Query service uses read-only key
query_client = QdrantClient(
    url="https://qdrant.example.com:6333",
    api_key=os.environ["QDRANT_READ_ONLY_KEY"]
)

# Admin service uses full key
admin_client = QdrantClient(
    url="https://qdrant.example.com:6333",
    api_key=os.environ["QDRANT_API_KEY"]
)
```

### Key Rotation Strategy

```python
import os
from datetime import datetime, timedelta
import secrets

class APIKeyManager:
    """
    Manages API key rotation for vector databases.
    """
    def __init__(self, rotation_days: int = 90):
        self.rotation_days = rotation_days
    
    def generate_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def should_rotate(self, created_at: datetime) -> bool:
        """Check if key should be rotated."""
        age = datetime.now() - created_at
        return age > timedelta(days=self.rotation_days)
    
    def rotate_keys(self, current_key: str) -> dict:
        """
        Generate new key while keeping old one temporarily valid.
        
        Rotation process:
        1. Generate new key
        2. Add new key to vector DB config
        3. Update all applications to use new key
        4. Remove old key after transition period
        """
        new_key = self.generate_key()
        return {
            "new_key": new_key,
            "old_key_valid_until": datetime.now() + timedelta(days=7),
            "action_required": "Update applications within 7 days"
        }

# Example rotation workflow
key_manager = APIKeyManager(rotation_days=90)
rotation_result = key_manager.rotate_keys(os.environ["QDRANT_API_KEY"])
```

---

## JWT-Based Access Control (Qdrant)

For fine-grained, per-collection access control:

```yaml
# config.yaml - JWT RBAC Configuration (v1.9.0+)
service:
  api_key: "${QDRANT_API_KEY}"  # Still needed as fallback
  jwt_rbac: true  # Enable JWT-based RBAC
```

```python
import jwt
from datetime import datetime, timedelta

class QdrantTokenManager:
    """
    Generate JWT tokens with collection-level access control.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_token(
        self, 
        collections_access: dict,
        expires_in: timedelta = timedelta(hours=1)
    ) -> str:
        """
        Generate JWT with specific collection permissions.
        
        Args:
            collections_access: {collection_name: "r" or "rw"}
            expires_in: Token validity period
        """
        payload = {
            "exp": datetime.utcnow() + expires_in,
            "access": collections_access
        }
        
        return jwt.encode(payload, self.api_key, algorithm="HS256")

# Usage: Generate limited token for specific tenant
token_manager = QdrantTokenManager(os.environ["QDRANT_API_KEY"])

# Token with read-only access to customer_A collection only
limited_token = token_manager.generate_token(
    collections_access={
        "customer_A_embeddings": "r"  # Read-only
    },
    expires_in=timedelta(hours=1)
)

# Client uses JWT token
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://qdrant.example.com:6333",
    api_key=limited_token  # JWT token instead of API key
)

# This client can ONLY read from customer_A_embeddings
# All other collections are inaccessible
```

---

## Audit Logging

### Pinecone Audit Logs (Enterprise)

```python
# Audit log entry format (JSON)
"""
{
    "timestamp": "2024-01-15T10:30:45Z",
    "event_type": "data.upsert",
    "project": "production",
    "index": "product-embeddings",
    "user_id": "user_abc123",
    "api_key_id": "key_xyz789",
    "source_ip": "203.0.113.42",
    "vector_count": 500,
    "status": "success"
}
"""

# Audit log analysis for security monitoring
def analyze_audit_logs(logs: list) -> dict:
    """Analyze audit logs for security concerns."""
    from collections import defaultdict
    
    analysis = {
        "high_volume_users": [],
        "unusual_ips": [],
        "failed_operations": []
    }
    
    user_operations = defaultdict(int)
    ip_operations = defaultdict(int)
    
    for log in logs:
        user_operations[log["user_id"]] += 1
        ip_operations[log["source_ip"]] += 1
        
        if log["status"] != "success":
            analysis["failed_operations"].append(log)
    
    # Flag high-volume users (potential exfiltration)
    for user, count in user_operations.items():
        if count > 10000:  # Threshold
            analysis["high_volume_users"].append({
                "user": user,
                "operation_count": count
            })
    
    return analysis
```

### Custom Audit Logging

```python
import logging
from functools import wraps
from datetime import datetime
import json

# Configure structured logging
logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)
audit_logger = logging.getLogger('vector_audit')

def audit_vector_operation(operation: str):
    """Decorator to audit vector database operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            try:
                result = func(*args, **kwargs)
                status = "success"
                error = None
            except Exception as e:
                result = None
                status = "failure"
                error = str(e)
                raise
            finally:
                audit_entry = {
                    "timestamp": start_time.isoformat(),
                    "operation": operation,
                    "function": func.__name__,
                    "status": status,
                    "error": error,
                    "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "user_id": kwargs.get("user_id", "unknown"),
                    "collection": kwargs.get("collection", "unknown")
                }
                audit_logger.info(json.dumps(audit_entry))
            
            return result
        return wrapper
    return decorator

# Usage
class AuditedVectorStore:
    @audit_vector_operation("upsert")
    def upsert(self, collection: str, vectors: list, user_id: str):
        # Upsert logic
        pass
    
    @audit_vector_operation("query")
    def query(self, collection: str, query_vector: list, user_id: str):
        # Query logic
        pass
    
    @audit_vector_operation("delete")
    def delete(self, collection: str, ids: list, user_id: str):
        # Delete logic
        pass
```

---

## Network Security

### Binding and Isolation

```yaml
# Qdrant: Network binding configuration
service:
  # Development: bind to localhost only
  host: 127.0.0.1
  
  # Production: bind to private network interface
  # host: 10.0.1.100  # Private IP
  
  http_port: 6333
  grpc_port: 6334
```

### Firewall Rules

```bash
# Only allow access from application servers
# Using iptables (Linux)

# Allow from app server subnet
iptables -A INPUT -p tcp --dport 6333 -s 10.0.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 6334 -s 10.0.1.0/24 -j ACCEPT

# Block all other access
iptables -A INPUT -p tcp --dport 6333 -j DROP
iptables -A INPUT -p tcp --dport 6334 -j DROP
```

### Pinecone Private Endpoints

```python
# AWS PrivateLink for Pinecone (Enterprise)
# Access Pinecone without traversing public internet

# Step 1: Create VPC endpoint in AWS
"""
aws ec2 create-vpc-endpoint \
    --vpc-id vpc-xxx \
    --service-name com.amazonaws.vpce.us-east-1.pinecone \
    --subnet-ids subnet-xxx \
    --security-group-ids sg-xxx
"""

# Step 2: Configure DNS resolution
# Pinecone endpoint resolves to private IP

# Step 3: Use normally - traffic stays private
import pinecone

pc = pinecone.Pinecone(api_key="your-key")
# Traffic routes through PrivateLink, not public internet
```

---

## Production Security Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│              Vector DB Security Checklist                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENCRYPTION                                                     │
│  □ Encryption at rest enabled                                  │
│  □ TLS 1.2+ enforced for all connections                       │
│  □ CMEK configured (if available/required)                     │
│                                                                 │
│  AUTHENTICATION                                                 │
│  □ Strong API keys generated (32+ bytes)                       │
│  □ Read-only keys for query services                           │
│  □ Key rotation schedule established                           │
│  □ Keys stored in secrets manager (not code)                   │
│                                                                 │
│  AUTHORIZATION                                                  │
│  □ Least privilege access applied                              │
│  □ Per-collection access control (if multi-tenant)            │
│  □ Service account separation                                  │
│                                                                 │
│  NETWORK                                                        │
│  □ Vector DB not exposed to internet                           │
│  □ Firewall rules restrict access                              │
│  □ Private endpoints configured (cloud)                        │
│                                                                 │
│  MONITORING                                                     │
│  □ Audit logging enabled                                       │
│  □ Alerts for unusual access patterns                          │
│  □ Failed authentication monitoring                            │
│                                                                 │
│  OPERATIONAL                                                    │
│  □ Backups encrypted                                           │
│  □ Disaster recovery tested                                    │
│  □ Incident response plan documented                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

✅ **Enable encryption at rest** using AES-256 or filesystem encryption  
✅ **Enforce TLS 1.2+** for all vector database connections  
✅ **Use separate API keys** for different access levels (read-only vs full)  
✅ **Rotate keys regularly** (90 days recommended)  
✅ **Implement audit logging** to detect suspicious access patterns  
✅ **Restrict network access** to private networks only

---

**Next:** [Multi-Tenant Isolation →](./04-multi-tenant-isolation.md)

---

<!-- 
Sources Consulted:
- Pinecone Security: https://docs.pinecone.io/guides/security/overview
- Qdrant Security: https://qdrant.tech/documentation/guides/security/
- AWS KMS Best Practices: https://docs.aws.amazon.com/kms/latest/developerguide/best-practices.html
- OWASP API Security: https://owasp.org/API-Security/
-->
