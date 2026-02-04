---
title: "Multi-Tenant Isolation"
---

# Multi-Tenant Isolation

## Introduction

When multiple customers share a vector database, you need isolation guarantees. One customer's queries should never return another customer's data. One tenant's misconfiguration shouldn't affect others.

This lesson covers multi-tenancy strategies across major vector databases, from full physical isolation to efficient logical partitioning.

### What We'll Cover

- Multi-tenancy architecture patterns
- Milvus isolation strategies (4 levels)
- Qdrant collection-based isolation
- Pinecone project and namespace separation
- Query-level tenant filtering
- Preventing cross-tenant data leakage

### Prerequisites

- Understanding of [secure storage practices](./03-secure-storage-practices.md)
- Familiarity with vector database architecture
- Basic knowledge of access control patterns

---

## Multi-Tenancy Architecture Patterns

### The Isolation Spectrum

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Tenancy Isolation Spectrum                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FULL ISOLATION                              SHARED RESOURCES   │
│  ◄───────────────────────────────────────────────────────────► │
│                                                                 │
│  Separate       Separate      Separate       Partition    Metadata │
│  Cluster        Database      Collection     Key          Filter │
│    │               │              │            │             │   │
│    │  Most         │              │            │   Most      │   │
│    │  Secure       │              │            │   Efficient │   │
│    │               │              │            │             │   │
│    ▼               ▼              ▼            ▼             ▼   │
│                                                                 │
│  Cost: $$$$       $$$           $$            $             $   │
│  Scale: Limited   ~64           ~65K          ~1K/coll     ∞   │
│  RBAC: Full       Full          Full          Limited      App  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Choosing the Right Strategy

| Factor | Full Isolation | Collection-Level | Partition/Filter |
|--------|---------------|------------------|------------------|
| Tenant count | < 50 | 50-10,000 | 10,000+ |
| Compliance requirements | High | Medium | Low |
| Performance isolation | Yes | Partial | No |
| Cost sensitivity | Low | Medium | High |
| Schema flexibility | Full | Per-collection | Shared |

---

## Milvus Multi-Tenancy Strategies

Milvus offers four distinct multi-tenancy approaches:

### Strategy 1: Database-Level Isolation

```python
from pymilvus import connections, db

# Connect to Milvus
connections.connect(host="localhost", port="19530")

class DatabaseLevelMultiTenancy:
    """
    One database per tenant.
    Maximum isolation, but limited to ~64 databases.
    """
    def create_tenant(self, tenant_id: str):
        """Create isolated database for tenant."""
        db_name = f"tenant_{tenant_id}"
        db.create_database(db_name)
        
        # Use the tenant's database
        db.using_database(db_name)
        
        # Create collections within tenant's database
        # Each tenant has completely separate namespace
        return db_name
    
    def use_tenant(self, tenant_id: str):
        """Switch to tenant's database."""
        db.using_database(f"tenant_{tenant_id}")
    
    def delete_tenant(self, tenant_id: str):
        """Delete tenant and all their data."""
        db.drop_database(f"tenant_{tenant_id}")

# Characteristics:
# ✅ Complete data isolation (separate storage)
# ✅ Full RBAC support
# ✅ Independent schema per tenant
# ❌ Maximum ~64 databases
# ❌ Higher resource overhead
```

### Strategy 2: Collection-Level Isolation

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

class CollectionLevelMultiTenancy:
    """
    One collection per tenant.
    Good balance of isolation and scale (~65,536 collections).
    """
    def __init__(self, db_name: str = "default"):
        self.db_name = db_name
        db.using_database(db_name)
    
    def create_tenant_collection(self, tenant_id: str, dim: int = 1536):
        """Create collection for tenant."""
        collection_name = f"tenant_{tenant_id}_vectors"
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, description=f"Vectors for tenant {tenant_id}")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
        
        return collection
    
    def get_tenant_collection(self, tenant_id: str) -> Collection:
        """Get tenant's collection."""
        return Collection(f"tenant_{tenant_id}_vectors")
    
    def search(self, tenant_id: str, query_vector: list, top_k: int = 10):
        """Search within tenant's collection only."""
        collection = self.get_tenant_collection(tenant_id)
        collection.load()
        
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k
        )
        
        return results

# Characteristics:
# ✅ Physical isolation (separate storage files)
# ✅ RBAC support
# ✅ Per-tenant schema customization
# ✅ Scale to ~65,536 tenants
# ❌ More collections = more memory overhead
```

### Strategy 3: Partition-Level Isolation

```python
class PartitionLevelMultiTenancy:
    """
    One partition per tenant within shared collection.
    More efficient, but limited to ~1,024 partitions per collection.
    """
    def __init__(self, collection_name: str = "multi_tenant_vectors"):
        self.collection = Collection(collection_name)
    
    def create_tenant_partition(self, tenant_id: str):
        """Create partition for tenant."""
        partition_name = f"tenant_{tenant_id}"
        self.collection.create_partition(partition_name)
        return partition_name
    
    def insert(self, tenant_id: str, data: list):
        """Insert into tenant's partition."""
        partition_name = f"tenant_{tenant_id}"
        self.collection.insert(data, partition_name=partition_name)
    
    def search(self, tenant_id: str, query_vector: list, top_k: int = 10):
        """Search only within tenant's partition."""
        partition_name = f"tenant_{tenant_id}"
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            partition_names=[partition_name]  # Restrict to tenant's partition
        )
        
        return results
    
    def delete_tenant(self, tenant_id: str):
        """Delete tenant's partition."""
        self.collection.drop_partition(f"tenant_{tenant_id}")

# Characteristics:
# ✅ Physical isolation within collection
# ✅ Efficient resource sharing
# ✅ Fast tenant deletion
# ❌ Limited to ~1,024 partitions per collection
# ❌ No RBAC at partition level
# ❌ Shared schema
```

### Strategy 4: Partition Key Isolation

```python
class PartitionKeyMultiTenancy:
    """
    Partition key field for automatic tenant routing.
    Most scalable approach (millions of tenants).
    """
    def create_collection_with_partition_key(self, dim: int = 1536):
        """Create collection with tenant_id as partition key."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=100, is_partition_key=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields, description="Multi-tenant vectors with partition key")
        collection = Collection(name="partition_key_vectors", schema=schema)
        
        return collection
    
    def insert(self, tenant_id: str, embeddings: list, contents: list, ids: list):
        """Insert with tenant_id - Milvus routes automatically."""
        data = [
            ids,
            [tenant_id] * len(ids),  # Same tenant_id for all records
            embeddings,
            contents
        ]
        self.collection.insert(data)
    
    def search(self, tenant_id: str, query_vector: list, top_k: int = 10):
        """Search with tenant filter - Milvus optimizes automatically."""
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=f'tenant_id == "{tenant_id}"'  # Partition key filter
        )
        
        return results

# Characteristics:
# ✅ Millions of tenants supported
# ✅ Automatic partition management
# ✅ Efficient storage
# ❌ Weaker isolation (logical only)
# ❌ No RBAC at tenant level
# ❌ Shared schema
```

### Milvus Strategy Comparison

| Strategy | Max Tenants | Isolation Level | RBAC | Schema Flexibility | Performance |
|----------|-------------|-----------------|------|-------------------|-------------|
| Database | ~64 | Complete | ✅ | Full | Isolated |
| Collection | ~65,536 | Physical | ✅ | Per-tenant | Isolated |
| Partition | ~1,024/coll | Physical | ❌ | Shared | Shared |
| Partition Key | Millions | Logical | ❌ | Shared | Optimized |

---

## Qdrant Multi-Tenancy

### Collection-Per-Tenant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantMultiTenancy:
    """
    Collection-per-tenant in Qdrant.
    """
    def __init__(self, url: str, api_key: str):
        self.client = QdrantClient(url=url, api_key=api_key)
    
    def create_tenant(self, tenant_id: str, dim: int = 1536):
        """Create collection for tenant."""
        collection_name = f"tenant_{tenant_id}"
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
        return collection_name
    
    def search(self, tenant_id: str, query_vector: list, top_k: int = 10):
        """Search within tenant's collection."""
        collection_name = f"tenant_{tenant_id}"
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return results
    
    def delete_tenant(self, tenant_id: str):
        """Delete tenant's collection."""
        self.client.delete_collection(f"tenant_{tenant_id}")
```

### JWT-Based Tenant Isolation

```python
import jwt
from datetime import datetime, timedelta

class QdrantTenantTokenManager:
    """
    Generate JWT tokens with tenant-specific collection access.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_tenant_token(self, tenant_id: str, access: str = "r") -> str:
        """
        Generate token for specific tenant's collection.
        
        Args:
            tenant_id: The tenant identifier
            access: "r" for read-only, "rw" for read-write
        """
        collection_name = f"tenant_{tenant_id}"
        
        payload = {
            "exp": datetime.utcnow() + timedelta(hours=1),
            "access": {
                collection_name: access
            }
        }
        
        return jwt.encode(payload, self.api_key, algorithm="HS256")

# Usage: Application gets tenant-specific token
token_manager = QdrantTenantTokenManager(api_key="your-api-key")

# Generate token for tenant_123 with read-only access
tenant_token = token_manager.generate_tenant_token("123", access="r")

# Client can ONLY access tenant_123's collection
tenant_client = QdrantClient(
    url="https://qdrant.example.com",
    api_key=tenant_token
)
```

---

## Pinecone Multi-Tenancy

### Namespace-Based Isolation

```python
import pinecone

class PineconeMultiTenancy:
    """
    Namespace-per-tenant in Pinecone.
    Single index, multiple namespaces.
    """
    def __init__(self, api_key: str, index_name: str):
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
    
    def upsert(self, tenant_id: str, vectors: list):
        """Upsert into tenant's namespace."""
        namespace = f"tenant_{tenant_id}"
        
        self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )
    
    def search(self, tenant_id: str, query_vector: list, top_k: int = 10):
        """Search within tenant's namespace only."""
        namespace = f"tenant_{tenant_id}"
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        return results
    
    def delete_tenant(self, tenant_id: str):
        """Delete all vectors in tenant's namespace."""
        namespace = f"tenant_{tenant_id}"
        self.index.delete(delete_all=True, namespace=namespace)

# Characteristics:
# ✅ Efficient (single index, shared infrastructure)
# ✅ Easy tenant deletion
# ✅ Automatic isolation in queries
# ❌ All tenants share index configuration (dimension, metric)
# ❌ Large tenant can impact others
```

### Project-Based Isolation

```python
# For higher isolation: separate projects
# Each project has its own:
# - API keys
# - Indexes
# - Billing

# Configure via Pinecone console:
# 1. Create project "customer_a"
# 2. Create project "customer_b"
# 3. Generate separate API keys per project

# Application routes to correct project
def get_pinecone_client(tenant_tier: str, tenant_id: str):
    """Get appropriate Pinecone client based on tenant tier."""
    if tenant_tier == "enterprise":
        # Enterprise tenants get dedicated project
        api_key = secrets_manager.get(f"pinecone_key_{tenant_id}")
    else:
        # Standard tenants share project with namespaces
        api_key = secrets_manager.get("pinecone_key_shared")
    
    return pinecone.Pinecone(api_key=api_key)
```

---

## Query-Level Tenant Filtering

### Defense in Depth: Always Filter

Even with proper isolation, add query-level tenant filtering:

```python
class SecureMultiTenantSearch:
    """
    Defense-in-depth: Always validate tenant context.
    """
    def __init__(self, vector_store, tenant_resolver):
        self.store = vector_store
        self.tenant_resolver = tenant_resolver
    
    def search(self, request, query: str, top_k: int = 10):
        """Search with mandatory tenant filtering."""
        # Step 1: Resolve tenant from request context
        tenant_id = self.tenant_resolver.get_tenant(request)
        if not tenant_id:
            raise ValueError("Tenant context required for search")
        
        # Step 2: Validate tenant access
        if not self.tenant_resolver.can_access(request.user, tenant_id):
            raise PermissionError(f"User cannot access tenant {tenant_id}")
        
        # Step 3: Create embedding
        query_vector = embedding_model.encode(query)
        
        # Step 4: Search with tenant filter (defense in depth)
        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter={"tenant_id": tenant_id}  # Always filter
        )
        
        # Step 5: Validate results belong to tenant
        validated_results = [
            r for r in results 
            if r.metadata.get("tenant_id") == tenant_id
        ]
        
        return validated_results
```

### Audit Tenant Access

```python
import logging

class TenantAccessAuditor:
    """
    Log all cross-tenant access attempts.
    """
    def __init__(self):
        self.logger = logging.getLogger('tenant_audit')
    
    def audit_access(
        self, 
        user_id: str, 
        requested_tenant: str, 
        allowed_tenants: list,
        action: str
    ):
        """Log access attempt with context."""
        is_allowed = requested_tenant in allowed_tenants
        
        log_entry = {
            "user_id": user_id,
            "requested_tenant": requested_tenant,
            "allowed_tenants": allowed_tenants,
            "action": action,
            "allowed": is_allowed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if is_allowed:
            self.logger.info(f"Tenant access granted: {log_entry}")
        else:
            self.logger.warning(f"CROSS-TENANT ACCESS ATTEMPT: {log_entry}")
            # Alert security team
            security_alerts.send(log_entry)
```

---

## Preventing Cross-Tenant Leakage

### Common Leakage Vectors

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Tenant Leakage Risks                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RISK 1: FILTER BYPASS                                          │
│  • Attacker manipulates tenant_id in request                   │
│  • Defense: Server-side tenant resolution from auth token      │
│                                                                 │
│  RISK 2: NAMESPACE CONFUSION                                    │
│  • Wrong namespace used in query                               │
│  • Defense: Centralized tenant routing, never trust input      │
│                                                                 │
│  RISK 3: SHARED EMBEDDING MODEL LEAKAGE                         │
│  • Similar content returns similar vectors                     │
│  • Defense: Accept as inherent limitation of shared models     │
│                                                                 │
│  RISK 4: METADATA EXPOSURE                                      │
│  • Metadata field reveals other tenant's info                  │
│  • Defense: Strict metadata validation on insert               │
│                                                                 │
│  RISK 5: INDEX-LEVEL INFERENCE                                  │
│  • Approximate counts reveal tenant data sizes                 │
│  • Defense: Disable stats endpoints in production             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Secure Tenant Routing Pattern

```python
from typing import Optional
import jwt

class SecureTenantRouter:
    """
    Server-side tenant resolution - never trust client input.
    """
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
    
    def get_tenant_from_request(self, request) -> Optional[str]:
        """
        Extract tenant from authenticated session only.
        Never from query parameters or request body.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload.get("tenant_id")
        except jwt.InvalidTokenError:
            return None
    
    def validate_tenant_access(self, request, target_tenant: str) -> bool:
        """
        Validate user can access target tenant.
        """
        auth_tenant = self.get_tenant_from_request(request)
        
        if auth_tenant is None:
            return False
        
        # User can only access their own tenant
        # Unless they have admin role
        if auth_tenant == target_tenant:
            return True
        
        # Check for cross-tenant admin permission
        user_roles = self._get_user_roles(request)
        if "cross_tenant_admin" in user_roles:
            return True
        
        return False
```

---

## Summary

✅ **Choose isolation level based on tenant count and compliance needs**  
✅ **Database-level isolation for highest security** (< 64 tenants)  
✅ **Collection-level for balanced isolation** (< 65K tenants)  
✅ **Partition keys for massive scale** (millions of tenants)  
✅ **Always filter by tenant at query level** as defense-in-depth  
✅ **Resolve tenant from authentication, never from user input**

---

**Next:** [Compliance Considerations →](./05-compliance-considerations.md)

---

<!-- 
Sources Consulted:
- Milvus Multi-tenancy Guide: https://milvus.io/docs/multi_tenancy.md
- Qdrant Security: https://qdrant.tech/documentation/guides/security/
- Pinecone Namespaces: https://docs.pinecone.io/guides/data/understanding-namespaces
- OWASP Multi-tenancy Security: https://owasp.org/www-community/Multi_Tenancy
-->
