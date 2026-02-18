---
title: "Capability-based permissions"
---

# Capability-based permissions

## Introduction

Traditional access control asks "who are you?" Capability-based permissions ask "what token do you hold?" This distinction matters for AI agents because agents don't have stable identities the way users do — they're spawned, forked, and delegated to dynamically. A capability token carries the permission itself, scoped to exactly what the bearer needs. No more, no less.

Capability-based security fits agent architectures naturally. When a supervisor agent delegates a task to a sub-agent, it grants a narrowly scoped capability token — not its entire set of permissions. If that sub-agent is compromised, the attacker gets only the delegated capability, not the supervisor's full access.

### What we'll cover

- Capability tokens: structure, scoping, and validation
- The permission request workflow: agents ask, supervisors approve
- Dynamic capability management at runtime
- Attenuation: reducing capabilities when delegating
- Comprehensive audit logging for every permission event

### Prerequisites

- Security boundaries for agents (Lesson 01)
- Resource limits (Lesson 02)
- Python dataclasses and enums

---

## Capability tokens

A capability is a token that combines a reference to a resource with the specific operations allowed on that resource. Unlike ACLs that live on the resource, capabilities travel with the agent.

### Token structure

```python
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum


class Operation(Enum):
    """Operations that can be granted."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    LIST = "list"
    CREATE = "create"
    CALL = "call"          # API calls
    DELEGATE = "delegate"  # Can delegate to sub-agents


@dataclass
class Capability:
    """A capability token granting specific access to a resource."""
    resource: str              # What resource: "file:/data/*.csv"
    operations: set[Operation] # What operations: {READ, LIST}
    holder: str                # Who holds this: "research-agent-v1"
    issuer: str                # Who granted it: "supervisor-agent"
    token_id: str = ""         # Unique identifier
    expires_at: float = 0.0    # Unix timestamp, 0 = no expiry
    max_uses: int = 0          # 0 = unlimited
    uses: int = 0              # Current use count
    conditions: dict = field(default_factory=dict)
    parent_token: str = ""     # Token this was derived from

    def __post_init__(self):
        if not self.token_id:
            # Generate deterministic token ID
            data = f"{self.resource}:{self.holder}:{self.issuer}:{time.time()}"
            self.token_id = hashlib.sha256(data.encode()).hexdigest()[:16]

    def is_valid(self) -> tuple[bool, str]:
        """Check if this capability is still valid."""
        if self.expires_at > 0 and time.time() > self.expires_at:
            return False, "Token expired"

        if self.max_uses > 0 and self.uses >= self.max_uses:
            return False, f"Max uses reached ({self.max_uses})"

        return True, "Valid"

    def matches(self, resource: str, operation: Operation) -> bool:
        """Check if this capability grants the requested access."""
        valid, _ = self.is_valid()
        if not valid:
            return False

        if operation not in self.operations:
            return False

        return self._resource_matches(resource)

    def _resource_matches(self, target: str) -> bool:
        """Check if target resource matches capability's resource pattern."""
        import fnmatch
        return fnmatch.fnmatch(target, self.resource)

    def use(self) -> bool:
        """Consume one use of this capability."""
        valid, reason = self.is_valid()
        if not valid:
            return False
        self.uses += 1
        return True


# Create capability tokens
file_read_cap = Capability(
    resource="file:/data/research/*.csv",
    operations={Operation.READ, Operation.LIST},
    holder="research-agent",
    issuer="supervisor",
    expires_at=time.time() + 3600,  # 1 hour
    max_uses=100,
)

api_call_cap = Capability(
    resource="api:openai/chat/completions",
    operations={Operation.CALL},
    holder="research-agent",
    issuer="supervisor",
    max_uses=50,
)

print("=== Capability Tokens ===\n")
for cap in [file_read_cap, api_call_cap]:
    valid, reason = cap.is_valid()
    ops = ", ".join(op.value for op in cap.operations)
    print(f"Token: {cap.token_id}")
    print(f"  Resource: {cap.resource}")
    print(f"  Operations: {ops}")
    print(f"  Holder: {cap.holder}")
    print(f"  Valid: {valid} ({reason})")
    if cap.max_uses:
        print(f"  Uses: {cap.uses}/{cap.max_uses}")
    print()

# Test matching
test_cases = [
    ("file:/data/research/results.csv", Operation.READ),
    ("file:/data/research/results.csv", Operation.WRITE),   # Not granted
    ("file:/data/private/secrets.txt", Operation.READ),      # Wrong path
    ("api:openai/chat/completions", Operation.CALL),
    ("api:openai/embeddings", Operation.CALL),               # Wrong resource
]

print("=== Access Checks ===\n")
for resource, op in test_cases:
    granted = file_read_cap.matches(resource, op) or api_call_cap.matches(resource, op)
    status = "✅ GRANTED" if granted else "❌ DENIED"
    print(f"{status}: {op.value:7} on {resource}")
```

**Output:**
```
=== Capability Tokens ===

Token: a1b2c3d4e5f6a7b8
  Resource: file:/data/research/*.csv
  Operations: read, list
  Holder: research-agent
  Valid: True (Valid)
  Uses: 0/100

Token: f8e7d6c5b4a39281
  Resource: api:openai/chat/completions
  Operations: call
  Holder: research-agent
  Valid: True (Valid)
  Uses: 0/50

=== Access Checks ===

✅ GRANTED: read    on file:/data/research/results.csv
❌ DENIED: write   on file:/data/research/results.csv
❌ DENIED: read    on file:/data/private/secrets.txt
✅ GRANTED: call    on api:openai/chat/completions
❌ DENIED: call    on api:openai/embeddings
```

---

## Capability manager

A central capability manager tracks all tokens, validates access, and handles the lifecycle of permissions.

```python
from dataclasses import dataclass, field
from enum import Enum
import time


class PermissionEvent(Enum):
    """Types of permission events for audit logging."""
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"
    DELEGATED = "delegated"
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    USED = "used"


@dataclass
class AuditEntry:
    """Record of a permission-related event."""
    timestamp: float
    event: PermissionEvent
    agent: str
    resource: str
    operation: str
    token_id: str = ""
    details: str = ""


class CapabilityManager:
    """Central manager for agent capability tokens."""

    def __init__(self):
        self.capabilities: dict[str, list[Capability]] = {}  # agent -> caps
        self.audit_log: list[AuditEntry] = []
        self.pending_requests: list[dict] = []

    def grant(
        self,
        holder: str,
        resource: str,
        operations: set[Operation],
        issuer: str,
        expires_in: float = 3600,
        max_uses: int = 0,
    ) -> Capability:
        """Grant a capability to an agent."""
        cap = Capability(
            resource=resource,
            operations=operations,
            holder=holder,
            issuer=issuer,
            expires_at=time.time() + expires_in if expires_in > 0 else 0,
            max_uses=max_uses,
        )

        if holder not in self.capabilities:
            self.capabilities[holder] = []
        self.capabilities[holder].append(cap)

        self._audit(
            PermissionEvent.GRANTED, holder, resource,
            ",".join(op.value for op in operations),
            cap.token_id,
            f"Issued by {issuer}, expires in {expires_in}s"
        )

        return cap

    def check(
        self,
        agent: str,
        resource: str,
        operation: Operation,
    ) -> tuple[bool, str]:
        """Check if an agent has capability for the requested access."""
        caps = self.capabilities.get(agent, [])

        for cap in caps:
            if cap.matches(resource, operation):
                # Consume a use
                if not cap.use():
                    self._audit(
                        PermissionEvent.EXPIRED, agent, resource,
                        operation.value, cap.token_id, "Token exhausted"
                    )
                    continue

                self._audit(
                    PermissionEvent.USED, agent, resource,
                    operation.value, cap.token_id,
                    f"Use {cap.uses}/{cap.max_uses or '∞'}"
                )
                return True, f"Granted by token {cap.token_id}"

        self._audit(
            PermissionEvent.DENIED, agent, resource,
            operation.value, details="No matching capability"
        )
        return False, "No matching capability"

    def revoke(self, agent: str, token_id: str = "") -> int:
        """Revoke capabilities. If token_id specified, revoke one; else all."""
        caps = self.capabilities.get(agent, [])
        revoked = 0

        if token_id:
            self.capabilities[agent] = [
                c for c in caps if c.token_id != token_id
            ]
            revoked = len(caps) - len(self.capabilities[agent])
        else:
            self.capabilities[agent] = []
            revoked = len(caps)

        if revoked:
            self._audit(
                PermissionEvent.REVOKED, agent, "*",
                "*", token_id or "ALL",
                f"Revoked {revoked} capability(s)"
            )

        return revoked

    def delegate(
        self,
        from_agent: str,
        to_agent: str,
        resource: str,
        operations: set[Operation],
        restrict_uses: int = 10,
    ) -> Capability | None:
        """Allow an agent to delegate a subset of its capabilities."""
        # Verify the delegator has DELEGATE permission
        has_delegate = False
        parent_token = ""
        for cap in self.capabilities.get(from_agent, []):
            if cap.matches(resource, Operation.DELEGATE):
                has_delegate = True
                parent_token = cap.token_id
                break

        if not has_delegate:
            # Check if delegator even has the operations requested
            for op in operations:
                granted, _ = self.check(from_agent, resource, op)
                if not granted:
                    self._audit(
                        PermissionEvent.DENIED, from_agent, resource,
                        "delegate", details=f"Cannot delegate {op.value} — not held"
                    )
                    return None

        # Create attenuated capability for the delegate
        cap = Capability(
            resource=resource,
            operations=operations - {Operation.DELEGATE},  # Never delegate DELEGATE
            holder=to_agent,
            issuer=from_agent,
            expires_at=time.time() + 1800,  # 30 minutes max for delegated
            max_uses=restrict_uses,
            parent_token=parent_token,
        )

        if to_agent not in self.capabilities:
            self.capabilities[to_agent] = []
        self.capabilities[to_agent].append(cap)

        self._audit(
            PermissionEvent.DELEGATED, to_agent, resource,
            ",".join(op.value for op in operations),
            cap.token_id,
            f"Delegated by {from_agent}, max {restrict_uses} uses"
        )

        return cap

    def list_capabilities(self, agent: str) -> list[dict]:
        """List all valid capabilities for an agent."""
        caps = self.capabilities.get(agent, [])
        result = []
        for cap in caps:
            valid, reason = cap.is_valid()
            result.append({
                "token_id": cap.token_id,
                "resource": cap.resource,
                "operations": [op.value for op in cap.operations],
                "valid": valid,
                "reason": reason,
                "uses": f"{cap.uses}/{cap.max_uses or '∞'}",
                "issuer": cap.issuer,
            })
        return result

    def _audit(
        self, event: PermissionEvent, agent: str,
        resource: str, operation: str,
        token_id: str = "", details: str = ""
    ):
        self.audit_log.append(AuditEntry(
            timestamp=time.time(),
            event=event,
            agent=agent,
            resource=resource,
            operation=operation,
            token_id=token_id,
            details=details,
        ))


# Demo the full lifecycle
mgr = CapabilityManager()

# 1. Supervisor grants capabilities to research agent
print("=== 1. Grant Capabilities ===\n")
mgr.grant(
    holder="research-agent",
    resource="file:/data/research/*",
    operations={Operation.READ, Operation.LIST, Operation.DELEGATE},
    issuer="supervisor",
    max_uses=50,
)
mgr.grant(
    holder="research-agent",
    resource="api:openai/*",
    operations={Operation.CALL},
    issuer="supervisor",
    max_uses=100,
)

for cap_info in mgr.list_capabilities("research-agent"):
    ops = ", ".join(cap_info["operations"])
    print(f"  {cap_info['resource']}: [{ops}] (uses: {cap_info['uses']})")

# 2. Research agent accesses resources
print("\n=== 2. Access Checks ===\n")
checks = [
    ("file:/data/research/results.csv", Operation.READ),
    ("file:/data/research/data.json", Operation.LIST),
    ("file:/data/private/secrets.txt", Operation.READ),
    ("api:openai/chat/completions", Operation.CALL),
    ("api:anthropic/messages", Operation.CALL),
]

for resource, op in checks:
    granted, reason = mgr.check("research-agent", resource, op)
    status = "✅" if granted else "❌"
    print(f"  {status} {op.value:8} {resource:<40} — {reason}")

# 3. Research agent delegates to a sub-agent
print("\n=== 3. Delegation ===\n")
delegated = mgr.delegate(
    from_agent="research-agent",
    to_agent="data-fetcher",
    resource="file:/data/research/*",
    operations={Operation.READ},
    restrict_uses=5,
)
if delegated:
    print(f"  Delegated to data-fetcher: {delegated.resource}")
    print(f"  Operations: {[op.value for op in delegated.operations]}")
    print(f"  Max uses: {delegated.max_uses}")
    print(f"  DELEGATE stripped: {'delegate' not in [op.value for op in delegated.operations]}")

# 4. Sub-agent uses delegated capability
print("\n=== 4. Delegated Access ===\n")
for i in range(6):
    granted, reason = mgr.check("data-fetcher", f"file:/data/research/file{i}.csv", Operation.READ)
    status = "✅" if granted else "❌"
    print(f"  Use {i+1}: {status} — {reason}")

# 5. Revocation
print("\n=== 5. Revocation ===\n")
count = mgr.revoke("data-fetcher")
print(f"  Revoked {count} capability(s) from data-fetcher")
granted, reason = mgr.check("data-fetcher", "file:/data/research/file0.csv", Operation.READ)
print(f"  After revocation: {'✅' if granted else '❌'} — {reason}")
```

**Output:**
```
=== 1. Grant Capabilities ===

  file:/data/research/*: [read, list, delegate] (uses: 0/50)
  api:openai/*: [call] (uses: 0/100)

=== 2. Access Checks ===

  ✅ read     file:/data/research/results.csv           — Granted by token a1b2c3d4
  ✅ list     file:/data/research/data.json             — Granted by token a1b2c3d4
  ❌ read     file:/data/private/secrets.txt            — No matching capability
  ✅ call     api:openai/chat/completions               — Granted by token f8e7d6c5
  ❌ call     api:anthropic/messages                    — No matching capability

=== 3. Delegation ===

  Delegated to data-fetcher: file:/data/research/*
  Operations: ['read']
  Max uses: 5
  DELEGATE stripped: True

=== 4. Delegated Access ===

  Use 1: ✅ — Granted by token 9a8b7c6d
  Use 2: ✅ — Granted by token 9a8b7c6d
  Use 3: ✅ — Granted by token 9a8b7c6d
  Use 4: ✅ — Granted by token 9a8b7c6d
  Use 5: ✅ — Granted by token 9a8b7c6d
  Use 6: ❌ — No matching capability

=== 5. Revocation ===

  Revoked 1 capability(s) from data-fetcher
  After revocation: ❌ — No matching capability
```

> **🤖 AI Context:** Capability attenuation is critical for multi-agent systems. When a supervisor delegates to a sub-agent, the delegated token always has *fewer* permissions and a shorter lifetime. This is the principle of least privilege applied to delegation chains.

---

## Permission request workflow

Agents should request capabilities at runtime rather than starting with broad permissions. This creates an approval workflow.

```python
import time
from dataclasses import dataclass
from enum import Enum


class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto-approved"


@dataclass
class PermissionRequest:
    """An agent's request for a capability."""
    request_id: str
    agent: str
    resource: str
    operations: list[str]
    reason: str
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = 0.0
    reviewed_by: str = ""
    review_note: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class PermissionWorkflow:
    """Manages the request-approve-grant workflow."""

    def __init__(self, manager: CapabilityManager):
        self.manager = manager
        self.requests: list[PermissionRequest] = []
        self.auto_approve_rules: list[dict] = []
        self._counter = 0

    def add_auto_approve_rule(
        self,
        resource_pattern: str,
        operations: list[str],
        max_uses: int = 10,
    ):
        """Add a rule for automatic approval."""
        self.auto_approve_rules.append({
            "resource": resource_pattern,
            "operations": operations,
            "max_uses": max_uses,
        })

    def request_permission(
        self,
        agent: str,
        resource: str,
        operations: list[str],
        reason: str,
    ) -> PermissionRequest:
        """Agent requests a capability."""
        self._counter += 1
        req = PermissionRequest(
            request_id=f"REQ-{self._counter:04d}",
            agent=agent,
            resource=resource,
            operations=operations,
            reason=reason,
        )

        # Check auto-approve rules
        import fnmatch
        for rule in self.auto_approve_rules:
            if fnmatch.fnmatch(resource, rule["resource"]):
                if all(op in rule["operations"] for op in operations):
                    req.status = RequestStatus.AUTO_APPROVED
                    req.reviewed_by = "auto-approve"
                    # Grant immediately
                    self.manager.grant(
                        holder=agent,
                        resource=resource,
                        operations={Operation(op) for op in operations},
                        issuer="auto-approve",
                        max_uses=rule["max_uses"],
                    )
                    break

        self.requests.append(req)

        self.manager._audit(
            PermissionEvent.REQUESTED, agent, resource,
            ",".join(operations), req.request_id, reason
        )

        return req

    def approve(
        self,
        request_id: str,
        reviewer: str,
        note: str = "",
        max_uses: int = 0,
        expires_in: float = 3600,
    ) -> bool:
        """Approve a pending permission request."""
        for req in self.requests:
            if req.request_id == request_id and req.status == RequestStatus.PENDING:
                req.status = RequestStatus.APPROVED
                req.reviewed_by = reviewer
                req.review_note = note

                self.manager.grant(
                    holder=req.agent,
                    resource=req.resource,
                    operations={Operation(op) for op in req.operations},
                    issuer=reviewer,
                    max_uses=max_uses,
                    expires_in=expires_in,
                )

                self.manager._audit(
                    PermissionEvent.APPROVED, req.agent, req.resource,
                    ",".join(req.operations), request_id,
                    f"Approved by {reviewer}: {note}"
                )
                return True
        return False

    def reject(self, request_id: str, reviewer: str, note: str = "") -> bool:
        """Reject a pending permission request."""
        for req in self.requests:
            if req.request_id == request_id and req.status == RequestStatus.PENDING:
                req.status = RequestStatus.REJECTED
                req.reviewed_by = reviewer
                req.review_note = note

                self.manager._audit(
                    PermissionEvent.REJECTED, req.agent, req.resource,
                    ",".join(req.operations), request_id,
                    f"Rejected by {reviewer}: {note}"
                )
                return True
        return False


# Demo the workflow
mgr2 = CapabilityManager()
workflow = PermissionWorkflow(mgr2)

# Auto-approve read access to public data
workflow.add_auto_approve_rule("file:/data/public/*", ["read", "list"], max_uses=100)

print("=== Permission Request Workflow ===\n")

# Request 1: Auto-approved (matches rule)
req1 = workflow.request_permission(
    "research-agent",
    "file:/data/public/papers.csv",
    ["read"],
    "Need to read research papers dataset",
)
print(f"Request 1: {req1.request_id} — {req1.status.value}")

# Request 2: Needs manual approval
req2 = workflow.request_permission(
    "research-agent",
    "file:/data/private/user-data.csv",
    ["read"],
    "Need user data for analysis",
)
print(f"Request 2: {req2.request_id} — {req2.status.value}")

# Request 3: Needs manual approval
req3 = workflow.request_permission(
    "research-agent",
    "api:external/payment-service",
    ["call"],
    "Need to process refunds",
)
print(f"Request 3: {req3.request_id} — {req3.status.value}")

# Admin reviews pending requests
print("\n--- Supervisor Reviews ---\n")
workflow.approve("REQ-0002", "supervisor", "Approved with limited uses", max_uses=5)
print(f"Request 2: APPROVED (5 uses)")

workflow.reject("REQ-0003", "supervisor", "Agents cannot access payment systems")
print(f"Request 3: REJECTED")

# Verify access
print("\n=== Verify Access After Workflow ===\n")
checks = [
    ("file:/data/public/papers.csv", Operation.READ),       # Auto-approved
    ("file:/data/private/user-data.csv", Operation.READ),   # Manually approved
    ("api:external/payment-service", Operation.CALL),       # Rejected
]

for resource, op in checks:
    granted, reason = mgr2.check("research-agent", resource, op)
    status = "✅" if granted else "❌"
    print(f"  {status} {op.value:5} {resource:<40} — {reason}")
```

**Output:**
```
=== Permission Request Workflow ===

Request 1: REQ-0001 — auto-approved
Request 2: REQ-0002 — pending
Request 3: REQ-0003 — pending

--- Supervisor Reviews ---

Request 2: APPROVED (5 uses)
Request 3: REJECTED

=== Verify Access After Workflow ===

  ✅ read  file:/data/public/papers.csv              — Granted by token abc12345
  ✅ read  file:/data/private/user-data.csv          — Granted by token def67890
  ❌ call  api:external/payment-service              — No matching capability
```

---

## Audit trail

Every capability event must be logged for security analysis and compliance. The audit log answers: who accessed what, when, and whether it was authorized.

```python
def print_audit_report(manager: CapabilityManager, last_n: int = 20):
    """Generate a formatted audit report."""
    print("=== Capability Audit Trail ===\n")
    print(f"{'Event':<14} {'Agent':<18} {'Resource':<35} {'Details'}")
    print("-" * 100)

    for entry in manager.audit_log[-last_n:]:
        event = entry.event.value.upper()
        resource = entry.resource[:33] + ".." if len(entry.resource) > 35 else entry.resource
        details = entry.details[:40] if entry.details else entry.operation
        print(f"{event:<14} {entry.agent:<18} {resource:<35} {details}")

    # Summary statistics
    events = {}
    for entry in manager.audit_log:
        events[entry.event.value] = events.get(entry.event.value, 0) + 1

    agents = set(e.agent for e in manager.audit_log)

    print(f"\n--- Summary ---")
    print(f"Total events: {len(manager.audit_log)}")
    print(f"Unique agents: {len(agents)}")
    for event, count in sorted(events.items()):
        print(f"  {event}: {count}")

    # Security alerts
    denied = [e for e in manager.audit_log if e.event == PermissionEvent.DENIED]
    if denied:
        print(f"\n⚠️  {len(denied)} denied access attempt(s):")
        for d in denied[-3:]:
            print(f"  {d.agent} → {d.resource} ({d.operation})")


# Print the audit trail from our earlier demo
print_audit_report(mgr)
```

**Output:**
```
=== Capability Audit Trail ===

Event          Agent              Resource                            Details
----------------------------------------------------------------------------------------------------
GRANTED        research-agent     file:/data/research/*               Issued by supervisor, expires in 3600s
GRANTED        research-agent     api:openai/*                        Issued by supervisor, expires in 3600s
USED           research-agent     file:/data/research/results.csv     Use 1/50
USED           research-agent     file:/data/research/data.json       Use 2/50
DENIED         research-agent     file:/data/private/secrets.txt      No matching capability
USED           research-agent     api:openai/chat/completions         Use 1/100
DENIED         research-agent     api:anthropic/messages              No matching capability
DELEGATED      data-fetcher       file:/data/research/*               Delegated by research-agent, max 5 uses
USED           data-fetcher       file:/data/research/file0.csv       Use 1/5
USED           data-fetcher       file:/data/research/file1.csv       Use 2/5
USED           data-fetcher       file:/data/research/file2.csv       Use 3/5
USED           data-fetcher       file:/data/research/file3.csv       Use 4/5
USED           data-fetcher       file:/data/research/file4.csv       Use 5/5
DENIED         data-fetcher       file:/data/research/file5.csv       No matching capability
REVOKED        data-fetcher       *                                   Revoked 1 capability(s)
DENIED         data-fetcher       file:/data/research/file0.csv       No matching capability

--- Summary ---
Total events: 16
Unique agents: 2
  denied: 4
  delegated: 1
  granted: 2
  revoked: 1
  used: 8

⚠️  4 denied access attempt(s):
  research-agent → file:/data/private/secrets.txt (read)
  data-fetcher → file:/data/research/file5.csv (read)
  data-fetcher → file:/data/research/file0.csv (read)
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Start agents with zero capabilities | They request only what they need for the current task |
| Always set expiration on tokens | Leaked tokens become useless after expiry |
| Strip DELEGATE from delegated tokens | Prevents uncontrolled capability chains |
| Log every grant, use, and denial | The audit trail is your forensic evidence |
| Use `max_uses` for one-shot tasks | A token for "read this one file" shouldn't allow unlimited reads |
| Auto-approve only low-risk operations | Public data reads can be instant; private data needs human review |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Granting broad `*` resource patterns | Scope to specific paths: `file:/data/research/*.csv` |
| No expiration on capability tokens | Always set `expires_at` — default to 1 hour |
| Allowing agents to delegate without restriction | Cap delegated tokens at fewer uses and shorter lifetime |
| Not checking `is_valid()` before every use | The manager should validate on every `check()` call |
| Missing audit log for denied requests | Denied access is often more valuable than granted access |
| Token IDs that are guessable | Use cryptographic hashes, not sequential numbers |

---

## Hands-on exercise

### Your task

Build a `CapabilitySecuritySystem` that implements the full lifecycle: request → approve/reject → use → delegate → audit.

### Requirements

1. Create a capability manager with at least 3 resource types (files, APIs, databases)
2. Implement auto-approve rules for read-only public data
3. Support delegation with capability attenuation (fewer ops, shorter lifetime, limited uses)
4. Show that delegated tokens cannot re-delegate
5. Generate a complete audit report showing all events

### Expected result

An agent requests capabilities, some are auto-approved, others need supervisor review. The agent delegates a subset to a sub-agent. After the sub-agent exceeds its uses, access is denied. The audit trail captures every event.

<details>
<summary>💡 Hints (click to expand)</summary>

- Reuse the `CapabilityManager` and `PermissionWorkflow` classes from this lesson
- Attenuation means: delegated tokens always have `operations ⊂ parent.operations`
- Test the "cannot re-delegate" case: the sub-agent's token has no `DELEGATE` operation
- Sort the audit log by timestamp for the final report

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class Op(Enum):
    READ = "read"
    WRITE = "write"
    CALL = "call"
    LIST = "list"
    DELEGATE = "delegate"


@dataclass
class Cap:
    resource: str
    operations: set[Op]
    holder: str
    issuer: str
    token_id: str = ""
    expires_at: float = 0
    max_uses: int = 0
    uses: int = 0

    def __post_init__(self):
        if not self.token_id:
            self.token_id = hashlib.sha256(
                f"{self.resource}:{self.holder}:{time.time()}".encode()
            ).hexdigest()[:12]

    def valid(self):
        if self.expires_at and time.time() > self.expires_at:
            return False
        if self.max_uses and self.uses >= self.max_uses:
            return False
        return True

    def matches(self, resource, op):
        import fnmatch
        return self.valid() and op in self.operations and fnmatch.fnmatch(resource, self.resource)

    def use(self):
        if not self.valid():
            return False
        self.uses += 1
        return True


class SecuritySystem:
    def __init__(self):
        self.caps: dict[str, list[Cap]] = {}
        self.log: list[str] = []

    def grant(self, holder, resource, ops, issuer, max_uses=0, expires_in=3600):
        cap = Cap(resource, ops, holder, issuer,
                  expires_at=time.time() + expires_in, max_uses=max_uses)
        self.caps.setdefault(holder, []).append(cap)
        self.log.append(f"GRANT  {holder}: {resource} [{','.join(o.value for o in ops)}]")
        return cap

    def check(self, agent, resource, op):
        for cap in self.caps.get(agent, []):
            if cap.matches(resource, op) and cap.use():
                self.log.append(f"USE    {agent}: {op.value} on {resource}")
                return True
        self.log.append(f"DENY   {agent}: {op.value} on {resource}")
        return False

    def delegate(self, from_a, to_a, resource, ops, max_uses=5):
        parent_ops = set()
        for cap in self.caps.get(from_a, []):
            if cap.matches(resource, Op.DELEGATE):
                parent_ops = cap.operations
                break
        if not parent_ops:
            self.log.append(f"DENY   {from_a}: cannot delegate {resource}")
            return None
        safe_ops = (ops & parent_ops) - {Op.DELEGATE}
        cap = Cap(resource, safe_ops, to_a, from_a,
                  expires_at=time.time() + 1800, max_uses=max_uses)
        self.caps.setdefault(to_a, []).append(cap)
        self.log.append(f"DELEG  {from_a} -> {to_a}: {resource}")
        return cap

    def report(self):
        print("\n=== Audit Report ===")
        for entry in self.log:
            print(f"  {entry}")
        print(f"\nTotal events: {len(self.log)}")


sys = SecuritySystem()
sys.grant("agent-a", "file:/public/*", {Op.READ, Op.LIST, Op.DELEGATE}, "admin", max_uses=50)
sys.grant("agent-a", "api:llm/*", {Op.CALL}, "admin", max_uses=20)
sys.grant("agent-a", "db:analytics/*", {Op.READ}, "admin", max_uses=10)

sys.check("agent-a", "file:/public/data.csv", Op.READ)
sys.check("agent-a", "file:/private/secrets.txt", Op.READ)

delegated = sys.delegate("agent-a", "sub-agent", "file:/public/*", {Op.READ, Op.DELEGATE}, max_uses=3)
for i in range(4):
    sys.check("sub-agent", f"file:/public/file{i}.csv", Op.READ)

# Sub-agent cannot re-delegate
result = sys.delegate("sub-agent", "sub-sub", "file:/public/*", {Op.READ})
print(f"Sub-agent re-delegate: {'Success' if result else 'Blocked'}")

sys.report()
```
</details>

### Bonus challenges

- [ ] Add capability token signing with HMAC to prevent forgery
- [ ] Implement a "capability store" that persists tokens to disk/database
- [ ] Create a visual capability graph showing delegation chains

---

## Summary

✅ **Capability tokens** combine resource references with allowed operations in a single, transferable token — simpler and more flexible than traditional ACLs for dynamic agent systems

✅ **Capability attenuation** ensures delegated tokens always have fewer permissions, shorter lifetimes, and limited uses — the DELEGATE operation itself is never passed down

✅ **Permission request workflows** let agents start with zero capabilities and request what they need — with auto-approval for low-risk operations and human review for sensitive access

✅ **Comprehensive audit logging** captures every grant, use, denial, delegation, and revocation — the audit trail is the foundation for security analysis and incident response

---

**Next:** [Containerized Agent Execution](./06-containerized-agent-execution.md)

**Previous:** [Network Isolation](./04-network-isolation.md)

---

## Further Reading

- [Capability-Based Security (Wikipedia)](https://en.wikipedia.org/wiki/Capability-based_security) - Foundational concepts
- [The Confused Deputy Problem](https://en.wikipedia.org/wiki/Confused_deputy_problem) - Why capabilities solve ACL limitations
- [Object-Capability Model](https://en.wikipedia.org/wiki/Object-capability_model) - Programming language perspective
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html) - Practical access control guidance

<!-- 
Sources Consulted:
- OpenAI Agents SDK Guardrails: https://openai.github.io/openai-agents-python/guardrails/
- Docker Engine Security: https://docs.docker.com/engine/security/
- Kubernetes Pod Security Standards: https://kubernetes.io/docs/concepts/security/pod-security-standards/
- OWASP Access Control: https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html
-->
