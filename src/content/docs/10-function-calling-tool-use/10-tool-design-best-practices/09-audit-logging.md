---
title: "Audit Logging"
---

# Audit Logging

## Introduction

When an AI agent updates a customer record, refunds an order, or sends an email — who did it? Was it the model's decision, the user's request, or an automated rule? **Audit logging** answers these questions by recording every tool call with full context: who triggered it, what was called, what arguments were used, and what the result was.

Audit logs serve three purposes: **debugging** (what went wrong), **compliance** (proving what happened), and **improvement** (understanding how tools are being used).

### What we'll cover

- What to log for every tool call
- Structured logging for AI tool interactions
- User attribution and accountability chains
- Compliance requirements for regulated industries
- Building an audit trail that's actually useful

### Prerequisites

- [Security Best Practices](./07-security-best-practices.md) — Authorization and data protection
- [Rate Limiting](./08-rate-limiting.md) — Call tracking

---

## What to log

Every tool call should record these fields:

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | When the call happened (UTC) | `2025-07-06T14:23:01.456Z` |
| `conversation_id` | Which conversation triggered it | `conv_abc123` |
| `turn_number` | Which turn in the conversation | `7` |
| `user_id` | The human user who started the conversation | `usr_98765` |
| `tool_name` | Which tool was called | `update_order_status` |
| `arguments` | The parameters passed to the tool | `{"order_id": "ORD-123", "status": "shipped"}` |
| `result_summary` | Outcome (success/error + key data) | `{"success": true, "old_status": "processing"}` |
| `duration_ms` | How long the call took | `145` |
| `model` | Which AI model made the call | `gpt-4.1` |
| `session_metadata` | Additional context (user_role, client_ip) | `{"role": "support_agent", "ip": "10.0.1.5"}` |

```python
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import json
import logging

logger = logging.getLogger("tool_audit")
logger.setLevel(logging.INFO)


@dataclass
class AuditEntry:
    """A single audit log entry for a tool call."""
    timestamp: str
    conversation_id: str
    turn_number: int
    user_id: str
    tool_name: str
    arguments: dict
    result_summary: dict
    duration_ms: float
    model: str
    success: bool
    session_metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Log every tool call for debugging, compliance, and analytics."""
    
    def __init__(self):
        self._entries: list[AuditEntry] = []
    
    def log_call(
        self,
        tool_name: str,
        arguments: dict,
        result: dict,
        duration_ms: float,
        context: dict
    ) -> AuditEntry:
        """Record a tool call with full context."""
        
        # Sanitize arguments — remove sensitive fields
        safe_args = self._sanitize(arguments)
        
        # Summarize result — don't log full payloads
        result_summary = self._summarize_result(result)
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            conversation_id=context.get("conversation_id", "unknown"),
            turn_number=context.get("turn_number", 0),
            user_id=context.get("user_id", "unknown"),
            tool_name=tool_name,
            arguments=safe_args,
            result_summary=result_summary,
            duration_ms=round(duration_ms, 2),
            model=context.get("model", "unknown"),
            success="error" not in result,
            session_metadata={
                "user_role": context.get("user_role"),
                "client_ip": context.get("client_ip"),
                "user_agent": context.get("user_agent"),
            }
        )
        
        self._entries.append(entry)
        logger.info(entry.to_json())
        
        return entry
    
    def _sanitize(self, data: dict) -> dict:
        """Remove sensitive fields from logged data."""
        SENSITIVE_FIELDS = {
            "password", "token", "api_key", "secret",
            "credit_card", "ssn", "authorization"
        }
        
        sanitized = {}
        for key, value in data.items():
            if key.lower() in SENSITIVE_FIELDS:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _summarize_result(self, result: dict) -> dict:
        """Create a compact summary of the tool result."""
        summary = {"success": "error" not in result}
        
        if "error" in result:
            summary["error"] = result["error"]
        
        # Include key identifiers but not full payloads
        for key in ["id", "order_id", "customer_id", "status", "count"]:
            if key in result:
                summary[key] = result[key]
        
        # Note if result was large
        result_str = json.dumps(result, default=str)
        if len(result_str) > 1000:
            summary["_result_size"] = len(result_str)
            summary["_truncated"] = True
        
        return summary
```

---

## Integrating audit logging with tool execution

Wrap your tool execution in an audit-aware middleware:

```python
import time

audit = AuditLogger()

def execute_tool_with_audit(
    tool_name: str,
    arguments: dict,
    context: dict
) -> dict:
    """Execute a tool and log the call for audit purposes."""
    
    start_time = time.perf_counter()
    
    try:
        # Execute the tool
        handler = get_handler(tool_name)
        result = handler(**arguments)
        
    except Exception as e:
        result = {"error": str(e), "error_type": type(e).__name__}
    
    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    # Log the audit entry
    audit.log_call(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        duration_ms=duration_ms,
        context=context
    )
    
    return result
```

**Example log output:**
```json
{
    "timestamp": "2025-07-06T14:23:01.456789+00:00",
    "conversation_id": "conv_abc123",
    "turn_number": 3,
    "user_id": "usr_98765",
    "tool_name": "update_order_status",
    "arguments": {
        "order_id": "ORD-54321",
        "status": "shipped"
    },
    "result_summary": {
        "success": true,
        "order_id": "ORD-54321",
        "status": "shipped"
    },
    "duration_ms": 145.23,
    "model": "gpt-4.1",
    "success": true,
    "session_metadata": {
        "user_role": "support_agent",
        "client_ip": "10.0.1.5",
        "user_agent": "SupportDashboard/2.1"
    }
}
```

---

## User attribution

AI tool calls have a unique attribution challenge: the **model** makes the call, but a **human user** initiated the conversation. Track the full chain:

```python
@dataclass
class AttributionChain:
    """Who is responsible for this tool call?"""
    initiator: str          # The human user who started the conversation
    initiator_role: str     # Their role (customer, agent, admin)
    model: str              # The AI model that decided to call the tool
    tool_name: str          # The tool that was called
    trigger: str            # What caused the call (user_request, proactive, retry)
    
    def summary(self) -> str:
        return (
            f"User {self.initiator} ({self.initiator_role}) → "
            f"{self.model} → {self.tool_name} [{self.trigger}]"
        )


def build_attribution(
    context: dict,
    tool_name: str,
    call_reason: str = "user_request"
) -> AttributionChain:
    """Build a clear attribution chain for a tool call."""
    return AttributionChain(
        initiator=context["user_id"],
        initiator_role=context.get("user_role", "unknown"),
        model=context.get("model", "unknown"),
        tool_name=tool_name,
        trigger=call_reason
    )
```

### Classifying call triggers

```python
from enum import Enum

class CallTrigger(Enum):
    USER_REQUEST = "user_request"    # User explicitly asked
    PROACTIVE = "proactive"          # Model decided on its own
    RETRY = "retry"                  # Retrying a failed call
    FOLLOW_UP = "follow_up"         # Completing a multi-step workflow
    SYSTEM = "system"               # System-triggered (scheduled, automated)


# Include trigger in audit log
def log_with_trigger(
    tool_name: str,
    arguments: dict,
    result: dict,
    context: dict,
    trigger: CallTrigger
) -> None:
    """Log with explicit trigger classification."""
    context_with_trigger = {
        **context,
        "call_trigger": trigger.value
    }
    audit.log_call(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        duration_ms=0,
        context=context_with_trigger
    )
```

---

## Compliance requirements

For regulated industries (finance, healthcare, government), audit logs must meet specific standards:

### Immutability

Logs must be append-only. No one should be able to modify or delete audit records:

```python
import hashlib

class ImmutableAuditLog:
    """Append-only audit log with hash chain integrity."""
    
    def __init__(self):
        self._entries: list[dict] = []
        self._previous_hash: str = "genesis"
    
    def append(self, entry: AuditEntry) -> dict:
        """Add an entry with hash chain verification."""
        entry_dict = entry.to_dict()
        
        # Create hash chain — each entry references the previous
        entry_dict["_previous_hash"] = self._previous_hash
        entry_dict["_hash"] = hashlib.sha256(
            json.dumps(entry_dict, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        self._previous_hash = entry_dict["_hash"]
        self._entries.append(entry_dict)
        
        return entry_dict
    
    def verify_integrity(self) -> bool:
        """Verify the hash chain is unbroken."""
        expected_prev = "genesis"
        
        for entry in self._entries:
            if entry["_previous_hash"] != expected_prev:
                return False
            
            # Recalculate hash
            verify_entry = dict(entry)
            stored_hash = verify_entry.pop("_hash")
            recalculated = hashlib.sha256(
                json.dumps(verify_entry, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            if recalculated != stored_hash:
                return False
            
            expected_prev = stored_hash
        
        return True
```

### Retention policies

```python
RETENTION_POLICIES = {
    "financial": {
        "retention_days": 2555,  # 7 years
        "tools": ["charge_payment", "issue_refund", "apply_credit"]
    },
    "healthcare": {
        "retention_days": 2190,  # 6 years
        "tools": ["access_patient_record", "update_treatment"]
    },
    "general": {
        "retention_days": 365,   # 1 year
        "tools": "*"             # All other tools
    }
}

def get_retention_period(tool_name: str) -> int:
    """Get the retention period in days for a tool's audit logs."""
    for policy_name, policy in RETENTION_POLICIES.items():
        if policy["tools"] == "*":
            continue
        if tool_name in policy["tools"]:
            return policy["retention_days"]
    
    return RETENTION_POLICIES["general"]["retention_days"]
```

---

## Querying audit logs

Make your audit logs queryable for debugging and compliance:

```python
from typing import Generator

class AuditQueryEngine:
    """Query audit logs for debugging and compliance."""
    
    def __init__(self, log: AuditLogger):
        self._log = log
    
    def by_conversation(self, conversation_id: str) -> list[AuditEntry]:
        """Get all tool calls in a conversation — for debugging."""
        return [
            e for e in self._log._entries
            if e.conversation_id == conversation_id
        ]
    
    def by_user(
        self, user_id: str, since: datetime | None = None
    ) -> list[AuditEntry]:
        """Get all tool calls by a user — for usage analysis."""
        entries = [
            e for e in self._log._entries
            if e.user_id == user_id
        ]
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.timestamp) >= since
            ]
        return entries
    
    def failures(
        self, since: datetime | None = None
    ) -> list[AuditEntry]:
        """Get all failed tool calls — for reliability monitoring."""
        entries = [e for e in self._log._entries if not e.success]
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.timestamp) >= since
            ]
        return entries
    
    def by_tool(self, tool_name: str) -> dict:
        """Get usage statistics for a tool — for optimization."""
        calls = [
            e for e in self._log._entries
            if e.tool_name == tool_name
        ]
        
        if not calls:
            return {"tool": tool_name, "total_calls": 0}
        
        durations = [e.duration_ms for e in calls]
        successes = [e for e in calls if e.success]
        
        return {
            "tool": tool_name,
            "total_calls": len(calls),
            "success_rate": round(len(successes) / len(calls), 3),
            "avg_duration_ms": round(sum(durations) / len(durations), 1),
            "max_duration_ms": round(max(durations), 1),
            "unique_users": len(set(e.user_id for e in calls))
        }
    
    def conversation_timeline(
        self, conversation_id: str
    ) -> list[str]:
        """Human-readable timeline of a conversation's tool usage."""
        entries = self.by_conversation(conversation_id)
        
        timeline = []
        for e in entries:
            status = "✅" if e.success else "❌"
            timeline.append(
                f"[Turn {e.turn_number}] {status} {e.tool_name}"
                f"({json.dumps(e.arguments)}) "
                f"→ {json.dumps(e.result_summary)} "
                f"[{e.duration_ms}ms]"
            )
        
        return timeline
```

**Example output from `conversation_timeline()`:**
```
[Turn 1] ✅ get_customer({"customer_id": "C-123"}) → {"success": true, "customer_id": "C-123"} [45ms]
[Turn 2] ✅ get_order({"order_id": "ORD-456"}) → {"success": true, "order_id": "ORD-456", "status": "shipped"} [62ms]
[Turn 3] ❌ update_order_status({"order_id": "ORD-456", "status": "cancelled"}) → {"success": false, "error": "Cannot cancel shipped order"} [38ms]
[Turn 3] ✅ initiate_return({"order_id": "ORD-456"}) → {"success": true, "return_id": "RET-789"} [156ms]
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Log every tool call — including failures | Complete trail for debugging and compliance |
| Sanitize sensitive data before logging | Passwords, tokens, and PII must never appear in logs |
| Include attribution chain (user → model → tool) | Know who is accountable for each action |
| Summarize results, don't log full payloads | Keeps logs compact and manageable |
| Use structured JSON logging | Enables querying, filtering, and analysis |
| Implement hash chains for tamper detection | Proves logs haven't been modified |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Logging passwords or API keys in arguments | Sanitize sensitive fields before logging |
| Logging full database query results (thousands of rows) | Summarize: count, key fields, truncation note |
| No way to query logs by conversation | Index by conversation_id for debugging |
| Mutable log entries (can be edited/deleted) | Use append-only storage with hash chain integrity |
| Logging only successful calls | Log failures too — they're often more important |
| No retention policy | Define retention periods by tool category and regulation |

---

## Hands-on exercise

### Your task

Build an audit logging system for a **medical records assistant** that:
- Logs access to patient records
- Tracks who viewed what and when
- Meets HIPAA-style requirements (immutable, 6-year retention)
- Can generate compliance reports

### Requirements

1. Log every tool call with full attribution
2. Sanitize patient PII (names, SSN, dates of birth) from log entries
3. Implement immutable logging with hash chain
4. Create a `generate_access_report` function that shows all accesses to a specific patient's records over a date range
5. Ensure logs cannot be deleted or modified

### Expected result

An audit system that records, protects, and reports on all patient record access.

<details>
<summary>💡 Hints (click to expand)</summary>

- Patient names should be logged as hashed identifiers, not plain text
- The hash chain ensures any tampering is detectable
- The access report should show: who accessed, when, which tool, what data was viewed
- Consider logging the "purpose" of access (treatment, billing, research)

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import json


# PII fields to hash (not log in plain text)
PII_FIELDS = {"patient_name", "ssn", "date_of_birth", "address", "phone"}

def hash_pii(value: str) -> str:
    """Hash PII for logging — preserves linkability without exposure."""
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}"


@dataclass
class MedicalAuditEntry:
    timestamp: str
    user_id: str
    user_role: str
    tool_name: str
    patient_id: str           # Internal ID — OK to log
    purpose: str              # treatment, billing, research
    arguments_sanitized: dict
    result_summary: dict
    success: bool
    previous_hash: str = ""
    entry_hash: str = ""


class HIPAAAuditLog:
    RETENTION_DAYS = 2190  # 6 years
    
    def __init__(self):
        self._entries: list[MedicalAuditEntry] = []
        self._previous_hash = "genesis"
    
    def log_access(
        self,
        tool_name: str,
        arguments: dict,
        result: dict,
        context: dict
    ) -> MedicalAuditEntry:
        """Log patient record access with PII sanitization."""
        
        # Sanitize PII from arguments
        safe_args = {}
        for key, value in arguments.items():
            if key in PII_FIELDS:
                safe_args[key] = hash_pii(str(value))
            else:
                safe_args[key] = value
        
        # Sanitize PII from results
        safe_result = {}
        for key, value in result.items():
            if key in PII_FIELDS:
                safe_result[key] = hash_pii(str(value))
            elif key == "error":
                safe_result[key] = value
            else:
                safe_result[key] = value
        
        entry = MedicalAuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=context["user_id"],
            user_role=context.get("user_role", "unknown"),
            tool_name=tool_name,
            patient_id=arguments.get("patient_id", "unknown"),
            purpose=context.get("access_purpose", "unspecified"),
            arguments_sanitized=safe_args,
            result_summary=safe_result,
            success="error" not in result,
            previous_hash=self._previous_hash
        )
        
        # Compute hash chain
        entry_data = json.dumps({
            "timestamp": entry.timestamp,
            "user_id": entry.user_id,
            "tool_name": entry.tool_name,
            "patient_id": entry.patient_id,
            "previous_hash": entry.previous_hash
        }, sort_keys=True)
        entry.entry_hash = hashlib.sha256(entry_data.encode()).hexdigest()
        self._previous_hash = entry.entry_hash
        
        self._entries.append(entry)
        return entry
    
    def generate_access_report(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Generate a compliance report for a patient's records."""
        
        accesses = []
        for entry in self._entries:
            if entry.patient_id != patient_id:
                continue
            
            entry_time = datetime.fromisoformat(entry.timestamp)
            if start_date <= entry_time <= end_date:
                accesses.append({
                    "timestamp": entry.timestamp,
                    "accessed_by": entry.user_id,
                    "role": entry.user_role,
                    "tool": entry.tool_name,
                    "purpose": entry.purpose,
                    "success": entry.success
                })
        
        return {
            "patient_id": patient_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_accesses": len(accesses),
            "unique_users": len(set(a["accessed_by"] for a in accesses)),
            "accesses": accesses,
            "integrity_verified": self.verify_integrity()
        }
    
    def verify_integrity(self) -> bool:
        """Verify hash chain is unbroken."""
        expected_prev = "genesis"
        for entry in self._entries:
            if entry.previous_hash != expected_prev:
                return False
            expected_prev = entry.entry_hash
        return True
```

</details>

### Bonus challenges

- [ ] Add real-time alerts for unusual access patterns (e.g., same patient accessed 10+ times in an hour)
- [ ] Implement log export in a standard format (e.g., FHIR AuditEvent)
- [ ] Add a "break the glass" emergency access mechanism that logs with elevated urgency

---

## Summary

✅ **Log every tool call** — including failures, arguments, results, and timing

✅ **Sanitize before logging** — never store passwords, API keys, or PII in plain text

✅ **Track the full attribution chain** — user → model → tool → result

✅ **Use immutable logging** — append-only with hash chain for tamper detection

✅ **Make logs queryable** — index by conversation, user, tool, and time range

✅ Retention policies must match regulatory requirements (7 years for finance, 6 for healthcare)

**Next:** [Lesson 11: Model Context Protocol (MCP) →](../11-model-context-protocol-mcp/00-model-context-protocol-mcp.md)

---

[← Previous: Rate Limiting](./08-rate-limiting.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- HIPAA Audit Log Requirements: https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html
-->
