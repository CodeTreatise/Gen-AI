---
title: "Idempotency"
---

# Idempotency

## Introduction

Models retry. Networks fail. Users double-click. If your `charge_payment` tool runs twice, the customer gets charged twice. Idempotency ensures that **calling the same operation multiple times produces the same result as calling it once**.

This isn't just a "nice to have" for AI tool use — it's critical. When a model doesn't receive a tool response (network timeout, context window overflow), it will often re-issue the same call. If your tool isn't idempotent, that retry creates a duplicate action.

### What we'll cover

- What makes an operation idempotent (and what doesn't)
- Idempotency keys for non-naturally-idempotent operations
- State verification before action
- Designing retry-safe tool handlers
- Duplicate detection patterns

### Prerequisites

- [Safe Defaults](./04-safe-defaults.md) — Defensive tool design
- [Lesson 06: Handling Responses](../06-handling-responses/00-handling-responses.md) — Processing tool results

---

## Naturally idempotent operations

Some operations are idempotent by nature — calling them multiple times has the same effect as calling once:

| Operation | Idempotent? | Why |
|-----------|-------------|-----|
| `GET /users/123` | ✅ Yes | Reading doesn't change state |
| `PUT /users/123 {"name": "Alice"}` | ✅ Yes | Setting to same value = same result |
| `DELETE /users/123` | ✅ Yes | Deleting already-deleted = no effect |
| `POST /orders` | ❌ No | Creates a new order every time |
| `POST /payments` | ❌ No | Charges money every time |
| `PATCH /counter {"increment": 1}` | ❌ No | Adds 1 each time |

Map this to tool design:

```python
# ✅ Naturally idempotent — safe to retry
idempotent_tools = [
    {
        "name": "get_customer",       # Read — always safe
        "description": "Get customer details by ID"
    },
    {
        "name": "set_order_status",   # Set (not toggle) — idempotent
        "description": "Set order status to a specific value"
    },
    {
        "name": "deactivate_account", # State transition — can't deactivate twice
        "description": "Deactivate an active account"
    }
]

# ❌ NOT naturally idempotent — needs protection
non_idempotent_tools = [
    {
        "name": "create_order",      # Creates duplicate if retried
        "description": "Create a new order"
    },
    {
        "name": "send_email",        # Sends duplicate emails
        "description": "Send an email to the customer"
    },
    {
        "name": "charge_payment",    # Double-charges the customer
        "description": "Process a payment"
    }
]
```

---

## Idempotency keys

For non-naturally-idempotent operations, use **idempotency keys** — unique identifiers that let the server recognize duplicate requests:

```python
import hashlib
import time
from datetime import datetime, timedelta


class IdempotencyStore:
    """Track idempotency keys to prevent duplicate operations."""
    
    def __init__(self, ttl_hours: int = 24):
        self._store: dict[str, dict] = {}
        self._ttl = timedelta(hours=ttl_hours)
    
    def check_and_set(self, key: str, result: dict | None = None) -> dict | None:
        """
        Check if an idempotency key has been used.
        Returns the cached result if duplicate, None if new.
        """
        self._cleanup_expired()
        
        if key in self._store:
            entry = self._store[key]
            return entry["result"]  # Return cached result
        
        # Mark as in-progress
        self._store[key] = {
            "created_at": datetime.utcnow(),
            "result": result
        }
        return None
    
    def complete(self, key: str, result: dict) -> None:
        """Store the result for a completed operation."""
        if key in self._store:
            self._store[key]["result"] = result
    
    def _cleanup_expired(self) -> None:
        """Remove expired keys."""
        now = datetime.utcnow()
        expired = [
            k for k, v in self._store.items()
            if now - v["created_at"] > self._ttl
        ]
        for k in expired:
            del self._store[k]


# Using idempotency keys in tool handlers
idempotency = IdempotencyStore()

def create_order(items: list[dict], idempotency_key: str) -> dict:
    """Create an order with idempotency protection."""
    
    # Check for duplicate
    cached = idempotency.check_and_set(idempotency_key)
    if cached is not None:
        return {
            **cached,
            "_note": "Duplicate request — returning cached result"
        }
    
    # Execute the operation
    order = db.orders.create(items=items)
    result = {
        "order_id": order.id,
        "status": "created",
        "items": items,
        "total": order.total
    }
    
    # Cache the result
    idempotency.complete(idempotency_key, result)
    return result
```

### Generating idempotency keys

The model shouldn't generate idempotency keys — your code should:

```python
def generate_idempotency_key(
    tool_name: str,
    args: dict,
    conversation_id: str,
    turn_number: int
) -> str:
    """
    Generate a deterministic idempotency key from the call context.
    Same tool + same args + same conversation turn = same key.
    """
    key_data = f"{tool_name}:{sorted(args.items())}:{conversation_id}:{turn_number}"
    return hashlib.sha256(key_data.encode()).hexdigest()


# In your tool execution middleware
def execute_tool(tool_name: str, args: dict, context: dict) -> dict:
    """Execute a tool with automatic idempotency protection."""
    
    # Generate key from call context
    key = generate_idempotency_key(
        tool_name=tool_name,
        args=args,
        conversation_id=context["conversation_id"],
        turn_number=context["turn_number"]
    )
    
    # Check for duplicate
    cached = idempotency.check_and_set(key)
    if cached is not None:
        return cached
    
    # Execute the actual tool
    handler = get_handler(tool_name)
    result = handler(**args)
    
    # Cache result
    idempotency.complete(key, result)
    return result
```

> **Note:** The idempotency key is generated from the **tool name + arguments + conversation context**, not from a random ID. This way, if the model retries the exact same call in the same turn, it gets the cached result automatically.

---

## State verification before action

Instead of blindly executing, verify the current state matches what the model expects:

```python
def cancel_order(order_id: str) -> dict:
    """Cancel an order — idempotent through state checking."""
    
    order = db.orders.get(order_id)
    
    if order is None:
        return {"error": "Order not found", "order_id": order_id}
    
    # Already cancelled — idempotent response
    if order.status == "cancelled":
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Order was already cancelled",
            "cancelled_at": order.cancelled_at.isoformat()
        }
    
    # Can't cancel if already shipped
    if order.status in ("shipped", "delivered"):
        return {
            "error": f"Cannot cancel — order is {order.status}",
            "order_id": order_id,
            "suggestion": "Use initiate_return instead"
        }
    
    # Execute cancellation
    order.status = "cancelled"
    order.cancelled_at = datetime.utcnow()
    db.orders.update(order)
    
    return {
        "order_id": order_id,
        "status": "cancelled",
        "cancelled_at": order.cancelled_at.isoformat()
    }
```

The key insight: if the operation was already performed, **return the same result** instead of an error. The model and user see the same outcome regardless of how many times the tool is called.

---

## Designing retry-safe handlers

A comprehensive pattern for tools that must be retry-safe:

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

class OperationStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class OperationRecord:
    key: str
    status: OperationStatus
    result: dict | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

class RetrySafeHandler:
    """Base class for retry-safe tool handlers."""
    
    def __init__(self):
        self._operations: dict[str, OperationRecord] = {}
    
    def execute(
        self,
        operation_key: str,
        action: callable,
        **kwargs
    ) -> dict:
        """Execute an operation with full retry safety."""
        
        # 1. Check if already completed
        if operation_key in self._operations:
            record = self._operations[operation_key]
            
            if record.status == OperationStatus.COMPLETED:
                return {
                    **record.result,
                    "_idempotent": True,
                    "_original_time": record.completed_at.isoformat()
                }
            
            if record.status == OperationStatus.PENDING:
                # Previous attempt still running or crashed
                # Check if it timed out (> 30 seconds)
                if datetime.utcnow() - record.created_at > timedelta(seconds=30):
                    # Timed out — allow retry
                    pass
                else:
                    return {
                        "status": "in_progress",
                        "message": "This operation is currently being processed"
                    }
        
        # 2. Mark as pending
        self._operations[operation_key] = OperationRecord(
            key=operation_key,
            status=OperationStatus.PENDING
        )
        
        # 3. Execute
        try:
            result = action(**kwargs)
            
            # 4. Mark as completed
            record = self._operations[operation_key]
            record.status = OperationStatus.COMPLETED
            record.result = result
            record.completed_at = datetime.utcnow()
            
            return result
            
        except Exception as e:
            # 5. Mark as failed (allows retry)
            record = self._operations[operation_key]
            record.status = OperationStatus.FAILED
            record.result = {"error": str(e)}
            
            return {"error": str(e), "retryable": True}


# Usage
handler = RetrySafeHandler()

def charge_payment(amount: float, customer_id: str, order_id: str) -> dict:
    """Process a payment — retry-safe."""
    
    # Generate deterministic key
    key = hashlib.sha256(
        f"charge:{customer_id}:{order_id}:{amount}".encode()
    ).hexdigest()
    
    return handler.execute(
        operation_key=key,
        action=payment_gateway.charge,
        amount=amount,
        customer_id=customer_id,
        order_id=order_id
    )
```

---

## Duplicate detection for messages

Sending duplicate messages (emails, SMS, notifications) is a common idempotency failure:

```python
class MessageDeduplicator:
    """Prevent duplicate message sends."""
    
    def __init__(self):
        self._sent: dict[str, datetime] = {}
    
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        conversation_id: str
    ) -> dict:
        """Send an email with deduplication."""
        
        # Create content hash for deduplication
        content_key = hashlib.sha256(
            f"{to}:{subject}:{conversation_id}".encode()
        ).hexdigest()
        
        if content_key in self._sent:
            sent_at = self._sent[content_key]
            return {
                "status": "already_sent",
                "sent_at": sent_at.isoformat(),
                "message": "This email was already sent in this conversation"
            }
        
        # Send the email
        result = email_service.send(to=to, subject=subject, body=body)
        self._sent[content_key] = datetime.utcnow()
        
        return {
            "status": "sent",
            "message_id": result.id,
            "to": to,
            "subject": subject
        }
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Make all write operations idempotent | Models and networks retry — duplicates are inevitable |
| Generate idempotency keys server-side | Don't rely on the model to provide unique keys |
| Include conversation context in keys | Same tool + args in different conversations should be independent |
| Return cached results for duplicates, not errors | The model treats the retry as successful |
| Set TTL on idempotency records | Don't store keys forever — 24 hours is usually sufficient |
| Verify state before acting | "Already cancelled" is better than "can't cancel — already cancelled" |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No idempotency protection on create/send operations | Add idempotency keys and deduplication |
| Returning errors for duplicate operations | Return the cached successful result instead |
| Making the model generate idempotency keys | Generate them server-side from call context |
| Storing idempotency keys forever | Set a 24-hour TTL and clean up expired entries |
| Only checking the tool name, not the arguments | Include arguments in the idempotency key |
| Toggle operations (increment, toggle) without guards | Use set-to-value instead of increment/toggle |

---

## Hands-on exercise

### Your task

Implement an idempotent **email notification** tool handler for an e-commerce system that:
- Sends order confirmation emails
- Sends shipping notification emails
- Sends delivery confirmation emails

### Requirements

1. Same email type for the same order should never send twice
2. Different email types for the same order are allowed (confirmation + shipping)
3. Return the cached result when a duplicate is detected
4. Include a TTL so very old orders can be re-notified
5. Handle the case where the email service is down (allow retry)

### Expected result

A handler class that passes these test scenarios:
- First send: delivers email ✅
- Immediate retry: returns cached result ✅
- Different email type, same order: delivers email ✅
- After TTL expires: delivers email ✅
- Email service error: allows retry ✅

<details>
<summary>💡 Hints (click to expand)</summary>

- The idempotency key should include: order_id + email_type
- Don't include email body in the key (it might change slightly)
- Failed sends should NOT be stored as "completed" — allow retry
- Use a TTL of 7 days for order emails

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass


class EmailType(Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SHIPPING_NOTIFICATION = "shipping_notification"
    DELIVERY_CONFIRMATION = "delivery_confirmation"


@dataclass
class SentRecord:
    key: str
    result: dict
    sent_at: datetime


class OrderEmailHandler:
    """Idempotent email notification handler."""
    
    def __init__(self, ttl_days: int = 7):
        self._sent: dict[str, SentRecord] = {}
        self._ttl = timedelta(days=ttl_days)
    
    def _make_key(self, order_id: str, email_type: EmailType) -> str:
        """Key = order + email type. Not body (might change)."""
        return f"{order_id}:{email_type.value}"
    
    def _is_expired(self, record: SentRecord) -> bool:
        """Check if a sent record has expired."""
        return datetime.utcnow() - record.sent_at > self._ttl
    
    def send_notification(
        self,
        order_id: str,
        email_type: str,
        recipient: str,
        subject: str,
        body: str
    ) -> dict:
        """Send an order notification email — idempotent."""
        
        # Parse email type
        try:
            etype = EmailType(email_type)
        except ValueError:
            return {"error": f"Invalid email type: {email_type}"}
        
        key = self._make_key(order_id, etype)
        
        # Check for existing send
        if key in self._sent:
            record = self._sent[key]
            
            if not self._is_expired(record):
                return {
                    **record.result,
                    "_duplicate": True,
                    "_original_sent_at": record.sent_at.isoformat()
                }
            else:
                # Expired — allow re-send
                del self._sent[key]
        
        # Attempt to send
        try:
            result = email_service.send(
                to=recipient,
                subject=subject,
                body=body
            )
        except EmailServiceError as e:
            # Failed — don't cache, allow retry
            return {
                "error": "Email service unavailable",
                "retryable": True,
                "details": str(e)
            }
        
        # Cache successful send
        send_result = {
            "status": "sent",
            "message_id": result.id,
            "order_id": order_id,
            "email_type": email_type,
            "recipient": recipient
        }
        
        self._sent[key] = SentRecord(
            key=key,
            result=send_result,
            sent_at=datetime.utcnow()
        )
        
        return send_result
```

</details>

### Bonus challenges

- [ ] Add a `resend_notification` tool that bypasses idempotency when the user explicitly requests it
- [ ] Implement persistent storage (database) instead of in-memory for production use
- [ ] Add metrics tracking: count duplicates detected vs. new sends

---

## Summary

✅ **All write operations should be idempotent** — retries are inevitable with AI tool calling

✅ **Generate idempotency keys server-side** from tool name + arguments + conversation context

✅ **Return cached results for duplicates** — not errors

✅ **Verify state before acting** — "already done" is a success, not a failure

✅ **Set TTLs on idempotency records** — 24 hours for most operations, longer for critical ones

✅ **Failed operations should allow retry** — only cache successful completions

**Next:** [Security Best Practices →](./07-security-best-practices.md)

---

[← Previous: Temperature Settings](./05-temperature-settings.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
- Stripe Idempotency Keys: https://stripe.com/docs/api/idempotent_requests
-->
