---
title: "Safe Defaults"
---

# Safe Defaults

## Introduction

When a model calls `delete_all_records()` because the user said "clean up my data," you don't get a second chance. Safe defaults mean your tools are designed so the **least dangerous thing happens by default**, and destructive actions require deliberate escalation.

This isn't about distrusting the model — it's about engineering systems where mistakes have limited blast radius. The same principle applies in software engineering: `rm` requires `-rf` for a reason.

### What we'll cover

- Read-before-write patterns that verify state first
- Confirmation gates for destructive operations
- Limited default scope and pagination
- Reversible actions and soft deletes
- Human-in-the-loop for high-stakes decisions

### Prerequisites

- [Atomic vs. Composite Tools](./01-atomic-vs-composite-tools.md) — Tool granularity
- [System Prompt Guidance](./03-system-prompt-guidance.md) — Usage rules

---

## Read before write

Every tool that **modifies** state should have a corresponding **read** tool. The model should verify current state before changing it:

```python
# Tool set: read-before-write pattern
tools = [
    {
        "name": "get_subscription",
        "description": (
            "Get current subscription details for a customer. "
            "Returns plan name, billing cycle, price, and renewal date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "change_subscription",
        "description": (
            "Change a customer's subscription plan. "
            "IMPORTANT: Call get_subscription first to verify the current plan. "
            "Returns the updated subscription details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "new_plan": {
                    "type": "string",
                    "enum": ["free", "starter", "professional", "enterprise"]
                }
            },
            "required": ["customer_id", "new_plan"]
        }
    }
]
```

Reinforce this in the system prompt:

```python
system_prompt = """
## Tool Rules
- ALWAYS call get_subscription BEFORE change_subscription
- Confirm the change with the user: "You're currently on [plan], 
  change to [new_plan] at [$price/month]?"
- Never change a subscription without showing the user 
  what will change and the price difference
"""
```

### Server-side enforcement

Don't rely on the model alone — enforce read-before-write in your handler:

```python
class SubscriptionHandler:
    def __init__(self):
        self._last_read: dict[str, dict] = {}  # Cache of recent reads
    
    def get_subscription(self, customer_id: str) -> dict:
        """Read current subscription — cached for verification."""
        subscription = db.get_subscription(customer_id)
        self._last_read[customer_id] = subscription
        return subscription
    
    def change_subscription(
        self, customer_id: str, new_plan: str
    ) -> dict:
        """Change subscription — requires prior read."""
        # Verify the model read the current state
        if customer_id not in self._last_read:
            return {
                "error": "Must call get_subscription first",
                "action_required": "Read current subscription before changing"
            }
        
        current = self._last_read[customer_id]
        
        # Prevent no-op changes
        if current["plan"] == new_plan:
            return {
                "error": "Customer is already on this plan",
                "current_plan": current["plan"]
            }
        
        # Execute the change
        result = db.update_subscription(customer_id, new_plan)
        del self._last_read[customer_id]  # Invalidate cache
        return result
```

---

## Confirmation gates

Destructive operations should require explicit user confirmation. Design your tools to support a two-step process:

```python
# Step 1: Preview what will happen
{
    "name": "preview_delete",
    "description": (
        "Preview what would be deleted without actually deleting anything. "
        "Returns the list of items that match the deletion criteria "
        "and their total count."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "description": "Filter criteria for items to delete"
            }
        },
        "required": ["filter"]
    }
}

# Step 2: Execute with confirmation token
{
    "name": "execute_delete",
    "description": (
        "Delete items matching a previously previewed filter. "
        "Requires the confirmation_token from preview_delete. "
        "This action is irreversible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "confirmation_token": {
                "type": "string",
                "description": "Token from preview_delete that confirms the scope"
            }
        },
        "required": ["confirmation_token"]
    }
}
```

### Server-side confirmation tokens

```python
import hashlib
import time
from dataclasses import dataclass

@dataclass
class PendingAction:
    action: str
    scope: dict
    item_count: int
    created_at: float
    token: str

class ConfirmationGate:
    """Require preview + confirm for destructive operations."""
    
    def __init__(self, expiry_seconds: int = 300):
        self._pending: dict[str, PendingAction] = {}
        self._expiry = expiry_seconds
    
    def preview(self, action: str, scope: dict) -> dict:
        """Preview a destructive action and return a confirmation token."""
        # Find what would be affected
        items = db.query(scope)
        
        # Generate a token tied to this specific action
        token = hashlib.sha256(
            f"{action}:{scope}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        self._pending[token] = PendingAction(
            action=action,
            scope=scope,
            item_count=len(items),
            created_at=time.time(),
            token=token
        )
        
        return {
            "confirmation_token": token,
            "action": action,
            "items_affected": len(items),
            "preview": [item.summary() for item in items[:10]],
            "message": f"This will {action} {len(items)} items. Confirm?"
        }
    
    def execute(self, token: str) -> dict:
        """Execute a previously previewed action."""
        if token not in self._pending:
            return {"error": "Invalid or expired confirmation token"}
        
        pending = self._pending[token]
        
        # Check expiry
        if time.time() - pending.created_at > self._expiry:
            del self._pending[token]
            return {
                "error": "Confirmation expired (5 minute limit). "
                         "Please preview again."
            }
        
        # Execute the action
        result = db.execute_action(pending.action, pending.scope)
        del self._pending[token]
        
        return {
            "success": True,
            "items_affected": pending.item_count,
            "action": pending.action
        }
```

> **Note:** Gemini specifically recommends: "Validate the call with the user before executing it." This preview-then-confirm pattern does exactly that.

---

## Limited default scope

Tools should default to the **narrowest possible scope**:

```python
# ❌ Dangerous default — returns everything
{
    "name": "list_users",
    "description": "List users in the system",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max results (default: unlimited)"
            }
        }
    }
}

# ✅ Safe default — limited scope and pagination
{
    "name": "list_users",
    "description": (
        "List users with pagination. Returns up to 25 users per page. "
        "Use the next_cursor from the response to get the next page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "page_cursor": {
                "type": "string",
                "description": "Cursor from previous response for next page. Omit for first page."
            },
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "suspended"],
                "description": "Filter by account status"
            }
        }
    }
}
```

### Apply limits at the handler level

```python
MAX_RESULTS = 25
MAX_BULK_ACTION = 50

def list_users(page_cursor: str | None = None, status: str | None = None) -> dict:
    """Always paginate, never return unbounded results."""
    query = db.users.query()
    
    if status:
        query = query.filter(status=status)
    if page_cursor:
        query = query.after(page_cursor)
    
    # Hard cap regardless of what the model asks for
    results = query.limit(MAX_RESULTS + 1).execute()
    
    has_more = len(results) > MAX_RESULTS
    users = results[:MAX_RESULTS]
    
    return {
        "users": [u.to_dict() for u in users],
        "count": len(users),
        "next_cursor": users[-1].id if has_more else None,
        "has_more": has_more
    }

def bulk_update(user_ids: list[str], updates: dict) -> dict:
    """Cap bulk operations to prevent mass changes."""
    if len(user_ids) > MAX_BULK_ACTION:
        return {
            "error": f"Bulk update limited to {MAX_BULK_ACTION} users at a time",
            "requested": len(user_ids),
            "suggestion": "Split into smaller batches"
        }
    
    results = []
    for uid in user_ids:
        results.append(db.users.update(uid, updates))
    
    return {"updated": len(results), "results": results}
```

---

## Reversible actions and soft deletes

Prefer operations that can be undone:

```python
# ❌ Hard delete — permanent
def delete_project(project_id: str) -> dict:
    db.projects.delete(project_id)  # Gone forever
    return {"deleted": project_id}

# ✅ Soft delete — recoverable
def archive_project(project_id: str) -> dict:
    """Move project to archive. Can be restored within 30 days."""
    project = db.projects.get(project_id)
    if not project:
        return {"error": "Project not found"}
    
    project.status = "archived"
    project.archived_at = datetime.utcnow()
    project.auto_delete_at = datetime.utcnow() + timedelta(days=30)
    db.projects.update(project)
    
    return {
        "archived": project_id,
        "can_restore_until": project.auto_delete_at.isoformat(),
        "message": "Project archived. It can be restored within 30 days."
    }
```

Name the tool `archive_project`, not `delete_project`. The name itself communicates reversibility.

### Expose undo as a companion tool

```python
tools = [
    {
        "name": "archive_project",
        "description": "Archive a project. Can be restored within 30 days."
    },
    {
        "name": "restore_project",
        "description": (
            "Restore a previously archived project. "
            "Only works within 30 days of archiving."
        )
    }
]
```

---

## Human-in-the-loop for high-stakes decisions

Some actions should always involve a human decision-maker. Design tools that **pause for approval** rather than executing immediately:

```python
def request_approval(
    action: str,
    details: dict,
    approver_role: str = "manager"
) -> dict:
    """Request human approval for a high-stakes action."""
    request_id = create_approval_request(
        action=action,
        details=details,
        approver_role=approver_role
    )
    
    # Notify the approver
    notify_approver(request_id, approver_role)
    
    return {
        "status": "pending_approval",
        "request_id": request_id,
        "message": (
            f"This action requires {approver_role} approval. "
            f"Request {request_id} has been submitted. "
            "You'll be notified when it's approved or denied."
        )
    }

# System prompt guidance
system_prompt = """
## High-Stakes Actions (Require Approval)
These actions submit an approval request instead of executing immediately:
- Refunds over $500 → manager approval
- Account deletion → customer confirmation + admin approval
- Data export → security team approval
- Bulk operations on 100+ records → team lead approval

When calling these, tell the user: "I've submitted this for approval. 
You'll receive a notification when it's processed."
"""
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Read before write — always verify current state | Prevents acting on stale or wrong assumptions |
| Preview before execute for destructive actions | User sees impact before anything changes |
| Hard-cap all list/query results | Prevents unbounded data returns |
| Use soft deletes (archive) over hard deletes | Mistakes are recoverable |
| Expire confirmation tokens after 5 minutes | Prevents stale confirmations from being reused |
| Name tools by their safe behavior | `archive_project` not `delete_project` |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Delete tool with no confirmation step | Add preview + confirmation token pattern |
| Unlimited query results by default | Hard-cap at 25-50 results with pagination |
| Trusting the model to always read before writing | Enforce read-before-write in your handler code |
| Permanent deletion as the only option | Implement soft delete with time-limited restoration |
| Confirmation that never expires | Add 5-minute expiry to confirmation tokens |
| Relying solely on system prompt for safety | Enforce constraints in handler code — defense in depth |

---

## Hands-on exercise

### Your task

Design a safe tool set for a **user account management** system that can:
- View account details
- Update email address
- Change account role (user, editor, admin)
- Deactivate an account

### Requirements

1. Implement the read-before-write pattern
2. Add confirmation gates for role changes and deactivation
3. Use soft deactivation (not deletion)
4. Write the system prompt section that enforces safe usage
5. Include at least one server-side safety check in your handler

### Expected result

A set of 3-5 tool definitions plus a system prompt section plus a handler with safety enforcement.

<details>
<summary>💡 Hints (click to expand)</summary>

- Changing someone to admin is high-stakes — require extra confirmation
- Deactivation should be reversible (reactivation tool)
- The system prompt should require `get_account` before any modification
- Server-side: check that you can't deactivate the last admin

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# Tool definitions
account_tools = [
    {
        "name": "get_account",
        "description": (
            "Get account details for a user. Returns name, email, role, "
            "status (active/deactivated), and last login date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID, e.g. 'USR-12345'"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "update_email",
        "description": (
            "Update a user's email address. Sends a verification email "
            "to the new address. Call get_account first to confirm identity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "new_email": {"type": "string", "description": "New email address"}
            },
            "required": ["user_id", "new_email"]
        }
    },
    {
        "name": "change_role",
        "description": (
            "Change a user's account role. Admin role changes require "
            "additional confirmation. Call get_account first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "new_role": {
                    "type": "string",
                    "enum": ["user", "editor", "admin"]
                }
            },
            "required": ["user_id", "new_role"]
        }
    },
    {
        "name": "deactivate_account",
        "description": (
            "Deactivate a user account. The account is preserved and can be "
            "reactivated within 90 days. Call get_account first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "reason": {
                    "type": "string",
                    "enum": ["user_request", "policy_violation", "inactivity"]
                }
            },
            "required": ["user_id", "reason"]
        }
    },
    {
        "name": "reactivate_account",
        "description": (
            "Reactivate a deactivated account. Only works within 90 days "
            "of deactivation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"}
            },
            "required": ["user_id"]
        }
    }
]

# System prompt section
system_prompt_section = """
## Account Management Rules
1. ALWAYS call get_account before any modification tool
2. Show the user the current state and what will change
3. For change_role to admin: ask "Are you sure? This grants full system access."
4. For deactivate_account: confirm with "This will deactivate [name]'s account. 
   It can be reactivated within 90 days. Proceed?"
5. NEVER deactivate without asking for a reason
"""

# Server-side handler with safety checks
class AccountHandler:
    def __init__(self):
        self._verified: set[str] = set()  # Users that were read first
    
    def get_account(self, user_id: str) -> dict:
        account = db.get_user(user_id)
        self._verified.add(user_id)
        return account.to_dict()
    
    def change_role(self, user_id: str, new_role: str) -> dict:
        if user_id not in self._verified:
            return {"error": "Call get_account first"}
        
        current = db.get_user(user_id)
        
        # Safety: don't demote the last admin
        if current.role == "admin" and new_role != "admin":
            admin_count = db.count_users(role="admin", status="active")
            if admin_count <= 1:
                return {
                    "error": "Cannot remove the last admin. "
                             "Promote another user to admin first."
                }
        
        db.update_user(user_id, role=new_role)
        return {"updated": user_id, "old_role": current.role, "new_role": new_role}
    
    def deactivate_account(self, user_id: str, reason: str) -> dict:
        if user_id not in self._verified:
            return {"error": "Call get_account first"}
        
        current = db.get_user(user_id)
        
        # Safety: don't deactivate the last admin
        if current.role == "admin":
            admin_count = db.count_users(role="admin", status="active")
            if admin_count <= 1:
                return {"error": "Cannot deactivate the last admin"}
        
        db.update_user(
            user_id,
            status="deactivated",
            deactivated_at=datetime.utcnow(),
            deactivation_reason=reason,
            auto_purge_at=datetime.utcnow() + timedelta(days=90)
        )
        
        return {
            "deactivated": user_id,
            "reason": reason,
            "can_reactivate_until": (
                datetime.utcnow() + timedelta(days=90)
            ).isoformat()
        }
```

</details>

### Bonus challenges

- [ ] Add a `permanently_delete_account` tool that requires a separate approval workflow
- [ ] Implement rate limiting so `change_role` can only be called 3 times per hour per user
- [ ] Add an audit log that records who made each change and when

---

## Summary

✅ **Read before write** — verify current state before modifying anything

✅ **Preview before execute** — show the user what will change before it changes

✅ **Limit default scope** — paginate results, cap bulk operations

✅ **Prefer reversible actions** — soft delete and archive over permanent deletion

✅ **Enforce safety server-side** — system prompts guide the model, but handlers enforce constraints

✅ **Use confirmation tokens with expiry** — prevent stale or replayed confirmations

**Next:** [Temperature Settings →](./05-temperature-settings.md)

---

[← Previous: System Prompt Guidance](./03-system-prompt-guidance.md) | [Back to Lesson Overview](./00-tool-design-best-practices.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Google Gemini Function Calling (Best Practices): https://ai.google.dev/gemini-api/docs/function-calling
- Anthropic Tool Use Overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
-->
