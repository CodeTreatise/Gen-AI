---
title: "Confirmation workflows"
---

# Confirmation workflows

## Introduction

Before an agent sends an email, deletes a record, or processes a payment, the user should see exactly what's about to happen and have the chance to say "yes," "no," or "change this." Confirmation workflows are the most common human-in-the-loop pattern — they protect against irreversible mistakes while keeping the agent in control of the overall task.

In this lesson, we'll build confirmation systems that preview actions clearly, collect approve/reject decisions, allow modifications before execution, and handle batches of actions efficiently.

### What we'll cover

- Presenting action previews that humans can quickly evaluate
- Building approve/reject flows with LangGraph's `interrupt()`
- Allowing users to modify proposed actions before execution
- Processing multiple confirmations efficiently with batch approval

### Prerequisites

- [Human-in-the-Loop overview](./00-human-in-the-loop.md) — the HITL spectrum
- [Error Handling — Human Escalation](../08-error-handling-recovery/06-human-escalation-triggers.md) — when to involve humans
- [State Management](../07-state-management/) — persisting state across pauses

---

## Action preview

A good confirmation workflow starts with a clear preview. The user needs to understand *what* the agent wants to do, *why* it wants to do it, and *what will change* — all at a glance.

### Designing action previews

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionRisk(Enum):
    LOW = "low"          # Reversible, no external impact
    MEDIUM = "medium"    # External impact but recoverable
    HIGH = "high"        # Irreversible or sensitive
    CRITICAL = "critical"  # Financial, legal, or safety impact


class ActionType(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEND = "send"
    EXECUTE = "execute"


@dataclass
class ActionPreview:
    """A human-readable preview of what the agent wants to do."""
    
    action_type: ActionType
    description: str        # "Send email to jane@example.com"
    reason: str             # "User asked to notify Jane about the meeting change"
    risk_level: ActionRisk
    details: dict = field(default_factory=dict)  # Action-specific data
    reversible: bool = True
    estimated_impact: str = ""  # "1 recipient will receive an email"
    
    def format_for_human(self) -> str:
        """Create a clear, scannable preview."""
        risk_icons = {
            ActionRisk.LOW: "🟢",
            ActionRisk.MEDIUM: "🟡",
            ActionRisk.HIGH: "🔴",
            ActionRisk.CRITICAL: "🚨"
        }
        icon = risk_icons[self.risk_level]
        reversible_text = "✅ Reversible" if self.reversible else "⚠️ Irreversible"
        
        lines = [
            f"{icon} **{self.action_type.value.upper()}** — {self.description}",
            f"",
            f"**Why:** {self.reason}",
            f"**Risk:** {self.risk_level.value} | {reversible_text}",
        ]
        
        if self.estimated_impact:
            lines.append(f"**Impact:** {self.estimated_impact}")
        
        if self.details:
            lines.append("")
            lines.append("**Details:**")
            for key, value in self.details.items():
                display_key = key.replace("_", " ").title()
                lines.append(f"  • {display_key}: {value}")
        
        return "\n".join(lines)


# Usage
preview = ActionPreview(
    action_type=ActionType.SEND,
    description="Send email to jane@example.com",
    reason="User asked to notify Jane about the meeting time change",
    risk_level=ActionRisk.MEDIUM,
    reversible=False,
    estimated_impact="1 recipient will receive an email",
    details={
        "to": "jane@example.com",
        "subject": "Meeting rescheduled to 3 PM",
        "body": "Hi Jane, the project sync has been moved to 3 PM today..."
    }
)

print(preview.format_for_human())
```

**Output:**
```
🟡 **SEND** — Send email to jane@example.com

**Why:** User asked to notify Jane about the meeting time change
**Risk:** medium | ⚠️ Irreversible
**Impact:** 1 recipient will receive an email

**Details:**
  • To: jane@example.com
  • Subject: Meeting rescheduled to 3 PM
  • Body: Hi Jane, the project sync has been moved to 3 PM today...
```

> **🔑 Key concept:** A preview should answer three questions in under 5 seconds: *What?* (the action), *Why?* (the reason), and *So what?* (the impact). If it takes longer to read the preview than to do the task manually, the HITL pattern is adding friction without value.

---

## Approve/reject with LangGraph

LangGraph's `interrupt()` function is purpose-built for confirmation workflows. It pauses the graph, sends a payload to the caller, and resumes when the human responds:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal


class EmailState(TypedDict):
    user_request: str
    email_to: str
    email_subject: str
    email_body: str
    status: str


def draft_email(state: EmailState) -> dict:
    """Agent drafts the email based on user request."""
    # In production, an LLM generates this
    return {
        "email_to": "jane@example.com",
        "email_subject": "Meeting rescheduled to 3 PM",
        "email_body": "Hi Jane, the project sync has been moved to 3 PM today.",
        "status": "drafted"
    }


def confirm_send(state: EmailState) -> Command[Literal["send_email", "cancel"]]:
    """Pause for human confirmation before sending."""
    
    # This payload appears in result["__interrupt__"]
    response = interrupt({
        "action": "send_email",
        "preview": {
            "to": state["email_to"],
            "subject": state["email_subject"],
            "body": state["email_body"]
        },
        "question": "Send this email?",
        "options": ["approve", "reject", "modify"]
    })
    
    # This code runs AFTER the human responds
    if response.get("decision") == "approve":
        return Command(goto="send_email")
    elif response.get("decision") == "modify":
        # Human provided modifications
        return Command(
            goto="send_email",
            update={
                "email_to": response.get("to", state["email_to"]),
                "email_subject": response.get("subject", state["email_subject"]),
                "email_body": response.get("body", state["email_body"])
            }
        )
    else:
        return Command(goto="cancel")


def send_email(state: EmailState) -> dict:
    """Actually send the email."""
    print(f"📧 Email sent to {state['email_to']}: {state['email_subject']}")
    return {"status": "sent"}


def cancel(state: EmailState) -> dict:
    """Cancel the email."""
    print("❌ Email cancelled by user")
    return {"status": "cancelled"}


# Build the graph
builder = StateGraph(EmailState)
builder.add_node("draft", draft_email)
builder.add_node("confirm", confirm_send)
builder.add_node("send_email", send_email)
builder.add_node("cancel", cancel)

builder.add_edge(START, "draft")
builder.add_edge("draft", "confirm")
builder.add_edge("send_email", END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=MemorySaver())

# --- Execution Flow ---
config = {"configurable": {"thread_id": "email-1"}}

# Step 1: Run until interrupt
result = graph.invoke(
    {
        "user_request": "Email Jane about the meeting change",
        "email_to": "", "email_subject": "", "email_body": "",
        "status": ""
    },
    config=config
)
# Graph is now PAUSED — result["__interrupt__"] contains the preview

# Step 2: Human approves
result = graph.invoke(
    Command(resume={"decision": "approve"}),
    config=config
)
print(f"Status: {result['status']}")
```

**Output:**
```
📧 Email sent to jane@example.com: Meeting rescheduled to 3 PM
Status: sent
```

> **Warning:** Remember the rules of `interrupt()`:
> 1. **Never wrap in try/except** — `interrupt()` raises a special exception internally
> 2. **The node re-runs from the beginning on resume** — side effects before `interrupt()` must be idempotent
> 3. **Keep payloads JSON-serializable** — no Python objects or functions
> 4. **Don't reorder interrupt calls** — index-based matching requires consistent order

---

## Modification options

Often, the human doesn't want to fully approve or fully reject — they want to *tweak* the action. A good confirmation workflow supports inline modifications:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModifiableAction:
    """An action where specific fields can be modified before execution."""
    
    action_type: str
    fields: dict[str, Any]
    editable_fields: list[str]  # Which fields the user can change
    locked_fields: list[str] = field(default_factory=list)  # Fields that can't change
    
    def get_editable_summary(self) -> dict:
        """Return the fields a human can modify."""
        return {
            "editable": {
                k: v for k, v in self.fields.items() 
                if k in self.editable_fields
            },
            "locked": {
                k: v for k, v in self.fields.items() 
                if k in self.locked_fields
            }
        }
    
    def apply_modifications(self, modifications: dict) -> dict:
        """Apply human modifications, enforcing locked fields."""
        result = dict(self.fields)
        
        for key, value in modifications.items():
            if key in self.locked_fields:
                print(f"⚠️ Cannot modify locked field: {key}")
                continue
            if key in self.editable_fields:
                result[key] = value
            else:
                print(f"⚠️ Unknown field: {key}")
        
        return result


# Usage
action = ModifiableAction(
    action_type="send_email",
    fields={
        "to": "jane@example.com",
        "subject": "Meeting rescheduled to 3 PM",
        "body": "Hi Jane, the project sync has been moved...",
        "from": "agent@company.com",  # Can't change the sender
    },
    editable_fields=["to", "subject", "body"],
    locked_fields=["from"]
)

# Human wants to change the subject
summary = action.get_editable_summary()
print("Editable:", list(summary["editable"].keys()))
print("Locked:", list(summary["locked"].keys()))

modified = action.apply_modifications({
    "subject": "URGENT: Meeting rescheduled to 3 PM",
    "from": "hacker@evil.com"  # Blocked!
})
print(f"\nFinal subject: {modified['subject']}")
print(f"Final from: {modified['from']}")
```

**Output:**
```
Editable: ['to', 'subject', 'body']
Locked: ['from']
⚠️ Cannot modify locked field: from

Final subject: URGENT: Meeting rescheduled to 3 PM
Final from: agent@company.com
```

### Integrating modification with LangGraph interrupt

In a LangGraph node, the modification flow uses the resume value to carry edits:

```python
from langgraph.types import interrupt, Command
from typing import Literal


def confirm_with_edits(state: dict) -> Command[Literal["execute", "cancel"]]:
    """Confirmation that supports approve, reject, and modify."""
    
    response = interrupt({
        "action": "database_update",
        "preview": {
            "table": state["table"],
            "record_id": state["record_id"],
            "changes": state["proposed_changes"]
        },
        "editable_fields": ["changes"],
        "locked_fields": ["table", "record_id"],
        "options": ["approve", "modify", "reject"]
    })
    
    decision = response.get("decision", "reject")
    
    if decision == "approve":
        return Command(goto="execute")
    elif decision == "modify":
        # Apply only allowed modifications
        edits = response.get("edits", {})
        allowed_edits = {
            k: v for k, v in edits.items()
            if k not in ["table", "record_id"]
        }
        return Command(goto="execute", update=allowed_edits)
    else:
        return Command(goto="cancel")
```

---

## Batch approval

When an agent generates multiple actions — like sending 20 emails or updating 50 records — asking for individual confirmation on each one is impractical. Batch approval lets humans review and decide on groups of actions together.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BatchDecision(Enum):
    APPROVE_ALL = "approve_all"
    REJECT_ALL = "reject_all"
    SELECTIVE = "selective"  # Review each item


@dataclass
class BatchItem:
    """A single item in a batch for review."""
    item_id: str
    description: str
    risk_level: str
    approved: Optional[bool] = None


@dataclass
class BatchConfirmation:
    """Manages batch approval workflows."""
    
    items: list[BatchItem] = field(default_factory=list)
    auto_approve_below: str = "low"  # Auto-approve low-risk items
    
    def add_item(self, item_id: str, description: str, risk_level: str):
        self.items.append(BatchItem(item_id, description, risk_level))
    
    def get_summary(self) -> dict:
        """Summary for human review."""
        risk_counts = {}
        for item in self.items:
            risk_counts[item.risk_level] = risk_counts.get(item.risk_level, 0) + 1
        
        return {
            "total_items": len(self.items),
            "risk_breakdown": risk_counts,
            "needs_review": [
                {
                    "id": item.item_id,
                    "description": item.description,
                    "risk": item.risk_level
                }
                for item in self.items
                if item.risk_level != self.auto_approve_below
            ],
            "auto_approved": sum(
                1 for item in self.items 
                if item.risk_level == self.auto_approve_below
            )
        }
    
    def apply_decision(self, decision: BatchDecision,
                       selective_decisions: Optional[dict] = None) -> dict:
        """Apply human's batch decision."""
        
        approved = []
        rejected = []
        
        for item in self.items:
            if decision == BatchDecision.APPROVE_ALL:
                item.approved = True
                approved.append(item.item_id)
            
            elif decision == BatchDecision.REJECT_ALL:
                item.approved = False
                rejected.append(item.item_id)
            
            elif decision == BatchDecision.SELECTIVE:
                if item.risk_level == self.auto_approve_below:
                    item.approved = True
                    approved.append(item.item_id)
                elif selective_decisions and item.item_id in selective_decisions:
                    item.approved = selective_decisions[item.item_id]
                    if item.approved:
                        approved.append(item.item_id)
                    else:
                        rejected.append(item.item_id)
                else:
                    item.approved = False
                    rejected.append(item.item_id)
        
        return {
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "approved_ids": approved,
            "rejected_ids": rejected
        }


# Usage
batch = BatchConfirmation(auto_approve_below="low")

# Agent wants to update 5 records
batch.add_item("rec-1", "Update user name: John → Jonathan", "low")
batch.add_item("rec-2", "Update user name: Jane → Janet", "low")
batch.add_item("rec-3", "Update user email: old@co.com → new@co.com", "medium")
batch.add_item("rec-4", "Delete inactive user: bob@co.com", "high")
batch.add_item("rec-5", "Update user role: viewer → admin", "high")

summary = batch.get_summary()
print(f"Total: {summary['total_items']} items")
print(f"Auto-approved (low risk): {summary['auto_approved']}")
print(f"Needs review: {len(summary['needs_review'])} items")
for item in summary["needs_review"]:
    print(f"  [{item['risk']}] {item['description']}")

# Human selectively approves
result = batch.apply_decision(
    BatchDecision.SELECTIVE,
    selective_decisions={"rec-3": True, "rec-4": False, "rec-5": True}
)
print(f"\n✅ Approved: {result['approved_count']}")
print(f"❌ Rejected: {result['rejected_count']}")
```

**Output:**
```
Total: 5 items
Auto-approved (low risk): 2
Needs review: 3 items
  [medium] Update user email: old@co.com → new@co.com
  [high] Delete inactive user: bob@co.com
  [high] Update user role: viewer → admin

✅ Approved: 4
❌ Rejected: 1
```

> **💡 Tip:** Use risk-based auto-approval to reduce human effort. Low-risk, reversible actions can be auto-approved while high-risk actions require explicit review. This keeps the human focused on decisions that actually matter.

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Show a preview *before* asking for a decision | Humans can't approve what they don't understand |
| Include the "why" in every preview | Context makes approval decisions faster and more accurate |
| Support "modify" alongside approve/reject | Most real decisions are "yes, but change this one thing" |
| Use risk-based auto-approval for batches | Reviewing 50 low-risk items wastes human attention |
| Keep previews scannable (< 5 seconds) | Lengthy previews cause fatigue and rubber-stamping |
| Make cancellation safe and consequence-free | Users should never hesitate to reject a bad action |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Asking for confirmation on every action, including reads | Only confirm actions with side effects (create, update, delete, send) |
| Showing raw JSON or API payloads as the preview | Format previews into human-readable summaries with clear labels |
| No modification option — only approve/reject | Let users edit specific fields without rejecting the entire action |
| No timeout on confirmation requests | Set a maximum wait time; auto-reject or notify if no response |
| Applying modifications without validation | Validate human edits the same way you'd validate agent actions |
| Asking for batch confirmation without a summary | Always show counts, risk breakdown, and auto-approval status |

---

## Hands-on exercise

### Your task

Build a `ConfirmationWorkflow` class that generates previews, collects decisions, supports modifications, and handles batch approval.

### Requirements

1. Create `ActionPreview` objects from tool calls, including risk assessment
2. Support three decision types: approve, reject, and modify
3. Implement batch confirmation with risk-based auto-approval
4. Track confirmation history (what was approved, when, by whom)
5. Include a `format_receipt()` method that shows what was decided after the fact

### Expected result

```python
workflow = ConfirmationWorkflow()

# Single confirmation
preview = workflow.create_preview(
    action="send_email",
    details={"to": "jane@co.com", "subject": "Hello"},
    risk="medium"
)
result = workflow.confirm(preview, decision="modify", edits={"subject": "Hi Jane!"})
# result = {"approved": True, "modified": True, "final_action": {...}}

# Batch confirmation  
batch = workflow.create_batch([...])
result = workflow.confirm_batch(batch, decision="selective", ...)
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `ActionPreview` and `BatchConfirmation` classes from this lesson as building blocks
- The confirmation history should include timestamps and the human's identity
- `format_receipt()` can iterate over the history and produce a markdown-style summary
- Consider adding a `requires_confirmation()` method that checks risk level to decide if confirmation is needed at all

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ConfirmationRecord:
    action: str
    decision: str  # "approved", "rejected", "modified"
    original_details: dict
    final_details: dict
    decided_by: str
    decided_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class ConfirmationWorkflow:
    """Manages confirmation workflows with history tracking."""
    
    def __init__(self, auto_approve_risk: str = "low"):
        self.auto_approve_risk = auto_approve_risk
        self.history: list[ConfirmationRecord] = []
        self.risk_levels = ["low", "medium", "high", "critical"]
    
    def requires_confirmation(self, risk: str) -> bool:
        """Check if this risk level needs human confirmation."""
        return self.risk_levels.index(risk) > self.risk_levels.index(
            self.auto_approve_risk
        )
    
    def create_preview(
        self, action: str, details: dict, risk: str, reason: str = ""
    ) -> dict:
        """Create an action preview for human review."""
        return {
            "action": action,
            "details": details,
            "risk": risk,
            "reason": reason,
            "needs_confirmation": self.requires_confirmation(risk)
        }
    
    def confirm(
        self,
        preview: dict,
        decision: str,
        edits: Optional[dict] = None,
        decided_by: str = "human"
    ) -> dict:
        """Process a confirmation decision."""
        final_details = dict(preview["details"])
        
        if decision == "modify" and edits:
            final_details.update(edits)
            effective_decision = "modified"
        elif decision == "approve":
            effective_decision = "approved"
        else:
            effective_decision = "rejected"
        
        record = ConfirmationRecord(
            action=preview["action"],
            decision=effective_decision,
            original_details=preview["details"],
            final_details=final_details,
            decided_by=decided_by
        )
        self.history.append(record)
        
        return {
            "approved": decision in ("approve", "modify"),
            "modified": decision == "modify",
            "final_action": {
                "action": preview["action"],
                "details": final_details
            }
        }
    
    def create_batch(self, items: list[dict]) -> list[dict]:
        """Create a batch of previews."""
        return [
            self.create_preview(**item)
            for item in items
        ]
    
    def confirm_batch(
        self,
        batch: list[dict],
        decision: str = "selective",
        selective: Optional[dict] = None,
        decided_by: str = "human"
    ) -> dict:
        """Process batch confirmation."""
        results = {"approved": [], "rejected": []}
        
        for i, preview in enumerate(batch):
            if decision == "approve_all":
                item_decision = "approve"
            elif decision == "reject_all":
                item_decision = "reject"
            elif not preview["needs_confirmation"]:
                item_decision = "approve"
            elif selective and str(i) in selective:
                item_decision = selective[str(i)]
            else:
                item_decision = "reject"
            
            result = self.confirm(
                preview, item_decision, decided_by=decided_by
            )
            key = "approved" if result["approved"] else "rejected"
            results[key].append(preview["action"])
        
        return results
    
    def format_receipt(self) -> str:
        """Generate a receipt of all confirmation decisions."""
        lines = ["## Confirmation Receipt", ""]
        for i, record in enumerate(self.history, 1):
            icon = {"approved": "✅", "rejected": "❌", "modified": "✏️"}
            lines.append(
                f"{i}. {icon.get(record.decision, '❓')} "
                f"**{record.action}** — {record.decision} "
                f"by {record.decided_by} at {record.decided_at}"
            )
        return "\n".join(lines)
```
</details>

### Bonus challenges

- [ ] Add a confirmation timeout that auto-rejects after a configurable duration
- [ ] Implement a "remember this decision" feature that auto-approves future identical actions
- [ ] Build a web-based confirmation UI using Flask or FastAPI that renders previews as HTML

---

## Summary

✅ **Action previews** should answer What/Why/Impact in under 5 seconds — raw JSON payloads are not acceptable for human review

✅ **LangGraph's `interrupt()`** creates natural confirmation points where the graph pauses, shows a preview, and resumes based on the human's decision

✅ **Modification support** is essential — most real-world decisions are "approve with changes," not binary approve/reject

✅ **Batch approval** with risk-based auto-approval keeps humans focused on decisions that matter, not rubber-stamping low-risk actions

**Next:** [Approval Gates](./02-approval-gates.md)

---

## Further reading

- [LangGraph — Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) — approve/reject and review/edit patterns
- [Google PAIR — Feedback + Control](https://pair.withgoogle.com/chapter/feedback-controls/) — designing for user control
- [LangGraph — Review Tool Calls](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/) — interrupts in tools

*[Back to Human-in-the-Loop overview](./00-human-in-the-loop.md)*

<!-- 
Sources Consulted:
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph how-to review tool calls: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/
- Google PAIR Feedback + Control: https://pair.withgoogle.com/chapter/feedback-controls/
-->
