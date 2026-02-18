---
title: "Human-in-the-Loop"
---

# Human-in-the-Loop

## Introduction

Not every tool call should execute automatically. Deleting a database, sending a payment, or publishing content — these actions have consequences that can't be undone. **Human-in-the-loop** (HITL) patterns insert a checkpoint between the model's tool call and its execution, giving the user a chance to review, approve, modify, or reject the action.

This is an application-level pattern. None of the providers have built-in confirmation gates — you build them into your agentic loop.

### What we'll cover

- Classifying tools by risk level
- Confirmation workflows with approve/reject/modify
- Preview-before-execute for data-transforming tools
- Building a `ConfirmationGate` for the agentic loop
- Multi-level approval with escalation

### Prerequisites

- [Multi-Turn Function Calling](../07-multi-turn-function-calling/00-multi-turn-function-calling.md) — The agentic loop
- [Nested Function Calling](./04-nested-function-calling.md) — Loop controls
- [Error Handling](../08-error-handling/00-error-handling.md) — Handling rejections gracefully

---

## Classifying tools by risk level

The first step is deciding which tools need confirmation. Not all actions are equal:

```python
from enum import Enum
from dataclasses import dataclass


class RiskLevel(Enum):
    LOW = "low"           # Read-only, no side effects
    MEDIUM = "medium"     # Writes data, but reversible
    HIGH = "high"         # Destructive or irreversible
    CRITICAL = "critical" # Financial, legal, or public-facing


@dataclass
class ToolPolicy:
    """Defines how a tool should be handled before execution."""
    name: str
    risk_level: RiskLevel
    requires_confirmation: bool
    requires_preview: bool = False
    max_auto_executions: int | None = None  # None = unlimited

# Define policies for each tool
TOOL_POLICIES = {
    "search_products": ToolPolicy(
        name="search_products",
        risk_level=RiskLevel.LOW,
        requires_confirmation=False
    ),
    "update_profile": ToolPolicy(
        name="update_profile",
        risk_level=RiskLevel.MEDIUM,
        requires_confirmation=False,
        requires_preview=True
    ),
    "delete_account": ToolPolicy(
        name="delete_account",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True
    ),
    "process_payment": ToolPolicy(
        name="process_payment",
        risk_level=RiskLevel.CRITICAL,
        requires_confirmation=True,
        requires_preview=True
    ),
    "send_email": ToolPolicy(
        name="send_email",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        requires_preview=True
    )
}
```

---

## The confirmation gate

A `ConfirmationGate` intercepts tool calls that require approval and routes them to the user:

```python
import json
from enum import Enum
from dataclasses import dataclass


class Decision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


@dataclass
class ConfirmationResult:
    decision: Decision
    modified_arguments: str | None = None  # Only set for MODIFY
    reason: str | None = None              # Optional user explanation


class ConfirmationGate:
    """Intercepts tool calls that require human approval."""
    
    def __init__(self, policies: dict[str, ToolPolicy]):
        self.policies = policies
        self._auto_execution_counts: dict[str, int] = {}
    
    def needs_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires human confirmation."""
        policy = self.policies.get(tool_name)
        if not policy:
            # Unknown tools always require confirmation
            return True
        
        if not policy.requires_confirmation:
            # Check auto-execution limits
            if policy.max_auto_executions is not None:
                count = self._auto_execution_counts.get(tool_name, 0)
                if count >= policy.max_auto_executions:
                    return True
            return False
        
        return True
    
    def needs_preview(self, tool_name: str) -> bool:
        """Check if a tool should show a preview before execution."""
        policy = self.policies.get(tool_name)
        return policy.requires_preview if policy else True
    
    def request_confirmation(
        self,
        tool_name: str,
        arguments: dict,
        preview: str | None = None
    ) -> ConfirmationResult:
        """Present the tool call to the user for approval."""
        policy = self.policies.get(tool_name)
        risk = policy.risk_level.value if policy else "unknown"
        
        print(f"\n{'='*50}")
        print(f"⚠️  CONFIRMATION REQUIRED")
        print(f"{'='*50}")
        print(f"Tool:      {tool_name}")
        print(f"Risk:      {risk.upper()}")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        if preview:
            print(f"\nPreview:")
            print(f"  {preview}")
        
        print(f"\nOptions: [a]pprove / [r]eject / [m]odify")
        
        choice = input("Your choice: ").strip().lower()
        
        if choice in ("a", "approve"):
            return ConfirmationResult(decision=Decision.APPROVE)
        elif choice in ("m", "modify"):
            new_args = input("Enter modified arguments (JSON): ")
            return ConfirmationResult(
                decision=Decision.MODIFY,
                modified_arguments=new_args
            )
        else:
            reason = input("Reason for rejection (optional): ")
            return ConfirmationResult(
                decision=Decision.REJECT,
                reason=reason or "User rejected the action"
            )
    
    def record_auto_execution(self, tool_name: str) -> None:
        """Track auto-executions for tools with limits."""
        self._auto_execution_counts[tool_name] = (
            self._auto_execution_counts.get(tool_name, 0) + 1
        )
```

**Output (when processing a payment):**
```
==================================================
⚠️  CONFIRMATION REQUIRED
==================================================
Tool:      process_payment
Risk:      CRITICAL
Arguments: {
  "amount": 149.99,
  "currency": "USD",
  "recipient": "vendor@example.com"
}

Preview:
  Payment of $149.99 USD to vendor@example.com

Options: [a]pprove / [r]eject / [m]odify
Your choice: a
```

---

## Integrating the gate into the agentic loop

The confirmation gate sits between parsing the model's tool calls and executing them:

```python
def run_with_confirmation(
    client,
    model: str,
    tools: list[dict],
    messages: list[dict],
    gate: ConfirmationGate,
    preview_handlers: dict = None
) -> str:
    """Agentic loop with human-in-the-loop confirmation."""
    input_messages = list(messages)
    preview_handlers = preview_handlers or {}
    
    while True:
        response = client.responses.create(
            model=model,
            input=input_messages,
            tools=tools
        )
        
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]
        
        if not tool_calls:
            return response.output_text
        
        input_messages += response.output
        
        for call in tool_calls:
            args = json.loads(call.arguments)
            
            if gate.needs_confirmation(call.name):
                # Generate preview if available
                preview = None
                if gate.needs_preview(call.name) and call.name in preview_handlers:
                    preview = preview_handlers[call.name](args)
                
                # Ask the user
                result = gate.request_confirmation(
                    call.name, args, preview
                )
                
                if result.decision == Decision.APPROVE:
                    output = execute_function(call.name, call.arguments)
                    
                elif result.decision == Decision.MODIFY:
                    output = execute_function(
                        call.name, result.modified_arguments
                    )
                    
                else:  # REJECT
                    output = {
                        "status": "rejected",
                        "reason": result.reason,
                        "message": "The user rejected this action."
                    }
            else:
                # Auto-execute low-risk tools
                output = execute_function(call.name, call.arguments)
                gate.record_auto_execution(call.name)
            
            input_messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(output)
            })
    
    return response.output_text
```

---

## Preview handlers

For tools that need a preview, we generate a human-readable summary of what will happen *before* the action runs:

```python
def preview_payment(args: dict) -> str:
    """Generate a human-readable preview for a payment."""
    amount = args.get("amount", 0)
    currency = args.get("currency", "USD")
    recipient = args.get("recipient", "unknown")
    return (
        f"💰 Payment of {amount:.2f} {currency} "
        f"to {recipient}\n"
        f"  This cannot be reversed once processed."
    )


def preview_email(args: dict) -> str:
    """Generate a preview for an email send."""
    to = args.get("to", "unknown")
    subject = args.get("subject", "(no subject)")
    body_preview = args.get("body", "")[:100]
    return (
        f"📧 Email to: {to}\n"
        f"  Subject: {subject}\n"
        f"  Body: {body_preview}..."
    )


def preview_delete(args: dict) -> str:
    """Generate a preview for a deletion."""
    target = args.get("id", args.get("name", "unknown"))
    return (
        f"🗑️ DELETE: {target}\n"
        f"  ⚠️ This action is permanent and cannot be undone."
    )


# Register previews
preview_handlers = {
    "process_payment": preview_payment,
    "send_email": preview_email,
    "delete_account": preview_delete,
}
```

---

## Multi-level approval

For critical actions, a single user confirmation may not be enough. Consider multi-level approval where different users authorize different risk levels:

```python
@dataclass
class Approver:
    name: str
    max_risk_level: RiskLevel
    can_approve: list[str]  # Tool names this approver can authorize


class MultiLevelGate:
    """Requires different approvers for different risk levels."""
    
    def __init__(
        self,
        policies: dict[str, ToolPolicy],
        approvers: list[Approver]
    ):
        self.policies = policies
        self.approvers = {a.name: a for a in approvers}
    
    def get_required_approver(self, tool_name: str) -> Approver | None:
        """Find the appropriate approver for a tool."""
        policy = self.policies.get(tool_name)
        if not policy or not policy.requires_confirmation:
            return None
        
        # Find an approver authorized for this tool and risk level
        for approver in self.approvers.values():
            if (tool_name in approver.can_approve and
                policy.risk_level.value <= approver.max_risk_level.value):
                return approver
        
        return None
    
    def request_approval(
        self,
        tool_name: str,
        arguments: dict,
        approver: Approver
    ) -> bool:
        """Request approval from a specific approver."""
        print(f"\n📋 Approval needed from: {approver.name}")
        print(f"   Action: {tool_name}")
        print(f"   Args: {json.dumps(arguments, indent=2)}")
        
        # In production, this sends a notification to the approver
        # and waits for their response asynchronously
        response = input(f"   {approver.name}, approve? [y/n]: ")
        return response.strip().lower() == "y"


# Define approvers with different authority levels
approvers = [
    Approver(
        name="Support Agent",
        max_risk_level=RiskLevel.MEDIUM,
        can_approve=["update_profile", "apply_discount"]
    ),
    Approver(
        name="Team Lead",
        max_risk_level=RiskLevel.HIGH,
        can_approve=["delete_account", "issue_refund", "send_email"]
    ),
    Approver(
        name="Finance Director",
        max_risk_level=RiskLevel.CRITICAL,
        can_approve=["process_payment", "wire_transfer"]
    )
]
```

---

## Handling rejections gracefully

When a user rejects a tool call, the model needs to know *why* and adapt. Return a structured rejection so the model can suggest alternatives:

```python
def handle_rejection(
    tool_name: str,
    arguments: dict,
    reason: str
) -> dict:
    """Create a structured rejection result for the model."""
    return {
        "status": "rejected_by_user",
        "tool": tool_name,
        "reason": reason,
        "guidance": (
            "The user has rejected this action. "
            "Acknowledge their decision, explain what you "
            "were trying to do, and offer alternatives if possible."
        )
    }

# The model receives this as the function result and can respond:
# "I understand you don't want to delete the account. 
#  Instead, I could deactivate it (which is reversible), 
#  or export your data first. Would either option work?"
```

> **🤖 AI Context:** When the model receives a rejection result, it naturally adapts — suggesting alternatives, asking for clarification, or proceeding with the remaining (approved) actions. You don't need special prompt engineering for this.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Default to confirmation for unknown tools | Safety net for tools added without a policy |
| Show previews for write operations | Users can't approve what they don't understand |
| Include a "modify" option alongside approve/reject | Users often want to adjust parameters, not cancel entirely |
| Return structured rejections to the model | The model can suggest alternatives instead of failing silently |
| Log all confirmations and rejections | Audit trail for compliance and debugging |
| Set auto-execution limits for medium-risk tools | Prevents unlimited write operations even for "safe" tools |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Confirming every single tool call | Only confirm medium+ risk — low-risk confirmations cause "alert fatigue" |
| No way to modify arguments | Users often want to adjust amounts, recipients, etc. |
| Silently blocking without telling the model | Return a result so the model can adapt its response |
| Hardcoding risk levels | Use configurable policies — risk varies by deployment context |
| No timeout on confirmation prompts | In async systems, set a timeout and reject on expiry |

---

## Hands-on exercise

### Your task

Build a confirmation system for a customer support chatbot that handles:
- `search_orders` (LOW) — auto-execute
- `update_shipping_address` (MEDIUM) — preview, auto-execute
- `issue_refund` (HIGH) — confirmation + preview
- `cancel_account` (CRITICAL) — confirmation + preview

### Requirements

1. Define `ToolPolicy` for each tool with appropriate risk levels
2. Create preview handlers for `issue_refund` and `cancel_account`
3. Build a `ConfirmationGate` that routes tools based on risk level
4. Simulate a conversation where the model calls all four tools

### Expected result

```
search_orders("ORD-123") → auto-executed ✅
update_shipping_address({...}) → previewed, auto-executed ✅
issue_refund({"order": "ORD-123", "amount": 49.99}) → 
  ⚠️ CONFIRMATION REQUIRED
  Preview: 💰 Refund of $49.99 for order ORD-123
  → User approves ✅
cancel_account({"user_id": "U-456"}) →
  ⚠️ CONFIRMATION REQUIRED
  Preview: 🗑️ Permanently cancel account U-456
  → User rejects ❌
  → Model suggests deactivation alternative
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `RiskLevel` enum and `ToolPolicy` dataclass from this lesson
- Preview handlers should return readable strings, not raw data
- For the simulation, you can hardcode user decisions instead of using `input()`
- Return a structured rejection dict so the "model" can respond to it

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

# Tool policies
policies = {
    "search_orders": ToolPolicy(
        name="search_orders",
        risk_level=RiskLevel.LOW,
        requires_confirmation=False
    ),
    "update_shipping_address": ToolPolicy(
        name="update_shipping_address",
        risk_level=RiskLevel.MEDIUM,
        requires_confirmation=False,
        requires_preview=True
    ),
    "issue_refund": ToolPolicy(
        name="issue_refund",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        requires_preview=True
    ),
    "cancel_account": ToolPolicy(
        name="cancel_account",
        risk_level=RiskLevel.CRITICAL,
        requires_confirmation=True,
        requires_preview=True
    )
}

# Preview handlers
def preview_refund(args: dict) -> str:
    order = args.get("order_id", "unknown")
    amount = args.get("amount", 0)
    return f"💰 Refund of ${amount:.2f} for order {order}"

def preview_cancel(args: dict) -> str:
    user_id = args.get("user_id", "unknown")
    return f"🗑️ Permanently cancel account {user_id}\n  ⚠️ This cannot be undone."

previews = {
    "issue_refund": preview_refund,
    "cancel_account": preview_cancel
}

# Simulate tool calls
simulated_calls = [
    ("search_orders", {"order_id": "ORD-123"}),
    ("update_shipping_address", {
        "order_id": "ORD-123", 
        "address": "456 New St"
    }),
    ("issue_refund", {"order_id": "ORD-123", "amount": 49.99}),
    ("cancel_account", {"user_id": "U-456"})
]

# Simulated user decisions
user_decisions = {
    "issue_refund": Decision.APPROVE,
    "cancel_account": Decision.REJECT
}

gate = ConfirmationGate(policies)

for tool_name, args in simulated_calls:
    if gate.needs_confirmation(tool_name):
        preview = None
        if gate.needs_preview(tool_name) and tool_name in previews:
            preview = previews[tool_name](args)
        
        decision = user_decisions.get(tool_name, Decision.REJECT)
        
        print(f"\n⚠️  CONFIRMATION: {tool_name}")
        if preview:
            print(f"  Preview: {preview}")
        print(f"  Decision: {decision.value}")
        
        if decision == Decision.REJECT:
            result = handle_rejection(
                tool_name, args, "User declined"
            )
            print(f"  → Rejected. Model should suggest alternatives.")
        else:
            print(f"  → Approved and executed ✅")
    else:
        if gate.needs_preview(tool_name) and tool_name in previews:
            print(f"📋 Preview: {previews[tool_name](args)}")
        print(f"✅ {tool_name}({json.dumps(args)}) → auto-executed")
        gate.record_auto_execution(tool_name)
```

</details>

### Bonus challenges

- [ ] Add an async confirmation flow using `asyncio.Event` for web-based approvals
- [ ] Implement a confirmation timeout that auto-rejects after 60 seconds
- [ ] Build a confirmation log that records all approvals/rejections with timestamps

---

## Summary

✅ **Risk classification** is the foundation — categorize every tool as LOW, MEDIUM, HIGH, or CRITICAL

✅ **Confirmation gates** intercept high-risk tool calls and route them to the user before execution

✅ **Preview handlers** generate readable summaries so users know exactly what they're approving

✅ The **modify option** lets users adjust parameters (amounts, recipients) without rejecting entirely

✅ **Structured rejections** returned to the model enable it to suggest alternatives gracefully

✅ This is an **application-level pattern** — all three providers support it through the agentic loop

**Next:** [Function Call Streaming →](./06-function-call-streaming.md)

---

[← Previous: Nested Function Calling](./04-nested-function-calling.md) | [Back to Lesson Overview](./00-advanced-patterns.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use Documentation: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
