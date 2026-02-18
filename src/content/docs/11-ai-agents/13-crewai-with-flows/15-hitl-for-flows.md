---
title: "HITL for Flows"
---

# HITL for Flows

## Introduction

Beyond the `@human_feedback` decorator (covered in [Lesson 09](./09-human-feedback-decorator.md)), CrewAI v1.8+ provides **global human-in-the-loop (HITL) configuration** for Flows. This includes `HumanFeedbackPending` state handling, streaming tool call events, `EventListener` for custom integrations, and `TraceCollectionListener` for comprehensive execution tracing.

These advanced patterns are essential for building production systems where human oversight, audit trails, and custom event handling are requirements.

### What We'll Cover

- Global HITL configuration for Flows
- `HumanFeedbackPending` handling
- EventListener for custom event processing
- TraceCollectionListener for execution tracing
- Streaming tool call events
- Building audit trails

### Prerequisites

- Completed [Human Feedback Decorator](./09-human-feedback-decorator.md)
- Completed [Streaming Flow Execution](./11-streaming-flow-execution.md)

---

## Global HITL Configuration

Instead of decorating individual methods with `@human_feedback`, you can configure HITL behavior at the Flow class level:

```python
from crewai.flow.flow import Flow, start, listen


class ReviewPipelineFlow(Flow):
    # Global HITL settings
    require_human_approval = True
    approval_timeout = 300  # seconds
    default_on_timeout = "approve"
    
    @start()
    def generate_content(self):
        return "Draft: AI agents are revolutionizing development..."
    
    @listen(generate_content)
    def review_step(self, content):
        # This step automatically requests human approval
        # based on global HITL configuration
        return content
    
    @listen(review_step)
    def publish(self, content):
        print(f"✅ Published: {content[:50]}...")
```

### Global vs Per-Method HITL

| Feature | `@human_feedback` (per-method) | Global HITL Config |
|---------|-------------------------------|-------------------|
| Scope | Individual methods | All methods or flow-wide |
| Configuration | Decorator parameters | Class-level attributes |
| Flexibility | Fine-grained control | Consistent behavior |
| Best for | Specific approval gates | Compliance-heavy workflows |

---

## HumanFeedbackPending

When a flow pauses for human input, it enters a `HumanFeedbackPending` state. This state can be detected, serialized, and resumed:

```python
from crewai.flow.flow import Flow, start, listen
from crewai.flow.human_feedback import human_feedback


class ApprovableFlow(Flow):
    
    @start()
    def prepare(self):
        self.state["document"] = "Quarterly report: Revenue up 15%"
        return self.state["document"]
    
    @human_feedback(
        message="Approve this report for distribution?",
        emit=["distribute", "hold"],
    )
    @listen(prepare)
    def approval_gate(self):
        return self.state["document"]
    
    @listen("distribute")
    def send_report(self):
        print("📤 Report distributed")
    
    @listen("hold")
    def hold_report(self):
        print("⏸️ Report held for revision")
```

### Detecting Pending State

```python
flow = ApprovableFlow()

# In a web application, you might:
# 1. Start the flow
# 2. Detect when it's pending
# 3. Show a UI to the user
# 4. Resume with their decision

try:
    result = flow.kickoff()
except HumanFeedbackPending as pending:
    # Flow is waiting for human input
    print(f"Waiting for approval: {pending.message}")
    print(f"Options: {pending.options}")
    
    # Later, when the human responds:
    flow.resume(decision="distribute", feedback="Looks good")
```

---

## EventListener

`EventListener` lets you hook into Flow execution events for logging, monitoring, and custom integrations:

```python
from crewai.flow.events import EventListener


class MonitoringListener(EventListener):
    """Custom listener for flow monitoring."""
    
    def on_flow_started(self, flow_id: str, state: dict):
        print(f"🚀 Flow {flow_id} started")
        # Send to monitoring system
    
    def on_method_started(self, flow_id: str, method_name: str):
        print(f"  ▶️ {method_name} started")
    
    def on_method_completed(self, flow_id: str, method_name: str, result: any):
        print(f"  ✅ {method_name} completed")
    
    def on_method_failed(self, flow_id: str, method_name: str, error: Exception):
        print(f"  ❌ {method_name} failed: {error}")
        # Alert on-call team
    
    def on_human_feedback_requested(self, flow_id: str, message: str, options: list):
        print(f"  🧑 Feedback requested: {message}")
        # Send Slack notification
    
    def on_flow_completed(self, flow_id: str, result: any):
        print(f"🏁 Flow {flow_id} completed")


# Attach listener to a flow
flow = ApprovableFlow()
flow.add_listener(MonitoringListener())
flow.kickoff()
```

### EventListener Methods

| Method | When Fired |
|--------|-----------|
| `on_flow_started` | Flow `kickoff()` begins |
| `on_method_started` | Any decorated method begins execution |
| `on_method_completed` | Any decorated method completes successfully |
| `on_method_failed` | Any decorated method throws an exception |
| `on_human_feedback_requested` | `@human_feedback` pauses for input |
| `on_human_feedback_received` | Human responds to feedback request |
| `on_flow_completed` | Flow finishes all execution |
| `on_flow_failed` | Flow terminates due to unhandled error |

---

## TraceCollectionListener

For comprehensive execution tracing, use `TraceCollectionListener`:

```python
from crewai.flow.events import TraceCollectionListener


class AuditTraceListener(TraceCollectionListener):
    """Collects execution traces for audit compliance."""
    
    def __init__(self):
        super().__init__()
        self.traces = []
    
    def on_trace(self, trace_event: dict):
        self.traces.append(trace_event)
        
        # Log to audit system
        event_type = trace_event.get("type")
        timestamp = trace_event.get("timestamp")
        details = trace_event.get("details", {})
        
        print(f"[{timestamp}] {event_type}: {details.get('method', 'flow')}")
    
    def get_audit_report(self) -> list[dict]:
        return self.traces


# Usage
audit = AuditTraceListener()
flow = ApprovableFlow()
flow.add_listener(audit)
flow.kickoff()

# After execution
print(f"\nAudit trail: {len(audit.traces)} events")
for trace in audit.traces:
    print(f"  {trace['type']}: {trace.get('details', {})}")
```

### Trace Event Structure

```python
{
    "type": "method_completed",
    "timestamp": "2025-01-15T10:30:00Z",
    "flow_id": "abc-123",
    "details": {
        "method": "prepare",
        "duration_ms": 150,
        "result_size": 45,
    }
}
```

---

## Streaming Tool Call Events

When agents use tools during flow execution, you can stream tool call events:

```python
class ToolMonitorListener(EventListener):
    """Monitor tool usage within crews."""
    
    def on_tool_call_started(self, flow_id: str, agent: str, tool: str, input_data: dict):
        print(f"  🔧 {agent} → {tool}({input_data})")
    
    def on_tool_call_completed(self, flow_id: str, agent: str, tool: str, result: str):
        print(f"  ✅ {tool} returned: {result[:100]}...")
    
    def on_tool_call_failed(self, flow_id: str, agent: str, tool: str, error: str):
        print(f"  ❌ {tool} failed: {error}")
```

This gives visibility into:
- Which tools agents are calling
- What inputs they're providing
- What results they're getting
- How long tool calls take

---

## Building Production Audit Trails

Combine listeners for comprehensive production monitoring:

```python
import json
from datetime import datetime


class ProductionListener(EventListener):
    """Production-grade event listener with JSON logging."""
    
    def __init__(self, log_file: str = "flow_audit.jsonl"):
        self.log_file = log_file
    
    def _log(self, event: dict):
        event["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def on_flow_started(self, flow_id: str, state: dict):
        self._log({"event": "flow_started", "flow_id": flow_id, "state_keys": list(state.keys())})
    
    def on_method_completed(self, flow_id: str, method_name: str, result: any):
        self._log({
            "event": "method_completed",
            "flow_id": flow_id,
            "method": method_name,
            "result_type": type(result).__name__,
        })
    
    def on_human_feedback_received(self, flow_id: str, decision: str, feedback: str):
        self._log({
            "event": "human_decision",
            "flow_id": flow_id,
            "decision": decision,
            "feedback": feedback,
        })
    
    def on_flow_completed(self, flow_id: str, result: any):
        self._log({"event": "flow_completed", "flow_id": flow_id})


# Attach to flow
listener = ProductionListener(log_file="audit_trail.jsonl")
flow = ApprovableFlow()
flow.add_listener(listener)
flow.kickoff()
```

The resulting `audit_trail.jsonl` provides a complete record of every decision, execution step, and human interaction — essential for compliance and debugging.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `EventListener` for monitoring, not control flow | Listeners should observe, not modify execution |
| Log to structured formats (JSON, JSONL) | Enables automated analysis and alerting |
| Add `TraceCollectionListener` for regulated industries | Full audit trails for compliance |
| Handle `HumanFeedbackPending` in web apps | Serialize pending state and resume later |
| Combine multiple listeners | Monitoring + auditing + alerting simultaneously |
| Keep listener callbacks fast | Slow listeners delay flow execution |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Putting business logic in listeners | Listeners are for observation, not control flow |
| Blocking listeners with slow I/O | Use async logging or background queues |
| Not handling `HumanFeedbackPending` | Catch it explicitly in web applications |
| Missing listener methods | Implement all relevant event handlers |
| Logging sensitive data in traces | Filter PII and secrets before logging |
| No timeout on human feedback | Set `default_on_timeout` or `default_outcome` |

---

## Hands-on Exercise

### Your Task

Build a Flow with an EventListener that logs all execution events.

### Requirements

1. Create a `LoggingListener(EventListener)` that records events to a list
2. Create a simple 3-step Flow
3. Attach the listener and run the flow
4. After execution, print the complete event log

### Expected Result

```
Event Log:
  1. flow_started — ID: abc-123
  2. method_started — step_one
  3. method_completed — step_one
  4. method_started — step_two
  5. method_completed — step_two
  6. method_started — step_three
  7. method_completed — step_three
  8. flow_completed — ID: abc-123
```

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from crewai.flow.flow import Flow, start, listen
from crewai.flow.events import EventListener


class LoggingListener(EventListener):
    def __init__(self):
        self.events = []
    
    def on_flow_started(self, flow_id, state):
        self.events.append(f"flow_started — ID: {flow_id[:8]}")
    
    def on_method_started(self, flow_id, method_name):
        self.events.append(f"method_started — {method_name}")
    
    def on_method_completed(self, flow_id, method_name, result):
        self.events.append(f"method_completed — {method_name}")
    
    def on_flow_completed(self, flow_id, result):
        self.events.append(f"flow_completed — ID: {flow_id[:8]}")


class SimpleFlow(Flow):
    @start()
    def step_one(self):
        return "Step 1 done"
    
    @listen(step_one)
    def step_two(self, data):
        return "Step 2 done"
    
    @listen(step_two)
    def step_three(self, data):
        return "Step 3 done"


logger = LoggingListener()
flow = SimpleFlow()
flow.add_listener(logger)
flow.kickoff()

print("\nEvent Log:")
for i, event in enumerate(logger.events, 1):
    print(f"  {i}. {event}")
```

</details>

---

## Summary

✅ Global HITL configuration applies human approval requirements across entire Flows

✅ `HumanFeedbackPending` can be caught and handled in web applications for async approval UIs

✅ `EventListener` hooks into every flow event — starts, completions, failures, and feedback

✅ `TraceCollectionListener` provides comprehensive execution traces for audit compliance

✅ Streaming tool call events give visibility into agent tool usage within Flows

**Next:** [AutoGen AgentChat](../14-autogen-agentchat/00-autogen-agentchat.md)

---

## Further Reading

- [CrewAI Flows Documentation](https://docs.crewai.com/concepts/flows) — Advanced flow features
- [CrewAI Enterprise](https://www.crewai.com/enterprise) — Managed monitoring and observability

*Back to [CrewAI with Flows Overview](./00-crewai-with-flows.md)*

<!-- 
Sources Consulted:
- CrewAI Flows: https://docs.crewai.com/concepts/flows
- CrewAI Crews: https://docs.crewai.com/concepts/crews
-->
