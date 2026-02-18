---
title: "Thought/action logging"
---

# Thought/action logging

## Introduction

Step-through debugging works when we can reproduce a bug locally. But agents in production fail in ways we can't reproduce on demand — the LLM makes a different decision, the context window has shifted, or a rare tool chain triggers a subtle error. For these cases, we need comprehensive logging that captures the agent's entire reasoning process.

Thought/action logging records every decision the agent makes: what it thought, which tool it chose, what arguments it generated, what the tool returned, and how it incorporated that result. When something goes wrong, we reconstruct the exact chain of events from logs instead of guessing.

### What we'll cover

- Configuring Python's `logging` module for agent workflows
- Using `capture_run_messages()` to record full conversations
- Building structured JSON logs for agent actions
- Log level strategies for development vs. production
- Storing and searching agent logs effectively

### Prerequisites

- Python `logging` module basics
- Agent fundamentals (Lessons 1-5)
- Understanding of tools and tool registration
- JSON and structured data concepts

---

## Setting up agent-specific logging

The Python standard library `logging` module is the foundation. We create dedicated loggers for different aspects of agent execution.

### Logger hierarchy for agents

```python
import logging

# Create a logger hierarchy for agent components
agent_logger = logging.getLogger("agent")
tool_logger = logging.getLogger("agent.tools")
llm_logger = logging.getLogger("agent.llm")
memory_logger = logging.getLogger("agent.memory")

# Configure the root agent logger
agent_logger.setLevel(logging.DEBUG)

# Console handler with readable format for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
console_handler.setFormatter(console_format)
agent_logger.addHandler(console_handler)
```

**Output:**

```
14:23:01 [agent.tools] INFO: Tool 'search_flights' called with: origin='NYC', destination='Tokyo'
14:23:01 [agent.tools] INFO: Tool 'search_flights' returned: 3 results
14:23:02 [agent.llm] INFO: LLM generation completed in 1.2s, 245 tokens
14:23:02 [agent.tools] INFO: Tool 'book_flight' called with: flight_id='FL-123'
```

### Adding logging to tool functions

Wrap every tool function with logging to capture its inputs and outputs:

```python
import json
import time
import logging
from functools import wraps
from pydantic_ai import Agent, RunContext

tool_logger = logging.getLogger("agent.tools")

def log_tool_call(func):
    """Decorator that logs tool function inputs and outputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract arguments (skip RunContext)
        log_args = {k: v for k, v in kwargs.items()}
        tool_logger.info(
            f"Tool '{func.__name__}' called with: {json.dumps(log_args, default=str)}"
        )
        
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            
            # Truncate long results for readability
            result_preview = str(result)[:200]
            tool_logger.info(
                f"Tool '{func.__name__}' returned in {duration:.3f}s: {result_preview}"
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            tool_logger.error(
                f"Tool '{func.__name__}' FAILED in {duration:.3f}s: {type(e).__name__}: {e}"
            )
            raise
    
    return wrapper

agent = Agent("openai:gpt-4o", instructions="You help with research.")

@agent.tool
@log_tool_call
def search_web(ctx: RunContext[None], query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return f"Found {max_results} results for '{query}'"

@agent.tool
@log_tool_call
def summarize(ctx: RunContext[None], text: str, max_words: int = 100) -> str:
    """Summarize a piece of text."""
    words = text.split()[:max_words]
    return " ".join(words) + "..."
```

**Output:**

```
14:23:01 [agent.tools] INFO: Tool 'search_web' called with: {"query": "AI trends 2026", "max_results": 5}
14:23:01 [agent.tools] INFO: Tool 'search_web' returned in 0.002s: Found 5 results for 'AI trends 2026'
14:23:02 [agent.tools] INFO: Tool 'summarize' called with: {"text": "AI continues to...", "max_words": 100}
14:23:02 [agent.tools] INFO: Tool 'summarize' returned in 0.001s: AI continues to...
```

---

## Capturing full conversations with Pydantic AI

Pydantic AI provides `capture_run_messages()` — a context manager that records every message in the agent's conversation. This is the most powerful debugging tool for understanding why an agent made specific decisions.

### Basic message capture

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent(
    "openai:gpt-4o",
    instructions="You are a customer service agent. Always verify the order ID before making changes.",
)

@agent.tool
def lookup_order(ctx, order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: 2x Widget Pro, status: shipped, ETA: Feb 20"

@agent.tool
def cancel_order(ctx, order_id: str, reason: str) -> str:
    """Cancel an order."""
    return f"Order {order_id} cancelled. Reason: {reason}"

# Capture all messages during the run
with capture_run_messages() as messages:
    result = agent.run_sync(
        "I need to cancel order ORD-5521, I found a better price elsewhere"
    )

# Log the complete conversation
import json
for i, msg in enumerate(messages):
    print(f"\n{'='*60}")
    print(f"Message {i}: {msg.__class__.__name__}")
    for part in msg.parts:
        kind = part.part_kind
        if kind == "user-prompt":
            print(f"  [USER] {part.content}")
        elif kind == "tool-call":
            print(f"  [TOOL CALL] {part.tool_name}({json.dumps(part.args)})")
        elif kind == "tool-return":
            print(f"  [TOOL RESULT] {part.tool_name} → {part.content}")
        elif kind == "text":
            print(f"  [ASSISTANT] {part.content}")
```

**Output:**

```
============================================================
Message 0: ModelRequest
  [USER] I need to cancel order ORD-5521, I found a better price elsewhere

============================================================
Message 1: ModelResponse
  [TOOL CALL] lookup_order({"order_id": "ORD-5521"})

============================================================
Message 2: ModelRequest
  [TOOL RESULT] lookup_order → Order ORD-5521: 2x Widget Pro, status: shipped, ETA: Feb 20

============================================================
Message 3: ModelResponse
  [TOOL CALL] cancel_order({"order_id": "ORD-5521", "reason": "Customer found a better price"})

============================================================
Message 4: ModelRequest
  [TOOL RESULT] cancel_order → Order ORD-5521 cancelled. Reason: Customer found a better price

============================================================
Message 5: ModelResponse
  [ASSISTANT] I've cancelled order ORD-5521 for you. The reason has been recorded...
```

> **🤖 AI Context:** Message capture reveals whether the agent followed its instructions. In this case, we can verify the agent looked up the order before cancelling it — following the "always verify first" instruction.

### Logging message capture to a file

For persistent debugging, write captured messages to a structured log file:

```python
import json
import logging
from datetime import datetime, timezone
from pydantic_ai import Agent, capture_run_messages

# Set up file logging for agent conversations
conv_logger = logging.getLogger("agent.conversation")
file_handler = logging.FileHandler("agent_conversations.jsonl")
file_handler.setFormatter(logging.Formatter("%(message)s"))
conv_logger.addHandler(file_handler)
conv_logger.setLevel(logging.INFO)

def log_conversation(messages, run_input: str, run_output: str):
    """Log a complete agent conversation as structured JSON."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": run_input,
        "output": run_output,
        "turn_count": len(messages),
        "tools_called": [],
        "messages": [],
    }
    
    for msg in messages:
        for part in msg.parts:
            kind = part.part_kind
            if kind == "tool-call":
                record["tools_called"].append(part.tool_name)
                record["messages"].append({
                    "type": "tool_call",
                    "tool": part.tool_name,
                    "args": part.args,
                })
            elif kind == "tool-return":
                record["messages"].append({
                    "type": "tool_return",
                    "tool": part.tool_name,
                    "result": part.content[:500],  # Truncate long results
                })
            elif kind == "user-prompt":
                record["messages"].append({
                    "type": "user",
                    "content": part.content,
                })
            elif kind == "text":
                record["messages"].append({
                    "type": "assistant",
                    "content": part.content,
                })
    
    conv_logger.info(json.dumps(record))

# Usage
agent = Agent("openai:gpt-4o", instructions="Help users with orders.")
user_input = "Cancel order ORD-5521"

with capture_run_messages() as messages:
    result = agent.run_sync(user_input)

log_conversation(messages, user_input, result.output)
```

**Resulting JSONL file (one JSON object per line):**

```json
{"timestamp": "2026-02-17T14:23:01Z", "input": "Cancel order ORD-5521", "output": "Order cancelled.", "turn_count": 6, "tools_called": ["lookup_order", "cancel_order"], "messages": [{"type": "user", "content": "Cancel order ORD-5521"}, {"type": "tool_call", "tool": "lookup_order", "args": {"order_id": "ORD-5521"}}, ...]}
```

---

## Structured log formats

Plain text logs are human-readable but hard to search and analyze at scale. Structured JSON logging makes agent logs queryable and parseable.

### JSON structured logging setup

```python
import json
import logging
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for machine parsing."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include extra fields if provided
        if hasattr(record, "agent_name"):
            log_entry["agent_name"] = record.agent_name
        if hasattr(record, "tool_name"):
            log_entry["tool_name"] = record.tool_name
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "token_count"):
            log_entry["token_count"] = record.token_count
        
        # Include exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_entry)

# Configure JSON logging
logger = logging.getLogger("agent")
json_handler = logging.FileHandler("agent_debug.jsonl")
json_handler.setFormatter(JSONFormatter())
logger.addHandler(json_handler)
logger.setLevel(logging.DEBUG)

# Usage with extra fields
logger.info(
    "Tool executed successfully",
    extra={
        "agent_name": "OrderBot",
        "tool_name": "cancel_order",
        "duration_ms": 45,
    }
)
```

**Output in `agent_debug.jsonl`:**

```json
{"timestamp": "2026-02-17T14:23:01Z", "level": "INFO", "logger": "agent", "message": "Tool executed successfully", "agent_name": "OrderBot", "tool_name": "cancel_order", "duration_ms": 45}
```

### Agent action logging class

For consistent structured logging across all agent operations, create a dedicated action logger:

```python
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

@dataclass
class AgentAction:
    """Represents a single agent action for logging."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action_type: str = ""         # "llm_call", "tool_call", "tool_result", "error", "decision"
    agent_name: str = ""
    details: dict = field(default_factory=dict)
    duration_ms: float = 0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

class AgentActionLogger:
    """Structured logger for agent actions."""
    
    def __init__(self, agent_name: str, log_file: str = "agent_actions.jsonl"):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agent.actions.{agent_name}")
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.actions: list[AgentAction] = []
    
    def log_tool_call(self, tool_name: str, arguments: dict, duration_ms: float = 0):
        action = AgentAction(
            action_type="tool_call",
            agent_name=self.agent_name,
            details={"tool": tool_name, "arguments": arguments},
            duration_ms=duration_ms,
        )
        self.actions.append(action)
        self.logger.info(action.to_json())
    
    def log_tool_result(self, tool_name: str, result: Any, success: bool = True):
        action = AgentAction(
            action_type="tool_result",
            agent_name=self.agent_name,
            details={
                "tool": tool_name,
                "result": str(result)[:500],
                "success": success,
            },
        )
        self.actions.append(action)
        self.logger.info(action.to_json())
    
    def log_llm_call(self, model: str, input_tokens: int, output_tokens: int, duration_ms: float):
        action = AgentAction(
            action_type="llm_call",
            agent_name=self.agent_name,
            details={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            duration_ms=duration_ms,
        )
        self.actions.append(action)
        self.logger.info(action.to_json())
    
    def log_decision(self, decision: str, reasoning: str = ""):
        action = AgentAction(
            action_type="decision",
            agent_name=self.agent_name,
            details={"decision": decision, "reasoning": reasoning},
        )
        self.actions.append(action)
        self.logger.info(action.to_json())
    
    def log_error(self, error: Exception, context: str = ""):
        action = AgentAction(
            action_type="error",
            agent_name=self.agent_name,
            details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            },
        )
        self.actions.append(action)
        self.logger.error(action.to_json())
    
    def get_summary(self) -> dict:
        """Get a summary of all logged actions."""
        return {
            "agent": self.agent_name,
            "total_actions": len(self.actions),
            "tool_calls": sum(1 for a in self.actions if a.action_type == "tool_call"),
            "errors": sum(1 for a in self.actions if a.action_type == "error"),
            "total_duration_ms": sum(a.duration_ms for a in self.actions),
        }

# Usage
action_logger = AgentActionLogger("OrderBot")
action_logger.log_tool_call("search_orders", {"customer_id": "C-123"}, duration_ms=45)
action_logger.log_tool_result("search_orders", "Found 3 orders", success=True)
action_logger.log_decision("Cancel order", reasoning="Customer requested cancellation")
print(action_logger.get_summary())
```

**Output:**

```python
{'agent': 'OrderBot', 'total_actions': 3, 'tool_calls': 1, 'errors': 0, 'total_duration_ms': 45.0}
```

---

## Log level strategies

Different environments need different levels of detail. Too much logging slows down production; too little makes debugging impossible.

### Log levels for agent debugging

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Development only — full message content, all state | `DEBUG: LLM prompt: "You are a travel agent..."` |
| `INFO` | Normal operations — tool calls, decisions, timing | `INFO: Tool 'search' called, 3 results in 0.4s` |
| `WARNING` | Unusual but handled situations | `WARNING: Tool 'search' returned empty results, retrying` |
| `ERROR` | Failures that stop the current operation | `ERROR: Tool 'book_flight' raised ValueError: invalid ID` |
| `CRITICAL` | Failures that stop the entire agent | `CRITICAL: LLM API connection failed after 3 retries` |

### Environment-specific configuration

```python
import os
import logging

def configure_agent_logging():
    """Configure logging based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    logger = logging.getLogger("agent")
    
    if env == "development":
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(handler)
        
    elif env == "staging":
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("agent_staging.jsonl")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    elif env == "production":
        logger.setLevel(logging.WARNING)
        handler = logging.FileHandler("agent_production.jsonl")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        # Also log errors to stderr for alerting
        error_handler = logging.StreamHandler()
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    
    return logger
```

> **Warning:** Never log full user prompts or LLM responses at `INFO` level in production. They may contain sensitive data. Use `DEBUG` level for content, `INFO` for metadata (tool names, timing, token counts).

### Dynamic log level adjustment

Sometimes you need to increase logging detail for a specific agent run without redeploying:

```python
import logging
from contextlib import contextmanager

@contextmanager
def debug_logging(logger_name: str = "agent"):
    """Temporarily enable DEBUG logging for an agent run."""
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    # Add a temporary console handler
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        "🔍 %(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(debug_handler)
    
    try:
        yield logger
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(debug_handler)

# Usage: temporarily enable debug for a suspicious request
with debug_logging("agent"):
    result = agent.run_sync("This request seems to cause issues")
```

---

## Logging with OpenAI Agents SDK tracing

The OpenAI Agents SDK has built-in tracing that automatically logs all agent operations. We can tap into this system with custom trace processors:

```python
from agents import Agent, Runner, trace, custom_span
from agents.tracing import TracingProcessor

class DebugTracingProcessor(TracingProcessor):
    """Custom processor that logs all trace events for debugging."""
    
    def on_trace_start(self, trace):
        print(f"\n{'='*60}")
        print(f"🚀 Trace started: {trace.workflow_name}")
        print(f"   Trace ID: {trace.trace_id}")
    
    def on_trace_end(self, trace):
        print(f"✅ Trace ended: {trace.workflow_name}")
        print(f"{'='*60}\n")
    
    def on_span_start(self, span):
        span_type = type(span.span_data).__name__
        print(f"  ▶ Span started: {span_type}")
    
    def on_span_end(self, span):
        span_type = type(span.span_data).__name__
        duration = (span.ended_at - span.started_at) if span.ended_at else 0
        print(f"  ◀ Span ended: {span_type} ({duration:.0f}ms)")

# Register the custom processor
from agents.tracing import add_trace_processor
add_trace_processor(DebugTracingProcessor())

# Now all agent runs will be logged
agent = Agent(name="Debug Agent", instructions="Be helpful.")

async def main():
    with trace("Debug Session"):
        result = await Runner.run(agent, "What is 2+2?")
        print(f"Result: {result.final_output}")
```

**Output:**

```
============================================================
🚀 Trace started: Debug Session
   Trace ID: trace_abc123def456
  ▶ Span started: AgentSpanData
  ▶ Span started: GenerationSpanData
  ◀ Span ended: GenerationSpanData (1234ms)
  ◀ Span ended: AgentSpanData (1240ms)
✅ Trace ended: Debug Session
============================================================
Result: 2 + 2 = 4
```

---

## Searching and analyzing logs

Structured logs are only useful if we can search them efficiently.

### Searching JSONL agent logs

```python
import json
from datetime import datetime, timezone

def search_agent_logs(
    log_file: str,
    tool_name: str | None = None,
    action_type: str | None = None,
    errors_only: bool = False,
    since: str | None = None,
) -> list[dict]:
    """Search structured agent logs with filters."""
    results = []
    
    with open(log_file) as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            # Apply filters
            if errors_only and record.get("action_type") != "error":
                continue
            if tool_name and record.get("details", {}).get("tool") != tool_name:
                continue
            if action_type and record.get("action_type") != action_type:
                continue
            if since:
                record_time = record.get("timestamp", "")
                if record_time < since:
                    continue
            
            results.append(record)
    
    return results

# Find all errors from the last hour
errors = search_agent_logs(
    "agent_actions.jsonl",
    errors_only=True,
    since="2026-02-17T13:00:00Z",
)
print(f"Found {len(errors)} errors")

# Find all calls to a specific tool
search_calls = search_agent_logs(
    "agent_actions.jsonl",
    tool_name="search_orders",
    action_type="tool_call",
)
print(f"search_orders was called {len(search_calls)} times")
```

**Output:**

```
Found 2 errors
search_orders was called 15 times
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use structured JSON logging in production | Enables search, filtering, and automated analysis |
| Log tool inputs and outputs at every call | This is the primary debugging data for agent issues |
| Truncate long content in logs | LLM responses can be thousands of tokens — log a preview |
| Never log API keys or credentials | Even in debug mode, mask sensitive values |
| Use `capture_run_messages()` during development | The full conversation is the best debugging artifact |
| Separate log files by agent or environment | Keeps log volumes manageable and searchable |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Logging full prompt/response content in production | Log metadata only (tool names, timing, token counts) at INFO level |
| Not truncating tool results in logs | Truncate to 200-500 chars with `str(result)[:500]` |
| Using `print()` instead of `logging` | Use `logging` — it supports levels, handlers, and structured output |
| Forgetting to log errors with stack traces | Use `logger.error("msg", exc_info=True)` to include the traceback |
| Logging every LLM token during streaming | Log the complete response after streaming finishes, not token-by-token |
| Not including timestamps in logs | Always include ISO timestamps — essential for correlating events |

---

## Hands-on exercise

### Your task

Build a comprehensive logging system for a customer service agent that handles order lookups, cancellations, and refunds. The system should produce searchable JSONL logs that capture every decision and tool call.

### Requirements

1. Create an `AgentActionLogger` with methods for tool calls, results, decisions, and errors
2. Wrap three tool functions (`lookup_order`, `cancel_order`, `process_refund`) with logging
3. Use `capture_run_messages()` to log the complete conversation
4. Write a search function that can find errors and filter by tool name
5. Run the agent with a test prompt and verify the logs contain all expected entries

### Expected result

After running the agent, the JSONL log file should contain structured entries for every tool call, result, and the final decision. The search function should be able to filter by tool name and find only error entries.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use the `log_tool_call` decorator pattern from this lesson to wrap all three tools
- The JSONL format writes one JSON object per line — easy to search with Python or `grep`
- Remember to log both successful and failed tool executions
- Include a `run_id` field to correlate all entries from a single agent run

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from functools import wraps
from pydantic_ai import Agent, RunContext, capture_run_messages

# --- Logging Setup ---

class AgentLogger:
    def __init__(self, agent_name: str, log_file: str = "agent_debug.jsonl"):
        self.agent_name = agent_name
        self.run_id = str(uuid.uuid4())[:8]
        self.logger = logging.getLogger(f"agent.{agent_name}")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _log(self, action_type: str, details: dict):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": self.agent_name,
            "action": action_type,
            **details,
        }
        self.logger.info(json.dumps(record, default=str))
    
    def tool_call(self, name: str, args: dict, duration_ms: float):
        self._log("tool_call", {"tool": name, "args": args, "duration_ms": duration_ms})
    
    def tool_result(self, name: str, result: str, success: bool):
        self._log("tool_result", {"tool": name, "result": result[:500], "success": success})
    
    def error(self, error: Exception, context: str):
        self._log("error", {"error": type(error).__name__, "message": str(error), "context": context})

logger = AgentLogger("CustomerService")

# --- Decorated Tools ---

def logged_tool(log: AgentLogger):
    def decorator(func):
        @wraps(func)
        def wrapper(ctx, **kwargs):
            start = time.perf_counter()
            try:
                result = func(ctx, **kwargs)
                ms = (time.perf_counter() - start) * 1000
                log.tool_call(func.__name__, kwargs, ms)
                log.tool_result(func.__name__, str(result), True)
                return result
            except Exception as e:
                ms = (time.perf_counter() - start) * 1000
                log.tool_call(func.__name__, kwargs, ms)
                log.error(e, f"Tool {func.__name__}")
                raise
        return wrapper
    return decorator

agent = Agent("openai:gpt-4o", instructions="You handle customer orders.")

@agent.tool
@logged_tool(logger)
def lookup_order(ctx: RunContext[None], order_id: str) -> str:
    return f"Order {order_id}: 2x Widget, shipped, ETA Feb 20"

@agent.tool
@logged_tool(logger)
def cancel_order(ctx: RunContext[None], order_id: str, reason: str) -> str:
    return f"Order {order_id} cancelled. Reason: {reason}"

@agent.tool
@logged_tool(logger)
def process_refund(ctx: RunContext[None], order_id: str, amount: float) -> str:
    if amount <= 0:
        raise ValueError(f"Invalid refund amount: {amount}")
    return f"Refund of ${amount:.2f} processed for order {order_id}"

# --- Run & Inspect ---

with capture_run_messages() as messages:
    result = agent.run_sync("Cancel order ORD-5521 and refund $29.99")

# Log conversation summary
for msg in messages:
    for part in msg.parts:
        print(f"  {part.part_kind}: {getattr(part, 'content', getattr(part, 'tool_name', ''))}")

# Search logs
def search_logs(log_file: str, tool_name: str | None = None):
    with open(log_file) as f:
        for line in f:
            record = json.loads(line)
            if tool_name and record.get("tool") != tool_name:
                continue
            print(json.dumps(record, indent=2))

search_logs("agent_debug.jsonl", tool_name="cancel_order")
```

</details>

### Bonus challenges

- [ ] Add a `run_id` field that correlates all log entries from a single `agent.run_sync()` call
- [ ] Create a log analyzer that computes average tool call duration and error rates
- [ ] Implement log rotation using Python's `RotatingFileHandler` to prevent unbounded file growth

---

## Summary

✅ Python's `logging` module with JSON formatting creates searchable, structured agent logs that work in development and production

✅ `capture_run_messages()` in Pydantic AI records the complete conversation — every user prompt, LLM decision, tool call, and result — making it the most valuable debugging artifact

✅ A `log_tool_call` decorator wrapping every tool function ensures consistent logging of inputs, outputs, duration, and errors without cluttering tool logic

✅ Log levels should match the environment: `DEBUG` in development (full content), `INFO` in staging (metadata), `WARNING+` in production (anomalies and errors only)

✅ OpenAI Agents SDK's custom `TracingProcessor` lets you tap into the built-in tracing system to log all spans and traces to any destination

---

**Next:** [Visualization of Agent Decisions](./03-visualization-agent-decisions.md)

**Previous:** [Step-Through Debugging](./01-step-through-debugging.md)

---

## Further Reading

- [Python logging Documentation](https://docs.python.org/3/library/logging.html) - Standard library logging
- [Python logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html) - Advanced logging patterns
- [Pydantic AI Testing](https://ai.pydantic.dev/testing/) - capture_run_messages and message inspection
- [OpenAI Agents SDK Tracing](https://openai.github.io/openai-agents-python/tracing/) - Custom trace processors
- [Pydantic Logfire](https://ai.pydantic.dev/logfire/) - Structured observability for Pydantic AI

<!-- 
Sources Consulted:
- Python logging docs: https://docs.python.org/3/library/logging.html
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- OpenAI Agents SDK Tracing: https://openai.github.io/openai-agents-python/tracing/
- Pydantic Logfire docs: https://ai.pydantic.dev/logfire/
- Python logging cookbook: https://docs.python.org/3/howto/logging-cookbook.html
-->
