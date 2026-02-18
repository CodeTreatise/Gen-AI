---
title: "Breakpoints in agent loops"
---

# Breakpoints in agent loops

## Introduction

Agents execute in loops. The core pattern — think, act, observe, repeat — means the same code path runs multiple times with different state. A traditional breakpoint fires on every iteration, forcing us to step through five correct cycles just to reach the sixth one where the bug occurs.

Agent-aware breakpoints solve this. We use conditional breakpoints that trigger only when specific tools are called, when the iteration count exceeds a threshold, or when the agent state matches a particular condition. We also build programmatic breakpoints directly into the agent execution loop — pausing execution on specific tool calls, state changes, or exceptions.

### What we'll cover

- Conditional breakpoints that trigger on specific agent actions
- Tool call interception for debugging specific tool invocations
- State change triggers that pause when agent state meets a condition
- Exception breakpoints for catching and inspecting errors in context
- Iteration-aware debugging for multi-step agent loops

### Prerequisites

- Step-through debugging (Lesson 22-01)
- Agent execution loop concepts (Lesson 6)
- Python `pdb` and VS Code debugger basics
- Understanding of the ReAct loop pattern

---

## Conditional breakpoints in agent code

A conditional breakpoint pauses execution only when a specified Python expression evaluates to `True`. This is essential for agent debugging because agent loops execute the same code path many times.

### Programmatic conditional breakpoints

Python's `breakpoint()` doesn't support conditions directly. We build conditional breakpoints manually:

```python
from pydantic_ai import Agent, RunContext

agent = Agent("openai:gpt-4o", instructions="You are a research assistant.")

iteration_count = 0

@agent.tool
def search(ctx: RunContext[None], query: str) -> str:
    """Search for information."""
    global iteration_count
    iteration_count += 1
    
    # Conditional breakpoint: only pause after 3rd tool call
    if iteration_count >= 3:
        breakpoint()  # Only triggers on iteration 3+
    
    return f"Results for '{query}': Article 1, Article 2"

@agent.tool
def analyze(ctx: RunContext[None], data: str) -> str:
    """Analyze data."""
    # Conditional breakpoint: only pause for suspicious input
    if len(data) > 500:
        breakpoint()  # Trigger on unexpectedly long input
    
    return f"Analysis of: {data[:100]}..."
```

### Tool-name conditional breakpoints

The most common agent debugging pattern is pausing only when a specific tool is called:

```python
import functools
from pydantic_ai import Agent, RunContext

def debug_on_tool(target_tool: str):
    """Decorator: pause execution when a specific tool is called."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if func.__name__ == target_tool:
                print(f"\n🔴 Breakpoint: Tool '{target_tool}' called")
                print(f"   Arguments: {kwargs}")
                breakpoint()
            return func(*args, **kwargs)
        return wrapper
    return decorator

agent = Agent("openai:gpt-4o", instructions="Help with data analysis.")

@agent.tool
@debug_on_tool("execute_query")  # Only break on this tool
def execute_query(ctx: RunContext[None], sql: str) -> str:
    """Execute a SQL query."""
    # This breakpoint only fires for execute_query, not for other tools
    return f"Query result: 42 rows"

@agent.tool
def format_report(ctx: RunContext[None], data: str, format: str = "table") -> str:
    """Format data into a report."""
    # No breakpoint here — this tool works fine
    return f"Formatted report ({format}): {data[:50]}"
```

**Output:**

```
🔴 Breakpoint: Tool 'execute_query' called
   Arguments: {'sql': 'SELECT * FROM products WHERE price < 50'}
> /path/to/script.py(15)wrapper()
-> return func(*args, **kwargs)
(Pdb) p kwargs['sql']
'SELECT * FROM products WHERE price < 50'
(Pdb) c
```

---

## Tool call interception

Sometimes we need more than a simple breakpoint — we want to intercept a tool call, inspect it, optionally modify it, and then decide whether to proceed. This is a "debugging middleware" pattern.

### Building a tool call interceptor

```python
import json
from typing import Callable, Any
from pydantic_ai import Agent, RunContext

class ToolInterceptor:
    """Intercept and optionally modify tool calls during debugging."""
    
    def __init__(self):
        self.intercept_tools: set[str] = set()
        self.call_log: list[dict] = []
        self.pause_on_error: bool = True
    
    def watch(self, *tool_names: str):
        """Register tools to intercept."""
        self.intercept_tools.update(tool_names)
        return self
    
    def wrap(self, func: Callable) -> Callable:
        """Wrap a tool function with interception logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            
            # Log every call
            call_record = {
                "tool": tool_name,
                "args": {k: v for k, v in kwargs.items()},
                "call_number": len(self.call_log) + 1,
            }
            self.call_log.append(call_record)
            
            # Interactive pause for watched tools
            if tool_name in self.intercept_tools:
                print(f"\n{'='*50}")
                print(f"🔍 INTERCEPTED: {tool_name} (call #{call_record['call_number']})")
                print(f"   Args: {json.dumps(call_record['args'], default=str, indent=2)}")
                print(f"   Previous calls: {len(self.call_log) - 1}")
                print(f"{'='*50}")
                
                # Interactive debugging prompt
                action = input("   [c]ontinue, [i]nspect, [s]kip, [b]reakpoint: ").strip()
                
                if action == "b":
                    breakpoint()
                elif action == "s":
                    return f"[SKIPPED] Tool '{tool_name}' was skipped during debugging"
                elif action == "i":
                    print(f"\n   Full call log:")
                    for record in self.call_log:
                        print(f"     #{record['call_number']}: {record['tool']}({record['args']})")
                    input("   Press Enter to continue...")
            
            # Execute the actual tool
            try:
                result = func(*args, **kwargs)
                call_record["result"] = str(result)[:200]
                call_record["success"] = True
                return result
            except Exception as e:
                call_record["error"] = str(e)
                call_record["success"] = False
                if self.pause_on_error:
                    print(f"\n❌ Tool '{tool_name}' raised: {e}")
                    breakpoint()
                raise
        
        return wrapper

# Usage
interceptor = ToolInterceptor()
interceptor.watch("execute_query", "delete_record")  # Watch dangerous tools

agent = Agent("openai:gpt-4o", instructions="Manage the database.")

@agent.tool
@interceptor.wrap
def execute_query(ctx: RunContext[None], sql: str) -> str:
    """Execute a SQL query."""
    return f"Result: 42 rows"

@agent.tool
@interceptor.wrap
def delete_record(ctx: RunContext[None], table: str, record_id: str) -> str:
    """Delete a record from a table."""
    return f"Deleted {record_id} from {table}"

@agent.tool
@interceptor.wrap
def list_tables(ctx: RunContext[None]) -> str:
    """List all database tables."""
    # Not watched — runs without interruption
    return "Tables: users, orders, products"
```

> **💡 Tip:** The interceptor pattern is especially useful for debugging agents that call "dangerous" tools like database writes, file deletions, or API mutations. Watch those tools specifically while letting read-only tools run freely.

---

## State change triggers

Agents maintain state that evolves across iterations — memory, context, accumulated tool results. State change triggers pause execution when the state meets a specific condition.

### Monitoring agent state changes

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentState:
    """Observable agent state that triggers breakpoints on changes."""
    
    _data: dict = field(default_factory=dict)
    _watchers: dict = field(default_factory=dict)
    _history: list[dict] = field(default_factory=list)
    
    def set(self, key: str, value: Any):
        """Set a state value. Triggers watchers if conditions are met."""
        old_value = self._data.get(key)
        self._data[key] = value
        
        # Record history
        self._history.append({
            "key": key,
            "old": old_value,
            "new": value,
            "change_number": len(self._history) + 1,
        })
        
        # Check watchers
        if key in self._watchers:
            condition = self._watchers[key]
            if condition(value, old_value):
                print(f"\n🔴 State breakpoint triggered!")
                print(f"   Key: '{key}'")
                print(f"   Old value: {old_value}")
                print(f"   New value: {value}")
                print(f"   State history ({len(self._history)} changes):")
                for h in self._history[-5:]:
                    print(f"     {h['key']}: {h['old']} → {h['new']}")
                breakpoint()
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def watch(self, key: str, condition):
        """Set a watcher that triggers breakpoint when condition is True.
        
        condition(new_value, old_value) -> bool
        """
        self._watchers[key] = condition

# Usage
state = AgentState()

# Trigger breakpoint when iteration count exceeds 10 (possible infinite loop)
state.watch("iteration_count", lambda new, old: new > 10)

# Trigger when context grows too large (approaching context window limit)
state.watch("context_tokens", lambda new, old: new > 100_000)

# Trigger when a specific error state is reached
state.watch("error_count", lambda new, old: new >= 3)

# Trigger on any value change for a critical field
state.watch("active_agent", lambda new, old: new != old and old is not None)

# Simulate agent execution
for i in range(15):
    state.set("iteration_count", i + 1)
    state.set("context_tokens", (i + 1) * 8500)
    # Breakpoint will trigger at iteration 11
```

**Output:**

```
🔴 State breakpoint triggered!
   Key: 'iteration_count'
   Old value: 10
   New value: 11
   State history (21 changes):
     iteration_count: 9 → 10
     context_tokens: 85000 → 85000
     iteration_count: 10 → 11
> /path/to/script.py(32)set()
(Pdb) 
```

### Watching LangGraph state changes

For LangGraph agents, state transitions are first-class concepts. We can intercept them directly:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
import operator

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    tool_calls: int
    current_step: str

def debug_state_node(state: AgentState) -> dict:
    """Debug node that inspects state between steps."""
    tool_calls = state.get("tool_calls", 0)
    current_step = state.get("current_step", "unknown")
    
    # Conditional breakpoint on state
    if tool_calls > 5:
        print(f"⚠️ Agent has made {tool_calls} tool calls")
        print(f"   Current step: {current_step}")
        print(f"   Messages: {len(state.get('messages', []))}")
        breakpoint()
    
    return {}  # No state changes — this is a debug-only node

# Insert debug nodes between real nodes in the graph
builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("debug_after_plan", debug_state_node)  # Debug checkpoint
builder.add_node("execute", execute_node)
builder.add_node("debug_after_execute", debug_state_node)  # Debug checkpoint
builder.add_edge("plan", "debug_after_plan")
builder.add_edge("debug_after_plan", "execute")
builder.add_edge("execute", "debug_after_execute")
```

> **Warning:** Remove debug nodes before deploying to production. They add latency and can interfere with state management. Use feature flags or environment checks to enable/disable them.

---

## Exception breakpoints

Exception breakpoints pause execution the moment an error occurs — before the exception propagates up the call stack. This lets us inspect the exact state that caused the failure.

### Catching tool execution errors

```python
import sys
import functools
from pydantic_ai import Agent, RunContext

def break_on_exception(func):
    """Decorator: drop into debugger when a tool raises an exception."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n❌ Exception in tool '{func.__name__}':")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {e}")
            print(f"   Args: {kwargs}")
            
            # Post-mortem debugging — inspect the state at the point of failure
            import pdb
            pdb.post_mortem(sys.exc_info()[2])
            
            raise  # Re-raise after debugging
    return wrapper

agent = Agent("openai:gpt-4o", instructions="Process customer orders.")

@agent.tool
@break_on_exception
def process_payment(ctx: RunContext[None], amount: float, currency: str) -> str:
    """Process a payment."""
    if amount <= 0:
        raise ValueError(f"Invalid payment amount: {amount}")
    if currency not in ("USD", "EUR", "GBP"):
        raise ValueError(f"Unsupported currency: {currency}")
    return f"Payment of {amount} {currency} processed"
```

**When the LLM passes an invalid amount:**

```
❌ Exception in tool 'process_payment':
   Type: ValueError
   Message: Invalid payment amount: -5.0
   Args: {'amount': -5.0, 'currency': 'USD'}
> /path/to/script.py(8)process_payment()
-> raise ValueError(f"Invalid payment amount: {amount}")
(Pdb) p amount
-5.0
(Pdb) p currency
'USD'
(Pdb) w
  ...
  /path/to/pydantic_ai/agent.py(234)_run_tool()
  /path/to/script.py(12)wrapper()
> /path/to/script.py(8)process_payment()
```

### VS Code exception breakpoints

VS Code has built-in exception breakpoint support. Configure it in the debug panel:

1. Open the **Run and Debug** panel (Ctrl+Shift+D)
2. In the **Breakpoints** section, check:
   - **Raised Exceptions** — pause on every exception (noisy but thorough)
   - **Uncaught Exceptions** — pause only on unhandled exceptions (recommended for agents)
3. Optionally add **conditional exception breakpoints**:
   - Right-click → "Add Conditional Exception Breakpoint"
   - Condition: `type(e).__name__ == 'ValueError'`

### Catching specific exception types in agent loops

```python
import functools

def break_on(exception_types: tuple, tools: set[str] | None = None):
    """Break only on specific exception types, optionally for specific tools."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if tools is None or func.__name__ in tools:
                    print(f"\n🎯 Caught {type(e).__name__} in '{func.__name__}'")
                    breakpoint()
                raise
        return wrapper
    return decorator

# Only break on ValueError and KeyError, only in specific tools
@agent.tool
@break_on((ValueError, KeyError), tools={"process_payment", "update_order"})
def process_payment(ctx: RunContext[None], amount: float, currency: str) -> str:
    """Process a payment."""
    if amount <= 0:
        raise ValueError(f"Invalid amount: {amount}")
    return f"Processed {amount} {currency}"
```

---

## Iteration-aware debugging

Agent loops iterate multiple times. We need breakpoints that understand the iteration count and pause at specific iterations.

### Iteration counter with smart breakpoints

```python
from pydantic_ai import Agent, RunContext

class IterationDebugger:
    """Track agent loop iterations and break at specific points."""
    
    def __init__(self):
        self.iteration = 0
        self.tool_calls_per_iteration: dict[int, list[str]] = {}
        self.break_at_iteration: int | None = None
        self.break_after_n_calls: int | None = None
        self.total_tool_calls = 0
    
    def on_tool_call(self, tool_name: str, args: dict):
        """Call this at the start of every tool function."""
        self.total_tool_calls += 1
        
        if self.iteration not in self.tool_calls_per_iteration:
            self.tool_calls_per_iteration[self.iteration] = []
        self.tool_calls_per_iteration[self.iteration].append(tool_name)
        
        # Break conditions
        should_break = False
        reason = ""
        
        if self.break_at_iteration and self.iteration >= self.break_at_iteration:
            should_break = True
            reason = f"Reached iteration {self.iteration}"
        
        if self.break_after_n_calls and self.total_tool_calls >= self.break_after_n_calls:
            should_break = True
            reason = f"Reached {self.total_tool_calls} total tool calls"
        
        # Detect potential infinite loop (same tool called 5+ times in a row)
        recent_calls = self.tool_calls_per_iteration.get(self.iteration, [])
        if len(recent_calls) >= 5 and len(set(recent_calls[-5:])) == 1:
            should_break = True
            reason = f"Possible loop: '{tool_name}' called 5 times in iteration {self.iteration}"
        
        if should_break:
            print(f"\n🔴 Iteration breakpoint: {reason}")
            print(f"   Iteration: {self.iteration}")
            print(f"   Total tool calls: {self.total_tool_calls}")
            print(f"   Current tool: {tool_name}({args})")
            breakpoint()
    
    def next_iteration(self):
        """Call this at the start of each agent loop iteration."""
        self.iteration += 1
    
    def summary(self) -> str:
        """Print a summary of all iterations."""
        lines = [f"Agent completed in {self.iteration} iterations:"]
        for i, tools in sorted(self.tool_calls_per_iteration.items()):
            lines.append(f"  Iteration {i}: {', '.join(tools)}")
        lines.append(f"  Total tool calls: {self.total_tool_calls}")
        return "\n".join(lines)

# Usage
debugger = IterationDebugger()
debugger.break_after_n_calls = 10  # Break if agent makes 10+ tool calls

agent = Agent("openai:gpt-4o", instructions="Research a topic thoroughly.")

@agent.tool
def search(ctx: RunContext[None], query: str) -> str:
    debugger.on_tool_call("search", {"query": query})
    return f"Results for '{query}'"

@agent.tool
def read_article(ctx: RunContext[None], url: str) -> str:
    debugger.on_tool_call("read_article", {"url": url})
    return f"Article content from {url}"

result = agent.run_sync("Research the history of quantum computing")
print(debugger.summary())
```

**Output:**

```
Agent completed in 3 iterations:
  Iteration 0: search, read_article, read_article
  Iteration 0: search, read_article
  Total tool calls: 5
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use conditional breakpoints over unconditional ones | Agents loop many times — unconditional breaks waste time |
| Watch "dangerous" tools specifically | Database writes, deletions, and API mutations need inspection |
| Monitor iteration count for infinite loop detection | Break when iterations exceed a reasonable threshold |
| Use post-mortem debugging for exceptions | Inspects state at the exact point of failure |
| Remove debug nodes before deploying | Debug checkpoints add latency and can leak state |
| Combine breakpoints with logging | Breakpoints pause execution; logs provide the full history |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `breakpoint()` in async callbacks | Place breakpoints in the main coroutine, not in callbacks |
| Breaking on every tool call in a long loop | Use conditional breakpoints: `if tool_name == 'target'` |
| Forgetting to remove debug interceptors | Use environment checks: `if os.getenv('DEBUG_AGENT')` |
| Not tracking iteration count | Use an `IterationDebugger` class to count and break smartly |
| Catching exceptions too broadly in agent code | Use `break_on((ValueError, KeyError))` for specific types |
| Not inspecting full state at breakpoint | Use `pp state` or `pp messages` to see the complete picture |

---

## Hands-on exercise

### Your task

Build an iteration-aware debugging system for a research agent that searches for information, reads articles, and synthesizes findings. The system should detect infinite loops and pause on specific tool calls.

### Requirements

1. Create an `IterationDebugger` that tracks tool calls per iteration
2. Add a conditional breakpoint that triggers if the agent calls `search` more than 3 times with the same query
3. Add a state change watcher that triggers when the context exceeds a token threshold
4. Wire the debugger into three tool functions: `search`, `read_article`, `synthesize`
5. Run the agent and verify the debugger produces a meaningful summary

### Expected result

The debugger should track all tool calls, detect duplicate searches, and produce a summary like: "Agent completed in 4 iterations, 8 tool calls, 1 duplicate search detected."

<details>
<summary>💡 Hints (click to expand)</summary>

- Track previous search queries in a `set` to detect duplicates
- Use the `IterationDebugger.on_tool_call()` pattern from this lesson
- The context token threshold should be based on your model's context window (e.g., 128K for GPT-4o)
- Consider using a `defaultdict(list)` to group tool calls by iteration

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from collections import defaultdict
from pydantic_ai import Agent, RunContext

class ResearchDebugger:
    def __init__(self, max_iterations: int = 20, max_duplicate_searches: int = 3):
        self.iteration = 0
        self.tool_calls: list[dict] = []
        self.search_queries: dict[str, int] = defaultdict(int)
        self.max_iterations = max_iterations
        self.max_duplicate_searches = max_duplicate_searches
        self.context_tokens = 0
        self.warnings: list[str] = []
    
    def on_tool_call(self, tool_name: str, args: dict):
        self.tool_calls.append({"tool": tool_name, "args": args, "iteration": self.iteration})
        
        # Check for duplicate searches
        if tool_name == "search":
            query = args.get("query", "")
            self.search_queries[query] += 1
            if self.search_queries[query] > self.max_duplicate_searches:
                warning = f"Duplicate search '{query}' ({self.search_queries[query]} times)"
                self.warnings.append(warning)
                print(f"\n⚠️ {warning}")
                breakpoint()
        
        # Check iteration limit
        if self.iteration > self.max_iterations:
            warning = f"Exceeded max iterations ({self.max_iterations})"
            self.warnings.append(warning)
            print(f"\n🔴 {warning}")
            breakpoint()
    
    def add_tokens(self, count: int, threshold: int = 100_000):
        self.context_tokens += count
        if self.context_tokens > threshold:
            print(f"\n⚠️ Context tokens: {self.context_tokens} (threshold: {threshold})")
            breakpoint()
    
    def summary(self) -> str:
        tools_by_iter = defaultdict(list)
        for call in self.tool_calls:
            tools_by_iter[call["iteration"]].append(call["tool"])
        
        duplicates = sum(1 for count in self.search_queries.values() if count > 1)
        
        lines = [
            f"Agent completed in {self.iteration} iterations, "
            f"{len(self.tool_calls)} tool calls, "
            f"{duplicates} duplicate search(es) detected.",
            "",
        ]
        for i, tools in sorted(tools_by_iter.items()):
            lines.append(f"  Iteration {i}: {', '.join(tools)}")
        if self.warnings:
            lines.append(f"\n  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    ⚠️ {w}")
        return "\n".join(lines)

# Wire it up
debugger = ResearchDebugger()
agent = Agent("openai:gpt-4o", instructions="Research topics thoroughly.")

@agent.tool
def search(ctx: RunContext[None], query: str) -> str:
    debugger.on_tool_call("search", {"query": query})
    debugger.add_tokens(500)
    return f"Results for '{query}': 3 articles found"

@agent.tool
def read_article(ctx: RunContext[None], url: str) -> str:
    debugger.on_tool_call("read_article", {"url": url})
    debugger.add_tokens(2000)
    return f"Article content from {url}: Lorem ipsum..."

@agent.tool
def synthesize(ctx: RunContext[None], findings: str) -> str:
    debugger.on_tool_call("synthesize", {"findings": findings[:50]})
    debugger.add_tokens(1000)
    return f"Synthesis: {findings[:100]}..."

result = agent.run_sync("Research quantum computing breakthroughs in 2025")
print(debugger.summary())
```

</details>

### Bonus challenges

- [ ] Add a "replay mode" that re-runs the same tool calls from a previous session without LLM calls
- [ ] Create a VS Code launch configuration with conditional exception breakpoints for `ValueError` only
- [ ] Build a dashboard that visualizes tool calls per iteration as a bar chart

---

## Summary

✅ Conditional breakpoints (`if tool_name == 'target': breakpoint()`) let us skip irrelevant iterations and pause only on the actions that matter

✅ Tool call interceptors provide a debugging middleware — inspect, modify, or skip tool calls interactively during agent execution

✅ State change watchers trigger breakpoints when agent state crosses thresholds — iteration count, token usage, error count, or agent handoffs

✅ Exception breakpoints with `pdb.post_mortem()` let us inspect the exact state at the moment of failure, before the exception propagates

✅ Iteration-aware debuggers track tool calls per iteration and automatically detect infinite loops — the most common agent failure mode

---

**Next:** [Replay and Reproduce Issues](./05-replay-reproduce-issues.md)

**Previous:** [Visualization of Agent Decisions](./03-visualization-agent-decisions.md)

---

## Further Reading

- [Python pdb Documentation](https://docs.python.org/3/library/pdb.html) - Conditional breakpoints and post-mortem debugging
- [VS Code Conditional Breakpoints](https://code.visualstudio.com/docs/editor/debugging#_conditional-breakpoints) - IDE breakpoint configuration
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/) - Interrupt agent execution for inspection
- [OpenAI Agents SDK Tracing](https://openai.github.io/openai-agents-python/tracing/) - Built-in execution tracing
- [Pydantic AI Testing](https://ai.pydantic.dev/testing/) - TestModel for controlled execution

<!-- 
Sources Consulted:
- Python pdb docs: https://docs.python.org/3/library/pdb.html
- VS Code debugging: https://code.visualstudio.com/docs/editor/debugging
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- OpenAI Agents SDK Tracing: https://openai.github.io/openai-agents-python/tracing/
- LangGraph docs: https://langchain-ai.github.io/langgraph/
-->
