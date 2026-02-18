---
title: "Maximum Iteration Limits"
---

# Maximum Iteration Limits

## Introduction

A well-intentioned agent can still run forever. A research agent might keep finding "one more source" to check. A coding agent might rewrite the same function endlessly, never satisfied. Maximum iteration limits are the ultimate safety net — a hard ceiling that guarantees every agent loop terminates, regardless of what the LLM decides.

This lesson covers the practical engineering of iteration limits: how to set them, what to do when they're hit, how to detect stuck agents, and how to gracefully escalate instead of simply crashing.

### What we'll cover

- Setting `max_turns` in OpenAI Agents SDK and `max_iterations` in custom loops
- Handling `MaxTurnsExceeded` and other limit exceptions
- Stuck detection — identifying agents caught in repetitive cycles
- Cost protection through iteration budgets
- Escalation strategies: fallback responses, human handoff, and partial results
- Error handlers for graceful limit management

### Prerequisites

- [Loop Termination Conditions](./05-loop-termination-conditions.md) — termination strategies
- [Simple Loop](./01-simple-loop.md) — the basic agent execution cycle
- Python exception handling (`try`/`except`, custom exceptions)

---

## Setting iteration limits

### OpenAI Agents SDK: max_turns

The OpenAI Agents SDK provides `max_turns` as a parameter on `Runner.run()`. Each turn is one complete LLM call plus any resulting tool executions:

```python
from agents import Agent, Runner

agent = Agent(
    name="Research Agent",
    instructions="Research the topic thoroughly using available tools.",
    tools=[search_tool, read_tool],
)

# Limit to 5 turns — 5 LLM calls maximum
result = await Runner.run(
    agent,
    input="Explain quantum entanglement with real-world applications",
    max_turns=5,
)
print(result.final_output)
```

If the agent exceeds 5 turns without producing a final output, the SDK raises `MaxTurnsExceeded`:

```python
from agents import MaxTurnsExceeded

try:
    result = await Runner.run(agent, input="...", max_turns=3)
except MaxTurnsExceeded as e:
    print(f"Agent exceeded {e.max_turns} turns without completing")
    # Handle gracefully — don't just crash
```

### Custom loops: explicit counter

For hand-built loops, the counter is explicit:

```python
def run_agent(task: str, max_iterations: int = 10) -> str:
    """Agent loop with explicit iteration limit."""
    messages = [
        {"role": "system", "content": "Complete the task using available tools."},
        {"role": "user", "content": task},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content  # Success

        # Execute tools and continue...
        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            result = execute_tool(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Fell through the loop — max iterations reached
    return f"Task incomplete after {max_iterations} iterations. Last progress: {messages[-1].get('content', 'N/A')}"
```

### Choosing the right limit

| Task type | Suggested max_turns | Reasoning |
|-----------|-------------------|-----------|
| Simple Q&A with one tool | 3-5 | 1-2 tool calls then answer |
| Multi-step research | 8-15 | Multiple searches, reading, synthesizing |
| Code generation with testing | 10-20 | Write, test, fix, repeat |
| Open-ended exploration | 15-25 | Many avenues to explore |
| Iterative refinement | 5-8 | Diminishing returns after ~5 passes |

> **Warning:** Setting limits too low causes premature termination on legitimate tasks. Setting them too high wastes budget on stuck agents. Start with 10 as a default and adjust based on observed behavior.

---

## Handling MaxTurnsExceeded

When an agent hits its limit, you have several options beyond simply raising an exception.

### OpenAI Agents SDK error handlers

The SDK provides a dedicated `error_handlers` mechanism for `max_turns`:

```python
from agents import Agent, Runner
from agents.types import RunErrorHandlerResult, RunErrorHandlerInput

async def on_max_turns(ctx: RunErrorHandlerInput) -> RunErrorHandlerResult:
    """Handle max turns gracefully — return a partial answer instead of crashing."""
    # Extract what the agent found so far from conversation history
    findings = []
    for item in ctx.items:
        if hasattr(item, 'content') and item.content:
            findings.append(item.content)

    partial_answer = (
        f"I wasn't able to complete the task within {ctx.max_turns} turns. "
        f"Here's what I found so far:\n\n"
        + "\n".join(findings[-3:])  # Last 3 messages
    )

    return RunErrorHandlerResult(
        final_output=partial_answer,
        include_in_history=False,  # Don't pollute conversation with the error
    )

agent = Agent(
    name="Research Agent",
    instructions="Research the topic using tools.",
    tools=[search_tool],
)

result = await Runner.run(
    agent,
    input="Comprehensive analysis of renewable energy trends",
    max_turns=5,
    error_handlers={"max_turns": on_max_turns},
)

# No exception raised — the error handler produced a partial result
print(result.final_output)
```

**Output:**
```
I wasn't able to complete the task within 5 turns. Here's what I found so far:

Solar energy capacity grew by 45% globally in 2024...
Wind energy investment reached $120 billion...
Battery storage costs declined 15% year-over-year...
```

> **🔑 Key concept:** The `error_handlers` dict lets you register handlers for specific error types. The `"max_turns"` key catches `MaxTurnsExceeded`. The handler receives the full conversation context and returns a `RunErrorHandlerResult` with a `final_output` — turning a crash into a graceful degradation.

### Custom exception pattern

For hand-built loops, define a custom exception hierarchy:

```python
class AgentError(Exception):
    """Base exception for agent errors."""
    pass

class MaxIterationsError(AgentError):
    """Agent exceeded its iteration limit."""
    def __init__(self, max_iterations: int, partial_result: str = ""):
        self.max_iterations = max_iterations
        self.partial_result = partial_result
        super().__init__(f"Agent exceeded {max_iterations} iterations")

class BudgetExceededError(AgentError):
    """Agent exceeded its cost budget."""
    def __init__(self, budget: float, actual_cost: float):
        self.budget = budget
        self.actual_cost = actual_cost
        super().__init__(f"Cost ${actual_cost:.4f} exceeded budget ${budget:.4f}")

# Usage
try:
    result = run_agent("complex task", max_iterations=10)
except MaxIterationsError as e:
    print(f"Partial result: {e.partial_result}")
except BudgetExceededError as e:
    print(f"Over budget by ${e.actual_cost - e.budget:.4f}")
except AgentError as e:
    print(f"Agent error: {e}")
```

---

## Stuck detection

An agent can hit its iteration limit in two very different ways:

1. **Productive but slow** — the agent is making progress but ran out of turns
2. **Stuck in a loop** — the agent is repeating the same actions with no progress

Detecting the difference lets you respond appropriately.

### Detecting repetitive tool calls

```python
from collections import Counter

def detect_stuck_agent(tool_call_history: list[dict], window: int = 4) -> bool:
    """Detect if the agent is repeating the same tool calls.

    Args:
        tool_call_history: List of {"name": ..., "args": ...} dicts.
        window: Number of recent calls to analyze.

    Returns:
        True if the agent appears stuck.
    """
    if len(tool_call_history) < window:
        return False

    recent = tool_call_history[-window:]
    # Convert to hashable tuples for counting
    recent_calls = [
        (call["name"], tuple(sorted(call["args"].items())))
        for call in recent
    ]

    counts = Counter(recent_calls)
    most_common_call, most_common_count = counts.most_common(1)[0]

    # If the same call appears in >50% of recent calls, agent is stuck
    if most_common_count / window > 0.5:
        name, args = most_common_call
        print(f"  🔄 Stuck: called {name}({dict(args)}) {most_common_count}/{window} times")
        return True

    return False

# Example
history = [
    {"name": "search", "args": {"query": "python GIL"}},
    {"name": "search", "args": {"query": "python GIL"}},
    {"name": "read_url", "args": {"url": "https://example.com"}},
    {"name": "search", "args": {"query": "python GIL"}},
]
detect_stuck_agent(history)
```

**Output:**
```
  🔄 Stuck: called search({'query': 'python GIL'}) 3/4 times
```

### Detecting identical outputs

Sometimes the agent calls different tools but produces the same output:

```python
def detect_output_repetition(
    output_history: list[str],
    window: int = 3,
    similarity_threshold: float = 0.9,
) -> bool:
    """Detect if the agent is producing the same output repeatedly.

    Args:
        output_history: List of agent outputs at each iteration.
        window: Number of recent outputs to compare.
        similarity_threshold: How similar outputs must be to count as "same."

    Returns:
        True if recent outputs are too similar.
    """
    if len(output_history) < window:
        return False

    recent = output_history[-window:]

    # Simple overlap check — count matching words
    for i in range(len(recent) - 1):
        words_a = set(recent[i].lower().split())
        words_b = set(recent[i + 1].lower().split())
        if not words_a or not words_b:
            continue
        overlap = len(words_a & words_b) / max(len(words_a), len(words_b))

        if overlap < similarity_threshold:
            return False  # At least one pair is different enough

    print(f"  🔄 Output repetition detected in last {window} iterations")
    return True
```

### Integrating stuck detection into the loop

```python
def run_agent_with_stuck_detection(
    task: str,
    max_iterations: int = 10,
    stuck_window: int = 4,
) -> str:
    """Agent loop with stuck detection."""
    messages = [{"role": "user", "content": task}]
    tool_call_history = []
    output_history = []

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        # Track tool calls
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_call_history.append({"name": tc.function.name, "args": args})

        # Check for stuck agent
        if detect_stuck_agent(tool_call_history, window=stuck_window):
            # Inject a "nudge" message to break the cycle
            messages.append({
                "role": "user",
                "content": (
                    "You seem to be repeating the same actions. "
                    "Try a different approach or provide your best answer "
                    "with the information you have."
                ),
            })
            print(f"  💡 Injected nudge at iteration {iteration}")

        # ... execute tools, continue loop ...

    return f"Max iterations reached. Last result: {output_history[-1] if output_history else 'N/A'}"
```

> **💡 Tip:** Instead of immediately terminating a stuck agent, try injecting a "nudge" message first. This gives the LLM a chance to break out of its loop. If the agent is still stuck after the nudge, then terminate.

---

## Cost protection strategies

Iteration limits alone don't protect against cost overruns. A single turn with a large context window can cost more than 10 turns with a small one.

### Token budget tracking

```python
class TokenBudget:
    """Track and enforce token usage limits."""

    def __init__(self, max_input_tokens: int = 100_000, max_output_tokens: int = 20_000):
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens
        self.total_input = 0
        self.total_output = 0

    def record(self, input_tokens: int, output_tokens: int) -> bool:
        """Record token usage. Returns False if budget is exceeded."""
        self.total_input += input_tokens
        self.total_output += output_tokens

        if self.total_input > self.max_input:
            print(f"  ⚠️ Input token budget exceeded: {self.total_input}/{self.max_input}")
            return False

        if self.total_output > self.max_output:
            print(f"  ⚠️ Output token budget exceeded: {self.total_output}/{self.max_output}")
            return False

        return True

    def remaining(self) -> dict:
        return {
            "input_remaining": self.max_input - self.total_input,
            "output_remaining": self.max_output - self.total_output,
            "input_pct": f"{(self.total_input / self.max_input) * 100:.0f}%",
            "output_pct": f"{(self.total_output / self.max_output) * 100:.0f}%",
        }
```

### Context window management

As the conversation grows, each LLM call sends more tokens. Managing context growth is a form of cost protection:

```python
from agents import Agent, Runner
from agents.types import CallModelData, ModelInputData

def limit_context(data: CallModelData) -> ModelInputData:
    """Trim conversation history to prevent context overflow.

    This function is used as a call_model_input_filter in RunConfig.
    """
    messages = data.input.messages

    # Keep system prompt + last 10 messages
    if len(messages) > 11:
        system_msgs = [m for m in messages if getattr(m, "role", None) == "system"]
        recent_msgs = messages[-10:]
        trimmed = system_msgs + recent_msgs
        print(f"  ✂️ Trimmed context: {len(messages)} → {len(trimmed)} messages")
        return ModelInputData(messages=trimmed, tools=data.input.tools)

    return data.input

# Use in RunConfig
result = await Runner.run(
    agent,
    input="...",
    run_config=RunConfig(call_model_input_filter=limit_context),
)
```

> **🤖 AI Context:** The `call_model_input_filter` in OpenAI Agents SDK lets you modify the LLM input before each call. This is the ideal hook for implementing context trimming, token counting, and message filtering — all without modifying the agent's core logic.

---

## Escalation strategies

When an agent hits its limit, the worst option is returning nothing. Escalation strategies provide useful outcomes even in failure:

### Strategy 1: Partial result with disclaimer

```python
def escalate_partial_result(findings: list[str], task: str) -> str:
    """Return whatever the agent found with a disclaimer."""
    return (
        f"⚠️ I couldn't fully complete the task within the allowed iterations.\n\n"
        f"**Task:** {task}\n\n"
        f"**What I found:**\n"
        + "\n".join(f"- {f}" for f in findings[-5:])
        + "\n\n**Recommendation:** Ask a follow-up question for the remaining details."
    )
```

### Strategy 2: Human handoff

```python
class EscalationResult:
    """Result when an agent needs to hand off to a human."""

    def __init__(self, task: str, findings: list[str], reason: str):
        self.task = task
        self.findings = findings
        self.reason = reason
        self.ticket_id = f"ESC-{hash(task) % 100000:05d}"

    def to_message(self) -> str:
        return (
            f"I've created escalation ticket **{self.ticket_id}**.\n\n"
            f"**Reason:** {self.reason}\n"
            f"**Progress so far:** {len(self.findings)} findings collected.\n\n"
            f"A human agent will continue this task."
        )
```

### Strategy 3: Fallback to simpler approach

```python
async def escalate_to_simple_response(task: str, context: str) -> str:
    """Fall back to a single-shot LLM call without tools."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "The agent loop has been exhausted. Based on the context below, "
                    "provide the best possible answer without using any tools."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nContext gathered so far:\n{context}",
            },
        ],
    )
    return response.choices[0].message.content
```

### Choosing an escalation strategy

| Situation | Strategy | Example |
|-----------|----------|---------|
| Agent made significant progress | Partial result | Research agent found 4/5 answers |
| Agent is completely stuck | Human handoff | Complex debugging with no clear path |
| Task is answerable from context | Fallback simple response | Agent has data but keeps searching |
| Cost/time critical | Immediate fallback | Customer chat with timeout pressure |

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Always set a `max_turns` / `max_iterations` | No agent should run unbounded — even a high limit is better than none |
| Implement stuck detection alongside iteration limits | Limits stop runaway loops; stuck detection identifies *why* they happen |
| Use error handlers, not try/except, in SDK-based agents | Error handlers produce `RunResult` objects with `final_output`, not exceptions |
| Track token usage, not just iteration count | One iteration with a 100K context costs more than ten with 1K |
| Inject nudge messages before terminating stuck agents | Give the LLM a chance to self-correct before pulling the plug |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Catching `MaxTurnsExceeded` and returning an empty string | Use error handlers to produce a meaningful partial result |
| Same `max_turns` for all tasks | Scale limits to task complexity — 3 for simple, 15 for complex |
| No distinction between "stuck" and "slow but productive" | Track tool call diversity and output changes, not just iteration count |
| Only limiting iterations, not tokens or cost | An agent can burn through budget in 2 turns with large contexts |
| Terminating without logging the reason | Always log which condition triggered the stop and the agent's last state |

---

## Hands-on exercise

### Your task

Build a production-ready `SafeAgentRunner` class that wraps any agent loop with comprehensive safety limits: iteration cap, token budget, cost budget, stuck detection, and configurable escalation. The runner should never raise an exception to the caller — every termination produces a meaningful result.

### Requirements

1. Configure `max_iterations`, `max_tokens` (input + output), `max_cost`, and `stuck_window`
2. Implement stuck detection that checks for repeated tool calls
3. Try a "nudge" message before terminating a stuck agent
4. On any limit trigger, produce a structured result with: `status` ("completed" | "partial" | "stuck" | "budget_exceeded"), `output`, `iterations_used`, `tokens_used`, `cost`, and `termination_reason`
5. Never raise an exception — all failures are returned as structured results

### Expected result

```python
runner = SafeAgentRunner(max_iterations=10, max_cost=0.50, stuck_window=3)
result = runner.run("Analyze the pros and cons of microservices")

# Always returns a result, never raises
print(result.status)             # "completed" or "partial"
print(result.output)             # The agent's answer
print(result.iterations_used)    # 7
print(result.tokens_used)        # 12450
print(result.cost)               # 0.0234
print(result.termination_reason) # "goal_achieved" or "max_iterations"
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a `dataclass` for the structured result type
- The stuck detection function should return a boolean — if True, inject a nudge first, then check again on the next iteration
- Track cost using `response.usage.prompt_tokens` and `response.usage.completion_tokens`
- Wrap the entire loop in a single try/except as a final safety net

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
import time
from dataclasses import dataclass, field
from collections import Counter
from openai import OpenAI

client = OpenAI()

@dataclass
class AgentResult:
    status: str  # "completed", "partial", "stuck", "budget_exceeded", "error"
    output: str
    iterations_used: int
    tokens_used: int
    cost: float
    termination_reason: str
    tool_calls_made: int = 0
    duration_seconds: float = 0.0

class SafeAgentRunner:
    PRICING = {"gpt-4o-mini": (0.00015, 0.0006), "gpt-4o": (0.0025, 0.01)}

    def __init__(
        self,
        max_iterations: int = 10,
        max_tokens: int = 100_000,
        max_cost: float = 1.0,
        stuck_window: int = 3,
        model: str = "gpt-4o-mini",
    ):
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.stuck_window = stuck_window
        self.model = model

    def _is_stuck(self, history: list[tuple], window: int) -> bool:
        if len(history) < window:
            return False
        recent = history[-window:]
        counts = Counter(recent)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / window > 0.5

    def run(self, task: str, tools: list = None, tool_functions: dict = None) -> AgentResult:
        start = time.time()
        total_tokens = 0
        total_cost = 0.0
        tool_call_history = []
        tool_calls_made = 0
        nudged = False
        last_output = ""

        messages = [
            {"role": "system", "content": "Complete the task using tools. When done, respond without tool calls."},
            {"role": "user", "content": task},
        ]

        try:
            for iteration in range(self.max_iterations):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or [],
                )
                choice = response.choices[0]

                # Track usage
                inp_rate, out_rate = self.PRICING.get(self.model, (0.01, 0.03))
                inp_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
                total_tokens += inp_tokens + out_tokens
                total_cost += (inp_tokens / 1000) * inp_rate + (out_tokens / 1000) * out_rate

                # Check token budget
                if total_tokens > self.max_tokens:
                    return AgentResult(
                        status="budget_exceeded", output=last_output,
                        iterations_used=iteration + 1, tokens_used=total_tokens,
                        cost=total_cost, termination_reason="token_budget",
                        tool_calls_made=tool_calls_made,
                        duration_seconds=time.time() - start,
                    )

                # Check cost budget
                if total_cost > self.max_cost:
                    return AgentResult(
                        status="budget_exceeded", output=last_output,
                        iterations_used=iteration + 1, tokens_used=total_tokens,
                        cost=total_cost, termination_reason="cost_budget",
                        tool_calls_made=tool_calls_made,
                        duration_seconds=time.time() - start,
                    )

                # Check for final output
                if choice.finish_reason == "stop":
                    return AgentResult(
                        status="completed", output=choice.message.content,
                        iterations_used=iteration + 1, tokens_used=total_tokens,
                        cost=total_cost, termination_reason="goal_achieved",
                        tool_calls_made=tool_calls_made,
                        duration_seconds=time.time() - start,
                    )

                # Process tool calls
                messages.append(choice.message)
                if choice.message.tool_calls and tool_functions:
                    for tc in choice.message.tool_calls:
                        args = json.loads(tc.function.arguments)
                        tool_call_history.append(
                            (tc.function.name, tuple(sorted(args.items())))
                        )
                        tool_calls_made += 1
                        result = tool_functions.get(tc.function.name, lambda **k: "{}")(**args)
                        last_output = result
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id, "content": result,
                        })

                # Stuck detection
                if self._is_stuck(tool_call_history, self.stuck_window):
                    if not nudged:
                        messages.append({
                            "role": "user",
                            "content": "You're repeating actions. Try a different approach or answer now.",
                        })
                        nudged = True
                    else:
                        return AgentResult(
                            status="stuck", output=last_output,
                            iterations_used=iteration + 1, tokens_used=total_tokens,
                            cost=total_cost, termination_reason="stuck_after_nudge",
                            tool_calls_made=tool_calls_made,
                            duration_seconds=time.time() - start,
                        )

            # Max iterations reached
            return AgentResult(
                status="partial", output=last_output,
                iterations_used=self.max_iterations, tokens_used=total_tokens,
                cost=total_cost, termination_reason="max_iterations",
                tool_calls_made=tool_calls_made,
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            return AgentResult(
                status="error", output=str(e),
                iterations_used=0, tokens_used=total_tokens,
                cost=total_cost, termination_reason=f"exception: {type(e).__name__}",
                tool_calls_made=tool_calls_made,
                duration_seconds=time.time() - start,
            )
```

</details>

### Bonus challenges

- [ ] Add exponential backoff for rate-limited API calls inside the runner
- [ ] Implement a `dry_run` mode that estimates cost without making real API calls
- [ ] Add async support with `asyncio.wait_for` for timeout-based termination

---

## Summary

✅ **Always set `max_turns`** — even a generous limit is better than no limit at all

✅ Use **error handlers** (not try/except) in the OpenAI Agents SDK for graceful limit handling

✅ **Stuck detection** identifies agents caught in repetitive cycles — try a nudge before terminating

✅ Track **tokens and cost**, not just iterations — one large-context call can exceed budgets fast

✅ Every termination should produce a **meaningful result** — partial answers beat empty responses

**Next:** [State Management](../07-state-management/00-state-management.md)

---

## Further reading

- [OpenAI Agents SDK — Running Agents](https://openai.github.io/openai-agents-python/running_agents/) — `max_turns`, `MaxTurnsExceeded`, error handlers, and `RunConfig`
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — stopping conditions and iteration limits
- [LangGraph — Agent Patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — conditional routing to END

*[Back to Execution Loop Patterns](./00-execution-loop-patterns.md)*

<!--
Sources Consulted:
- OpenAI Agents SDK Running Agents: https://openai.github.io/openai-agents-python/running_agents/
- Anthropic Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph Workflows and Agents: https://docs.langchain.com/oss/python/langgraph/workflows-agents
-->
