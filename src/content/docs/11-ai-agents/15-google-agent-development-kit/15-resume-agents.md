---
title: "Resume Agents in Google ADK"
---

# Resume Agents in Google ADK

Long-running agent workflows are inherently fragile. A multi-step pipeline that classifies documents, calls external APIs, processes data, and generates summaries can fail at any point — a network timeout at step three, an API rate limit at step four, or a simple process crash. Without a recovery mechanism, we must restart the entire workflow from scratch, wasting compute, burning API credits, and re-executing side effects that may not be idempotent.

Google's Agent Development Kit (ADK) addresses this with **resumability** (available in v1.14.0+). Resume agents allow workflows to pick up exactly where they left off after a failure, skipping already-completed steps and retrying only from the point of failure. In this lesson, we'll explore how resumability works, how to enable it for built-in agent types, and how to implement custom checkpoint logic for our own agents.

**Prerequisites:**

- Familiarity with ADK agent types (`SequentialAgent`, `ParallelAgent`, `LoopAgent`)
- Understanding of ADK's event system and session state
- Python 3.10+ and `google-adk >= 1.14.0`

---

## The Problem: Wasted Work on Failure

Let's consider a concrete scenario. We have a `SequentialAgent` that runs five steps in order — data ingestion, validation, transformation, enrichment, and export. Each step takes time and costs money (LLM calls, API requests, database writes).

```text
Step 1: Ingest Data        ✅ (30 seconds, 2 API calls)
Step 2: Validate Schema    ✅ (10 seconds, 1 LLM call)
Step 3: Transform Records  ❌ TIMEOUT after 45 seconds
Step 4: Enrich with AI     ⏸️ Never reached
Step 5: Export Results      ⏸️ Never reached
```

**Without resume**, our only option is to restart the entire pipeline from step one. We re-ingest data we already have, re-validate a schema we already checked, and pay for LLM calls we already made. For pipelines that take minutes or hours, this is unacceptable.

**With resume**, ADK recognizes that steps one and two completed successfully. It skips them entirely and retries from step three. We save time, money, and avoid duplicating side effects.

```text
Resume Pipeline:
Step 1: Ingest Data        ⏭️ SKIPPED (already completed)
Step 2: Validate Schema    ⏭️ SKIPPED (already completed)
Step 3: Transform Records  🔄 RETRY (picks up here)
Step 4: Enrich with AI     ▶️ Runs normally
Step 5: Export Results      ▶️ Runs normally
```

---

## How Resume Works Under the Hood

ADK's resumability builds on top of its **event system** and **session state**. Here's the mechanism in detail:

1. **Step tracking via Events** — Each time a sub-agent completes, ADK emits an event recording the completion.
2. **State checkpointing** — Completed steps generate state checkpoints that persist in the `SessionService`.
3. **Event history replay** — On restart, ADK reads the event history from the session.
4. **Skip logic** — Already-completed steps are identified and skipped entirely.
5. **Resume point** — Execution resumes from the first incomplete step.

The following diagram illustrates this flow:

```mermaid
flowchart TD
    A[Start Pipeline] --> B[Step 1 ✅]
    B --> C[Step 2 ✅]
    C --> D[Step 3 ❌ Failed]
    D --> E[Save Checkpoint]
    E --> F[Resume Pipeline]
    F --> G[Skip Step 1]
    G --> H[Skip Step 2]
    H --> I[Retry Step 3]
    I --> J[Step 4]
    J --> K[Step 5]
    K --> L[Complete]
```

The critical requirement is a **persistent `SessionService`**. If we use `InMemorySessionService`, all state is lost when the process restarts — which defeats the entire purpose. We must use `DatabaseSessionService` (or another persistent backend) to retain event history across process restarts.

---

## Enabling Resumability

Enabling resume for a workflow is straightforward. We wrap our agent in an `App` instance with a `ResumabilityConfig`:

```python
from google.adk.app import App
from google.adk.agents import SequentialAgent
from google.adk.app import ResumabilityConfig

# Define individual step agents (LlmAgent, custom, etc.)
step1 = LlmAgent(name="ingest", instruction="Ingest raw data...")
step2 = LlmAgent(name="validate", instruction="Validate the schema...")
step3 = LlmAgent(name="transform", instruction="Transform records...")
step4 = LlmAgent(name="enrich", instruction="Enrich with metadata...")
step5 = LlmAgent(name="export", instruction="Export final results...")

# Compose into a sequential pipeline
pipeline = SequentialAgent(
    name="data_pipeline",
    sub_agents=[step1, step2, step3, step4, step5],
)

# Wrap in App with resumability enabled
app = App(
    agent=pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

```text
Expected: The App is now configured so that if the pipeline fails and restarts
with the same session, it will skip completed steps and resume from the failure
point.
```

That single `ResumabilityConfig(is_resumable=True)` flag tells ADK to track step completion and support resume on restart. No other changes are needed for built-in agent types like `SequentialAgent`.

### Using a Persistent Session Service

Resume only works if checkpoint data survives process restarts. We need to pair our resumable app with a persistent session backend:

```python
from google.adk.sessions import DatabaseSessionService

# Use a persistent database-backed session service
session_service = DatabaseSessionService(
    db_url="postgresql://user:pass@localhost:5432/agents_db"
)

app = App(
    agent=pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
    session_service=session_service,
)
```

```text
Expected: Sessions and their event histories are now stored in PostgreSQL,
surviving process restarts and enabling resume from checkpoints.
```

---

## Resume Behavior by Agent Type

Not all agent types resume the same way. The behavior depends on the agent's execution model:

| Agent Type | Resume Behavior |
|---|---|
| `SequentialAgent` | Skips completed sub-agents, resumes from first incomplete |
| `ParallelAgent` | Re-runs all sub-agents (parallel work is atomic) |
| `LoopAgent` | Resumes from the last incomplete iteration |
| `LlmAgent` | Re-runs (LLM calls are not checkpointed) |
| Custom Agent | Requires manual `BaseAgentState` implementation |

### SequentialAgent

This is the most natural fit for resumability. Each sub-agent runs in order, and ADK can identify exactly which ones completed. On resume, completed sub-agents are skipped and execution picks up at the first incomplete one.

### ParallelAgent

Parallel execution is treated as **atomic**. If any branch fails, all branches re-run on resume. This is because partial parallel results may leave state in an inconsistent condition — it's safer to re-execute the entire parallel block.

### LoopAgent

Loops track iteration state. If a `LoopAgent` completes three of five iterations before failing, it resumes from iteration four on restart. Each iteration boundary acts as a checkpoint.

### LlmAgent

Individual LLM calls are **not** checkpointed. If an `LlmAgent` fails mid-generation, the entire LLM call re-runs on resume. This is expected — LLM responses are non-deterministic, and caching partial generations would introduce subtle bugs.

---

## Custom Agent Resumability

For custom agents that subclass `BaseAgent`, ADK does not automatically know how to checkpoint and resume. We must implement state tracking ourselves using `BaseAgentState` and the state event API.

```python
from google.adk.agents import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from enum import Enum

class WorkflowStep(str, Enum):
    CLASSIFY = "classify"
    PROCESS = "process"
    SUMMARIZE = "summarize"

class ResumableRouter(BaseAgent):
    classifier: Agent
    processor: Agent
    summarizer: Agent

    @property
    def sub_agents(self):
        return [self.classifier, self.processor, self.summarizer]

    async def _run_async_impl(self, ctx):
        # Check if resuming from a previous state
        state = self._get_agent_state(ctx)
        current_step = state.current_step if state else WorkflowStep.CLASSIFY

        if current_step <= WorkflowStep.CLASSIFY:
            self._create_agent_state_event(ctx, WorkflowStep.CLASSIFY)
            async for event in self.classifier.run_async(ctx):
                yield event

        if current_step <= WorkflowStep.PROCESS:
            self._create_agent_state_event(ctx, WorkflowStep.PROCESS)
            async for event in self.processor.run_async(ctx):
                yield event

        if current_step <= WorkflowStep.SUMMARIZE:
            self._create_agent_state_event(ctx, WorkflowStep.SUMMARIZE)
            async for event in self.summarizer.run_async(ctx):
                yield event
```

```text
Expected: On first run, the agent executes all three steps (classify → process
→ summarize). If it fails at "process" and restarts, _get_agent_state returns
the last checkpoint, and the agent skips "classify" and retries from "process".
```

### How the Custom State Pattern Works

Let's break down the key methods:

- **`_get_agent_state(ctx)`** — Reads the last recorded state from the session's event history. Returns `None` on first run (no prior state).
- **`_create_agent_state_event(ctx, step)`** — Emits a state checkpoint event marking the beginning of a step. This is what ADK reads on resume to determine where to restart.
- **Step ordering via Enum comparison** — The `WorkflowStep` enum's ordering (`CLASSIFY < PROCESS < SUMMARIZE`) lets us use `<=` comparisons to determine which steps to run. Steps at or after the current checkpoint execute; steps before it are skipped.

This pattern gives us full control over checkpoint granularity. We can checkpoint at the sub-agent level (as shown), at individual tool calls, or even at specific lines of logic within a step.

---

## Resume vs. Retry

ADK supports both **resume** (workflow-level recovery) and **retry** (operation-level recovery via callbacks). These solve different problems and can be used together:

| Feature | Resume | Retry (Callback) |
|---|---|---|
| **Scope** | Workflow-level | Individual operation |
| **Trigger** | Process restart | Immediate on failure |
| **State** | Persistent checkpoints | In-memory |
| **Cost** | Skips completed work | Re-runs failed operation only |
| **Use case** | Crash recovery, deployments | Transient API errors, rate limits |

**When to use resume**: Our process crashed, the server restarted, or we're deploying a new version mid-pipeline. We need to recover across process boundaries.

**When to use retry**: An API returned a 429 (rate limit) or 503 (service unavailable). We want to wait and retry immediately without restarting the process.

**Using both together** gives us defense in depth — retries handle transient errors within a running process, and resume handles catastrophic failures that kill the process:

```python
from google.adk.agents import LlmAgent
from google.adk.app import App, ResumabilityConfig

# Agent with retry for transient failures
step3 = LlmAgent(
    name="transform",
    instruction="Transform records...",
    # Retry handles transient errors within the process
)

pipeline = SequentialAgent(
    name="data_pipeline",
    sub_agents=[step1, step2, step3, step4, step5],
)

# Resume handles process-level crashes
app = App(
    agent=pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
    session_service=DatabaseSessionService(db_url="..."),
)
```

```text
Expected: Transient errors trigger retries within the running process. If the
process crashes entirely, resume picks up from the last completed checkpoint
on restart.
```

---

## Important Caveats

Before enabling resumability in production, we need to understand these critical limitations:

### Tools Always Re-Execute

When resuming, tools within a re-run step always execute at least once — even if they ran successfully before the crash. This means **side effects will repeat**. If a tool sends an email, it sends it again. If it writes to a database, it writes again.

**Solution**: Make all tools **idempotent**. Use unique request IDs, upserts instead of inserts, and check-before-act patterns.

```python
async def idempotent_save(ctx, data: dict) -> str:
    """Save data with idempotency key to prevent duplicates."""
    idempotency_key = f"{ctx.session.id}:{ctx.invocation_id}"
    existing = await db.find_one({"idempotency_key": idempotency_key})
    if existing:
        return f"Already saved (idempotent skip): {existing['id']}"
    data["idempotency_key"] = idempotency_key
    result = await db.insert_one(data)
    return f"Saved: {result.inserted_id}"
```

```text
Expected: On first execution, inserts the record. On resume (re-execution),
detects the existing idempotency key and skips the insert.
```

### LLM Calls Are Not Cached

LLM responses are not stored as part of the checkpoint. On resume, any `LlmAgent` step that re-runs will make a fresh LLM call, potentially returning a different response. If determinism matters, consider caching LLM responses externally.

### InMemorySessionService Loses State

The default `InMemorySessionService` stores everything in process memory. When the process dies, all checkpoint data is lost. Always use `DatabaseSessionService` or a custom persistent backend for resumable workflows.

---

## Best Practices

| Practice | Description |
|---|---|
| **Use persistent sessions** | Always pair resumability with `DatabaseSessionService` or equivalent |
| **Make tools idempotent** | Use idempotency keys, upserts, and check-before-act patterns |
| **Checkpoint at meaningful boundaries** | Checkpoint after expensive operations, not trivial ones |
| **Test resume paths explicitly** | Simulate failures at each step and verify correct resume behavior |
| **Combine resume with retry** | Use retry for transient errors, resume for process crashes |
| **Log checkpoint events** | Emit structured logs at each checkpoint for observability |
| **Keep step granularity balanced** | Too many steps adds overhead; too few reduces resume benefit |
| **Version your agent state** | If `WorkflowStep` enum changes between deploys, handle migration |

---

## Common Pitfalls

| | Pitfall | Fix |
|---|---|---|
| ❌ | Using `InMemorySessionService` with resume — state lost on crash | ✅ Use `DatabaseSessionService` for persistent checkpoints |
| ❌ | Non-idempotent tools that duplicate side effects on resume | ✅ Add idempotency keys and check-before-act guards |
| ❌ | Assuming LLM responses are cached across resume | ✅ Accept non-determinism or cache LLM responses externally |
| ❌ | Expecting `ParallelAgent` to skip completed branches | ✅ Understand that parallel execution is atomic — all branches re-run |
| ❌ | No error handling around `_get_agent_state()` returning `None` | ✅ Always default to the first step when state is `None` |
| ❌ | Changing the step enum between deploys without migration | ✅ Version your agent state or handle unknown steps gracefully |
| ❌ | Checkpointing after every trivial operation | ✅ Checkpoint at meaningful, expensive boundaries only |

---

## Hands-on Exercise

**Build a resumable document processing pipeline.**

Create a `SequentialAgent` with four steps:

1. **Fetch** — Simulates fetching a document from an API (use a mock tool)
2. **Parse** — Extracts key fields from the document
3. **Analyze** — Uses an LLM to analyze the extracted data
4. **Store** — Saves the analysis results (with idempotency)

Requirements:
- Enable resumability via `ResumabilityConfig`
- Use `DatabaseSessionService` (SQLite is fine for testing)
- Make the "Store" tool idempotent
- Simulate a failure at step 3 on the first run, then resume successfully

<details>
<summary>💡 Hints</summary>

1. Use `DatabaseSessionService(db_url="sqlite:///sessions.db")` for a lightweight persistent backend.
2. For simulating failure, use a counter in session state: fail on the first attempt, succeed on the second.
3. The idempotent store tool should check for an existing record by a unique key before inserting.
4. Wrap everything in an `App` with `ResumabilityConfig(is_resumable=True)`.

</details>

<details>
<summary>✅ Solution</summary>

```python
import asyncio
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.app import App, ResumabilityConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import FunctionTool

# --- Tools ---
async def fetch_document(ctx) -> str:
    """Simulate fetching a document from an external API."""
    ctx.state["document"] = {
        "title": "Q4 Report",
        "content": "Revenue grew 15% year over year...",
    }
    return "Document fetched successfully."

async def parse_document(ctx) -> str:
    """Extract key fields from the fetched document."""
    doc = ctx.state.get("document", {})
    ctx.state["parsed"] = {
        "title": doc.get("title"),
        "word_count": len(doc.get("content", "").split()),
    }
    return f"Parsed: {ctx.state['parsed']}"

async def store_results(ctx, analysis: str) -> str:
    """Idempotent store — checks before writing."""
    idempotency_key = f"analysis:{ctx.session.id}"
    # In production, check a real database
    if ctx.state.get(f"stored:{idempotency_key}"):
        return "Already stored (idempotent skip)."
    ctx.state[f"stored:{idempotency_key}"] = True
    ctx.state["final_analysis"] = analysis
    return f"Stored analysis: {analysis[:100]}..."

# --- Agents ---
step1 = LlmAgent(
    name="fetch",
    instruction="Fetch the document using the fetch_document tool.",
    tools=[FunctionTool(fetch_document)],
)

step2 = LlmAgent(
    name="parse",
    instruction="Parse the document using the parse_document tool.",
    tools=[FunctionTool(parse_document)],
)

step3 = LlmAgent(
    name="analyze",
    instruction=(
        "Analyze the parsed document data in state and provide "
        "a summary with key insights."
    ),
)

step4 = LlmAgent(
    name="store",
    instruction="Store the analysis results using the store_results tool.",
    tools=[FunctionTool(store_results)],
)

# --- Pipeline ---
pipeline = SequentialAgent(
    name="doc_pipeline",
    sub_agents=[step1, step2, step3, step4],
)

# --- App with Resumability ---
session_service = DatabaseSessionService(
    db_url="sqlite:///sessions.db"
)

app = App(
    agent=pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
    session_service=session_service,
)

# Run the app (use ADK CLI or call app.run())
if __name__ == "__main__":
    asyncio.run(app.run())
```

```text
Expected first run (simulated failure at step 3):
  Step 1 (fetch): ✅ Completed
  Step 2 (parse): ✅ Completed
  Step 3 (analyze): ❌ Failed (timeout / error)

Expected resume run:
  Step 1 (fetch): ⏭️ Skipped
  Step 2 (parse): ⏭️ Skipped
  Step 3 (analyze): 🔄 Retried → ✅ Completed
  Step 4 (store): ▶️ Completed
  Pipeline: ✅ Done
```

</details>

---

## Summary

✅ Long-running agent workflows are fragile — crashes, timeouts, and API errors can halt multi-step pipelines mid-execution.

✅ ADK's `ResumabilityConfig` (v1.14.0+) enables workflows to pick up where they left off by tracking completed steps via event checkpoints.

✅ `SequentialAgent` and `LoopAgent` support fine-grained resume; `ParallelAgent` re-runs atomically; `LlmAgent` always re-executes.

✅ Persistent session storage (`DatabaseSessionService`) is mandatory — `InMemorySessionService` loses all checkpoint data on restart.

✅ Custom agents implement resumability through `_get_agent_state()` and `_create_agent_state_event()` with ordered enum steps.

✅ All tools must be idempotent when using resumability, since tools re-execute on resume even if they ran before the failure.

✅ Resume handles process-level crashes; retry callbacks handle transient errors — use both together for robust production systems.

---

## Navigation

← Previous: [Plugins System](./14-plugins-system.md) | Next: [Third-Party Tool Integrations](./16-third-party-tool-integrations.md) →

[Back to Google ADK Overview](./00-google-agent-development-kit.md)

---

## Further Reading

- [Google ADK Documentation — Resumability](https://google.github.io/adk-docs/agents/resumability/)
- [ADK Sessions and State Management](https://google.github.io/adk-docs/sessions/)
- [DatabaseSessionService Reference](https://google.github.io/adk-docs/sessions/database/)
- [Building Reliable Agent Pipelines](https://google.github.io/adk-docs/guides/reliable-pipelines/)
- [Idempotency Patterns for Agent Tools](https://cloud.google.com/blog/topics/developers-practitioners/idempotency-patterns)

<!-- Sources:
  - Google ADK Documentation: https://google.github.io/adk-docs/
  - ADK Resumability Guide: https://google.github.io/adk-docs/agents/resumability/
  - ADK GitHub Repository: https://github.com/google/adk-python
  - ADK Sessions Documentation: https://google.github.io/adk-docs/sessions/
  - ADK v1.14.0 Release Notes (resumability feature)
-->
