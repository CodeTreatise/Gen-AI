---
title: "State Management in Flows"
---

# State Management in Flows

## Introduction

State is what makes Flows more than just a chain of function calls. It lets methods share data without passing everything through return values, persists across flow restarts, and provides a clear data model for complex workflows.

CrewAI Flows offer two approaches to state: **unstructured** (flexible dictionary) and **structured** (typed Pydantic models). This lesson covers both, along with the `@persist` decorator for saving state to disk.

### What We'll Cover

- Unstructured state with dictionaries
- Structured state with Pydantic models
- Automatic UUID generation
- The `@persist` decorator and `SQLiteFlowPersistence`
- Choosing between state approaches

### Prerequisites

- Completed [Flow Decorators In-Depth](./03-flow-decorators-in-depth.md)
- Familiarity with Pydantic models

---

## Unstructured State

By default, every Flow has a `self.state` dictionary that any method can read and write:

```python
from crewai.flow.flow import Flow, start, listen


class SimpleStateFlow(Flow):
    
    @start()
    def collect_input(self):
        self.state["topic"] = "AI Agents"
        self.state["word_count"] = 0
        print(f"📝 Topic set: {self.state['topic']}")
    
    @listen(collect_input)
    def research(self):
        findings = ["Multi-agent systems growing", "Tool use is key", "RAG is standard"]
        self.state["findings"] = findings
        self.state["word_count"] += 50
        print(f"🔍 Found {len(findings)} items")
    
    @listen(research)
    def summarize(self):
        topic = self.state["topic"]
        findings = self.state["findings"]
        self.state["word_count"] += 100
        summary = f"Report on {topic}: {len(findings)} key findings"
        print(f"📋 {summary} ({self.state['word_count']} words)")
        return summary


flow = SimpleStateFlow()
result = flow.kickoff()
```

**Output:**
```
📝 Topic set: AI Agents
🔍 Found 3 items
📋 Report on AI Agents: 3 key findings (150 words)
```

### Unstructured State Characteristics

| Feature | Detail |
|---------|--------|
| Type | `dict` — any key, any value |
| Auto-generated `id` | Every state gets a UUID automatically |
| Access | `self.state["key"]` or `self.state.get("key", default)` |
| Flexibility | Add any key at any time |
| Validation | None — no type checking |

```python
class ShowIdFlow(Flow):
    
    @start()
    def show_auto_id(self):
        print(f"Flow ID: {self.state['id']}")
        # Output: Flow ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

> **Note:** The auto-generated `id` is a UUID that uniquely identifies each flow execution. This is useful for logging, persistence, and debugging.

---

## Structured State

For production flows, **structured state** with Pydantic models provides type safety, validation, and IDE autocompletion:

```python
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel


class ResearchState(BaseModel):
    topic: str = ""
    findings: list[str] = []
    word_count: int = 0
    is_complete: bool = False


class StructuredFlow(Flow[ResearchState]):
    
    @start()
    def collect_input(self):
        self.state.topic = "AI Agents"
        print(f"📝 Topic: {self.state.topic}")
    
    @listen(collect_input)
    def research(self):
        self.state.findings = [
            "Multi-agent frameworks maturing",
            "Tool use becoming standard",
            "RAG integrated by default",
        ]
        self.state.word_count += 75
        print(f"🔍 Found {len(self.state.findings)} items")
    
    @listen(research)
    def finalize(self):
        self.state.is_complete = True
        self.state.word_count += 100
        print(f"✅ Complete: {self.state.word_count} words on '{self.state.topic}'")


flow = StructuredFlow()
flow.kickoff()
```

**Output:**
```
📝 Topic: AI Agents
🔍 Found 3 items
✅ Complete: 175 words on 'AI Agents'
```

### Structured vs Unstructured Comparison

| Feature | Unstructured (`dict`) | Structured (Pydantic) |
|---------|----------------------|----------------------|
| Definition | Automatic | Define a `BaseModel` subclass |
| Type safety | ❌ None | ✅ Full type checking |
| IDE autocomplete | ❌ No | ✅ Yes (`self.state.topic`) |
| Default values | Manual | Declared in model |
| Validation | None | Pydantic validation |
| Access syntax | `self.state["key"]` | `self.state.key` |
| Auto UUID | ✅ `self.state["id"]` | ✅ `self.state.id` |
| Best for | Prototyping, simple flows | Production, complex flows |

### State Model with Validation

```python
from pydantic import BaseModel, Field, field_validator


class ContentState(BaseModel):
    title: str = ""
    body: str = ""
    word_count: int = Field(default=0, ge=0)
    tags: list[str] = []
    status: str = "draft"
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed = {"draft", "review", "published"}
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return v


class ContentFlow(Flow[ContentState]):
    
    @start()
    def create_draft(self):
        self.state.title = "Understanding AI Agents"
        self.state.status = "draft"
        self.state.tags = ["AI", "agents", "tutorial"]
```

> **🤖 AI Context:** Structured state is similar to how LangGraph uses `TypedDict` for its state schema. Both frameworks enforce that state follows a defined structure, but CrewAI uses Pydantic while LangGraph uses Python's type system.

---

## Initializing State

You can pass initial state values when creating a flow:

### Unstructured

```python
class MyFlow(Flow):
    @start()
    def begin(self):
        print(f"Starting with: {self.state['user_input']}")


flow = MyFlow()
flow.kickoff(inputs={"user_input": "Build me an AI agent"})
```

### Structured

```python
class MyState(BaseModel):
    user_input: str = ""
    max_iterations: int = 5


class MyFlow(Flow[MyState]):
    @start()
    def begin(self):
        print(f"Input: {self.state.user_input}")
        print(f"Max iterations: {self.state.max_iterations}")


flow = MyFlow()
flow.kickoff(inputs={"user_input": "Analyze this dataset", "max_iterations": 10})
```

**Output:**
```
Input: Analyze this dataset
Max iterations: 10
```

---

## The @persist Decorator

The `@persist` decorator saves flow state to disk, allowing flows to be resumed after interruptions:

### Class-Level Persistence

Apply `@persist` to the entire flow class to save state after every method:

```python
from crewai.flow.flow import Flow, start, listen, persist


@persist  # Saves state after every method execution
class PersistentFlow(Flow):
    
    @start()
    def step_one(self):
        self.state["progress"] = "step_one_complete"
        print("Step 1 done")
    
    @listen(step_one)
    def step_two(self):
        self.state["progress"] = "step_two_complete"
        print("Step 2 done")
```

### Method-Level Persistence

Apply `@persist` to specific methods to save state only at critical checkpoints:

```python
from crewai.flow.flow import Flow, start, listen, persist


class SelectivePersistFlow(Flow):
    
    @start()
    def quick_setup(self):
        self.state["data"] = "initial"
        print("Setup complete (not persisted)")
    
    @persist  # Only this step saves to disk
    @listen(quick_setup)
    def expensive_computation(self):
        self.state["result"] = "computed_value"
        print("Computation complete (persisted!)")
    
    @listen(expensive_computation)
    def final_step(self):
        print(f"Using: {self.state['result']}")
```

### SQLiteFlowPersistence

By default, `@persist` uses `SQLiteFlowPersistence`, which stores state in a local SQLite database:

```python
from crewai.flow.persistence import SQLiteFlowPersistence

# Custom database path
@persist(SQLiteFlowPersistence(db_path="my_flow_state.db"))
class CustomPersistFlow(Flow):
    @start()
    def begin(self):
        self.state["status"] = "running"
```

| Setting | Default |
|---------|---------|
| Storage | SQLite database file |
| Location | Current working directory |
| File name | `crewai_flows.db` (default) |
| Custom path | `SQLiteFlowPersistence(db_path="path/to/db")` |

### When to Use Persistence

| Scenario | Persist? |
|----------|----------|
| Long-running flows (minutes to hours) | ✅ Yes |
| Flows with expensive API calls | ✅ Yes |
| Flows requiring human approval mid-way | ✅ Yes |
| Quick prototyping / development | ❌ No |
| Stateless, idempotent flows | ❌ No |

---

## State Patterns

### Pattern 1: Accumulating Results

```python
class AccumulatorFlow(Flow):
    
    @start()
    def initialize(self):
        self.state["results"] = []
    
    @listen(initialize)
    def gather_a(self):
        self.state["results"].append("Result from A")
    
    @listen(initialize)
    def gather_b(self):
        self.state["results"].append("Result from B")
    
    @listen(and_(gather_a, gather_b))
    def summarize(self):
        print(f"All results: {self.state['results']}")
```

### Pattern 2: Progress Tracking

```python
class ProgressState(BaseModel):
    current_step: int = 0
    total_steps: int = 5
    errors: list[str] = []
    
    @property
    def progress_pct(self) -> float:
        return (self.current_step / self.total_steps) * 100


class TrackedFlow(Flow[ProgressState]):
    
    @start()
    def step(self):
        self.state.current_step += 1
        print(f"Progress: {self.state.progress_pct:.0f}%")
```

### Pattern 3: Configuration State

```python
class PipelineConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    output_format: str = "markdown"
    debug: bool = False


class ConfigurableFlow(Flow[PipelineConfig]):
    
    @start()
    def process(self):
        print(f"Using {self.state.model} at temp {self.state.temperature}")


# Override defaults at runtime
flow = ConfigurableFlow()
flow.kickoff(inputs={"model": "gpt-4o", "temperature": 0.2, "debug": True})
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use structured state for production flows | Type safety catches bugs early |
| Keep state models flat (avoid deep nesting) | Easier to persist, debug, and serialize |
| Set sensible defaults in your state model | Flows work out of the box without explicit initialization |
| Use `@persist` for flows with expensive steps | Avoid re-running costly API calls after failures |
| Access state with attribute syntax when structured | `self.state.topic` not `self.state["topic"]` |
| Use method-level `@persist` for selective checkpoints | Avoid unnecessary I/O on cheap operations |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Modifying state in parallel methods without coordination | Use `and_()` to synchronize, or use separate state keys |
| Forgetting that unstructured state has no validation | Switch to structured state for data integrity |
| Storing large objects (files, images) in state | Store file paths or references, not raw data |
| Not handling missing state keys in unstructured state | Use `self.state.get("key", default)` |
| Using `@persist` on every method in development | Only persist at critical checkpoints to avoid slowdowns |

---

## Hands-on Exercise

### Your Task

Build a content pipeline flow with structured state that tracks progress through multiple stages.

### Requirements

1. Define a `PipelineState` Pydantic model with: `topic` (str), `outline` (list[str]), `draft` (str), `word_count` (int), `stage` (str, default "init")
2. Create a `ContentPipeline(Flow[PipelineState])` with four stages:
   - `start_pipeline` (`@start`): Set topic and stage to "outline"
   - `create_outline` (`@listen`): Add 3 outline items, set stage to "drafting"
   - `write_draft` (`@listen`): Create a draft string from the outline, update word count, set stage to "review"
   - `review` (`@listen`): Print the final state summary
3. Initialize the flow with `inputs={"topic": "AI Agents in Production"}`

### Expected Result

```
Stage: outline — Topic: AI Agents in Production
Stage: drafting — 3 outline items
Stage: review — Draft: 45 words
✅ Pipeline complete for 'AI Agents in Production'
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Flow[PipelineState]` as the generic type parameter
- Each method updates `self.state.stage` to track progress
- Calculate `word_count` with `len(self.state.draft.split())`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel


class PipelineState(BaseModel):
    topic: str = ""
    outline: list[str] = []
    draft: str = ""
    word_count: int = 0
    stage: str = "init"


class ContentPipeline(Flow[PipelineState]):
    
    @start()
    def start_pipeline(self):
        self.state.stage = "outline"
        print(f"Stage: {self.state.stage} — Topic: {self.state.topic}")
    
    @listen(start_pipeline)
    def create_outline(self):
        self.state.outline = [
            f"Introduction to {self.state.topic}",
            f"Key challenges in {self.state.topic}",
            f"Best practices for {self.state.topic}",
        ]
        self.state.stage = "drafting"
        print(f"Stage: {self.state.stage} — {len(self.state.outline)} outline items")
    
    @listen(create_outline)
    def write_draft(self):
        sections = [f"## {item}\nContent about {item}." for item in self.state.outline]
        self.state.draft = "\n\n".join(sections)
        self.state.word_count = len(self.state.draft.split())
        self.state.stage = "review"
        print(f"Stage: {self.state.stage} — Draft: {self.state.word_count} words")
    
    @listen(write_draft)
    def review(self):
        print(f"✅ Pipeline complete for '{self.state.topic}'")
        return self.state.draft


flow = ContentPipeline()
result = flow.kickoff(inputs={"topic": "AI Agents in Production"})
```

</details>

### Bonus Challenges

- [ ] Add `@persist` to the `write_draft` method (the most expensive step)
- [ ] Add a `Pydantic` field validator to ensure `stage` is one of: `init`, `outline`, `drafting`, `review`, `complete`
- [ ] Extend the state with `errors: list[str]` and add error tracking

---

## Summary

✅ **Unstructured state** (`dict`) is flexible for prototyping — any key, any value

✅ **Structured state** (Pydantic `BaseModel`) provides type safety, validation, and IDE support

✅ Every flow state gets an automatic **UUID** for tracking and persistence

✅ `@persist` saves state to SQLite — use at class level for full persistence or method level for checkpoints

✅ Pass initial values via `kickoff(inputs={...})` to configure flows at runtime

**Next:** [Crews Within Flows](./05-crews-within-flows.md)

---

## Further Reading

- [CrewAI Flows Documentation](https://docs.crewai.com/concepts/flows) — State management and persistence
- [Pydantic Documentation](https://docs.pydantic.dev/) — Model validation and serialization

*Back to [CrewAI with Flows Overview](./00-crewai-with-flows.md)*

<!-- 
Sources Consulted:
- CrewAI Flows: https://docs.crewai.com/concepts/flows
- Pydantic BaseModel: https://docs.pydantic.dev/latest/concepts/models/
-->
