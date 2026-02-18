---
title: "Streaming in LangGraph"
---

# Streaming in LangGraph

## Introduction

When an agent takes seconds or minutes to complete, users need feedback that something is happening. Streaming provides real-time visibility into graph execution — from high-level node transitions to individual LLM tokens. LangGraph supports multiple streaming modes that can be combined to give your application exactly the level of detail it needs.

This lesson covers all five streaming modes, how to stream LLM tokens from any point in your graph, how to emit custom data, and how to stream from subgraphs.

### What We'll Cover

- The five stream modes: `values`, `updates`, `messages`, `custom`, `debug`
- Streaming LLM tokens from nodes and tools
- Emitting custom data with `get_stream_writer()`
- Combining multiple stream modes
- Streaming from subgraphs
- Filtering streams by node or LLM invocation

### Prerequisites

- Completed [Core Concepts](./01-core-concepts.md) and [Graph Construction Patterns](./02-graph-construction-patterns.md)
- Understanding of async Python (`async for`, `await`)
- A LangChain-compatible LLM set up for streaming

---

## Stream Modes Overview

LangGraph provides five stream modes, each offering a different level of detail:

| Mode | What It Streams | Best For |
|------|----------------|----------|
| `values` | Full state after each node | Monitoring complete state changes |
| `updates` | Only the state delta from each node | Efficient progress tracking |
| `messages` | LLM tokens as `(chunk, metadata)` tuples | Real-time chat UX |
| `custom` | User-defined data from inside nodes | Progress bars, status updates |
| `debug` | Maximum detail for every step | Development and debugging |

---

## Streaming State Updates

### Updates Mode

The most common mode — streams only what changed after each node:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream({"topic": "ice cream"}, stream_mode="updates"):
    print(chunk)
```

**Output:**
```
{'refine_topic': {'topic': 'ice cream and cats'}}
{'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
```

Each chunk is a dict with the node name as key and the state update as value.

### Values Mode

Streams the complete state after each step:

```python
for chunk in graph.stream({"topic": "ice cream"}, stream_mode="values"):
    print(chunk)
```

**Output:**
```
{'topic': 'ice cream'}
{'topic': 'ice cream and cats'}
{'topic': 'ice cream and cats', 'joke': 'This is a joke about ice cream and cats'}
```

> **💡 Tip:** Use `updates` mode in production for efficiency. Use `values` mode during development when you need to see the full state at every step.

---

## Streaming LLM Tokens

The `messages` mode streams individual LLM tokens as they're generated, along with metadata about where they came from:

```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class State:
    topic: str
    joke: str = ""


model = init_chat_model(model="gpt-4.1-mini")


def call_model(state: State):
    response = model.invoke(
        [{"role": "user", "content": f"Tell a short joke about {state.topic}"}]
    )
    return {"joke": response.content}


graph = (
    StateGraph(State)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# Stream individual tokens
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

**Output:**
```
Why|did|the|ice|cream|truck|break|down|?|Because|of|the|rocky|road|!|
```

### Filtering by Node

Stream tokens from specific nodes only:

```python
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    # Only show tokens from the "generate" node
    if msg.content and metadata["langgraph_node"] == "generate":
        print(msg.content, end="", flush=True)
```

### Filtering by LLM Tag

Associate tags with LLM invocations to filter specific model calls:

```python
from langchain.chat_models import init_chat_model

# Tag different models
joke_model = init_chat_model(model="gpt-4.1-mini", tags=["joke"])
poem_model = init_chat_model(model="gpt-4.1-mini", tags=["poem"])

# Filter by tag during streaming
async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="", flush=True)
```

> **🤖 AI Context:** Token-level streaming is critical for chat applications. Without it, users wait for the entire LLM response before seeing anything — with it, text appears progressively, just like ChatGPT or Claude's interface.

---

## Custom Data Streaming

Send arbitrary data from inside your nodes using `get_stream_writer()`:

```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    answer: str


def process(state: State):
    writer = get_stream_writer()
    
    # Emit progress updates
    writer({"status": "Starting analysis..."})
    writer({"progress": 0.5, "status": "Halfway done..."})
    writer({"progress": 1.0, "status": "Complete!"})
    
    return {"answer": f"Processed: {state['query']}"}


graph = (
    StateGraph(State)
    .add_node(process)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile()
)

for chunk in graph.stream({"query": "analyze this"}, stream_mode="custom"):
    print(chunk)
```

**Output:**
```
{'status': 'Starting analysis...'}
{'progress': 0.5, 'status': 'Halfway done...'}
{'progress': 1.0, 'status': 'Complete!'}
```

### Custom Streaming from Tools

You can also stream custom data from inside tool functions:

```python
from langchain_core.tools import tool
from langgraph.config import get_stream_writer


@tool
def search_documents(query: str) -> str:
    """Search documents with progress updates."""
    writer = get_stream_writer()
    
    writer({"step": "Searching index..."})
    # ... search logic ...
    
    writer({"step": "Ranking results..."})
    # ... ranking logic ...
    
    writer({"step": "Done!", "results_found": 42})
    return "Found 42 matching documents"
```

### Using with Non-LangChain LLMs

`custom` mode lets you stream from any LLM API, even if it doesn't implement the LangChain interface:

```python
from langgraph.config import get_stream_writer


def call_custom_llm(state):
    writer = get_stream_writer()
    
    # Use your own streaming client
    for chunk in your_custom_streaming_client(state["topic"]):
        writer({"custom_llm_chunk": chunk})
    
    return {"result": "completed"}


# Stream the custom data
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
```

---

## Combining Multiple Stream Modes

Pass a list of modes to receive all of them simultaneously:

```python
for mode, chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
):
    if mode == "updates":
        print(f"State update: {chunk}")
    elif mode == "custom":
        print(f"Custom data: {chunk}")
```

**Output:**
```
Custom data: {'status': 'Starting...'}
Custom data: {'progress': 1.0}
State update: {'process': {'answer': 'Processed: analyze this'}}
```

When using multiple modes, each streamed item is a tuple of `(mode, chunk)`.

---

## Streaming from Subgraphs

To include outputs from nested subgraphs, set `subgraphs=True`:

```python
for chunk in graph.stream(
    {"foo": "bar"},
    subgraphs=True,
    stream_mode="updates",
):
    print(chunk)
```

The output includes a namespace tuple showing the path through the graph:

```python
# Parent graph output
((), {'parent_node': {'value': 'updated'}})

# Subgraph output (shows the path: parent_node -> child_node)
(('parent_node:task_id',), {'child_node': {'value': 'from subgraph'}})
```

### HITL Streaming with Subgraphs

Combine streaming with human-in-the-loop interrupts for interactive agents:

```python
async for metadata, mode, chunk in graph.astream(
    initial_input,
    stream_mode=["messages", "updates"],
    subgraphs=True,
    config=config,
):
    if mode == "messages":
        msg, _ = chunk
        if hasattr(msg, "content") and msg.content:
            print(msg.content, end="", flush=True)
    elif mode == "updates":
        if "__interrupt__" in chunk:
            interrupt_info = chunk["__interrupt__"][0].value
            user_response = input(f"\n{interrupt_info}: ")
            initial_input = Command(resume=user_response)
            break
```

---

## Debug Mode

The `debug` mode streams maximum information — useful during development:

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
):
    print(chunk)
```

This includes node names, full state, timestamps, and internal metadata for every step.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `updates` mode for production | More efficient than `values` — only sends what changed |
| Use `messages` for chat UX | Token-level streaming creates responsive, ChatGPT-like interfaces |
| Use `custom` for progress indicators | Provides structured progress data for complex operations |
| Combine modes when needed | `["messages", "updates"]` gives both token streaming and node tracking |
| Filter by node or tag | Prevents noise from irrelevant LLM calls in multi-node graphs |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `stream_mode="messages"` with non-LangChain LLMs | Use `custom` mode with `get_stream_writer()` instead |
| Forgetting `subgraphs=True` for nested graphs | Subgraph outputs are hidden by default — enable explicitly |
| Not handling the tuple format with multiple modes | With multiple modes, each item is `(mode, chunk)`, not just `chunk` |
| Using `get_stream_writer()` in async Python < 3.11 | Pass a `writer` parameter directly instead of using `get_stream_writer()` |
| Streaming without checking `msg.content` | Filter out empty content chunks to avoid printing empty strings |

---

## Hands-on Exercise

### Your Task

Build a streaming research agent that provides real-time progress updates.

### Requirements

1. Create a three-node graph: `plan`, `research`, `summarize`
2. The `plan` node emits custom data: `{"phase": "planning", "steps": 3}`
3. The `research` node emits progress updates: `{"phase": "researching", "progress": 0.33}`, etc.
4. The `summarize` node produces the final result
5. Stream with `["updates", "custom"]` modes combined
6. Print custom data as progress bars and updates as state changes

### Expected Result

```
[Custom] Phase: planning, Steps: 3
[Custom] Phase: researching, Progress: 33%
[Custom] Phase: researching, Progress: 67%
[Custom] Phase: researching, Progress: 100%
[Update] summarize: {'result': 'Research complete!'}
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `get_stream_writer()` in each node that needs custom output
- Stream with `stream_mode=["updates", "custom"]`
- Each streamed item is `(mode, chunk)` — check `mode` to distinguish
- Format progress as a percentage: `f"{progress * 100:.0f}%"`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    topic: str
    findings: str
    result: str


def plan(state: State):
    writer = get_stream_writer()
    writer({"phase": "planning", "steps": 3})
    return {"findings": ""}


def research(state: State):
    writer = get_stream_writer()
    sources = ["Wikipedia", "ArXiv", "News"]
    results = []
    for i, source in enumerate(sources, 1):
        writer({"phase": "researching", "progress": i / len(sources)})
        results.append(f"Data from {source}")
    return {"findings": "; ".join(results)}


def summarize(state: State):
    return {"result": f"Research on '{state['topic']}': {state['findings']}"}


graph = (
    StateGraph(State)
    .add_node(plan)
    .add_node(research)
    .add_node(summarize)
    .add_edge(START, "plan")
    .add_edge("plan", "research")
    .add_edge("research", "summarize")
    .add_edge("summarize", END)
    .compile()
)

for mode, chunk in graph.stream({"topic": "LangGraph"}, stream_mode=["updates", "custom"]):
    if mode == "custom":
        phase = chunk.get("phase", "")
        progress = chunk.get("progress")
        if progress:
            print(f"[Custom] Phase: {phase}, Progress: {progress * 100:.0f}%")
        else:
            print(f"[Custom] Phase: {phase}, Steps: {chunk.get('steps')}")
    elif mode == "updates":
        for node_name, update in chunk.items():
            if "result" in update:
                print(f"[Update] {node_name}: {update}")
```

**Output:**
```
[Custom] Phase: planning, Steps: 3
[Custom] Phase: researching, Progress: 33%
[Custom] Phase: researching, Progress: 67%
[Custom] Phase: researching, Progress: 100%
[Update] summarize: {'result': "Research on 'LangGraph': Data from Wikipedia; Data from ArXiv; Data from News"}
```

</details>

### Bonus Challenges

- [ ] Add LLM token streaming with `messages` mode alongside `custom` and `updates`
- [ ] Stream from a subgraph using `subgraphs=True`
- [ ] Build an async version using `astream()` and `async for`

---

## Summary

✅ Five stream modes (`values`, `updates`, `messages`, `custom`, `debug`) cover every use case

✅ `messages` mode streams LLM tokens for real-time chat interfaces

✅ `custom` mode with `get_stream_writer()` lets you emit arbitrary progress data

✅ Multiple modes can be combined — items arrive as `(mode, chunk)` tuples

✅ `subgraphs=True` includes nested graph outputs in the stream

**Next:** [Subgraphs and Composition](./08-subgraphs-and-composition.md)

---

## Further Reading

- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — Complete streaming documentation
- [LangGraph Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) — Streaming from subgraphs
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) — HITL with streaming

*Back to [LangGraph Agent Orchestration](./00-langgraph-agent-orchestration.md)*

<!-- 
Sources Consulted:
- LangGraph Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- LangGraph Subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- LangGraph Interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
-->
