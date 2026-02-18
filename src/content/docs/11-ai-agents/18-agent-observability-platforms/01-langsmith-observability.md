---
title: "LangSmith Observability"
---

# LangSmith Observability

## Introduction

LangSmith is LangChain's full-lifecycle platform for developing, debugging, and monitoring LLM applications. It provides **end-to-end trace visualization**, cost tracking, evaluation datasets, and prompt management — all in one place. Whether you're using LangChain, LangGraph, or raw OpenAI calls, LangSmith captures every step your agent takes and presents it as an explorable run tree.

LangSmith is framework-agnostic: you can use it with or without LangChain's open-source libraries. This makes it a versatile choice for any Python or TypeScript agent project.

### What we'll cover

- Setting up LangSmith tracing
- Wrapping OpenAI calls with `wrap_openai`
- Using the `@traceable` decorator for custom functions
- Exploring traces and run trees in the dashboard
- Token and cost tracking
- Evaluation datasets and prompt versioning

### Prerequisites

- Python 3.10+ installed
- An OpenAI API key
- A free LangSmith account at [smith.langchain.com](https://smith.langchain.com/)
- Familiarity with agent fundamentals (Unit 11, Lessons 1–5)

---

## Setting up LangSmith

LangSmith requires minimal configuration. We set environment variables and install the SDK — tracing starts automatically.

### Installation

```bash
pip install -U langsmith openai
```

### Environment configuration

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

> **Note:** Sign up at [smith.langchain.com](https://smith.langchain.com/) for free. Create an API key under Settings → API Keys.

---

## Tracing LLM calls with `wrap_openai`

The fastest way to start tracing is wrapping your OpenAI client. LangSmith captures every `chat.completions.create()` call automatically.

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

# Wrap the OpenAI client — all subsequent calls are traced
client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is observability?"},
    ],
)
print(response.choices[0].message.content)
```

**Output:**
```
Observability is the ability to understand the internal state of a system
by examining its outputs, such as logs, metrics, and traces...
```

After running this code, navigate to your LangSmith project in the UI. You'll see a trace containing the LLM call with full input/output, model name, token counts, and latency.

---

## Tracing entire applications with `@traceable`

The `@traceable` decorator wraps any Python function so it appears as a span in your trace. We use this to capture the full pipeline — retrieval, processing, and LLM calls — in a single trace.

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

def retriever(query: str) -> list[str]:
    """Simulate document retrieval."""
    return [
        "LangSmith provides trace visualization and evaluation tools.",
        "Traces capture LLM calls, tool invocations, and custom spans.",
    ]

@traceable
def rag_agent(question: str) -> str:
    """RAG agent that retrieves context and generates an answer."""
    docs = retriever(question)
    system_message = (
        "Answer using only the provided context:\n"
        + "\n".join(docs)
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    answer = rag_agent("What does LangSmith do?")
    print(answer)
```

**Output:**
```
LangSmith provides trace visualization and evaluation tools. It captures
LLM calls, tool invocations, and custom spans for debugging.
```

In the LangSmith UI, you'll see a **run tree**: the top-level `rag_agent` span contains the nested `ChatOpenAI` LLM call span, with timing and token counts for each.

---

## Run tree exploration

The run tree is the core visualization in LangSmith. It shows every operation as a **hierarchical tree of spans**:

```
rag_agent (3.2s, $0.002)
├── retriever (0.1ms)
└── ChatOpenAI (3.1s, 847 tokens)
    ├── Input: [system + user messages]
    └── Output: "LangSmith provides..."
```

### What each span captures

| Field | Description |
|-------|-------------|
| **Name** | Function name or LLM provider |
| **Start/End time** | Precise timestamps for latency calculation |
| **Input** | Arguments passed to the function or LLM |
| **Output** | Return value or LLM response |
| **Token count** | Input + output tokens (LLM spans only) |
| **Cost** | Calculated cost based on model pricing |
| **Status** | Success, error, or in-progress |
| **Metadata** | Custom key-value pairs you attach |

### Adding metadata to traces

```python
@traceable(metadata={"version": "v2", "experiment": "rag-improved"})
def rag_agent(question: str) -> str:
    # ...same code...
    pass
```

Metadata enables **filtering and grouping** in the dashboard. Use it to track experiments, deployment versions, or user segments.

---

## Token and cost tracking

LangSmith automatically calculates token usage and cost for every LLM call. In the dashboard, you can see:

- **Per-trace cost**: Total tokens and dollars for an entire agent run
- **Per-span cost**: Cost breakdown by individual LLM call
- **Aggregate cost**: Daily/weekly/monthly totals across all traces

### Viewing costs in the UI

Navigate to your project → Traces → Click any trace. The trace header shows:
- Total tokens (input + output)
- Estimated cost (based on model pricing tables)
- Latency (total and per-span)

### Programmatic cost access

```python
from langsmith import Client

ls_client = Client()

# Get runs for a specific project
runs = ls_client.list_runs(
    project_name="my-agent-project",
    run_type="llm",
    start_time="2026-02-01",
)

total_cost = 0
for run in runs:
    if run.total_cost:
        total_cost += run.total_cost
        print(f"{run.name}: {run.total_tokens} tokens, ${run.total_cost:.4f}")

print(f"\nTotal cost: ${total_cost:.2f}")
```

**Output:**
```
ChatOpenAI: 847 tokens, $0.0021
ChatOpenAI: 1203 tokens, $0.0030
ChatOpenAI: 562 tokens, $0.0014

Total cost: $0.01
```

---

## Evaluation datasets

LangSmith datasets let you create **curated sets of inputs and expected outputs** for systematic testing. Instead of manually checking responses, you define evaluators that score agent outputs automatically.

### Creating a dataset

```python
from langsmith import Client

ls_client = Client()

# Create a dataset
dataset = ls_client.create_dataset(
    "Agent QA Benchmark",
    description="Questions to evaluate the RAG agent",
)

# Add examples
examples = [
    {
        "input": {"question": "What does LangSmith do?"},
        "output": {"answer": "LangSmith provides trace visualization and evaluation tools."},
    },
    {
        "input": {"question": "What are traces?"},
        "output": {"answer": "Traces capture the full record of an agent workflow execution."},
    },
]

for ex in examples:
    ls_client.create_example(
        inputs=ex["input"],
        outputs=ex["output"],
        dataset_id=dataset.id,
    )

print(f"Created dataset '{dataset.name}' with {len(examples)} examples")
```

**Output:**
```
Created dataset 'Agent QA Benchmark' with 2 examples
```

### Running evaluations

```python
from langsmith import Client, evaluate

ls_client = Client()

# Define an evaluator
def correctness(outputs: dict, reference_outputs: dict) -> dict:
    """Check if the agent's answer matches the expected answer."""
    predicted = outputs.get("answer", "").lower()
    expected = reference_outputs.get("answer", "").lower()
    # Simple keyword overlap check
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    overlap = len(expected_words & predicted_words) / len(expected_words)
    return {"key": "correctness", "score": overlap}

# Run evaluation
results = evaluate(
    lambda inputs: {"answer": rag_agent(inputs["question"])},
    data="Agent QA Benchmark",
    evaluators=[correctness],
    experiment_prefix="rag-v2",
)

print(f"Average correctness: {results.aggregate_metrics['correctness']:.2%}")
```

**Output:**
```
Average correctness: 85.00%
```

---

## Prompt hub and versioning

LangSmith includes a **Prompt Hub** for managing, versioning, and deploying prompts separately from your code. This lets non-engineering team members iterate on prompts while developers focus on the pipeline.

### Key features

| Feature | Benefit |
|---------|---------|
| **Version control** | Every prompt change is tracked with diffs |
| **Deployment labels** | Tag versions as "production", "staging", "experiment" |
| **A/B testing** | Serve different prompt versions to compare results |
| **Linked traces** | See which prompt version generated each trace |
| **Collaboration** | Team members edit prompts via UI without code changes |

---

## Tracing LangGraph agents

If you use LangGraph, tracing is automatic when `LANGSMITH_TRACING=true`. Every node, edge, and tool call in the graph appears in the run tree.

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(llm, tools=[])
result = agent.invoke({"messages": [("user", "What is 2 + 2?")]})
print(result["messages"][-1].content)
```

The LangSmith dashboard shows the full graph execution: which nodes ran, in what order, what each tool returned, and how the agent decided to stop.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Enable tracing from day one | Easier to debug early; traces are lightweight |
| Use `@traceable` on all agent functions | See the full pipeline, not just LLM calls |
| Add metadata for experiment tracking | Filter and compare runs in the dashboard |
| Create evaluation datasets early | Catch regressions before they reach users |
| Monitor token costs weekly | Multi-step agents can burn through budgets fast |
| Use project names to organize traces | Separate dev/staging/production environments |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Only tracing LLM calls | Wrap retrieval, processing, and tool functions with `@traceable` too |
| Ignoring cost tracking | Check the dashboard weekly; set budget alerts |
| No evaluation datasets | Create 10–20 examples for your core use cases from day one |
| Hardcoding prompts in code | Use the Prompt Hub for version control and team collaboration |
| Forgetting to set environment variables | Add `LANGSMITH_TRACING=true` to your `.env` file permanently |
| Not filtering by metadata | Add version/experiment metadata so you can compare runs later |

---

## Hands-on exercise

### Your task

Set up LangSmith tracing for a simple agent and create an evaluation dataset.

### Requirements

1. Install `langsmith` and `openai`
2. Wrap the OpenAI client with `wrap_openai`
3. Create a `@traceable` function that answers questions using an LLM
4. Create a dataset with at least 3 question/answer examples
5. Write a simple evaluator that checks keyword overlap between predicted and expected answers

### Expected result

After running the agent, you should see traces in the LangSmith dashboard with full run trees showing nested spans, token counts, and latency.

<details>
<summary>💡 Hints (click to expand)</summary>

- Set `LANGSMITH_TRACING=true` before importing anything
- `wrap_openai(OpenAI())` wraps all subsequent OpenAI calls
- Use `@traceable` on your main agent function
- `ls_client.create_dataset()` and `ls_client.create_example()` for the dataset
- The `evaluate()` function runs your agent against the dataset

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-key"

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, Client, evaluate

client = wrap_openai(OpenAI())
ls_client = Client()

@traceable
def answer_question(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

# Create dataset
dataset = ls_client.create_dataset("Exercise QA")
for q, a in [
    ("What is Python?", "Python is a programming language."),
    ("What is an API?", "An API is an application programming interface."),
    ("What is JSON?", "JSON is a data format for structured data."),
]:
    ls_client.create_example(
        inputs={"question": q},
        outputs={"answer": a},
        dataset_id=dataset.id,
    )

# Evaluator
def keyword_overlap(outputs, reference_outputs):
    pred = set(outputs["answer"].lower().split())
    ref = set(reference_outputs["answer"].lower().split())
    score = len(pred & ref) / max(len(ref), 1)
    return {"key": "overlap", "score": score}

# Run evaluation
results = evaluate(
    lambda inputs: {"answer": answer_question(inputs["question"])},
    data="Exercise QA",
    evaluators=[keyword_overlap],
)
```

</details>

### Bonus challenges

- [ ] Add metadata to traces with experiment version and user ID
- [ ] Create an LLM-as-a-judge evaluator that uses GPT-4o to score correctness
- [ ] Use the LangSmith SDK to export all traces from the past week as JSON

---

## Summary

✅ **`wrap_openai`** wraps the OpenAI client for zero-config trace capture  
✅ **`@traceable`** decorates any Python function to appear as a span in the run tree  
✅ **Run trees** show hierarchical agent execution with timing, tokens, and costs  
✅ **Evaluation datasets** enable systematic quality testing with custom evaluators  
✅ **Prompt Hub** provides version control and team collaboration for prompts  

**Next:** [OpenAI Agents Tracing](./02-openai-agents-tracing.md)

---

## Further Reading

- [LangSmith Tracing Quickstart](https://docs.langchain.com/langsmith/observability-quickstart) — Official getting started guide
- [LangSmith Evaluation Quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart) — Evaluation workflow
- [LangSmith Python SDK Reference](https://docs.smith.langchain.com/reference/python/) — Full API docs
- [LangChain Academy](https://academy.langchain.com/) — Free courses including LangSmith

<!--
Sources Consulted:
- LangSmith docs: https://docs.langchain.com/langsmith
- LangSmith tracing quickstart: https://docs.langchain.com/langsmith/observability-quickstart
- LangSmith evaluation quickstart: https://docs.langchain.com/langsmith/evaluation-quickstart
-->
