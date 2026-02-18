---
title: "Langfuse Open-Source"
---

# Langfuse Open-Source

## Introduction

Langfuse is an **open-source LLM engineering platform** that provides observability, prompt management, and evaluation — all self-hostable. If you need full control over your data, want to avoid vendor lock-in, or need to run tracing infrastructure on-premises, Langfuse is the leading open-source choice. It integrates with OpenAI, LangChain, LlamaIndex, Vercel AI SDK, and many more frameworks through native SDKs, OpenTelemetry, and drop-in API wrappers.

Langfuse joined ClickHouse in 2025, bringing enterprise-grade analytics infrastructure to its open-source core. With SOC 2 Type II, ISO 27001, GDPR, and HIPAA compliance, it's suitable for regulated industries.

### What we'll cover

- Setting up Langfuse (cloud and self-hosted)
- Drop-in OpenAI SDK tracing
- Python SDK instrumentation with decorators
- Trace and span data model
- Cost analytics and dashboards
- Prompt management and versioning
- Evaluation with scores and datasets

### Prerequisites

- Python 3.10+ installed
- An OpenAI API key
- A free Langfuse account at [cloud.langfuse.com](https://cloud.langfuse.com/) (or self-hosted instance)
- Basic agent knowledge (Unit 11, Lessons 1–5)

---

## Setting up Langfuse

### Installation

```bash
pip install langfuse openai
```

### Getting API keys

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com/auth/sign-up) (or self-host)
2. Create a new project
3. Go to project Settings → API Keys → Create new keys

### Environment configuration

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"  # EU region
# export LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"  # US region
export OPENAI_API_KEY="your-openai-api-key"
```

> **Tip:** For self-hosted instances, set `LANGFUSE_BASE_URL` to your deployment URL (e.g., `http://localhost:3000`).

---

## Drop-in OpenAI SDK tracing

The fastest way to start is Langfuse's **drop-in OpenAI wrapper**. Change one import line and all OpenAI calls are traced automatically.

```python
# Replace this:
# from openai import OpenAI

# With this:
from langfuse.openai import openai

response = openai.chat.completions.create(
    name="test-chat",
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Langfuse?"},
    ],
    metadata={"user_id": "u42", "session": "onboarding"},
)
print(response.choices[0].message.content)
```

**Output:**
```
Langfuse is an open-source LLM engineering platform that provides
observability, prompt management, and evaluation tools for AI applications.
```

After running this code, visit the Langfuse dashboard. You'll see a trace with the full LLM call — input messages, output, model, tokens, cost, and latency.

> **Note:** The `name` and `metadata` parameters are Langfuse-specific. They're stripped before the request reaches OpenAI, so they don't affect your API call.

---

## Python SDK instrumentation

For full control, use Langfuse's Python SDK with the `@observe()` decorator. This is similar to LangSmith's `@traceable` but with Langfuse's data model.

### The `@observe()` decorator

```python
from langfuse.openai import openai
from langfuse.decorators import observe, langfuse_context

@observe()
def retriever(query: str) -> list[str]:
    """Simulate document retrieval."""
    # Langfuse tracks this function as a span
    return [
        "Langfuse provides open-source LLM observability.",
        "It supports self-hosting and cloud deployment.",
    ]

@observe()
def rag_pipeline(question: str) -> str:
    """Full RAG pipeline — observed as the top-level trace."""
    docs = retriever(question)
    
    context = "\n".join(docs)
    response = openai.chat.completions.create(
        name="rag-generation",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using this context:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    
    return response.choices[0].message.content

# Run the pipeline
answer = rag_pipeline("How does Langfuse handle tracing?")
print(answer)
```

**Output:**
```
Langfuse handles tracing through its open-source observability platform,
which captures LLM calls, custom spans, and metadata for debugging.
```

In the Langfuse UI, you see a **trace tree**:

```
rag_pipeline (2.8s, $0.001)
├── retriever (0.1ms)
└── rag-generation (2.7s, 623 tokens)
```

### Adding scores and metadata

```python
@observe()
def agent_with_scoring(question: str) -> str:
    answer = rag_pipeline(question)
    
    # Attach a score to the current trace
    langfuse_context.score_current_trace(
        name="user_satisfaction",
        value=0.9,
        comment="Answer was relevant and concise",
    )
    
    # Update the current trace with metadata
    langfuse_context.update_current_trace(
        user_id="user_42",
        session_id="session_abc",
        tags=["rag", "production"],
    )
    
    return answer
```

---

## Trace and span data model

Langfuse uses a hierarchical data model based on **traces, spans, generations, and events**.

| Concept | Description | Example |
|---------|-------------|---------|
| **Trace** | Top-level container for a single request | One user query through the agent |
| **Span** | A timed operation within a trace | Retrieval step, preprocessing |
| **Generation** | An LLM call (special span type) | `chat.completions.create()` |
| **Event** | A point-in-time occurrence | "User clicked regenerate" |
| **Score** | A numeric quality rating on a trace | Correctness: 0.85 |
| **Session** | Groups related traces | Multi-turn conversation |

### Session tracking

Group multi-turn conversations into sessions:

```python
@observe()
def chat_turn(session_id: str, message: str) -> str:
    langfuse_context.update_current_trace(session_id=session_id)
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content

# Multiple turns in the same session
chat_turn("session_42", "What's the weather?")
chat_turn("session_42", "What about tomorrow?")
chat_turn("session_42", "Thanks!")
```

All three traces are linked in the Langfuse UI under session `session_42`, showing the full conversation timeline.

---

## Cost analytics

Langfuse automatically tracks token usage and calculates costs based on model pricing. The dashboard provides:

- **Per-trace cost**: Total tokens and cost for each request
- **Daily/weekly trends**: Cost over time with model breakdown
- **Per-user cost**: How much each user is costing (with user tracking)
- **Model comparison**: Cost differences between gpt-4o, gpt-4o-mini, Claude, etc.

### Viewing cost data

Navigate to Dashboard → Metrics in the Langfuse UI. Key metrics include:

| Metric | What It Shows |
|--------|--------------|
| Total cost | Aggregate spend over selected time period |
| Cost per trace | Average cost per request |
| Token distribution | Input vs. output tokens |
| Model breakdown | Cost split by model name |
| Daily trend | Cost trajectory for budget planning |

### Custom cost queries via SDK

```python
from langfuse import Langfuse

lf = Langfuse()

# Fetch recent traces with costs
traces = lf.fetch_traces(limit=50)
for t in traces.data:
    if t.total_cost:
        print(f"{t.name}: {t.total_cost:.4f} USD, {t.latency:.1f}s")
```

---

## Prompt management

Langfuse includes a built-in **prompt management system** with version control, deployment labels, and a playground for testing.

### Creating and using managed prompts

```python
from langfuse import Langfuse

lf = Langfuse()

# Create a prompt (or use the UI)
lf.create_prompt(
    name="rag-system-prompt",
    prompt="Answer the user's question using only the provided context.\n\nContext: {{context}}\n\nQuestion: {{question}}",
    labels=["production"],
)
```

### Fetching prompts at runtime

```python
from langfuse import Langfuse
from langfuse.openai import openai

lf = Langfuse()

# Fetch the production version
prompt = lf.get_prompt("rag-system-prompt", label="production")
compiled = prompt.compile(context="Langfuse is open-source.", question="What is Langfuse?")

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": compiled},
        {"role": "user", "content": "Answer the question."},
    ],
    langfuse_prompt=prompt,  # Links trace to this prompt version
)
```

### Key prompt management features

| Feature | Description |
|---------|-------------|
| **Version control** | Every edit creates a new version with diff tracking |
| **Labels** | Tag versions as "production", "staging", "experiment" |
| **Variables** | Use `{{variable}}` syntax for dynamic content |
| **Playground** | Test prompts against models directly in the UI |
| **Linked traces** | See which prompt version generated each trace |
| **Composability** | Include prompts within other prompts |

---

## Evaluation framework

Langfuse supports multiple evaluation methods:

### 1. LLM-as-a-judge

Configure automated evaluators that use an LLM to score traces:

```python
from langfuse import Langfuse

lf = Langfuse()

# Score a trace with LLM-as-a-judge (via UI or SDK)
lf.score(
    trace_id="trace_abc123",
    name="relevance",
    value=0.9,
    comment="Response directly addresses the user's question",
)
```

### 2. Datasets and experiments

```python
# Create a dataset
dataset = lf.create_dataset("QA Benchmark")

# Add items
lf.create_dataset_item(
    dataset_name="QA Benchmark",
    input={"question": "What is Langfuse?"},
    expected_output="Langfuse is an open-source LLM observability platform.",
)

# Run experiments against the dataset
for item in lf.get_dataset("QA Benchmark").items:
    result = rag_pipeline(item.input["question"])
    
    item.link(
        trace_id=langfuse_context.get_current_trace_id(),
        run_name="rag-v2-experiment",
    )
```

### 3. User feedback

Capture user feedback (thumbs up/down) and link it to traces:

```python
# After user gives feedback in the UI
lf.score(
    trace_id="trace_abc123",
    name="user_feedback",
    value=1,  # 1 = positive, 0 = negative
    comment="User clicked thumbs up",
)
```

---

## Self-hosting Langfuse

Langfuse can be self-hosted via Docker:

```bash
# Clone and start with Docker Compose
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then set `LANGFUSE_BASE_URL=http://localhost:3000` in your application.

| Hosting Option | Best For |
|---------------|----------|
| **Langfuse Cloud** | Quick start, managed infrastructure |
| **Self-hosted (Docker)** | Full data control, air-gapped environments |
| **Self-hosted (Kubernetes)** | Enterprise scale, HA deployments |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use the OpenAI drop-in wrapper for quick starts | One import change gives you full tracing |
| Add `@observe()` to all pipeline functions | See the complete execution flow, not just LLM calls |
| Track sessions for multi-turn conversations | Understand conversation quality over time |
| Use prompt management for version control | Iterate on prompts without code deployments |
| Score traces with evaluation methods | Measure quality systematically, not anecdotally |
| Self-host for data-sensitive applications | Keep all trace data within your infrastructure |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting to flush traces at exit | Call `langfuse_context.flush()` or Langfuse auto-flushes on exit |
| Not setting `LANGFUSE_BASE_URL` | Defaults to EU cloud; set explicitly for US or self-hosted |
| Ignoring session tracking | Use `session_id` to group multi-turn conversations |
| Using the SDK without the OpenAI wrapper | Combine `@observe()` with the drop-in wrapper for maximum visibility |
| Not linking prompts to traces | Pass `langfuse_prompt=prompt` to connect prompt versions with their traces |
| Skipping evaluation | Set up at least user feedback scoring from day one |

---

## Hands-on exercise

### Your task

Set up Langfuse tracing for a RAG pipeline and score the results.

### Requirements

1. Install `langfuse` and use the OpenAI drop-in wrapper
2. Create a `@observe()` decorated pipeline with retrieval and generation steps
3. Track two queries under the same session
4. Score one trace with a quality rating
5. View the traces and session in the Langfuse dashboard

### Expected result

The Langfuse dashboard shows two linked traces under the same session, with the retrieval and generation spans visible in the trace tree, and a score attached to one trace.

<details>
<summary>💡 Hints (click to expand)</summary>

- Import `from langfuse.openai import openai` for the drop-in wrapper
- Use `langfuse_context.update_current_trace(session_id=...)` for session tracking
- Use `langfuse_context.score_current_trace(name=..., value=...)` for scoring
- Call `langfuse_context.flush()` at the end to ensure all data is sent

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langfuse.openai import openai
from langfuse.decorators import observe, langfuse_context

@observe()
def retrieve(query: str) -> list[str]:
    return ["Langfuse is open-source.", "It supports self-hosting."]

@observe()
def generate_answer(question: str) -> str:
    docs = retrieve(question)
    langfuse_context.update_current_trace(
        session_id="exercise_session",
        user_id="student_1",
    )
    
    response = openai.chat.completions.create(
        name="answer-gen",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Context: {docs}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

# Query 1
a1 = generate_answer("What is Langfuse?")
print(f"A1: {a1}")

# Score the first trace
langfuse_context.score_current_trace(
    name="quality", value=0.9, comment="Accurate answer"
)

# Query 2 (same session)
a2 = generate_answer("Can I self-host it?")
print(f"A2: {a2}")

# Flush to ensure traces are sent
langfuse_context.flush()
```

</details>

### Bonus challenges

- [ ] Create a managed prompt and use it in your pipeline
- [ ] Set up a dataset with 5 questions and run an experiment
- [ ] Self-host Langfuse with Docker and point your SDK at `localhost:3000`

---

## Summary

✅ **Drop-in OpenAI wrapper** traces all LLM calls with a single import change  
✅ **`@observe()` decorator** captures custom functions as spans in the trace tree  
✅ **Session tracking** groups multi-turn conversations for conversation-level analysis  
✅ **Cost analytics** provide per-trace, per-user, and aggregate cost visibility  
✅ **Prompt management** with version control, labels, and playground for team collaboration  
✅ **Self-hosting** gives full data control for regulated environments  

**Previous:** [OpenAI Agents Tracing](./02-openai-agents-tracing.md)  
**Next:** [Helicone Integration](./04-helicone-integration.md)

---

## Further Reading

- [Langfuse Documentation](https://langfuse.com/docs) — Full platform docs
- [Langfuse Getting Started](https://langfuse.com/docs/observability/get-started) — Tracing quickstart
- [Langfuse GitHub](https://github.com/langfuse/langfuse) — Open-source repository (22K+ stars)
- [Langfuse Self-Hosting Guide](https://langfuse.com/self-hosting) — Docker and Kubernetes deployment

<!--
Sources Consulted:
- Langfuse docs overview: https://langfuse.com/docs
- Langfuse get started: https://langfuse.com/docs/observability/get-started
- Langfuse OpenAI integration: https://langfuse.com/integrations/model-providers/openai-py
-->
