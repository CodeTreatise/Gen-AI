---
title: "Streaming Flow Execution"
---

# Streaming Flow Execution

## Introduction

Streaming lets you receive **incremental output** from CrewAI Flows and Crews as they execute, rather than waiting for the entire run to complete. This is essential for building responsive user interfaces where users see progress in real time — similar to how ChatGPT streams tokens as they're generated.

### What We'll Cover

- Enabling streaming on Flows
- Enabling streaming on Crews
- Iterating over streamed chunks
- Accessing the final result after streaming
- Building streaming-aware applications

### Prerequisites

- Completed [Flow System Architecture](./02-flow-system-architecture.md)
- Understanding of Python iterators and generators

---

## Streaming in Flows

Enable streaming by setting `stream=True` as a class attribute on your Flow:

```python
from crewai.flow.flow import Flow, start, listen


class StreamingFlow(Flow):
    stream = True  # Enable streaming
    
    @start()
    def generate_content(self):
        return "AI agents are transforming how we build software."
    
    @listen(generate_content)
    def expand_content(self, content):
        return f"Expanded: {content} They automate research, writing, and analysis."


flow = StreamingFlow()

# Iterate over streamed chunks
for chunk in flow.kickoff():
    print(chunk, end="", flush=True)
```

When `stream=True`, `kickoff()` returns an **iterator** instead of a final result. Each chunk is yielded as it becomes available.

---

## Streaming in Crews

Crews also support streaming with the `stream` parameter:

```python
from crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="Research Analyst",
    goal="Research AI trends",
    backstory="Expert researcher.",
    llm="gpt-4o-mini",
)

task = Task(
    description="Write a summary of AI trends in 2025",
    expected_output="A 200-word summary",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    stream=True,  # Enable streaming
)

# Iterate over streamed output
for chunk in crew.kickoff():
    print(chunk, end="", flush=True)

print()  # Newline after streaming completes
```

**Output:**
```
AI agents are rapidly transforming...the landscape of software...development in 2025...
```

Each chunk is a partial token or text segment from the LLM response.

---

## Accessing the Final Result

After iterating through all chunks, you can access the complete result:

```python
crew = Crew(
    agents=[researcher],
    tasks=[task],
    stream=True,
)

streaming = crew.kickoff()

# Collect chunks
full_output = ""
for chunk in streaming:
    full_output += str(chunk)
    print(chunk, end="", flush=True)

# Access the final result object
result = streaming.result
print(f"\n\nTotal tokens: {result.token_usage.total_tokens}")
print(f"Tasks completed: {len(result.tasks_output)}")
```

### Streaming Object Properties

| Property | Description |
|----------|-------------|
| Iterator (`for chunk in ...`) | Yields text chunks as they arrive |
| `.result` | The complete `CrewOutput` after streaming finishes |

---

## Streaming in Web Applications

Streaming is most valuable in web applications. Here's a pattern using FastAPI with Server-Sent Events (SSE):

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from crewai import Agent, Crew, Process, Task

app = FastAPI()


def create_crew(topic: str) -> Crew:
    agent = Agent(
        role="Writer",
        goal=f"Write about {topic}",
        backstory="Expert technical writer.",
        llm="gpt-4o-mini",
    )
    task = Task(
        description=f"Write a summary about {topic}",
        expected_output="A concise summary",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], stream=True)


@app.get("/stream/{topic}")
async def stream_content(topic: str):
    crew = create_crew(topic)
    
    def generate():
        for chunk in crew.kickoff():
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Frontend Integration

```javascript
// Client-side consumption of the stream
const eventSource = new EventSource(`/stream/AI%20agents`);

eventSource.onmessage = (event) => {
    if (event.data === "[DONE]") {
        eventSource.close();
        return;
    }
    document.getElementById("output").textContent += event.data;
};
```

---

## Streaming with Flows and Crews Together

When a Flow contains Crews, you can stream at both levels:

```python
class StreamingPipelineFlow(Flow):
    stream = True  # Flow-level streaming
    
    @start()
    def research(self):
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            stream=True,  # Crew-level streaming
        )
        result = crew.kickoff()
        # When inside a flow, collect the result
        full_text = ""
        for chunk in result:
            full_text += str(chunk)
        self.state["research"] = full_text
    
    @listen(research)
    def write(self):
        # Use research from state
        return self.state["research"]
```

> **Note:** When streaming both Flow and Crew, the outer Flow's streaming captures the overall flow progress, while the inner Crew's streaming provides token-level output.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `flush=True` when printing chunks | Ensures immediate display in terminals |
| Collect chunks into a string for post-processing | Streaming is ephemeral — save if you need the full text |
| Access `.result` after iteration for metadata | Token counts and task outputs are only available after streaming ends |
| Use SSE (Server-Sent Events) for web apps | Standard pattern for streaming to browsers |
| Stream only user-facing output | Internal processing steps don't need streaming |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Accessing `.result` before iterating | Iterate through all chunks first — `.result` is populated after |
| Forgetting `end=""` and `flush=True` in print | Without these, output appears all at once, not progressively |
| Setting `stream=True` but not iterating | You must iterate over `kickoff()` to consume the stream |
| Mixing streaming and non-streaming Crews in one Flow | Decide per-Crew; collect streaming results fully before proceeding |
| Not handling stream disconnections in web apps | Add error handling for client disconnects in SSE endpoints |

---

## Hands-on Exercise

### Your Task

Build a streaming Crew that displays output progressively.

### Requirements

1. Create a single-agent Crew with `stream=True`
2. Define a task that generates a 100+ word response
3. Iterate over the stream, printing each chunk with `end=""` and `flush=True`
4. After streaming, print the total token count from `.result`

### Expected Result

```
The future of AI agents...lies in...their ability to...
(progressive output appearing word by word)

Total tokens: 234
```

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from crewai import Agent, Crew, Process, Task

agent = Agent(
    role="AI Futurist",
    goal="Write about the future of AI agents",
    backstory="Visionary technologist who's been tracking AI for 15 years.",
    llm="gpt-4o-mini",
)

task = Task(
    description="Write a 150-word prediction about AI agents in 2026",
    expected_output="A thoughtful, specific prediction about AI agent capabilities",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    stream=True,
)

streaming = crew.kickoff()

for chunk in streaming:
    print(chunk, end="", flush=True)

print(f"\n\nTotal tokens: {streaming.result.token_usage.total_tokens}")
```

</details>

---

## Summary

✅ Enable streaming with `stream=True` on Crews or as a Flow class attribute

✅ `kickoff()` returns an iterator when streaming — iterate to consume chunks

✅ Access `.result` after iteration for token usage and complete output metadata

✅ Use Server-Sent Events (SSE) for streaming to web frontends

✅ Always use `flush=True` in print statements for real-time terminal output

**Next:** [Agent Within Flow](./12-agent-within-flow.md)

---

## Further Reading

- [CrewAI Flows Documentation](https://docs.crewai.com/concepts/flows) — Streaming section
- [CrewAI Crews Documentation](https://docs.crewai.com/concepts/crews) — Crew streaming
- [MDN Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — SSE reference

*Back to [CrewAI with Flows Overview](./00-crewai-with-flows.md)*

<!-- 
Sources Consulted:
- CrewAI Flows: https://docs.crewai.com/concepts/flows
- CrewAI Crews: https://docs.crewai.com/concepts/crews
-->
