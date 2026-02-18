---
title: "Structured Output and Streaming"
---

# Structured Output and Streaming

## Introduction

When you integrate AI agents into real applications, two problems surface quickly: unpredictable response formats and slow perceived latency. A user waiting five seconds for a wall of unstructured text is a poor experience. A downstream service that receives free-form prose instead of typed JSON will break.

AutoGen AgentChat solves both problems directly. **Structured output** forces an agent to respond with a validated Pydantic model, giving you typed, predictable data every time. **Streaming** pushes tokens to the client as they are generated, so users see progress immediately instead of staring at a spinner. And **model context management** controls how much conversation history the agent sends to the LLM, keeping costs down and responses focused.

This lesson shows you how to use all three capabilities together to build agents that are fast, reliable, and production-ready.

### What You'll Cover

- Forcing typed responses with `output_content_type` and `StructuredMessage`
- Streaming token-by-token output with `model_client_stream` and `ModelClientStreamingChunkEvent`
- Managing conversation history with `UnboundedChatCompletionContext`, `BufferedChatCompletionContext`, and `TokenLimitedChatCompletionContext`
- Generating human-readable tool summaries with `reflect_on_tool_use`

### Prerequisites

- Familiarity with AutoGen `AssistantAgent` basics (see [Agent Types and Configuration](./02-agent-types-and-configuration.md))
- Working knowledge of Pydantic models
- Python 3.10+ with `autogen-agentchat` installed
- A configured model client (OpenAI, Azure, etc.)

---

## Structured Output with Pydantic

By default, an `AssistantAgent` returns free-form text. That works for chat, but not when you need to parse the response programmatically. The `output_content_type` parameter changes this — it tells the agent to always respond with an instance of a specific Pydantic model.

### Defining a Response Schema

Start by defining the shape of the data you want back:

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating from 1.0 to 10.0")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief overall summary")
```

The `Field(description=...)` metadata is not just for documentation — AutoGen passes these descriptions to the LLM to guide its output. Clear descriptions produce more accurate structured responses.

### Creating a Structured Agent

Pass your Pydantic model as `output_content_type` when constructing the agent:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage

agent = AssistantAgent(
    name="movie_reviewer",
    model_client=model_client,
    system_message="You are a professional movie critic. Provide detailed reviews.",
    output_content_type=MovieReview,  # Forces structured output
)

result = await agent.run(task="Review the movie Inception")
last = result.messages[-1]

assert isinstance(last, StructuredMessage)
review = last.content  # Typed as MovieReview

print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Pros: {', '.join(review.pros)}")
print(f"Summary: {review.summary}")
```

**Output:**

```
Title: Inception
Rating: 9.2/10
Pros: Innovative concept, Stunning visuals, Strong ensemble cast
Summary: Christopher Nolan delivers a mind-bending thriller that layers dreams
within dreams, creating a cinematic puzzle that rewards repeat viewings.
```

### How StructuredMessage Works

When you set `output_content_type`, the agent's final response is always a `StructuredMessage` rather than a plain `TextMessage`. The key difference:

| Property | `TextMessage` | `StructuredMessage` |
|---|---|---|
| `.content` | `str` | Pydantic model instance |
| `.type` | `"TextMessage"` | `"StructuredMessage"` |
| Validation | None | Automatic via Pydantic |

AutoGen handles the parsing and validation internally. If the LLM returns malformed data, AutoGen retries the request. You always receive a valid, fully typed object.

### Structured Output in Teams

Structured output works in multi-agent teams too. When a team's final agent uses `output_content_type`, the entire team pipeline produces structured data:

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

class AnalysisReport(BaseModel):
    topic: str
    key_findings: list[str]
    confidence: float
    recommendation: str

analyst = AssistantAgent(
    name="analyst",
    model_client=model_client,
    output_content_type=AnalysisReport,
)

team = RoundRobinGroupChat(
    participants=[researcher, analyst],  # analyst speaks last
    termination_condition=MaxMessageTermination(max_messages=4),
)

result = await team.run(task="Analyze the impact of remote work on productivity")
# The final StructuredMessage contains an AnalysisReport instance
```

This pattern is powerful: upstream agents gather and discuss information in free-form text, then the final agent distills everything into a clean, typed object.

---

## Streaming Tokens in Real-Time

For long-form responses — stories, analyses, code generation — users should not wait for the entire response to complete. Streaming sends tokens to the client as the model generates them, dramatically improving perceived responsiveness.

### Enabling Streaming

Set `model_client_stream=True` on the agent, then call `run_stream()` instead of `run()`:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import ModelClientStreamingChunkEvent

agent = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="You are a creative writer.",
    model_client_stream=True,  # Enable streaming
)

async for message in agent.run_stream(task="Write a haiku about Python programming"):
    if isinstance(message, ModelClientStreamingChunkEvent):
        print(message.content, end="", flush=True)
```

**Output (appears token by token):**

```
Indentation rules,
Serpent logic coils through code,
Whitespace speaks volumes.
```

### Understanding the Stream Events

The `run_stream()` method yields several event types. Filter them based on what you need:

```python
from autogen_agentchat.base import TaskResult, ModelClientStreamingChunkEvent

async for event in agent.run_stream(task="Explain recursion in three sentences"):
    if isinstance(event, ModelClientStreamingChunkEvent):
        # Individual token chunks — use for real-time display
        print(event.content, end="", flush=True)
    elif isinstance(event, TaskResult):
        # Final result — contains all messages
        print(f"\n\nTotal messages: {len(event.messages)}")
```

**Output:**

```
Recursion is a technique where a function calls itself to solve smaller
instances of the same problem. Each call moves closer to a base case that
stops the recursion. Think of it like Russian nesting dolls — you open each
one until you reach the smallest doll inside.

Total messages: 2
```

The `ModelClientStreamingChunkEvent` carries a `.content` string containing one or more tokens. The `TaskResult` arrives last and contains the complete conversation history, exactly as if you had called `run()`.

### Streaming to a Web Client

In a web application, pipe streaming chunks directly to an SSE (Server-Sent Events) or WebSocket endpoint:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_stream(task: str):
    async for event in agent.run_stream(task=task):
        if isinstance(event, ModelClientStreamingChunkEvent):
            yield f"data: {event.content}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_response(task: str):
    return StreamingResponse(
        generate_stream(task),
        media_type="text/event-stream",
    )
```

This gives your frontend a ChatGPT-like typing experience with minimal effort.

---

## Managing Model Context

Every time an agent calls the LLM, it sends conversation history as context. By default, AutoGen sends *everything* — every message from the entire conversation. For short interactions this is fine. For long-running agents or multi-turn conversations, it leads to ballooning costs and eventually hits the model's context window limit.

Model context strategies control which messages get sent.

### Unbounded Context (Default)

```python
from autogen_core.model_context import UnboundedChatCompletionContext

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    model_context=UnboundedChatCompletionContext(),  # This is the default
)
```

All messages are kept and sent to the model on every turn. Simple, but expensive for long conversations.

**Use when:** Conversations are short (under ~20 turns) or you need full history for accuracy.

### Buffered Context

```python
from autogen_core.model_context import BufferedChatCompletionContext

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    model_context=BufferedChatCompletionContext(buffer_size=10),
)
```

Only the last `N` messages are sent to the model. Older messages are silently dropped from the LLM's view (though they remain in the `TaskResult` history).

**Use when:** You want a simple sliding window and the agent does not need to reference old context. Good for customer support bots and chatbots where recent context matters most.

### Token-Limited Context

```python
from autogen_core.model_context import TokenLimitedChatCompletionContext

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    model_context=TokenLimitedChatCompletionContext(max_token=4000),
)
```

Messages are kept as long as they fit within the specified token budget. When adding a new message would exceed the limit, the oldest messages are removed first.

**Use when:** You need precise control over costs or are working with models that have small context windows. This is the most production-appropriate strategy for long-running agents.

### Comparing Context Strategies

| Strategy | History Sent | Cost | Risk |
|---|---|---|---|
| `UnboundedChatCompletionContext` | All messages | High (grows linearly) | Context window overflow |
| `BufferedChatCompletionContext` | Last N messages | Fixed | Loses older context |
| `TokenLimitedChatCompletionContext` | Within token budget | Capped | Loses older context |

### Choosing the Right Strategy

A practical rule of thumb: start with `BufferedChatCompletionContext(buffer_size=20)` for most agents. Switch to `TokenLimitedChatCompletionContext` if you need tighter cost control or your messages vary widely in length. Reserve `UnboundedChatCompletionContext` for short-lived, single-task agents.

---

## Reflecting on Tool Use

When an agent calls a tool, the raw tool output is returned as the agent's response by default. For APIs that return JSON blobs or technical data, this is not user-friendly.

The `reflect_on_tool_use` parameter tells the agent to make a second LLM call after receiving tool output, generating a natural-language summary instead of forwarding raw results.

### Without Reflection

```python
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
)
result = await agent.run(task="What's the weather in Tokyo?")
print(result.messages[-1].content)
```

**Output:**

```
{"location": "Tokyo", "temp_c": 22, "condition": "Partly Cloudy", "humidity": 65}
```

### With Reflection

```python
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    reflect_on_tool_use=True,  # Summarize tool output
)
result = await agent.run(task="What's the weather in Tokyo?")
print(result.messages[-1].content)
```

**Output:**

```
The weather in Tokyo is currently 22°C and partly cloudy with 65% humidity.
A pleasant day overall — no umbrella needed!
```

The reflected response reads naturally and is suitable for display to end users. Under the hood, AutoGen sends the raw tool output back to the LLM with the conversation context, and the LLM generates a coherent summary.

**Trade-off:** Reflection adds one extra LLM call per tool invocation. For latency-sensitive applications, consider whether the improved readability justifies the added round-trip.

---

## Combining All Three

These features compose naturally. Here is an agent that streams structured output with a bounded context window:

```python
from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.base import TaskResult, ModelClientStreamingChunkEvent
from autogen_core.model_context import BufferedChatCompletionContext

class CodeReview(BaseModel):
    file_name: str = Field(description="Name of the reviewed file")
    issues: list[str] = Field(description="List of issues found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    quality_score: int = Field(description="Code quality score from 1 to 10")

agent = AssistantAgent(
    name="code_reviewer",
    model_client=model_client,
    system_message="You review code for quality, bugs, and best practices.",
    output_content_type=CodeReview,
    model_client_stream=True,
    model_context=BufferedChatCompletionContext(buffer_size=5),
)

# Stream the response, then extract structured data
async for event in agent.run_stream(task="Review this function:\ndef add(a, b): return a + b"):
    if isinstance(event, ModelClientStreamingChunkEvent):
        print(event.content, end="", flush=True)
    elif isinstance(event, TaskResult):
        last = event.messages[-1]
        if isinstance(last, StructuredMessage):
            review = last.content
            print(f"\n\nQuality Score: {review.quality_score}/10")
            print(f"Issues: {review.issues}")
```

---

## Best Practices

1. **Add Field descriptions to every Pydantic field.** The LLM uses these to understand what data to generate. Vague or missing descriptions produce unreliable output.

2. **Keep structured models flat when possible.** Deeply nested Pydantic models increase the chance of malformed responses. If you need nested data, test thoroughly.

3. **Use streaming for any response that takes more than ~2 seconds.** Users perceive streamed responses as faster even when total time is identical.

4. **Set a model context strategy from day one.** Switching from unbounded to bounded context later can change agent behavior in subtle ways. Start with a bounded strategy and loosen it only if needed.

5. **Pair `reflect_on_tool_use` with user-facing agents only.** Backend agents that pass tool output to other agents do not need reflection — it just adds latency and cost.

6. **Validate structured output at the application layer too.** Pydantic catches type errors, but you should still validate business logic (e.g., rating is between 1 and 10).

---

## Common Pitfalls

| Pitfall | What Happens | Fix |
|---|---|---|
| Forgetting `await` with `run()` | Returns a coroutine object, not results | Always `await agent.run(...)` |
| Using `run()` instead of `run_stream()` with streaming enabled | Works, but no streaming benefit — you get the full result at once | Call `run_stream()` to see chunks |
| Setting `buffer_size` too small | Agent loses important context, gives incoherent responses | Start with 10–20 and tune down |
| Complex Pydantic models with `Optional` unions | LLM may produce invalid JSON for complex union types | Simplify models, avoid deeply optional unions |
| Not handling `TaskResult` in streaming loop | You miss the final structured message | Always check for `TaskResult` at the end |

---

## Hands-On Exercise

Build a **product review analyzer** that combines structured output, streaming, and context management:

1. Define a `ProductAnalysis` Pydantic model with fields: `product_name` (str), `sentiment` (str: positive/negative/mixed), `key_themes` (list of str), `confidence_score` (float 0–1), and `summary` (str).

2. Create an `AssistantAgent` with:
   - `output_content_type=ProductAnalysis`
   - `model_client_stream=True`
   - `model_context=BufferedChatCompletionContext(buffer_size=5)`

3. Run the agent with `run_stream()`, printing each streaming chunk as it arrives.

4. After the stream completes, extract the `ProductAnalysis` from the final `StructuredMessage` and print each field.

5. Run three different product reviews in sequence to verify the buffered context drops old messages correctly.

**Stretch goal:** Add a tool that fetches fake product data and enable `reflect_on_tool_use=True`. Verify the agent summarizes the tool output before generating the structured analysis.

---

## Summary

- **Structured output** (`output_content_type`) forces agents to respond with validated Pydantic models via `StructuredMessage`, eliminating parsing and format errors.
- **Streaming** (`model_client_stream=True` + `run_stream()`) delivers tokens in real-time through `ModelClientStreamingChunkEvent`, improving perceived latency.
- **Model context** strategies (`Unbounded`, `Buffered`, `TokenLimited`) control how much conversation history reaches the LLM, managing cost and preventing context window overflow.
- **Tool reflection** (`reflect_on_tool_use=True`) converts raw tool output into natural-language summaries at the cost of one extra LLM call.

These four capabilities make the difference between a prototype agent and a production agent. Use them together to build systems that are predictable, responsive, and cost-effective.

**Next:** [Component Serialization](./14-component-serialization.md)

---

## Further Reading

- [AutoGen Structured Output Documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/structured-output.html)
- [AutoGen Streaming Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/streaming.html)
- [Pydantic Model Documentation](https://docs.pydantic.dev/latest/)
- [AutoGen Model Context Reference](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/model-context.html)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources:
- AutoGen AgentChat documentation (microsoft.github.io/autogen)
- AutoGen GitHub repository (github.com/microsoft/autogen)
- Pydantic v2 documentation (docs.pydantic.dev)
-->
