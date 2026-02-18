---
title: "Function Call Streaming"
---

# Function Call Streaming

## Introduction

In standard function calling, you wait for the model to finish generating its entire response — including all tool call names and arguments — before you can act. With **streaming**, you receive tool call data incrementally as the model generates it, enabling progressive UI updates, earlier error detection, and lower perceived latency.

Each provider streams function calls differently. We'll cover all three.

### What we'll cover

- Why streaming matters for function calling
- OpenAI streaming with the Responses API
- Anthropic streaming with `input_json_delta` events
- Google Gemini streaming with `GenerateContentResponse` chunks
- Accumulating deltas into complete tool calls
- Handling multiple parallel streamed tool calls

### Prerequisites

- [Multi-Turn Function Calling](../07-multi-turn-function-calling/00-multi-turn-function-calling.md) — The agentic loop
- [Lesson 05: Asynchronous JavaScript](../../01-web-development-fundamentals/05-asynchronous-javascript/) — Event streams
- Familiarity with server-sent events (SSE) or streaming APIs

---

## Why stream function calls?

Without streaming, the timeline looks like this:

```
Model generates ██████████████████████ (3 seconds)
                                       ↓ You get everything at once
Execute tools   ██████ (1 second)
Total wall time: 4 seconds of waiting
```

With streaming:

```
Model generates ██████████████████████ (3 seconds)
            ↓ partial args at 0.5s — show spinner
                ↓ tool name at 0.8s — show "Looking up..."
                           ↓ args complete at 2.5s — start executing!
Execute tools   ██████ (overlaps with generation)
Total perceived time: ~3 seconds
```

Streaming lets you:
- **Show progress** — display which tool is being called as soon as the name arrives
- **Validate early** — catch invalid tool names before arguments finish generating
- **Start parallel work** — begin executing a completed tool while the model still generates the next one

---

## OpenAI streaming (Responses API)

With the Responses API, streaming delivers events for each output item. Function calls arrive as `response.function_call_arguments.delta` events:

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

# Stream the response
stream = client.responses.create(
    model="gpt-4.1",
    input=[{
        "role": "user",
        "content": "What's the weather in London and Tokyo?"
    }],
    tools=tools,
    stream=True
)

# Accumulate streamed tool calls
pending_calls = {}  # item_id → {name, arguments_buffer}

for event in stream:
    match event.type:
        case "response.output_item.added":
            # A new output item started
            if event.item.type == "function_call":
                pending_calls[event.item.id] = {
                    "name": event.item.name,
                    "call_id": event.item.call_id,
                    "arguments": ""
                }
                print(f"🔧 Tool call started: {event.item.name}")
        
        case "response.function_call_arguments.delta":
            # Arguments being streamed incrementally
            call = pending_calls.get(event.item_id)
            if call:
                call["arguments"] += event.delta
        
        case "response.function_call_arguments.done":
            # Arguments complete for this tool call
            call = pending_calls.get(event.item_id)
            if call:
                args = json.loads(call["arguments"])
                print(f"✅ {call['name']} complete: {args}")
        
        case "response.output_text.delta":
            # Text content streaming
            print(event.delta, end="", flush=True)
        
        case "response.completed":
            print("\n📦 Response complete")
```

**Output:**
```
🔧 Tool call started: get_weather
✅ get_weather complete: {'location': 'London', 'units': 'celsius'}
🔧 Tool call started: get_weather
✅ get_weather complete: {'location': 'Tokyo', 'units': 'celsius'}
📦 Response complete
```

> **Note:** With parallel tool calls, multiple `response.output_item.added` events arrive. Each tool call streams its arguments independently, identified by `item_id`.

---

## Anthropic streaming (Messages API)

Anthropic streams tool calls through `content_block_start` and `content_block_delta` events. Tool call arguments arrive as `input_json_delta` deltas:

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

# Stream the response
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "What's the weather in London and Tokyo?"
    }]
) as stream:
    current_tool = None
    arguments_buffer = ""
    
    for event in stream:
        match event.type:
            case "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    current_tool = {
                        "id": block.id,
                        "name": block.name
                    }
                    arguments_buffer = ""
                    print(f"🔧 Tool call started: {block.name}")
                elif block.type == "text":
                    pass  # Text block starting
            
            case "content_block_delta":
                delta = event.delta
                if delta.type == "input_json_delta":
                    # Partial JSON for tool arguments
                    arguments_buffer += delta.partial_json
                elif delta.type == "text_delta":
                    print(delta.text, end="", flush=True)
            
            case "content_block_stop":
                if current_tool and arguments_buffer:
                    args = json.loads(arguments_buffer)
                    print(f"✅ {current_tool['name']} complete: {args}")
                    current_tool = None
                    arguments_buffer = ""
            
            case "message_stop":
                print("\n📦 Message complete")
```

**Output:**
```
🔧 Tool call started: get_weather
✅ get_weather complete: {'location': 'London', 'units': 'celsius'}
🔧 Tool call started: get_weather
✅ get_weather complete: {'location': 'Tokyo', 'units': 'celsius'}
📦 Message complete
```

> **Important:** Anthropic streams tool use arguments as `input_json_delta` events with a `partial_json` field. You must concatenate all `partial_json` strings and parse the result only when `content_block_stop` fires.

---

## Google Gemini streaming

Gemini streams function calls through the standard `generate_content` streaming interface. Each chunk may contain function call parts:

```python
from google import genai
from google.genai import types
import json

client = genai.Client()

weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a location",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "location": types.Schema(type="STRING"),
                    "units": types.Schema(
                        type="STRING",
                        enum=["celsius", "fahrenheit"]
                    )
                },
                required=["location"]
            )
        )
    ]
)

# Stream the response
response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="What's the weather in London and Tokyo?",
    config=types.GenerateContentConfig(
        tools=[weather_tool]
    )
)

for chunk in response_stream:
    for part in chunk.candidates[0].content.parts:
        if part.function_call:
            fc = part.function_call
            print(f"🔧 Tool call: {fc.name}")
            print(f"   Args: {dict(fc.args)}")
        elif part.text:
            print(part.text, end="", flush=True)
```

**Output:**
```
🔧 Tool call: get_weather
   Args: {'location': 'London', 'units': 'celsius'}
🔧 Tool call: get_weather
   Args: {'location': 'Tokyo', 'units': 'celsius'}
```

> **Note:** Gemini typically delivers function calls as complete objects within streaming chunks, rather than streaming individual argument characters. The streaming benefit here is receiving the first function call before the second one is generated.

---

## A unified streaming accumulator

For cross-provider applications, use an accumulator that normalizes streamed tool calls:

```python
from dataclasses import dataclass, field


@dataclass
class StreamedToolCall:
    """A tool call being accumulated from stream deltas."""
    id: str
    name: str
    arguments_buffer: str = ""
    is_complete: bool = False
    
    @property
    def arguments(self) -> dict:
        """Parse the accumulated arguments."""
        if not self.arguments_buffer:
            return {}
        return json.loads(self.arguments_buffer)


class ToolCallAccumulator:
    """Accumulates streamed tool call deltas into complete calls."""
    
    def __init__(self, on_call_started=None, on_call_complete=None):
        self._calls: dict[str, StreamedToolCall] = {}
        self._completed: list[StreamedToolCall] = []
        self._on_started = on_call_started
        self._on_complete = on_call_complete
    
    def start_call(self, call_id: str, name: str) -> None:
        """Register a new tool call."""
        call = StreamedToolCall(id=call_id, name=name)
        self._calls[call_id] = call
        if self._on_started:
            self._on_started(call)
    
    def append_arguments(self, call_id: str, delta: str) -> None:
        """Append argument data to a pending call."""
        call = self._calls.get(call_id)
        if call:
            call.arguments_buffer += delta
    
    def complete_call(self, call_id: str) -> StreamedToolCall | None:
        """Mark a call as complete and return it."""
        call = self._calls.get(call_id)
        if call:
            call.is_complete = True
            self._completed.append(call)
            if self._on_complete:
                self._on_complete(call)
            return call
        return None
    
    @property
    def completed_calls(self) -> list[StreamedToolCall]:
        """Return all completed tool calls."""
        return list(self._completed)
    
    @property
    def pending_count(self) -> int:
        """Number of calls still being streamed."""
        return sum(
            1 for c in self._calls.values() if not c.is_complete
        )


# Usage with callbacks
accumulator = ToolCallAccumulator(
    on_call_started=lambda c: print(f"🔧 Started: {c.name}"),
    on_call_complete=lambda c: print(
        f"✅ Complete: {c.name}({c.arguments})"
    )
)

# Simulate streaming events
accumulator.start_call("call-1", "get_weather")
accumulator.append_arguments("call-1", '{"locat')
accumulator.append_arguments("call-1", 'ion": "Lo')
accumulator.append_arguments("call-1", 'ndon"}')
accumulator.complete_call("call-1")

print(f"Completed: {len(accumulator.completed_calls)}")
print(f"Pending: {accumulator.pending_count}")
```

**Output:**
```
🔧 Started: get_weather
✅ Complete: get_weather({'location': 'London'})
Completed: 1
Pending: 0
```

---

## Progressive UI updates

The real power of streaming is keeping users informed. Here's a pattern for a chat UI:

```python
class StreamingUI:
    """Manages UI updates during streamed function calling."""
    
    def __init__(self):
        self._status_lines: list[str] = []
    
    def show_thinking(self) -> None:
        print("🤔 Thinking...")
    
    def show_tool_started(self, call: StreamedToolCall) -> None:
        """Show which tool is being prepared."""
        friendly_names = {
            "get_weather": "Checking weather",
            "search_flights": "Searching flights",
            "get_stock_price": "Looking up stock price"
        }
        label = friendly_names.get(call.name, f"Running {call.name}")
        self._status_lines.append(f"  ⏳ {label}...")
        print(self._status_lines[-1])
    
    def show_tool_executing(self, call: StreamedToolCall) -> None:
        """Update status when tool starts executing."""
        print(f"  🔄 Executing {call.name}...")
    
    def show_tool_result(
        self, call: StreamedToolCall, result: dict
    ) -> None:
        """Show the result of a completed tool."""
        print(f"  ✅ {call.name} complete")
    
    def show_text(self, text: str) -> None:
        """Stream text response to the user."""
        print(text, end="", flush=True)


# Usage in the streaming loop
ui = StreamingUI()
accumulator = ToolCallAccumulator(
    on_call_started=ui.show_tool_started,
    on_call_complete=ui.show_tool_executing
)
```

---

## Cross-provider streaming reference

| Feature | OpenAI (Responses API) | Anthropic | Gemini |
|---------|----------------------|-----------|--------|
| Stream parameter | `stream=True` | `.messages.stream()` | `generate_content_stream()` |
| Tool call start event | `response.output_item.added` | `content_block_start` (type: `tool_use`) | Part with `function_call` |
| Argument delta event | `response.function_call_arguments.delta` | `content_block_delta` (type: `input_json_delta`) | — (delivered complete) |
| Argument complete event | `response.function_call_arguments.done` | `content_block_stop` | — |
| Call identifier | `item_id` / `call_id` | `content_block.id` | Part index |
| Parallel calls in stream | Yes — interleaved by `item_id` | Yes — sequential blocks | Yes — multiple parts per chunk |
| Arguments arrive as | Character-level deltas | JSON string deltas (`partial_json`) | Complete JSON object |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use an accumulator pattern, not string concatenation | Handles multiple parallel calls cleanly |
| Parse JSON only after the complete event fires | Partial JSON will throw parse errors |
| Start tool execution as soon as its arguments are complete | Don't wait for all calls — execute in parallel |
| Show tool names to the user immediately | Reduces perceived latency even before results arrive |
| Handle stream interruptions | Network issues can cut a stream mid-arguments |
| Use callbacks for UI updates | Decouples streaming logic from display logic |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Parsing arguments before the block/item is complete | Wait for `content_block_stop` (Anthropic) or `arguments.done` (OpenAI) |
| Assuming one tool call per stream | Parallel calls interleave — track by call ID |
| Not handling the `stop_reason` / finish reason | Check for `"tool_use"` (Anthropic) or `"tool_calls"` (OpenAI) to know if more turns are needed |
| Ignoring text blocks mixed with tool calls | Models can emit text and tool calls in the same response |
| No timeout on streams | Set connection timeouts — a stuck stream blocks the entire loop |

---

## Hands-on exercise

### Your task

Build a `ToolCallAccumulator` and simulate processing a streamed response with two parallel tool calls.

### Requirements

1. Create an accumulator with `on_call_started` and `on_call_complete` callbacks
2. Simulate interleaved argument deltas for two calls
3. Track and display the state after each delta event
4. Execute both tools once their arguments are complete

### Expected result

```
🔧 Started: get_weather (call-1)
  Delta for call-1: {"loc
  Delta for call-1: ation": "P
🔧 Started: search_hotels (call-2)
  Delta for call-2: {"city
  Delta for call-1: aris"}
✅ call-1 complete: get_weather({"location": "Paris"})
  Delta for call-2: ": "Paris",
  Delta for call-2:  "stars": 4}
✅ call-2 complete: search_hotels({"city": "Paris", "stars": 4})
All calls complete. Executing...
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Define the interleaved events as a list of tuples: `(call_id, event_type, data)`
- Event types: `"start"`, `"delta"`, `"complete"`
- Process events in order to simulate real streaming
- Use the `ToolCallAccumulator` class from this lesson

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json

# Simulated interleaved stream events
events = [
    ("call-1", "start", "get_weather"),
    ("call-1", "delta", '{"loc'),
    ("call-1", "delta", 'ation": "P'),
    ("call-2", "start", "search_hotels"),
    ("call-2", "delta", '{"city'),
    ("call-1", "delta", 'aris"}'),
    ("call-1", "complete", None),
    ("call-2", "delta", '": "Paris",'),
    ("call-2", "delta", ' "stars": 4}'),
    ("call-2", "complete", None),
]

# Create accumulator with callbacks
accumulator = ToolCallAccumulator(
    on_call_started=lambda c: print(
        f"🔧 Started: {c.name} ({c.id})"
    ),
    on_call_complete=lambda c: print(
        f"✅ {c.id} complete: {c.name}({c.arguments})"
    )
)

# Process events
for call_id, event_type, data in events:
    match event_type:
        case "start":
            accumulator.start_call(call_id, data)
        case "delta":
            accumulator.append_arguments(call_id, data)
            print(f"  Delta for {call_id}: {data}")
        case "complete":
            accumulator.complete_call(call_id)

# Execute completed calls
print(f"\nAll {len(accumulator.completed_calls)} calls complete.")
for call in accumulator.completed_calls:
    print(f"  Executing {call.name}({call.arguments})...")
```

</details>

### Bonus challenges

- [ ] Add a timeout: if a tool call takes more than 5 seconds of deltas without completing, cancel it
- [ ] Implement early execution: start running call-1 as soon as it completes, even while call-2 is still streaming
- [ ] Add progress tracking: estimate completion percentage based on expected argument size

---

## Summary

✅ **Streaming function calls** reduces perceived latency by delivering tool call data incrementally

✅ **OpenAI** streams arguments character-by-character via `response.function_call_arguments.delta` events

✅ **Anthropic** uses `content_block_start` + `input_json_delta` deltas + `content_block_stop` for each tool call

✅ **Gemini** delivers function calls as complete objects within streaming chunks — first call arrives before second is generated

✅ The **accumulator pattern** tracks multiple parallel calls by ID and fires callbacks when each completes

✅ **Parse JSON only after the complete signal** — partial JSON will throw errors

**Next:** [Custom Tools →](./07-custom-tools.md)

---

[← Previous: Human-in-the-Loop](./05-human-in-the-loop.md) | [Back to Lesson Overview](./00-advanced-patterns.md)

<!-- 
Sources Consulted:
- OpenAI Function Calling Guide (streaming): https://platform.openai.com/docs/guides/function-calling
- OpenAI Responses API Reference (streaming events): https://platform.openai.com/docs/api-reference/responses/streaming
- Anthropic Streaming Messages: https://platform.claude.com/docs/en/api/messages-streaming
- Anthropic Tool Use (streaming section): https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling: https://ai.google.dev/gemini-api/docs/function-calling
-->
