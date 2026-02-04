---
title: "Thinking with Tools and Streaming"
---

# Thinking with Tools and Streaming

## Introduction

Extended thinking introduces complexity when combined with tool use and streaming. The model thinks, may call tools, thinks again based on results, and produces a final response. Understanding how these features interact ensures robust implementations.

This lesson covers the technical patterns for using extended thinking with tool calling and streaming responses.

### What We'll Cover

- Tool use with extended thinking
- Preserving thinking blocks across turns
- Interleaved thinking mode
- Streaming thinking content
- Toggling thinking during conversations

### Prerequisites

- [Extended Thinking Overview](./00-extended-thinking-overview.md)
- [Thinking Budget Configuration](./01-thinking-budget-configuration.md)
- Understanding of tool use patterns

---

## Tool Use with Extended Thinking

### How It Works

When extended thinking is enabled with tool use:

1. Model thinks about the problem
2. Model decides to call tool(s)
3. You execute tools and return results
4. Model thinks about tool results
5. Model produces final response

```
User Prompt
    ↓
[Extended Thinking] ← Reasons about problem
    ↓
Tool Call(s)
    ↓
You: Execute & Return Results
    ↓
[Extended Thinking] ← Reasons about results
    ↓
Final Response
```

### Basic Tool Use Pattern

```python
from anthropic import Anthropic

client = Anthropic()

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
]

# Initial request with thinking
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo and Paris?"}]
)

# Check for tool use
if response.stop_reason == "tool_use":
    # Process tool calls
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })
    
    # Continue conversation with tool results
    final_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo and Paris?"},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results}
        ]
    )
```

---

## Preserving Thinking Blocks

> **⚠️ Critical:** When continuing conversations with tool results, you must preserve thinking blocks from previous assistant responses.

### The Rule

Include ALL content blocks (thinking, text, and tool_use) in the assistant message:

```python
# ❌ WRONG: Only passing tool calls
messages = [
    {"role": "user", "content": original_prompt},
    {"role": "assistant", "content": [
        # Missing thinking blocks!
        {"type": "tool_use", "id": "123", ...}
    ]},
    {"role": "user", "content": tool_results}
]

# ✅ CORRECT: Preserve entire response.content
messages = [
    {"role": "user", "content": original_prompt},
    {"role": "assistant", "content": response.content},  # All blocks
    {"role": "user", "content": tool_results}
]
```

### Why This Matters

The model's thinking provides context for its decisions. Removing it:
- Breaks the logical flow
- May cause inconsistent behavior
- Loses the reasoning behind tool calls

### Complete Example

```python
def call_with_tools_and_thinking(client, prompt, tools):
    """Execute a multi-turn tool-using conversation with thinking."""
    
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            tools=tools,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response
        
        if response.stop_reason == "tool_use":
            # Preserve ENTIRE response content (including thinking)
            messages.append({
                "role": "assistant",
                "content": response.content  # Includes thinking + tool calls
            })
            
            # Execute tools and add results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({
                "role": "user",
                "content": tool_results
            })
```

---

## Tool Choice Restrictions

> **Important:** With extended thinking, only `tool_choice: "auto"` is supported.

```python
# ✅ Supported: auto (default)
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=tools,
    tool_choice={"type": "auto"},  # Default, can be omitted
    messages=messages
)

# ❌ NOT Supported with extended thinking
# tool_choice={"type": "any"}     # Error
# tool_choice={"type": "tool", "name": "specific_tool"}  # Error
```

### Why This Limitation?

Extended thinking requires the model to reason about *whether* to use tools. Forcing tool use short-circuits this reasoning process.

---

## Interleaved Thinking (Beta)

Standard extended thinking puts all thinking *before* the response. **Interleaved thinking** allows the model to think *between* actions:

```
[Think] → [Tool Call] → [Think about results] → [Another Tool] → [Think] → [Response]
```

### Enabling Interleaved Thinking

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    betas=["interleaved-thinking-2025-05-14"],  # Beta header
    tools=tools,
    messages=messages
)
```

### Budget Behavior with Interleaving

With interleaved thinking, `budget_tokens` can exceed `max_tokens`:

```python
# Without interleaving: budget_tokens < max_tokens (required)
thinking={"type": "enabled", "budget_tokens": 8000},
max_tokens=10000

# With interleaving: budget_tokens can exceed max_tokens
betas=["interleaved-thinking-2025-05-14"],
thinking={"type": "enabled", "budget_tokens": 20000},
max_tokens=10000  # Budget > max is now allowed
```

### When to Use Interleaved Thinking

| Use Case | Interleaved? |
|----------|-------------|
| Simple, single-tool calls | Not needed |
| Complex multi-step workflows | ✅ Recommended |
| Agent loops with many tool calls | ✅ Recommended |
| Research requiring progressive reasoning | ✅ Recommended |

---

## Streaming Extended Thinking

### Event Types

When streaming with extended thinking enabled, you receive special events:

| Event | Content | Purpose |
|-------|---------|---------|
| `thinking` | Thinking text | The model's reasoning |
| `thinking_delta` | Incremental thinking | Streamed thinking chunks |
| `signature_delta` | Cryptographic signature | Verify thinking authenticity |

### Basic Streaming Pattern

```python
from anthropic import Anthropic

client = Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": prompt}]
) as stream:
    
    current_block = None
    
    for event in stream:
        if event.type == "content_block_start":
            current_block = event.content_block.type
            if current_block == "thinking":
                print("\n[THINKING]")
            elif current_block == "text":
                print("\n[RESPONSE]")
        
        elif event.type == "content_block_delta":
            if hasattr(event.delta, "thinking"):
                print(event.delta.thinking, end="", flush=True)
            elif hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
            elif hasattr(event.delta, "signature"):
                # Collect signature (usually at end)
                pass
```

### TypeScript Streaming

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = await client.messages.stream({
  model: "claude-sonnet-4-5-20250929",
  max_tokens: 16000,
  thinking: { type: "enabled", budget_tokens: 10000 },
  messages: [{ role: "user", content: prompt }],
});

for await (const event of stream) {
  if (event.type === "content_block_delta") {
    if ("thinking" in event.delta) {
      process.stdout.write(event.delta.thinking);
    } else if ("text" in event.delta) {
      process.stdout.write(event.delta.text);
    }
  }
}
```

### Streaming Required for Large Budgets

> **Note:** When `max_tokens > 21,333`, streaming is required.

```python
# max_tokens > 21,333 requires streaming
# This will ERROR without streaming:
response = client.messages.create(
    max_tokens=30000,  # > 21,333
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)

# Use streaming instead:
with client.messages.stream(
    max_tokens=30000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
) as stream:
    response = stream.get_final_message()
```

---

## Thought Signatures

### What Are They?

Thinking blocks include a cryptographic `signature` field that verifies the thinking is authentic and unmodified:

```python
response = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)

for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
        print(f"Signature: {block.signature}")  # Cryptographic signature
```

### Using Signatures in Multi-Turn

When passing thinking blocks back in conversation history, the signature ensures integrity:

```python
# First turn
response1 = client.messages.create(...)

# Second turn - pass back the thinking with signature intact
response2 = client.messages.create(
    messages=[
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": response1.content},  # Includes signature
        {"role": "user", "content": followup}
    ]
)
```

> **Important:** Don't modify thinking blocks or signatures. The API may reject tampered content.

---

## Toggling Thinking Mid-Conversation

### Graceful Degradation

If you toggle thinking mode during a conversation, the model handles it gracefully:

```python
# Turn 1: Thinking enabled
response1 = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve this math problem..."}]
)

# Turn 2: Thinking disabled (still works)
response2 = client.messages.create(
    # No thinking parameter
    messages=[
        {"role": "user", "content": "Solve this math problem..."},
        {"role": "assistant", "content": response1.content},  # Has thinking
        {"role": "user", "content": "Now explain it simply."}
    ]
)
```

### Best Practice

Complete the current assistant turn before changing thinking mode:

```python
# ✅ Good: Toggle between complete turns
turn1 = create_with_thinking(prompt1)
turn2 = create_without_thinking(followup)  # Fine after turn1 completes

# ⚠️ Avoid: Don't toggle mid-turn with tool use
```

---

## Gemini Tool Use with Thinking

### Gemini 2.5 with thinkingBudget

```python
from google import genai
from google.genai.types import Tool, GenerateContentConfig

client = genai.Client()

weather_tool = Tool(
    function_declarations=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Seattle?",
    config=GenerateContentConfig(
        tools=[weather_tool],
        thinking_config={"thinking_budget": 8000}
    )
)
```

### Gemini 3 with thinkingLevel

```python
response = client.models.generate_content(
    model="gemini-3.0-flash",
    contents="What's the weather in Seattle?",
    config=GenerateContentConfig(
        tools=[weather_tool],
        thinking_config={"thinking_level": "medium"}  # low, medium, or high
    )
)
```

### Thought Summaries for Multi-Turn (Gemini)

Pass thought summaries back for function calling continuations:

```python
# Get thought summaries
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=GenerateContentConfig(
        thinking_config={"thinking_budget": 8000, "include_thoughts": True}
    )
)

# Extract thought summary
thought_summary = None
for part in response.candidates[0].content.parts:
    if hasattr(part, "thought") and part.thought:
        thought_summary = part.text
        break

# Pass back in continuation
next_contents = [
    {"role": "user", "parts": [{"text": original_prompt}]},
    {"role": "model", "parts": response.candidates[0].content.parts},
    {"role": "user", "parts": [{"text": followup}]}
]
```

---

## Common Patterns

### Agent Loop with Thinking

```python
def agent_loop(client, initial_prompt, tools, max_iterations=10):
    """Run an agent loop with extended thinking."""
    
    messages = [{"role": "user", "content": initial_prompt}]
    
    for i in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            betas=["interleaved-thinking-2025-05-14"],
            tools=tools,
            messages=messages
        )
        
        # Log thinking for debugging
        for block in response.content:
            if block.type == "thinking":
                print(f"[Step {i+1} Thinking]: {block.thinking[:200]}...")
        
        if response.stop_reason == "end_turn":
            return extract_final_response(response)
        
        # Preserve full content and add tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": execute_tools(response)})
    
    raise Exception("Max iterations reached")
```

### Streaming Agent with Progress Updates

```python
async def streaming_agent(client, prompt, tools):
    """Stream thinking for real-time progress updates."""
    
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        async with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            tools=tools,
            messages=messages
        ) as stream:
            
            thinking_preview = []
            
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "thinking"):
                        thinking_preview.append(event.delta.thinking)
                        # Show progress to user
                        yield {"type": "thinking", "content": event.delta.thinking}
                    elif hasattr(event.delta, "text"):
                        yield {"type": "text", "content": event.delta.text}
            
            response = await stream.get_final_message()
        
        if response.stop_reason == "end_turn":
            return
        
        # Continue with tool execution
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": await execute_tools(response)})
```

---

## Hands-on Exercise

### Your Task

Build a function that uses extended thinking with tools, properly preserving thinking blocks:

**Requirements:**
1. Enable extended thinking with a 10,000 token budget
2. Define a simple tool (e.g., `calculate`)
3. Handle tool calls with proper thinking preservation
4. Stream the thinking output to console

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `response.content` directly when appending to messages
- Check `response.stop_reason` to determine if tool use occurred
- Streaming requires the `.stream()` method

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "calculate",
        "description": "Perform a calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    }
]

def execute_calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def run_with_thinking_and_tools(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        # Stream the response
        print("\n--- Streaming Response ---")
        
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            tools=tools,
            messages=messages
        ) as stream:
            
            current_block = None
            
            for event in stream:
                if event.type == "content_block_start":
                    current_block = event.content_block.type
                    if current_block == "thinking":
                        print("\n[THINKING]")
                    elif current_block == "text":
                        print("\n[RESPONSE]")
                    elif current_block == "tool_use":
                        print(f"\n[TOOL: {event.content_block.name}]")
                        
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "thinking"):
                        print(event.delta.thinking, end="", flush=True)
                    elif hasattr(event.delta, "text"):
                        print(event.delta.text, end="", flush=True)
            
            response = stream.get_final_message()
        
        # Check if done
        if response.stop_reason == "end_turn":
            print("\n\n--- Done ---")
            return response
        
        # Handle tool use - preserve full content including thinking
        if response.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": response.content  # Preserves thinking blocks!
            })
            
            # Execute tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "calculate":
                        result = execute_calculate(block.input["expression"])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                        print(f"[RESULT: {result}]")
            
            messages.append({
                "role": "user",
                "content": tool_results
            })

# Test
if __name__ == "__main__":
    run_with_thinking_and_tools(
        "Calculate the factorial of 10, then tell me if it's divisible by 7."
    )
```

</details>

---

## Summary

✅ **Preserve thinking blocks** when continuing tool-use conversations
✅ **Only `tool_choice: auto`** is supported with extended thinking
✅ **Use interleaved thinking** for complex multi-tool workflows
✅ **Stream for large budgets** (required when `max_tokens > 21,333`)
✅ **Keep signatures intact** when passing thinking back
✅ **Complete turns before toggling** thinking mode

**Next:** [Debugging and Optimization](./04-debugging-optimization.md)

---

## Further Reading

- [Anthropic Extended Thinking with Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use)
- [Gemini Thinking with Function Calling](https://ai.google.dev/gemini-api/docs/thinking#function-calling)
- [Anthropic Streaming Guide](https://docs.anthropic.com/en/api/streaming)

---

<!-- 
Sources Consulted:
- Anthropic Extended Thinking: Tool use patterns, interleaved thinking beta
- Anthropic: Streaming events, signature handling, thinking preservation
- Gemini Thinking: Thought signatures for function calling
-->
