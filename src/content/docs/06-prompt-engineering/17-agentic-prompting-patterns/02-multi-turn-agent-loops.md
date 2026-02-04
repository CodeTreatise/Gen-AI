---
title: "Multi-Turn Agent Loops"
---

# Multi-Turn Agent Loops

## Introduction

Single tool calls are simple—the model calls a function, you return a result, done. But real agents need to **iterate**: search, analyze results, search again, combine findings, take action. Multi-turn loops are what transform a model with tools into an autonomous agent.

This lesson covers how to build reliable agent loops that maintain context, handle tool results, and know when to stop.

> **🤖 AI Context:** The ReAct pattern (Reasoning + Acting) is the most common framework for agent loops. Each turn: think about what to do → take action → observe result → repeat.

### What We'll Cover

- Agent loop architecture
- Context management across turns
- Tool result handling patterns
- ReAct implementation
- Stopping conditions and safeguards
- Streaming agent responses

### Prerequisites

- [Agentic Prompting Overview](./00-agentic-prompting-overview.md)
- [Tool Definition Prompts](./01-tool-definition-prompts.md)

---

## Agent Loop Architecture

### The Basic Loop

Every agent follows this fundamental pattern:

```
┌─────────────────────────────────────────────────────────┐
│                        START                             │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Send messages + tools to model              │
└───────────────────────────┬─────────────────────────────┘
                            ▼
                    ┌───────────────┐
                    │  Model returns │
                    │    response    │
                    └───────┬───────┘
                            ▼
              ┌─────────────────────────────┐
              │     Check stop_reason       │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐        ┌───────────┐       ┌─────────┐
    │end_turn │        │ tool_use  │       │  error  │
    │  /stop  │        │           │       │         │
    └────┬────┘        └─────┬─────┘       └────┬────┘
         │                   │                  │
         ▼                   ▼                  ▼
    ┌─────────┐        ┌───────────┐       ┌─────────┐
    │  DONE   │        │  Execute  │       │ Handle  │
    │ Return  │        │   tool    │       │  error  │
    └─────────┘        └─────┬─────┘       └─────────┘
                             │
                             ▼
                       ┌───────────┐
                       │  Append   │
                       │  result   │
                       └─────┬─────┘
                             │
                             ▼
                      ┌────────────┐
                      │  LOOP BACK │
                      │  to start  │
                      └────────────┘
```

### OpenAI Implementation

```python
from openai import OpenAI

client = OpenAI()

def run_agent(user_message: str, tools: list, max_iterations: int = 10):
    """Run an agent loop with tool calling."""
    
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # Send request to model
        response = client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
        
        # Check if we're done
        if response.output.finish_reason == "stop":
            # Extract final text response
            for item in response.output:
                if item.type == "message":
                    return item.content[0].text
            return None
        
        # Process tool calls
        tool_calls = [item for item in response.output if item.type == "function_call"]
        
        if not tool_calls:
            # No tool calls and not stopped - unusual, break
            break
        
        # Add assistant response to context
        messages.append({"role": "assistant", "content": response.output})
        
        # Execute each tool call
        for tool_call in tool_calls:
            result = execute_tool(tool_call.name, tool_call.arguments)
            
            # Add tool result to context
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": json.dumps(result)
            })
    
    raise Exception(f"Agent exceeded {max_iterations} iterations")
```

### Anthropic Implementation

```python
import anthropic

client = anthropic.Anthropic()

def run_agent(user_message: str, tools: list, max_iterations: int = 10):
    """Run an agent loop with Claude."""
    
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # Send request to Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # Check if we're done
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return None
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Execute each tool and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            # Add tool results as user message
            messages.append({
                "role": "user",
                "content": tool_results
            })
    
    raise Exception(f"Agent exceeded {max_iterations} iterations")
```

---

## Context Management

### The Context Window Challenge

Every iteration adds to the context:
- Previous messages
- Tool calls (with full arguments)
- Tool results (potentially large)

This accumulates quickly and can exceed context limits.

### Strategies for Managing Context

#### 1. Summarize Tool Results

```python
def format_tool_result(result: dict, max_length: int = 2000) -> str:
    """Format tool result with size limits."""
    
    result_str = json.dumps(result, indent=2)
    
    if len(result_str) <= max_length:
        return result_str
    
    # Summarize large results
    if isinstance(result, list):
        return json.dumps({
            "summary": f"Returned {len(result)} items",
            "first_3": result[:3],
            "note": "Full results truncated. Ask for specific items if needed."
        })
    
    # Truncate with notice
    return result_str[:max_length] + "\n... (truncated)"
```

#### 2. Use Response Chaining (OpenAI)

OpenAI's `previous_response_id` maintains context server-side:

```python
def run_agent_with_chaining(user_message: str, tools: list, max_iterations: int = 10):
    """Use response chaining for efficient context management."""
    
    previous_response_id = None
    
    for iteration in range(max_iterations):
        # Build input
        if previous_response_id:
            # Continue from previous response
            input_data = [{"role": "user", "content": user_message}]
        else:
            # First request
            input_data = [{"role": "user", "content": user_message}]
        
        response = client.responses.create(
            model="gpt-4.1",
            input=input_data,
            tools=tools,
            previous_response_id=previous_response_id  # Chain context
        )
        
        previous_response_id = response.id
        
        if response.output.finish_reason == "stop":
            return extract_text(response)
        
        # Process tool calls...
        for tool_call in get_tool_calls(response):
            result = execute_tool(tool_call.name, tool_call.arguments)
            
            # Update input with tool result for next iteration
            user_message = json.dumps({
                "tool_call_id": tool_call.call_id,
                "result": result
            })
```

#### 3. Sliding Window

Keep only recent context:

```python
def manage_context_window(messages: list, max_messages: int = 20) -> list:
    """Keep most recent messages, always preserve system and first user message."""
    
    if len(messages) <= max_messages:
        return messages
    
    # Always keep: system message (if any) + first user message + recent context
    preserved = []
    
    # Keep system message
    if messages[0]["role"] == "system":
        preserved.append(messages[0])
        messages = messages[1:]
    
    # Keep first user message for original task context
    if messages[0]["role"] == "user":
        preserved.append(messages[0])
        messages = messages[1:]
    
    # Add recent messages
    recent = messages[-(max_messages - len(preserved)):]
    
    return preserved + recent
```

---

## Tool Result Handling

### Success Results

```python
def handle_tool_success(tool_name: str, result: any) -> dict:
    """Format successful tool result."""
    return {
        "status": "success",
        "tool": tool_name,
        "result": result
    }
```

### Error Results

Always return structured errors so the model can reason about them:

```python
def handle_tool_error(tool_name: str, error: Exception) -> dict:
    """Format tool error for the model."""
    
    error_type = type(error).__name__
    
    return {
        "status": "error",
        "tool": tool_name,
        "error_type": error_type,
        "message": str(error),
        "suggestion": get_error_suggestion(error_type)
    }

def get_error_suggestion(error_type: str) -> str:
    """Provide actionable suggestions for common errors."""
    
    suggestions = {
        "ConnectionError": "The service may be temporarily unavailable. Try again in a moment.",
        "TimeoutError": "The request timed out. Try with a simpler query or wait and retry.",
        "ValidationError": "The input parameters were invalid. Check the required format.",
        "NotFoundError": "The requested resource was not found. Verify the ID/name.",
        "PermissionError": "Access denied. This operation may require additional permissions."
    }
    
    return suggestions.get(error_type, "An unexpected error occurred. Try a different approach.")
```

### Handling Large Results

```python
def format_large_result(result: list, max_items: int = 5) -> dict:
    """Handle large list results intelligently."""
    
    total = len(result)
    
    if total <= max_items:
        return {"items": result, "total": total}
    
    return {
        "items": result[:max_items],
        "total": total,
        "showing": max_items,
        "note": f"Showing first {max_items} of {total} results. Use pagination or filters to see more."
    }
```

---

## The ReAct Pattern

ReAct (Reasoning + Acting) structures agent thinking explicitly:

```
Thought: I need to find information about X
Action: search(query="X")
Observation: [search results]
Thought: The results show Y, but I also need Z
Action: search(query="Z")
Observation: [more results]
Thought: Now I have enough information to answer
Action: respond(answer="...")
```

### Implementing ReAct with System Prompts

```python
REACT_SYSTEM_PROMPT = """
You are a helpful assistant that solves problems step by step.

For each step:
1. THINK: Explain your reasoning about what to do next
2. ACT: Use a tool to gather information or take action
3. OBSERVE: Analyze the tool's result

Continue until you have enough information to provide a complete answer.

When you have the final answer, respond directly without using tools.

Important:
- Always explain your thinking before acting
- If a tool returns an error, reason about alternatives
- Don't repeat the same action with the same inputs
- Ask for clarification if the request is ambiguous
"""

def run_react_agent(user_message: str, tools: list):
    """Run an agent with explicit ReAct reasoning."""
    
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    return run_agent(messages, tools)
```

### Extracting Reasoning

```python
def extract_reasoning(response) -> tuple[str, list]:
    """Separate reasoning from tool calls."""
    
    reasoning = ""
    tool_calls = []
    
    for item in response.output:
        if item.type == "message":
            reasoning = item.content[0].text
        elif item.type == "function_call":
            tool_calls.append(item)
    
    return reasoning, tool_calls
```

---

## Stopping Conditions

### Maximum Iterations

```python
MAX_ITERATIONS = 10  # Prevent infinite loops

for iteration in range(MAX_ITERATIONS):
    response = call_model()
    
    if is_done(response):
        return response
    
    process_tool_calls(response)

# If we get here, we've exceeded the limit
raise AgentLoopError("Maximum iterations exceeded")
```

### Cost-Based Limits

```python
def run_agent_with_cost_limit(
    user_message: str,
    tools: list,
    max_cost_usd: float = 1.0
):
    """Stop agent if cost exceeds limit."""
    
    total_cost = 0.0
    messages = [{"role": "user", "content": user_message}]
    
    while total_cost < max_cost_usd:
        response = client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
        
        # Calculate cost (simplified)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        iteration_cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)
        total_cost += iteration_cost
        
        if response.output.finish_reason == "stop":
            return response, total_cost
        
        # Process tools...
    
    raise CostLimitExceeded(f"Agent exceeded ${max_cost_usd} limit")
```

### Task Completion Detection

```python
def is_task_complete(response, original_task: str) -> bool:
    """Check if the agent has completed the original task."""
    
    # Model returned final answer (no tool calls)
    if response.output.finish_reason == "stop":
        return True
    
    # Check for explicit completion signals
    text = extract_text(response)
    completion_phrases = [
        "task complete",
        "i have finished",
        "here is the final",
        "the answer is"
    ]
    
    return any(phrase in text.lower() for phrase in completion_phrases)
```

---

## Parallel Tool Calls

Some models can call multiple tools simultaneously:

### Handling Parallel Calls

```python
def process_parallel_tool_calls(tool_calls: list) -> list:
    """Execute multiple tool calls in parallel."""
    
    import concurrent.futures
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tool calls
        futures = {
            executor.submit(execute_tool, tc.name, tc.arguments): tc
            for tc in tool_calls
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            tool_call = futures[future]
            try:
                result = future.result()
                results.append({
                    "tool_call_id": tool_call.call_id,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.call_id,
                    "status": "error",
                    "error": str(e)
                })
    
    return results
```

### Disabling Parallel Calls

When tool order matters, disable parallel execution:

```python
# OpenAI
response = client.responses.create(
    model="gpt-4.1",
    input=messages,
    tools=tools,
    parallel_tool_calls=False  # Force sequential execution
)
```

---

## Streaming Agent Responses

For real-time user feedback, stream agent progress:

```python
async def stream_agent(user_message: str, tools: list):
    """Stream agent reasoning and actions."""
    
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        stream = client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools,
            stream=True
        )
        
        tool_calls = []
        current_text = ""
        
        for event in stream:
            if event.type == "response.output_text.delta":
                # Stream reasoning text to user
                yield {"type": "reasoning", "text": event.delta}
                current_text += event.delta
            
            elif event.type == "response.function_call_arguments.delta":
                # Tool call being constructed
                yield {"type": "tool_call_progress", "delta": event.delta}
            
            elif event.type == "response.function_call_arguments.done":
                tool_calls.append(event.function_call)
        
        # Check if done
        if not tool_calls:
            yield {"type": "complete", "text": current_text}
            return
        
        # Execute tools and stream results
        for tc in tool_calls:
            yield {"type": "executing_tool", "name": tc.name}
            result = execute_tool(tc.name, tc.arguments)
            yield {"type": "tool_result", "name": tc.name, "result": result}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": json.dumps(result)
            })
```

---

## Common Mistakes

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No iteration limit | Always set `max_iterations` |
| Ignoring tool errors | Return structured errors to model |
| Context overflow | Summarize or truncate large results |
| No cost tracking | Monitor tokens/cost per iteration |
| Blocking on long tools | Use async execution with timeouts |
| Silent failures | Log each iteration for debugging |

---

## Hands-on Exercise

### Your Task

Build a simple research agent that:
1. Searches for information (mock search function)
2. Synthesizes findings
3. Returns a summary

The agent should:
- Use the ReAct pattern with explicit reasoning
- Handle search errors gracefully
- Stop after finding sufficient information
- Limit to 5 iterations maximum

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import json
from openai import OpenAI

client = OpenAI()

# Mock search function
def mock_search(query: str) -> dict:
    """Simulated search function."""
    results = {
        "python async": [
            {"title": "Python asyncio guide", "snippet": "asyncio is Python's library for async programming..."},
            {"title": "Async vs threading", "snippet": "Use async for I/O-bound tasks..."}
        ],
        "default": [
            {"title": "General result", "snippet": "Some information about the topic..."}
        ]
    }
    
    for key in results:
        if key in query.lower():
            return {"results": results[key], "query": query}
    
    return {"results": results["default"], "query": query}

def execute_tool(name: str, arguments: str) -> dict:
    """Execute a tool by name."""
    args = json.loads(arguments)
    
    if name == "search":
        return mock_search(args["query"])
    
    return {"error": f"Unknown tool: {name}"}

# Tool definition
tools = [
    {
        "type": "function",
        "name": "search",
        "description": "Search for information on a topic. Returns relevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]

SYSTEM_PROMPT = """
You are a research assistant. For each request:

1. THINK: Explain what information you need
2. SEARCH: Use the search tool to find information
3. ANALYZE: Review the results
4. REPEAT or CONCLUDE: Either search for more info or provide your final answer

When you have enough information, respond with a clear summary.
Maximum 3 searches per request.
"""

def run_research_agent(query: str, max_iterations: int = 5):
    """Run a research agent with ReAct pattern."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    search_count = 0
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Print reasoning
        if message.content:
            print(f"Reasoning: {message.content}")
        
        # Check if done
        if message.tool_calls is None:
            print(f"\nFinal Answer: {message.content}")
            return message.content
        
        # Process tool calls
        messages.append(message)
        
        for tool_call in message.tool_calls:
            print(f"Searching: {tool_call.function.arguments}")
            search_count += 1
            
            if search_count > 3:
                result = {"error": "Search limit reached. Please conclude with available information."}
            else:
                result = execute_tool(tool_call.function.name, tool_call.function.arguments)
            
            print(f"Results: {json.dumps(result, indent=2)[:200]}...")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    return "Research incomplete - maximum iterations reached"

# Run the agent
if __name__ == "__main__":
    result = run_research_agent("What are best practices for Python async programming?")
    print(f"\n=== Final Result ===\n{result}")
```

</details>

---

## Summary

✅ **Agent loops iterate** until task completion or limits reached
✅ **Manage context** by summarizing, truncating, or using response chaining
✅ **Handle errors gracefully** with structured error returns
✅ **ReAct pattern** structures reasoning explicitly
✅ **Set hard limits** on iterations, cost, and time

**Next:** [MCP Agentic Patterns](./03-mcp-agentic-patterns.md)

---

## Further Reading

- [OpenAI Agents Guide](https://platform.openai.com/docs/guides/agents)
- [Anthropic Agent Loop Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [ReAct Paper: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

---

<!-- 
Sources Consulted:
- OpenAI Responses API: previous_response_id, streaming
- Anthropic Tool Use: message structure, stop_reason handling
- ReAct pattern: Thought-Action-Observation cycle
-->
