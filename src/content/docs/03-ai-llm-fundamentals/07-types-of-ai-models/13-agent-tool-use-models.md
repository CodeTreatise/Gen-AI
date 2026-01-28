---
title: "Agent & Tool-Use Models"
---

# Agent & Tool-Use Models

## Introduction

Agent and tool-use models can plan multi-step tasks and execute tools to accomplish goals. These models form the foundation of AI agents that can take actions in the real world.

### What We'll Cover

- What makes a model "agentic"
- Tool/function calling
- Multi-step planning
- Agent architectures

---

## Agentic Capabilities

### What Makes a Model Agentic?

```python
agentic_capabilities = {
    "tool_use": "Can call external functions/APIs",
    "planning": "Can break down tasks into steps",
    "reasoning": "Can think through problems",
    "memory": "Can maintain context across turns",
    "self_correction": "Can recognize and fix mistakes",
    "autonomy": "Can work with minimal guidance",
}

# Traditional LLM: Text in → Text out
# Agentic LLM: Goal in → Actions + Results out
```

### The Agent Loop

```
┌─────────────────────────────────────────────────┐
│                  AGENT LOOP                      │
├─────────────────────────────────────────────────┤
│                                                  │
│   1. OBSERVE: Receive task/feedback              │
│        ↓                                         │
│   2. THINK: Analyze situation, plan next step   │
│        ↓                                         │
│   3. ACT: Choose and call a tool                │
│        ↓                                         │
│   4. OBSERVE: Get tool result                   │
│        ↓                                         │
│   5. Repeat until task complete                 │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Tool/Function Calling

### OpenAI Function Calling

```python
from openai import OpenAI

client = OpenAI()

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Call with tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

### Anthropic Tool Use

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

# Check for tool use
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
```

---

## Complete Tool Loop

### Executing Tools and Continuing

```python
import json

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Actual weather API call"""
    # Simulated response
    return {"temperature": 22, "condition": "sunny", "unit": unit}

def search_web(query: str) -> str:
    """Actual web search"""
    return f"Search results for: {query}"

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return result"""
    tools_map = {
        "get_weather": get_weather,
        "search_web": search_web,
    }
    
    if name in tools_map:
        result = tools_map[name](**args)
        return json.dumps(result)
    return json.dumps({"error": f"Unknown tool: {name}"})

def agent_loop(user_message: str, tools: list) -> str:
    """Run agent loop until completion"""
    
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # Check if done (no tool calls)
        if not assistant_message.tool_calls:
            return assistant_message.content
        
        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# Run agent
result = agent_loop("What's the weather in Tokyo and NYC?", tools)
print(result)
```

---

## Multi-Step Planning

### ReAct Pattern

```python
class ReActAgent:
    """Reasoning and Acting agent"""
    
    def __init__(self, tools: list):
        self.tools = tools
        self.client = OpenAI()
    
    def run(self, task: str, max_steps: int = 10) -> str:
        """Execute task with ReAct pattern"""
        
        system_prompt = """You are an AI agent that solves tasks step by step.

For each step:
1. THOUGHT: Reason about what to do next
2. ACTION: Choose a tool and arguments
3. OBSERVATION: Analyze the result

Continue until the task is complete, then provide FINAL ANSWER."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # Check for completion
            if not message.tool_calls:
                if "FINAL ANSWER" in message.content:
                    return message.content
            
            # Execute tools
            for tool_call in message.tool_calls or []:
                result = self._execute(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        return "Max steps reached"
    
    def _execute(self, tool_call) -> str:
        # Tool execution logic
        pass
```

### Plan-and-Execute

```python
class PlanExecuteAgent:
    """Plan first, then execute"""
    
    def run(self, task: str) -> str:
        # Step 1: Create plan
        plan = self._create_plan(task)
        print(f"Plan: {plan}")
        
        results = []
        
        # Step 2: Execute each step
        for step in plan["steps"]:
            result = self._execute_step(step, results)
            results.append(result)
        
        # Step 3: Synthesize final answer
        return self._synthesize(task, results)
    
    def _create_plan(self, task: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Create a step-by-step plan for this task:
                
Task: {task}

Return JSON: {{"steps": ["step 1", "step 2", ...]}}"""
            }],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

---

## Agent Architectures

### Single Agent

```
User → [Agent] → Tools → Result
         ↑         ↓
         └─────────┘
         (feedback loop)
```

### Multi-Agent

```
User → [Orchestrator Agent]
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
[Research] [Code] [Review]
  Agent    Agent   Agent
    ↓         ↓         ↓
    └─────────┼─────────┘
              ↓
         Final Result
```

### Hierarchical

```
[Supervisor Agent]
       ↓
   ┌───┴───┐
   ↓       ↓
[Team A] [Team B]
   ↓       ↓
[Worker] [Worker]
[Worker] [Worker]
```

---

## Model Comparison for Agents

| Model | Tool Use | Planning | Reasoning |
|-------|----------|----------|-----------|
| GPT-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Claude 3.5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Gemini 1.5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| GPT-4o-mini | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## Hands-on Exercise

### Your Task

Build a simple research agent:

```python
from openai import OpenAI
import json

client = OpenAI()

class ResearchAgent:
    """Agent that can search and summarize"""
    
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize",
                    "description": "Summarize text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "length": {"type": "string", "enum": ["short", "medium", "long"]}
                        },
                        "required": ["text"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "search":
            # Simulated search
            return f"Found information about {args['query']}: [simulated results]"
        elif name == "summarize":
            return f"Summary of text: [simulated summary]"
        return "Unknown tool"
    
    def research(self, topic: str) -> str:
        messages = [
            {"role": "system", "content": "You are a research assistant. Use tools to find and summarize information."},
            {"role": "user", "content": f"Research this topic: {topic}"}
        ]
        
        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                return message.content
            
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = self._execute_tool(tool_call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

# Test
agent = ResearchAgent()
result = agent.research("Latest developments in quantum computing")
print(result)
```

---

## Summary

✅ **Agentic models** can plan and use tools

✅ **Function calling** is the foundation of tool use

✅ **Agent loop**: Observe → Think → Act → Repeat

✅ **ReAct pattern**: Interleave reasoning and action

✅ **GPT-4o and Claude 3.5**: Best for agentic tasks

**Next:** [Computer Use Models](./14-computer-use-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Document Understanding](./12-document-understanding-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Computer Use](./14-computer-use-models.md) |

