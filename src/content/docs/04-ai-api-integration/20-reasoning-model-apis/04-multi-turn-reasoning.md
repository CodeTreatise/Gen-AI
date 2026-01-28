---
title: "Multi-Turn Reasoning"
---

# Multi-Turn Reasoning

## Introduction

Reasoning models can maintain their thinking process across multiple conversation turns. By passing encrypted reasoning items back to the API, you enable the model to build upon previous analysis without repeating work—reducing latency and costs while improving coherence.

### What We'll Cover

- Passing reasoning items between turns
- Using `include` for encrypted content
- Context preservation strategies
- Maintaining reasoning state with tool calls

### Prerequisites

- Reasoning models overview
- Understanding of conversation context
- Basic API multi-turn patterns

---

## Passing Reasoning Items

### Understanding Reasoning Continuity

```python
from dataclasses import dataclass
from typing import List, Optional, Any
from enum import Enum


class ReasoningItemType(str, Enum):
    """Types of reasoning-related items in responses."""
    
    REASONING = "reasoning"
    MESSAGE = "message"
    TOOL_CALL = "function_call"


@dataclass
class ReasoningItem:
    """Represents a reasoning item from response."""
    
    item_type: str
    id: str
    encrypted_content: Optional[str] = None
    summary: Optional[str] = None
    status: str = "completed"


@dataclass
class ConversationTurn:
    """A single turn in multi-turn reasoning."""
    
    turn_number: int
    user_input: str
    reasoning_items: List[ReasoningItem]
    output_text: str
    reasoning_preserved: bool = False


# Why preserve reasoning across turns?
REASONING_CONTINUITY_BENEFITS = [
    {
        "benefit": "Reduced latency",
        "description": "Model doesn't re-derive conclusions from scratch",
        "impact": "30-50% faster responses on follow-up turns"
    },
    {
        "benefit": "Cost efficiency",
        "description": "Less reasoning tokens spent on repeated analysis",
        "impact": "20-40% token savings per turn"
    },
    {
        "benefit": "Improved coherence",
        "description": "Consistent reasoning thread throughout conversation",
        "impact": "Higher quality multi-step problem solving"
    },
    {
        "benefit": "Better context",
        "description": "Model remembers why it made previous decisions",
        "impact": "More accurate follow-up responses"
    }
]


print("Benefits of Preserving Reasoning Across Turns")
print("=" * 60)

for item in REASONING_CONTINUITY_BENEFITS:
    print(f"\n✅ {item['benefit']}")
    print(f"   {item['description']}")
    print(f"   📊 Impact: {item['impact']}")


print("""

📊 Multi-Turn Reasoning Flow

Turn 1:
┌─────────────┐
│ User Input  │
└──────┬──────┘
       ↓
┌─────────────────────────────────┐
│ Model Reasoning (encrypted)     │ ← Generated
│ + Output Message                │
└──────┬──────────────────────────┘
       ↓
Turn 2:
┌─────────────┐   ┌──────────────────┐
│ User Input  │ + │ Prev. Reasoning  │ ← Passed back
└──────┬──────┘   └────────┬─────────┘
       └─────────┬─────────┘
                 ↓
┌─────────────────────────────────┐
│ Model builds on prev. reasoning │ ← Continued
│ + New Output                    │
└─────────────────────────────────┘
""")
```

---

## The Include Parameter

### Requesting Encrypted Reasoning

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class IncludeOption:
    """An option for the include parameter."""
    
    value: str
    description: str
    use_case: str
    required_for: str


INCLUDE_OPTIONS = [
    IncludeOption(
        value="reasoning.encrypted_content",
        description="Include encrypted reasoning for multi-turn",
        use_case="Preserve reasoning across conversation turns",
        required_for="Multi-turn reasoning, conversation continuity"
    )
]


class MultiTurnReasoningClient:
    """Client for multi-turn reasoning conversations."""
    
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.conversation_items: List[Dict[str, Any]] = []
        self.reasoning_items: List[Dict[str, Any]] = []
    
    def create_request(
        self,
        user_message: str,
        include_reasoning: bool = True
    ) -> dict:
        """Create a multi-turn request with reasoning preservation."""
        
        # Build input items
        input_items = []
        
        # Add previous conversation items
        input_items.extend(self.conversation_items)
        
        # Add previous reasoning items (for continuity)
        if include_reasoning:
            input_items.extend(self.reasoning_items)
        
        # Add new user message
        input_items.append({
            "type": "message",
            "role": "user",
            "content": user_message
        })
        
        # Build request
        request = {
            "model": self.model,
            "input": input_items
        }
        
        # Include encrypted reasoning in response
        if include_reasoning:
            request["include"] = ["reasoning.encrypted_content"]
        
        return request
    
    def process_response(self, response: dict) -> str:
        """Process response and extract reasoning items."""
        
        output_text = ""
        new_reasoning = []
        new_messages = []
        
        for item in response.get("output", []):
            item_type = item.get("type", "")
            
            if item_type == "reasoning":
                # Store reasoning item for next turn
                new_reasoning.append({
                    "type": "reasoning",
                    "id": item.get("id", ""),
                    "encrypted_content": item.get("encrypted_content", "")
                })
            
            elif item_type == "message":
                # Extract text content
                content = item.get("content", [])
                for part in content:
                    if part.get("type") == "output_text":
                        output_text = part.get("text", "")
                
                # Store message for context
                new_messages.append({
                    "type": "message",
                    "role": "assistant",
                    "content": output_text
                })
        
        # Update state
        self.reasoning_items = new_reasoning
        self.conversation_items.extend(new_messages)
        
        return output_text
    
    def get_state(self) -> dict:
        """Get current conversation state."""
        
        return {
            "message_count": len(self.conversation_items),
            "has_reasoning": len(self.reasoning_items) > 0,
            "reasoning_items_count": len(self.reasoning_items)
        }


print("\n\nMulti-Turn Request Structure")
print("=" * 60)

client = MultiTurnReasoningClient("gpt-5")

# First turn
request1 = client.create_request("What is the square root of 144?")

print("\n📤 Turn 1 Request:")
print(f"   Model: {request1['model']}")
print(f"   Include: {request1.get('include', [])}")
print(f"   Input items: {len(request1['input'])}")

# Simulate response
mock_response1 = {
    "output": [
        {
            "type": "reasoning",
            "id": "reasoning_001",
            "encrypted_content": "eyJ0aGlua2luZyI6ICJTcXVhcmUgcm9vdCBvZiAxNDQuLi4ifQ=="
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The square root of 144 is 12."}]
        }
    ]
}

output1 = client.process_response(mock_response1)
print(f"\n📥 Turn 1 Response: {output1}")
print(f"   State: {client.get_state()}")

# Second turn (with reasoning context)
request2 = client.create_request("What about 169?")

print("\n📤 Turn 2 Request:")
print(f"   Input items: {len(request2['input'])}")
print(f"   Includes reasoning from turn 1: ✅")
```

### Complete Multi-Turn Flow

```python
@dataclass
class ConversationSession:
    """Manages a multi-turn reasoning session."""
    
    session_id: str
    model: str
    reasoning_items: List[dict] = field(default_factory=list)
    messages: List[dict] = field(default_factory=list)
    turn_count: int = 0
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append({
            "type": "message",
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.messages.append({
            "type": "message",
            "role": "assistant",
            "content": content
        })
    
    def update_reasoning(self, items: List[dict]):
        """Update reasoning items (replaces previous)."""
        self.reasoning_items = items
        self.turn_count += 1
    
    def build_input(self) -> List[dict]:
        """Build input for next request."""
        # Combine messages and current reasoning
        return self.messages + self.reasoning_items


from dataclasses import field


class ReasoningConversationManager:
    """Manage multi-turn reasoning conversations."""
    
    def __init__(self, default_model: str = "gpt-5"):
        self.default_model = default_model
        self.sessions: Dict[str, ConversationSession] = {}
    
    def create_session(self, session_id: str, model: str = None) -> ConversationSession:
        """Create a new conversation session."""
        
        session = ConversationSession(
            session_id=session_id,
            model=model or self.default_model
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve an existing session."""
        return self.sessions.get(session_id)
    
    def prepare_request(
        self,
        session_id: str,
        user_message: str,
        effort: str = "medium"
    ) -> dict:
        """Prepare a multi-turn request."""
        
        session = self.sessions[session_id]
        session.add_user_message(user_message)
        
        return {
            "model": session.model,
            "input": session.build_input(),
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": effort}
        }
    
    def handle_response(
        self,
        session_id: str,
        response: dict
    ) -> str:
        """Handle response and update session."""
        
        session = self.sessions[session_id]
        
        output_text = ""
        reasoning_items = []
        
        for item in response.get("output", []):
            if item.get("type") == "reasoning":
                reasoning_items.append({
                    "type": "reasoning",
                    "id": item.get("id"),
                    "encrypted_content": item.get("encrypted_content")
                })
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        output_text = part.get("text", "")
        
        session.add_assistant_message(output_text)
        session.update_reasoning(reasoning_items)
        
        return output_text


print("\n\nConversation Session Management")
print("=" * 60)

manager = ReasoningConversationManager("gpt-5")
session = manager.create_session("demo_session")

print(f"\n📝 Session created: {session.session_id}")
print(f"   Model: {session.model}")
print(f"   Initial state: {session.turn_count} turns")

# Demonstrate request preparation
request = manager.prepare_request(
    "demo_session",
    "Explain the concept of recursion"
)

print(f"\n📤 Request prepared:")
print(f"   Model: {request['model']}")
print(f"   Include: {request['include']}")
print(f"   Reasoning effort: {request['reasoning']['effort']}")
```

---

## Context Preservation Strategies

### Managing Conversation Length

```python
@dataclass
class ContextConfig:
    """Configuration for context management."""
    
    max_messages: int = 20
    max_total_tokens: int = 100000
    preserve_system: bool = True
    preserve_reasoning: bool = True
    truncation_strategy: str = "sliding_window"


class ContextManager:
    """Manage conversation context for multi-turn reasoning."""
    
    def __init__(self, config: ContextConfig):
        self.config = config
    
    def prepare_context(
        self,
        messages: List[dict],
        reasoning_items: List[dict],
        system_message: Optional[str] = None
    ) -> List[dict]:
        """Prepare context within limits."""
        
        items = []
        
        # Always include system message first
        if system_message and self.config.preserve_system:
            items.append({
                "type": "message",
                "role": "system",
                "content": system_message
            })
        
        # Apply truncation strategy to messages
        truncated_messages = self._truncate_messages(messages)
        items.extend(truncated_messages)
        
        # Include reasoning items (latest only)
        if self.config.preserve_reasoning and reasoning_items:
            items.extend(reasoning_items)
        
        return items
    
    def _truncate_messages(self, messages: List[dict]) -> List[dict]:
        """Truncate messages based on strategy."""
        
        if len(messages) <= self.config.max_messages:
            return messages
        
        if self.config.truncation_strategy == "sliding_window":
            # Keep most recent messages
            return messages[-self.config.max_messages:]
        
        elif self.config.truncation_strategy == "summarize_old":
            # Keep first few + recent
            keep_start = 3
            keep_end = self.config.max_messages - keep_start - 1
            
            result = messages[:keep_start]
            result.append({
                "type": "message",
                "role": "system",
                "content": f"[{len(messages) - keep_start - keep_end} messages summarized]"
            })
            result.extend(messages[-keep_end:])
            return result
        
        return messages[-self.config.max_messages:]
    
    def should_start_fresh(
        self,
        messages: List[dict],
        reasoning_items: List[dict]
    ) -> dict:
        """Check if conversation should start fresh."""
        
        reasons = []
        
        if len(messages) > self.config.max_messages * 1.5:
            reasons.append("Message count too high")
        
        # Estimate token count (rough)
        estimated_tokens = sum(
            len(str(m.get("content", ""))) // 4 
            for m in messages
        )
        
        if estimated_tokens > self.config.max_total_tokens * 0.8:
            reasons.append("Token count approaching limit")
        
        return {
            "should_reset": len(reasons) > 0,
            "reasons": reasons,
            "recommendation": "Start new session with summary" if reasons else "Continue"
        }


print("\n\nContext Management Strategies")
print("=" * 60)

config = ContextConfig(
    max_messages=10,
    truncation_strategy="sliding_window"
)

context_mgr = ContextManager(config)

# Simulate long conversation
long_conversation = [
    {"type": "message", "role": "user", "content": f"Message {i}"}
    for i in range(25)
]

truncated = context_mgr.prepare_context(
    messages=long_conversation,
    reasoning_items=[{"type": "reasoning", "id": "r1"}],
    system_message="You are a helpful assistant."
)

print(f"\n📊 Context Preparation:")
print(f"   Original messages: {len(long_conversation)}")
print(f"   After truncation: {len(truncated)}")
print(f"   Strategy: {config.truncation_strategy}")

check = context_mgr.should_start_fresh(long_conversation, [])
print(f"\n🔍 Fresh start check:")
print(f"   Should reset: {check['should_reset']}")
if check['reasons']:
    for reason in check['reasons']:
        print(f"   • {reason}")
print(f"   Recommendation: {check['recommendation']}")
```

---

## Reasoning with Tool Calls

### Maintaining State During Tool Use

```python
from typing import Callable


@dataclass
class ToolResult:
    """Result from a tool call."""
    
    tool_id: str
    result: Any
    success: bool
    error: Optional[str] = None


class ReasoningToolHandler:
    """Handle tool calls while preserving reasoning state."""
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.pending_reasoning: List[dict] = []
    
    def process_response(self, response: dict) -> dict:
        """Process response, handling tool calls and reasoning."""
        
        output_text = None
        tool_calls = []
        reasoning = []
        is_complete = True
        
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                reasoning.append({
                    "type": "reasoning",
                    "id": item.get("id"),
                    "encrypted_content": item.get("encrypted_content")
                })
            
            elif item_type == "function_call":
                is_complete = False
                tool_calls.append({
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", {})
                })
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        output_text = part.get("text")
        
        # Store reasoning for next call
        self.pending_reasoning = reasoning
        
        return {
            "output_text": output_text,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "is_complete": is_complete
        }
    
    def execute_tools(self, tool_calls: List[dict]) -> List[dict]:
        """Execute tool calls and format results."""
        
        results = []
        
        for call in tool_calls:
            tool_name = call["name"]
            
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](**call["arguments"])
                    results.append({
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": str(result)
                    })
                except Exception as e:
                    results.append({
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": f"Error: {str(e)}"
                    })
            else:
                results.append({
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": f"Unknown tool: {tool_name}"
                })
        
        return results
    
    def build_continuation_input(
        self,
        previous_input: List[dict],
        tool_results: List[dict]
    ) -> List[dict]:
        """Build input for continuation after tool calls."""
        
        # Include previous input
        new_input = previous_input.copy()
        
        # Add tool results
        new_input.extend(tool_results)
        
        # Include preserved reasoning
        new_input.extend(self.pending_reasoning)
        
        return new_input


print("\n\nTool Calls with Reasoning Preservation")
print("=" * 60)

# Define sample tools
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72°F, Sunny"

def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"

tools = {
    "get_weather": get_weather,
    "calculate": calculate
}

handler = ReasoningToolHandler(tools)

# Simulate response with tool call
mock_response = {
    "output": [
        {
            "type": "reasoning",
            "id": "reason_1",
            "encrypted_content": "base64encodedcontent..."
        },
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": {"city": "San Francisco"}
        }
    ]
}

result = handler.process_response(mock_response)

print(f"\n📤 Response Processing:")
print(f"   Is complete: {result['is_complete']}")
print(f"   Tool calls: {len(result['tool_calls'])}")
print(f"   Reasoning items preserved: {len(result['reasoning'])}")

if result['tool_calls']:
    # Execute tools
    tool_results = handler.execute_tools(result['tool_calls'])
    
    print(f"\n🔧 Tool Execution:")
    for res in tool_results:
        print(f"   {res['call_id']}: {res['output']}")
    
    # Build continuation
    prev_input = [{"type": "message", "role": "user", "content": "What's the weather?"}]
    continuation = handler.build_continuation_input(prev_input, tool_results)
    
    print(f"\n📤 Continuation Input:")
    print(f"   Total items: {len(continuation)}")
    print(f"   Includes reasoning: ✅")
```

---

## Best Practices for Multi-Turn Reasoning

### Optimization Guidelines

```python
MULTI_TURN_BEST_PRACTICES = [
    {
        "practice": "Always include reasoning on follow-ups",
        "why": "Prevents re-derivation of conclusions",
        "how": 'include=["reasoning.encrypted_content"]',
        "impact": "30-50% faster, 20-40% cheaper"
    },
    {
        "practice": "Use consistent effort levels",
        "why": "Maintains reasoning depth across turns",
        "how": "Set effort at session start, adjust rarely",
        "impact": "More coherent multi-step reasoning"
    },
    {
        "practice": "Truncate old messages, keep recent reasoning",
        "why": "Reasoning items are more valuable than old context",
        "how": "Sliding window on messages, always pass latest reasoning",
        "impact": "Optimal context utilization"
    },
    {
        "practice": "Start fresh on topic changes",
        "why": "Old reasoning may not apply to new topics",
        "how": "Detect topic shifts, reset session when needed",
        "impact": "Prevents confusion, cleaner reasoning"
    },
    {
        "practice": "Preserve reasoning during tool calls",
        "why": "Tool results should feed into ongoing reasoning",
        "how": "Include reasoning items in tool call continuation",
        "impact": "Maintains coherent tool-assisted reasoning"
    }
]


print("Multi-Turn Reasoning Best Practices")
print("=" * 60)

for i, bp in enumerate(MULTI_TURN_BEST_PRACTICES, 1):
    print(f"\n{i}. {bp['practice']}")
    print(f"   Why: {bp['why']}")
    print(f"   How: {bp['how']}")
    print(f"   📊 Impact: {bp['impact']}")
```

---

## Hands-on Exercise

### Your Task

Build a multi-turn reasoning conversation system that properly preserves reasoning across turns, handles context limits, and manages tool calls.

### Requirements

1. Manage conversation sessions with reasoning preservation
2. Implement context truncation when needed
3. Handle tool calls while maintaining reasoning state
4. Track conversation quality metrics

<details>
<summary>💡 Hints</summary>

- Store reasoning items separately from messages
- Use session IDs to manage multiple conversations
- Truncate messages before reasoning items
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import json


class SessionState(str, Enum):
    """State of a conversation session."""
    
    ACTIVE = "active"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SessionMetrics:
    """Metrics for a session."""
    
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    tool_calls_made: int = 0
    reasoning_preserved_count: int = 0
    
    def add_usage(self, usage: dict):
        """Add usage from a response."""
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_reasoning_tokens += usage.get(
            "output_tokens_details", {}
        ).get("reasoning_tokens", 0)


@dataclass
class ReasoningSession:
    """A multi-turn reasoning session."""
    
    session_id: str
    model: str
    created_at: datetime
    state: SessionState = SessionState.ACTIVE
    messages: List[dict] = field(default_factory=list)
    reasoning_items: List[dict] = field(default_factory=list)
    pending_tool_calls: List[dict] = field(default_factory=list)
    system_message: Optional[str] = None
    effort: str = "medium"
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    max_messages: int = 20


class MultiTurnReasoningSystem:
    """Complete multi-turn reasoning conversation system."""
    
    def __init__(self, default_model: str = "gpt-5"):
        self.default_model = default_model
        self.sessions: Dict[str, ReasoningSession] = {}
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, func: Callable, schema: dict):
        """Register a tool for use in conversations."""
        self.tools[name] = {"function": func, "schema": schema}
    
    def create_session(
        self,
        session_id: str,
        model: str = None,
        system_message: str = None,
        effort: str = "medium"
    ) -> ReasoningSession:
        """Create a new reasoning session."""
        
        session = ReasoningSession(
            session_id=session_id,
            model=model or self.default_model,
            created_at=datetime.now(),
            system_message=system_message,
            effort=effort
        )
        
        self.sessions[session_id] = session
        return session
    
    def send_message(
        self,
        session_id: str,
        user_message: str
    ) -> dict:
        """Send a message and get the API request."""
        
        session = self.sessions[session_id]
        
        if session.state == SessionState.WAITING_TOOL:
            return {
                "error": "Session waiting for tool results",
                "pending_tools": session.pending_tool_calls
            }
        
        # Add user message
        session.messages.append({
            "type": "message",
            "role": "user",
            "content": user_message
        })
        
        # Build input
        input_items = self._build_input(session)
        
        # Build request
        request = {
            "model": session.model,
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": session.effort}
        }
        
        # Add tools if registered
        if self.tools:
            request["tools"] = [
                {"type": "function", "function": t["schema"]}
                for t in self.tools.values()
            ]
        
        return {"request": request, "session_id": session_id}
    
    def _build_input(self, session: ReasoningSession) -> List[dict]:
        """Build input items with truncation."""
        
        items = []
        
        # System message first
        if session.system_message:
            items.append({
                "type": "message",
                "role": "system",
                "content": session.system_message
            })
        
        # Truncate messages if needed
        messages = session.messages
        if len(messages) > session.max_messages:
            # Keep first 2 + last (max - 3)
            keep_end = session.max_messages - 3
            messages = messages[:2] + [
                {"type": "message", "role": "system", 
                 "content": f"[{len(session.messages) - 2 - keep_end} messages omitted]"}
            ] + messages[-keep_end:]
        
        items.extend(messages)
        
        # Add reasoning items (always include latest)
        items.extend(session.reasoning_items)
        
        return items
    
    def process_response(
        self,
        session_id: str,
        response: dict
    ) -> dict:
        """Process API response and update session."""
        
        session = self.sessions[session_id]
        
        output_text = None
        tool_calls = []
        reasoning = []
        
        # Update metrics
        if "usage" in response:
            session.metrics.add_usage(response["usage"])
        
        # Parse output
        for item in response.get("output", []):
            item_type = item.get("type")
            
            if item_type == "reasoning":
                reasoning.append({
                    "type": "reasoning",
                    "id": item.get("id"),
                    "encrypted_content": item.get("encrypted_content")
                })
            
            elif item_type == "function_call":
                tool_calls.append({
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", {})
                })
            
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        output_text = part.get("text")
        
        # Update session state
        session.reasoning_items = reasoning
        session.metrics.turn_count += 1
        
        if reasoning:
            session.metrics.reasoning_preserved_count += 1
        
        if tool_calls:
            session.state = SessionState.WAITING_TOOL
            session.pending_tool_calls = tool_calls
            session.metrics.tool_calls_made += len(tool_calls)
            
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "reasoning_preserved": len(reasoning) > 0
            }
        
        if output_text:
            session.messages.append({
                "type": "message",
                "role": "assistant",
                "content": output_text
            })
        
        return {
            "type": "message",
            "content": output_text,
            "reasoning_preserved": len(reasoning) > 0
        }
    
    def submit_tool_results(
        self,
        session_id: str,
        tool_results: Dict[str, Any]
    ) -> dict:
        """Submit tool results and continue conversation."""
        
        session = self.sessions[session_id]
        
        if session.state != SessionState.WAITING_TOOL:
            return {"error": "No pending tool calls"}
        
        # Build tool result items
        result_items = []
        for call in session.pending_tool_calls:
            call_id = call["call_id"]
            result = tool_results.get(call_id, "No result provided")
            
            result_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(result)
            })
        
        # Build continuation request
        input_items = self._build_input(session)
        input_items.extend(result_items)
        
        # Clear pending and update state
        session.pending_tool_calls = []
        session.state = SessionState.ACTIVE
        
        request = {
            "model": session.model,
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": session.effort}
        }
        
        if self.tools:
            request["tools"] = [
                {"type": "function", "function": t["schema"]}
                for t in self.tools.values()
            ]
        
        return {"request": request, "session_id": session_id}
    
    def get_session_info(self, session_id: str) -> dict:
        """Get session information and metrics."""
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "model": session.model,
            "state": session.state.value,
            "message_count": len(session.messages),
            "has_reasoning": len(session.reasoning_items) > 0,
            "metrics": {
                "turns": session.metrics.turn_count,
                "input_tokens": session.metrics.total_input_tokens,
                "output_tokens": session.metrics.total_output_tokens,
                "reasoning_tokens": session.metrics.total_reasoning_tokens,
                "tool_calls": session.metrics.tool_calls_made,
                "reasoning_preserved": session.metrics.reasoning_preserved_count
            }
        }
    
    def should_reset_session(self, session_id: str) -> dict:
        """Check if session should be reset."""
        
        session = self.sessions[session_id]
        
        reasons = []
        
        if len(session.messages) > session.max_messages * 2:
            reasons.append("Too many messages")
        
        if session.metrics.turn_count > 50:
            reasons.append("Too many turns")
        
        estimated_tokens = (
            session.metrics.total_input_tokens + 
            session.metrics.total_output_tokens
        )
        if estimated_tokens > 500000:
            reasons.append("High token usage")
        
        return {
            "should_reset": len(reasons) > 0,
            "reasons": reasons,
            "suggestion": "Create new session with summary" if reasons else "Continue"
        }


# Demo the system
print("\nMulti-Turn Reasoning System Demo")
print("=" * 60)

system = MultiTurnReasoningSystem("gpt-5")

# Register a tool
system.register_tool(
    "calculate",
    lambda expression: eval(expression),
    {
        "name": "calculate",
        "description": "Calculate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        }
    }
)

# Create session
session = system.create_session(
    "demo_session",
    system_message="You are a helpful math tutor.",
    effort="medium"
)

print(f"\n✅ Session created: {session.session_id}")

# Send first message
result = system.send_message("demo_session", "What is 15 * 23?")
print(f"\n📤 Request prepared (Turn 1)")
print(f"   Input items: {len(result['request']['input'])}")

# Simulate response with tool call
mock_response = {
    "output": [
        {"type": "reasoning", "id": "r1", "encrypted_content": "abc123"},
        {"type": "function_call", "call_id": "c1", "name": "calculate", 
         "arguments": {"expression": "15 * 23"}}
    ],
    "usage": {"input_tokens": 100, "output_tokens": 500,
              "output_tokens_details": {"reasoning_tokens": 400}}
}

result = system.process_response("demo_session", mock_response)
print(f"\n📥 Response (tool call)")
print(f"   Type: {result['type']}")
print(f"   Reasoning preserved: {result['reasoning_preserved']}")

# Submit tool result
tool_results = {"c1": "345"}
result = system.submit_tool_results("demo_session", tool_results)
print(f"\n📤 Continuation request prepared")

# Simulate final response
final_response = {
    "output": [
        {"type": "reasoning", "id": "r2", "encrypted_content": "def456"},
        {"type": "message", "role": "assistant", 
         "content": [{"type": "output_text", "text": "15 × 23 = 345"}]}
    ],
    "usage": {"input_tokens": 150, "output_tokens": 300,
              "output_tokens_details": {"reasoning_tokens": 250}}
}

result = system.process_response("demo_session", final_response)
print(f"\n📥 Final response: {result['content']}")

# Check session info
info = system.get_session_info("demo_session")
print(f"\n📊 Session Metrics:")
print(f"   Turns: {info['metrics']['turns']}")
print(f"   Reasoning tokens: {info['metrics']['reasoning_tokens']}")
print(f"   Tool calls: {info['metrics']['tool_calls']}")
print(f"   Reasoning preserved: {info['metrics']['reasoning_preserved']} times")
```

</details>

---

## Summary

✅ Use `include: ["reasoning.encrypted_content"]` to preserve reasoning  
✅ Pass reasoning items back in subsequent turns for continuity  
✅ Truncate old messages but always keep latest reasoning  
✅ Preserve reasoning state during tool call loops  
✅ Start fresh sessions when topics change significantly

**Next:** [Encrypted Reasoning](./05-encrypted-reasoning.md)

---

## Further Reading

- [OpenAI Multi-Turn Conversations](https://platform.openai.com/docs/guides/reasoning) — Official guide
- [Responses API](https://platform.openai.com/docs/api-reference/responses) — API reference
- [Tool Use](https://platform.openai.com/docs/guides/function-calling) — Function calling guide
