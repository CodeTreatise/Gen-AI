---
title: "Streaming Thinking Blocks"
---

# Streaming Thinking Blocks

## Introduction

When reasoning takes 30-60+ seconds, users need feedback. Streaming thinking blocks provides progressive updates during extended reasoning, improving the user experience.

### What We'll Cover

- Showing thinking process to users
- Hiding vs. revealing reasoning
- UX patterns for thinking display
- Progressive disclosure

---

## Streaming with Claude

### Basic Streaming

```python
from anthropic import Anthropic

client = Anthropic()

def stream_thinking(problem: str):
    """Stream both thinking and answer"""
    
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{"role": "user", "content": problem}]
    ) as stream:
        current_block = None
        
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    print("\n🤔 Thinking...")
                    current_block = "thinking"
                elif event.content_block.type == "text":
                    print("\n✅ Answer:")
                    current_block = "text"
            
            elif event.type == "content_block_delta":
                if hasattr(event.delta, 'thinking'):
                    print(event.delta.thinking, end="", flush=True)
                elif hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)

# Usage
stream_thinking("Explain why the sky is blue")
```

### Async Streaming

```python```python
import asyncio
from anthropic import AsyncAnthropic

async_client = AsyncAnthropic()

async def stream_thinking_async(problem: str):
    """Async streaming for web applications"""
    
    async with async_client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": problem}]
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'thinking'):
                    yield {"type": "thinking", "content": event.delta.thinking}
                elif hasattr(event.delta, 'text'):
                    yield {"type": "answer", "content": event.delta.text}

# Usage with FastAPI
async def websocket_handler(websocket, problem: str):
    async for chunk in stream_thinking_async(problem):
        await websocket.send_json(chunk)
```

---

## UX Patterns for Thinking Display

### Pattern 1: Hidden Thinking (Production Default)

```
┌─────────────────────────────────────┐
│  🤔 Thinking deeply...              │
│  ████████░░░░░░░░ 45 seconds        │
│                                     │
│  [Show thinking process]            │
└─────────────────────────────────────┘
```

Best for: End-user applications where internal reasoning is distracting.

### Pattern 2: Progressive Disclosure

```
┌─────────────────────────────────────┐
│  🤔 Analyzing your question...      │
│                                     │
│  ▼ Step 1: Understanding context ✓  │
│  ▼ Step 2: Evaluating options...    │
│    └── Considering 3 approaches     │
│  ○ Step 3: Formulating response     │
└─────────────────────────────────────┘
```

Best for: Complex queries where users want visibility into progress.

### Pattern 3: Full Transparency (Developer Tools)

```
┌─────────────────────────────────────┐
│  THINKING:                          │
│  Let me analyze this step by step.  │
│  First, I need to consider...       │
│  The key factors are...             │
│  ─────────────────────────────────  │
│  ANSWER:                            │
│  Based on my analysis...            │
└─────────────────────────────────────┘
```

Best for: Debugging, auditing, developer-facing tools.

---

## Implementation: React Component

```typescript
import { useState, useEffect } from 'react';

interface ThinkingDisplayProps {
  problem: string;
  showThinking?: boolean;
}

function ThinkingDisplay({ problem, showThinking = false }: ThinkingDisplayProps) {
  const [thinking, setThinking] = useState('');
  const [answer, setAnswer] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [expanded, setExpanded] = useState(showThinking);

  useEffect(() => {
    const eventSource = new EventSource(`/api/reason?q=${encodeURIComponent(problem)}`);
    setIsThinking(true);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'thinking') {
        setThinking(prev => prev + data.content);
      } else if (data.type === 'answer') {
        setIsThinking(false);
        setAnswer(prev => prev + data.content);
      }
    };

    return () => eventSource.close();
  }, [problem]);

  return (
    <div className="thinking-display">
      {isThinking && (
        <div className="thinking-indicator">
          <span className="spinner" />
          <span>Thinking deeply...</span>
          <button onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Hide' : 'Show'} thinking
          </button>
        </div>
      )}
      
      {expanded && thinking && (
        <pre className="thinking-content">{thinking}</pre>
      )}
      
      {answer && (
        <div className="answer-content">{answer}</div>
      )}
    </div>
  );
}
```

---

## Handling Long-Running Reasoning

### Timeout Management

```python
import asyncio
from anthropic import Anthropic

client = Anthropic()

async def stream_with_timeout(problem: str, timeout_seconds: int = 120):
    """Stream with graceful timeout handling"""
    
    try:
        async with asyncio.timeout(timeout_seconds):
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
                messages=[{"role": "user", "content": problem}]
            ) as stream:
                for event in stream:
                    yield event
                    
    except asyncio.TimeoutError:
        yield {
            "type": "error",
            "message": "Reasoning exceeded time limit. Consider breaking into smaller steps."
        }
```

### Progress Estimation

```python
def estimate_thinking_time(budget_tokens: int, complexity: str) -> int:
    """Estimate thinking duration based on budget and complexity"""
    
    base_rates = {
        "simple": 500,    # tokens per second
        "medium": 200,
        "complex": 100
    }
    
    rate = base_rates.get(complexity, 200)
    estimated_seconds = budget_tokens / rate
    
    return int(estimated_seconds * 1.2)  # Add 20% buffer
```

---

## Summary

✅ Streaming provides real-time feedback during extended reasoning  
✅ Choose UX pattern based on audience (hidden, progressive, or transparent)  
✅ Implement timeout handling for long-running reasoning  
✅ Use async streaming for web applications  

**Next:** [Verifiable Reasoning](./08-verifiable-reasoning.md)

---

## Further Reading

- [Anthropic Streaming Documentation](https://docs.anthropic.com/en/api/streaming)
- [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [WebSocket Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
