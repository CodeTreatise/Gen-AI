---
title: "Use Cases by Response Mode"
---

# Use Cases by Response Mode

## Introduction

Different applications have different latency requirements. This lesson provides concrete guidance on when to use streaming vs. non-streaming for specific use cases.

### What We'll Cover

- Chatbots and conversational UIs
- Background processing
- Batch jobs
- Real-time applications
- Hybrid architectures

---

## Chatbots: Streaming Essential

Conversational interfaces demand streaming for natural interaction.

### Why Streaming for Chat

```python
chatbot_requirements = {
    "responsiveness": "Users expect immediate feedback",
    "natural_flow": "Conversation feels real-time",
    "engagement": "Users read along as response appears",
    "early_abort": "Can stop if going wrong direction",
    "typing_indicator": "Streaming IS the typing indicator",
}

# Without streaming: "Hello, how are you?" → [5 second pause] → Full response
# With streaming: "Hello, how are you?" → "I'm" → "doing" → "well" → "..."
```

### Chatbot Implementation

```python
from openai import OpenAI

client = OpenAI()

class StreamingChatbot:
    """Chatbot with streaming responses"""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_message: str):
        """Send message and stream response"""
        
        self.messages.append({"role": "user", "content": user_message})
        
        print("Assistant: ", end="", flush=True)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=self.messages,
            stream=True
        )
        
        assistant_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_message += content
                print(content, end="", flush=True)
        
        print()  # Newline after response
        
        self.messages.append({
            "role": "assistant", 
            "content": assistant_message
        })
        
        return assistant_message

# Usage
bot = StreamingChatbot()
bot.chat("What's the weather like today?")
bot.chat("Tell me more about clouds")
```

### Web Chat with SSE

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: dict):
    """Streaming chat API endpoint"""
    
    async def generate():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=request["messages"],
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## Background Processing: Non-Streaming Fine

When no user is watching, streaming adds unnecessary complexity.

### Background Job Examples

```python
import asyncio
from typing import List

class BackgroundProcessor:
    """Process AI tasks without user waiting"""
    
    def __init__(self):
        self.results = {}
    
    async def process_document(self, doc_id: str, content: str) -> dict:
        """Summarize document in background"""
        
        # Non-streaming is simpler for background work
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user", 
                "content": f"Summarize this document:\n\n{content}"
            }],
            stream=False  # No one watching, don't need streaming
        )
        
        result = {
            "doc_id": doc_id,
            "summary": response.choices[0].message.content,
            "status": "complete"
        }
        
        self.results[doc_id] = result
        return result
    
    async def bulk_process(self, documents: List[dict]):
        """Process multiple documents"""
        tasks = [
            self.process_document(doc["id"], doc["content"])
            for doc in documents
        ]
        
        return await asyncio.gather(*tasks)

# Usage (runs in background)
processor = BackgroundProcessor()
asyncio.run(processor.bulk_process([
    {"id": "doc1", "content": "..."},
    {"id": "doc2", "content": "..."},
]))
```

### Email/Notification Generation

```python
def generate_email_notification(user_data: dict) -> str:
    """Generate personalized email - no streaming needed"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"""Generate a personalized email for:
            Name: {user_data['name']}
            Last purchase: {user_data['last_purchase']}
            Preference: {user_data['preference']}
            
            Make it warm and engaging."""
        }],
        stream=False  # Will send via email system later
    )
    
    return response.choices[0].message.content

# Scheduled job - no user waiting
def nightly_email_generation():
    users = get_users_for_email()
    emails = [generate_email_notification(u) for u in users]
    queue_for_sending(emails)
```

---

## Batch Jobs: Non-Streaming Preferred

High-throughput processing works best without streaming overhead.

### Batch Processing Pattern

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def batch_classify(items: list, batch_size: int = 10) -> list:
    """Classify many items efficiently"""
    
    results = []
    
    def classify_one(item):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Fast model for batch
            messages=[{
                "role": "user",
                "content": f"Classify as positive/negative/neutral: {item}"
            }],
            max_tokens=10,
            stream=False  # Simpler for batch
        )
        return {
            "item": item,
            "classification": response.choices[0].message.content.strip()
        }
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(classify_one, item): item for item in items}
        
        for future in tqdm(as_completed(futures), total=len(items)):
            results.append(future.result())
    
    return results

# Process 1000 items
items = ["Great product!", "Terrible service", "It's okay", ...]
classifications = batch_classify(items)
```

### Data Pipeline Integration

```python
def etl_with_ai_enrichment(data_batch: list) -> list:
    """ETL pipeline with AI enrichment step"""
    
    # Non-streaming for pipeline reliability
    enriched = []
    
    for record in data_batch:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Extract entities: {record['text']}"
            }],
            response_format={"type": "json_object"},
            stream=False  # Need complete JSON
        )
        
        record["entities"] = json.loads(response.choices[0].message.content)
        enriched.append(record)
    
    return enriched
```

---

## Real-Time Applications: Streaming Required

Some applications need real-time updates beyond just chatbots.

### Live Transcription Enhancement

```python
async def enhance_live_transcription(transcript_stream):
    """Real-time transcript enhancement with streaming"""
    
    async for transcript_chunk in transcript_stream:
        # Stream correction/enhancement back
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Correct grammar: {transcript_chunk}"
            }],
            stream=True  # Real-time updates
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### Live Dashboard Insights

```python
async def stream_insights_to_dashboard(metrics: dict):
    """Stream AI insights to live dashboard"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Analyze these metrics and provide insights: {metrics}"
        }],
        stream=True
    )
    
    # Push to dashboard websocket as chunks arrive
    for chunk in response:
        if chunk.choices[0].delta.content:
            await dashboard_ws.send(json.dumps({
                "type": "insight_chunk",
                "content": chunk.choices[0].delta.content
            }))
```

### Collaborative Writing

```python
class CollaborativeWriter:
    """Real-time collaborative writing with AI"""
    
    async def suggest_continuation(self, current_text: str, websocket):
        """Stream AI suggestions to collaborators"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Continue this text naturally:\n\n{current_text}"
            }],
            stream=True
        )
        
        # Stream to all connected users
        for chunk in response:
            if chunk.choices[0].delta.content:
                await websocket.broadcast({
                    "type": "ai_suggestion",
                    "text": chunk.choices[0].delta.content
                })
```

---

## Hybrid Architectures

Many real applications combine both modes.

### API with Optional Streaming

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.post("/generate")
async def generate(
    request: dict,
    stream: bool = Query(False, description="Enable streaming")
):
    """Endpoint that supports both modes"""
    
    if stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream"
        )
    else:
        return await generate_complete(request)

async def generate_stream(request: dict):
    response = client.chat.completions.create(
        model=request.get("model", "gpt-4"),
        messages=request["messages"],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
    yield "data: [DONE]\n\n"

async def generate_complete(request: dict):
    response = client.chat.completions.create(
        model=request.get("model", "gpt-4"),
        messages=request["messages"],
        stream=False
    )
    return {"content": response.choices[0].message.content}
```

### Stream to User, Log Complete

```python
async def stream_and_log(messages: list, request_id: str):
    """Stream to user while collecting for logging"""
    
    full_content = ""
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            yield content  # Stream to user
    
    # Log complete interaction
    await log_interaction({
        "request_id": request_id,
        "messages": messages,
        "response": full_content,
        "duration": time.time() - start_time
    })
```

---

## Decision Cheat Sheet

### Quick Reference

| Use Case | Mode | Reason |
|----------|------|--------|
| Chatbot | **Streaming** | UX essential |
| Code IDE | **Streaming** | Watch code appear |
| Email generation | Non-streaming | Sent later |
| Batch classification | Non-streaming | Throughput |
| Data pipeline | Non-streaming | Reliability |
| Live transcription | **Streaming** | Real-time |
| Document processing | Non-streaming | Background |
| Interactive search | **Streaming** | Responsiveness |
| Report generation | Non-streaming | Download later |
| Writing assistant | **Streaming** | Collaboration |

### Decision Tree

```
Is there a user actively watching?
│
├── NO → Non-streaming
│   ├── Batch job
│   ├── Background process
│   └── Scheduled task
│
└── YES → Is response time > 1-2 seconds?
    │
    ├── NO → Either works (non-streaming simpler)
    │   └── Short Q&A, simple classifications
    │
    └── YES → Streaming
        ├── Chatbots
        ├── Code generation
        ├── Long-form content
        └── Real-time features
```

---

## Hands-on Exercise

### Your Task

Build a multi-mode application:

```python
from openai import OpenAI
from enum import Enum
from typing import Iterator, Union

client = OpenAI()

class ResponseMode(Enum):
    STREAMING = "streaming"
    NON_STREAMING = "non_streaming"
    AUTO = "auto"

class MultiModeAI:
    """AI client supporting multiple response modes"""
    
    def __init__(self):
        self.use_case_modes = {
            "chat": ResponseMode.STREAMING,
            "batch": ResponseMode.NON_STREAMING,
            "analysis": ResponseMode.NON_STREAMING,
            "interactive": ResponseMode.STREAMING,
        }
    
    def query(
        self,
        prompt: str,
        mode: ResponseMode = ResponseMode.AUTO,
        use_case: str = None
    ) -> Union[str, Iterator[str]]:
        """Query with specified or automatic mode selection"""
        
        # Determine mode
        if mode == ResponseMode.AUTO:
            mode = self._select_mode(prompt, use_case)
        
        messages = [{"role": "user", "content": prompt}]
        
        if mode == ResponseMode.STREAMING:
            return self._stream_query(messages)
        else:
            return self._simple_query(messages)
    
    def _select_mode(self, prompt: str, use_case: str) -> ResponseMode:
        """Automatically select best mode"""
        
        if use_case and use_case in self.use_case_modes:
            return self.use_case_modes[use_case]
        
        # Heuristics
        if len(prompt) > 500:  # Long input = probably long output
            return ResponseMode.STREAMING
        
        if any(word in prompt.lower() for word in ["explain", "describe", "write"]):
            return ResponseMode.STREAMING
        
        return ResponseMode.NON_STREAMING
    
    def _stream_query(self, messages: list) -> Iterator[str]:
        """Streaming query"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _simple_query(self, messages: list) -> str:
        """Non-streaming query"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content

# Test different modes
ai = MultiModeAI()

# Chat mode (streaming)
print("Chat (streaming):")
for chunk in ai.query("Tell me a story", use_case="chat"):
    print(chunk, end="", flush=True)
print()

# Batch mode (non-streaming)
print("\nBatch (non-streaming):")
result = ai.query("Classify: great product", use_case="batch")
print(result)

# Auto mode
print("\nAuto mode:")
for chunk in ai.query("Explain quantum computing in detail"):
    print(chunk, end="", flush=True)
print()
```

---

## Summary

✅ **Chatbots need streaming** for natural conversation feel

✅ **Background jobs use non-streaming** for simplicity

✅ **Batch processing prefers non-streaming** for throughput

✅ **Real-time apps require streaming** for live updates

✅ **Hybrid architectures** combine both modes appropriately

✅ **Match mode to use case** for optimal UX and efficiency

**Next Lesson:** [Types of AI Models](../07-types-of-ai-models/00-types-of-ai-models.md)

---

## Further Reading

- [OpenAI Streaming Guide](https://platform.openai.com/docs/api-reference/streaming) — Implementation details
- [Real-Time AI Applications](https://platform.openai.com/docs/guides/realtime) — OpenAI real-time features

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Full Response Latency](./06-full-response-latency.md) | [Streaming Response Modes](./00-streaming-response-modes.md) | [Types of AI Models](../07-types-of-ai-models/00-types-of-ai-models.md) |

<!-- 
Sources Consulted:
- OpenAI Streaming: https://platform.openai.com/docs/api-reference/streaming
- Anthropic Streaming: https://docs.anthropic.com/claude/reference/streaming
-->

