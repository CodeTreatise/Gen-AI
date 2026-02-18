---
title: "Real-Time Agent Interfaces"
---

# Real-Time Agent Interfaces

## Introduction

Some agent interactions demand more than request-response HTTP patterns. **Real-time interfaces** keep a persistent connection open between the agent and the user, enabling bidirectional communication, live collaboration, voice interaction, and push-based updates. These patterns are essential for agents that maintain ongoing sessions, respond to external events, or need sub-second latency.

This lesson covers WebSocket-based agent sessions, voice-powered agents using the Web Speech API, collaborative multi-user agent interactions, and push notification patterns for long-running agent tasks.

### What we'll cover

- WebSocket agent sessions with persistent connections
- Voice-powered agents with speech recognition and synthesis
- Collaborative agents for multi-user environments
- Push-based updates for long-running agent tasks
- Reconnection and resilience patterns

### Prerequisites

- WebSocket fundamentals (Unit 1, Lesson 7)
- Server-side agent hosting (Lesson 17-01)
- Streaming agents to the browser (Lesson 17-02)
- JavaScript async/await (Unit 1, Lesson 5)

---

## WebSocket agent sessions

While SSE works well for one-way streaming, WebSockets provide **full-duplex communication** — both the client and server can send messages at any time. This is ideal for:

- Agents that push updates without being asked
- Multi-turn conversations with low latency
- Agents that need to interrupt the user
- Real-time collaboration between multiple users and an agent

### Server: WebSocket agent with FastAPI

```python
# server.py
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from anthropic import AsyncAnthropic

app = FastAPI()
client = AsyncAnthropic()

class AgentSession:
    """Manages a persistent agent conversation over WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.messages: list[dict] = []
        self.is_processing = False

    async def send_event(self, event_type: str, data: dict):
        await self.ws.send_json({"type": event_type, **data})

    async def process_message(self, user_input: str):
        self.is_processing = True
        self.messages.append({"role": "user", "content": user_input})

        await self.send_event("status", {"status": "thinking"})

        try:
            async with client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=self.messages,
            ) as stream:
                full_response = ""
                async for text in stream.text_stream:
                    full_response += text
                    await self.send_event("token", {"content": text})

            self.messages.append({
                "role": "assistant",
                "content": full_response,
            })
            await self.send_event("done", {"content": full_response})

        except Exception as e:
            await self.send_event("error", {"message": str(e)})

        finally:
            self.is_processing = False

    async def handle_interrupt(self):
        """Handle user interruption during agent processing."""
        if self.is_processing:
            self.is_processing = False
            await self.send_event("interrupted", {})

@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    session = AgentSession(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "message":
                await session.process_message(data["content"])
            elif data["type"] == "interrupt":
                await session.handle_interrupt()
            elif data["type"] == "clear":
                session.messages = []
                await session.send_event("cleared", {})

    except WebSocketDisconnect:
        pass  # Client disconnected
```

### Client: WebSocket agent class

```typescript
type AgentEventType =
  | 'status' | 'token' | 'done' | 'error'
  | 'interrupted' | 'cleared';

interface AgentEvent {
  type: AgentEventType;
  content?: string;
  status?: string;
  message?: string;
}

class WebSocketAgent {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnects = 5;
  private listeners = new Map<string, Set<(event: AgentEvent) => void>>();

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onmessage = (event) => {
        const data: AgentEvent = JSON.parse(event.data);
        this.emit(data.type, data);
      };

      this.ws.onclose = () => {
        this.handleDisconnect();
      };

      this.ws.onerror = () => {
        reject(new Error('WebSocket connection failed'));
      };
    });
  }

  sendMessage(content: string) {
    this.ws?.send(JSON.stringify({ type: 'message', content }));
  }

  interrupt() {
    this.ws?.send(JSON.stringify({ type: 'interrupt' }));
  }

  clearHistory() {
    this.ws?.send(JSON.stringify({ type: 'clear' }));
  }

  on(event: string, callback: (data: AgentEvent) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  private emit(event: string, data: AgentEvent) {
    this.listeners.get(event)?.forEach(cb => cb(data));
    this.listeners.get('*')?.forEach(cb => cb(data)); // Wildcard
  }

  private async handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnects) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 30000);
      console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
      await new Promise(r => setTimeout(r, delay));
      try {
        await this.connect();
      } catch {
        this.handleDisconnect();
      }
    }
  }

  disconnect() {
    this.maxReconnects = 0; // Prevent reconnection
    this.ws?.close();
  }
}
```

**Usage:**
```typescript
const agent = new WebSocketAgent('ws://localhost:8000/ws/agent');
await agent.connect();

agent.on('token', (event) => {
  process.stdout.write(event.content || '');
});

agent.on('done', (event) => {
  console.log('\n--- Agent done ---');
});

agent.sendMessage('What are the latest AI trends?');
```

---

## Voice-powered agents

The Web Speech API lets you build voice-in, voice-out agent interfaces — no external services needed for basic voice interaction.

### Speech recognition (voice input)

```typescript
class VoiceInput {
  private recognition: SpeechRecognition;
  private isListening = false;

  constructor(private onResult: (text: string) => void) {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      const last = event.results[event.results.length - 1];
      if (last.isFinal) {
        this.onResult(last[0].transcript);
      }
    };

    this.recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };
  }

  start() {
    if (!this.isListening) {
      this.isListening = true;
      this.recognition.start();
    }
  }

  stop() {
    this.isListening = false;
    this.recognition.stop();
  }
}
```

### Speech synthesis (voice output)

```typescript
class VoiceOutput {
  private synth = window.speechSynthesis;
  private currentUtterance: SpeechSynthesisUtterance | null = null;

  speak(text: string, options: { rate?: number; pitch?: number } = {}) {
    // Cancel any ongoing speech
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = options.rate || 1.0;
    utterance.pitch = options.pitch || 1.0;

    // Pick a natural-sounding voice
    const voices = this.synth.getVoices();
    const preferred = voices.find(v =>
      v.name.includes('Google') || v.name.includes('Samantha')
    );
    if (preferred) utterance.voice = preferred;

    this.currentUtterance = utterance;
    this.synth.speak(utterance);

    return new Promise<void>((resolve) => {
      utterance.onend = () => resolve();
    });
  }

  stop() {
    this.synth.cancel();
  }

  get isSpeaking() {
    return this.synth.speaking;
  }
}
```

### Combining voice with a WebSocket agent

```typescript
class VoiceAgent {
  private wsAgent: WebSocketAgent;
  private voiceInput: VoiceInput;
  private voiceOutput: VoiceOutput;
  private responseBuffer = '';

  constructor(wsUrl: string) {
    this.voiceOutput = new VoiceOutput();

    this.wsAgent = new WebSocketAgent(wsUrl);

    this.voiceInput = new VoiceInput((text) => {
      console.log(`🎤 You said: "${text}"`);
      this.wsAgent.sendMessage(text);
    });

    // Collect tokens and speak when done
    this.wsAgent.on('token', (event) => {
      this.responseBuffer += event.content || '';
    });

    this.wsAgent.on('done', async () => {
      console.log(`🔊 Agent: "${this.responseBuffer}"`);
      await this.voiceOutput.speak(this.responseBuffer);
      this.responseBuffer = '';
      // Resume listening after agent speaks
      this.voiceInput.start();
    });
  }

  async start() {
    await this.wsAgent.connect();
    this.voiceInput.start();
    console.log('🎤 Listening... Speak to the agent.');
  }

  stop() {
    this.voiceInput.stop();
    this.voiceOutput.stop();
    this.wsAgent.disconnect();
  }
}
```

**Usage:**
```typescript
const voiceAgent = new VoiceAgent('ws://localhost:8000/ws/agent');
await voiceAgent.start();
// Now speak — the agent hears, responds, and speaks back
```

> **Warning:** The Web Speech API has limited browser support. Chrome has the best support for `SpeechRecognition`. For production voice agents, consider using cloud speech services (Google Cloud Speech, Azure Speech, OpenAI Whisper/TTS) for better accuracy and cross-browser compatibility.

---

## Collaborative agent sessions

In collaborative environments, multiple users interact with the same agent. Each user sees the agent's responses and other users' messages in real-time.

```python
# server.py — Collaborative WebSocket agent
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI()

class CollaborativeRoom:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.clients: dict[str, WebSocket] = {}
        self.messages: list[dict] = []

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.clients[user_id] = websocket
        await self.broadcast({
            "type": "user_joined",
            "user_id": user_id,
            "active_users": list(self.clients.keys()),
        })

    async def disconnect(self, user_id: str):
        self.clients.pop(user_id, None)
        await self.broadcast({
            "type": "user_left",
            "user_id": user_id,
            "active_users": list(self.clients.keys()),
        })

    async def broadcast(self, message: dict, exclude: str = None):
        for uid, ws in self.clients.items():
            if uid != exclude:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass  # Client disconnected

    async def handle_message(self, user_id: str, content: str):
        # Broadcast user message to all participants
        user_msg = {
            "type": "user_message",
            "user_id": user_id,
            "content": content,
        }
        await self.broadcast(user_msg)

        # Get agent response (simplified — use your agent here)
        self.messages.append({"role": "user", "content": f"[{user_id}]: {content}"})

        await self.broadcast({"type": "agent_thinking"})

        # ... Agent streaming would go here ...
        agent_response = f"Response to {user_id}'s question about: {content[:50]}..."
        self.messages.append({"role": "assistant", "content": agent_response})

        await self.broadcast({
            "type": "agent_response",
            "content": agent_response,
        })

rooms: dict[str, CollaborativeRoom] = {}

@app.websocket("/ws/room/{room_id}/{user_id}")
async def room_ws(websocket: WebSocket, room_id: str, user_id: str):
    if room_id not in rooms:
        rooms[room_id] = CollaborativeRoom(room_id)

    room = rooms[room_id]
    await room.connect(user_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "message":
                await room.handle_message(user_id, data["content"])
    except WebSocketDisconnect:
        await room.disconnect(user_id)
```

---

## Push-based updates for long-running tasks

Some agent tasks (research, code generation, data analysis) take minutes, not seconds. Push-based patterns let users navigate away and get notified when the task completes.

```typescript
// Pattern: Submit task, get notified when done

class LongRunningAgentTask {
  private taskId: string | null = null;
  private eventSource: EventSource | null = null;

  async submit(prompt: string): Promise<string> {
    // Submit the task — server returns immediately with a task ID
    const res = await fetch('/api/agent/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const { taskId } = await res.json();
    this.taskId = taskId;
    return taskId;
  }

  subscribe(callbacks: {
    onProgress?: (progress: number, message: string) => void;
    onComplete?: (result: string) => void;
    onError?: (error: string) => void;
  }) {
    if (!this.taskId) throw new Error('No task submitted');

    // Subscribe to task updates via SSE
    this.eventSource = new EventSource(
      `/api/agent/tasks/${this.taskId}/stream`
    );

    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'progress':
          callbacks.onProgress?.(data.percent, data.message);
          break;
        case 'complete':
          callbacks.onComplete?.(data.result);
          this.eventSource?.close();
          break;
        case 'error':
          callbacks.onError?.(data.message);
          this.eventSource?.close();
          break;
      }
    };
  }

  cancel() {
    this.eventSource?.close();
    if (this.taskId) {
      fetch(`/api/agent/tasks/${this.taskId}`, { method: 'DELETE' });
    }
  }
}
```

**Usage:**
```typescript
const task = new LongRunningAgentTask();
const taskId = await task.submit('Research the history of quantum computing');

// Show notification when done — user can navigate away
task.subscribe({
  onProgress: (percent, msg) => {
    console.log(`${percent}%: ${msg}`);
  },
  onComplete: (result) => {
    new Notification('Agent Task Complete', {
      body: result.slice(0, 100) + '...',
    });
  },
  onError: (error) => {
    console.error('Task failed:', error);
  },
});
```

**Output:**
```
10%: Searching for relevant sources...
35%: Analyzing 12 documents...
70%: Synthesizing findings...
100%: Research complete
```

---

## Reconnection and resilience

Real-time connections drop. Here's a resilience pattern for maintaining agent sessions across reconnections:

```typescript
class ResilientAgentConnection {
  private sessionId: string;
  private lastEventId = 0;

  constructor(private baseUrl: string) {
    this.sessionId = crypto.randomUUID();
  }

  async connect() {
    // Resume from last known event
    const url = new URL(`${this.baseUrl}/ws/agent`);
    url.searchParams.set('session', this.sessionId);
    url.searchParams.set('after', String(this.lastEventId));

    const ws = new WebSocket(url);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.eventId) {
        this.lastEventId = data.eventId;
      }
      // Process the event...
    };

    ws.onclose = () => {
      // Exponential backoff reconnection
      setTimeout(() => this.connect(), this.getBackoffDelay());
    };
  }

  private reconnectCount = 0;
  private getBackoffDelay(): number {
    const delay = Math.min(1000 * 2 ** this.reconnectCount, 30000);
    this.reconnectCount++;
    return delay + Math.random() * 1000; // Add jitter
  }
}
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use WebSockets for bidirectional, SSE for unidirectional | Don't add WebSocket complexity when SSE is sufficient |
| Implement exponential backoff with jitter | Prevents thundering herd on reconnection |
| Track session IDs across reconnections | Resume conversations without losing context |
| Use push notifications for long tasks | Users shouldn't have to keep the tab open |
| Fall back to cloud speech for production voice | Web Speech API is Chrome-only for recognition |
| Limit concurrent WebSocket connections | Each connection holds server resources open |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using WebSockets for everything | SSE is simpler for one-way streaming; use it when possible |
| Not handling reconnection | Connections drop; always implement auto-reconnect with backoff |
| Blocking voice recognition during agent speech | Stop `SpeechRecognition` while `SpeechSynthesis` is active |
| Sending full message history on every WebSocket message | Keep state server-side; send only new messages |
| No rate limiting on WebSocket endpoints | Add per-connection message rate limits to prevent abuse |
| Forgetting to clean up WebSocket connections | Close connections in component cleanup / `useEffect` return |

---

## Hands-on exercise

### Your task

Build a real-time agent interface with a WebSocket connection that supports:
1. Persistent conversation sessions
2. Message interruption (cancel current generation)
3. Automatic reconnection on disconnect

### Requirements

1. Create a WebSocket server that maintains agent conversation state
2. Build a client class that handles connection, messaging, and reconnection
3. Add an interrupt button that cancels the agent's current response
4. Implement exponential backoff reconnection with a maximum of 5 attempts

### Expected result

A chat interface that maintains connection state, allows users to interrupt agent responses mid-stream, and automatically reconnects if the connection drops.

<details>
<summary>💡 Hints (click to expand)</summary>

- Store conversation history in an `AgentSession` class on the server
- Use a `processing` flag to track whether the agent is generating
- Send an `interrupt` event type from the client to stop generation
- Calculate backoff delay as `min(1000 * 2^attempt, 30000)` + random jitter

</details>

### Bonus challenges

- [ ] Add voice input using `SpeechRecognition` that sends transcribed text via WebSocket
- [ ] Implement a collaborative room where multiple users see the same agent responses
- [ ] Add browser push notifications for long-running agent tasks

---

## Summary

✅ **WebSockets** enable full-duplex agent sessions with interruption, push updates, and low latency  
✅ The **Web Speech API** provides voice input/output for browser-based voice agents  
✅ **Collaborative sessions** let multiple users interact with a shared agent in real-time  
✅ **Push-based patterns** keep users informed about long-running tasks without holding connections open  
✅ Always implement **reconnection with exponential backoff** for resilient real-time connections

**Next:** [Agent UI/UX Patterns](./08-agent-ui-ux-patterns.md)

---

## Further Reading

- [MDN WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket) - WebSocket reference
- [MDN Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) - Speech recognition & synthesis
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/) - Server-side WebSocket handling
- [Reconnecting WebSocket](https://github.com/pladaria/reconnecting-websocket) - Auto-reconnection library

<!--
Sources Consulted:
- MDN WebSocket API: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
- MDN Web Speech API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
-->
