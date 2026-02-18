---
title: "Streaming Agents to Browser"
---

# Streaming Agents to Browser

## Introduction

When an agent thinks for 10 seconds before responding, users stare at a blank screen wondering if the app froze. Streaming solves this by delivering the agent's output **incrementally** — tokens appear as they're generated, tool calls are shown as they execute, and progress updates keep users informed throughout the entire reasoning process.

This lesson focuses on the **browser-side consumption** of agent streams. We'll cover how to parse SSE streams, render tokens progressively, display tool call chains, and handle the various event types that agent backends emit.

### What we'll cover

- Consuming SSE streams in the browser with `fetch` and `EventSource`
- Token-by-token rendering for smooth text display
- Streaming thought processes and reasoning chains
- Displaying tool execution progress
- Partial result rendering patterns
- Error handling in streams

### Prerequisites

- Server-side agent hosting (Lesson 17-01)
- JavaScript async/await and Promises (Unit 1, Lesson 5)
- DOM manipulation basics (Unit 1, Lesson 4)
- Fetch API (Unit 1, Lesson 6)

---

## Consuming SSE streams with fetch

The `EventSource` API is the traditional way to consume SSE, but it only supports `GET` requests. Since agent endpoints typically use `POST` (to send conversation history), we use `fetch` with a streaming reader instead.

### The fetch + ReadableStream pattern

```javascript
async function streamAgent(message) {
  const response = await fetch('/api/agent/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Decode the chunk and add to buffer
    buffer += decoder.decode(value, { stream: true });

    // Parse complete SSE events from buffer
    const events = buffer.split('\n\n');
    buffer = events.pop(); // Keep incomplete event in buffer

    for (const event of events) {
      if (event.startsWith('data: ')) {
        const jsonStr = event.slice(6); // Remove 'data: ' prefix
        try {
          const data = JSON.parse(jsonStr);
          handleAgentEvent(data);
        } catch (e) {
          console.warn('Failed to parse SSE event:', jsonStr);
        }
      }
    }
  }
}

function handleAgentEvent(event) {
  switch (event.type) {
    case 'thinking':
      showThinkingIndicator(event.content);
      break;
    case 'tool_call':
      showToolExecution(event.tool, event.input);
      break;
    case 'tool_result':
      showToolResult(event.tool, event.output);
      break;
    case 'token':
      appendToken(event.content);
      break;
    case 'done':
      finishResponse();
      break;
    case 'error':
      showError(event.content);
      break;
  }
}
```

**Output (in browser console):**
```
[thinking] Analyzing your question...
[tool_call] Running search_web({query: "AI trends 2025"})
[tool_result] search_web → Found 5 results...
[token] Based
[token] on
[token] my
[token] research...
[done] Response complete
```

> **Note:** The buffer pattern is critical. Network chunks don't always align with SSE event boundaries. The `split('\n\n')` + `pop()` technique ensures we only process complete events and preserve partial data for the next chunk.

### EventSource for GET endpoints

If your agent endpoint supports `GET` requests (less common, but useful for simple queries):

```javascript
function connectAgent(query) {
  const source = new EventSource(`/api/agent/stream?q=${encodeURIComponent(query)}`);
  
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleAgentEvent(data);
    
    // Close when done
    if (data.type === 'done') {
      source.close();
    }
  };
  
  source.onerror = (error) => {
    console.error('SSE connection error:', error);
    source.close();
  };
  
  return source; // Return for external cancellation
}
```

> **Tip:** `EventSource` automatically reconnects on connection loss — useful for long-running agents, but be aware it will restart the request from scratch.

---

## Token-by-token rendering

The core of streaming UIs is rendering text tokens as they arrive. We need to handle token batching, Markdown rendering, and smooth visual updates.

### Basic token appender

```javascript
class TokenRenderer {
  constructor(containerElement) {
    this.container = containerElement;
    this.fullText = '';
  }

  appendToken(token) {
    this.fullText += token;
    this.container.textContent = this.fullText;
    
    // Auto-scroll to bottom
    this.container.scrollTop = this.container.scrollHeight;
  }

  clear() {
    this.fullText = '';
    this.container.textContent = '';
  }
}

// Usage
const output = document.getElementById('agent-output');
const renderer = new TokenRenderer(output);

// Each token event from SSE:
renderer.appendToken('Hello');
renderer.appendToken(' world');
renderer.appendToken('!');
```

**Output (progressive):**
```
Frame 1: "Hello"
Frame 2: "Hello world"
Frame 3: "Hello world!"
```

### Markdown-aware streaming

Agents often output Markdown. We need to render it progressively without breaking incomplete syntax:

```javascript
import { marked } from 'marked';

class MarkdownStreamRenderer {
  constructor(containerElement) {
    this.container = containerElement;
    this.rawText = '';
    this.renderScheduled = false;
  }

  appendToken(token) {
    this.rawText += token;
    
    // Debounce rendering to avoid excessive DOM updates
    if (!this.renderScheduled) {
      this.renderScheduled = true;
      requestAnimationFrame(() => {
        this.render();
        this.renderScheduled = false;
      });
    }
  }

  render() {
    // Render complete Markdown, handling incomplete blocks gracefully
    const safeText = this.closeIncompleteBlocks(this.rawText);
    this.container.innerHTML = marked.parse(safeText);
    this.container.scrollTop = this.container.scrollHeight;
  }

  closeIncompleteBlocks(text) {
    // Close incomplete code blocks to prevent rendering errors
    const codeBlockCount = (text.match(/```/g) || []).length;
    if (codeBlockCount % 2 !== 0) {
      text += '\n```';
    }
    return text;
  }

  getFullText() {
    return this.rawText;
  }

  clear() {
    this.rawText = '';
    this.container.innerHTML = '';
  }
}
```

**Output (during streaming):**
```html
<!-- Token 1-3: "Here is " + "some " + "**bold**" -->
<p>Here is some <strong>bold</strong></p>

<!-- Token 4-6: " text" + "\n```python\n" + "print('hello')" -->
<p>Here is some <strong>bold</strong> text</p>
<pre><code class="language-python">print('hello')
</code></pre>
```

> **Important:** Use `requestAnimationFrame` to batch DOM updates. Without it, updating the DOM on every single token (which can arrive milliseconds apart) causes visible jank and wastes CPU cycles.

---

## Streaming thought and action chains

Agents don't just generate text — they reason, plan, call tools, and iterate. A rich streaming UI shows each of these phases:

### Agent event stream renderer

```javascript
class AgentStreamUI {
  constructor(container) {
    this.container = container;
    this.currentSection = null;
    this.tokenRenderer = null;
  }

  handleEvent(event) {
    switch (event.type) {
      case 'thinking':
        this.showThinking(event.content);
        break;
      case 'tool_call':
        this.showToolCall(event.tool, event.input, event.status);
        break;
      case 'tool_result':
        this.showToolResult(event.tool, event.output);
        break;
      case 'token':
        this.appendResponseToken(event.content);
        break;
      case 'step_start':
        this.showStepBoundary(event.step);
        break;
      case 'done':
        this.finalize();
        break;
      case 'error':
        this.showError(event.content);
        break;
    }
  }

  showThinking(content) {
    const el = document.createElement('div');
    el.className = 'agent-thinking';
    el.innerHTML = `
      <span class="thinking-icon">🤔</span>
      <span class="thinking-text">${content}</span>
      <span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
    `;
    this.container.appendChild(el);
    this.currentSection = el;
  }

  showToolCall(toolName, input, status) {
    // Remove thinking indicator
    this.removeThinking();

    const el = document.createElement('div');
    el.className = 'agent-tool-call';
    el.dataset.tool = toolName;
    el.innerHTML = `
      <div class="tool-header">
        <span class="tool-icon">🔧</span>
        <span class="tool-name">${toolName}</span>
        <span class="tool-status ${status}">${status === 'running' ? '⏳' : '✅'}</span>
      </div>
      <div class="tool-input">
        <pre>${JSON.stringify(input, null, 2)}</pre>
      </div>
    `;
    this.container.appendChild(el);
    this.currentSection = el;
  }

  showToolResult(toolName, output) {
    // Update the existing tool call element
    const toolEl = this.container.querySelector(
      `.agent-tool-call[data-tool="${toolName}"]`
    );
    if (toolEl) {
      const statusEl = toolEl.querySelector('.tool-status');
      statusEl.textContent = '✅';
      statusEl.className = 'tool-status complete';

      const resultEl = document.createElement('div');
      resultEl.className = 'tool-result';
      resultEl.innerHTML = `<pre>${JSON.stringify(output, null, 2)}</pre>`;
      toolEl.appendChild(resultEl);
    }
  }

  appendResponseToken(token) {
    if (!this.tokenRenderer) {
      const el = document.createElement('div');
      el.className = 'agent-response';
      this.container.appendChild(el);
      this.tokenRenderer = new MarkdownStreamRenderer(el);
    }
    this.tokenRenderer.appendToken(token);
  }

  showStepBoundary(stepNumber) {
    const el = document.createElement('hr');
    el.className = 'step-boundary';
    this.container.appendChild(el);
    this.tokenRenderer = null; // Reset for next response section
  }

  showError(content) {
    const el = document.createElement('div');
    el.className = 'agent-error';
    el.innerHTML = `<span class="error-icon">❌</span> ${content}`;
    this.container.appendChild(el);
  }

  removeThinking() {
    const thinkingEl = this.container.querySelector('.agent-thinking');
    if (thinkingEl) thinkingEl.remove();
  }

  finalize() {
    this.removeThinking();
    this.currentSection = null;
  }
}
```

### CSS for the streaming UI

```css
.agent-thinking {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  color: #6b7280;
  font-style: italic;
}

.thinking-dots span {
  animation: blink 1.4s infinite both;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
  0%, 80%, 100% { opacity: 0; }
  40% { opacity: 1; }
}

.agent-tool-call {
  margin: 8px 0;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
}

.tool-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f9fafb;
  font-weight: 500;
}

.tool-status.running { color: #f59e0b; }
.tool-status.complete { color: #10b981; }

.tool-input, .tool-result {
  padding: 8px 12px;
  font-size: 0.875rem;
}

.tool-input pre, .tool-result pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}

.tool-result {
  background: #f0fdf4;
  border-top: 1px solid #e5e7eb;
}

.agent-response {
  padding: 12px 0;
  line-height: 1.6;
}

.agent-error {
  padding: 12px 16px;
  color: #dc2626;
  background: #fef2f2;
  border-radius: 8px;
}

.step-boundary {
  margin: 16px 0;
  border: none;
  border-top: 1px dashed #d1d5db;
}
```

---

## Progress updates for long-running agents

Agents that perform complex tasks (research, code generation, data analysis) may run for 30 seconds or more. Progress updates keep users engaged:

### Progress event pattern

```javascript
class AgentProgressTracker {
  constructor(container) {
    this.container = container;
    this.progressBar = null;
    this.stepsList = null;
    this.steps = [];
  }

  handleEvent(event) {
    switch (event.type) {
      case 'progress':
        this.updateProgress(event.percent, event.message);
        break;
      case 'step_start':
        this.addStep(event.step, event.description);
        break;
      case 'step_complete':
        this.completeStep(event.step);
        break;
    }
  }

  updateProgress(percent, message) {
    if (!this.progressBar) {
      this.createProgressUI();
    }

    const fill = this.progressBar.querySelector('.progress-fill');
    const label = this.progressBar.querySelector('.progress-label');
    fill.style.width = `${percent}%`;
    label.textContent = `${message} (${percent}%)`;
  }

  addStep(stepId, description) {
    if (!this.stepsList) {
      this.stepsList = document.createElement('ul');
      this.stepsList.className = 'agent-steps';
      this.container.appendChild(this.stepsList);
    }

    const li = document.createElement('li');
    li.dataset.step = stepId;
    li.className = 'step-item running';
    li.innerHTML = `<span class="step-indicator">⏳</span> ${description}`;
    this.stepsList.appendChild(li);
    this.steps.push(stepId);
  }

  completeStep(stepId) {
    const stepEl = this.stepsList?.querySelector(`[data-step="${stepId}"]`);
    if (stepEl) {
      stepEl.className = 'step-item complete';
      stepEl.querySelector('.step-indicator').textContent = '✅';
    }
  }

  createProgressUI() {
    this.progressBar = document.createElement('div');
    this.progressBar.className = 'agent-progress';
    this.progressBar.innerHTML = `
      <div class="progress-track">
        <div class="progress-fill" style="width: 0%"></div>
      </div>
      <div class="progress-label">Starting...</div>
    `;
    this.container.appendChild(this.progressBar);
  }
}
```

**Output (visual progression):**
```
⏳ Searching knowledge base...        [████░░░░░░] 35%
✅ Searching knowledge base
⏳ Analyzing results...               [██████░░░░] 60%
✅ Analyzing results
⏳ Generating response...             [████████░░] 85%
```

---

## Stream cancellation

Users need the ability to stop an agent mid-execution. With `fetch`, we use `AbortController`:

```javascript
class CancellableAgentStream {
  constructor() {
    this.controller = null;
  }

  async start(message, onEvent) {
    // Create a new abort controller for this request
    this.controller = new AbortController();

    try {
      const response = await fetch('/api/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: this.controller.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const event of events) {
          if (event.startsWith('data: ')) {
            const data = JSON.parse(event.slice(6));
            onEvent(data);
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        onEvent({ type: 'cancelled', content: 'Stream cancelled by user' });
      } else {
        onEvent({ type: 'error', content: error.message });
      }
    }
  }

  cancel() {
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
  }
}

// Usage
const stream = new CancellableAgentStream();

// Start button
document.getElementById('send').onclick = () => {
  stream.start('Explain quantum computing', (event) => {
    console.log(event.type, event.content);
  });
};

// Stop button
document.getElementById('stop').onclick = () => {
  stream.cancel();
};
```

**Output (when cancelled):**
```
thinking: Analyzing your question...
token: Quantum
token: computing
token: is
cancelled: Stream cancelled by user
```

---

## React streaming component

Here's a complete React component that handles agent streaming with all the patterns above:

```jsx
import { useState, useRef, useCallback } from 'react';

function useAgentStream() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const controllerRef = useRef(null);

  const sendMessage = useCallback(async (userMessage) => {
    controllerRef.current = new AbortController();
    setIsStreaming(true);

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    // Create assistant message placeholder
    const assistantMsg = { role: 'assistant', content: '', tools: [], thinking: '' };
    setMessages(prev => [...prev, assistantMsg]);

    try {
      const response = await fetch('/api/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
        signal: controllerRef.current.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const rawEvent of events) {
          if (!rawEvent.startsWith('data: ')) continue;
          const event = JSON.parse(rawEvent.slice(6));

          setMessages(prev => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };

            switch (event.type) {
              case 'thinking':
                last.thinking = event.content;
                break;
              case 'tool_call':
                last.tools = [...last.tools, {
                  name: event.tool,
                  input: event.input,
                  status: 'running',
                }];
                break;
              case 'tool_result':
                last.tools = last.tools.map(t =>
                  t.name === event.tool
                    ? { ...t, output: event.output, status: 'complete' }
                    : t
                );
                break;
              case 'token':
                last.content += event.content;
                last.thinking = ''; // Clear thinking when tokens start
                break;
            }

            updated[updated.length - 1] = last;
            return updated;
          });
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Stream error:', error);
      }
    } finally {
      setIsStreaming(false);
    }
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  return { messages, sendMessage, cancel, isStreaming };
}

// Component usage
function AgentChat() {
  const { messages, sendMessage, cancel, isStreaming } = useAgentStream();
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isStreaming) {
      sendMessage(input);
      setInput('');
    }
  };

  return (
    <div className="agent-chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.thinking && (
              <div className="thinking">🤔 {msg.thinking}...</div>
            )}
            {msg.tools?.map((tool, j) => (
              <div key={j} className={`tool-call ${tool.status}`}>
                <span>🔧 {tool.name}</span>
                {tool.status === 'running' && <span>⏳</span>}
                {tool.output && <pre>{JSON.stringify(tool.output, null, 2)}</pre>}
              </div>
            ))}
            {msg.content && <div className="content">{msg.content}</div>}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the agent..."
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button type="button" onClick={cancel}>Stop</button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </div>
  );
}
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use `requestAnimationFrame` for DOM updates | Batches rendering, prevents jank during fast token streams |
| Buffer SSE events at `\n\n` boundaries | Network chunks don't align with event boundaries |
| Close incomplete Markdown blocks during streaming | Prevents broken HTML rendering mid-stream |
| Provide cancel/stop functionality | Users must be able to halt long-running agents |
| Show structured events (tools, thinking) | Users need transparency into what the agent is doing |
| Debounce Markdown re-rendering | Full re-parse on every token is expensive |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Updating DOM on every single token | Batch updates with `requestAnimationFrame` |
| Not handling incomplete SSE events | Use the buffer + `split('\n\n')` + `pop()` pattern |
| Forgetting to close `AbortController` | Always abort on unmount or cancel |
| Re-rendering entire chat on each token | Only update the last message (use React state carefully) |
| Not showing thinking/tool states | Users think the app is frozen — always show agent activity |
| Ignoring stream errors | Catch `AbortError` separately from real errors |

---

## Hands-on exercise

### Your task

Build a vanilla JavaScript page that connects to an agent SSE endpoint, renders tokens with Markdown support, displays tool calls inline, and supports stream cancellation.

### Requirements

1. Use `fetch` with `ReadableStream` to consume the SSE stream
2. Render tokens progressively using `requestAnimationFrame` batching
3. Show a thinking indicator that disappears when tokens start
4. Display tool calls with their inputs and results
5. Add a "Stop" button that cancels the stream via `AbortController`

### Expected result

A chat interface that shows: thinking indicator → tool execution cards → streaming text response, with a working cancel button.

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with the `CancellableAgentStream` class from this lesson
- Combine it with the `AgentStreamUI` renderer
- Use `handleAgentEvent` as the bridge between stream and UI
- For Markdown, you can use `marked` library or just render plain text

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html>
<head>
  <title>Agent Stream Demo</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; }
    #output { min-height: 200px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .thinking { color: #6b7280; font-style: italic; }
    .tool-call { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; margin: 8px 0; }
    .tool-call.complete { border-color: #10b981; }
    .error { color: #dc2626; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; }
    button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; background: #3b82f6; color: white; }
    button.stop { background: #ef4444; }
  </style>
</head>
<body>
  <h1>Agent Stream</h1>
  <div id="output"></div>
  <form id="form">
    <input id="input" placeholder="Ask the agent..." />
    <button type="submit" id="sendBtn">Send</button>
    <button type="button" id="stopBtn" class="stop" style="display:none">Stop</button>
  </form>

  <script>
    const output = document.getElementById('output');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const stopBtn = document.getElementById('stopBtn');
    let controller = null;
    let responseText = '';

    form.onsubmit = async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;

      output.innerHTML = '';
      responseText = '';
      input.value = '';
      sendBtn.style.display = 'none';
      stopBtn.style.display = 'inline-block';
      controller = new AbortController();

      try {
        const res = await fetch('/api/agent/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message }),
          signal: controller.signal,
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let responseEl = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop();

          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            const event = JSON.parse(part.slice(6));

            if (event.type === 'thinking') {
              const el = document.createElement('div');
              el.className = 'thinking';
              el.textContent = `🤔 ${event.content}...`;
              output.appendChild(el);
            } else if (event.type === 'tool_call') {
              document.querySelector('.thinking')?.remove();
              const el = document.createElement('div');
              el.className = 'tool-call';
              el.dataset.tool = event.tool;
              el.textContent = `🔧 ${event.tool}: ${JSON.stringify(event.input)}`;
              output.appendChild(el);
            } else if (event.type === 'tool_result') {
              const el = document.querySelector(`.tool-call[data-tool="${event.tool}"]`);
              if (el) { el.classList.add('complete'); el.textContent += ` → ${JSON.stringify(event.output)}`; }
            } else if (event.type === 'token') {
              document.querySelector('.thinking')?.remove();
              if (!responseEl) { responseEl = document.createElement('div'); output.appendChild(responseEl); }
              responseText += event.content;
              requestAnimationFrame(() => { responseEl.textContent = responseText; });
            }
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          const el = document.createElement('div');
          el.className = 'error';
          el.textContent = '⛔ Cancelled';
          output.appendChild(el);
        }
      }
      sendBtn.style.display = 'inline-block';
      stopBtn.style.display = 'none';
    };

    stopBtn.onclick = () => controller?.abort();
  </script>
</body>
</html>
```

</details>

### Bonus challenges

- [ ] Add Markdown rendering using the `marked` library
- [ ] Implement a "typing cursor" animation at the end of streaming text
- [ ] Add a progress bar for multi-step agent operations

---

## Summary

✅ Use `fetch` + `ReadableStream` to consume SSE from `POST` endpoints — the standard pattern for agent streaming  
✅ Buffer SSE data at `\n\n` boundaries — network chunks don't align with events  
✅ Batch DOM updates with `requestAnimationFrame` — prevents jank during fast token delivery  
✅ Show structured agent activity — thinking indicators, tool calls, and progress updates keep users engaged  
✅ Always provide stream cancellation — `AbortController` gives users control over long-running agents

**Next:** [LangChain.js Essentials](./03-langchainjs-essentials.md)

---

## Further Reading

- [MDN ReadableStream](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream) - Streaming API reference
- [MDN AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController) - Request cancellation API
- [MDN Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) - EventSource and SSE protocol
- [marked.js](https://marked.js.org/) - Markdown parser for streaming rendering

<!--
Sources Consulted:
- MDN ReadableStream: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
- MDN AbortController: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
- MDN Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Vercel AI SDK streaming patterns: https://ai-sdk.dev/docs/foundations/streaming
-->
