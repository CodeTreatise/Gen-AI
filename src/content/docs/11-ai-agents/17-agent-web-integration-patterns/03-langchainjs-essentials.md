---
title: "LangChain.js Essentials"
---

# LangChain.js Essentials

## Introduction

LangChain.js brings the power of LangChain's agent framework to JavaScript and TypeScript. It runs in both **Node.js** and **browser** environments, enabling developers to build AI-powered applications using the same language on frontend and backend. With its `createAgent` abstraction, model integrations, and tool system, LangChain.js provides a high-level way to build agents without managing the raw API loop yourself.

This lesson covers how to use LangChain.js for web-facing agent applications — from creating simple agents in Node.js to integrating them with React and Vue frontends via streaming.

### What we'll cover

- LangChain.js architecture and core concepts
- Creating agents with `createAgent`
- Model integrations and provider flexibility
- Streaming agent responses in Node.js
- React integration patterns
- Client-side chains and limitations
- Vue integration patterns

### Prerequisites

- JavaScript/TypeScript ES6+ (Unit 1, Lessons 3, 12)
- Async/await and Promises (Unit 1, Lesson 5)
- Agent fundamentals (Unit 11, Lessons 1–3)
- React basics (helpful but not required)

---

## LangChain.js architecture

LangChain.js is organized into a modular package structure:

| Package | Purpose | Install |
|---------|---------|---------|
| `langchain` | Core agents, chains, and abstractions | `npm install langchain` |
| `@langchain/core` | Base interfaces and types | Included with `langchain` |
| `@langchain/openai` | OpenAI model provider | `npm install @langchain/openai` |
| `@langchain/anthropic` | Anthropic model provider | `npm install @langchain/anthropic` |
| `@langchain/google-genai` | Google Gemini provider | `npm install @langchain/google-genai` |
| `@langchain/langgraph` | Stateful graph-based agents | `npm install @langchain/langgraph` |

```bash
# Install LangChain.js with Anthropic provider
npm install langchain @langchain/core @langchain/anthropic
```

> **Note:** LangChain.js agents are built on top of LangGraph. The `createAgent` function provides a high-level abstraction, while LangGraph gives you full control over the agent's execution graph.

---

## Creating agents with createAgent

The simplest way to build an agent in LangChain.js is the `createAgent` function. It handles the tool loop, message management, and model interaction:

```typescript
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

// Define a tool
const getWeather = tool(
  async ({ city }) => {
    // In production, call a real weather API
    const conditions = ['sunny', 'cloudy', 'rainy', 'windy'];
    const temp = Math.floor(Math.random() * 30) + 50;
    return `${city}: ${temp}°F, ${conditions[Math.floor(Math.random() * conditions.length)]}`;
  },
  {
    name: 'get_weather',
    description: 'Get the current weather for a city',
    schema: z.object({
      city: z.string().describe('The city name'),
    }),
  }
);

// Create an agent
const agent = createAgent({
  model: 'claude-sonnet-4-5-20250929',
  tools: [getWeather],
});

// Invoke the agent
const result = await agent.invoke({
  messages: [{ role: 'user', content: 'What is the weather in Tokyo?' }],
});

console.log(result.messages[result.messages.length - 1].content);
```

**Output:**
```
The weather in Tokyo is currently 68°F and sunny.
```

### Multiple tools

Agents can use multiple tools and call them in sequence:

```typescript
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

const searchWeb = tool(
  async ({ query }) => `Search results for "${query}": [Result 1, Result 2, Result 3]`,
  {
    name: 'search_web',
    description: 'Search the web for information',
    schema: z.object({
      query: z.string().describe('Search query'),
    }),
  }
);

const calculator = tool(
  async ({ expression }) => {
    try {
      // Simple safe eval for math expressions
      const result = Function(`"use strict"; return (${expression})`)();
      return `${expression} = ${result}`;
    } catch {
      return `Error evaluating: ${expression}`;
    }
  },
  {
    name: 'calculator',
    description: 'Evaluate a mathematical expression',
    schema: z.object({
      expression: z.string().describe('Math expression like "2 + 3 * 4"'),
    }),
  }
);

const agent = createAgent({
  model: 'claude-sonnet-4-5-20250929',
  tools: [searchWeb, calculator],
});

const result = await agent.invoke({
  messages: [
    { role: 'user', content: 'What is the population of France, and what is that divided by 1000?' },
  ],
});
```

---

## Streaming agent responses

For web applications, streaming is essential. LangChain.js supports streaming via the `.stream()` method:

### Basic streaming in Node.js

```typescript
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

const getWeather = tool(
  async ({ city }) => `${city}: 72°F, sunny`,
  {
    name: 'get_weather',
    description: 'Get weather for a city',
    schema: z.object({ city: z.string() }),
  }
);

const agent = createAgent({
  model: 'claude-sonnet-4-5-20250929',
  tools: [getWeather],
});

// Stream the agent's response
const stream = await agent.stream({
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
});

for await (const event of stream) {
  // Each event contains the updates from a single step
  if (event.messages) {
    for (const msg of event.messages) {
      if (msg.content) {
        process.stdout.write(msg.content);
      }
    }
  }
}
```

**Output (streamed):**
```
The weather in Paris is currently 72°F and sunny.
```

### Express.js SSE endpoint with LangChain.js

```typescript
import express from 'express';
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

const app = express();
app.use(express.json());

const getWeather = tool(
  async ({ city }) => `${city}: 72°F, sunny`,
  {
    name: 'get_weather',
    description: 'Get weather for a city',
    schema: z.object({ city: z.string() }),
  }
);

const agent = createAgent({
  model: 'claude-sonnet-4-5-20250929',
  tools: [getWeather],
});

app.post('/api/chat/stream', async (req, res) => {
  const { messages } = req.body;

  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const stream = await agent.stream({
      messages: messages.map((m: any) => ({
        role: m.role,
        content: m.content,
      })),
    });

    for await (const event of stream) {
      if (event.messages) {
        for (const msg of event.messages) {
          // Tool call events
          if (msg.tool_calls?.length) {
            for (const tc of msg.tool_calls) {
              res.write(`data: ${JSON.stringify({
                type: 'tool_call',
                tool: tc.name,
                input: tc.args,
              })}\n\n`);
            }
          }
          // Text content
          if (msg.content && typeof msg.content === 'string') {
            res.write(`data: ${JSON.stringify({
              type: 'text',
              content: msg.content,
            })}\n\n`);
          }
        }
      }
    }

    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
  } catch (error) {
    res.write(`data: ${JSON.stringify({
      type: 'error',
      content: error.message,
    })}\n\n`);
  }

  res.end();
});

app.listen(3001, () => console.log('LangChain.js agent server on :3001'));
```

---

## React integration patterns

LangChain.js can be used with React in two patterns: **server-side agent with React frontend** (recommended) or **client-side chains** (limited use cases).

### Pattern 1: Server-side agent + React fetch

The most common pattern: agent runs on the server, React consumes the SSE stream.

```tsx
import { useState, useCallback, useRef } from 'react';

interface AgentMessage {
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: Array<{
    name: string;
    input: Record<string, unknown>;
    result?: string;
  }>;
}

function useLangChainAgent(endpoint: string) {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    const userMsg: AgentMessage = { role: 'user', content };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);
    controllerRef.current = new AbortController();

    const assistantMsg: AgentMessage = {
      role: 'assistant',
      content: '',
      toolCalls: [],
    };
    setMessages(prev => [...prev, assistantMsg]);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMsg].map(m => ({
            role: m.role,
            content: m.content,
          })),
        }),
        signal: controllerRef.current.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop()!;

        for (const raw of events) {
          if (!raw.startsWith('data: ')) continue;
          const event = JSON.parse(raw.slice(6));

          setMessages(prev => {
            const updated = [...prev];
            const last = { ...updated[updated.length - 1] };

            if (event.type === 'text') {
              last.content += event.content;
            } else if (event.type === 'tool_call') {
              last.toolCalls = [
                ...(last.toolCalls || []),
                { name: event.tool, input: event.input },
              ];
            }

            updated[updated.length - 1] = last;
            return updated;
          });
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('Agent error:', error);
      }
    } finally {
      setIsLoading(false);
    }
  }, [endpoint, messages]);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  return { messages, sendMessage, cancel, isLoading };
}

// Usage in a component
function ChatWithAgent() {
  const { messages, sendMessage, cancel, isLoading } = useLangChainAgent(
    '/api/chat/stream'
  );
  const [input, setInput] = useState('');

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={msg.role}>
          {msg.toolCalls?.map((tc, j) => (
            <div key={j} className="tool-badge">
              🔧 {tc.name}({JSON.stringify(tc.input)})
            </div>
          ))}
          <p>{msg.content}</p>
        </div>
      ))}

      <form onSubmit={(e) => {
        e.preventDefault();
        if (input.trim()) { sendMessage(input); setInput(''); }
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        {isLoading ? (
          <button type="button" onClick={cancel}>Stop</button>
        ) : (
          <button type="submit">Send</button>
        )}
      </form>
    </div>
  );
}
```

### Pattern 2: Client-side chains (limited)

For simple use cases where you don't need tools or agent loops, LangChain.js can run directly in the browser:

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

// ⚠️ WARNING: This exposes your API key in the browser!
// Only use for prototyping, never in production
const model = new ChatOpenAI({
  modelName: 'gpt-4o-mini',
  openAIApiKey: 'sk-...', // NEVER do this in production
  streaming: true,
});

const stream = await model.stream([
  new HumanMessage('Explain React hooks in 3 sentences'),
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
```

> **Warning:** Running LLM calls client-side exposes API keys. Only use this pattern for internal tools, prototypes, or when proxying through your own backend API. For production, always run LangChain.js agents on the server.

---

## Vue integration patterns

LangChain.js works with Vue using the same server-side agent pattern:

```vue
<script setup lang="ts">
import { ref, reactive } from 'vue';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const messages = reactive<Message[]>([]);
const input = ref('');
const isLoading = ref(false);
let controller: AbortController | null = null;

async function sendMessage() {
  const content = input.value.trim();
  if (!content || isLoading.value) return;

  messages.push({ role: 'user', content });
  input.value = '';
  isLoading.value = true;
  controller = new AbortController();

  messages.push({ role: 'assistant', content: '' });

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages.map(m => ({ role: m.role, content: m.content })),
      }),
      signal: controller.signal,
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop()!;

      for (const raw of events) {
        if (!raw.startsWith('data: ')) continue;
        const event = JSON.parse(raw.slice(6));

        if (event.type === 'text') {
          messages[messages.length - 1].content += event.content;
        }
      }
    }
  } catch (error) {
    if ((error as Error).name !== 'AbortError') {
      console.error('Error:', error);
    }
  } finally {
    isLoading.value = false;
  }
}

function cancel() {
  controller?.abort();
}
</script>

<template>
  <div class="chat">
    <div v-for="(msg, i) in messages" :key="i" :class="msg.role">
      <p>{{ msg.content }}</p>
    </div>

    <form @submit.prevent="sendMessage">
      <input v-model="input" :disabled="isLoading" placeholder="Ask something..." />
      <button v-if="isLoading" type="button" @click="cancel">Stop</button>
      <button v-else type="submit">Send</button>
    </form>
  </div>
</template>
```

---

## LangChain.js vs Vercel AI SDK

When should you use LangChain.js vs the Vercel AI SDK?

| Feature | LangChain.js | Vercel AI SDK |
|---------|-------------|---------------|
| **Primary focus** | Agent framework + model abstraction | UI hooks + streaming |
| **Agent abstraction** | `createAgent` with LangGraph backend | `ToolLoopAgent` class |
| **Streaming UI** | Manual SSE consumption | `useChat`, `useCompletion` hooks |
| **Framework support** | React, Vue, Svelte, Node.js | React, Next.js, Svelte, Vue, Nuxt |
| **Tool system** | Zod-based tools with rich types | Zod-based tools with typed UI parts |
| **Model providers** | 30+ via `@langchain/*` packages | 20+ via `@ai-sdk/*` packages |
| **Best for** | Complex agent logic, multi-model | Chat UIs, streaming, Next.js apps |
| **Observability** | LangSmith integration | OpenTelemetry support |
| **Edge runtime** | Limited browser support | Full edge runtime support |

> **🤖 AI Context:** Many production applications use **both** — LangChain.js / LangGraph for backend agent orchestration and Vercel AI SDK for frontend streaming UI. They complement rather than compete with each other.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Run agents server-side, never client-side | Protects API keys and enables full tool access |
| Use `createAgent` for simple agents | Handles the tool loop and message management automatically |
| Use LangGraph for complex workflows | When you need cycles, state persistence, or human-in-loop |
| Stream responses over SSE | Users see progress; prevents timeout on long agent runs |
| Use provider-specific packages | `@langchain/anthropic` instead of generic — better types and features |
| Pin package versions | LangChain.js evolves rapidly; pin versions to avoid breaking changes |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Exposing API keys in browser code | Always run agent logic on the server |
| Using `langchain` for everything | Use specific provider packages (`@langchain/openai`, etc.) |
| Not handling streaming errors | Wrap stream consumption in try/catch |
| Ignoring package version conflicts | Use `npm ls` to check for version mismatches between `@langchain/*` packages |
| Building complex agents with `createAgent` | Use LangGraph for agents needing cycles, branching, or persistence |

---

## Hands-on exercise

### Your task

Build a Node.js Express server using LangChain.js `createAgent` that exposes an SSE streaming endpoint, and a simple HTML frontend that consumes the stream and displays the agent's responses.

### Requirements

1. Create an agent with at least 2 tools (e.g., `get_weather` and `search`)
2. Expose a `POST /api/chat` SSE endpoint that streams agent events
3. Build a frontend page that sends messages and renders streaming responses
4. Handle tool calls by displaying them as status badges in the UI

### Expected result

A working chat interface where user messages trigger agent reasoning with tool calls, and responses stream in token by token with tool execution visible.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `agent.stream()` to get streaming events
- Format events as SSE: `data: {json}\n\n`
- On the frontend, use `fetch` + `ReadableStream` (from Lesson 02)
- Check `event.messages` for both text content and tool calls

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```typescript
// server.ts
import express from 'express';
import { createAgent, tool } from 'langchain';
import * as z from 'zod';

const app = express();
app.use(express.json());

const weatherTool = tool(
  async ({ city }) => `${city}: 72°F, sunny`,
  {
    name: 'get_weather',
    description: 'Get weather for a city',
    schema: z.object({ city: z.string() }),
  }
);

const searchTool = tool(
  async ({ query }) => `Results for "${query}": Found 3 relevant articles.`,
  {
    name: 'search',
    description: 'Search for information',
    schema: z.object({ query: z.string() }),
  }
);

const agent = createAgent({
  model: 'claude-sonnet-4-5-20250929',
  tools: [weatherTool, searchTool],
});

app.post('/api/chat', async (req, res) => {
  const { messages } = req.body;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');

  const stream = await agent.stream({ messages });

  for await (const event of stream) {
    if (event.messages) {
      for (const msg of event.messages) {
        if (msg.tool_calls?.length) {
          for (const tc of msg.tool_calls) {
            res.write(`data: ${JSON.stringify({
              type: 'tool_call', tool: tc.name, input: tc.args
            })}\n\n`);
          }
        }
        if (msg.content && typeof msg.content === 'string') {
          res.write(`data: ${JSON.stringify({
            type: 'text', content: msg.content
          })}\n\n`);
        }
      }
    }
  }

  res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
  res.end();
});

app.listen(3001);
```

</details>

### Bonus challenges

- [ ] Add conversation memory by persisting messages across requests
- [ ] Implement a model selector that lets users choose between OpenAI and Anthropic
- [ ] Add LangSmith tracing to observe agent execution steps

---

## Summary

✅ **LangChain.js** provides a modular JavaScript/TypeScript framework for building agents with `createAgent`  
✅ Agents should run **server-side** — use Express or Next.js API routes with SSE for streaming  
✅ The **SSE + fetch** pattern works identically for React and Vue frontends  
✅ Use **provider-specific packages** (`@langchain/anthropic`, `@langchain/openai`) for better types  
✅ For complex stateful workflows, graduate from `createAgent` to **LangGraph.js** (next lesson)

**Next:** [LangGraph.js for Frontend](./04-langgraphjs-for-frontend.md)

---

## Further Reading

- [LangChain.js Documentation](https://docs.langchain.com/oss/javascript/langchain/overview) - Official docs
- [LangChain.js Agents](https://docs.langchain.com/oss/javascript/langchain/agents) - Agent abstraction guide
- [LangChain.js Models](https://docs.langchain.com/oss/javascript/langchain/models) - Model provider integrations
- [LangChain.js Tools](https://docs.langchain.com/oss/javascript/langchain/tools) - Tool creation reference
- [LangChain Academy](https://academy.langchain.com/) - Free courses on LangGraph

<!--
Sources Consulted:
- LangChain.js overview: https://docs.langchain.com/oss/javascript/langchain/overview
- LangChain.js agents: https://docs.langchain.com/oss/javascript/langchain/agents
- LangGraph.js overview: https://docs.langchain.com/oss/javascript/langgraph/overview
- LangChain.js Deep Agents: https://docs.langchain.com/oss/javascript/deepagents/overview
-->
