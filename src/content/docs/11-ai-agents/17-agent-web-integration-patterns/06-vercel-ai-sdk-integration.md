---
title: "Vercel AI SDK Integration"
---

# Vercel AI SDK Integration

## Introduction

The **Vercel AI SDK** (now at version 6) is the most comprehensive TypeScript toolkit for building AI-powered applications. It provides two main libraries: **AI SDK Core** for server-side LLM operations (`streamText`, `generateText`, `ToolLoopAgent`) and **AI SDK UI** for client-side React hooks (`useChat`, `useCompletion`). Together, they eliminate the boilerplate of connecting agents to frontends.

This lesson covers how to build full-stack agent applications using the AI SDK — from defining agents with `ToolLoopAgent`, to streaming tool calls with `streamText`, to rendering typed tool parts in React with `useChat`.

### What we'll cover

- AI SDK Core: `streamText`, `generateText`, and tools
- `ToolLoopAgent` class for reusable agent definitions
- AI SDK UI: `useChat` hook with typed tool parts
- Tool calling patterns (server-side, client-side, approval)
- Multi-step agent loops with `stopWhen`
- `createAgentUIStreamResponse` for API routes

### Prerequisites

- Next.js App Router basics (Lesson 17-05)
- JavaScript/TypeScript async/await (Unit 1, Lesson 5)
- AI agent concepts (Unit 11, Lessons 01-05)
- React hooks (`useState`, `useEffect`)

---

## Setting up the AI SDK

Install the core packages and a provider:

```bash
npm install ai @ai-sdk/react @ai-sdk/anthropic zod
```

The AI SDK uses a **provider pattern** — you import model constructors from provider packages:

| Provider Package | Models |
|-----------------|--------|
| `@ai-sdk/anthropic` | Claude Sonnet, Opus, Haiku |
| `@ai-sdk/openai` | GPT-4o, GPT-4, o1, o3 |
| `@ai-sdk/google` | Gemini Pro, Flash |
| `@ai-sdk/xai` | Grok |
| `@ai-sdk/amazon-bedrock` | Bedrock models |

You can also use the **AI Gateway** for provider-agnostic model strings:

```typescript
import { generateText } from 'ai';

// Gateway pattern — no provider import needed
const { text } = await generateText({
  model: 'anthropic/claude-sonnet-4.5',
  prompt: 'What is love?',
});
```

---

## AI SDK Core: streamText and generateText

These are the two fundamental functions for server-side LLM calls.

### generateText — one-shot generation

```typescript
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const { text, usage } = await generateText({
  model: anthropic('claude-sonnet-4-5-20250929'),
  prompt: 'Explain quantum entanglement in one paragraph.',
});

console.log(text);
console.log(`Tokens used: ${usage.totalTokens}`);
```

**Output:**
```
Quantum entanglement is a phenomenon where two particles become linked...
Tokens used: 142
```

### streamText — token-by-token streaming

```typescript
import { streamText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const result = streamText({
  model: anthropic('claude-sonnet-4-5-20250929'),
  prompt: 'Write a haiku about programming.',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

**Output:**
```
Bugs hide in the code
Debugging through the long night
Tests finally pass
```

### streamText in a Next.js route handler

```typescript
// app/api/chat/route.ts
import { streamText, UIMessage, convertToModelMessages } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-5-20250929'),
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
```

> **Note:** `toUIMessageStreamResponse()` returns a streaming response in the AI SDK's UI message stream format, which the `useChat` hook on the client understands natively. No manual SSE formatting needed.

---

## Defining tools

Tools let the model interact with external systems. The AI SDK provides a type-safe `tool()` function with Zod schemas:

```typescript
import { tool } from 'ai';
import { z } from 'zod';

const getWeather = tool({
  description: 'Get the current weather in a given city',
  inputSchema: z.object({
    city: z.string().describe('The city to get weather for'),
  }),
  execute: async ({ city }) => {
    // Call a weather API
    const temps: Record<string, string> = {
      'Tokyo': '72°F, sunny',
      'London': '58°F, cloudy',
      'New York': '65°F, partly cloudy',
    };
    return temps[city] || '60°F, unknown';
  },
});
```

### Tool types in the AI SDK

| Type | Has `execute`? | Runs Where | Use Case |
|------|---------------|-----------|----------|
| **Server-side** | ✅ Yes | Server | API calls, database queries, file ops |
| **Client-side (auto)** | ❌ No | Browser | Location, camera, localStorage |
| **Client-side (interactive)** | ❌ No | Browser | Confirmation dialogs, user input |

---

## ToolLoopAgent class

The `ToolLoopAgent` class (new in AI SDK 6) encapsulates agent configuration into a reusable, type-safe object:

```typescript
import { ToolLoopAgent, tool, stepCountIs } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';

const researchAgent = new ToolLoopAgent({
  model: anthropic('claude-sonnet-4-5-20250929'),
  instructions: `You are a research assistant. 
When asked a question:
1. Search for relevant information
2. Analyze the results  
3. Provide a clear, cited answer`,
  tools: {
    searchWeb: tool({
      description: 'Search the web for information',
      inputSchema: z.object({ query: z.string() }),
      execute: async ({ query }) => {
        return `Results for "${query}": AI advances in 2025...`;
      },
    }),
    calculateMetric: tool({
      description: 'Calculate a numerical metric',
      inputSchema: z.object({ expression: z.string() }),
      execute: async ({ expression }) => {
        return String(eval(expression)); // Use safe parser in production
      },
    }),
  },
  stopWhen: stepCountIs(10),
});
```

### Using the agent

```typescript
// One-shot generation
const { text } = await researchAgent.generate({
  prompt: 'What are the top 3 AI trends in 2025?',
});
console.log(text);

// Streaming
const result = await researchAgent.stream({
  prompt: 'Analyze the JavaScript ecosystem',
});
for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### Agent in a Next.js route handler

The `createAgentUIStreamResponse` function connects an agent directly to the `useChat` hook:

```typescript
// app/api/chat/route.ts
import { createAgentUIStreamResponse } from 'ai';

export async function POST(request: Request) {
  const { messages } = await request.json();

  return createAgentUIStreamResponse({
    agent: researchAgent,
    uiMessages: messages,
  });
}
```

> **🤖 AI Context:** `ToolLoopAgent` handles the entire agent loop automatically — calling the LLM, executing tools, feeding results back, and repeating until the agent generates a text response or hits the step limit. Before AI SDK 6, you had to implement this loop manually.

---

## useChat hook

The `useChat` hook is the client-side counterpart. It manages message state, sends messages to the API, and processes the streaming response:

```tsx
// app/chat/page.tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState } from 'react';

export default function ChatPage() {
  const { messages, sendMessage } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
    }),
  });
  const [input, setInput] = useState('');

  return (
    <div className="chat">
      {messages.map(message => (
        <div key={message.id} className={message.role}>
          {message.parts.map((part, i) => {
            switch (part.type) {
              case 'text':
                return <p key={i}>{part.text}</p>;
              default:
                return null;
            }
          })}
        </div>
      ))}

      <form
        onSubmit={e => {
          e.preventDefault();
          if (input.trim()) {
            sendMessage({ text: input });
            setInput('');
          }
        }}
      >
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask the agent..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
```

---

## Tool calling with useChat

The AI SDK's killer feature is **typed tool parts** — when the model calls a tool, the client receives strongly-typed UI parts that you can render directly.

### Server: Define tools in the route handler

```typescript
// app/api/chat/route.ts
import { streamText, UIMessage, convertToModelMessages } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-5-20250929'),
    messages: await convertToModelMessages(messages),
    tools: {
      // Server-side tool (auto-executed)
      getWeather: {
        description: 'Get weather in a city',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => {
          const options = ['sunny', 'cloudy', 'rainy'];
          return `${city}: ${options[Math.floor(Math.random() * 3)]}`;
        },
      },
      // Client-side tool (needs user confirmation)
      askForConfirmation: {
        description: 'Ask the user for confirmation',
        inputSchema: z.object({
          message: z.string().describe('What to ask confirmation for'),
        }),
        // No execute function = client-side tool
      },
    },
  });

  return result.toUIMessageStreamResponse();
}
```

### Client: Render typed tool parts

```tsx
// app/chat/page.tsx
'use client';

import { useChat } from '@ai-sdk/react';
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithToolCalls,
} from 'ai';
import { useState } from 'react';

export default function Chat() {
  const { messages, sendMessage, addToolOutput } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),

    // Auto-submit when all tool results are available
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,

    // Handle auto-executed client-side tools
    async onToolCall({ toolCall }) {
      if (toolCall.dynamic) return;

      // No client-side auto tools in this example
    },
  });
  const [input, setInput] = useState('');

  return (
    <>
      {messages?.map(message => (
        <div key={message.id}>
          <strong>{message.role}: </strong>
          {message.parts.map((part, index) => {
            switch (part.type) {
              case 'text':
                return <span key={index}>{part.text}</span>;

              // Typed tool part for weather
              case 'tool-getWeather':
                switch (part.state) {
                  case 'input-available':
                    return (
                      <div key={part.toolCallId}>
                        🌤️ Getting weather for {part.input.city}...
                      </div>
                    );
                  case 'output-available':
                    return (
                      <div key={part.toolCallId}>
                        🌤️ Weather: {part.output}
                      </div>
                    );
                  case 'output-error':
                    return (
                      <div key={part.toolCallId}>
                        ❌ Weather error: {part.errorText}
                      </div>
                    );
                }
                break;

              // Typed tool part for confirmation
              case 'tool-askForConfirmation':
                switch (part.state) {
                  case 'input-available':
                    return (
                      <div key={part.toolCallId}>
                        <p>❓ {part.input.message}</p>
                        <button
                          onClick={() =>
                            addToolOutput({
                              tool: 'askForConfirmation',
                              toolCallId: part.toolCallId,
                              output: 'Yes, confirmed.',
                            })
                          }
                        >
                          ✅ Yes
                        </button>
                        <button
                          onClick={() =>
                            addToolOutput({
                              tool: 'askForConfirmation',
                              toolCallId: part.toolCallId,
                              output: 'No, denied.',
                            })
                          }
                        >
                          ❌ No
                        </button>
                      </div>
                    );
                  case 'output-available':
                    return (
                      <div key={part.toolCallId}>
                        Confirmation: {part.output}
                      </div>
                    );
                }
                break;

              default:
                return null;
            }
          })}
        </div>
      ))}

      <form onSubmit={e => {
        e.preventDefault();
        if (input.trim()) {
          sendMessage({ text: input });
          setInput('');
        }
      }}>
        <input value={input} onChange={e => setInput(e.target.value)} />
      </form>
    </>
  );
}
```

---

## Multi-step agent loops

Agents often need multiple tool calls before generating a final response. Use `stopWhen` with `stepCountIs` to control the maximum number of steps:

```typescript
// app/api/agent/route.ts
import {
  streamText,
  UIMessage,
  convertToModelMessages,
  stepCountIs,
} from 'ai';
import { anthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-5-20250929'),
    messages: await convertToModelMessages(messages),
    tools: {
      searchDatabase: {
        description: 'Search the product database',
        inputSchema: z.object({ query: z.string() }),
        execute: async ({ query }) => {
          return `Found 3 products matching "${query}"`;
        },
      },
      getProductDetails: {
        description: 'Get detailed info about a product',
        inputSchema: z.object({ productId: z.string() }),
        execute: async ({ productId }) => {
          return `Product ${productId}: Widget Pro, $29.99, in stock`;
        },
      },
      compareProducts: {
        description: 'Compare two products',
        inputSchema: z.object({
          productA: z.string(),
          productB: z.string(),
        }),
        execute: async ({ productA, productB }) => {
          return `${productA} vs ${productB}: A is cheaper, B has better reviews`;
        },
      },
    },
    // Allow up to 5 tool call rounds
    stopWhen: stepCountIs(5),
    onStepFinish: async ({ usage, finishReason, toolCalls }) => {
      console.log('Step:', {
        tokens: usage.totalTokens,
        finishReason,
        tools: toolCalls?.map(tc => tc.toolName),
      });
    },
  });

  return result.toUIMessageStreamResponse();
}
```

### Rendering step boundaries

On the client, use `step-start` parts to show visual boundaries between tool call rounds:

```tsx
{message.parts.map((part, index) => {
  switch (part.type) {
    case 'step-start':
      return index > 0 ? (
        <hr key={index} className="step-divider" />
      ) : null;

    case 'text':
      return <p key={index}>{part.text}</p>;

    case 'tool-searchDatabase':
    case 'tool-getProductDetails':
    case 'tool-compareProducts':
      return (
        <div key={part.toolCallId} className="tool-call">
          {part.state === 'input-available' && (
            <span>🔧 Calling {part.type.replace('tool-', '')}...</span>
          )}
          {part.state === 'output-available' && (
            <span>✅ {part.output}</span>
          )}
        </div>
      );
  }
})}
```

---

## Tool execution approval

For sensitive operations, require user approval before the server executes a tool:

```typescript
// Server: Mark tool as needing approval
const tools = {
  deleteAccount: tool({
    description: 'Delete a user account',
    inputSchema: z.object({ userId: z.string() }),
    needsApproval: true, // ← Requires user approval
    execute: async ({ userId }) => {
      // Only runs after user approves
      return `Account ${userId} deleted`;
    },
  }),
};
```

```tsx
// Client: Handle approval UI
case 'tool-deleteAccount':
  switch (part.state) {
    case 'approval-requested':
      return (
        <div key={part.toolCallId} className="approval">
          <p>⚠️ Delete account {part.input.userId}?</p>
          <button onClick={() =>
            addToolApprovalResponse({
              id: part.approval.id,
              approved: true,
            })
          }>
            Approve
          </button>
          <button onClick={() =>
            addToolApprovalResponse({
              id: part.approval.id,
              approved: false,
            })
          }>
            Deny
          </button>
        </div>
      );
    case 'output-available':
      return <div key={part.toolCallId}>✅ {part.output}</div>;
  }
```

> **Important:** Use `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses` if you want the conversation to automatically continue after the user approves/denies.

---

## End-to-end type safety

The AI SDK provides type inference so your client components know exactly which tool parts exist:

```typescript
// lib/agent.ts
import { ToolLoopAgent, InferAgentUIMessage } from 'ai';

const myAgent = new ToolLoopAgent({
  // ... configuration with tools
});

// Infer the UIMessage type — includes typed tool parts
export type MyAgentUIMessage = InferAgentUIMessage<typeof myAgent>;
```

```tsx
// components/Chat.tsx
'use client';

import { useChat } from '@ai-sdk/react';
import type { MyAgentUIMessage } from '@/lib/agent';

export function Chat() {
  // Full type safety for messages and tool parts
  const { messages } = useChat<MyAgentUIMessage>();

  // TypeScript knows which tool-* part types are valid
  // and what .input and .output shapes each tool has
}
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Use `ToolLoopAgent` for reusable agents | Single source of truth for model, tools, instructions |
| Use `toUIMessageStreamResponse()` with `useChat` | Handles all streaming protocol details automatically |
| Set `maxDuration` on route handlers | Prevents Vercel function timeout for long agent runs |
| Use `stopWhen: stepCountIs(N)` | Prevents infinite loops and controls cost |
| Define `onStepFinish` for observability | Track token usage and tool calls per step |
| Use `InferAgentUIMessage` for type safety | Catch tool rendering bugs at compile time |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mixing up `streamText` and `generateText` | Use `streamText` for real-time UI, `generateText` for background tasks |
| Forgetting `sendAutomaticallyWhen` | Without it, tool results don't trigger the next agent step |
| Not handling `output-error` state in tool parts | Always render error states for every tool |
| Calling `await addToolOutput(...)` | Never await `addToolOutput` — it causes deadlocks |
| Using `onToolCall` without checking `toolCall.dynamic` | Check `if (toolCall.dynamic) return;` first for proper type narrowing |
| Not setting `maxDuration` in route exports | Agent calls may exceed default 10s Vercel timeout |

---

## Hands-on exercise

### Your task

Build a Next.js application using the Vercel AI SDK with a `ToolLoopAgent` that has three tools: a server-side search tool, a server-side calculator, and a client-side confirmation tool.

### Requirements

1. Define a `ToolLoopAgent` with `searchWeb`, `calculate`, and `askForConfirmation` tools
2. Create an API route using `createAgentUIStreamResponse`
3. Build a chat page with `useChat` that renders typed tool parts
4. Handle the confirmation tool with approve/deny buttons in the UI
5. Configure `sendAutomaticallyWhen` for automatic tool result submission

### Expected result

A chat interface where the agent can search, calculate, and ask for confirmation — with each tool call rendered as a distinct, interactive UI component.

<details>
<summary>💡 Hints (click to expand)</summary>

- Server-side tools have an `execute` function; client-side tools don't
- Use `tool-toolName` as the part type in the switch statement
- Use `addToolOutput` (without `await`) to provide client tool results
- Use `lastAssistantMessageIsCompleteWithToolCalls` for auto-submission

</details>

### Bonus challenges

- [ ] Add `needsApproval: true` to the search tool for sensitive queries
- [ ] Implement `onStepFinish` to log token usage to the console
- [ ] Add `InferAgentUIMessage` type inference for full type safety

---

## Summary

✅ **AI SDK Core** provides `streamText` and `generateText` for server-side LLM calls with automatic tool execution  
✅ **`ToolLoopAgent`** encapsulates agent config (model, tools, instructions, step limits) into reusable objects  
✅ **`useChat`** with typed tool parts enables rendering tool calls as interactive UI components  
✅ Use **`sendAutomaticallyWhen`** with `lastAssistantMessageIsCompleteWithToolCalls` for seamless multi-step flows  
✅ **Tool execution approval** adds human oversight for sensitive operations like deletions or payments

**Next:** [Real-Time Agent Interfaces](./07-real-time-agent-interfaces.md)

---

## Further Reading

- [AI SDK Introduction](https://ai-sdk.dev/docs/introduction) - Official overview
- [Building Agents](https://ai-sdk.dev/docs/agents/building-agents) - ToolLoopAgent class
- [Chatbot Tool Usage](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) - useChat with tools
- [Loop Control](https://ai-sdk.dev/docs/agents/loop-control) - stopWhen and step conditions
- [Generative User Interfaces](https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces) - Dynamic UI generation

<!--
Sources Consulted:
- AI SDK Introduction: https://ai-sdk.dev/docs/introduction
- AI SDK Building Agents: https://ai-sdk.dev/docs/agents/building-agents
- AI SDK Chatbot Tool Usage: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage
- Next.js Route Handlers: https://nextjs.org/docs/app/api-reference/file-conventions/route
-->
