---
title: "useChat Hook Fundamentals"
---

# useChat Hook Fundamentals

## Introduction

The `useChat` hook is the cornerstone of AI SDK's UI layer. It manages the complete lifecycle of a chat conversation — from sending messages to handling streaming responses to displaying errors. Understanding its fundamentals enables you to build production-ready chat interfaces with minimal code.

This lesson covers the essential configuration options and patterns you'll use in every chat application.

### What We'll Cover

- Installing and importing the AI SDK packages
- Initializing useChat with transport configuration
- Configuring API endpoints and custom options
- Understanding the hook's return values
- Setting up the corresponding server-side route

### Prerequisites

- React and React hooks experience
- Node.js/npm familiarity
- Basic Next.js App Router knowledge (for examples)

---

## Installation

The AI SDK is split into multiple packages for modularity:

```bash
# Core AI SDK package
npm install ai

# React-specific hooks (for React/Next.js)
npm install @ai-sdk/react

# Provider packages (install what you need)
npm install @ai-sdk/openai
npm install @ai-sdk/anthropic
npm install @ai-sdk/google
```

For a Next.js project, the minimum installation is:

```bash
npm install ai @ai-sdk/react @ai-sdk/openai
```

---

## Basic Initialization

### The Simplest Setup

At its most basic, `useChat` requires no configuration — it uses sensible defaults:

```tsx
'use client';

import { useChat } from '@ai-sdk/react';

export default function Chat() {
  const { messages, sendMessage, status } = useChat();
  // Uses default: POST /api/chat
  
  return (/* your UI */);
}
```

However, AI SDK 6.x encourages explicit transport configuration for clarity:

```tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

export default function Chat() {
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
    }),
  });
  
  return (/* your UI */);
}
```

### Why Explicit Transport?

The transport-based architecture provides several benefits:

| Benefit | Description |
|---------|-------------|
| **Clarity** | Explicit API endpoint declaration |
| **Flexibility** | Swap HTTP for WebSocket or direct calls |
| **Testability** | Mock transport for unit tests |
| **Type Safety** | Transport options are strongly typed |

---

## Transport Configuration Options

The `DefaultChatTransport` accepts several configuration options:

### API Endpoint

```tsx
transport: new DefaultChatTransport({
  api: '/api/chat',           // Default endpoint
  // OR
  api: '/api/v2/assistant',   // Custom path
  // OR
  api: 'https://api.example.com/chat',  // External API
})
```

### Custom Headers

Headers can be static or dynamic (function-based):

```tsx
// Static headers
transport: new DefaultChatTransport({
  api: '/api/chat',
  headers: {
    'Authorization': 'Bearer your-token',
    'X-Custom-Header': 'value',
  },
})

// Dynamic headers (evaluated on each request)
transport: new DefaultChatTransport({
  api: '/api/chat',
  headers: () => ({
    'Authorization': `Bearer ${getAuthToken()}`,
    'X-Request-ID': generateRequestId(),
  }),
})
```

### Additional Body Fields

Send extra data with every request:

```tsx
// Static body fields
transport: new DefaultChatTransport({
  api: '/api/chat',
  body: {
    userId: '12345',
    sessionId: 'abc-123',
  },
})

// Dynamic body fields
transport: new DefaultChatTransport({
  api: '/api/chat',
  body: () => ({
    userId: getCurrentUser().id,
    timestamp: Date.now(),
  }),
})
```

### Credentials

Control cookie handling for cross-origin requests:

```tsx
transport: new DefaultChatTransport({
  api: '/api/chat',
  credentials: 'include',     // Send cookies
  // OR
  credentials: 'same-origin', // Default
  // OR
  credentials: 'omit',        // Never send cookies
})
```

### Complete Configuration Example

```tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useAuth } from '@/hooks/useAuth';

export default function AuthenticatedChat() {
  const { token, userId } = useAuth();
  
  const { messages, sendMessage, status, error } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
      headers: () => ({
        'Authorization': `Bearer ${token}`,
      }),
      body: {
        userId,
        model: 'gpt-4o',
      },
      credentials: 'include',
    }),
  });
  
  // ... rest of component
}
```

---

## Hook Return Values

The `useChat` hook returns an object with state and methods:

### State Properties

```tsx
const {
  // Message array - the conversation history
  messages,
  
  // Current status: 'ready' | 'submitted' | 'streaming' | 'error'
  status,
  
  // Error object if status === 'error'
  error,
  
  // Unique chat ID
  id,
} = useChat({ /* config */ });
```

### Action Methods

```tsx
const {
  // Send a new message
  sendMessage,
  
  // Stop the current streaming response
  stop,
  
  // Regenerate the last assistant message
  regenerate,
  
  // Update messages programmatically
  setMessages,
  
  // Clear error state
  clearError,
  
  // Resume an interrupted stream
  resumeStream,
  
  // Add tool results (for function calling)
  addToolOutput,
  addToolApprovalResponse,
} = useChat({ /* config */ });
```

### Type Signatures

For TypeScript users, here are the key types:

```typescript
// Message type
interface UIMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  parts: UIMessagePart[];
  metadata?: unknown;
}

// sendMessage signature
sendMessage: (
  message?: { text: string; files?: FileList | FileUIPart[]; metadata?; messageId?: string },
  options?: ChatRequestOptions
) => Promise<void>;

// Status type
type ChatStatus = 'ready' | 'submitted' | 'streaming' | 'error';
```

---

## Chat ID Configuration

Each chat instance has a unique ID for identification:

```tsx
// Auto-generated ID
const { id } = useChat();
console.log(id); // e.g., "chat_abc123..."

// Custom ID (useful for persistence)
const { id } = useChat({
  id: 'user-session-12345',
  transport: new DefaultChatTransport({ api: '/api/chat' }),
});
```

Using custom IDs is essential when:
- Persisting conversations to a database
- Resuming interrupted streams
- Sharing chat state between components

---

## Initial Messages

Pre-populate the chat with existing messages:

```tsx
const { messages } = useChat({
  messages: [
    {
      id: 'welcome-1',
      role: 'assistant',
      parts: [{ type: 'text', text: 'Hello! How can I help you today?' }],
    },
  ],
  transport: new DefaultChatTransport({ api: '/api/chat' }),
});
```

This is useful for:
- Loading conversation history from a database
- Displaying a welcome message
- Continuing a previous session

---

## Server-Side Route Setup

The `useChat` hook sends requests to your API route. Here's the standard setup:

### Next.js App Router

```typescript
// app/api/chat/route.ts
import { convertToModelMessages, streamText, UIMessage } from 'ai';
import { openai } from '@ai-sdk/openai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  // Extract messages from request
  const { messages }: { messages: UIMessage[] } = await req.json();

  // Generate streaming response
  const result = streamText({
    model: openai('gpt-4o'),
    system: 'You are a helpful assistant.',
    messages: await convertToModelMessages(messages),
  });

  // Return as UI message stream
  return result.toUIMessageStreamResponse();
}
```

### Key Server Components

| Function | Purpose |
|----------|---------|
| `convertToModelMessages` | Converts UI messages to model format |
| `streamText` | Generates streaming text from AI model |
| `toUIMessageStreamResponse` | Formats response for useChat consumption |

### Accessing Custom Body Fields

If you send extra fields via the transport's `body` option:

```typescript
export async function POST(req: Request) {
  const { 
    messages, 
    userId,     // Custom field
    sessionId,  // Custom field
  }: { 
    messages: UIMessage[]; 
    userId: string;
    sessionId: string;
  } = await req.json();

  // Use custom fields for authorization, logging, etc.
  console.log(`Request from user: ${userId}`);
  
  // ... generate response
}
```

---

## Complete Working Example

Here's a full example bringing everything together:

### Client Component

```tsx
// components/Chat.tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState } from 'react';

export default function Chat() {
  const [input, setInput] = useState('');
  
  const { messages, sendMessage, status, error, stop } = useChat({
    id: 'main-chat',
    transport: new DefaultChatTransport({
      api: '/api/chat',
    }),
    messages: [
      {
        id: 'welcome',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Hello! I am your AI assistant. How can I help?' }],
      },
    ],
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || status !== 'ready') return;
    
    sendMessage({ text: input });
    setInput('');
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map(message => (
          <div
            key={message.id}
            className={`p-3 rounded-lg ${
              message.role === 'user'
                ? 'bg-blue-100 ml-auto max-w-[80%]'
                : 'bg-gray-100 mr-auto max-w-[80%]'
            }`}
          >
            {message.parts.map((part, i) =>
              part.type === 'text' ? <p key={i}>{part.text}</p> : null
            )}
          </div>
        ))}
        
        {/* Loading indicator */}
        {status === 'submitted' && (
          <div className="text-gray-500 italic">Thinking...</div>
        )}
        {status === 'streaming' && (
          <div className="text-gray-500 italic">Typing...</div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-100 text-red-700 p-3 rounded mb-4">
          Error: {error.message}
        </div>
      )}

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={status !== 'ready'}
          placeholder="Type your message..."
          className="flex-1 p-2 border rounded"
        />
        {status === 'streaming' || status === 'submitted' ? (
          <button
            type="button"
            onClick={stop}
            className="px-4 py-2 bg-red-500 text-white rounded"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
          >
            Send
          </button>
        )}
      </form>
    </div>
  );
}
```

### API Route

```typescript
// app/api/chat/route.ts
import { convertToModelMessages, streamText, UIMessage } from 'ai';
import { openai } from '@ai-sdk/openai';

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    system: `You are a helpful, friendly assistant. Keep responses concise.`,
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
```

**Expected Behavior:**

1. Page loads with welcome message
2. User types a message and clicks Send
3. Input clears and shows "Thinking..."
4. Response streams in word-by-word, showing "Typing..."
5. When complete, status returns to "ready"
6. Stop button appears during streaming to cancel

---

## Best Practices

### ✅ Do

| Practice | Reason |
|----------|--------|
| Use explicit transport configuration | Clearer code, easier debugging |
| Manage input state yourself | Full control over form behavior |
| Disable input during streaming | Prevent duplicate submissions |
| Handle error state | Show user-friendly error messages |
| Use custom chat IDs for persistence | Enables conversation recovery |

### ❌ Don't

| Anti-pattern | Problem |
|--------------|---------|
| Ignore error state | Users see broken UI |
| Allow submit while streaming | Creates race conditions |
| Hardcode API keys client-side | Security vulnerability |
| Skip `maxDuration` on routes | Streaming may timeout |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting `'use client'` directive | Add `'use client'` at top of component file |
| Missing `maxDuration` export | Add `export const maxDuration = 30;` in API route |
| Not awaiting `convertToModelMessages` | It returns a Promise — always use `await` |
| Input not clearing after send | Manually clear with `setInput('')` after `sendMessage` |

---

## Hands-on Exercise

### Your Task

Build a themed chat interface with the following features:

1. A chat with a custom AI persona (e.g., "You are a pirate who gives sailing advice")
2. Custom styling that matches the persona theme
3. Error handling with a retry button
4. Message timestamps

### Requirements

1. Create a new chat component with useChat
2. Configure transport with custom headers including a `X-Theme` header
3. Display timestamps for each message
4. Show an error state with a "Try Again" button that calls `regenerate()`
5. Style appropriately for your chosen theme

### Expected Result

A functional themed chatbot where:
- Messages appear with timestamps
- Errors show a styled error box with retry option
- The AI responds in character
- The UI matches the persona theme

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Date.now()` when sending to track timestamps via metadata
- Pass theme info via `body` in transport configuration
- Access `regenerate()` from useChat return value
- Use message metadata to store and display timestamps

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState } from 'react';

export default function PirateChat() {
  const [input, setInput] = useState('');
  
  const { messages, sendMessage, status, error, regenerate } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
      headers: {
        'X-Theme': 'pirate',
      },
      body: {
        persona: 'pirate',
      },
    }),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    sendMessage({ 
      text: input,
      metadata: { timestamp: Date.now() },
    });
    setInput('');
  };

  const formatTime = (timestamp?: number) => {
    if (!timestamp) return '';
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="bg-amber-50 min-h-screen p-4">
      <h1 className="text-2xl font-bold text-amber-900 mb-4">
        🏴‍☠️ Captain AI's Sailing Advice
      </h1>
      
      <div className="space-y-4 mb-4">
        {messages.map(message => (
          <div
            key={message.id}
            className={`p-3 rounded ${
              message.role === 'user'
                ? 'bg-blue-200 ml-8'
                : 'bg-amber-200 mr-8 border-2 border-amber-400'
            }`}
          >
            <div className="text-xs text-gray-600 mb-1">
              {message.role === 'user' ? 'You' : '🏴‍☠️ Captain AI'} 
              {' • '}
              {formatTime((message.metadata as any)?.timestamp)}
            </div>
            {message.parts.map((part, i) =>
              part.type === 'text' ? <p key={i}>{part.text}</p> : null
            )}
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 p-3 rounded mb-4">
          <p className="text-red-700">Arrr! Something went wrong!</p>
          <button 
            onClick={() => regenerate()}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded"
          >
            Try Again, Matey!
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask the captain..."
          className="flex-1 p-2 border-2 border-amber-400 rounded"
          disabled={status !== 'ready'}
        />
        <button
          type="submit"
          className="px-4 py-2 bg-amber-600 text-white rounded"
          disabled={status !== 'ready'}
        >
          ⚓ Send
        </button>
      </form>
    </div>
  );
}
```

</details>

### Bonus Challenges

- [ ] Add a "Clear Chat" button using `setMessages([])`
- [ ] Persist the chat ID to localStorage
- [ ] Add multiple persona options in a dropdown

---

## Summary

✅ The `useChat` hook provides complete chat state management with minimal configuration

✅ `DefaultChatTransport` configures HTTP communication with support for headers, body, and credentials

✅ The hook returns state (`messages`, `status`, `error`) and actions (`sendMessage`, `stop`, `regenerate`)

✅ Server routes use `streamText` and `toUIMessageStreamResponse` to stream AI responses

✅ Custom chat IDs enable conversation persistence and stream resumption

**Next:** [Message Parts Structure](./02-message-parts-structure.md)

---

## Further Reading

- [useChat API Reference](https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat) — Complete parameter documentation
- [Transport Documentation](https://ai-sdk.dev/docs/ai-sdk-ui/transport) — Advanced transport patterns
- [AI SDK Examples](https://github.com/vercel/ai/tree/main/examples) — Official code examples

---

<!-- 
Sources Consulted:
- AI SDK Chatbot Guide: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- useChat Reference: https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat
- Transport Documentation: https://ai-sdk.dev/docs/ai-sdk-ui/transport
-->
