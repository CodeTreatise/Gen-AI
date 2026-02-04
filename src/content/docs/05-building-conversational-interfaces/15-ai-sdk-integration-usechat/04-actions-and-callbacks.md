---
title: "Actions and Callbacks"
---

# Actions and Callbacks

## Introduction

The `useChat` hook provides a rich set of actions and callbacks that give you complete control over the chat experience. Actions let you send messages, stop streams, regenerate responses, and manipulate message history. Callbacks notify you when important events occur — when a response finishes, errors happen, or data arrives.

Mastering these methods enables building sophisticated chat interfaces with features like message editing, conversation branching, and real-time analytics.

### What We'll Cover

- Action methods: sendMessage, stop, regenerate, setMessages
- Event callbacks: onFinish, onError, onData
- Request-level options for customizing individual sends
- Message manipulation patterns
- Error recovery strategies

### Prerequisites

- Completed [useChat Hook Fundamentals](./01-usechat-hook-fundamentals.md)
- Completed [Message Parts Structure](./02-message-parts-structure.md)
- Completed [Status Management](./03-status-management.md)

---

## Action Methods Overview

The `useChat` hook returns several action methods:

```tsx
const {
  sendMessage,        // Send a new message
  stop,               // Cancel current stream
  regenerate,         // Retry last response
  setMessages,        // Modify message history
  clearError,         // Clear error state
  resumeStream,       // Resume interrupted stream
  addToolOutput,      // Provide tool results
  addToolApprovalResponse,  // Respond to tool approval
} = useChat({ /* config */ });
```

---

## sendMessage

The primary method for sending user messages.

### Basic Usage

```tsx
// Send a simple text message
sendMessage({ text: 'Hello, how are you?' });
```

### With Files

```tsx
// Send message with file attachments
const fileInput = document.querySelector<HTMLInputElement>('input[type="file"]');

sendMessage({
  text: 'What is in this image?',
  files: fileInput?.files,  // FileList from input
});
```

### With Metadata

Attach custom data to the message:

```tsx
sendMessage({
  text: 'Help me with this code',
  metadata: {
    timestamp: Date.now(),
    sessionId: 'abc-123',
    context: 'code-review',
  },
});
```

### Request-Level Options

Override transport settings for a specific request:

```tsx
sendMessage(
  { text: 'Urgent request' },
  {
    headers: {
      'X-Priority': 'high',
    },
    body: {
      temperature: 0.2,  // Lower temperature for more focused response
      maxTokens: 500,
    },
    metadata: {
      urgent: true,
    },
  }
);
```

### Message Editing

Replace an existing message by providing its ID:

```tsx
// Edit a previous message
sendMessage({
  text: 'Updated question with more context',
  messageId: 'message-to-replace',
});
```

When you edit a message, the AI regenerates the response based on the updated content.

---

## stop

Cancels the current streaming response.

### Basic Usage

```tsx
const { stop, status } = useChat({ /* config */ });

// Stop button
<button 
  onClick={stop} 
  disabled={status !== 'streaming' && status !== 'submitted'}
>
  Stop Generation
</button>
```

### What Happens on Stop

1. The HTTP request is aborted
2. Streaming stops immediately  
3. Partial content is preserved in messages
4. Status returns to 'ready'
5. `onFinish` is called with `isAbort: true`

### Handling Stopped Responses

```tsx
const { messages, sendMessage } = useChat({
  onFinish: ({ message, isAbort }) => {
    if (isAbort) {
      console.log('User stopped generation');
      // Optionally mark the message as incomplete
    }
  },
});
```

---

## regenerate

Regenerates the last assistant message (or a specific message).

### Basic Usage

```tsx
const { regenerate, status } = useChat({ /* config */ });

// Regenerate button
<button 
  onClick={() => regenerate()} 
  disabled={status !== 'ready' && status !== 'error'}
>
  🔄 Regenerate
</button>
```

### Regenerating a Specific Message

```tsx
// Regenerate a specific assistant message
regenerate({ messageId: 'specific-message-id' });
```

### With Request Options

```tsx
// Regenerate with different parameters
regenerate({
  body: {
    temperature: 0.9,  // More creative this time
  },
});
```

### How Regenerate Works

1. Removes the last assistant message
2. Re-sends the conversation to the API
3. Streams a new response
4. Replaces the old message with the new one

---

## setMessages

Programmatically update the message array.

### Setting Messages Directly

```tsx
const { setMessages } = useChat({ /* config */ });

// Clear all messages
setMessages([]);

// Set specific messages
setMessages([
  {
    id: 'welcome',
    role: 'assistant',
    parts: [{ type: 'text', text: 'Hello! How can I help you today?' }],
  },
]);
```

### Functional Update

Use a function when updating based on current messages:

```tsx
// Remove the last message
setMessages(messages => messages.slice(0, -1));

// Delete a specific message
setMessages(messages => 
  messages.filter(m => m.id !== 'message-to-delete')
);

// Add a system message
setMessages(messages => [
  {
    id: 'system-1',
    role: 'system',
    parts: [{ type: 'text', text: 'You are a helpful assistant.' }],
  },
  ...messages,
]);
```

### Building a Delete Button

```tsx
function MessageActions({ messageId }: { messageId: string }) {
  const { setMessages } = useChat({ /* config */ });
  
  const handleDelete = () => {
    setMessages(messages => messages.filter(m => m.id !== messageId));
  };
  
  return (
    <button onClick={handleDelete} className="text-red-500">
      🗑️ Delete
    </button>
  );
}
```

### Loading Conversation History

```tsx
function ChatWithHistory({ conversationId }: { conversationId: string }) {
  const { messages, setMessages } = useChat({
    id: conversationId,
    transport: new DefaultChatTransport({ api: '/api/chat' }),
  });
  
  useEffect(() => {
    // Load history from database
    async function loadHistory() {
      const response = await fetch(`/api/conversations/${conversationId}`);
      const history = await response.json();
      setMessages(history.messages);
    }
    
    loadHistory();
  }, [conversationId, setMessages]);
  
  return (/* render chat */);
}
```

---

## Event Callbacks

Configure callbacks when initializing useChat:

```tsx
const { messages, sendMessage } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  
  onFinish: ({ message, messages, isAbort, isError }) => {
    // Called when response is complete
  },
  
  onError: (error) => {
    // Called on errors
  },
  
  onData: (dataPart) => {
    // Called when data parts arrive
  },
});
```

---

## onFinish

Called when the assistant's response is complete.

### Callback Parameters

```typescript
interface OnFinishOptions {
  message: UIMessage;      // The response message
  messages: UIMessage[];   // All messages including response
  isAbort: boolean;        // True if user stopped generation
  isDisconnect: boolean;   // True if network disconnected
  isError: boolean;        // True if error occurred during stream
  finishReason?: string;   // Why model stopped: 'stop', 'length', etc.
}
```

### Usage Examples

```tsx
const { messages, sendMessage } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  
  onFinish: ({ message, messages, isAbort, finishReason }) => {
    // Log analytics
    console.log('Response complete', {
      messageId: message.id,
      finishReason,
      wasAborted: isAbort,
    });
    
    // Save to database
    saveConversation(messages);
    
    // Access metadata (token usage, etc.)
    if (message.metadata?.totalTokens) {
      updateTokenUsage(message.metadata.totalTokens);
    }
  },
});
```

### Handling Different Finish Scenarios

```tsx
onFinish: ({ isAbort, isDisconnect, isError, finishReason }) => {
  if (isAbort) {
    // User manually stopped
    showToast('Generation stopped');
    return;
  }
  
  if (isDisconnect) {
    // Network issue
    showToast('Connection lost - response may be incomplete');
    return;
  }
  
  if (isError) {
    // Error during streaming
    showToast('Error occurred during generation');
    return;
  }
  
  if (finishReason === 'length') {
    // Hit token limit
    showToast('Response reached maximum length');
  }
}
```

---

## onError

Called when an error occurs during the request.

### Basic Error Handling

```tsx
const { messages, sendMessage, error } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  
  onError: (error) => {
    console.error('Chat error:', error);
    
    // Send to error tracking
    errorTracker.capture(error);
    
    // Show user-friendly message
    showToast('Something went wrong. Please try again.');
  },
});
```

### Error Types

```tsx
onError: (error) => {
  // Network errors
  if (error.message.includes('fetch')) {
    showToast('Network error - check your connection');
    return;
  }
  
  // Rate limiting
  if (error.message.includes('429') || error.message.includes('rate')) {
    showToast('Too many requests - please wait a moment');
    return;
  }
  
  // Authentication
  if (error.message.includes('401') || error.message.includes('unauthorized')) {
    redirectToLogin();
    return;
  }
  
  // Generic fallback
  showToast('An error occurred');
}
```

---

## onData

Called when data parts arrive during streaming. This is especially useful for handling transient data that won't appear in message.parts.

### Basic Usage

```tsx
const { messages, sendMessage } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  
  onData: (dataPart) => {
    console.log('Received data:', dataPart);
    
    // Handle different data types
    if (dataPart.type === 'data-notification') {
      showNotification(dataPart.data.message);
    }
    
    if (dataPart.type === 'data-progress') {
      updateProgressBar(dataPart.data.percent);
    }
  },
});
```

### Transient Data

Transient data parts are only accessible via `onData` — they don't appear in message.parts:

```tsx
const [progress, setProgress] = useState(0);

const { messages, sendMessage } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  
  onData: (dataPart) => {
    // Server sent: { type: 'data-progress', data: { percent: 50 }, transient: true }
    if (dataPart.type === 'data-progress') {
      setProgress(dataPart.data.percent);
    }
  },
});
```

### Aborting in onData

You can abort processing by throwing an error:

```tsx
onData: (dataPart) => {
  // Check for problematic content
  if (dataPart.type === 'text' && containsProfanity(dataPart.text)) {
    throw new Error('Content moderation: inappropriate content detected');
  }
}
```

---

## Combining Actions and Callbacks

### Message Editing with Regeneration

```tsx
function EditableMessage({ message }: { message: UIMessage }) {
  const { sendMessage, setMessages } = useChat({ /* config */ });
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');
  
  const startEdit = () => {
    // Get the text content from parts
    const textContent = message.parts
      .filter(p => p.type === 'text')
      .map(p => p.text)
      .join('');
    setEditText(textContent);
    setIsEditing(true);
  };
  
  const saveEdit = () => {
    // Remove this message and all after it
    setMessages(messages => {
      const index = messages.findIndex(m => m.id === message.id);
      return messages.slice(0, index);
    });
    
    // Send the edited message
    sendMessage({ text: editText });
    setIsEditing(false);
  };
  
  if (message.role !== 'user') return null;
  
  return (
    <div>
      {isEditing ? (
        <div>
          <textarea 
            value={editText} 
            onChange={e => setEditText(e.target.value)} 
          />
          <button onClick={saveEdit}>Save & Regenerate</button>
          <button onClick={() => setIsEditing(false)}>Cancel</button>
        </div>
      ) : (
        <button onClick={startEdit}>✏️ Edit</button>
      )}
    </div>
  );
}
```

### Conversation Branching

Create alternative responses without losing the original:

```tsx
function BranchButton({ messageId }: { messageId: string }) {
  const { messages, setMessages, regenerate } = useChat({ /* config */ });
  const [branches, setBranches] = useState<Map<string, UIMessage[]>>(new Map());
  
  const createBranch = () => {
    // Save current state as a branch
    const branchId = `branch-${Date.now()}`;
    setBranches(prev => new Map(prev).set(branchId, [...messages]));
    
    // Regenerate from this point
    regenerate({ messageId });
  };
  
  const loadBranch = (branchId: string) => {
    const branchMessages = branches.get(branchId);
    if (branchMessages) {
      setMessages(branchMessages);
    }
  };
  
  return (
    <div>
      <button onClick={createBranch}>🌿 Create Branch</button>
      {/* Branch selector UI */}
    </div>
  );
}
```

### Auto-Save with onFinish

```tsx
function AutoSaveChat({ conversationId }: { conversationId: string }) {
  const { messages, sendMessage } = useChat({
    id: conversationId,
    transport: new DefaultChatTransport({ api: '/api/chat' }),
    
    onFinish: async ({ messages }) => {
      // Auto-save after each response
      await fetch(`/api/conversations/${conversationId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    },
  });
  
  return (/* render chat */);
}
```

---

## Action Reference Table

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `sendMessage(msg, opts?)` | Send a message | User submits input |
| `stop()` | Cancel streaming | User clicks stop |
| `regenerate(opts?)` | Retry response | User wants different response |
| `setMessages(msgs)` | Update history | Loading, editing, branching |
| `clearError()` | Clear error state | Dismiss error |
| `resumeStream()` | Resume interrupted stream | After network recovery |
| `addToolOutput(opts)` | Provide tool result | Client-side tool execution |
| `addToolApprovalResponse(opts)` | Approve/deny tool | Tool approval workflow |

---

## Best Practices

### ✅ Do

| Practice | Reason |
|----------|--------|
| Clear input after sendMessage | Prevents duplicate sends |
| Handle all onFinish scenarios | Complete error handling |
| Use functional setMessages updates | Avoids race conditions |
| Save conversations in onFinish | Reliable persistence |
| Track analytics in callbacks | Non-blocking metrics |

### ❌ Don't

| Anti-pattern | Problem |
|--------------|---------|
| Call sendMessage while streaming | Race conditions |
| Ignore isAbort in onFinish | Mishandle stopped responses |
| Mutate messages directly | React won't update |
| Throw in onFinish | Breaks completion handling |

---

## Hands-on Exercise

### Your Task

Build a chat interface with these features:

1. **Message Actions**: Copy, delete, and regenerate buttons for each assistant message
2. **Auto-title**: When the first assistant response completes, extract a title from the conversation
3. **Usage Tracking**: Display total tokens used across all messages
4. **Export**: Button to export the conversation as JSON

### Requirements

1. Use `onFinish` to track token usage from metadata
2. Use `setMessages` for delete functionality
3. Use `regenerate` for the regenerate button
4. Access message metadata for token counts

<details>
<summary>💡 Hints (click to expand)</summary>

- Store total tokens in component state, update in onFinish
- Use `navigator.clipboard.writeText()` for copy functionality
- For export, use `JSON.stringify(messages, null, 2)`
- Create a Blob and download link for file export

</details>

---

## Summary

✅ `sendMessage` sends messages with optional files, metadata, and request options

✅ `stop` cancels streaming, preserving partial content

✅ `regenerate` retries the last assistant message with optional different parameters

✅ `setMessages` enables editing, deleting, loading, and branching conversations

✅ `onFinish` provides completion info including abort/error flags and finish reason

✅ `onError` handles request failures with access to the error object

✅ `onData` receives streaming data parts including transient notifications

**Next:** [Transport Configuration](./05-transport-configuration.md)

---

## Further Reading

- [useChat API Reference](https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat) — Complete method signatures
- [Chatbot Guide](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot) — Action examples
- [Message Persistence](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-message-persistence) — Saving conversations

---

<!-- 
Sources Consulted:
- useChat Reference: https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat
- AI SDK Chatbot Guide: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- Streaming Data: https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data
-->
