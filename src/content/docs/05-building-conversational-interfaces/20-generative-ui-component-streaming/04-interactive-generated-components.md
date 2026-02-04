---
title: "Interactive Generated Components"
---

# Interactive Generated Components

## Introduction

Generated components aren't just for display—they can be interactive. Users can click buttons, fill forms, and trigger actions that feed back into the AI conversation. This lesson covers client-side tools, user confirmation flows, and state management for dynamic generated UI.

### What We'll Cover

- Client-side tools for browser interactions
- Tool execution approval (`needsApproval`)
- User interaction handling in generated components
- Event callbacks to the AI
- Component lifecycle in chat
- Automatic submission with `sendAutomaticallyWhen`

### Prerequisites

- [Streaming React Components](./01-streaming-react-components.md)
- [Server-Side Generation](./03-server-side-generation.md)
- Understanding of React event handling

---

## Client-Side Tools

Client-side tools execute in the browser rather than on the server. They're useful for:

- Accessing browser APIs (geolocation, clipboard)
- Updating client-side state
- Triggering user interactions

### Defining Client-Side Tools

Client-side tools have no `execute` function on the server:

```typescript
// app/api/chat/route.ts
import { streamText, convertToModelMessages, UIMessage } from 'ai';
import { z } from 'zod';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    messages: await convertToModelMessages(messages),
    tools: {
      // Server-side tool (has execute)
      getWeather: {
        description: 'Get weather for a location',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => {
          return await fetchWeather(city);
        },
      },

      // Client-side tool (no execute)
      askForConfirmation: {
        description: 'Ask the user to confirm an action',
        inputSchema: z.object({
          message: z.string().describe('The confirmation message'),
          action: z.string().describe('What will happen if confirmed'),
        }),
        // No execute function = client handles it
      },

      // Client-side tool for browser APIs
      getLocation: {
        description: 'Get the user current location',
        inputSchema: z.object({}),
        // Client will use navigator.geolocation
      },
    },
  });

  return result.toUIMessageStreamResponse();
}
```

### Handling Client-Side Tools with onToolCall

```tsx
'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

export default function Chat() {
  const { messages, sendMessage, addToolOutput } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),

    // Handle automatically executed client-side tools
    async onToolCall({ toolCall }) {
      // Check for dynamic tools first
      if (toolCall.dynamic) return;

      if (toolCall.toolName === 'getLocation') {
        try {
          const position = await new Promise<GeolocationPosition>(
            (resolve, reject) => {
              navigator.geolocation.getCurrentPosition(resolve, reject);
            }
          );

          // Provide tool result (no await to avoid deadlock)
          addToolOutput({
            tool: 'getLocation',
            toolCallId: toolCall.toolCallId,
            output: {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
            },
          });
        } catch (error) {
          addToolOutput({
            tool: 'getLocation',
            toolCallId: toolCall.toolCallId,
            state: 'output-error',
            errorText: 'Location access denied',
          });
        }
      }
    },
  });

  // ... render
}
```

---

## User Interaction Tools

Some tools require user input before providing a result. Render interactive UI for these:

### Confirmation Dialog Pattern

```tsx
function MessagePart({ part, addToolOutput }: { 
  part: any; 
  addToolOutput: any;
}) {
  if (part.type === 'tool-askForConfirmation') {
    switch (part.state) {
      case 'input-streaming':
        return <div>Loading confirmation...</div>;

      case 'input-available':
        // Render interactive confirmation UI
        return (
          <ConfirmationDialog
            message={part.input.message}
            action={part.input.action}
            onConfirm={() => {
              addToolOutput({
                tool: 'askForConfirmation',
                toolCallId: part.toolCallId,
                output: { confirmed: true, timestamp: Date.now() },
              });
            }}
            onDeny={() => {
              addToolOutput({
                tool: 'askForConfirmation',
                toolCallId: part.toolCallId,
                output: { confirmed: false, reason: 'User declined' },
              });
            }}
          />
        );

      case 'output-available':
        // Show result after user responded
        return (
          <div className="confirmation-result">
            {part.output.confirmed ? (
              <span className="text-green-600">✓ Confirmed</span>
            ) : (
              <span className="text-red-600">✗ Declined</span>
            )}
          </div>
        );

      case 'output-error':
        return <div className="error">{part.errorText}</div>;
    }
  }

  return null;
}
```

### Confirmation Dialog Component

```tsx
interface ConfirmationDialogProps {
  message: string;
  action: string;
  onConfirm: () => void;
  onDeny: () => void;
}

function ConfirmationDialog({ 
  message, 
  action, 
  onConfirm, 
  onDeny 
}: ConfirmationDialogProps) {
  return (
    <div className="confirmation-dialog">
      <div className="dialog-icon">⚠️</div>
      <div className="dialog-content">
        <p className="dialog-message">{message}</p>
        <p className="dialog-action">
          <strong>Action:</strong> {action}
        </p>
      </div>
      <div className="dialog-buttons">
        <button onClick={onDeny} className="btn-secondary">
          Cancel
        </button>
        <button onClick={onConfirm} className="btn-primary">
          Confirm
        </button>
      </div>
    </div>
  );
}
```

```css
.confirmation-dialog {
  border: 1px solid #fbbf24;
  background: #fffbeb;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.dialog-icon {
  font-size: 2rem;
  text-align: center;
}

.dialog-message {
  font-size: 1rem;
  color: #1f2937;
  margin: 0;
}

.dialog-action {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 4px 0 0;
}

.dialog-buttons {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.btn-secondary {
  padding: 8px 16px;
  border: 1px solid #d1d5db;
  background: white;
  border-radius: 6px;
  cursor: pointer;
}

.btn-primary {
  padding: 8px 16px;
  border: none;
  background: #2563eb;
  color: white;
  border-radius: 6px;
  cursor: pointer;
}
```

---

## Tool Execution Approval

For sensitive server-side tools, require user approval before execution:

### Server Configuration

```typescript
// app/api/chat/route.ts
import { streamText, tool } from 'ai';
import { z } from 'zod';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    messages,
    tools: {
      deleteFile: tool({
        description: 'Delete a file from the system',
        inputSchema: z.object({
          filename: z.string(),
          permanent: z.boolean(),
        }),
        needsApproval: true,  // Require user approval
        execute: async ({ filename, permanent }) => {
          await deleteFile(filename, permanent);
          return { success: true, deleted: filename };
        },
      }),

      // Dynamic approval based on input
      transferMoney: tool({
        description: 'Transfer money between accounts',
        inputSchema: z.object({
          amount: z.number(),
          toAccount: z.string(),
        }),
        // Only need approval for large amounts
        needsApproval: ({ amount }) => amount > 1000,
        execute: async ({ amount, toAccount }) => {
          return await processTransfer(amount, toAccount);
        },
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
```

### Client Approval UI

```tsx
import { useChat } from '@ai-sdk/react';

export default function Chat() {
  const { messages, addToolApprovalResponse } = useChat();

  return (
    <div>
      {messages.map((message) => (
        <div key={message.id}>
          {message.parts.map((part, i) => {
            if (part.type === 'tool-deleteFile') {
              if (part.state === 'approval-requested') {
                return (
                  <ApprovalRequest
                    key={i}
                    title="Delete File"
                    details={`Delete "${part.input.filename}"${
                      part.input.permanent ? ' permanently' : ''
                    }?`}
                    onApprove={() => {
                      addToolApprovalResponse({
                        id: part.approval.id,
                        approved: true,
                      });
                    }}
                    onDeny={() => {
                      addToolApprovalResponse({
                        id: part.approval.id,
                        approved: false,
                      });
                    }}
                  />
                );
              }

              if (part.state === 'output-available') {
                return (
                  <div key={i} className="success">
                    ✓ Deleted {part.output.deleted}
                  </div>
                );
              }
            }

            // ... handle other parts
          })}
        </div>
      ))}
    </div>
  );
}
```

### Approval Request Component

```tsx
interface ApprovalRequestProps {
  title: string;
  details: string;
  onApprove: () => void;
  onDeny: () => void;
}

function ApprovalRequest({ 
  title, 
  details, 
  onApprove, 
  onDeny 
}: ApprovalRequestProps) {
  return (
    <div className="approval-request">
      <div className="approval-header">
        <span className="approval-icon">🔐</span>
        <span className="approval-title">{title}</span>
      </div>
      
      <p className="approval-details">{details}</p>
      
      <div className="approval-warning">
        This action requires your approval before proceeding.
      </div>
      
      <div className="approval-actions">
        <button onClick={onDeny} className="btn-deny">
          Deny
        </button>
        <button onClick={onApprove} className="btn-approve">
          Approve
        </button>
      </div>
    </div>
  );
}
```

```css
.approval-request {
  border: 2px solid #dc2626;
  background: #fef2f2;
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
}

.approval-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #991b1b;
}

.approval-details {
  margin: 12px 0;
  color: #1f2937;
}

.approval-warning {
  font-size: 0.875rem;
  color: #b91c1c;
  padding: 8px 12px;
  background: #fee2e2;
  border-radius: 6px;
  margin-bottom: 12px;
}

.approval-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.btn-deny {
  padding: 8px 16px;
  border: 1px solid #d1d5db;
  background: white;
  border-radius: 6px;
  cursor: pointer;
}

.btn-approve {
  padding: 8px 16px;
  border: none;
  background: #dc2626;
  color: white;
  border-radius: 6px;
  cursor: pointer;
}
```

---

## Automatic Submission

Use `sendAutomaticallyWhen` to continue the conversation after tool results:

### After Tool Calls Complete

```tsx
import { useChat } from '@ai-sdk/react';
import { 
  DefaultChatTransport, 
  lastAssistantMessageIsCompleteWithToolCalls 
} from 'ai';

export default function Chat() {
  const { messages, sendMessage, addToolOutput } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),

    // Auto-submit when all tool results are available
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,

    async onToolCall({ toolCall }) {
      if (toolCall.toolName === 'getLocation') {
        // ... handle tool
        addToolOutput({ /* ... */ });
        // Chat will auto-submit after this
      }
    },
  });

  // ...
}
```

### After Approvals

```tsx
import { lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';

const { messages, addToolApprovalResponse } = useChat({
  // Auto-submit after all approvals are handled
  sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
});
```

### Custom Conditions

```tsx
const { messages, sendMessage } = useChat({
  sendAutomaticallyWhen: (messages) => {
    const lastMessage = messages[messages.length - 1];
    
    // Custom logic
    if (lastMessage?.role !== 'assistant') return false;
    
    const hasUnresolvedTools = lastMessage.parts.some(
      part => part.type.startsWith('tool-') && 
              part.state === 'input-available'
    );
    
    return !hasUnresolvedTools;
  },
});
```

---

## Event Callbacks from Components

Generated components can trigger new AI interactions:

### Callback Pattern

```tsx
function ProductCard({ 
  product, 
  onAskQuestion 
}: { 
  product: any; 
  onAskQuestion: (question: string) => void;
}) {
  const [question, setQuestion] = useState('');

  return (
    <div className="product-card">
      <img src={product.image} alt={product.name} />
      <h3>{product.name}</h3>
      <p>{product.price}</p>
      
      <div className="product-questions">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about this product..."
        />
        <button onClick={() => {
          onAskQuestion(`About ${product.name}: ${question}`);
          setQuestion('');
        }}>
          Ask
        </button>
      </div>
      
      <div className="quick-questions">
        <button onClick={() => onAskQuestion(`What are the specs of ${product.name}?`)}>
          Specs
        </button>
        <button onClick={() => onAskQuestion(`Is ${product.name} in stock?`)}>
          Stock
        </button>
      </div>
    </div>
  );
}
```

### Using in Chat

```tsx
export default function Chat() {
  const { messages, sendMessage } = useChat();

  const handleAskQuestion = (question: string) => {
    sendMessage({ text: question });
  };

  return (
    <div>
      {messages.map((message) => (
        <div key={message.id}>
          {message.parts.map((part, i) => {
            if (part.type === 'tool-showProduct' && part.state === 'output-available') {
              return (
                <ProductCard
                  key={i}
                  product={part.output}
                  onAskQuestion={handleAskQuestion}
                />
              );
            }
            // ... other parts
          })}
        </div>
      ))}
    </div>
  );
}
```

---

## Component State Management

Generated components can have their own state:

### Stateful Interactive Component

```tsx
function InteractiveChart({ 
  data, 
  onRangeChange 
}: { 
  data: any[]; 
  onRangeChange: (range: string) => void;
}) {
  const [selectedRange, setSelectedRange] = useState('1M');
  const [hoveredPoint, setHoveredPoint] = useState<any>(null);

  const handleRangeChange = (range: string) => {
    setSelectedRange(range);
    onRangeChange(range);
  };

  return (
    <div className="interactive-chart">
      <div className="chart-controls">
        {['1D', '1W', '1M', '3M', '1Y'].map((range) => (
          <button
            key={range}
            className={selectedRange === range ? 'active' : ''}
            onClick={() => handleRangeChange(range)}
          >
            {range}
          </button>
        ))}
      </div>

      <div className="chart-area">
        <Chart 
          data={data} 
          onHover={setHoveredPoint}
        />
      </div>

      {hoveredPoint && (
        <div className="chart-tooltip">
          <strong>{hoveredPoint.date}</strong>
          <span>${hoveredPoint.value}</span>
        </div>
      )}
    </div>
  );
}
```

### Persisting Component State

For components that should remember state across re-renders:

```tsx
// Use message ID as state key
function StatefulToolPart({ 
  part, 
  messageId 
}: { 
  part: any; 
  messageId: string;
}) {
  // Key state by message + tool call ID
  const stateKey = `${messageId}-${part.toolCallId}`;
  
  const [localState, setLocalState] = useLocalStorage(stateKey, {
    expanded: true,
    selectedTab: 0,
  });

  return (
    <div className={localState.expanded ? 'expanded' : 'collapsed'}>
      {/* Component with persisted state */}
    </div>
  );
}
```

---

## Handling Tool Errors

Provide good UX for tool failures:

```tsx
function ToolErrorHandler({ 
  part, 
  onRetry 
}: { 
  part: any; 
  onRetry: () => void;
}) {
  if (part.state !== 'output-error') return null;

  return (
    <div className="tool-error">
      <div className="error-icon">⚠️</div>
      <div className="error-content">
        <h4>Something went wrong</h4>
        <p>{part.errorText}</p>
      </div>
      <button onClick={onRetry} className="retry-button">
        Try Again
      </button>
    </div>
  );
}
```

```css
.tool-error {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  margin: 8px 0;
}

.error-icon {
  font-size: 1.5rem;
}

.error-content h4 {
  margin: 0 0 4px;
  color: #991b1b;
}

.error-content p {
  margin: 0;
  color: #7f1d1d;
  font-size: 0.875rem;
}

.retry-button {
  margin-left: auto;
  padding: 8px 16px;
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
```

---

## Summary

✅ Client-side tools execute in browser using `onToolCall`

✅ User interaction tools render UI and call `addToolOutput` on user action

✅ `needsApproval` requires explicit user approval before execution

✅ `sendAutomaticallyWhen` continues conversation after tool results

✅ Generated components can trigger new AI interactions via callbacks

✅ Component state can be managed locally or persisted

**Next:** [Generative UI Use Cases](./05-generative-ui-use-cases.md)

---

## Further Reading

- [Chatbot Tool Usage](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) — Complete tool handling
- [Tool Execution Approval](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling#tool-execution-approval) — Approval patterns
- [Client-Side Tools](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage#example) — Browser API access

---

<!-- 
Sources Consulted:
- AI SDK Chatbot Tool Usage: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage
- Tool Execution Approval: https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling
-->
