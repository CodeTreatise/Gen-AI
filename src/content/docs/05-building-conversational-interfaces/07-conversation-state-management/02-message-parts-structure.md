---
title: "Message Parts Structure"
---

# Message Parts Structure

## Introduction

Modern AI responses are no longer just text. They include tool invocations, reasoning steps, source citations, and file references. The message parts structure (introduced in AI SDK 4.0 and adopted widely in 2024-2025) provides a type-safe way to handle this complexity.

In this lesson, we'll implement the parts array pattern for rich message content.

### What We'll Cover

- The `message.parts` array format
- Part types: text, tool-invocation, tool-result
- Reasoning parts for thinking models
- Source and citation parts
- Type-safe part rendering

### Prerequisites

- [Message Data Structures](./01-message-data-structures.md)
- TypeScript generics
- Understanding of AI tool/function calling

---

## Why Message Parts?

### The Problem with String Content

```typescript
// Old approach - everything is a string
interface OldMessage {
  role: 'assistant';
  content: string;  // "Let me search for that... [tool call] Found: ..."
}

// Problems:
// 1. Can't distinguish tool calls from text
// 2. No structured data for tool results
// 3. Reasoning is mixed with response
// 4. Citations are embedded in prose
```

### The Parts Solution

```typescript
// Modern approach - structured parts
interface ModernMessage {
  role: 'assistant';
  content: string;  // Still here for backwards compatibility
  parts: MessagePart[];  // Structured content
}

// Each part has a specific type and structure
type MessagePart = 
  | TextPart
  | ToolInvocationPart
  | ToolResultPart
  | ReasoningPart
  | SourcePart
  | FilePart;
```

---

## Part Type Definitions

### Text Part

```typescript
interface TextPart {
  type: 'text';
  text: string;
}

// Example
const textPart: TextPart = {
  type: 'text',
  text: 'Based on the search results, here are the top 3 options...'
};
```

### Tool Invocation Part

```typescript
interface ToolInvocationPart {
  type: 'tool-invocation';
  toolInvocationId: string;
  toolName: string;
  args: Record<string, unknown>;
  state: 'pending' | 'running' | 'complete' | 'error';
}

// Example
const toolInvocation: ToolInvocationPart = {
  type: 'tool-invocation',
  toolInvocationId: 'call_abc123',
  toolName: 'web_search',
  args: { query: 'best restaurants in Austin' },
  state: 'running'
};
```

### Tool Result Part

```typescript
interface ToolResultPart {
  type: 'tool-result';
  toolInvocationId: string;
  toolName: string;
  result: unknown;
  isError?: boolean;
}

// Example
const toolResult: ToolResultPart = {
  type: 'tool-result',
  toolInvocationId: 'call_abc123',
  toolName: 'web_search',
  result: {
    results: [
      { title: 'Top 10 Austin Restaurants', url: '...' },
      { title: 'Best BBQ in Texas', url: '...' }
    ]
  }
};
```

### Reasoning Part

```typescript
interface ReasoningPart {
  type: 'reasoning';
  reasoning: string;
  // For redacted reasoning (some models)
  redactedReasoning?: string;
}

// Example (from o1/o3 models)
const reasoningPart: ReasoningPart = {
  type: 'reasoning',
  reasoning: `Let me think about this step by step:
1. The user wants restaurant recommendations
2. I should consider cuisine variety
3. Location accessibility matters
4. Price range wasn't specified, include variety`
};
```

### Source Parts

```typescript
interface SourceUrlPart {
  type: 'source-url';
  sourceId: string;
  url: string;
  title?: string;
  snippet?: string;
}

interface SourceDocumentPart {
  type: 'source-document';
  sourceId: string;
  documentId: string;
  documentName: string;
  pageNumber?: number;
  snippet?: string;
}

type SourcePart = SourceUrlPart | SourceDocumentPart;

// Example
const sourcePart: SourceUrlPart = {
  type: 'source-url',
  sourceId: 'src_1',
  url: 'https://austin.eater.com/best-restaurants',
  title: 'Austin Eater Guide 2025',
  snippet: 'The definitive guide to dining in Austin...'
};
```

### File Part

```typescript
interface FilePart {
  type: 'file';
  fileId: string;
  fileName: string;
  mimeType: string;
  url?: string;
  size?: number;
}

// Example
const filePart: FilePart = {
  type: 'file',
  fileId: 'file_xyz789',
  fileName: 'report.pdf',
  mimeType: 'application/pdf',
  url: '/api/files/file_xyz789',
  size: 245000
};
```

---

## Complete Type Definition

```typescript
// === Message Parts Types ===

export type MessagePart =
  | TextPart
  | ToolInvocationPart
  | ToolResultPart
  | ReasoningPart
  | SourceUrlPart
  | SourceDocumentPart
  | FilePart;

export interface TextPart {
  type: 'text';
  text: string;
}

export interface ToolInvocationPart {
  type: 'tool-invocation';
  toolInvocationId: string;
  toolName: string;
  args: Record<string, unknown>;
  state: 'pending' | 'running' | 'complete' | 'error';
}

export interface ToolResultPart {
  type: 'tool-result';
  toolInvocationId: string;
  toolName: string;
  result: unknown;
  isError?: boolean;
}

export interface ReasoningPart {
  type: 'reasoning';
  reasoning: string;
  redactedReasoning?: string;
}

export interface SourceUrlPart {
  type: 'source-url';
  sourceId: string;
  url: string;
  title?: string;
  snippet?: string;
}

export interface SourceDocumentPart {
  type: 'source-document';
  sourceId: string;
  documentId: string;
  documentName: string;
  pageNumber?: number;
  snippet?: string;
}

export interface FilePart {
  type: 'file';
  fileId: string;
  fileName: string;
  mimeType: string;
  url?: string;
  size?: number;
}

// === Extended Message Interface ===

export interface MessageWithParts {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;          // Backwards-compatible text
  parts: MessagePart[];     // Structured content
  createdAt: Date;
  status: MessageStatus;
}
```

---

## Type Guards

```typescript
// Type guard functions for safe narrowing

export function isTextPart(part: MessagePart): part is TextPart {
  return part.type === 'text';
}

export function isToolInvocationPart(part: MessagePart): part is ToolInvocationPart {
  return part.type === 'tool-invocation';
}

export function isToolResultPart(part: MessagePart): part is ToolResultPart {
  return part.type === 'tool-result';
}

export function isReasoningPart(part: MessagePart): part is ReasoningPart {
  return part.type === 'reasoning';
}

export function isSourcePart(part: MessagePart): part is SourceUrlPart | SourceDocumentPart {
  return part.type === 'source-url' || part.type === 'source-document';
}

export function isFilePart(part: MessagePart): part is FilePart {
  return part.type === 'file';
}

// Extract parts by type
export function getTextParts(parts: MessagePart[]): TextPart[] {
  return parts.filter(isTextPart);
}

export function getToolCalls(parts: MessagePart[]): ToolInvocationPart[] {
  return parts.filter(isToolInvocationPart);
}

export function getSources(parts: MessagePart[]): (SourceUrlPart | SourceDocumentPart)[] {
  return parts.filter(isSourcePart);
}
```

---

## Rendering Parts

### Part Renderer Component

```tsx
import { type MessagePart } from './types';

interface PartRendererProps {
  part: MessagePart;
}

export function PartRenderer({ part }: PartRendererProps) {
  switch (part.type) {
    case 'text':
      return <TextPartView text={part.text} />;
      
    case 'tool-invocation':
      return (
        <ToolInvocationView
          name={part.toolName}
          args={part.args}
          state={part.state}
        />
      );
      
    case 'tool-result':
      return (
        <ToolResultView
          name={part.toolName}
          result={part.result}
          isError={part.isError}
        />
      );
      
    case 'reasoning':
      return <ReasoningView reasoning={part.reasoning} />;
      
    case 'source-url':
      return (
        <SourceUrlView
          url={part.url}
          title={part.title}
          snippet={part.snippet}
        />
      );
      
    case 'source-document':
      return (
        <SourceDocumentView
          name={part.documentName}
          page={part.pageNumber}
          snippet={part.snippet}
        />
      );
      
    case 'file':
      return (
        <FilePartView
          name={part.fileName}
          mimeType={part.mimeType}
          url={part.url}
        />
      );
      
    default:
      // Exhaustive check
      const _exhaustive: never = part;
      return null;
  }
}
```

### Message with Parts Renderer

```tsx
function MessageWithPartsRenderer({ message }: { message: MessageWithParts }) {
  // Group parts by type for organized rendering
  const textParts = message.parts.filter(isTextPart);
  const toolParts = message.parts.filter(
    p => isToolInvocationPart(p) || isToolResultPart(p)
  );
  const sources = message.parts.filter(isSourcePart);
  const reasoning = message.parts.filter(isReasoningPart);
  
  return (
    <div className={`message ${message.role}`}>
      {/* Show reasoning first (collapsible) */}
      {reasoning.length > 0 && (
        <ReasoningSection parts={reasoning} />
      )}
      
      {/* Tool calls inline */}
      {toolParts.map((part, i) => (
        <PartRenderer key={i} part={part} />
      ))}
      
      {/* Main text content */}
      {textParts.map((part, i) => (
        <PartRenderer key={i} part={part} />
      ))}
      
      {/* Sources at the bottom */}
      {sources.length > 0 && (
        <SourcesSection sources={sources} />
      )}
    </div>
  );
}
```

---

## Building Parts from Streams

### Streaming Part Accumulator

```typescript
class StreamingPartsBuilder {
  private parts: MessagePart[] = [];
  private currentTextBuffer = '';
  
  addTextDelta(delta: string): void {
    this.currentTextBuffer += delta;
  }
  
  startToolCall(id: string, name: string): void {
    // Flush text buffer first
    this.flushText();
    
    this.parts.push({
      type: 'tool-invocation',
      toolInvocationId: id,
      toolName: name,
      args: {},
      state: 'pending'
    });
  }
  
  updateToolArgs(id: string, argsJson: string): void {
    const part = this.parts.find(
      p => p.type === 'tool-invocation' && p.toolInvocationId === id
    ) as ToolInvocationPart | undefined;
    
    if (part) {
      try {
        part.args = JSON.parse(argsJson);
        part.state = 'running';
      } catch {
        // Partial JSON, wait for more
      }
    }
  }
  
  addToolResult(id: string, result: unknown, isError = false): void {
    // Mark invocation as complete
    const invocation = this.parts.find(
      p => p.type === 'tool-invocation' && p.toolInvocationId === id
    ) as ToolInvocationPart | undefined;
    
    if (invocation) {
      invocation.state = isError ? 'error' : 'complete';
    }
    
    // Find the tool name from invocation
    const toolName = invocation?.toolName || 'unknown';
    
    this.parts.push({
      type: 'tool-result',
      toolInvocationId: id,
      toolName,
      result,
      isError
    });
  }
  
  addReasoning(reasoning: string): void {
    this.parts.push({
      type: 'reasoning',
      reasoning
    });
  }
  
  addSource(source: SourceUrlPart | SourceDocumentPart): void {
    this.parts.push(source);
  }
  
  private flushText(): void {
    if (this.currentTextBuffer.trim()) {
      this.parts.push({
        type: 'text',
        text: this.currentTextBuffer
      });
      this.currentTextBuffer = '';
    }
  }
  
  finalize(): MessagePart[] {
    this.flushText();
    return this.parts;
  }
  
  // Get current state (for UI updates during streaming)
  getCurrentParts(): MessagePart[] {
    const result = [...this.parts];
    
    if (this.currentTextBuffer) {
      result.push({
        type: 'text',
        text: this.currentTextBuffer
      });
    }
    
    return result;
  }
}
```

### Usage in Stream Handler

```typescript
async function handleStream(response: Response) {
  const builder = new StreamingPartsBuilder();
  
  for await (const event of parseSSE(response.body)) {
    switch (event.type) {
      case 'content_block_delta':
        builder.addTextDelta(event.delta.text);
        break;
        
      case 'tool_use':
        builder.startToolCall(event.id, event.name);
        break;
        
      case 'tool_result':
        builder.addToolResult(event.tool_use_id, event.content);
        break;
        
      case 'reasoning':
        builder.addReasoning(event.text);
        break;
    }
    
    // Update UI with current state
    updateMessage({
      parts: builder.getCurrentParts()
    });
  }
  
  // Final message
  return {
    parts: builder.finalize()
  };
}
```

---

## Converting Legacy Messages

```typescript
function convertLegacyMessage(legacy: LegacyMessage): MessageWithParts {
  const parts: MessagePart[] = [];
  
  // Convert string content to text part
  if (legacy.content) {
    parts.push({
      type: 'text',
      text: legacy.content
    });
  }
  
  // Convert tool_calls to tool-invocation parts
  if (legacy.tool_calls) {
    for (const call of legacy.tool_calls) {
      parts.push({
        type: 'tool-invocation',
        toolInvocationId: call.id,
        toolName: call.function.name,
        args: JSON.parse(call.function.arguments),
        state: 'complete'
      });
    }
  }
  
  return {
    id: legacy.id || generateId(),
    role: legacy.role,
    content: legacy.content || '',
    parts,
    createdAt: new Date(),
    status: 'complete'
  };
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use discriminated unions | Use generic object types |
| Provide type guards | Cast types unsafely |
| Keep `content` for backwards compat | Remove string content |
| Order parts logically for UI | Rely on array order |
| Handle unknown part types | Crash on new types |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting exhaustive check | Add `never` case in switch |
| Tool result without invocation | Track invocation ID mapping |
| Text split across parts | Use buffer and flush |
| Reasoning in main content | Separate into reasoning parts |
| Sources mixed with text | Extract to dedicated parts |

---

## Hands-on Exercise

### Your Task

Build a `MessagePartsParser` that:
1. Accepts raw AI SDK stream events
2. Accumulates content into proper parts
3. Returns a typed MessagePart array

### Requirements

1. Handle text deltas
2. Track tool invocation lifecycle
3. Associate tool results with invocations
4. Support reasoning content

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a class to maintain state
- Buffer text until a different event type
- Map tool IDs to invocation parts
- Flush text before adding non-text parts

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `StreamingPartsBuilder` class in the "Building Parts from Streams" section above.

</details>

---

## Summary

✅ **Message parts** separate content types  
✅ **Type guards** enable safe narrowing  
✅ **Part renderers** map types to components  
✅ **Streaming builders** accumulate parts live  
✅ **Legacy conversion** maintains compatibility  
✅ **Exhaustive checks** catch new part types

---

## Further Reading

- [AI SDK Message Parts](https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat#message-parts)
- [OpenAI Tool Calls](https://platform.openai.com/docs/guides/function-calling)
- [TypeScript Discriminated Unions](https://www.typescriptlang.org/docs/handbook/2/narrowing.html)

---

**Previous:** [Message Data Structures](./01-message-data-structures.md)  
**Next:** [Conversation History Storage](./03-conversation-history-storage.md)

<!-- 
Sources Consulted:
- AI SDK Message Parts: https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
-->
