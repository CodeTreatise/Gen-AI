---
title: "Streaming Reasoning Display"
---

# Streaming Reasoning Display

## Introduction

Reasoning models often take 10-60+ seconds to think through complex problems. During this time, users need visual feedback that the AI is working. Streaming the thinking process in real-time creates engagement and transparency.

This lesson covers server-side configuration for streaming reasoning tokens, client-side rendering of live thinking, and animated indicators that maintain user engagement during long processing times.

### What We'll Cover

- Server-side `sendReasoning` configuration
- Real-time thinking visualization
- Animated thinking indicators
- Partial reasoning updates
- Completion detection
- Provider-specific streaming patterns

### Prerequisites

- [Thinking Section UI Patterns](./02-thinking-section-ui-patterns.md)
- AI SDK streaming basics
- Server-sent events (SSE) understanding

---

## Server-Side Configuration

### Enabling Reasoning Streaming

To stream reasoning tokens, configure your API route with `sendReasoning: true`:

```typescript
// app/api/chat/route.ts
import { streamText, convertToModelMessages, UIMessage } from 'ai';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();
  
  const result = streamText({
    model: 'deepseek/deepseek-r1', // or other reasoning model
    messages: await convertToModelMessages(messages),
  });
  
  return result.toUIMessageStreamResponse({
    sendReasoning: true, // Stream reasoning tokens to client
  });
}
```

### Model-Specific Configuration

Different providers have different reasoning APIs:

```typescript
// OpenAI o-series
import { openai } from '@ai-sdk/openai';

const result = streamText({
  model: openai('o4-mini'),
  messages: await convertToModelMessages(messages),
  experimental_reasoning: {
    effort: 'medium', // 'low' | 'medium' | 'high'
    summary: 'auto',  // Include reasoning summary
  },
});

return result.toUIMessageStreamResponse({
  sendReasoning: true,
});
```

```typescript
// Anthropic Claude with extended thinking
import { anthropic } from '@ai-sdk/anthropic';

const result = streamText({
  model: anthropic('claude-sonnet-4-5'),
  messages: await convertToModelMessages(messages),
  experimental_thinking: {
    type: 'enabled',
    budgetTokens: 10000, // Max thinking tokens
  },
});

return result.toUIMessageStreamResponse({
  sendReasoning: true,
});
```

```typescript
// DeepSeek R1
import { deepseek } from '@ai-sdk/deepseek';

const result = streamText({
  model: deepseek('deepseek-r1'),
  messages: await convertToModelMessages(messages),
});

return result.toUIMessageStreamResponse({
  sendReasoning: true, // Full reasoning tokens exposed
});
```

---

## Client-Side Streaming Display

### Basic Streaming Reasoning

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

export function StreamingReasoningChat() {
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),
  });
  
  const isThinking = status === 'streaming';
  
  return (
    <div className="chat">
      {messages.map(message => (
        <StreamingMessage 
          key={message.id} 
          message={message}
          isStreaming={isThinking && message === messages[messages.length - 1]}
        />
      ))}
    </div>
  );
}
```

### Streaming Message Component

Render parts as they arrive:

```tsx
import type { UIMessage } from 'ai';

interface StreamingMessageProps {
  message: UIMessage;
  isStreaming: boolean;
}

export function StreamingMessage({ message, isStreaming }: StreamingMessageProps) {
  // Separate thinking and response parts
  const reasoningParts = message.parts.filter(p => p.type === 'reasoning');
  const textParts = message.parts.filter(p => p.type === 'text');
  
  const hasReasoning = reasoningParts.length > 0;
  const hasText = textParts.length > 0;
  
  // Determine current streaming phase
  const isStreamingThinking = isStreaming && hasReasoning && !hasText;
  const isStreamingResponse = isStreaming && hasText;
  
  return (
    <div className={`message ${message.role}`}>
      {/* Thinking Section */}
      {hasReasoning && (
        <div className={`thinking-section ${isStreamingThinking ? 'streaming' : ''}`}>
          <div className="thinking-header">
            <span className="icon">💭</span>
            <span>Thinking</span>
            {isStreamingThinking && <StreamingDots />}
          </div>
          <div className="thinking-content">
            {reasoningParts.map((part, i) => (
              <pre key={i}>{part.text}</pre>
            ))}
            {isStreamingThinking && <BlinkingCursor />}
          </div>
        </div>
      )}
      
      {/* Response Section */}
      {hasText && (
        <div className={`response-section ${isStreamingResponse ? 'streaming' : ''}`}>
          {textParts.map((part, i) => (
            <div key={i}>{part.text}</div>
          ))}
          {isStreamingResponse && <BlinkingCursor />}
        </div>
      )}
      
      {/* Initial thinking indicator */}
      {isStreaming && !hasReasoning && !hasText && (
        <ThinkingIndicator />
      )}
    </div>
  );
}
```

---

## Animated Thinking Indicators

### Streaming Dots

```tsx
export function StreamingDots() {
  return (
    <span className="streaming-dots">
      <span className="dot">.</span>
      <span className="dot">.</span>
      <span className="dot">.</span>
    </span>
  );
}
```

```css
.streaming-dots {
  display: inline-flex;
  gap: 2px;
  margin-left: 8px;
}

.streaming-dots .dot {
  animation: bounce 1.4s infinite ease-in-out;
  color: #8b5cf6;
  font-weight: bold;
}

.streaming-dots .dot:nth-child(1) { animation-delay: 0s; }
.streaming-dots .dot:nth-child(2) { animation-delay: 0.2s; }
.streaming-dots .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% {
    transform: translateY(0);
    opacity: 0.5;
  }
  40% {
    transform: translateY(-4px);
    opacity: 1;
  }
}
```

### Blinking Cursor

```tsx
export function BlinkingCursor() {
  return <span className="blinking-cursor">▋</span>;
}
```

```css
.blinking-cursor {
  display: inline-block;
  animation: blink 1s infinite;
  color: #8b5cf6;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
```

### Thinking Spinner

```tsx
export function ThinkingIndicator() {
  return (
    <div className="thinking-indicator">
      <div className="spinner" />
      <span>AI is thinking...</span>
    </div>
  );
}
```

```css
.thinking-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
  border-radius: 12px;
  color: #7c3aed;
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e9d5ff;
  border-top-color: #8b5cf6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Brain Animation

A more elaborate thinking indicator:

```tsx
export function BrainThinking() {
  return (
    <div className="brain-thinking">
      <div className="brain-icon">🧠</div>
      <div className="brain-waves">
        <span className="wave" />
        <span className="wave" />
        <span className="wave" />
      </div>
      <span className="brain-text">Processing...</span>
    </div>
  );
}
```

```css
.brain-thinking {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  background: #fef3c7;
  border-radius: 9999px;
}

.brain-icon {
  font-size: 1.5rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.brain-waves {
  display: flex;
  gap: 3px;
}

.wave {
  width: 3px;
  height: 12px;
  background: #f59e0b;
  border-radius: 2px;
  animation: wave 1s infinite ease-in-out;
}

.wave:nth-child(1) { animation-delay: 0s; }
.wave:nth-child(2) { animation-delay: 0.15s; }
.wave:nth-child(3) { animation-delay: 0.3s; }

@keyframes wave {
  0%, 100% { height: 6px; }
  50% { height: 16px; }
}

.brain-text {
  color: #92400e;
  font-size: 0.875rem;
  font-weight: 500;
}
```

---

## Partial Reasoning Updates

### Streaming Text Effect

Create a typewriter effect for reasoning:

```tsx
import { useState, useEffect } from 'react';

interface TypewriterTextProps {
  text: string;
  speed?: number;
}

export function TypewriterText({ text, speed = 20 }: TypewriterTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [index, setIndex] = useState(0);
  
  useEffect(() => {
    if (index < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(prev => prev + text[index]);
        setIndex(prev => prev + 1);
      }, speed);
      
      return () => clearTimeout(timer);
    }
  }, [index, text, speed]);
  
  // Reset when text changes significantly
  useEffect(() => {
    if (text.length < displayedText.length) {
      setDisplayedText('');
      setIndex(0);
    }
  }, [text]);
  
  return (
    <span className="typewriter">
      {displayedText}
      {index < text.length && <BlinkingCursor />}
    </span>
  );
}
```

### Progressive Line Reveal

Show reasoning line by line:

```tsx
import { useState, useEffect, useMemo } from 'react';

interface ProgressiveReasoningProps {
  text: string;
  isStreaming: boolean;
}

export function ProgressiveReasoning({ text, isStreaming }: ProgressiveReasoningProps) {
  const lines = useMemo(() => text.split('\n'), [text]);
  const [visibleLines, setVisibleLines] = useState(0);
  
  useEffect(() => {
    if (isStreaming) {
      // Show all lines as they arrive during streaming
      setVisibleLines(lines.length);
    }
  }, [lines.length, isStreaming]);
  
  return (
    <div className="progressive-reasoning">
      {lines.slice(0, visibleLines).map((line, i) => (
        <div 
          key={i} 
          className="reasoning-line"
          style={{ 
            animationDelay: `${i * 50}ms`,
            opacity: i === visibleLines - 1 && isStreaming ? 0.7 : 1 
          }}
        >
          {line || '\u00A0'} {/* Non-breaking space for empty lines */}
        </div>
      ))}
      {isStreaming && <BlinkingCursor />}
    </div>
  );
}
```

```css
.progressive-reasoning {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
}

.reasoning-line {
  animation: fadeSlide 0.2s ease-out forwards;
  opacity: 0;
}

@keyframes fadeSlide {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

---

## Completion Detection

### Tracking Thinking vs Response Phases

```tsx
import { useChat } from '@ai-sdk/react';
import { useState, useEffect, useRef } from 'react';

type Phase = 'idle' | 'thinking' | 'responding' | 'complete';

export function PhaseAwareChat() {
  const { messages, status } = useChat();
  const [phase, setPhase] = useState<Phase>('idle');
  const lastMessageRef = useRef<string | null>(null);
  
  useEffect(() => {
    if (status === 'ready') {
      setPhase('idle');
      return;
    }
    
    if (status !== 'streaming') return;
    
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage || lastMessage.role !== 'assistant') return;
    
    const hasReasoning = lastMessage.parts.some(p => p.type === 'reasoning');
    const hasText = lastMessage.parts.some(p => p.type === 'text');
    
    if (hasReasoning && !hasText) {
      setPhase('thinking');
    } else if (hasText) {
      setPhase('responding');
    }
  }, [messages, status]);
  
  return (
    <div className="chat">
      <PhaseIndicator phase={phase} />
      {/* Messages */}
    </div>
  );
}

function PhaseIndicator({ phase }: { phase: Phase }) {
  if (phase === 'idle') return null;
  
  return (
    <div className={`phase-indicator ${phase}`}>
      {phase === 'thinking' && (
        <>
          <span className="icon">💭</span>
          <span>Thinking through the problem...</span>
        </>
      )}
      {phase === 'responding' && (
        <>
          <span className="icon">✍️</span>
          <span>Writing response...</span>
        </>
      )}
    </div>
  );
}
```

```css
.phase-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 0.875rem;
  transition: all 0.3s;
}

.phase-indicator.thinking {
  background: #faf5ff;
  color: #7c3aed;
}

.phase-indicator.responding {
  background: #ecfdf5;
  color: #059669;
}
```

### Transition Animation Between Phases

```tsx
export function PhaseTransition({ 
  from, 
  to 
}: { 
  from: 'thinking' | null;
  to: 'responding';
}) {
  return (
    <div className="phase-transition">
      <div className="transition-line" />
      <span className="transition-label">
        ✓ Thinking complete → Now responding
      </span>
    </div>
  );
}
```

```css
.phase-transition {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 12px 0;
  animation: fadeIn 0.3s ease;
}

.transition-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, #e9d5ff, #d1fae5);
}

.transition-label {
  font-size: 0.75rem;
  color: #9ca3af;
  white-space: nowrap;
}
```

---

## Full Streaming Example

### Complete Implementation

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState, useEffect, useRef } from 'react';
import type { UIMessage } from 'ai';

export function StreamingReasoningApp() {
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: '/api/chat' }),
  });
  
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="app">
      <div className="messages">
        {messages.map((message, index) => (
          <Message
            key={message.id}
            message={message}
            isLatest={index === messages.length - 1}
            isStreaming={status === 'streaming'}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={(e) => {
        e.preventDefault();
        if (input.trim() && status === 'ready') {
          sendMessage({ text: input });
          setInput('');
        }
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something complex..."
          disabled={status !== 'ready'}
        />
        <button type="submit" disabled={status !== 'ready'}>
          Send
        </button>
      </form>
    </div>
  );
}

interface MessageProps {
  message: UIMessage;
  isLatest: boolean;
  isStreaming: boolean;
}

function Message({ message, isLatest, isStreaming }: MessageProps) {
  const isCurrentlyStreaming = isLatest && isStreaming;
  
  if (message.role === 'user') {
    return (
      <div className="message user">
        {message.parts.map((part, i) => (
          part.type === 'text' ? <p key={i}>{part.text}</p> : null
        ))}
      </div>
    );
  }
  
  // Assistant message
  const reasoningParts = message.parts.filter(p => p.type === 'reasoning');
  const textParts = message.parts.filter(p => p.type === 'text');
  
  const isThinking = isCurrentlyStreaming && reasoningParts.length > 0 && textParts.length === 0;
  const isResponding = isCurrentlyStreaming && textParts.length > 0;
  const isWaiting = isCurrentlyStreaming && reasoningParts.length === 0 && textParts.length === 0;
  
  return (
    <div className="message assistant">
      {/* Waiting state */}
      {isWaiting && <ThinkingIndicator />}
      
      {/* Reasoning section */}
      {reasoningParts.length > 0 && (
        <details className="reasoning-block" open={isThinking}>
          <summary>
            💭 Thinking
            {isThinking && <StreamingDots />}
            {!isThinking && (
              <span className="complete-badge">✓</span>
            )}
          </summary>
          <div className="reasoning-content">
            {reasoningParts.map((part, i) => (
              <pre key={i}>{part.text}</pre>
            ))}
            {isThinking && <BlinkingCursor />}
          </div>
        </details>
      )}
      
      {/* Response section */}
      {textParts.length > 0 && (
        <div className="response-block">
          {textParts.map((part, i) => (
            <div key={i}>{part.text}</div>
          ))}
          {isResponding && <BlinkingCursor />}
        </div>
      )}
    </div>
  );
}
```

---

## Performance Considerations

### Throttling Updates

Reduce render frequency for smooth performance:

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

const { messages } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
  experimental_throttle: 50, // Throttle to 50ms
});
```

### Virtualization for Long Reasoning

For very long reasoning chains:

```tsx
import { FixedSizeList as List } from 'react-window';

function VirtualizedReasoning({ lines }: { lines: string[] }) {
  return (
    <List
      height={300}
      itemCount={lines.length}
      itemSize={24}
      width="100%"
    >
      {({ index, style }) => (
        <div style={style} className="reasoning-line">
          {lines[index]}
        </div>
      )}
    </List>
  );
}
```

---

## Summary

✅ Enable reasoning streaming with `sendReasoning: true` in `toUIMessageStreamResponse`

✅ Different providers have different streaming patterns (OpenAI summaries, Claude thinking blocks, DeepSeek full tokens)

✅ Use animated indicators (spinners, dots, cursors) during streaming

✅ Track thinking vs responding phases for appropriate UI feedback

✅ Throttle updates with `experimental_throttle` for smooth performance

✅ Auto-close thinking sections when response begins

**Next:** [Tool Invocation Display](./04-tool-invocation-display.md)

---

## Further Reading

- [AI SDK Streaming](https://ai-sdk.dev/docs/foundations/streaming) — Core streaming concepts
- [AI SDK Chatbot - Throttling](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot#throttling-ui-updates) — Performance optimization
- [OpenAI Streaming Responses](https://platform.openai.com/docs/guides/streaming-responses) — OpenAI streaming guide

---

<!-- 
Sources Consulted:
- AI SDK Chatbot: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- AI SDK Streaming: https://ai-sdk.dev/docs/foundations/streaming
- OpenAI Reasoning: https://platform.openai.com/docs/guides/reasoning
- Claude Extended Thinking: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
-->
