---
title: "Agent UI/UX Patterns"
---

# Agent UI/UX Patterns

## Introduction

A powerful agent is useless if users can't understand what it's doing. Unlike traditional web applications where UI states are predictable, agent interfaces must communicate **uncertainty, progress through multi-step reasoning, tool execution, errors, and partial results** — all while keeping users engaged and in control.

This lesson covers the UI/UX patterns that make agent interactions feel transparent, trustworthy, and responsive. We build thinking indicators, tool call visualizations, error recovery UI, feedback mechanisms, and accessibility-first agent components.

### What we'll cover

- Thinking and processing indicators
- Tool call visualization
- Error handling and recovery UI
- User feedback and correction mechanisms
- Streaming content rendering patterns
- Accessibility for agent interfaces

### Prerequisites

- React component basics (hooks, state, props)
- CSS fundamentals (Unit 1, Lesson 2)
- Streaming agents to the browser (Lesson 17-02)
- Accessibility fundamentals (Unit 1, Lesson 13)

---

## Thinking indicators

Users need to know the agent is working — silence feels broken. The indicator style should match the agent's current activity.

### Multi-state thinking indicator

```tsx
type AgentState =
  | 'idle'
  | 'thinking'
  | 'calling-tool'
  | 'reading-results'
  | 'generating';

function ThinkingIndicator({ state }: { state: AgentState }) {
  if (state === 'idle') return null;

  const labels: Record<AgentState, string> = {
    idle: '',
    thinking: 'Thinking...',
    'calling-tool': 'Using a tool...',
    'reading-results': 'Analyzing results...',
    generating: 'Writing response...',
  };

  const icons: Record<AgentState, string> = {
    idle: '',
    thinking: '🧠',
    'calling-tool': '🔧',
    'reading-results': '📖',
    generating: '✍️',
  };

  return (
    <div className="thinking-indicator" role="status" aria-live="polite">
      <span className="thinking-icon">{icons[state]}</span>
      <span className="thinking-label">{labels[state]}</span>
      <span className="thinking-dots">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </span>
    </div>
  );
}
```

```css
.thinking-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: #f0f4f8;
  border-radius: 12px;
  font-size: 14px;
  color: #4a5568;
}

.thinking-dots {
  display: flex;
  gap: 4px;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #a0aec0;
  animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
.dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
```

### Elapsed time indicator

For operations that take more than a few seconds, show elapsed time:

```tsx
import { useState, useEffect } from 'react';

function ElapsedTimer({ isActive }: { isActive: boolean }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setElapsed(0);
      return;
    }

    const start = Date.now();
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, [isActive]);

  if (!isActive || elapsed < 3) return null;

  return (
    <span className="elapsed" aria-label={`${elapsed} seconds elapsed`}>
      {elapsed}s
    </span>
  );
}
```

---

## Tool call visualization

When agents use tools, users should see **what the agent is doing and why**. Transparent tool visualization builds trust.

### Tool call timeline

```tsx
interface ToolCallStep {
  id: string;
  toolName: string;
  input: Record<string, unknown>;
  output?: string;
  status: 'pending' | 'running' | 'success' | 'error';
  duration?: number;
}

function ToolTimeline({ steps }: { steps: ToolCallStep[] }) {
  return (
    <div className="tool-timeline" role="list" aria-label="Agent tool calls">
      {steps.map((step, index) => (
        <div
          key={step.id}
          className={`tool-step tool-step--${step.status}`}
          role="listitem"
        >
          <div className="tool-step__connector">
            {index < steps.length - 1 && <div className="connector-line" />}
          </div>

          <div className="tool-step__icon">
            {step.status === 'pending' && '⏳'}
            {step.status === 'running' && '⚙️'}
            {step.status === 'success' && '✅'}
            {step.status === 'error' && '❌'}
          </div>

          <div className="tool-step__content">
            <div className="tool-step__header">
              <strong>{formatToolName(step.toolName)}</strong>
              {step.duration && (
                <span className="tool-step__duration">
                  {step.duration}ms
                </span>
              )}
            </div>

            <div className="tool-step__input">
              <code>{JSON.stringify(step.input)}</code>
            </div>

            {step.output && (
              <div className="tool-step__output">
                <details>
                  <summary>Result</summary>
                  <pre>{step.output}</pre>
                </details>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function formatToolName(name: string): string {
  return name
    .replace(/([A-Z])/g, ' $1')
    .replace(/_/g, ' ')
    .replace(/^\w/, c => c.toUpperCase())
    .trim();
}
```

```css
.tool-timeline {
  padding: 8px 0;
}

.tool-step {
  display: flex;
  gap: 12px;
  padding: 8px 0;
  position: relative;
}

.tool-step__connector {
  width: 24px;
  display: flex;
  justify-content: center;
}

.connector-line {
  position: absolute;
  top: 36px;
  width: 2px;
  height: calc(100% - 12px);
  background: #e2e8f0;
}

.tool-step--running .tool-step__icon {
  animation: spin 1s linear infinite;
}

.tool-step__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tool-step__duration {
  font-size: 12px;
  color: #a0aec0;
}

.tool-step__input code {
  font-size: 12px;
  background: #f7fafc;
  padding: 2px 6px;
  border-radius: 4px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

### Collapsible tool details

For agents that make many tool calls, use progressive disclosure:

```tsx
function CompactToolCall({ step }: { step: ToolCallStep }) {
  const statusIcons = {
    pending: '⏳',
    running: '⚙️',
    success: '✅',
    error: '❌',
  };

  return (
    <details className="compact-tool">
      <summary>
        <span>{statusIcons[step.status]}</span>
        <span>{formatToolName(step.toolName)}</span>
        {step.duration && <span className="dim">({step.duration}ms)</span>}
      </summary>
      <div className="compact-tool__details">
        <div>
          <strong>Input:</strong>
          <pre>{JSON.stringify(step.input, null, 2)}</pre>
        </div>
        {step.output && (
          <div>
            <strong>Output:</strong>
            <pre>{step.output}</pre>
          </div>
        )}
      </div>
    </details>
  );
}
```

---

## Error handling and recovery UI

Agent errors differ from traditional app errors — they can be partial, recoverable, or require user clarification.

### Error categories and UI patterns

| Error Type | Cause | UI Pattern |
|-----------|-------|------------|
| **Network error** | Connection lost | Auto-retry with progress bar |
| **Rate limit** | Too many requests | Countdown timer, then auto-retry |
| **Model error** | LLM refused or failed | "Try rephrasing" suggestion |
| **Tool error** | External API failed | Show which tool failed, offer retry |
| **Timeout** | Agent took too long | "Still working?" with cancel option |
| **Content filter** | Response blocked | Explain limitation, suggest alternative |

### Error recovery component

```tsx
interface AgentError {
  type: 'network' | 'rate_limit' | 'model' | 'tool' | 'timeout' | 'content';
  message: string;
  retryable: boolean;
  retryAfter?: number; // seconds
  toolName?: string;
}

function AgentErrorUI({
  error,
  onRetry,
  onRephrase,
}: {
  error: AgentError;
  onRetry: () => void;
  onRephrase: () => void;
}) {
  const [countdown, setCountdown] = useState(error.retryAfter || 0);

  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  // Auto-retry after countdown
  useEffect(() => {
    if (countdown === 0 && error.type === 'rate_limit') {
      onRetry();
    }
  }, [countdown]);

  return (
    <div
      className={`agent-error agent-error--${error.type}`}
      role="alert"
    >
      <div className="agent-error__header">
        <span className="agent-error__icon">
          {error.type === 'network' && '🌐'}
          {error.type === 'rate_limit' && '⏱️'}
          {error.type === 'model' && '🤖'}
          {error.type === 'tool' && '🔧'}
          {error.type === 'timeout' && '⏳'}
          {error.type === 'content' && '🚫'}
        </span>
        <span className="agent-error__title">
          {error.type === 'network' && 'Connection Lost'}
          {error.type === 'rate_limit' && 'Rate Limited'}
          {error.type === 'model' && 'Agent Error'}
          {error.type === 'tool' && `Tool Failed: ${error.toolName}`}
          {error.type === 'timeout' && 'Request Timed Out'}
          {error.type === 'content' && 'Response Filtered'}
        </span>
      </div>

      <p className="agent-error__message">{error.message}</p>

      <div className="agent-error__actions">
        {error.retryable && countdown <= 0 && (
          <button onClick={onRetry} className="btn btn--primary">
            Try Again
          </button>
        )}

        {countdown > 0 && (
          <span className="agent-error__countdown">
            Retrying in {countdown}s...
          </span>
        )}

        {error.type === 'model' && (
          <button onClick={onRephrase} className="btn btn--secondary">
            Rephrase Question
          </button>
        )}
      </div>
    </div>
  );
}
```

---

## User feedback mechanisms

Collecting user feedback on agent responses improves quality over time and gives users a sense of control.

### Inline feedback component

```tsx
type FeedbackType = 'positive' | 'negative' | null;

function MessageFeedback({
  messageId,
  onFeedback,
}: {
  messageId: string;
  onFeedback: (id: string, type: FeedbackType, comment?: string) => void;
}) {
  const [feedback, setFeedback] = useState<FeedbackType>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');

  const handleFeedback = (type: FeedbackType) => {
    setFeedback(type);
    if (type === 'negative') {
      setShowComment(true);
    } else {
      onFeedback(messageId, type);
    }
  };

  return (
    <div className="message-feedback" aria-label="Rate this response">
      <div className="feedback-buttons">
        <button
          onClick={() => handleFeedback('positive')}
          className={feedback === 'positive' ? 'active' : ''}
          aria-pressed={feedback === 'positive'}
          aria-label="Good response"
        >
          👍
        </button>
        <button
          onClick={() => handleFeedback('negative')}
          className={feedback === 'negative' ? 'active' : ''}
          aria-pressed={feedback === 'negative'}
          aria-label="Poor response"
        >
          👎
        </button>
      </div>

      {showComment && (
        <div className="feedback-comment">
          <textarea
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="What went wrong? (optional)"
            rows={2}
          />
          <button
            onClick={() => {
              onFeedback(messageId, 'negative', comment);
              setShowComment(false);
            }}
          >
            Submit
          </button>
        </div>
      )}

      {feedback === 'positive' && (
        <span className="feedback-thanks" role="status">
          Thanks for the feedback!
        </span>
      )}
    </div>
  );
}
```

### Correction / regeneration controls

```tsx
function ResponseActions({
  onRegenerate,
  onEdit,
  onCopy,
}: {
  onRegenerate: () => void;
  onEdit: () => void;
  onCopy: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    onCopy();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="response-actions">
      <button onClick={onRegenerate} title="Regenerate response">
        🔄 Regenerate
      </button>
      <button onClick={onEdit} title="Edit and resubmit">
        ✏️ Edit
      </button>
      <button onClick={handleCopy} title="Copy to clipboard">
        {copied ? '✅ Copied!' : '📋 Copy'}
      </button>
    </div>
  );
}
```

---

## Streaming content rendering

Rendering tokens as they arrive requires specific UI patterns to avoid layout jumps and unreadable partial content.

### Cursor-style streaming display

```css
.streaming-text {
  white-space: pre-wrap;
  line-height: 1.6;
}

.streaming-text.active::after {
  content: '▌';
  animation: blink 1s infinite;
  color: #3182ce;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}
```

```tsx
function StreamingMessage({
  content,
  isStreaming,
}: {
  content: string;
  isStreaming: boolean;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom during streaming
  useEffect(() => {
    if (isStreaming) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [content, isStreaming]);

  return (
    <div className={`streaming-text ${isStreaming ? 'active' : ''}`}>
      {content}
      <div ref={endRef} />
    </div>
  );
}
```

### Skeleton loading for structured content

When the agent generates structured responses (tables, lists, code), show skeletons:

```tsx
function AgentSkeleton({ type }: { type: 'text' | 'code' | 'list' }) {
  if (type === 'text') {
    return (
      <div className="skeleton" aria-label="Loading response">
        <div className="skeleton-line" style={{ width: '100%' }} />
        <div className="skeleton-line" style={{ width: '85%' }} />
        <div className="skeleton-line" style={{ width: '92%' }} />
        <div className="skeleton-line" style={{ width: '60%' }} />
      </div>
    );
  }

  if (type === 'code') {
    return (
      <div className="skeleton skeleton--code" aria-label="Loading code">
        <div className="skeleton-line" style={{ width: '40%' }} />
        <div className="skeleton-line" style={{ width: '65%', marginLeft: 16 }} />
        <div className="skeleton-line" style={{ width: '50%', marginLeft: 16 }} />
        <div className="skeleton-line" style={{ width: '30%' }} />
      </div>
    );
  }

  return (
    <div className="skeleton" aria-label="Loading list">
      {[1, 2, 3].map(i => (
        <div key={i} className="skeleton-line" style={{ width: `${70 + i * 10}%` }} />
      ))}
    </div>
  );
}
```

```css
.skeleton-line {
  height: 16px;
  background: linear-gradient(90deg, #e2e8f0 25%, #edf2f7 50%, #e2e8f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 8px;
}

.skeleton--code {
  background: #1a202c;
  padding: 16px;
  border-radius: 8px;
}

.skeleton--code .skeleton-line {
  background: linear-gradient(90deg, #2d3748 25%, #4a5568 50%, #2d3748 75%);
  background-size: 200% 100%;
  height: 14px;
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
```

---

## Accessibility for agent interfaces

Agent UIs introduce unique accessibility challenges. Screen readers need to announce streaming content, tool activity, and state changes without overwhelming the user.

### Key accessibility patterns

```tsx
// 1. Announce agent state changes
<div role="status" aria-live="polite">
  {agentState === 'thinking' && 'Agent is thinking...'}
  {agentState === 'done' && 'Agent response complete.'}
</div>

// 2. Label interactive elements
<button
  onClick={onRegenerate}
  aria-label="Regenerate the agent's last response"
>
  🔄
</button>

// 3. Announce new messages without reading entire history
<div
  aria-live="polite"
  aria-atomic="false"    // Only announce new additions
  aria-relevant="additions"
>
  {messages.map(m => (
    <div key={m.id} role="article" aria-label={`${m.role} message`}>
      {m.content}
    </div>
  ))}
</div>

// 4. Skip repetitive tool call announcements
<div aria-live="polite">
  {toolCalls.length > 0 && (
    <span className="sr-only">
      Agent is using {toolCalls.length} tool{toolCalls.length > 1 ? 's' : ''}
    </span>
  )}
</div>
```

| Pattern | Implementation |
|---------|---------------|
| Agent state announcements | `aria-live="polite"` with `role="status"` |
| New message notification | `aria-live="polite"` with `aria-relevant="additions"` |
| Tool call progress | Summarize count, not individual calls |
| Streaming text | Announce completion, not each token |
| Error messages | `role="alert"` for immediate announcement |
| Feedback buttons | `aria-pressed` for toggle state |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Show thinking indicators immediately | Silence > 200ms feels broken |
| Make tool calls visible but collapsible | Transparency builds trust; details can be hidden |
| Categorize errors and offer specific recovery | "Try again" is vague; "Rephrase your question" is helpful |
| Add feedback on every agent response | Users feel heard; data improves quality |
| Use `aria-live="polite"` for agent status | Screen readers announce changes without interrupting |
| Auto-scroll during streaming, stop on user scroll | Don't fight the user's scroll position |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No feedback during thinking | Show indicator within 200ms of user action |
| Announcing every streamed token to screen readers | Announce completion only: "Response complete" |
| Generic "Something went wrong" errors | Classify errors and offer specific recovery actions |
| Hiding tool calls completely | Show at least a summary — users want to know what happened |
| No way to stop or retry | Always provide cancel, retry, and regenerate controls |
| Layout jumps during streaming | Reserve space or use smooth scroll anchoring |

---

## Hands-on exercise

### Your task

Build a complete agent chat UI component that includes:
1. A multi-state thinking indicator
2. Tool call visualization with collapsible details
3. Error handling with categorized recovery actions
4. Inline feedback (thumbs up/down) on agent responses

### Requirements

1. Implement `ThinkingIndicator` with at least 3 states (thinking, calling-tool, generating)
2. Create a `ToolTimeline` that shows each tool call with status, input, and output
3. Add an `AgentErrorUI` component that offers context-specific recovery actions
4. Include `MessageFeedback` with thumbs up/down and optional comment

### Expected result

A polished chat interface where users can see what the agent is doing at each step, recover from errors gracefully, and provide feedback on responses.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `role="status"` with `aria-live="polite"` for the thinking indicator
- Use `role="alert"` for error messages so screen readers announce immediately
- Use CSS `animation` for the bouncing dots and shimmer effects
- Track tool call steps in a state array that updates as events arrive

</details>

### Bonus challenges

- [ ] Add keyboard shortcuts (Esc to cancel, Enter to retry)
- [ ] Implement a "confidence meter" that shows how certain the agent is about its response
- [ ] Add a dark mode theme for all agent UI components

---

## Summary

✅ **Thinking indicators** with multiple states keep users informed about what the agent is doing  
✅ **Tool call visualization** builds transparency and trust by showing agent reasoning  
✅ **Categorized error handling** with specific recovery actions helps users get back on track  
✅ **Inline feedback** gives users control and provides data for quality improvement  
✅ **Accessibility patterns** ensure agent interfaces work for all users, including screen reader users

**Previous:** [Real-Time Agent Interfaces](./07-real-time-agent-interfaces.md)  
**Back to:** [Agent Web Integration Patterns](./00-agent-web-integration-patterns.md)

---

## Further Reading

- [Nielsen Norman Group: AI UX Guidelines](https://www.nngroup.com/articles/ai-ux/) - Research-backed AI UX patterns
- [MDN ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions) - Accessibility for dynamic content
- [Vercel AI Chatbot Template](https://vercel.com/templates/next.js/nextjs-ai-chatbot) - Production reference implementation
- [Google Material Design: AI Patterns](https://m3.material.io/) - Design system for AI interfaces

<!--
Sources Consulted:
- MDN ARIA live regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- Nielsen Norman Group AI UX research: https://www.nngroup.com/articles/ai-ux/
- Vercel AI SDK chatbot template: https://vercel.com/templates/next.js/nextjs-ai-chatbot
-->
