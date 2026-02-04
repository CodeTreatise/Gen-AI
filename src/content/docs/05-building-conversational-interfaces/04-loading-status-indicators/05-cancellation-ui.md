---
title: "Cancellation UI"
---

# Cancellation UI

## Introduction

Stop buttons, cancel confirmations, and graceful termination—cancellation UI gives users control over long-running AI operations. A well-designed cancellation system respects partial work, enables quick restarts, and maintains trust.

In this lesson, we'll implement cancellation patterns for AI chat interfaces.

### What We'll Cover

- Stop button design and placement
- Cancel confirmation patterns
- Handling partial results
- Resume and retry options
- Keyboard accessibility

### Prerequisites

- [Status Messages](./04-status-messages.md)
- AI SDK `stop()` function
- React event handling

---

## Stop Button Design

### Basic Stop Button

```jsx
function StopButton({ onStop, isStreaming }) {
  if (!isStreaming) return null;
  
  return (
    <button 
      onClick={onStop}
      className="stop-button"
      aria-label="Stop generating"
    >
      <span className="stop-icon" aria-hidden="true">■</span>
      <span className="stop-text">Stop</span>
    </button>
  );
}
```

```css
.stop-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: 2px solid var(--stop-color, #dc3545);
  background: transparent;
  color: var(--stop-color, #dc3545);
  border-radius: 20px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.stop-button:hover {
  background: var(--stop-color, #dc3545);
  color: white;
}

.stop-icon {
  width: 12px;
  height: 12px;
  background: currentColor;
  border-radius: 2px;
}
```

### Icon-Only Stop Button

```jsx
function CompactStopButton({ onStop }) {
  return (
    <button 
      onClick={onStop}
      className="compact-stop-button"
      aria-label="Stop generating response"
      title="Stop generating"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
        <rect x="4" y="4" width="12" height="12" rx="2" />
      </svg>
    </button>
  );
}
```

```css
.compact-stop-button {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 50%;
  background: var(--stop-color, #dc3545);
  color: white;
  cursor: pointer;
  transition: transform 0.1s, background 0.2s;
}

.compact-stop-button:hover {
  background: var(--stop-hover, #c82333);
  transform: scale(1.05);
}

.compact-stop-button:active {
  transform: scale(0.95);
}
```

### Animated Stop Button

```jsx
function AnimatedStopButton({ onStop, isVisible }) {
  return (
    <button 
      onClick={onStop}
      className={`animated-stop-btn ${isVisible ? 'visible' : ''}`}
      aria-label="Stop generating"
    >
      <div className="stop-icon-container">
        <div className="stop-square" />
      </div>
      <span>Stop generating</span>
    </button>
  );
}
```

```css
.animated-stop-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  background: var(--stop-bg, #fee2e2);
  color: var(--stop-color, #dc2626);
  cursor: pointer;
  opacity: 0;
  transform: translateY(10px);
  transition: opacity 0.2s, transform 0.2s;
}

.animated-stop-btn.visible {
  opacity: 1;
  transform: translateY(0);
}

.stop-icon-container {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stop-square {
  width: 10px;
  height: 10px;
  background: currentColor;
  border-radius: 2px;
  animation: pulse-stop 1s ease-in-out infinite;
}

@keyframes pulse-stop {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(0.8); }
}

.animated-stop-btn:hover .stop-square {
  animation: none;
  transform: scale(1.1);
}
```

---

## Button Placement

### Inline with Message

```jsx
function StreamingMessage({ content, isStreaming, onStop }) {
  return (
    <div className="message streaming">
      <div className="message-content">
        {content}
        {isStreaming && <span className="cursor">▋</span>}
      </div>
      
      {isStreaming && (
        <div className="message-actions">
          <StopButton onStop={onStop} isStreaming={isStreaming} />
        </div>
      )}
    </div>
  );
}
```

### Fixed Position

```jsx
function ChatContainer({ messages, isStreaming, onStop }) {
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map(m => <Message key={m.id} {...m} />)}
      </div>
      
      {isStreaming && (
        <div className="floating-stop">
          <AnimatedStopButton 
            onStop={onStop} 
            isVisible={isStreaming}
          />
        </div>
      )}
    </div>
  );
}
```

```css
.floating-stop {
  position: sticky;
  bottom: 20px;
  display: flex;
  justify-content: center;
  pointer-events: none;
}

.floating-stop button {
  pointer-events: auto;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
```

### Input Area Integration

```jsx
function ChatInput({ isStreaming, onStop, onSubmit }) {
  return (
    <div className="chat-input-container">
      <input type="text" placeholder="Type a message..." />
      
      {isStreaming ? (
        <CompactStopButton onStop={onStop} />
      ) : (
        <button className="send-button" onClick={onSubmit}>
          Send
        </button>
      )}
    </div>
  );
}
```

---

## Cancel Confirmation

### Quick Confirm Pattern

```jsx
function ConfirmableStopButton({ onStop }) {
  const [showConfirm, setShowConfirm] = useState(false);
  const timeoutRef = useRef(null);
  
  const handleClick = () => {
    if (showConfirm) {
      onStop();
      setShowConfirm(false);
    } else {
      setShowConfirm(true);
      // Auto-reset after 3 seconds
      timeoutRef.current = setTimeout(() => {
        setShowConfirm(false);
      }, 3000);
    }
  };
  
  useEffect(() => {
    return () => clearTimeout(timeoutRef.current);
  }, []);
  
  return (
    <button 
      onClick={handleClick}
      className={`confirm-stop-btn ${showConfirm ? 'confirming' : ''}`}
    >
      {showConfirm ? 'Click to confirm' : 'Stop'}
    </button>
  );
}
```

```css
.confirm-stop-btn {
  padding: 8px 16px;
  border: 2px solid var(--stop-color, #dc3545);
  background: transparent;
  color: var(--stop-color, #dc3545);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.confirm-stop-btn.confirming {
  background: var(--stop-color, #dc3545);
  color: white;
  animation: shake 0.3s;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-2px); }
  75% { transform: translateX(2px); }
}
```

### Modal Confirmation

```jsx
function CancelModal({ isOpen, onConfirm, onCancel }) {
  if (!isOpen) return null;
  
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div 
        className="cancel-modal" 
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-labelledby="modal-title"
      >
        <h3 id="modal-title">Stop generating?</h3>
        <p>The current response will be incomplete.</p>
        
        <div className="modal-actions">
          <button onClick={onCancel} className="modal-cancel">
            Keep generating
          </button>
          <button onClick={onConfirm} className="modal-confirm">
            Stop
          </button>
        </div>
      </div>
    </div>
  );
}
```

### When to Confirm

```jsx
function SmartStopButton({ onStop, tokensGenerated, threshold = 100 }) {
  const [showModal, setShowModal] = useState(false);
  
  const handleStop = () => {
    // Only confirm if significant work done
    if (tokensGenerated > threshold) {
      setShowModal(true);
    } else {
      onStop();
    }
  };
  
  return (
    <>
      <StopButton onStop={handleStop} isStreaming={true} />
      <CancelModal 
        isOpen={showModal}
        onConfirm={() => { onStop(); setShowModal(false); }}
        onCancel={() => setShowModal(false)}
      />
    </>
  );
}
```

---

## Handling Partial Results

### Keep Partial Response

```jsx
function useStreamingChat() {
  const { messages, stop, append, isLoading } = useChat();
  const [partialMessage, setPartialMessage] = useState(null);
  
  const handleStop = () => {
    // Capture the partial response before stopping
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === 'assistant' && isLoading) {
      setPartialMessage({
        ...lastMessage,
        metadata: { 
          wasInterrupted: true,
          stoppedAt: new Date().toISOString()
        }
      });
    }
    stop();
  };
  
  return { 
    messages, 
    stop: handleStop, 
    append, 
    isLoading,
    partialMessage 
  };
}
```

### Partial Result Indicator

```jsx
function PartialResponseMessage({ message }) {
  const isPartial = message.metadata?.wasInterrupted;
  
  return (
    <div className={`message ${isPartial ? 'partial' : ''}`}>
      <div className="message-content">
        {message.content}
      </div>
      
      {isPartial && (
        <div className="partial-indicator">
          <span className="partial-icon">⚠️</span>
          <span className="partial-text">
            Response was stopped
          </span>
        </div>
      )}
    </div>
  );
}
```

```css
.message.partial {
  border-left: 3px solid var(--warning-color, #ffc107);
}

.partial-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  margin-top: 8px;
  background: var(--warning-bg, #fff3cd);
  border-radius: 4px;
  font-size: 13px;
  color: var(--warning-text, #856404);
}
```

### Discard vs Keep Options

```jsx
function StopOptions({ onKeep, onDiscard, partialContent }) {
  const wordCount = partialContent.split(/\s+/).length;
  
  return (
    <div className="stop-options">
      <p>Response stopped ({wordCount} words generated)</p>
      
      <div className="stop-actions">
        <button onClick={onKeep} className="keep-btn">
          Keep partial response
        </button>
        <button onClick={onDiscard} className="discard-btn">
          Discard and retry
        </button>
      </div>
    </div>
  );
}
```

---

## Resume and Retry

### Continue Button

```jsx
function ContinueButton({ onContinue, lastMessage }) {
  if (!lastMessage?.metadata?.wasInterrupted) return null;
  
  return (
    <button onClick={onContinue} className="continue-btn">
      <span className="continue-icon">▶</span>
      <span>Continue generating</span>
    </button>
  );
}
```

```css
.continue-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: 2px solid var(--primary-color, #007bff);
  background: transparent;
  color: var(--primary-color, #007bff);
  border-radius: 6px;
  cursor: pointer;
}

.continue-btn:hover {
  background: var(--primary-color, #007bff);
  color: white;
}

.continue-icon {
  font-size: 12px;
}
```

### Regenerate Button

```jsx
function RegenerateButton({ onRegenerate }) {
  return (
    <button onClick={onRegenerate} className="regenerate-btn">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z"/>
        <path d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z"/>
      </svg>
      <span>Regenerate</span>
    </button>
  );
}
```

### AI SDK Integration

```jsx
function ChatActions({ status, stop, reload }) {
  return (
    <div className="chat-actions">
      {status === 'streaming' && (
        <StopButton onStop={stop} isStreaming={true} />
      )}
      
      {status === 'ready' && (
        <RegenerateButton onRegenerate={reload} />
      )}
      
      {status === 'error' && (
        <button onClick={reload} className="retry-btn">
          Try again
        </button>
      )}
    </div>
  );
}
```

---

## Keyboard Accessibility

### Escape Key to Cancel

```jsx
function useEscapeToCancel(onCancel, isActive) {
  useEffect(() => {
    if (!isActive) return;
    
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onCancel, isActive]);
}

// Usage
function StreamingChat({ stop, isLoading }) {
  useEscapeToCancel(stop, isLoading);
  
  return (
    <div className="chat">
      {isLoading && (
        <p className="keyboard-hint">
          Press <kbd>Esc</kbd> to stop
        </p>
      )}
    </div>
  );
}
```

```css
.keyboard-hint {
  font-size: 12px;
  color: var(--text-secondary);
}

kbd {
  padding: 2px 6px;
  background: var(--kbd-bg, #f7f7f7);
  border: 1px solid var(--kbd-border, #ccc);
  border-radius: 3px;
  font-family: monospace;
  font-size: 11px;
}
```

### Focus Management

```jsx
function CancellableOperation({ isRunning, onCancel }) {
  const cancelRef = useRef(null);
  
  // Focus cancel button when operation starts
  useEffect(() => {
    if (isRunning && cancelRef.current) {
      cancelRef.current.focus();
    }
  }, [isRunning]);
  
  return (
    <div role="region" aria-live="polite">
      {isRunning && (
        <>
          <p id="operation-status">Operation in progress...</p>
          <button 
            ref={cancelRef}
            onClick={onCancel}
            aria-describedby="operation-status"
          >
            Cancel
          </button>
        </>
      )}
    </div>
  );
}
```

---

## Complete Cancellation System

```jsx
function CancellableChat() {
  const { messages, input, handleSubmit, stop, reload, status } = useChat();
  const [partialResponse, setPartialResponse] = useState(null);
  
  // Escape key to cancel
  useEscapeToCancel(stop, status === 'streaming');
  
  const handleStop = () => {
    // Capture partial before stopping
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === 'assistant') {
      setPartialResponse({
        content: lastMsg.content,
        stoppedAt: Date.now()
      });
    }
    stop();
  };
  
  const handleContinue = () => {
    // Send a "continue" prompt
    const continueMsg = `Continue from: "${partialResponse.content.slice(-100)}..."`;
    // Append as user message and let AI continue
    setPartialResponse(null);
  };
  
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map(msg => (
          <Message key={msg.id} {...msg} />
        ))}
      </div>
      
      {/* Stop Button - visible during streaming */}
      {status === 'streaming' && (
        <div className="cancel-area">
          <AnimatedStopButton 
            onStop={handleStop}
            isVisible={true}
          />
          <span className="hint">or press Esc</span>
        </div>
      )}
      
      {/* Post-Cancel Actions */}
      {partialResponse && status === 'ready' && (
        <StopOptions
          partialContent={partialResponse.content}
          onKeep={() => setPartialResponse(null)}
          onDiscard={() => { setPartialResponse(null); reload(); }}
        />
      )}
      
      {/* Continue/Regenerate */}
      {status === 'ready' && !partialResponse && (
        <ChatActions status={status} reload={reload} />
      )}
      
      <ChatInput onSubmit={handleSubmit} />
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Make stop button always visible during streaming | Hide cancel option |
| Support keyboard shortcuts (Esc) | Require mouse click only |
| Preserve partial work | Discard without asking |
| Provide post-cancel options | Leave user in limbo |
| Confirm only for significant work | Confirm every cancel |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Cancel doesn't stop actual request | Use AbortController or SDK `stop()` |
| No feedback after canceling | Show "Stopped" status briefly |
| Button placement changes during stream | Keep in consistent location |
| Modal blocks urgent cancellation | Use inline confirmation |
| Escape key conflicts | Check no input is focused |

---

## Hands-on Exercise

### Your Task

Implement a complete cancellation UI that:
1. Shows stop button during streaming
2. Supports Escape key
3. Preserves partial responses
4. Offers continue/regenerate options

### Requirements

1. Visible stop button with hover state
2. Escape key handler
3. Partial response indicator
4. Continue and regenerate buttons

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `useEffect` for keyboard listener
- Track `wasInterrupted` in message metadata
- Position stop button in sticky container
- Focus manage after cancel action

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `CancellableChat` component in the Complete Cancellation System section above.

</details>

---

## Summary

✅ **Stop buttons** must be visible and accessible  
✅ **Keyboard shortcuts** (Esc) for quick cancellation  
✅ **Partial results** should be preserved by default  
✅ **Post-cancel options** let users continue or restart  
✅ **Confirmation** only needed for significant work  
✅ **Consistent placement** prevents UI jumping

---

## Further Reading

- [AI SDK useChat Reference](https://sdk.vercel.ai/docs/ai-sdk-ui/overview)
- [AbortController MDN](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)
- [Focus Management Patterns](https://www.w3.org/WAI/ARIA/apg/patterns/)

---

**Previous:** [Status Messages](./04-status-messages.md)  
**Next:** [useChat Status Integration](./06-usechat-status-integration.md)

<!-- 
Sources Consulted:
- AI SDK useChat docs: https://sdk.vercel.ai/docs/ai-sdk-ui/overview
- MDN AbortController: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
- WAI-ARIA Authoring Practices: https://www.w3.org/WAI/ARIA/apg/
-->
