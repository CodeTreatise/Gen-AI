---
title: "Status Messages"
---

# Status Messages

## Introduction

Status messages communicate what's happening behind the scenes: "Connecting to server," "Generating response," "Processing image." They turn invisible processes into understandable states, building user confidence and reducing uncertainty.

In this lesson, we'll implement clear, informative status messages for chat interfaces.

### What We'll Cover

- Connection status indicators
- Processing stage messages
- Queue position display
- Completion notifications
- Error and retry states

### Prerequisites

- [Progress Indicators](./03-progress-indicators.md)
- React state management
- AI SDK status states

---

## Connection Status

### Connection State Indicator

```jsx
function ConnectionStatus({ state }) {
  const states = {
    connecting: { icon: '🔄', text: 'Connecting...', color: 'orange' },
    connected: { icon: '🟢', text: 'Connected', color: 'green' },
    disconnected: { icon: '🔴', text: 'Disconnected', color: 'red' },
    reconnecting: { icon: '🔄', text: 'Reconnecting...', color: 'orange' }
  };
  
  const current = states[state] || states.disconnected;
  
  return (
    <div 
      className={`connection-status ${state}`}
      role="status"
      aria-live="polite"
    >
      <span className="status-icon" aria-hidden="true">
        {current.icon}
      </span>
      <span className="status-text">{current.text}</span>
    </div>
  );
}
```

```css
.connection-status {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  background: var(--status-bg, #f5f5f5);
}

.connection-status.connected {
  color: var(--success-color, #28a745);
}

.connection-status.disconnected {
  color: var(--error-color, #dc3545);
}

.connection-status.connecting,
.connection-status.reconnecting {
  color: var(--warning-color, #ffc107);
}

.connection-status.connecting .status-icon,
.connection-status.reconnecting .status-icon {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### WebSocket Connection Hook

```jsx
function useConnectionStatus(url) {
  const [status, setStatus] = useState('disconnected');
  const wsRef = useRef(null);
  
  useEffect(() => {
    const connect = () => {
      setStatus('connecting');
      
      wsRef.current = new WebSocket(url);
      
      wsRef.current.onopen = () => {
        setStatus('connected');
      };
      
      wsRef.current.onclose = () => {
        setStatus('disconnected');
        // Auto-reconnect after delay
        setTimeout(reconnect, 3000);
      };
      
      wsRef.current.onerror = () => {
        setStatus('disconnected');
      };
    };
    
    const reconnect = () => {
      if (status !== 'connected') {
        setStatus('reconnecting');
        connect();
      }
    };
    
    connect();
    
    return () => {
      wsRef.current?.close();
    };
  }, [url]);
  
  return status;
}
```

### Banner Status

```jsx
function ConnectionBanner({ status, onRetry }) {
  if (status === 'connected') return null;
  
  return (
    <div className={`connection-banner ${status}`} role="alert">
      <span className="banner-icon">
        {status === 'disconnected' ? '⚠️' : '🔄'}
      </span>
      <span className="banner-text">
        {status === 'disconnected' && 'Connection lost'}
        {status === 'connecting' && 'Connecting to server...'}
        {status === 'reconnecting' && 'Attempting to reconnect...'}
      </span>
      {status === 'disconnected' && (
        <button onClick={onRetry} className="retry-btn">
          Retry
        </button>
      )}
    </div>
  );
}
```

```css
.connection-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--warning-bg, #fff3cd);
  border-bottom: 1px solid var(--warning-border, #ffc107);
}

.connection-banner.disconnected {
  background: var(--error-bg, #f8d7da);
  border-color: var(--error-border, #dc3545);
}

.retry-btn {
  margin-left: auto;
  padding: 4px 12px;
  border-radius: 4px;
  background: var(--primary-color, #007bff);
  color: white;
  border: none;
  cursor: pointer;
}
```

---

## Processing Stages

### Stage Indicator

```jsx
function ProcessingStage({ stage, stages }) {
  const currentIndex = stages.indexOf(stage);
  
  return (
    <div className="processing-stage" role="status" aria-live="polite">
      <div className="stage-spinner">
        <Spinner size={16} />
      </div>
      <div className="stage-info">
        <span className="stage-text">{stage}</span>
        <span className="stage-progress">
          Step {currentIndex + 1} of {stages.length}
        </span>
      </div>
    </div>
  );
}
```

```css
.processing-stage {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--stage-bg, #e3f2fd);
  border-radius: 8px;
}

.stage-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stage-text {
  font-weight: 500;
  color: var(--text-primary);
}

.stage-progress {
  font-size: 12px;
  color: var(--text-secondary);
}
```

### Animated Stage Transitions

```jsx
function AnimatedStage({ stage }) {
  const [displayStage, setDisplayStage] = useState(stage);
  const [isTransitioning, setIsTransitioning] = useState(false);
  
  useEffect(() => {
    if (stage !== displayStage) {
      setIsTransitioning(true);
      
      setTimeout(() => {
        setDisplayStage(stage);
        setIsTransitioning(false);
      }, 200);
    }
  }, [stage, displayStage]);
  
  return (
    <div className={`animated-stage ${isTransitioning ? 'transitioning' : ''}`}>
      <Spinner size={16} />
      <span className="stage-text">{displayStage}</span>
    </div>
  );
}
```

```css
.animated-stage {
  display: flex;
  align-items: center;
  gap: 8px;
  transition: opacity 0.2s;
}

.animated-stage.transitioning {
  opacity: 0.5;
}

.animated-stage .stage-text {
  transition: transform 0.2s;
}

.animated-stage.transitioning .stage-text {
  transform: translateY(4px);
}
```

### Multi-Stage Status

```jsx
function MultiStageStatus({ stages }) {
  const current = stages.find(s => s.status === 'active');
  const completed = stages.filter(s => s.status === 'complete').length;
  
  return (
    <div className="multi-stage-status">
      <div className="stage-summary">
        <span className="summary-icon">
          {current ? <Spinner size={14} /> : '✓'}
        </span>
        <span className="summary-text">
          {current ? current.label : 'Complete'}
        </span>
        <span className="summary-count">
          {completed}/{stages.length}
        </span>
      </div>
      
      <div className="stage-dots">
        {stages.map((stage, i) => (
          <span 
            key={i}
            className={`stage-dot ${stage.status}`}
            title={stage.label}
          />
        ))}
      </div>
    </div>
  );
}
```

```css
.stage-dots {
  display: flex;
  gap: 4px;
}

.stage-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dot-pending, #e0e0e0);
}

.stage-dot.complete {
  background: var(--success-color, #28a745);
}

.stage-dot.active {
  background: var(--primary-color, #007bff);
  animation: dot-pulse 1s ease-in-out infinite;
}

@keyframes dot-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.3); }
}
```

---

## Queue Position

### Queue Status Display

```jsx
function QueueStatus({ position, estimatedWait }) {
  if (position <= 0) return null;
  
  return (
    <div className="queue-status" role="status" aria-live="polite">
      <div className="queue-icon">⏳</div>
      <div className="queue-info">
        <span className="queue-position">
          Position {position} in queue
        </span>
        {estimatedWait && (
          <span className="queue-wait">
            Estimated wait: ~{estimatedWait}
          </span>
        )}
      </div>
    </div>
  );
}
```

```css
.queue-status {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: var(--queue-bg, #f8f9fa);
  border-radius: 8px;
  border: 1px solid var(--border-color, #dee2e6);
}

.queue-icon {
  font-size: 24px;
}

.queue-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.queue-position {
  font-weight: 500;
}

.queue-wait {
  font-size: 13px;
  color: var(--text-secondary);
}
```

### Animated Queue Position

```jsx
function AnimatedQueuePosition({ position }) {
  const prevPosition = useRef(position);
  const [isDecreasing, setIsDecreasing] = useState(false);
  
  useEffect(() => {
    if (position < prevPosition.current) {
      setIsDecreasing(true);
      setTimeout(() => setIsDecreasing(false), 500);
    }
    prevPosition.current = position;
  }, [position]);
  
  return (
    <span className={`queue-number ${isDecreasing ? 'decreasing' : ''}`}>
      #{position}
    </span>
  );
}
```

```css
.queue-number {
  font-size: 24px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  transition: transform 0.3s, color 0.3s;
}

.queue-number.decreasing {
  transform: scale(1.2);
  color: var(--success-color, #28a745);
}
```

### Queue Progress

```jsx
function QueueProgress({ position, totalAhead }) {
  const progress = ((totalAhead - position) / totalAhead) * 100;
  
  return (
    <div className="queue-progress">
      <div className="queue-header">
        <span>Waiting in queue</span>
        <span>{position} ahead of you</span>
      </div>
      <div className="queue-bar">
        <div 
          className="queue-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="queue-footer">
        <span>Started</span>
        <span>Your turn</span>
      </div>
    </div>
  );
}
```

---

## Completion Notifications

### Success Message

```jsx
function CompletionMessage({ type = 'success', message, onDismiss }) {
  const icons = {
    success: '✓',
    info: 'ℹ',
    warning: '⚠',
    error: '✗'
  };
  
  return (
    <div className={`completion-message ${type}`} role="alert">
      <span className="completion-icon">{icons[type]}</span>
      <span className="completion-text">{message}</span>
      {onDismiss && (
        <button 
          onClick={onDismiss} 
          className="dismiss-btn"
          aria-label="Dismiss"
        >
          ×
        </button>
      )}
    </div>
  );
}
```

```css
.completion-message {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  animation: slide-in 0.3s ease-out;
}

.completion-message.success {
  background: var(--success-bg, #d4edda);
  color: var(--success-text, #155724);
}

.completion-message.error {
  background: var(--error-bg, #f8d7da);
  color: var(--error-text, #721c24);
}

.completion-icon {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: currentColor;
  color: white;
  font-size: 14px;
}

.dismiss-btn {
  margin-left: auto;
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  opacity: 0.6;
}

.dismiss-btn:hover {
  opacity: 1;
}

@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### Auto-Dismiss Notification

```jsx
function AutoDismissNotification({ message, duration = 5000, onDismiss }) {
  const [visible, setVisible] = useState(true);
  const [progress, setProgress] = useState(100);
  
  useEffect(() => {
    const startTime = Date.now();
    
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100);
      setProgress(remaining);
      
      if (remaining === 0) {
        clearInterval(interval);
        setVisible(false);
        onDismiss?.();
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, [duration, onDismiss]);
  
  if (!visible) return null;
  
  return (
    <div className="auto-dismiss-notification">
      <span className="notification-text">{message}</span>
      <div 
        className="dismiss-progress"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
```

### Toast Notification System

```jsx
function useToast() {
  const [toasts, setToasts] = useState([]);
  
  const addToast = useCallback((message, type = 'info', duration = 5000) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type, duration }]);
    
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, duration);
  }, []);
  
  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);
  
  return { toasts, addToast, removeToast };
}

function ToastContainer({ toasts, onRemove }) {
  return (
    <div className="toast-container" role="region" aria-label="Notifications">
      {toasts.map(toast => (
        <CompletionMessage
          key={toast.id}
          type={toast.type}
          message={toast.message}
          onDismiss={() => onRemove(toast.id)}
        />
      ))}
    </div>
  );
}
```

```css
.toast-container {
  position: fixed;
  bottom: 24px;
  right: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1000;
  max-width: 400px;
}
```

---

## Error and Retry States

### Error Status

```jsx
function ErrorStatus({ error, onRetry, onDismiss }) {
  return (
    <div className="error-status" role="alert">
      <div className="error-header">
        <span className="error-icon">⚠️</span>
        <span className="error-title">Something went wrong</span>
      </div>
      
      <p className="error-message">{error.message}</p>
      
      <div className="error-actions">
        <button onClick={onRetry} className="retry-button">
          Try again
        </button>
        {onDismiss && (
          <button onClick={onDismiss} className="dismiss-button">
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
```

### Retry with Countdown

```jsx
function RetryCountdown({ seconds, onRetry, onCancel }) {
  const [remaining, setRemaining] = useState(seconds);
  
  useEffect(() => {
    if (remaining <= 0) {
      onRetry();
      return;
    }
    
    const timer = setTimeout(() => {
      setRemaining(prev => prev - 1);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [remaining, onRetry]);
  
  return (
    <div className="retry-countdown">
      <span>Retrying in {remaining}s...</span>
      <button onClick={onRetry}>Retry now</button>
      <button onClick={onCancel}>Cancel</button>
    </div>
  );
}
```

### AI SDK Error Integration

```jsx
function ChatErrorState({ error, reload, status }) {
  if (status !== 'error') return null;
  
  const errorMessages = {
    'rate-limit': 'Too many requests. Please wait a moment.',
    'network': 'Network error. Check your connection.',
    'server': 'Server error. We\'re working on it.',
    'default': 'Something went wrong.'
  };
  
  const errorType = error?.message?.includes('429') ? 'rate-limit' :
                    error?.message?.includes('network') ? 'network' :
                    error?.message?.includes('500') ? 'server' : 'default';
  
  return (
    <div className="chat-error">
      <ErrorStatus
        error={{ message: errorMessages[errorType] }}
        onRetry={reload}
      />
    </div>
  );
}
```

---

## Complete Status System

```jsx
function ChatStatusBar({ 
  connectionStatus,
  processingStage,
  queuePosition,
  error,
  onRetry 
}) {
  // Priority: Error > Queue > Processing > Connection
  
  if (error) {
    return (
      <div className="status-bar error">
        <span>⚠️ {error.message}</span>
        <button onClick={onRetry}>Retry</button>
      </div>
    );
  }
  
  if (queuePosition > 0) {
    return (
      <div className="status-bar queue">
        <span>⏳ Position {queuePosition} in queue</span>
      </div>
    );
  }
  
  if (processingStage) {
    return (
      <div className="status-bar processing">
        <Spinner size={14} />
        <span>{processingStage}</span>
      </div>
    );
  }
  
  if (connectionStatus !== 'connected') {
    return (
      <ConnectionBanner 
        status={connectionStatus} 
        onRetry={onRetry} 
      />
    );
  }
  
  return null;  // No status to show
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use clear, concise language | Use technical jargon |
| Show context (stage, position) | Only show spinner |
| Provide actions (retry, cancel) | Leave users stuck |
| Auto-dismiss success messages | Require manual dismissal |
| Animate transitions smoothly | Flash between states |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Status flickers rapidly | Debounce state changes |
| Generic "Error" message | Provide specific, actionable text |
| No recovery options | Always offer retry/dismiss |
| Status covers content | Use non-blocking positions |
| Multiple overlapping statuses | Prioritize and show one |

---

## Hands-on Exercise

### Your Task

Create a complete status message system that:
1. Shows connection status
2. Displays processing stages
3. Handles errors with retry
4. Auto-dismisses success messages

### Requirements

1. Three states: connected, disconnected, connecting
2. Processing stage with spinner
3. Error message with retry button
4. Toast notification for completion

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `useState` for each status type
- Priority: error > processing > connection
- Use `setTimeout` for auto-dismiss
- Include `role="status"` for accessibility

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `ChatStatusBar` component in the Complete Status System section above.

</details>

---

## Summary

✅ **Connection status** builds confidence in reliability  
✅ **Processing stages** explain what's happening  
✅ **Queue position** sets wait expectations  
✅ **Completion notifications** confirm success  
✅ **Error states** provide recovery options  
✅ **Prioritized display** prevents overwhelm

---

## Further Reading

- [Toast Notifications UX](https://uxplanet.org/toast-notification-or-how-to-bake-perfect-communication-with-the-user-8ade0fa5a81a)
- [Error Message Guidelines](https://www.nngroup.com/articles/error-message-guidelines/)
- [ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)

---

**Previous:** [Progress Indicators](./03-progress-indicators.md)  
**Next:** [Cancellation UI](./05-cancellation-ui.md)

<!-- 
Sources Consulted:
- MDN ARIA Live Regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- Nielsen Norman Group errors: https://www.nngroup.com/articles/error-message-guidelines/
- Toast notification patterns: https://web.dev/patterns/
-->
