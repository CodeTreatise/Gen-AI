---
title: "Error Message Display"
---

# Error Message Display

## Introduction

Error messages are inevitable in AI applications—API failures, rate limits, content filters, and network issues all happen. How you display these errors dramatically impacts user experience and trust.

In this lesson, we'll design error message patterns that are informative, actionable, and maintain conversation context.

### What We'll Cover

- Error categorization and severity levels
- Visual error message design
- Inline vs toast notifications
- Retry and recovery patterns
- Graceful degradation
- Error message accessibility

### Prerequisites

- [Message Container Structure](./01-message-container-structure.md)
- Understanding of HTTP status codes
- Basic error handling concepts

---

## Error Categorization

### Common AI Chat Errors

| Category | Examples | Severity | User Action |
|----------|----------|----------|-------------|
| **Network** | Offline, timeout | High | Retry when online |
| **Rate Limit** | 429 errors | Medium | Wait and retry |
| **Content Filter** | Blocked content | Medium | Modify request |
| **API Error** | 500, service unavailable | High | Retry later |
| **Auth Error** | Invalid/expired key | Critical | Re-authenticate |
| **Input Error** | Too long, invalid format | Low | Fix input |
| **Model Error** | Context length exceeded | Medium | Shorten conversation |

### Error Type Definitions

```typescript
type ErrorSeverity = 'info' | 'warning' | 'error' | 'critical';

interface ChatError {
  id: string;
  type: 
    | 'network'
    | 'rate_limit'
    | 'content_filter'
    | 'api_error'
    | 'auth_error'
    | 'input_error'
    | 'model_error';
  severity: ErrorSeverity;
  message: string;
  userMessage: string;      // Friendly message
  technicalDetails?: string; // For developers
  retryable: boolean;
  retryAfter?: number;      // Seconds until retry
  action?: 'retry' | 'modify' | 'authenticate' | 'contact_support';
}
```

### Error Factory

```javascript
const createError = {
  network: () => ({
    type: 'network',
    severity: 'error',
    userMessage: "Can't connect to the server. Check your internet connection.",
    retryable: true,
    action: 'retry'
  }),
  
  rateLimit: (retryAfter = 60) => ({
    type: 'rate_limit',
    severity: 'warning',
    userMessage: `Too many requests. Please wait ${retryAfter} seconds.`,
    retryable: true,
    retryAfter,
    action: 'retry'
  }),
  
  contentFilter: () => ({
    type: 'content_filter',
    severity: 'warning',
    userMessage: "Your message couldn't be processed due to content guidelines.",
    retryable: false,
    action: 'modify'
  }),
  
  contextLength: () => ({
    type: 'model_error',
    severity: 'warning',
    userMessage: "The conversation is too long. Consider starting a new chat.",
    retryable: false,
    action: 'modify'
  }),
  
  apiError: (status) => ({
    type: 'api_error',
    severity: 'error',
    userMessage: "Something went wrong on our end. Please try again.",
    technicalDetails: `API returned ${status}`,
    retryable: true,
    action: 'retry'
  })
};
```

---

## Visual Error Design

### Error Message Styles

```css
/* Base error container */
.error-message {
  display: flex;
  gap: 0.75rem;
  padding: 1rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0;
}

/* Severity variants */
.error-message.info {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1e40af;
}

.error-message.warning {
  background: #fffbeb;
  border: 1px solid #fde68a;
  color: #92400e;
}

.error-message.error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
}

.error-message.critical {
  background: #450a0a;
  border: 1px solid #dc2626;
  color: #fecaca;
}

/* Error icon */
.error-icon {
  flex-shrink: 0;
  width: 1.5rem;
  height: 1.5rem;
}

.error-message.info .error-icon { color: #3b82f6; }
.error-message.warning .error-icon { color: #f59e0b; }
.error-message.error .error-icon { color: #ef4444; }
.error-message.critical .error-icon { color: #f87171; }

/* Error content */
.error-content {
  flex: 1;
  min-width: 0;
}

.error-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.error-description {
  font-size: 0.875rem;
  line-height: 1.5;
}

/* Error actions */
.error-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.error-btn {
  padding: 0.375rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}

.error-btn-primary {
  background: currentColor;
  color: white;
  border: none;
}

.error-message.error .error-btn-primary {
  background: #dc2626;
}

.error-btn-primary:hover {
  opacity: 0.9;
}

.error-btn-secondary {
  background: transparent;
  border: 1px solid currentColor;
}

.error-btn-secondary:hover {
  background: rgba(0, 0, 0, 0.05);
}
```

### Error Icons

```html
<!-- Info icon -->
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <circle cx="12" cy="12" r="10"/>
  <line x1="12" y1="16" x2="12" y2="12"/>
  <line x1="12" y1="8" x2="12.01" y2="8"/>
</svg>

<!-- Warning icon -->
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <path d="M12 9v4M12 17h.01"/>
  <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
</svg>

<!-- Error icon -->
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <circle cx="12" cy="12" r="10"/>
  <line x1="15" y1="9" x2="9" y2="15"/>
  <line x1="9" y1="9" x2="15" y2="15"/>
</svg>

<!-- Critical icon -->
<svg viewBox="0 0 24 24" fill="currentColor">
  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
</svg>
```

---

## Inline Error Messages

### Error in Message Thread

When an error occurs after a user message:

```css
.message-wrapper.error {
  opacity: 1;
}

.message-error-response {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
  max-width: min(85%, 48rem);
}

.message-error-icon {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  padding: 0.375rem;
  background: #fee2e2;
  border-radius: 50%;
  color: #dc2626;
}

.message-error-content {
  flex: 1;
}

.message-error-title {
  font-weight: 600;
  color: #991b1b;
  margin-bottom: 0.25rem;
}

.message-error-text {
  font-size: 0.875rem;
  color: #b91c1c;
  margin-bottom: 0.75rem;
}

.message-error-actions {
  display: flex;
  gap: 0.5rem;
}

.message-retry-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.5rem 1rem;
  background: #dc2626;
  border: none;
  border-radius: 0.375rem;
  color: white;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
}

.message-retry-btn:hover {
  background: #b91c1c;
}

.message-retry-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

```html
<div class="message-wrapper assistant">
  <div class="message-avatar error-avatar">⚠️</div>
  <div class="message-error-response">
    <div class="message-error-icon">
      <svg><!-- error icon --></svg>
    </div>
    <div class="message-error-content">
      <div class="message-error-title">Unable to Generate Response</div>
      <p class="message-error-text">
        The server is temporarily unavailable. Please try again in a moment.
      </p>
      <div class="message-error-actions">
        <button class="message-retry-btn">
          <svg><!-- refresh icon --></svg>
          Retry
        </button>
        <button class="error-btn-secondary">
          Start New Chat
        </button>
      </div>
    </div>
  </div>
</div>
```

### Failed User Message

When a user's message fails to send:

```css
.message-wrapper.user.send-failed .message-container {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
}

.send-failed-indicator {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #dc2626;
}

.send-failed-indicator svg {
  width: 0.875rem;
  height: 0.875rem;
}
```

---

## Toast Notifications

### For Non-Blocking Errors

```css
.toast-container {
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  z-index: 1000;
  pointer-events: none;
}

.toast {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  max-width: 24rem;
  pointer-events: auto;
  animation: toast-in 0.3s ease-out;
}

.toast.exiting {
  animation: toast-out 0.2s ease-in forwards;
}

@keyframes toast-in {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes toast-out {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100%);
  }
}

.toast-icon {
  flex-shrink: 0;
  width: 1.25rem;
  height: 1.25rem;
}

.toast.error .toast-icon { color: #dc2626; }
.toast.warning .toast-icon { color: #f59e0b; }
.toast.success .toast-icon { color: #10b981; }

.toast-content {
  flex: 1;
  min-width: 0;
}

.toast-message {
  font-size: 0.875rem;
  color: #374151;
}

.toast-close {
  flex-shrink: 0;
  width: 1.25rem;
  height: 1.25rem;
  padding: 0;
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
}

.toast-close:hover {
  color: #6b7280;
}

/* Progress bar for auto-dismiss */
.toast-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3px;
  background: currentColor;
  opacity: 0.3;
  animation: toast-progress 5s linear forwards;
}

@keyframes toast-progress {
  from { width: 100%; }
  to { width: 0%; }
}
```

### Toast Manager

```javascript
class ToastManager {
  constructor() {
    this.container = this.createContainer();
    this.toasts = new Map();
  }
  
  createContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    container.setAttribute('role', 'alert');
    container.setAttribute('aria-live', 'polite');
    document.body.appendChild(container);
    return container;
  }
  
  show({ message, type = 'error', duration = 5000, action }) {
    const id = Date.now().toString();
    const toast = this.createToast(id, message, type, action);
    
    this.container.appendChild(toast);
    this.toasts.set(id, toast);
    
    if (duration > 0) {
      setTimeout(() => this.dismiss(id), duration);
    }
    
    return id;
  }
  
  createToast(id, message, type, action) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.dataset.id = id;
    
    toast.innerHTML = `
      <svg class="toast-icon">${this.getIcon(type)}</svg>
      <div class="toast-content">
        <p class="toast-message">${message}</p>
        ${action ? `<button class="toast-action">${action.label}</button>` : ''}
      </div>
      <button class="toast-close" aria-label="Dismiss">×</button>
      <div class="toast-progress"></div>
    `;
    
    toast.querySelector('.toast-close').onclick = () => this.dismiss(id);
    
    if (action) {
      toast.querySelector('.toast-action').onclick = () => {
        action.onClick();
        this.dismiss(id);
      };
    }
    
    return toast;
  }
  
  dismiss(id) {
    const toast = this.toasts.get(id);
    if (!toast) return;
    
    toast.classList.add('exiting');
    toast.addEventListener('animationend', () => {
      toast.remove();
      this.toasts.delete(id);
    });
  }
  
  getIcon(type) {
    const icons = {
      error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
      warning: '<path d="M12 9v4M12 17h.01"/><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>',
      success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'
    };
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${icons[type]}</svg>`;
  }
}

// Usage
const toast = new ToastManager();

toast.show({
  message: 'Rate limit exceeded. Please wait 60 seconds.',
  type: 'warning',
  duration: 10000,
  action: {
    label: 'Retry Now',
    onClick: () => retryRequest()
  }
});
```

---

## Retry and Recovery Patterns

### Auto-Retry with Backoff

```javascript
async function fetchWithRetry(fn, options = {}) {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 30000,
    onRetry
  } = options;
  
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (!isRetryable(error) || attempt === maxRetries) {
        throw error;
      }
      
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
        maxDelay
      );
      
      if (onRetry) {
        onRetry({ attempt, delay, error });
      }
      
      await sleep(delay);
    }
  }
  
  throw lastError;
}

function isRetryable(error) {
  // Network errors
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return true;
  }
  
  // Server errors (5xx) and rate limits (429)
  if (error.status >= 500 || error.status === 429) {
    return true;
  }
  
  return false;
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
```

### Retry UI Component

```jsx
function RetryableMessage({ message, onRetry }) {
  const [retryState, setRetryState] = useState('idle');
  const [countdown, setCountdown] = useState(0);
  
  const handleRetry = async () => {
    setRetryState('retrying');
    
    try {
      await onRetry(message);
      setRetryState('success');
    } catch (error) {
      if (error.retryAfter) {
        setRetryState('waiting');
        setCountdown(error.retryAfter);
        
        const timer = setInterval(() => {
          setCountdown(c => {
            if (c <= 1) {
              clearInterval(timer);
              setRetryState('idle');
              return 0;
            }
            return c - 1;
          });
        }, 1000);
      } else {
        setRetryState('failed');
      }
    }
  };
  
  return (
    <div className="retry-container">
      {retryState === 'retrying' && (
        <div className="retry-loading">
          <Spinner /> Retrying...
        </div>
      )}
      
      {retryState === 'waiting' && (
        <div className="retry-countdown">
          Rate limited. Retry in {countdown}s
          <div className="countdown-bar" style={{ width: `${(countdown / message.retryAfter) * 100}%` }} />
        </div>
      )}
      
      {retryState === 'idle' && (
        <button className="retry-btn" onClick={handleRetry}>
          <RefreshIcon /> Retry
        </button>
      )}
      
      {retryState === 'failed' && (
        <div className="retry-failed">
          <span>Retry failed</span>
          <button onClick={handleRetry}>Try again</button>
        </div>
      )}
    </div>
  );
}
```

---

## Graceful Degradation

### Fallback Responses

```javascript
async function getAIResponse(messages) {
  try {
    return await callPrimaryAPI(messages);
  } catch (error) {
    // Try backup model
    if (error.status === 503) {
      try {
        return await callBackupAPI(messages);
      } catch (backupError) {
        // Return cached/static response
        return getFallbackResponse(messages);
      }
    }
    throw error;
  }
}

function getFallbackResponse(messages) {
  const lastUserMessage = messages.filter(m => m.role === 'user').pop();
  
  return {
    role: 'assistant',
    content: `I'm currently experiencing high demand and can't process your request fully. 

Here's what I understood: "${lastUserMessage?.content.slice(0, 100)}..."

Please try again in a few moments, or rephrase your question.`,
    metadata: {
      isFallback: true,
      timestamp: new Date().toISOString()
    }
  };
}
```

### Offline Mode

```css
.offline-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: #fef3c7;
  border-bottom: 1px solid #fde68a;
  color: #92400e;
  font-size: 0.875rem;
}

.offline-banner-icon {
  width: 1rem;
  height: 1rem;
}

.offline-indicator {
  position: fixed;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.5rem 1rem;
  background: #1f2937;
  border-radius: 2rem;
  color: white;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  animation: slide-up 0.3s ease-out;
}

@keyframes slide-up {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(1rem);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}
```

```javascript
function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);
  
  return isOnline;
}
```

---

## Error Message Accessibility

### Screen Reader Announcements

```jsx
function ErrorAnnouncer({ error }) {
  return (
    <div 
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      className="sr-only"
    >
      {error && `Error: ${error.userMessage}`}
    </div>
  );
}
```

### Focus Management

```javascript
function handleError(error, containerRef) {
  // Create error element
  const errorElement = createErrorElement(error);
  containerRef.current.appendChild(errorElement);
  
  // Focus the error for screen readers
  errorElement.focus();
  
  // Announce to screen readers
  announceToScreenReader(`Error: ${error.userMessage}`);
}

function announceToScreenReader(message) {
  const announcement = document.createElement('div');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', 'polite');
  announcement.className = 'sr-only';
  announcement.textContent = message;
  
  document.body.appendChild(announcement);
  
  setTimeout(() => announcement.remove(), 1000);
}
```

### Keyboard Accessible Retry

```css
.error-retry-btn:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

.error-retry-btn:focus-visible {
  box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.3);
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use plain language in errors | Show technical jargon to users |
| Provide actionable next steps | Leave users stuck |
| Distinguish severity visually | Use same style for all errors |
| Auto-retry transient failures | Force manual retry always |
| Announce errors to screen readers | Ignore accessibility |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Showing raw API errors | Map to user-friendly messages |
| No retry option | Add retry button for retryable errors |
| Errors disappear too fast | Let users dismiss manually |
| Breaking conversation flow | Show error inline, preserve context |
| Infinite retry loops | Implement max retries with backoff |

---

## Hands-on Exercise

### Your Task

Create an error handling system with:
1. Error categorization (network, rate limit, content filter)
2. Inline error message component
3. Toast notifications for non-blocking errors
4. Retry with exponential backoff
5. Accessible announcements

### Requirements

1. Design error styles for 4 severity levels
2. Implement auto-dismiss toasts with progress bar
3. Add countdown for rate limit errors
4. Ensure keyboard and screen reader accessibility

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `aria-live="polite"` for toasts
- Implement backoff as `delay * 2^attempt`
- Store retry state in component state
- Use CSS custom properties for severity colors

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the complete implementations in the Toast Manager and RetryableMessage sections above.

</details>

---

## Summary

✅ **Categorize errors** by type and severity for appropriate handling  
✅ **Use friendly language** that explains what went wrong  
✅ **Provide clear actions** like retry, modify, or contact support  
✅ **Implement retry logic** with exponential backoff  
✅ **Use toasts** for non-blocking notifications  
✅ **Ensure accessibility** with ARIA announcements and focus management

---

## Further Reading

- [MDN ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)
- [Exponential Backoff](https://cloud.google.com/iot/docs/how-tos/exponential-backoff)
- [Error Message Guidelines](https://www.nngroup.com/articles/error-message-guidelines/)

---

**Previous:** [System Message Handling](./04-system-message-handling.md)  
**Next:** [Message Grouping Strategies](./06-message-grouping-strategies.md)

<!-- 
Sources Consulted:
- MDN ARIA Live Regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- Google Exponential Backoff: https://cloud.google.com/iot/docs/how-tos/exponential-backoff
- NN Group Error Messages: https://www.nngroup.com/articles/error-message-guidelines/
-->
