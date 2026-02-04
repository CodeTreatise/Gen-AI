---
title: "Offline Indicators"
---

# Offline Indicators

## Introduction

When users go offline, they need clear visual feedback about their connectivity status. Without proper indicators, users may think the app is broken or that their messages failed. Good offline indicators communicate status, set expectations, and guide user behavior.

This lesson covers designing and implementing effective offline indicators for chat applications.

### What We'll Cover

- Connection status displays
- Offline mode messaging patterns
- Feature availability indicators
- Reconnection attempt feedback
- Accessibility considerations

### Prerequisites

- [Caching Conversation History](./01-caching-conversation-history.md)
- [Sync on Reconnection](./03-sync-on-reconnection.md)
- Basic CSS knowledge

---

## Connection Status Display

### Status Banner Patterns

The most common approach is a persistent banner at the top of the screen:

```html
<div id="connection-status" class="connection-status hidden" role="status" aria-live="polite">
  <span class="status-icon" aria-hidden="true"></span>
  <span class="status-message"></span>
  <button class="status-action hidden">Retry</button>
</div>
```

```css
.connection-status {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: var(--warning-bg, #fff3cd);
  color: var(--warning-text, #856404);
  font-size: 0.875rem;
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.connection-status.hidden {
  transform: translateY(-100%);
  opacity: 0;
  pointer-events: none;
}

.connection-status.offline {
  background: var(--warning-bg, #fff3cd);
  color: var(--warning-text, #856404);
}

.connection-status.reconnecting {
  background: var(--info-bg, #cce5ff);
  color: var(--info-text, #004085);
}

.connection-status.error {
  background: var(--danger-bg, #f8d7da);
  color: var(--danger-text, #721c24);
}

.status-icon {
  font-size: 1.25rem;
}

.status-action {
  margin-left: auto;
  padding: 0.375rem 0.75rem;
  background: rgba(0, 0, 0, 0.1);
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  color: inherit;
  font-size: inherit;
}

.status-action:hover {
  background: rgba(0, 0, 0, 0.2);
}

.status-action.hidden {
  display: none;
}
```

### Status Controller

```javascript
// connectionStatusUI.js
import connectionManager from './connectionManager.js';

class ConnectionStatusUI {
  constructor() {
    this.container = document.getElementById('connection-status');
    this.iconEl = this.container.querySelector('.status-icon');
    this.messageEl = this.container.querySelector('.status-message');
    this.actionEl = this.container.querySelector('.status-action');
    
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    
    this.setupListeners();
  }
  
  setupListeners() {
    connectionManager.subscribe((event) => {
      if (event.type === 'online') {
        this.showOnline();
      } else if (event.type === 'offline') {
        this.showOffline();
      }
    });
    
    this.actionEl.addEventListener('click', () => {
      this.attemptReconnect();
    });
    
    // Initial state
    if (!navigator.onLine) {
      this.showOffline();
    }
  }
  
  showOffline() {
    this.container.className = 'connection-status offline';
    this.iconEl.textContent = '📡';
    this.messageEl.textContent = "You're offline. Messages will be sent when you reconnect.";
    this.actionEl.classList.add('hidden');
    
    this.show();
    this.startReconnectAttempts();
  }
  
  showReconnecting() {
    this.container.className = 'connection-status reconnecting';
    this.iconEl.textContent = '🔄';
    this.messageEl.textContent = `Reconnecting... (attempt ${this.reconnectAttempts})`;
    this.actionEl.classList.add('hidden');
  }
  
  showOnline() {
    this.stopReconnectAttempts();
    
    // Show brief "Back online" message
    this.container.className = 'connection-status online';
    this.iconEl.textContent = '✅';
    this.messageEl.textContent = 'Back online!';
    this.actionEl.classList.add('hidden');
    
    // Hide after 3 seconds
    setTimeout(() => this.hide(), 3000);
  }
  
  showError(message = 'Connection lost') {
    this.stopReconnectAttempts();
    
    this.container.className = 'connection-status error';
    this.iconEl.textContent = '⚠️';
    this.messageEl.textContent = message;
    
    this.actionEl.textContent = 'Retry';
    this.actionEl.classList.remove('hidden');
    
    this.show();
  }
  
  show() {
    this.container.classList.remove('hidden');
  }
  
  hide() {
    this.container.classList.add('hidden');
  }
  
  startReconnectAttempts() {
    this.reconnectAttempts = 0;
    this.scheduleReconnect();
  }
  
  stopReconnectAttempts() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = 0;
  }
  
  scheduleReconnect() {
    // Exponential backoff: 5s, 10s, 20s, 30s max
    const delays = [5000, 10000, 20000, 30000];
    const delay = delays[Math.min(this.reconnectAttempts, delays.length - 1)];
    
    this.reconnectTimer = setTimeout(() => {
      this.attemptReconnect();
    }, delay);
  }
  
  async attemptReconnect() {
    this.reconnectAttempts++;
    this.showReconnecting();
    
    const success = await connectionManager.verifyConnection();
    
    if (success) {
      // Will trigger 'online' event
    } else {
      this.showOffline();
      this.scheduleReconnect();
    }
  }
}

export default new ConnectionStatusUI();
```

---

## Offline Mode Messaging

When offline, adapt the chat interface to set correct expectations:

### Input Field States

```javascript
// chatInput.js
class ChatInput {
  constructor() {
    this.form = document.querySelector('.chat-input-form');
    this.input = this.form.querySelector('input');
    this.sendBtn = this.form.querySelector('button[type="submit"]');
    this.hint = this.form.querySelector('.input-hint');
    
    this.setupConnectionListener();
  }
  
  setupConnectionListener() {
    connectionManager.subscribe((event) => {
      if (event.type === 'offline') {
        this.setOfflineMode();
      } else if (event.type === 'online') {
        this.setOnlineMode();
      }
    });
    
    // Initial state
    if (!navigator.onLine) {
      this.setOfflineMode();
    }
  }
  
  setOfflineMode() {
    // Keep input enabled but update UI
    this.input.placeholder = 'Type a message (will send when online)...';
    this.sendBtn.textContent = 'Queue';
    this.sendBtn.classList.add('offline-mode');
    
    this.showHint('📤 Messages will be sent when you reconnect');
  }
  
  setOnlineMode() {
    this.input.placeholder = 'Type a message...';
    this.sendBtn.textContent = 'Send';
    this.sendBtn.classList.remove('offline-mode');
    
    this.hideHint();
  }
  
  showHint(message) {
    if (this.hint) {
      this.hint.textContent = message;
      this.hint.hidden = false;
    }
  }
  
  hideHint() {
    if (this.hint) {
      this.hint.hidden = true;
    }
  }
}
```

```css
/* Input styling for offline mode */
.chat-input-form {
  position: relative;
}

.chat-input-form .input-hint {
  position: absolute;
  bottom: 100%;
  left: 0;
  right: 0;
  padding: 0.5rem 1rem;
  background: var(--warning-bg, #fff3cd);
  color: var(--warning-text, #856404);
  font-size: 0.75rem;
  border-radius: 0.25rem 0.25rem 0 0;
}

.send-btn.offline-mode {
  background: var(--muted-color, #6c757d);
}
```

### Message Status Indicators

Show the status of each message clearly:

```html
<div class="message outgoing queued">
  <div class="message-content">Hello!</div>
  <div class="message-meta">
    <span class="time">10:32 AM</span>
    <span class="status" title="Queued to send">📤</span>
  </div>
</div>
```

```css
/* Message status indicators */
.message .status {
  font-size: 0.875rem;
}

/* Different states */
.message.sending .status::after { content: '⏳'; }
.message.queued .status::after { content: '📤'; }
.message.sent .status::after { content: '✓'; }
.message.delivered .status::after { content: '✓✓'; }
.message.read .status::after { content: '✓✓'; color: var(--primary-color); }
.message.failed .status::after { content: '❌'; }

/* Visual distinction for unsent messages */
.message.queued,
.message.sending {
  opacity: 0.8;
}

.message.failed {
  border-left: 3px solid var(--danger-color, #dc3545);
}

.message.failed .status {
  color: var(--danger-color, #dc3545);
}
```

### Tooltip for Status Details

```javascript
// messageStatus.js
class MessageStatusTooltip {
  constructor() {
    this.tooltip = this.createTooltip();
    this.setupListeners();
  }
  
  createTooltip() {
    const tooltip = document.createElement('div');
    tooltip.className = 'status-tooltip hidden';
    tooltip.setAttribute('role', 'tooltip');
    document.body.appendChild(tooltip);
    return tooltip;
  }
  
  setupListeners() {
    // Delegate to message list
    document.querySelector('.message-list').addEventListener('mouseenter', (e) => {
      const statusEl = e.target.closest('.message .status');
      if (statusEl) {
        this.show(statusEl);
      }
    }, true);
    
    document.querySelector('.message-list').addEventListener('mouseleave', (e) => {
      const statusEl = e.target.closest('.message .status');
      if (statusEl) {
        this.hide();
      }
    }, true);
  }
  
  show(element) {
    const message = element.closest('.message');
    const status = this.getStatusText(message);
    
    this.tooltip.textContent = status;
    this.tooltip.classList.remove('hidden');
    
    // Position near the element
    const rect = element.getBoundingClientRect();
    this.tooltip.style.top = `${rect.top - 30}px`;
    this.tooltip.style.left = `${rect.left}px`;
  }
  
  hide() {
    this.tooltip.classList.add('hidden');
  }
  
  getStatusText(message) {
    const statuses = {
      'sending': 'Sending...',
      'queued': 'Queued - will send when online',
      'sent': 'Sent',
      'delivered': 'Delivered',
      'read': 'Read',
      'failed': 'Failed to send - tap to retry'
    };
    
    for (const [status, text] of Object.entries(statuses)) {
      if (message.classList.contains(status)) {
        return text;
      }
    }
    
    return 'Unknown status';
  }
}
```

---

## Feature Availability Indicators

Some features don't work offline. Show this clearly:

### Feature Availability Matrix

| Feature | Online | Offline |
|---------|--------|---------|
| View cached messages | ✅ | ✅ |
| Send text messages | ✅ | ✅ (queued) |
| Send images/files | ✅ | ⚠️ (limited) |
| Search messages | ✅ | ✅ (local only) |
| Load more history | ✅ | ❌ |
| Voice messages | ✅ | ❌ |
| Read receipts | ✅ | ❌ |
| New conversation | ✅ | ❌ |

### Disabled Feature UI

```javascript
// featureAvailability.js
class FeatureAvailability {
  constructor() {
    this.features = {
      sendMessage: { offline: true, element: '.send-btn' },
      attachFile: { offline: false, element: '.attach-btn' },
      voiceMessage: { offline: false, element: '.voice-btn' },
      searchMessages: { offline: 'limited', element: '.search-btn' },
      loadHistory: { offline: false, element: '.load-more-btn' },
      newConversation: { offline: false, element: '.new-chat-btn' }
    };
    
    this.setupListeners();
  }
  
  setupListeners() {
    connectionManager.subscribe((event) => {
      this.updateFeatures(event.type === 'online');
    });
    
    // Initial state
    this.updateFeatures(navigator.onLine);
  }
  
  updateFeatures(isOnline) {
    Object.entries(this.features).forEach(([name, config]) => {
      const element = document.querySelector(config.element);
      if (!element) return;
      
      if (isOnline) {
        this.enableFeature(element);
      } else if (config.offline === false) {
        this.disableFeature(element, name);
      } else if (config.offline === 'limited') {
        this.limitFeature(element, name);
      }
    });
  }
  
  enableFeature(element) {
    element.disabled = false;
    element.classList.remove('feature-disabled', 'feature-limited');
    element.removeAttribute('title');
    element.removeAttribute('aria-disabled');
  }
  
  disableFeature(element, featureName) {
    element.disabled = true;
    element.classList.add('feature-disabled');
    element.setAttribute('title', `${this.formatName(featureName)} unavailable offline`);
    element.setAttribute('aria-disabled', 'true');
  }
  
  limitFeature(element, featureName) {
    element.disabled = false;
    element.classList.add('feature-limited');
    element.setAttribute('title', `${this.formatName(featureName)} limited offline`);
  }
  
  formatName(name) {
    return name.replace(/([A-Z])/g, ' $1').trim();
  }
}
```

```css
/* Disabled feature styles */
.feature-disabled {
  opacity: 0.5;
  cursor: not-allowed;
  position: relative;
}

.feature-disabled::after {
  content: '🚫';
  position: absolute;
  top: -5px;
  right: -5px;
  font-size: 0.75rem;
}

.feature-limited {
  position: relative;
}

.feature-limited::after {
  content: '⚠️';
  position: absolute;
  top: -5px;
  right: -5px;
  font-size: 0.75rem;
}

/* Prevent clicks on disabled buttons */
.feature-disabled {
  pointer-events: none;
}
```

### Showing Disabled Feature Explanations

```javascript
// When user clicks a disabled feature
document.querySelectorAll('.feature-disabled').forEach(el => {
  el.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    showToast({
      message: 'This feature requires an internet connection',
      type: 'warning',
      duration: 3000
    });
  });
});
```

---

## Reconnection Feedback

Keep users informed during reconnection attempts:

### Reconnection Progress

```javascript
// reconnectionUI.js
class ReconnectionUI {
  constructor() {
    this.overlay = null;
    this.attemptCount = 0;
    this.maxAttempts = 5;
  }
  
  show() {
    if (this.overlay) return;
    
    this.overlay = document.createElement('div');
    this.overlay.className = 'reconnection-overlay';
    this.overlay.innerHTML = `
      <div class="reconnection-content">
        <div class="reconnection-spinner" aria-hidden="true"></div>
        <h2 class="reconnection-title">Reconnecting</h2>
        <p class="reconnection-status">Attempting to reconnect...</p>
        <p class="reconnection-attempt">Attempt <span class="attempt-current">1</span> of ${this.maxAttempts}</p>
        <button class="reconnection-cancel">Cancel</button>
      </div>
    `;
    
    document.body.appendChild(this.overlay);
    
    this.overlay.querySelector('.reconnection-cancel').onclick = () => {
      this.cancel();
    };
  }
  
  hide() {
    if (this.overlay) {
      this.overlay.remove();
      this.overlay = null;
    }
    this.attemptCount = 0;
  }
  
  updateAttempt(attempt, status = null) {
    this.attemptCount = attempt;
    
    if (!this.overlay) return;
    
    const attemptEl = this.overlay.querySelector('.attempt-current');
    const statusEl = this.overlay.querySelector('.reconnection-status');
    
    if (attemptEl) {
      attemptEl.textContent = attempt;
    }
    
    if (statusEl && status) {
      statusEl.textContent = status;
    }
  }
  
  showFailed() {
    if (!this.overlay) return;
    
    const content = this.overlay.querySelector('.reconnection-content');
    content.innerHTML = `
      <div class="reconnection-icon">❌</div>
      <h2 class="reconnection-title">Connection Failed</h2>
      <p class="reconnection-status">Unable to reconnect after ${this.maxAttempts} attempts</p>
      <div class="reconnection-actions">
        <button class="reconnection-retry">Try Again</button>
        <button class="reconnection-offline">Stay Offline</button>
      </div>
    `;
    
    content.querySelector('.reconnection-retry').onclick = () => {
      this.hide();
      connectionManager.attemptReconnect();
    };
    
    content.querySelector('.reconnection-offline').onclick = () => {
      this.hide();
    };
  }
  
  showSuccess() {
    if (!this.overlay) return;
    
    const content = this.overlay.querySelector('.reconnection-content');
    content.innerHTML = `
      <div class="reconnection-icon">✅</div>
      <h2 class="reconnection-title">Connected!</h2>
      <p class="reconnection-status">Syncing your messages...</p>
    `;
    
    setTimeout(() => this.hide(), 2000);
  }
  
  cancel() {
    connectionManager.cancelReconnect();
    this.hide();
  }
}
```

```css
/* Reconnection overlay */
.reconnection-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.reconnection-content {
  background: var(--surface-color, #fff);
  padding: 2rem;
  border-radius: 1rem;
  text-align: center;
  max-width: 320px;
  width: 90%;
}

.reconnection-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid var(--border-color, #ddd);
  border-top-color: var(--primary-color, #007bff);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

.reconnection-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.reconnection-title {
  margin: 0 0 0.5rem;
  font-size: 1.25rem;
}

.reconnection-status {
  margin: 0 0 0.5rem;
  color: var(--muted-color, #6c757d);
}

.reconnection-attempt {
  margin: 0 0 1.5rem;
  font-size: 0.875rem;
  color: var(--muted-color);
}

.reconnection-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.reconnection-cancel,
.reconnection-retry,
.reconnection-offline {
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: 0.875rem;
}

.reconnection-retry {
  background: var(--primary-color);
  color: white;
  border: none;
}

.reconnection-cancel,
.reconnection-offline {
  background: transparent;
  border: 1px solid var(--border-color);
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

---

## Accessibility Considerations

Offline indicators must be accessible to all users:

### ARIA Live Regions

```html
<!-- Announce connection changes to screen readers -->
<div id="connection-announcer" 
     class="sr-only" 
     role="status" 
     aria-live="polite"
     aria-atomic="true">
</div>
```

```javascript
// connectionAnnouncer.js
class ConnectionAnnouncer {
  constructor() {
    this.announcer = document.getElementById('connection-announcer');
  }
  
  announce(message) {
    // Clear and set new message to trigger announcement
    this.announcer.textContent = '';
    
    // Small delay ensures the change is detected
    setTimeout(() => {
      this.announcer.textContent = message;
    }, 100);
  }
  
  announceOffline() {
    this.announce("You are now offline. Your messages will be saved and sent when you reconnect.");
  }
  
  announceOnline() {
    this.announce("Connection restored. Your messages are being synced.");
  }
  
  announceSyncComplete(count) {
    if (count > 0) {
      this.announce(`Sync complete. ${count} messages sent.`);
    } else {
      this.announce("Sync complete. You're up to date.");
    }
  }
}
```

```css
/* Screen reader only class */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

### Focus Management

```javascript
// When showing connection status banner, manage focus appropriately
class ConnectionStatusUI {
  showOffline() {
    // ... show UI ...
    
    // Don't steal focus - use aria-live for announcement
    // Only move focus if user action triggered it
  }
  
  showError(message) {
    // ... show UI ...
    
    // If there's an action button, consider focusing it
    // for keyboard users after a slight delay
    if (this.actionEl && !this.actionEl.classList.contains('hidden')) {
      // Don't auto-focus - let aria-live announce
      // Focus only makes sense if user was interacting
    }
  }
}
```

### Color and Contrast

```css
/* Don't rely on color alone - use icons and text */
.connection-status.offline {
  background: #fff3cd;
  color: #856404;
  /* Icon provides non-color indicator */
}

/* Ensure sufficient contrast (WCAG AA: 4.5:1 for text) */
.connection-status.offline .status-message {
  /* #856404 on #fff3cd = 4.68:1 ✓ */
}

.connection-status.error .status-message {
  /* #721c24 on #f8d7da = 4.71:1 ✓ */
}
```

### Keyboard Navigation

```javascript
// Ensure offline indicators are keyboard accessible
class ConnectionStatusUI {
  setupKeyboardHandlers() {
    this.container.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.hide();
      }
      
      if (e.key === 'Enter' && e.target === this.actionEl) {
        this.actionEl.click();
      }
    });
  }
}
```

---

## Complete Offline Indicator System

### Putting It All Together

```javascript
// offlineIndicatorSystem.js
import connectionManager from './connectionManager.js';
import syncManager from './syncManager.js';

class OfflineIndicatorSystem {
  constructor() {
    this.banner = new ConnectionStatusUI();
    this.features = new FeatureAvailability();
    this.announcer = new ConnectionAnnouncer();
    this.toast = new ToastNotification();
    
    this.setupListeners();
  }
  
  setupListeners() {
    // Connection changes
    connectionManager.subscribe((event) => {
      if (event.type === 'offline') {
        this.handleOffline();
      } else if (event.type === 'online') {
        this.handleOnline();
      }
    });
    
    // Sync events
    syncManager.subscribe((event) => {
      this.handleSyncEvent(event);
    });
    
    // Message queue changes
    messageQueue.subscribe((event, data) => {
      this.handleQueueEvent(event, data);
    });
  }
  
  handleOffline() {
    // 1. Show banner
    this.banner.showOffline();
    
    // 2. Update feature availability
    this.features.updateFeatures(false);
    
    // 3. Announce to screen readers
    this.announcer.announceOffline();
    
    // 4. Update input state
    this.updateInputState(false);
  }
  
  handleOnline() {
    // 1. Show brief reconnected message
    this.banner.showOnline();
    
    // 2. Restore features
    this.features.updateFeatures(true);
    
    // 3. Announce
    this.announcer.announceOnline();
    
    // 4. Update input
    this.updateInputState(true);
  }
  
  handleSyncEvent(event) {
    switch (event.type) {
      case 'sync-start':
        this.banner.showSyncing();
        break;
        
      case 'sync-progress':
        this.banner.updateProgress(event.data);
        break;
        
      case 'sync-complete':
        this.banner.showOnline();
        this.announcer.announceSyncComplete(event.data?.count || 0);
        break;
        
      case 'sync-error':
        this.banner.showError('Sync failed. Some messages may not be up to date.');
        break;
    }
  }
  
  handleQueueEvent(event, data) {
    if (event === 'enqueue' && !connectionManager.isOnline) {
      // Show toast for queued message
      this.toast.show({
        message: 'Message queued for sending',
        type: 'info',
        duration: 2000
      });
    }
  }
  
  updateInputState(isOnline) {
    const input = document.querySelector('.chat-input');
    const sendBtn = document.querySelector('.send-btn');
    const hint = document.querySelector('.input-hint');
    
    if (isOnline) {
      input.placeholder = 'Type a message...';
      sendBtn.textContent = 'Send';
      sendBtn.classList.remove('offline');
      if (hint) hint.hidden = true;
    } else {
      input.placeholder = 'Type a message (will send when online)';
      sendBtn.textContent = 'Queue';
      sendBtn.classList.add('offline');
      if (hint) {
        hint.textContent = '📤 Messages will be sent when you reconnect';
        hint.hidden = false;
      }
    }
  }
}

export default new OfflineIndicatorSystem();
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show status clearly at all times | Hide connectivity information |
| Keep input enabled offline | Disable all input when offline |
| Explain what will happen | Use vague "offline" messages |
| Show pending message count | Leave users guessing |
| Provide retry options | Make users refresh the page |
| Use icons AND text for status | Rely on color alone |
| Announce changes to screen readers | Only update visual UI |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Aggressive reconnection attempts | Use exponential backoff |
| No visual difference for queued messages | Distinct styling for each state |
| Banner blocks content | Use slideable/dismissible banners |
| Poor color contrast | Test against WCAG guidelines |
| Forgetting screen readers | Use aria-live regions |
| Blocking features unnecessarily | Enable all possible offline functionality |

---

## Hands-on Exercise

### Your Task

Implement an offline indicator system that includes:

1. **Connection status banner** that shows/hides based on connectivity
2. **Message status indicators** for sent, queued, and failed states
3. **Feature availability** that disables certain buttons when offline
4. **Screen reader announcements** for connection changes

### Requirements

1. Banner appears within 1 second of going offline
2. Banner auto-hides 3 seconds after coming back online
3. At least 3 different message statuses visually distinct
4. At least 2 features disabled when offline
5. Screen reader announces connection changes

<details>
<summary>💡 Hints (click to expand)</summary>

- Use CSS transitions for smooth banner show/hide
- `aria-live="polite"` for non-urgent announcements
- Use `classList.toggle()` for easy class switching
- Test with browser DevTools Network throttling

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!-- HTML Structure -->
<div id="connection-banner" class="banner hidden" role="status" aria-live="polite">
  <span class="banner-icon"></span>
  <span class="banner-message"></span>
</div>

<div id="sr-announcer" class="sr-only" aria-live="polite"></div>

<div class="chat">
  <div class="messages">
    <div class="message sent">
      <p>Sent message</p>
      <span class="status">✓</span>
    </div>
    <div class="message queued">
      <p>Queued message</p>
      <span class="status">📤</span>
    </div>
    <div class="message failed">
      <p>Failed message</p>
      <span class="status">❌</span>
    </div>
  </div>
  
  <div class="toolbar">
    <button class="attach-btn" data-online-only>📎 Attach</button>
    <button class="voice-btn" data-online-only>🎤 Voice</button>
    <button class="send-btn">Send</button>
  </div>
</div>
```

```css
/* CSS */
.banner {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  padding: 0.75rem;
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  transition: transform 0.3s ease;
  z-index: 100;
}

.banner.hidden { transform: translateY(-100%); }
.banner.offline { background: #fff3cd; color: #856404; }
.banner.online { background: #d4edda; color: #155724; }

.message { padding: 0.5rem; margin: 0.25rem; border-radius: 0.25rem; }
.message.sent { background: #e3f2fd; }
.message.queued { background: #fff8e1; opacity: 0.8; }
.message.failed { background: #ffebee; border-left: 3px solid #f44336; }

[data-online-only].disabled {
  opacity: 0.5;
  pointer-events: none;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0,0,0,0);
}
```

```javascript
// JavaScript
const banner = document.getElementById('connection-banner');
const announcer = document.getElementById('sr-announcer');
const onlineOnlyBtns = document.querySelectorAll('[data-online-only]');

function updateUI(isOnline) {
  // Banner
  banner.classList.remove('hidden', 'online', 'offline');
  
  if (isOnline) {
    banner.classList.add('online');
    banner.querySelector('.banner-icon').textContent = '✅';
    banner.querySelector('.banner-message').textContent = 'Back online!';
    
    setTimeout(() => banner.classList.add('hidden'), 3000);
    
    announcer.textContent = 'Connection restored';
  } else {
    banner.classList.add('offline');
    banner.querySelector('.banner-icon').textContent = '📡';
    banner.querySelector('.banner-message').textContent = "You're offline";
    
    announcer.textContent = 'You are now offline. Messages will be queued.';
  }
  
  // Features
  onlineOnlyBtns.forEach(btn => {
    btn.classList.toggle('disabled', !isOnline);
    btn.disabled = !isOnline;
  });
}

window.addEventListener('online', () => updateUI(true));
window.addEventListener('offline', () => updateUI(false));

// Initial state
updateUI(navigator.onLine);
```

</details>

---

## Summary

✅ Connection status banners provide immediate visual feedback  
✅ Message status indicators (sent, queued, failed) set correct expectations  
✅ Disable unavailable features rather than showing errors  
✅ Keep input enabled—let users compose messages offline  
✅ Use ARIA live regions for screen reader announcements  
✅ Provide retry options and clear explanations  

**Next:** [Progressive Enhancement](./05-progressive-enhancement.md)

---

<!-- 
Sources Consulted:
- MDN Navigator.onLine: https://developer.mozilla.org/en-US/docs/Web/API/Navigator/onLine
- MDN ARIA live regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
- WCAG 2.1 Use of Color: https://www.w3.org/WAI/WCAG21/Understanding/use-of-color
- web.dev Offline UX: https://web.dev/articles/offline-ux-design-guidelines
-->
