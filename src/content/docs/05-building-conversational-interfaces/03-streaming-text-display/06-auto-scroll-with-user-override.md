---
title: "Auto-Scroll with User Override"
---

# Auto-Scroll with User Override

## Introduction

Auto-scroll is helpful—until the user wants to read earlier content. Fighting for scroll control creates frustration. The solution: detect user intent and pause auto-scroll when they scroll up, resuming when they return to the bottom.

In this lesson, we'll implement intelligent scroll handling that respects user behavior.

### What We'll Cover

- Detecting user scroll intent
- Pausing auto-scroll on user interaction
- "New messages" indicator
- Resume auto-scroll patterns
- State management for scroll behavior

### Prerequisites

- [Scroll Behavior During Streaming](./05-scroll-behavior-during-streaming.md)
- JavaScript scroll events
- React state management

---

## Detecting User Scroll

### Scroll Direction Detection

```javascript
class ScrollDirectionDetector {
  constructor(container) {
    this.container = container;
    this.lastScrollTop = container.scrollTop;
    this.direction = 'none';
    this.callbacks = {
      up: [],
      down: []
    };
    
    this.handleScroll = this.handleScroll.bind(this);
    container.addEventListener('scroll', this.handleScroll, { passive: true });
  }
  
  handleScroll() {
    const currentScrollTop = this.container.scrollTop;
    
    if (currentScrollTop < this.lastScrollTop) {
      this.direction = 'up';
      this.callbacks.up.forEach(cb => cb());
    } else if (currentScrollTop > this.lastScrollTop) {
      this.direction = 'down';
      this.callbacks.down.forEach(cb => cb());
    }
    
    this.lastScrollTop = currentScrollTop;
  }
  
  onScrollUp(callback) {
    this.callbacks.up.push(callback);
  }
  
  onScrollDown(callback) {
    this.callbacks.down.push(callback);
  }
  
  destroy() {
    this.container.removeEventListener('scroll', this.handleScroll);
  }
}
```

### User vs Programmatic Scroll

Distinguish between user scrolling and code-triggered scrolling:

```javascript
class ScrollController {
  constructor(container) {
    this.container = container;
    this.isProgrammaticScroll = false;
    this.userHasScrolledUp = false;
    
    this.handleScroll = this.handleScroll.bind(this);
    container.addEventListener('scroll', this.handleScroll, { passive: true });
  }
  
  handleScroll() {
    if (this.isProgrammaticScroll) {
      // Ignore programmatic scrolls
      return;
    }
    
    // This is a user scroll
    const isAtBottom = this.checkIfAtBottom();
    
    if (!isAtBottom) {
      this.userHasScrolledUp = true;
    } else {
      this.userHasScrolledUp = false;
    }
  }
  
  checkIfAtBottom() {
    const { scrollTop, scrollHeight, clientHeight } = this.container;
    return scrollHeight - scrollTop - clientHeight < 50;
  }
  
  scrollToBottom() {
    this.isProgrammaticScroll = true;
    
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior: 'smooth'
    });
    
    // Reset flag after scroll completes
    setTimeout(() => {
      this.isProgrammaticScroll = false;
    }, 500);  // Approximate scroll duration
  }
  
  shouldAutoScroll() {
    return !this.userHasScrolledUp;
  }
}
```

---

## Pause and Resume Pattern

### Complete Auto-Scroll Manager

```javascript
class AutoScrollManager {
  constructor(container, options = {}) {
    this.container = container;
    this.threshold = options.threshold || 100;  // px from bottom
    this.resumeThreshold = options.resumeThreshold || 50;
    
    this.isEnabled = true;
    this.isPaused = false;
    this.onPauseCallbacks = [];
    this.onResumeCallbacks = [];
    
    this.lastScrollTop = container.scrollTop;
    this.handleScroll = this.handleScroll.bind(this);
    container.addEventListener('scroll', this.handleScroll, { passive: true });
  }
  
  handleScroll() {
    const { scrollTop, scrollHeight, clientHeight } = this.container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    // User scrolled up
    if (scrollTop < this.lastScrollTop && distanceFromBottom > this.threshold) {
      if (!this.isPaused) {
        this.pause();
      }
    }
    
    // User scrolled back to bottom
    if (distanceFromBottom < this.resumeThreshold) {
      if (this.isPaused) {
        this.resume();
      }
    }
    
    this.lastScrollTop = scrollTop;
  }
  
  pause() {
    this.isPaused = true;
    this.onPauseCallbacks.forEach(cb => cb());
  }
  
  resume() {
    this.isPaused = false;
    this.onResumeCallbacks.forEach(cb => cb());
  }
  
  onPause(callback) {
    this.onPauseCallbacks.push(callback);
    return () => {
      this.onPauseCallbacks = this.onPauseCallbacks.filter(cb => cb !== callback);
    };
  }
  
  onResume(callback) {
    this.onResumeCallbacks.push(callback);
    return () => {
      this.onResumeCallbacks = this.onResumeCallbacks.filter(cb => cb !== callback);
    };
  }
  
  scrollToBottom() {
    if (this.isPaused) return;
    
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior: 'smooth'
    });
  }
  
  forceScrollToBottom() {
    this.resume();
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior: 'smooth'
    });
  }
  
  destroy() {
    this.container.removeEventListener('scroll', this.handleScroll);
  }
}
```

---

## "New Messages" Indicator

### Visual Indicator Component

```html
<div class="messages-container">
  <div class="messages-list">
    <!-- messages here -->
  </div>
  
  <button class="new-messages-indicator" hidden>
    ↓ New messages
  </button>
</div>
```

```css
.new-messages-indicator {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 16px;
  background: var(--primary-color, #007bff);
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  transition: opacity 0.2s, transform 0.2s;
  z-index: 10;
}

.new-messages-indicator:hover {
  transform: translateX(-50%) scale(1.05);
}

.new-messages-indicator[hidden] {
  display: none;
}

.new-messages-indicator.animate-in {
  animation: slide-up-fade 0.3s ease-out;
}

@keyframes slide-up-fade {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}
```

### JavaScript Controller

```javascript
class NewMessagesIndicator {
  constructor(container, options = {}) {
    this.container = container;
    this.messageCount = 0;
    
    // Create indicator button
    this.indicator = document.createElement('button');
    this.indicator.className = 'new-messages-indicator';
    this.indicator.hidden = true;
    this.indicator.textContent = '↓ New messages';
    container.appendChild(this.indicator);
    
    // Click to scroll down
    this.indicator.addEventListener('click', () => {
      this.scrollToBottom();
      this.hide();
    });
  }
  
  show(count = 0) {
    this.messageCount = count;
    this.indicator.textContent = count > 0 
      ? `↓ ${count} new message${count > 1 ? 's' : ''}`
      : '↓ New messages';
    this.indicator.hidden = false;
    this.indicator.classList.add('animate-in');
  }
  
  hide() {
    this.indicator.hidden = true;
    this.indicator.classList.remove('animate-in');
    this.messageCount = 0;
  }
  
  increment() {
    this.messageCount++;
    this.indicator.textContent = `↓ ${this.messageCount} new message${this.messageCount > 1 ? 's' : ''}`;
  }
  
  scrollToBottom() {
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior: 'smooth'
    });
  }
}
```

### Integration with Auto-Scroll

```javascript
class ChatScrollController {
  constructor(container) {
    this.autoScroll = new AutoScrollManager(container);
    this.indicator = new NewMessagesIndicator(container);
    
    this.autoScroll.onPause(() => {
      // Will show indicator when new content arrives
    });
    
    this.autoScroll.onResume(() => {
      this.indicator.hide();
    });
  }
  
  onNewContent() {
    if (this.autoScroll.isPaused) {
      this.indicator.show();
      this.indicator.increment();
    } else {
      this.autoScroll.scrollToBottom();
    }
  }
}
```

---

## React Implementation

### Auto-Scroll Hook with Override

```jsx
function useAutoScrollWithOverride(options = {}) {
  const containerRef = useRef(null);
  const [isPaused, setIsPaused] = useState(false);
  const [newMessageCount, setNewMessageCount] = useState(0);
  const lastScrollTopRef = useRef(0);
  
  const threshold = options.threshold || 100;
  
  // Handle scroll events
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      
      // User scrolled up
      if (scrollTop < lastScrollTopRef.current && distanceFromBottom > threshold) {
        setIsPaused(true);
      }
      
      // User scrolled back to bottom
      if (distanceFromBottom < 50) {
        setIsPaused(false);
        setNewMessageCount(0);
      }
      
      lastScrollTopRef.current = scrollTop;
    };
    
    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [threshold]);
  
  // Scroll to bottom function
  const scrollToBottom = useCallback((force = false) => {
    const container = containerRef.current;
    if (!container) return;
    
    if (force || !isPaused) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
      setIsPaused(false);
      setNewMessageCount(0);
    } else {
      setNewMessageCount(prev => prev + 1);
    }
  }, [isPaused]);
  
  return {
    containerRef,
    isPaused,
    newMessageCount,
    scrollToBottom,
    forceScrollToBottom: () => scrollToBottom(true)
  };
}
```

### Messages Container Component

```jsx
function MessagesContainer({ messages, isStreaming }) {
  const {
    containerRef,
    isPaused,
    newMessageCount,
    scrollToBottom,
    forceScrollToBottom
  } = useAutoScrollWithOverride();
  
  // Auto-scroll on new messages
  useEffect(() => {
    scrollToBottom();
  }, [messages.length, scrollToBottom]);
  
  // Auto-scroll during streaming
  useEffect(() => {
    if (isStreaming && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      scrollToBottom();
    }
  }, [messages, isStreaming, scrollToBottom]);
  
  return (
    <div className="messages-wrapper">
      <div ref={containerRef} className="messages-container">
        {messages.map(msg => (
          <Message key={msg.id} {...msg} />
        ))}
      </div>
      
      {isPaused && newMessageCount > 0 && (
        <NewMessagesButton
          count={newMessageCount}
          onClick={forceScrollToBottom}
        />
      )}
    </div>
  );
}

function NewMessagesButton({ count, onClick }) {
  return (
    <button className="new-messages-indicator" onClick={onClick}>
      ↓ {count} new message{count > 1 ? 's' : ''}
    </button>
  );
}
```

### Complete Chat Component

```jsx
function Chat() {
  const { messages, input, handleInputChange, handleSubmit, status } = useChat();
  const isStreaming = status === 'streaming';
  
  return (
    <div className="chat">
      <MessagesContainer 
        messages={messages} 
        isStreaming={isStreaming}
      />
      <InputForm
        value={input}
        onChange={handleInputChange}
        onSubmit={handleSubmit}
        disabled={isStreaming}
      />
    </div>
  );
}
```

---

## Advanced: Scroll State Machine

For complex applications, use a state machine:

```javascript
const ScrollState = {
  AT_BOTTOM: 'at_bottom',
  SCROLLED_UP: 'scrolled_up',
  SCROLLING_DOWN: 'scrolling_down',
  LOCKED: 'locked'  // User explicitly paused
};

class ScrollStateMachine {
  constructor(container) {
    this.container = container;
    this.state = ScrollState.AT_BOTTOM;
    this.newMessageCount = 0;
    this.listeners = new Set();
    
    this.setupScrollListener();
  }
  
  setupScrollListener() {
    let lastScrollTop = this.container.scrollTop;
    
    this.container.addEventListener('scroll', () => {
      const { scrollTop, scrollHeight, clientHeight } = this.container;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      const isAtBottom = distanceFromBottom < 50;
      const scrolledUp = scrollTop < lastScrollTop;
      
      this.transition(isAtBottom, scrolledUp, distanceFromBottom);
      lastScrollTop = scrollTop;
    }, { passive: true });
  }
  
  transition(isAtBottom, scrolledUp, distanceFromBottom) {
    const prevState = this.state;
    
    switch (this.state) {
      case ScrollState.AT_BOTTOM:
        if (scrolledUp && distanceFromBottom > 100) {
          this.state = ScrollState.SCROLLED_UP;
        }
        break;
        
      case ScrollState.SCROLLED_UP:
        if (isAtBottom) {
          this.state = ScrollState.AT_BOTTOM;
          this.newMessageCount = 0;
        } else if (!scrolledUp) {
          this.state = ScrollState.SCROLLING_DOWN;
        }
        break;
        
      case ScrollState.SCROLLING_DOWN:
        if (isAtBottom) {
          this.state = ScrollState.AT_BOTTOM;
          this.newMessageCount = 0;
        } else if (scrolledUp) {
          this.state = ScrollState.SCROLLED_UP;
        }
        break;
        
      case ScrollState.LOCKED:
        // Only explicit unlock changes this state
        break;
    }
    
    if (prevState !== this.state) {
      this.notifyListeners();
    }
  }
  
  onNewContent() {
    if (this.state === ScrollState.AT_BOTTOM) {
      this.scrollToBottom();
    } else {
      this.newMessageCount++;
      this.notifyListeners();
    }
  }
  
  scrollToBottom() {
    this.container.scrollTo({
      top: this.container.scrollHeight,
      behavior: 'smooth'
    });
  }
  
  forceScrollToBottom() {
    this.state = ScrollState.AT_BOTTOM;
    this.newMessageCount = 0;
    this.scrollToBottom();
    this.notifyListeners();
  }
  
  lock() {
    this.state = ScrollState.LOCKED;
    this.notifyListeners();
  }
  
  unlock() {
    this.state = ScrollState.AT_BOTTOM;
    this.forceScrollToBottom();
  }
  
  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  
  notifyListeners() {
    const snapshot = {
      state: this.state,
      newMessageCount: this.newMessageCount,
      shouldAutoScroll: this.state === ScrollState.AT_BOTTOM
    };
    this.listeners.forEach(listener => listener(snapshot));
  }
}
```

### React Hook for State Machine

```jsx
function useScrollStateMachine() {
  const containerRef = useRef(null);
  const stateMachineRef = useRef(null);
  const [scrollState, setScrollState] = useState({
    state: ScrollState.AT_BOTTOM,
    newMessageCount: 0,
    shouldAutoScroll: true
  });
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    stateMachineRef.current = new ScrollStateMachine(containerRef.current);
    
    const unsubscribe = stateMachineRef.current.subscribe(setScrollState);
    
    return () => {
      unsubscribe();
    };
  }, []);
  
  const onNewContent = useCallback(() => {
    stateMachineRef.current?.onNewContent();
  }, []);
  
  const forceScrollToBottom = useCallback(() => {
    stateMachineRef.current?.forceScrollToBottom();
  }, []);
  
  return {
    containerRef,
    ...scrollState,
    onNewContent,
    forceScrollToBottom
  };
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Detect scroll direction | Only check distance from bottom |
| Provide clear resume indicator | Auto-resume without user action |
| Show new message count | Just show "new messages" |
| Use passive scroll listeners | Block scroll events |
| Allow explicit pause/resume | Force auto-scroll always |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Fighting user for scroll position | Pause on any upward scroll |
| Indicator obscures content | Position with padding, fade out of way |
| Resume too eagerly | Require near-bottom position |
| Losing scroll state on re-render | Store in ref, not just state |
| Indicator hard to dismiss | Click-to-scroll-and-dismiss |

---

## Hands-on Exercise

### Your Task

Create a complete scroll management system with:
1. Auto-scroll during streaming
2. Pause when user scrolls up
3. "New messages" indicator
4. Click to resume and scroll to bottom

### Requirements

1. Detect upward scroll > 100px from bottom
2. Show indicator with message count
3. Resume when user scrolls back to bottom
4. Click indicator to force scroll

<details>
<summary>💡 Hints (click to expand)</summary>

- Track `lastScrollTop` to detect direction
- Calculate `distanceFromBottom` for threshold
- Use state for `isPaused` and `newMessageCount`
- Reset count when returning to bottom

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `useAutoScrollWithOverride` hook and `MessagesContainer` component in the React Implementation section above.

</details>

---

## Summary

✅ **Detect scroll direction** to identify user intent  
✅ **Pause auto-scroll** when user scrolls upward  
✅ **Show new message indicator** with count  
✅ **Resume when at bottom** or on explicit action  
✅ **State machine approach** for complex scenarios  
✅ **Passive listeners** for performance

---

## Further Reading

- [Passive Event Listeners](https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#using_passive_listeners)
- [XState for State Machines](https://xstate.js.org/)
- [React useCallback](https://react.dev/reference/react/useCallback)

---

**Previous:** [Scroll Behavior During Streaming](./05-scroll-behavior-during-streaming.md)  
**Next:** [AI SDK Streaming States](./07-ai-sdk-streaming-states.md)

<!-- 
Sources Consulted:
- MDN Passive Event Listeners: https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener
- React useCallback: https://react.dev/reference/react/useCallback
- Vercel AI SDK: https://sdk.vercel.ai/docs
-->
