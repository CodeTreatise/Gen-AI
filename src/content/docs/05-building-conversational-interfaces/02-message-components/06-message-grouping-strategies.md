---
title: "Message Grouping Strategies"
---

# Message Grouping Strategies

## Introduction

In long conversations, individual message bubbles can become visually overwhelming. Message grouping consolidates related messages into cohesive visual units, reducing clutter and improving readability.

In this lesson, we'll explore different grouping strategies and implement them for chat interfaces.

### What We'll Cover

- Time-based message grouping
- Sender-based consecutive grouping
- Topic and context grouping
- Visual grouping techniques
- Accessibility considerations
- Performance optimization for large conversations

### Prerequisites

- [Message Container Structure](./01-message-container-structure.md)
- [User Message Styling](./02-user-message-styling.md)
- CSS Flexbox and positioning

---

## Why Group Messages?

### Benefits of Grouping

| Benefit | Description |
|---------|-------------|
| **Reduced visual noise** | Fewer repeated elements (avatars, timestamps) |
| **Faster scanning** | Users can quickly identify conversation sections |
| **Clearer context** | Related messages stay visually connected |
| **Improved performance** | Less DOM rendering with consolidated groups |
| **Better accessibility** | Semantic grouping aids screen readers |

### Grouping Strategies Comparison

| Strategy | Best For | Example |
|----------|----------|---------|
| **Time-based** | Separating conversation sessions | "Today", "Yesterday" |
| **Sender-based** | Consecutive messages from same person | Multiple user questions |
| **Topic-based** | Multi-turn focused discussions | "About your code question..." |
| **Hybrid** | Complex conversations | Combine time + sender grouping |

---

## Time-Based Grouping

### Date Separators

```css
.date-separator {
  display: flex;
  align-items: center;
  margin: 1.5rem 0;
  gap: 1rem;
}

.date-separator-line {
  flex: 1;
  height: 1px;
  background: #e5e7eb;
}

.date-separator-text {
  padding: 0.375rem 0.75rem;
  background: #f3f4f6;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: #6b7280;
  white-space: nowrap;
}
```

```html
<div class="date-separator" role="separator" aria-label="January 29, 2026">
  <div class="date-separator-line" aria-hidden="true"></div>
  <span class="date-separator-text">Today</span>
  <div class="date-separator-line" aria-hidden="true"></div>
</div>
```

### Time Gap Logic

```javascript
function shouldShowTimeSeparator(currentMsg, previousMsg) {
  if (!previousMsg) return true; // First message
  
  const current = new Date(currentMsg.createdAt);
  const previous = new Date(previousMsg.createdAt);
  
  // Show separator if gap > 1 hour
  const hourGap = (current - previous) / (1000 * 60 * 60);
  if (hourGap >= 1) return true;
  
  // Show separator if different day
  if (current.toDateString() !== previous.toDateString()) return true;
  
  return false;
}

function formatTimeSeparator(date) {
  const now = new Date();
  const messageDate = new Date(date);
  
  const isToday = messageDate.toDateString() === now.toDateString();
  const isYesterday = messageDate.toDateString() === 
    new Date(now - 86400000).toDateString();
  
  if (isToday) return 'Today';
  if (isYesterday) return 'Yesterday';
  
  const daysDiff = Math.floor((now - messageDate) / 86400000);
  
  if (daysDiff < 7) {
    return messageDate.toLocaleDateString(undefined, { weekday: 'long' });
  }
  
  return messageDate.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: messageDate.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  });
}
```

### React Implementation

```jsx
function MessageList({ messages }) {
  return (
    <div className="message-list" role="log" aria-label="Conversation">
      {messages.map((message, index) => {
        const previousMessage = messages[index - 1];
        const showTimeSeparator = shouldShowTimeSeparator(message, previousMessage);
        
        return (
          <React.Fragment key={message.id}>
            {showTimeSeparator && (
              <DateSeparator date={message.createdAt} />
            )}
            <Message message={message} />
          </React.Fragment>
        );
      })}
    </div>
  );
}

function DateSeparator({ date }) {
  const label = formatTimeSeparator(date);
  const fullDate = new Date(date).toLocaleDateString(undefined, {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
  
  return (
    <div 
      className="date-separator" 
      role="separator" 
      aria-label={fullDate}
    >
      <div className="date-separator-line" aria-hidden="true" />
      <span className="date-separator-text">{label}</span>
      <div className="date-separator-line" aria-hidden="true" />
    </div>
  );
}
```

---

## Sender-Based Grouping

### Visual Grouping for Consecutive Messages

```css
.message-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;  /* Tighter spacing within group */
  margin-bottom: 0.75rem;  /* Larger gap between groups */
}

.message-group.user {
  align-items: flex-end;
}

.message-group.assistant {
  align-items: flex-start;
}

/* First message in group shows avatar */
.message-group .message-wrapper:first-child .message-avatar {
  display: flex;
}

/* Subsequent messages hide avatar but keep spacing */
.message-group .message-wrapper:not(:first-child) .message-avatar {
  visibility: hidden;
}

/* Adjust border radius for grouped messages */
.message-group.user .message-wrapper:first-child .message-container {
  border-radius: 1.25rem 1.25rem 0.375rem 1.25rem;
}

.message-group.user .message-wrapper:not(:first-child):not(:last-child) .message-container {
  border-radius: 1.25rem 0.375rem 0.375rem 1.25rem;
}

.message-group.user .message-wrapper:last-child .message-container {
  border-radius: 1.25rem 0.375rem 1.25rem 1.25rem;
}

/* Single message in group */
.message-group.user .message-wrapper:only-child .message-container {
  border-radius: 1.25rem 1.25rem 0.375rem 1.25rem;
}
```

### Grouping Logic

```javascript
function groupMessages(messages) {
  const groups = [];
  let currentGroup = null;
  
  for (const message of messages) {
    const shouldStartNewGroup = 
      !currentGroup ||
      message.role !== currentGroup.role ||
      shouldShowTimeSeparator(message, currentGroup.messages.at(-1));
    
    if (shouldStartNewGroup) {
      currentGroup = {
        id: `group-${message.id}`,
        role: message.role,
        messages: [message],
        startTime: message.createdAt
      };
      groups.push(currentGroup);
    } else {
      currentGroup.messages.push(message);
    }
  }
  
  return groups;
}
```

### Grouped Message Component

```jsx
function GroupedMessageList({ messages }) {
  const groups = useMemo(() => groupMessages(messages), [messages]);
  
  return (
    <div className="message-list" role="log">
      {groups.map((group, groupIndex) => {
        const previousGroup = groups[groupIndex - 1];
        const showTimeSeparator = !previousGroup || 
          shouldShowTimeSeparator(group.messages[0], previousGroup.messages.at(-1));
        
        return (
          <React.Fragment key={group.id}>
            {showTimeSeparator && (
              <DateSeparator date={group.startTime} />
            )}
            <MessageGroup group={group} />
          </React.Fragment>
        );
      })}
    </div>
  );
}

function MessageGroup({ group }) {
  return (
    <div 
      className={`message-group ${group.role}`}
      role="group"
      aria-label={`${group.messages.length} messages from ${group.role}`}
    >
      {group.messages.map((message, index) => (
        <Message 
          key={message.id}
          message={message}
          isFirstInGroup={index === 0}
          isLastInGroup={index === group.messages.length - 1}
          showAvatar={index === 0}
          showTimestamp={index === group.messages.length - 1}
        />
      ))}
    </div>
  );
}
```

---

## Topic and Context Grouping

### Collapsible Thread Groups

For multi-turn discussions on a specific topic:

```css
.topic-group {
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  margin: 1rem 0;
  overflow: hidden;
}

.topic-group-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: #f9fafb;
  cursor: pointer;
}

.topic-group-header:hover {
  background: #f3f4f6;
}

.topic-group-icon {
  width: 1.25rem;
  height: 1.25rem;
  color: #6b7280;
  transition: transform 0.2s ease;
}

.topic-group[open] .topic-group-icon {
  transform: rotate(90deg);
}

.topic-group-title {
  flex: 1;
  font-weight: 500;
  color: #374151;
}

.topic-group-count {
  padding: 0.125rem 0.5rem;
  background: #e5e7eb;
  border-radius: 1rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.topic-group-content {
  padding: 0.5rem;
  background: white;
}
```

```html
<details class="topic-group" open>
  <summary class="topic-group-header">
    <svg class="topic-group-icon"><!-- chevron-right --></svg>
    <span class="topic-group-title">Debugging the authentication flow</span>
    <span class="topic-group-count">5 messages</span>
  </summary>
  <div class="topic-group-content">
    <!-- Messages here -->
  </div>
</details>
```

### AI-Detected Topics

```javascript
// Use AI to detect topic changes
async function detectTopicChange(messages) {
  const lastFewMessages = messages.slice(-5);
  
  const response = await fetch('/api/detect-topic', {
    method: 'POST',
    body: JSON.stringify({
      messages: lastFewMessages.map(m => ({
        role: m.role,
        content: m.content
      })),
      currentTopic: messages.currentTopic
    })
  });
  
  return response.json();
  // Returns: { isNewTopic: true, topicTitle: "Database optimization" }
}

// Simpler heuristic approach
function detectTopicHeuristic(message, previousMessage) {
  if (!previousMessage) return null;
  
  // Check for topic indicators
  const topicIndicators = [
    /(?:now|let's|moving on|next|about|regarding)\s+(?:talk|discuss|look at)/i,
    /(?:different|new|another)\s+(?:topic|question|issue)/i,
    /^(?:ok|okay|alright)[,.]?\s+/i
  ];
  
  for (const pattern of topicIndicators) {
    if (pattern.test(message.content)) {
      return extractTopicTitle(message.content);
    }
  }
  
  return null;
}
```

---

## Visual Grouping Techniques

### Connected Bubble Trail

```css
/* Messages connected with a visual line */
.message-group.connected {
  position: relative;
  padding-left: 3rem;
}

.message-group.connected::before {
  content: '';
  position: absolute;
  left: 1.125rem;
  top: 2.5rem;
  bottom: 0.5rem;
  width: 2px;
  background: #e5e7eb;
}

.message-group.connected .message-avatar {
  position: relative;
  z-index: 1;
}

.message-group.connected .message-wrapper:not(:first-child) {
  margin-left: 0;
}
```

### Color-Coded Groups

```css
.message-group[data-topic="code"] .message-container {
  border-left: 3px solid #3b82f6;
}

.message-group[data-topic="design"] .message-container {
  border-left: 3px solid #8b5cf6;
}

.message-group[data-topic="debug"] .message-container {
  border-left: 3px solid #ef4444;
}
```

### Compact Group View

```css
.message-group.compact .message-wrapper {
  gap: 0.5rem;
}

.message-group.compact .message-container {
  padding: 0.5rem 0.75rem;
}

.message-group.compact .message-header,
.message-group.compact .message-footer {
  display: none;
}

/* Show header only on first, footer only on last */
.message-group.compact .message-wrapper:first-child .message-header {
  display: flex;
}

.message-group.compact .message-wrapper:last-child .message-footer {
  display: flex;
}
```

---

## Accessibility Considerations

### Semantic Structure

```html
<!-- Use appropriate ARIA roles -->
<div class="message-list" role="log" aria-label="Conversation">
  
  <!-- Date separator with proper role -->
  <div role="separator" aria-label="Today, January 29, 2026">
    Today
  </div>
  
  <!-- Message group -->
  <div 
    class="message-group" 
    role="group"
    aria-label="3 messages from you"
  >
    <article class="message-wrapper user" aria-label="Your message at 2:30 PM">
      <!-- message content -->
    </article>
    <article class="message-wrapper user" aria-label="Your message at 2:31 PM">
      <!-- message content -->
    </article>
  </div>
  
</div>
```

### Screen Reader Announcements

```jsx
function MessageGroup({ group }) {
  const description = `${group.messages.length} ${
    group.messages.length === 1 ? 'message' : 'messages'
  } from ${group.role === 'user' ? 'you' : 'the assistant'}`;
  
  return (
    <div 
      className={`message-group ${group.role}`}
      role="group"
      aria-label={description}
    >
      <span className="sr-only">
        {description}, starting at{' '}
        {formatTime(group.messages[0].createdAt)}
      </span>
      {group.messages.map((message, index) => (
        <Message 
          key={message.id}
          message={message}
          // Only announce time for first and last
          announceTime={index === 0 || index === group.messages.length - 1}
        />
      ))}
    </div>
  );
}
```

### Keyboard Navigation

```javascript
function setupGroupNavigation(container) {
  const groups = container.querySelectorAll('.message-group');
  
  container.addEventListener('keydown', (e) => {
    if (!e.target.closest('.message-group')) return;
    
    const currentGroup = e.target.closest('.message-group');
    const groupsArray = Array.from(groups);
    const currentIndex = groupsArray.indexOf(currentGroup);
    
    if (e.key === 'ArrowUp' && e.altKey) {
      // Jump to previous group
      e.preventDefault();
      const prevGroup = groupsArray[currentIndex - 1];
      prevGroup?.querySelector('.message-container')?.focus();
    }
    
    if (e.key === 'ArrowDown' && e.altKey) {
      // Jump to next group
      e.preventDefault();
      const nextGroup = groupsArray[currentIndex + 1];
      nextGroup?.querySelector('.message-container')?.focus();
    }
  });
}
```

---

## Performance Optimization

### Virtualized Grouped List

```jsx
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedGroupedMessages({ messages }) {
  const parentRef = useRef(null);
  
  // Pre-compute groups
  const groups = useMemo(() => groupMessages(messages), [messages]);
  
  const virtualizer = useVirtualizer({
    count: groups.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => {
      // Estimate height based on message count
      const group = groups[index];
      const baseHeight = 60;
      const messageHeight = 80;
      return baseHeight + (group.messages.length * messageHeight);
    },
    overscan: 3,
  });
  
  return (
    <div ref={parentRef} className="message-list-container">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => {
          const group = groups[virtualItem.index];
          return (
            <div
              key={group.id}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              <MessageGroup group={group} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

### Lazy Rendering Groups

```jsx
function LazyMessageGroup({ group, isVisible }) {
  const [hasRendered, setHasRendered] = useState(false);
  
  useEffect(() => {
    if (isVisible && !hasRendered) {
      setHasRendered(true);
    }
  }, [isVisible, hasRendered]);
  
  if (!hasRendered) {
    // Placeholder with estimated height
    return (
      <div 
        className="message-group-placeholder"
        style={{ height: `${group.messages.length * 80}px` }}
        aria-label={`${group.messages.length} messages loading`}
      />
    );
  }
  
  return <MessageGroup group={group} />;
}
```

### Incremental Group Computation

```javascript
function useIncrementalGroups(messages) {
  const [groups, setGroups] = useState([]);
  const lastProcessedIndex = useRef(0);
  
  useEffect(() => {
    if (messages.length === lastProcessedIndex.current) return;
    
    const newMessages = messages.slice(lastProcessedIndex.current);
    
    setGroups(prevGroups => {
      const updatedGroups = [...prevGroups];
      
      for (const message of newMessages) {
        const lastGroup = updatedGroups[updatedGroups.length - 1];
        
        if (lastGroup && canAddToGroup(lastGroup, message)) {
          lastGroup.messages.push(message);
        } else {
          updatedGroups.push({
            id: `group-${message.id}`,
            role: message.role,
            messages: [message],
            startTime: message.createdAt
          });
        }
      }
      
      return updatedGroups;
    });
    
    lastProcessedIndex.current = messages.length;
  }, [messages]);
  
  return groups;
}
```

---

## Complete Implementation

```jsx
// Complete MessageGroupingSystem.jsx
function MessageGroupingSystem({ messages, groupingStrategy = 'sender' }) {
  const groups = useMemo(() => {
    switch (groupingStrategy) {
      case 'time':
        return groupByTime(messages);
      case 'sender':
        return groupBySender(messages);
      case 'topic':
        return groupByTopic(messages);
      default:
        return groupBySender(messages);
    }
  }, [messages, groupingStrategy]);
  
  return (
    <div className="message-list" role="log" aria-label="Conversation">
      {groups.map((group, index) => {
        const previousGroup = groups[index - 1];
        const showDateSeparator = shouldShowDateSeparator(group, previousGroup);
        
        return (
          <React.Fragment key={group.id}>
            {showDateSeparator && (
              <DateSeparator date={group.startTime} />
            )}
            
            {groupingStrategy === 'topic' ? (
              <TopicGroup group={group} />
            ) : (
              <MessageGroup 
                group={group}
                showAvatar={groupingStrategy === 'sender'}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Group consecutive same-sender messages | Show avatar for every message |
| Add date separators for clarity | Display timestamps on every message |
| Use semantic grouping with ARIA | Lose context with purely visual grouping |
| Maintain grouping during scroll | Re-compute groups on every render |
| Provide keyboard navigation between groups | Trap focus within groups |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Re-grouping on every message | Use incremental group computation |
| Groups too large (20+ messages) | Add time-based breaks within groups |
| Losing group context when streaming | Update last group incrementally |
| No visual distinction between groups | Add spacing, separators, or borders |
| Screen readers can't navigate groups | Add `role="group"` with descriptions |

---

## Hands-on Exercise

### Your Task

Implement a message grouping system with:
1. Time-based separators (Today, Yesterday, Date)
2. Sender-based consecutive grouping
3. Accessible group structure
4. Optimized for 100+ messages

### Requirements

1. Messages within 5 minutes group together
2. Avatar shows only on first message in group
3. Timestamp shows only on last message
4. Date separator between different days
5. ARIA labels for screen readers

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `useMemo` to avoid re-computing groups
- Track group boundaries with CSS `:first-child` and `:last-child`
- Use `role="group"` with `aria-label`
- Consider virtualization for 100+ messages

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the complete implementation in the sections above, combining `groupMessages()`, `MessageGroup`, and `DateSeparator` components.

</details>

---

## Summary

✅ **Time-based grouping** separates conversation sessions  
✅ **Sender-based grouping** reduces visual repetition  
✅ **Topic grouping** organizes multi-turn discussions  
✅ **Visual techniques** create cohesive message blocks  
✅ **ARIA roles** enable accessible navigation  
✅ **Performance optimization** handles long conversations

---

## Further Reading

- [TanStack Virtual](https://tanstack.com/virtual/latest)
- [ARIA Log Role](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/log_role)
- [Slack Message Grouping](https://slack.design/articles/creating-order-from-chaos-in-slack-message-groups/)

---

**Previous:** [Error Message Display](./05-error-message-display.md)  
**Next:** [AI-Generated File Display](./07-ai-generated-file-display.md)

<!-- 
Sources Consulted:
- TanStack Virtual: https://tanstack.com/virtual/latest
- MDN ARIA Log Role: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/log_role
-->
