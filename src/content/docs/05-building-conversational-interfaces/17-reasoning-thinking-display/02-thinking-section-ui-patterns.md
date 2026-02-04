---
title: "Thinking Section UI Patterns"
---

# Thinking Section UI Patterns

## Introduction

How you present reasoning tokens significantly impacts user experience. A wall of unformatted thinking text can overwhelm users, while a well-designed thinking section enhances transparency without cluttering the interface.

This lesson covers UI patterns for thinking sections: collapsible designs, toggle buttons, thinking duration display, and token count visualization. These patterns help users understand AI reasoning at their own pace.

### What We'll Cover

- Collapsible/expandable section designs
- "Show thinking" toggle patterns
- Thinking duration display
- Token count visualization
- Animation and transition effects
- Accessibility considerations

### Prerequisites

- [Rendering Reasoning Tokens](./01-rendering-reasoning-tokens.md)
- CSS transitions and animations
- React state management basics

---

## Collapsible Section Patterns

### Pattern 1: Accordion Style

The accordion pattern shows thinking in a dedicated collapsible panel:

```tsx
import { useState } from 'react';

interface ThinkingAccordionProps {
  thinking: string;
  defaultOpen?: boolean;
}

export function ThinkingAccordion({ 
  thinking, 
  defaultOpen = false 
}: ThinkingAccordionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <div className="thinking-accordion">
      <button
        className="accordion-header"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-controls="thinking-content"
      >
        <div className="header-left">
          <span className="accordion-icon" data-open={isOpen}>
            ▶
          </span>
          <span className="header-title">💭 Thinking Process</span>
        </div>
        <span className="header-badge">
          {thinking.split(/\s+/).length} words
        </span>
      </button>
      
      <div
        id="thinking-content"
        className="accordion-content"
        data-open={isOpen}
        aria-hidden={!isOpen}
      >
        <pre>{thinking}</pre>
      </div>
    </div>
  );
}
```

```css
.thinking-accordion {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 16px;
}

.accordion-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 14px 18px;
  background: linear-gradient(to right, #faf5ff, #f5f3ff);
  border: none;
  cursor: pointer;
  transition: background 0.2s;
}

.accordion-header:hover {
  background: linear-gradient(to right, #f3e8ff, #ede9fe);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.accordion-icon {
  font-size: 0.7rem;
  transition: transform 0.3s ease;
  color: #7c3aed;
}

.accordion-icon[data-open="true"] {
  transform: rotate(90deg);
}

.header-title {
  font-weight: 600;
  color: #6d28d9;
}

.header-badge {
  font-size: 0.75rem;
  color: #9ca3af;
  background: white;
  padding: 4px 10px;
  border-radius: 9999px;
}

.accordion-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease-out;
}

.accordion-content[data-open="true"] {
  max-height: 500px;
  overflow-y: auto;
}

.accordion-content pre {
  margin: 0;
  padding: 16px 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
  white-space: pre-wrap;
  color: #6b7280;
  background: #fafafa;
}
```

### Pattern 2: Inline Toggle

For minimal UI, use an inline toggle that reveals thinking below:

```tsx
import { useState } from 'react';

export function InlineThinkingToggle({ thinking }: { thinking: string }) {
  const [showThinking, setShowThinking] = useState(false);
  
  return (
    <div className="inline-thinking">
      <button
        className="inline-toggle"
        onClick={() => setShowThinking(!showThinking)}
      >
        {showThinking ? '🔽 Hide thinking' : '💭 Show thinking'}
      </button>
      
      {showThinking && (
        <div className="inline-content">
          <pre>{thinking}</pre>
        </div>
      )}
    </div>
  );
}
```

```css
.inline-thinking {
  margin-bottom: 12px;
}

.inline-toggle {
  background: none;
  border: 1px dashed #d1d5db;
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 0.8rem;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s;
}

.inline-toggle:hover {
  border-color: #9ca3af;
  color: #4b5563;
}

.inline-content {
  margin-top: 12px;
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}

.inline-content pre {
  margin: 0;
  font-size: 0.85rem;
  white-space: pre-wrap;
  color: #6b7280;
}
```

### Pattern 3: Tab-Style Toggle

For interfaces where thinking and response should be equally accessible:

```tsx
import { useState } from 'react';

type TabType = 'response' | 'thinking';

interface TabbedMessageProps {
  thinking: string;
  response: string;
}

export function TabbedMessage({ thinking, response }: TabbedMessageProps) {
  const [activeTab, setActiveTab] = useState<TabType>('response');
  
  return (
    <div className="tabbed-message">
      <div className="tab-header" role="tablist">
        <button
          role="tab"
          aria-selected={activeTab === 'response'}
          className={`tab ${activeTab === 'response' ? 'active' : ''}`}
          onClick={() => setActiveTab('response')}
        >
          Response
        </button>
        <button
          role="tab"
          aria-selected={activeTab === 'thinking'}
          className={`tab ${activeTab === 'thinking' ? 'active' : ''}`}
          onClick={() => setActiveTab('thinking')}
        >
          💭 Thinking
        </button>
      </div>
      
      <div className="tab-content" role="tabpanel">
        {activeTab === 'response' ? (
          <div className="response-panel">{response}</div>
        ) : (
          <pre className="thinking-panel">{thinking}</pre>
        )}
      </div>
    </div>
  );
}
```

```css
.tabbed-message {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
}

.tab-header {
  display: flex;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.tab {
  flex: 1;
  padding: 12px 16px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 0.875rem;
  color: #6b7280;
  transition: all 0.2s;
  position: relative;
}

.tab:hover {
  background: #f3f4f6;
}

.tab.active {
  color: #6d28d9;
  font-weight: 600;
}

.tab.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 2px;
  background: #6d28d9;
}

.tab-content {
  padding: 16px;
  min-height: 100px;
}

.thinking-panel {
  margin: 0;
  font-family: monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
  color: #6b7280;
}

.response-panel {
  line-height: 1.7;
}
```

---

## Thinking Duration Display

Showing how long the model spent thinking helps users understand processing complexity.

### Calculating Duration

Track thinking duration using timestamps:

```tsx
import { useChat } from '@ai-sdk/react';
import { useRef, useState, useEffect } from 'react';

export function ChatWithDuration() {
  const { messages, status } = useChat();
  const [thinkingDuration, setThinkingDuration] = useState<number | null>(null);
  const thinkingStartRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (status === 'streaming') {
      // Start timing when streaming begins
      if (thinkingStartRef.current === null) {
        thinkingStartRef.current = Date.now();
      }
    }
    
    if (status === 'ready' && thinkingStartRef.current !== null) {
      // Calculate duration when complete
      const duration = Date.now() - thinkingStartRef.current;
      setThinkingDuration(duration);
      thinkingStartRef.current = null;
    }
  }, [status]);
  
  return (
    <div>
      {/* Messages */}
      {messages.map(message => (
        <Message key={message.id} message={message} />
      ))}
      
      {/* Duration display */}
      {thinkingDuration && (
        <div className="duration-badge">
          Thought for {formatDuration(thinkingDuration)}
        </div>
      )}
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = (ms / 1000).toFixed(1);
  return `${seconds}s`;
}
```

### Duration Badge Component

```tsx
interface DurationBadgeProps {
  durationMs: number;
  tokenCount?: number;
}

export function DurationBadge({ durationMs, tokenCount }: DurationBadgeProps) {
  const seconds = (durationMs / 1000).toFixed(1);
  
  return (
    <div className="duration-badge">
      <span className="duration-icon">⏱️</span>
      <span className="duration-text">
        Thought for {seconds}s
      </span>
      {tokenCount && (
        <>
          <span className="separator">•</span>
          <span className="token-text">
            {tokenCount.toLocaleString()} reasoning tokens
          </span>
        </>
      )}
    </div>
  );
}
```

```css
.duration-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 9999px;
  font-size: 0.75rem;
  color: #166534;
}

.duration-icon {
  font-size: 0.875rem;
}

.separator {
  color: #86efac;
}

.token-text {
  color: #15803d;
}
```

### Animated Timer (During Thinking)

Show a live counter while thinking:

```tsx
import { useState, useEffect } from 'react';

export function LiveThinkingTimer({ isThinking }: { isThinking: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  
  useEffect(() => {
    if (!isThinking) {
      setElapsed(0);
      return;
    }
    
    const startTime = Date.now();
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime);
    }, 100);
    
    return () => clearInterval(interval);
  }, [isThinking]);
  
  if (!isThinking) return null;
  
  return (
    <div className="live-timer">
      <span className="thinking-spinner" />
      <span>Thinking... {(elapsed / 1000).toFixed(1)}s</span>
    </div>
  );
}
```

```css
.live-timer {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: #fef3c7;
  border-radius: 8px;
  font-size: 0.875rem;
  color: #92400e;
}

.thinking-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid #fcd34d;
  border-top-color: #f59e0b;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

---

## Token Count Visualization

### Simple Token Counter

```tsx
interface TokenDisplayProps {
  reasoningTokens: number;
  outputTokens: number;
}

export function TokenDisplay({ reasoningTokens, outputTokens }: TokenDisplayProps) {
  const total = reasoningTokens + outputTokens;
  const reasoningPercent = (reasoningTokens / total) * 100;
  
  return (
    <div className="token-display">
      <div className="token-bar">
        <div 
          className="reasoning-segment"
          style={{ width: `${reasoningPercent}%` }}
        />
        <div 
          className="output-segment"
          style={{ width: `${100 - reasoningPercent}%` }}
        />
      </div>
      
      <div className="token-legend">
        <span className="legend-item reasoning">
          <span className="dot" />
          Thinking: {reasoningTokens.toLocaleString()}
        </span>
        <span className="legend-item output">
          <span className="dot" />
          Output: {outputTokens.toLocaleString()}
        </span>
      </div>
    </div>
  );
}
```

```css
.token-display {
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
}

.token-bar {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.reasoning-segment {
  background: linear-gradient(90deg, #a78bfa, #8b5cf6);
  transition: width 0.3s;
}

.output-segment {
  background: linear-gradient(90deg, #60a5fa, #3b82f6);
  transition: width 0.3s;
}

.token-legend {
  display: flex;
  gap: 16px;
  font-size: 0.75rem;
  color: #6b7280;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-item .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.legend-item.reasoning .dot {
  background: #8b5cf6;
}

.legend-item.output .dot {
  background: #3b82f6;
}
```

### Detailed Token Breakdown

```tsx
interface TokenBreakdownProps {
  usage: {
    inputTokens: number;
    outputTokens: number;
    reasoningTokens: number;
  };
}

export function TokenBreakdown({ usage }: TokenBreakdownProps) {
  const total = usage.inputTokens + usage.outputTokens;
  
  return (
    <details className="token-breakdown">
      <summary>
        📊 Token usage: {total.toLocaleString()} total
      </summary>
      
      <table className="token-table">
        <tbody>
          <tr>
            <td>Input tokens</td>
            <td className="value">{usage.inputTokens.toLocaleString()}</td>
          </tr>
          <tr>
            <td>Output tokens</td>
            <td className="value">{usage.outputTokens.toLocaleString()}</td>
          </tr>
          <tr className="reasoning-row">
            <td>↳ Reasoning tokens</td>
            <td className="value">{usage.reasoningTokens.toLocaleString()}</td>
          </tr>
          <tr className="total-row">
            <td>Total</td>
            <td className="value">{total.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>
    </details>
  );
}
```

```css
.token-breakdown {
  margin-top: 12px;
  font-size: 0.8rem;
}

.token-breakdown summary {
  cursor: pointer;
  color: #6b7280;
  padding: 8px;
  border-radius: 6px;
}

.token-breakdown summary:hover {
  background: #f3f4f6;
}

.token-table {
  width: 100%;
  margin-top: 8px;
  border-collapse: collapse;
}

.token-table td {
  padding: 6px 12px;
  border-bottom: 1px solid #f3f4f6;
}

.token-table .value {
  text-align: right;
  font-family: monospace;
  color: #374151;
}

.reasoning-row {
  color: #8b5cf6;
  font-style: italic;
}

.total-row {
  font-weight: 600;
  border-top: 2px solid #e5e7eb;
}

.total-row td {
  border-bottom: none;
}
```

---

## Animation and Transition Effects

### Smooth Height Transitions

For accordion-style animations without fixed max-height:

```tsx
import { useRef, useEffect, useState } from 'react';

export function SmoothCollapse({ 
  isOpen, 
  children 
}: { 
  isOpen: boolean; 
  children: React.ReactNode;
}) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | undefined>(0);
  
  useEffect(() => {
    if (contentRef.current) {
      setHeight(isOpen ? contentRef.current.scrollHeight : 0);
    }
  }, [isOpen]);
  
  return (
    <div 
      className="smooth-collapse"
      style={{ height: height !== undefined ? `${height}px` : 'auto' }}
    >
      <div ref={contentRef}>
        {children}
      </div>
    </div>
  );
}
```

```css
.smooth-collapse {
  overflow: hidden;
  transition: height 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

### Thinking Pulse Animation

Visual feedback during thinking:

```tsx
export function ThinkingPulse({ isActive }: { isActive: boolean }) {
  if (!isActive) return null;
  
  return (
    <div className="thinking-pulse">
      <span className="pulse-dot" />
      <span className="pulse-dot" />
      <span className="pulse-dot" />
      <span className="pulse-text">Thinking</span>
    </div>
  );
}
```

```css
.thinking-pulse {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 16px;
  background: #faf5ff;
  border-radius: 9999px;
}

.pulse-dot {
  width: 6px;
  height: 6px;
  background: #a78bfa;
  border-radius: 50%;
  animation: pulse 1.4s ease-in-out infinite;
}

.pulse-dot:nth-child(2) {
  animation-delay: 0.2s;
}

.pulse-dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.4;
    transform: scale(0.8);
  }
  50% {
    opacity: 1;
    transform: scale(1);
  }
}

.pulse-text {
  margin-left: 8px;
  font-size: 0.8rem;
  color: #7c3aed;
}
```

---

## Accessibility Considerations

### ARIA Patterns

Ensure thinking sections are accessible:

```tsx
export function AccessibleThinkingSection({
  thinking,
  isExpanded,
  onToggle,
}: {
  thinking: string;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const contentId = 'thinking-content';
  
  return (
    <div 
      className="thinking-section"
      role="region"
      aria-label="AI thinking process"
    >
      <button
        aria-expanded={isExpanded}
        aria-controls={contentId}
        onClick={onToggle}
        className="toggle-button"
      >
        <span aria-hidden="true">
          {isExpanded ? '▼' : '▶'}
        </span>
        <span>
          {isExpanded ? 'Hide' : 'Show'} thinking process
        </span>
      </button>
      
      <div
        id={contentId}
        role="region"
        aria-hidden={!isExpanded}
        hidden={!isExpanded}
        className="thinking-content"
      >
        <pre>{thinking}</pre>
      </div>
    </div>
  );
}
```

### Keyboard Navigation

Support keyboard interactions:

```tsx
function handleKeyDown(
  event: React.KeyboardEvent,
  onToggle: () => void
) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    onToggle();
  }
}
```

### Reduced Motion Support

```css
@media (prefers-reduced-motion: reduce) {
  .accordion-icon,
  .smooth-collapse,
  .pulse-dot {
    animation: none;
    transition: none;
  }
}
```

---

## Combined Pattern: Full Thinking Component

```tsx
import { useState, useRef, useEffect } from 'react';

interface ThinkingSectionProps {
  thinking: string;
  durationMs?: number;
  tokenCount?: number;
  defaultOpen?: boolean;
}

export function ThinkingSection({
  thinking,
  durationMs,
  tokenCount,
  defaultOpen = false,
}: ThinkingSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);
  
  const wordCount = thinking.split(/\s+/).filter(Boolean).length;
  
  useEffect(() => {
    if (contentRef.current) {
      setHeight(isOpen ? contentRef.current.scrollHeight : 0);
    }
  }, [isOpen, thinking]);
  
  return (
    <div className="thinking-section">
      <button
        className="thinking-header"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="header-main">
          <span className="toggle-icon" data-open={isOpen}>▶</span>
          <span className="header-label">💭 Thinking</span>
        </div>
        
        <div className="header-stats">
          {durationMs && (
            <span className="stat">⏱️ {(durationMs / 1000).toFixed(1)}s</span>
          )}
          {tokenCount && (
            <span className="stat">🔢 {tokenCount.toLocaleString()} tokens</span>
          )}
          <span className="stat">{wordCount} words</span>
        </div>
      </button>
      
      <div 
        className="thinking-content"
        style={{ height: `${height}px` }}
        aria-hidden={!isOpen}
      >
        <div ref={contentRef}>
          <pre>{thinking}</pre>
        </div>
      </div>
    </div>
  );
}
```

---

## Summary

✅ Use collapsible sections to keep UI clean while maintaining transparency

✅ Accordion pattern works well for longer thinking content

✅ Show thinking duration to help users understand processing time

✅ Token visualization helps users understand API costs

✅ Support keyboard navigation and screen readers for accessibility

✅ Use CSS transitions with `prefers-reduced-motion` fallbacks

**Next:** [Streaming Reasoning Display](./03-streaming-reasoning-display.md)

---

## Further Reading

- [WAI-ARIA Disclosure Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/) — Accessible expandable sections
- [CSS Animation Best Practices](https://web.dev/animations-guide/) — Performance-optimized animations

---

<!-- 
Sources Consulted:
- WAI-ARIA Authoring Practices: https://www.w3.org/WAI/ARIA/apg/
- AI SDK Chatbot: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- web.dev Animation Guide: https://web.dev/animations-guide/
-->
