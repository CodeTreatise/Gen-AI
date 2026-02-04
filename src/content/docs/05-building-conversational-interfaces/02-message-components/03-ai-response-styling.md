---
title: "AI Response Styling"
---

# AI Response Styling

## Introduction

AI responses require special styling considerations that user messages don't need. From streaming animations to markdown rendering, code syntax highlighting to reasoning indicators—AI messages are complex content containers.

In this lesson, we'll build robust AI response styles that handle the full spectrum of LLM output.

### What We'll Cover

- Left-aligned layout with AI branding
- Streaming text animations
- Markdown content rendering
- Code block styling with copy functionality
- Model badges and token counts
- Thinking/reasoning display patterns
- Loading and regenerating states

### Prerequisites

- [Message Container Structure](./01-message-container-structure.md)
- [User Message Styling](./02-user-message-styling.md)
- Basic CSS animations

---

## Layout and Positioning

### Left-Aligned Design

AI messages appear on the left, opposite to user messages:

```css
.message-wrapper.assistant {
  flex-direction: row;
  align-self: flex-start;
}

.message-wrapper.assistant .message-container {
  background: var(--ai-bubble-bg, #f3f4f6);
  color: var(--ai-bubble-text, #1f2937);
  border-radius: 0.375rem 1.25rem 1.25rem 1.25rem;
  max-width: min(85%, 48rem);  /* Wider than user messages */
}
```

### AI Avatar/Icon

```css
.message-wrapper.assistant .message-avatar {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  background: linear-gradient(135deg, #8b5cf6, #3b82f6);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message-wrapper.assistant .message-avatar svg {
  width: 1.5rem;
  height: 1.5rem;
  color: white;
}

/* Animated avatar during streaming */
.message-wrapper.assistant.streaming .message-avatar {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(139, 92, 246, 0);
  }
}
```

---

## Streaming Animations

### Typing Indicator

Show something while the AI is thinking:

```css
.typing-indicator {
  display: flex;
  gap: 0.375rem;
  padding: 0.5rem 0;
}

.typing-indicator span {
  width: 0.5rem;
  height: 0.5rem;
  background: #9ca3af;
  border-radius: 50%;
  animation: typing-bounce 1.4s ease-in-out infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing-bounce {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-0.5rem);
  }
}
```

```html
<div class="message-wrapper assistant streaming">
  <div class="message-avatar">✨</div>
  <article class="message-container">
    <div class="typing-indicator" aria-label="AI is typing">
      <span></span>
      <span></span>
      <span></span>
    </div>
  </article>
</div>
```

### Streaming Cursor

When text is actively streaming:

```css
.message-wrapper.assistant.streaming .message-body::after {
  content: '▋';
  animation: cursor-blink 1s step-end infinite;
  color: #6b7280;
  margin-left: 0.125rem;
}

@keyframes cursor-blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Alternative: Underline cursor */
.message-wrapper.assistant.streaming.underline-cursor .message-body::after {
  content: '';
  display: inline-block;
  width: 0.5rem;
  height: 0.15rem;
  background: currentColor;
  animation: cursor-blink 1s step-end infinite;
}
```

### Smooth Text Reveal

Animate new words appearing:

```css
.streamed-word {
  animation: word-fade-in 0.15s ease-out;
}

@keyframes word-fade-in {
  from {
    opacity: 0.5;
    filter: blur(2px);
  }
  to {
    opacity: 1;
    filter: blur(0);
  }
}
```

```javascript
// JavaScript to add streaming animation
function appendStreamedContent(container, chunk) {
  const span = document.createElement('span');
  span.className = 'streamed-word';
  span.textContent = chunk;
  container.appendChild(span);
  
  // Remove animation class after completion
  span.addEventListener('animationend', () => {
    span.className = '';
  });
}
```

---

## Markdown Content Styling

### Typography Scale

```css
.message-wrapper.assistant .message-body {
  font-size: 0.9375rem;
  line-height: 1.7;
  color: #374151;
}

.message-wrapper.assistant .message-body h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 1.5rem 0 0.75rem;
  color: #111827;
}

.message-wrapper.assistant .message-body h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 1.25rem 0 0.625rem;
  color: #1f2937;
}

.message-wrapper.assistant .message-body h3 {
  font-size: 1.125rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem;
  color: #374151;
}

.message-wrapper.assistant .message-body p {
  margin: 0 0 0.875rem;
}

.message-wrapper.assistant .message-body p:last-child {
  margin-bottom: 0;
}
```

### Lists

```css
.message-wrapper.assistant .message-body ul,
.message-wrapper.assistant .message-body ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.message-wrapper.assistant .message-body li {
  margin: 0.375rem 0;
}

.message-wrapper.assistant .message-body li::marker {
  color: #6b7280;
}

/* Nested lists */
.message-wrapper.assistant .message-body li ul,
.message-wrapper.assistant .message-body li ol {
  margin: 0.25rem 0;
}
```

### Blockquotes

```css
.message-wrapper.assistant .message-body blockquote {
  margin: 0.75rem 0;
  padding: 0.75rem 1rem;
  border-left: 3px solid #8b5cf6;
  background: rgba(139, 92, 246, 0.05);
  border-radius: 0 0.375rem 0.375rem 0;
}

.message-wrapper.assistant .message-body blockquote p {
  margin: 0;
  font-style: italic;
  color: #4b5563;
}
```

### Links

```css
.message-wrapper.assistant .message-body a {
  color: #2563eb;
  text-decoration: underline;
  text-underline-offset: 2px;
  text-decoration-thickness: 1px;
}

.message-wrapper.assistant .message-body a:hover {
  color: #1d4ed8;
  text-decoration-thickness: 2px;
}

.message-wrapper.assistant .message-body a:visited {
  color: #7c3aed;
}
```

### Tables

```css
.message-wrapper.assistant .message-body table {
  width: 100%;
  margin: 0.75rem 0;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.message-wrapper.assistant .message-body th {
  padding: 0.625rem 0.75rem;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  font-weight: 600;
  text-align: left;
}

.message-wrapper.assistant .message-body td {
  padding: 0.5rem 0.75rem;
  border: 1px solid #e5e7eb;
}

.message-wrapper.assistant .message-body tr:nth-child(even) {
  background: #fafafa;
}
```

---

## Code Block Styling

### Inline Code

```css
.message-wrapper.assistant .message-body :not(pre) > code {
  padding: 0.125rem 0.375rem;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 0.25rem;
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.875em;
  color: #c7254e;
}
```

### Code Blocks with Header

```css
.code-block-wrapper {
  margin: 0.75rem 0;
  border-radius: 0.5rem;
  overflow: hidden;
  border: 1px solid #e5e7eb;
}

.code-block-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: #f3f4f6;
  border-bottom: 1px solid #e5e7eb;
  font-size: 0.75rem;
}

.code-block-language {
  color: #6b7280;
  font-family: 'Fira Code', monospace;
}

.code-block-copy {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  background: none;
  border: 1px solid #d1d5db;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.15s ease;
}

.code-block-copy:hover {
  background: #e5e7eb;
  color: #374151;
}

.code-block-copy.copied {
  background: #d1fae5;
  border-color: #10b981;
  color: #059669;
}

.message-wrapper.assistant .message-body pre {
  margin: 0;
  padding: 1rem;
  background: #1f2937;
  overflow-x: auto;
}

.message-wrapper.assistant .message-body pre code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #e5e7eb;
  background: none;
  padding: 0;
}
```

```html
<div class="code-block-wrapper">
  <div class="code-block-header">
    <span class="code-block-language">javascript</span>
    <button class="code-block-copy" aria-label="Copy code">
      <svg width="14" height="14"><!-- copy icon --></svg>
      Copy
    </button>
  </div>
  <pre><code class="language-javascript">function hello() {
  console.log("Hello, World!");
}</code></pre>
</div>
```

### Syntax Highlighting

Use a library like Prism.js or highlight.js:

```css
/* Prism.js token colors (dark theme) */
.token.comment { color: #6b7280; font-style: italic; }
.token.keyword { color: #c084fc; }
.token.string { color: #34d399; }
.token.number { color: #fbbf24; }
.token.function { color: #60a5fa; }
.token.operator { color: #f472b6; }
.token.class-name { color: #fcd34d; }
.token.punctuation { color: #9ca3af; }
```

---

## Model Badges and Metadata

### Model Indicator

```css
.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.25rem 0.625rem;
  background: #ede9fe;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: #6d28d9;
}

.model-badge-icon {
  width: 0.875rem;
  height: 0.875rem;
}

/* Model-specific colors */
.model-badge.gpt-4o { background: #d1fae5; color: #059669; }
.model-badge.gpt-4 { background: #dbeafe; color: #1d4ed8; }
.model-badge.claude { background: #fce7f3; color: #be185d; }
.model-badge.gemini { background: #fef3c7; color: #d97706; }
```

```html
<header class="message-header">
  <span class="sender-name">Assistant</span>
  <span class="model-badge gpt-4o">
    <svg class="model-badge-icon"><!-- sparkle icon --></svg>
    GPT-4o
  </span>
  <time class="message-time">2:31 PM</time>
</header>
```

### Token Count Display

```css
.token-count {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.token-count-icon {
  width: 0.875rem;
  height: 0.875rem;
}

.token-count-divider {
  margin: 0 0.5rem;
  color: #d1d5db;
}
```

```html
<footer class="message-footer">
  <span class="token-count">
    <svg class="token-count-icon"><!-- token icon --></svg>
    342 tokens
  </span>
  <span class="token-count-divider">·</span>
  <span class="response-time">1.2s</span>
</footer>
```

---

## Thinking/Reasoning Display

### Collapsible Reasoning Block

For models that show chain-of-thought:

```css
.reasoning-block {
  margin-bottom: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
}

.reasoning-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 0.625rem 1rem;
  background: #fafafa;
  border: none;
  cursor: pointer;
  font-size: 0.875rem;
  color: #6b7280;
}

.reasoning-toggle:hover {
  background: #f3f4f6;
}

.reasoning-toggle-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.reasoning-toggle-icon {
  transition: transform 0.2s ease;
}

.reasoning-block[open] .reasoning-toggle-icon {
  transform: rotate(180deg);
}

.reasoning-content {
  padding: 0.75rem 1rem;
  background: #fefce8;
  border-top: 1px solid #e5e7eb;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #713f12;
  white-space: pre-wrap;
}

/* Thinking animation during stream */
.reasoning-block.thinking .reasoning-toggle::after {
  content: '';
  width: 1rem;
  height: 1rem;
  border: 2px solid #e5e7eb;
  border-top-color: #8b5cf6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

```html
<details class="reasoning-block">
  <summary class="reasoning-toggle">
    <span class="reasoning-toggle-label">
      <svg class="reasoning-icon"><!-- brain icon --></svg>
      Thinking (3.2s)
    </span>
    <svg class="reasoning-toggle-icon"><!-- chevron icon --></svg>
  </summary>
  <div class="reasoning-content">
Let me break this down step by step...
1. First, I need to understand the user's question
2. Then, I'll consider the relevant context
3. Finally, I'll formulate a clear response
  </div>
</details>
```

### Inline Thinking Indicator

```css
.thinking-inline {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  background: #f3e8ff;
  border-radius: 1rem;
  font-size: 0.75rem;
  color: #7c3aed;
}

.thinking-inline-spinner {
  width: 0.75rem;
  height: 0.75rem;
  border: 2px solid rgba(124, 58, 237, 0.2);
  border-top-color: #7c3aed;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
```

---

## Loading and Regenerating States

### Regenerating Animation

```css
.message-wrapper.assistant.regenerating {
  opacity: 0.7;
  position: relative;
}

.message-wrapper.assistant.regenerating::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(139, 92, 246, 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  from { transform: translateX(-100%); }
  to { transform: translateX(100%); }
}

.regenerate-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(2px);
  border-radius: inherit;
}

.regenerate-spinner {
  width: 2rem;
  height: 2rem;
  border: 3px solid #e5e7eb;
  border-top-color: #8b5cf6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
```

### Error State

```css
.message-wrapper.assistant.error .message-container {
  background: #fef2f2;
  border: 1px solid #fecaca;
}

.ai-error-message {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.5rem 0;
  color: #991b1b;
}

.ai-error-icon {
  flex-shrink: 0;
  width: 1.25rem;
  height: 1.25rem;
  color: #dc2626;
}

.ai-error-content {
  flex: 1;
}

.ai-error-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.ai-error-details {
  font-size: 0.875rem;
  color: #b91c1c;
}

.ai-retry-button {
  margin-top: 0.5rem;
  padding: 0.5rem 1rem;
  background: #dc2626;
  border: none;
  border-radius: 0.375rem;
  color: white;
  font-size: 0.875rem;
  cursor: pointer;
}

.ai-retry-button:hover {
  background: #b91c1c;
}
```

---

## Complete Component

### React Implementation

```jsx
// AIMessage.jsx
function AIMessage({ 
  message, 
  isStreaming = false,
  onRegenerate,
  onCopy 
}) {
  const [copiedBlock, setCopiedBlock] = useState(null);
  const [reasoningOpen, setReasoningOpen] = useState(false);
  
  const handleCopyCode = async (code, index) => {
    await navigator.clipboard.writeText(code);
    setCopiedBlock(index);
    setTimeout(() => setCopiedBlock(null), 2000);
  };
  
  return (
    <div 
      className={`message-wrapper assistant ${isStreaming ? 'streaming' : ''}`}
      data-message-id={message.id}
    >
      <div className="message-avatar" aria-hidden="true">
        <AIIcon />
      </div>
      
      <article 
        className="message-container"
        tabIndex={0}
        aria-label="AI response"
        aria-busy={isStreaming}
      >
        <header className="message-header">
          <span className="sender-name">Assistant</span>
          {message.model && (
            <ModelBadge model={message.model} />
          )}
          <time className="message-time">
            {formatTime(message.createdAt)}
          </time>
        </header>
        
        {message.reasoning && (
          <ReasoningBlock
            content={message.reasoning}
            duration={message.reasoningDuration}
            isOpen={reasoningOpen}
            onToggle={() => setReasoningOpen(!reasoningOpen)}
          />
        )}
        
        <div className="message-body">
          <MarkdownRenderer 
            content={message.content}
            onCopyCode={handleCopyCode}
            copiedBlock={copiedBlock}
          />
          {isStreaming && <span className="streaming-cursor" />}
        </div>
        
        {!isStreaming && (
          <footer className="message-footer">
            {message.usage && (
              <span className="token-count">
                {message.usage.totalTokens.toLocaleString()} tokens
              </span>
            )}
          </footer>
        )}
      </article>
      
      {!isStreaming && (
        <MessageActions
          onCopy={() => onCopy(message)}
          onRegenerate={() => onRegenerate(message)}
        />
      )}
    </div>
  );
}

function ModelBadge({ model }) {
  const modelClass = model.toLowerCase().replace(/[\s.-]/g, '-');
  return (
    <span className={`model-badge ${modelClass}`}>
      <SparkleIcon className="model-badge-icon" />
      {model}
    </span>
  );
}

function ReasoningBlock({ content, duration, isOpen, onToggle }) {
  return (
    <details className="reasoning-block" open={isOpen}>
      <summary className="reasoning-toggle" onClick={onToggle}>
        <span className="reasoning-toggle-label">
          <BrainIcon />
          Thinking {duration && `(${duration})`}
        </span>
        <ChevronIcon className="reasoning-toggle-icon" />
      </summary>
      <div className="reasoning-content">
        {content}
      </div>
    </details>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show streaming feedback | Leave blank while loading |
| Use syntax highlighting for code | Display code as plain text |
| Include copy button on code blocks | Force users to select manually |
| Display model information | Hide which model responded |
| Handle errors gracefully | Show raw error objects |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No loading indicator | Add typing indicator immediately |
| Code overflow breaks layout | Use `overflow-x: auto` on `pre` |
| Hard to read long responses | Add proper typography scale |
| Missing streaming cursor | Show visual feedback during stream |
| Reasoning hidden by default | Auto-expand on short reasoning |

---

## Hands-on Exercise

### Your Task

Style an AI response message with:
1. Left-aligned layout with avatar
2. Streaming cursor animation
3. Model badge in header
4. Code block with copy button
5. Token count in footer

### Requirements

1. Use the three-layer container structure
2. Add proper markdown styling
3. Implement copy functionality for code
4. Show streaming state while loading

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `::after` pseudo-element for streaming cursor
- Apply `aria-busy="true"` during streaming
- The copy button should show "Copied!" feedback
- Use `<details>` for collapsible reasoning

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the complete React and CSS implementation in the "Complete Component" section above.

</details>

---

## Summary

✅ **Left alignment** distinguishes AI from user messages  
✅ **Streaming animations** provide real-time feedback  
✅ **Markdown styling** handles rich formatted content  
✅ **Code blocks with copy** improve developer experience  
✅ **Model badges** show which AI generated the response  
✅ **Reasoning blocks** display chain-of-thought when available

---

## Further Reading

- [Prism.js Syntax Highlighting](https://prismjs.com/)
- [CSS Animation Performance](https://web.dev/animations-overview/)
- [Streaming UI Patterns](https://vercel.com/blog/ai-sdk-3-generative-ui)

---

**Previous:** [User Message Styling](./02-user-message-styling.md)  
**Next:** [System Message Handling](./04-system-message-handling.md)

<!-- 
Sources Consulted:
- Prism.js: https://prismjs.com/
- web.dev CSS Animations: https://web.dev/animations-overview/
- Vercel AI SDK Patterns: https://vercel.com/blog/ai-sdk-3-generative-ui
-->
