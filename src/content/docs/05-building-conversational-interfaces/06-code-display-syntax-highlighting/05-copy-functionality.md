---
title: "Copy Functionality"
---

# Copy Functionality

## Introduction

Copy-to-clipboard is one of the most used features in code blocks. Users frequently copy AI-generated code to paste into their projects. A smooth copy experience with clear feedback increases trust and usability.

In this lesson, we'll implement robust copy functionality for code blocks.

### What We'll Cover

- Clipboard API usage
- Copy button design and placement
- Visual feedback on success/failure
- Handling large code blocks
- Fallback strategies

### Prerequisites

- [Line Numbers and Highlighting](./04-line-numbers.md)
- Browser Clipboard API
- React state management

---

## Basic Copy Implementation

### Using the Clipboard API

```javascript
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return { success: true };
  } catch (error) {
    console.error('Failed to copy:', error);
    return { success: false, error };
  }
}
```

### React Component

```jsx
function CopyButton({ text, onCopied }) {
  const [status, setStatus] = useState('idle');
  
  const handleCopy = async () => {
    setStatus('copying');
    
    const result = await copyToClipboard(text);
    
    if (result.success) {
      setStatus('copied');
      onCopied?.();
      
      // Reset after delay
      setTimeout(() => setStatus('idle'), 2000);
    } else {
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
  };
  
  return (
    <button 
      onClick={handleCopy}
      className={`copy-button ${status}`}
      disabled={status === 'copying'}
      aria-label={status === 'copied' ? 'Copied!' : 'Copy code'}
    >
      {status === 'idle' && <CopyIcon />}
      {status === 'copying' && <SpinnerIcon />}
      {status === 'copied' && <CheckIcon />}
      {status === 'error' && <ErrorIcon />}
    </button>
  );
}
```

### Icon Components

```jsx
const CopyIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
  </svg>
);

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const ErrorIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="15" y1="9" x2="9" y2="15" />
    <line x1="9" y1="9" x2="15" y2="15" />
  </svg>
);
```

---

## Code Block with Copy Button

### Full Implementation

```jsx
function CodeBlock({ code, language, showLineNumbers = true }) {
  const [copied, setCopied] = useState(false);
  
  return (
    <div className="code-block-container">
      <div className="code-header">
        <span className="language-label">{language}</span>
        <CopyButton 
          text={code} 
          onCopied={() => setCopied(true)}
        />
      </div>
      
      <pre className={`language-${language}`}>
        <code>
          {showLineNumbers 
            ? <LinesWithNumbers code={code} language={language} />
            : <HighlightedCode code={code} language={language} />
          }
        </code>
      </pre>
    </div>
  );
}
```

### Styling

```css
.code-block-container {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: #1e1e1e;
}

.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background: #252526;
  border-bottom: 1px solid #333;
}

.language-label {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.copy-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.25rem 0.5rem;
  background: transparent;
  border: 1px solid #555;
  border-radius: 4px;
  color: #888;
  cursor: pointer;
  transition: all 0.2s;
}

.copy-button:hover {
  background: #333;
  color: #fff;
  border-color: #666;
}

.copy-button.copied {
  background: #238636;
  border-color: #238636;
  color: #fff;
}

.copy-button.error {
  background: #da3633;
  border-color: #da3633;
  color: #fff;
}

.copy-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

---

## Visual Feedback Patterns

### Toast Notification

```jsx
function CodeBlockWithToast({ code, language }) {
  const [toast, setToast] = useState(null);
  
  const handleCopy = async () => {
    const result = await copyToClipboard(code);
    
    if (result.success) {
      setToast({ type: 'success', message: 'Copied to clipboard!' });
    } else {
      setToast({ type: 'error', message: 'Failed to copy' });
    }
    
    setTimeout(() => setToast(null), 2500);
  };
  
  return (
    <div className="code-block-container">
      <button onClick={handleCopy} className="copy-button">
        <CopyIcon />
        Copy
      </button>
      
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' ? <CheckIcon /> : <ErrorIcon />}
          {toast.message}
        </div>
      )}
      
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
```

```css
.toast {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  animation: toast-fade 2.5s ease-in-out;
  pointer-events: none;
}

.toast.success {
  background: rgba(35, 134, 54, 0.95);
  color: white;
}

.toast.error {
  background: rgba(218, 54, 51, 0.95);
  color: white;
}

@keyframes toast-fade {
  0% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
  10% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  80% { opacity: 1; }
  100% { opacity: 0; }
}
```

### Button Text Change

```jsx
function CopyButtonWithText({ text }) {
  const [status, setStatus] = useState('idle');
  
  const labels = {
    idle: 'Copy code',
    copying: 'Copying...',
    copied: 'Copied!',
    error: 'Failed'
  };
  
  const handleCopy = async () => {
    setStatus('copying');
    const result = await copyToClipboard(text);
    setStatus(result.success ? 'copied' : 'error');
    setTimeout(() => setStatus('idle'), 2000);
  };
  
  return (
    <button onClick={handleCopy} className={`copy-btn ${status}`}>
      {status === 'idle' && <CopyIcon />}
      {status === 'copied' && <CheckIcon />}
      {status === 'error' && <ErrorIcon />}
      <span>{labels[status]}</span>
    </button>
  );
}
```

### Floating Copy Button

```jsx
function FloatingCopyButton({ code }) {
  const [visible, setVisible] = useState(false);
  
  return (
    <div 
      className="code-wrapper"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <CopyButton 
        text={code} 
        className={`floating-copy ${visible ? 'visible' : ''}`}
      />
      
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
```

```css
.code-wrapper {
  position: relative;
}

.floating-copy {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.floating-copy.visible {
  opacity: 1;
}

/* Always visible on touch devices */
@media (hover: none) {
  .floating-copy {
    opacity: 1;
  }
}
```

---

## Handling Edge Cases

### Large Code Blocks

```jsx
function CopyLargeCode({ code }) {
  const [status, setStatus] = useState('idle');
  const codeSize = new Blob([code]).size;
  const isLarge = codeSize > 50000;  // 50KB threshold
  
  const handleCopy = async () => {
    setStatus('copying');
    
    // For large code, show progress
    if (isLarge) {
      // Copy in chunks for very large code
      try {
        await navigator.clipboard.writeText(code);
        setStatus('copied');
      } catch (error) {
        // Fallback for large content
        const textarea = document.createElement('textarea');
        textarea.value = code;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        setStatus('copied');
      }
    } else {
      const result = await copyToClipboard(code);
      setStatus(result.success ? 'copied' : 'error');
    }
    
    setTimeout(() => setStatus('idle'), 2000);
  };
  
  return (
    <button onClick={handleCopy} className={`copy-button ${status}`}>
      {status === 'copying' && isLarge && (
        <span className="copying-large">Copying {(codeSize / 1024).toFixed(1)}KB...</span>
      )}
      {status === 'idle' && (
        <>
          <CopyIcon />
          {isLarge && <span className="size-badge">{(codeSize / 1024).toFixed(0)}KB</span>}
        </>
      )}
      {status === 'copied' && <CheckIcon />}
    </button>
  );
}
```

### Copy Selected Lines Only

```jsx
function CopySelectedLines({ code, selectedLines }) {
  const lines = code.split('\n');
  
  const getSelectedCode = () => {
    if (selectedLines.length === 0) {
      return code;  // Copy all if none selected
    }
    
    return selectedLines
      .sort((a, b) => a - b)
      .map(lineNum => lines[lineNum - 1])
      .join('\n');
  };
  
  return (
    <CopyButton 
      text={getSelectedCode()} 
      label={
        selectedLines.length > 0 
          ? `Copy ${selectedLines.length} line${selectedLines.length > 1 ? 's' : ''}`
          : 'Copy all'
      }
    />
  );
}
```

### Code Without Line Numbers

Ensure copied content doesn't include line numbers:

```jsx
function CodeBlockWithCleanCopy({ code, language }) {
  // Store original code separately from display
  const originalCode = useRef(code);
  
  return (
    <div className="code-block">
      <CopyButton text={originalCode.current} />
      
      <pre>
        <code>
          {code.split('\n').map((line, i) => (
            <div key={i} className="line">
              <span className="line-number">{i + 1}</span>
              <span className="line-code">{line}</span>
            </div>
          ))}
        </code>
      </pre>
    </div>
  );
}
```

---

## Clipboard API Fallback

### Legacy Browser Support

```javascript
async function copyWithFallback(text) {
  // Modern API
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return { success: true, method: 'clipboard' };
    } catch (error) {
      // Fall through to legacy method
    }
  }
  
  // Legacy fallback
  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.cssText = 'position:fixed;left:-9999px;top:-9999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    
    const success = document.execCommand('copy');
    document.body.removeChild(textarea);
    
    return { 
      success, 
      method: 'execCommand',
      error: success ? null : new Error('execCommand failed')
    };
  } catch (error) {
    return { success: false, error, method: 'fallback' };
  }
}
```

### Secure Context Check

```javascript
function canUseClipboard() {
  // Clipboard API requires secure context
  if (!window.isSecureContext) {
    return { available: false, reason: 'Requires HTTPS' };
  }
  
  if (!navigator.clipboard) {
    return { available: false, reason: 'Not supported' };
  }
  
  return { available: true };
}

function CopyButtonSecure({ text }) {
  const clipboardStatus = canUseClipboard();
  
  if (!clipboardStatus.available) {
    return (
      <button 
        disabled 
        title={`Copy unavailable: ${clipboardStatus.reason}`}
      >
        <CopyIcon />
      </button>
    );
  }
  
  return <CopyButton text={text} />;
}
```

---

## Copy with Formatting

### Including Language Metadata

```jsx
function CopyWithMetadata({ code, language, filename }) {
  const [format, setFormat] = useState('plain');
  
  const getFormattedCode = () => {
    switch (format) {
      case 'markdown':
        return `\`\`\`${language}\n${code}\n\`\`\``;
      case 'filename':
        return `// ${filename}\n${code}`;
      case 'html':
        return `<pre><code class="language-${language}">${escapeHtml(code)}</code></pre>`;
      default:
        return code;
    }
  };
  
  return (
    <div className="copy-options">
      <select value={format} onChange={e => setFormat(e.target.value)}>
        <option value="plain">Plain text</option>
        <option value="markdown">Markdown</option>
        <option value="filename">With filename</option>
        <option value="html">HTML</option>
      </select>
      
      <CopyButton text={getFormattedCode()} />
    </div>
  );
}
```

### Copy as Rich Text

```javascript
async function copyAsRichText(plainText, htmlContent) {
  try {
    const clipboardItem = new ClipboardItem({
      'text/plain': new Blob([plainText], { type: 'text/plain' }),
      'text/html': new Blob([htmlContent], { type: 'text/html' })
    });
    
    await navigator.clipboard.write([clipboardItem]);
    return { success: true };
  } catch (error) {
    // Fallback to plain text
    return copyToClipboard(plainText);
  }
}
```

---

## Accessibility

### Keyboard and Screen Reader Support

```jsx
function AccessibleCopyButton({ text, codeBlockId }) {
  const [status, setStatus] = useState('idle');
  const [announcement, setAnnouncement] = useState('');
  
  const handleCopy = async () => {
    setStatus('copying');
    const result = await copyToClipboard(text);
    
    if (result.success) {
      setStatus('copied');
      setAnnouncement('Code copied to clipboard');
    } else {
      setStatus('error');
      setAnnouncement('Failed to copy code');
    }
    
    setTimeout(() => {
      setStatus('idle');
      setAnnouncement('');
    }, 2000);
  };
  
  return (
    <>
      <button
        onClick={handleCopy}
        className={`copy-button ${status}`}
        aria-describedby={codeBlockId}
        aria-label={status === 'copied' ? 'Copied!' : 'Copy code to clipboard'}
      >
        {status === 'idle' && <CopyIcon aria-hidden="true" />}
        {status === 'copied' && <CheckIcon aria-hidden="true" />}
        <span className="button-text">
          {status === 'copied' ? 'Copied!' : 'Copy'}
        </span>
      </button>
      
      {/* Live region for screen reader announcements */}
      <div 
        role="status" 
        aria-live="polite" 
        className="sr-only"
      >
        {announcement}
      </div>
    </>
  );
}
```

```css
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

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show clear success feedback | Use only color changes |
| Include button text with icon | Use unlabeled icons |
| Handle clipboard API failures | Assume copy always works |
| Use `aria-live` for announcements | Forget screen reader users |
| Show code size for large blocks | Copy silently for 1MB+ |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Copy includes line numbers | Store original code separately |
| No feedback on success | Animate button or show toast |
| Fails on HTTP localhost | Use fallback method |
| Button hard to find | Show on hover or in header |
| No error handling | Display friendly error message |

---

## Hands-on Exercise

### Your Task

Create a `SmartCopyButton` component that:
1. Uses Clipboard API with fallback
2. Shows different icons for idle/copied/error states
3. Displays code size for blocks > 10KB
4. Announces status to screen readers

### Requirements

1. Implement `copyWithFallback` function
2. Track status with useState
3. Add live region for announcements
4. Show size badge for large code

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `new Blob([text]).size` for byte count
- Set `aria-live="polite"` on announcement div
- Reset status after 2-3 seconds
- Style differently for each state

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
function SmartCopyButton({ text }) {
  const [status, setStatus] = useState('idle');
  const [announcement, setAnnouncement] = useState('');
  const size = new Blob([text]).size;
  const isLarge = size > 10240;
  
  const handleCopy = async () => {
    setStatus('copying');
    const result = await copyWithFallback(text);
    
    if (result.success) {
      setStatus('copied');
      setAnnouncement('Code copied to clipboard');
    } else {
      setStatus('error');
      setAnnouncement('Failed to copy code');
    }
    
    setTimeout(() => {
      setStatus('idle');
      setAnnouncement('');
    }, 2500);
  };
  
  return (
    <>
      <button onClick={handleCopy} className={`copy-btn ${status}`}>
        {status === 'idle' && <CopyIcon />}
        {status === 'copying' && <SpinnerIcon />}
        {status === 'copied' && <CheckIcon />}
        {status === 'error' && <ErrorIcon />}
        
        {isLarge && status === 'idle' && (
          <span className="size-badge">{(size / 1024).toFixed(0)}KB</span>
        )}
      </button>
      
      <div role="status" aria-live="polite" className="sr-only">
        {announcement}
      </div>
    </>
  );
}
```

</details>

---

## Summary

✅ **Clipboard API** is modern but needs fallback  
✅ **Visual feedback** confirms copy success  
✅ **Floating buttons** appear on hover  
✅ **Large code** needs size indicators  
✅ **Screen readers** need live announcements  
✅ **Selected lines** can be copied separately

---

## Further Reading

- [Clipboard API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API)
- [Async Clipboard API Guide](https://web.dev/async-clipboard/)
- [ARIA Live Regions](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions)

---

**Previous:** [Line Numbers and Highlighting](./04-line-numbers.md)  
**Next:** [Code Theming](./06-code-theming.md)

<!-- 
Sources Consulted:
- MDN Clipboard API: https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API
- web.dev Async Clipboard: https://web.dev/async-clipboard/
- ARIA Live Regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
-->
