---
title: "Copy Message Content"
---

# Copy Message Content

## Introduction

Copying message content is the most frequently used action in chat interfaces. Users copy AI responses to use in documents, share with colleagues, or save for later reference. A smooth copy experience requires proper Clipboard API usage, format options, and clear feedback.

In this lesson, we'll implement robust message copying with the modern Clipboard API.

### What We'll Cover

- Clipboard API fundamentals
- Copying formatted vs plain text
- Rich text (HTML) copying
- Copy success feedback
- Error handling and fallbacks

### Prerequisites

- [Message Actions Overview](./00-message-actions.md)
- JavaScript async/await
- Basic React patterns

---

## Clipboard API Basics

### The Modern Approach

```typescript
// ✅ Modern: Async Clipboard API
async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy:', error);
    return false;
  }
}
```

### Security Requirements

The Clipboard API requires:
- **Secure context** (HTTPS or localhost)
- **User activation** (triggered by user gesture)
- **Permissions** (browser may prompt or auto-grant)

```typescript
// Check clipboard availability
function isClipboardAvailable(): boolean {
  return !!(
    navigator.clipboard && 
    typeof navigator.clipboard.writeText === 'function'
  );
}
```

> **Note:** Most browsers auto-grant `clipboard-write` permission when triggered by user interaction. Reading requires explicit permission.

---

## Plain Text Copying

### Basic Implementation

```typescript
interface CopyResult {
  success: boolean;
  error?: Error;
}

async function copyText(text: string): Promise<CopyResult> {
  // Check availability
  if (!navigator.clipboard) {
    return { 
      success: false, 
      error: new Error('Clipboard API not available') 
    };
  }
  
  try {
    await navigator.clipboard.writeText(text);
    return { success: true };
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error : new Error('Copy failed') 
    };
  }
}
```

### React Hook

```typescript
interface UseCopyReturn {
  copy: (text: string) => Promise<boolean>;
  copied: boolean;
  error: Error | null;
}

export function useCopy(resetDelay = 2000): UseCopyReturn {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const timeoutRef = useRef<number | null>(null);
  
  const copy = useCallback(async (text: string): Promise<boolean> => {
    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    setError(null);
    
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      
      // Reset after delay
      timeoutRef.current = window.setTimeout(() => {
        setCopied(false);
      }, resetDelay);
      
      return true;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Copy failed');
      setError(error);
      setCopied(false);
      return false;
    }
  }, [resetDelay]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  return { copy, copied, error };
}
```

---

## Rich Text (HTML) Copying

### Copying as HTML

```typescript
async function copyAsHtml(html: string, plainText: string): Promise<boolean> {
  try {
    const htmlBlob = new Blob([html], { type: 'text/html' });
    const textBlob = new Blob([plainText], { type: 'text/plain' });
    
    const clipboardItem = new ClipboardItem({
      'text/html': htmlBlob,
      'text/plain': textBlob  // Fallback for apps that don't support HTML
    });
    
    await navigator.clipboard.write([clipboardItem]);
    return true;
  } catch (error) {
    console.error('Failed to copy HTML:', error);
    return false;
  }
}
```

### Usage for Messages

```typescript
function copyMessage(message: Message, format: 'plain' | 'rich' = 'plain') {
  if (format === 'plain') {
    // Get plain text content
    const text = extractPlainText(message);
    return copyText(text);
  } else {
    // Get rendered HTML
    const html = renderMessageToHtml(message);
    const text = extractPlainText(message);
    return copyAsHtml(html, text);
  }
}

function extractPlainText(message: Message): string {
  // Handle parts-based messages
  if (message.parts) {
    return message.parts
      .filter(part => part.type === 'text')
      .map(part => part.text)
      .join('\n');
  }
  
  // Legacy content property
  return message.content;
}
```

---

## Copy Button Component

### Basic Copy Button

```tsx
interface CopyButtonProps {
  text: string;
  className?: string;
}

function CopyButton({ text, className }: CopyButtonProps) {
  const { copy, copied, error } = useCopy();
  
  return (
    <button
      onClick={() => copy(text)}
      className={`
        flex items-center gap-1 px-2 py-1 rounded
        hover:bg-gray-100 transition-colors
        ${className}
      `}
      aria-label={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <>
          <CheckIcon className="w-4 h-4 text-green-500" />
          <span className="text-green-500 text-sm">Copied!</span>
        </>
      ) : (
        <>
          <ClipboardIcon className="w-4 h-4 text-gray-500" />
          <span className="text-gray-500 text-sm">Copy</span>
        </>
      )}
    </button>
  );
}
```

### Animated Copy Button

```tsx
function AnimatedCopyButton({ text }: { text: string }) {
  const { copy, copied } = useCopy();
  
  return (
    <button
      onClick={() => copy(text)}
      className="relative p-2 rounded hover:bg-gray-100"
      aria-label="Copy to clipboard"
    >
      {/* Clipboard icon - fades out when copied */}
      <ClipboardIcon 
        className={`
          w-5 h-5 transition-all duration-200
          ${copied ? 'scale-0 opacity-0' : 'scale-100 opacity-100'}
        `}
      />
      
      {/* Check icon - fades in when copied */}
      <CheckIcon 
        className={`
          absolute inset-0 m-auto w-5 h-5 text-green-500
          transition-all duration-200
          ${copied ? 'scale-100 opacity-100' : 'scale-0 opacity-0'}
        `}
      />
    </button>
  );
}
```

---

## Format Selection

### Copy With Format Options

```tsx
interface CopyWithFormatProps {
  message: Message;
}

function CopyWithFormat({ message }: CopyWithFormatProps) {
  const [showMenu, setShowMenu] = useState(false);
  const { copy, copied } = useCopy();
  
  const handleCopy = async (format: 'plain' | 'markdown' | 'html') => {
    let text: string;
    
    switch (format) {
      case 'plain':
        text = extractPlainText(message);
        break;
      case 'markdown':
        text = message.content;  // Already markdown
        break;
      case 'html':
        text = renderToHtml(message.content);
        break;
    }
    
    await copy(text);
    setShowMenu(false);
  };
  
  return (
    <div className="relative">
      <button
        onClick={() => setShowMenu(!showMenu)}
        className="flex items-center gap-1 p-2 rounded hover:bg-gray-100"
      >
        {copied ? <CheckIcon /> : <ClipboardIcon />}
        <ChevronDownIcon className="w-3 h-3" />
      </button>
      
      {showMenu && (
        <div className="absolute right-0 mt-1 bg-white border rounded-lg shadow-lg z-10">
          <button
            onClick={() => handleCopy('plain')}
            className="block w-full px-4 py-2 text-left hover:bg-gray-50"
          >
            Copy as plain text
          </button>
          <button
            onClick={() => handleCopy('markdown')}
            className="block w-full px-4 py-2 text-left hover:bg-gray-50"
          >
            Copy as Markdown
          </button>
          <button
            onClick={() => handleCopy('html')}
            className="block w-full px-4 py-2 text-left hover:bg-gray-50"
          >
            Copy as HTML
          </button>
        </div>
      )}
    </div>
  );
}
```

---

## Copy Feedback Patterns

### Toast Notification

```tsx
function CopyWithToast({ text }: { text: string }) {
  const [toast, setToast] = useState<{ show: boolean; message: string }>({
    show: false,
    message: ''
  });
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setToast({ show: true, message: 'Copied to clipboard!' });
    } catch {
      setToast({ show: true, message: 'Failed to copy' });
    }
    
    setTimeout(() => setToast({ show: false, message: '' }), 2000);
  };
  
  return (
    <>
      <button onClick={handleCopy}>Copy</button>
      
      {toast.show && (
        <div className="fixed bottom-4 right-4 bg-gray-900 text-white px-4 py-2 rounded-lg shadow-lg animate-fade-in">
          {toast.message}
        </div>
      )}
    </>
  );
}
```

### Inline Feedback

```tsx
function CopyWithInlineFeedback({ text }: { text: string }) {
  const { copy, copied, error } = useCopy();
  
  return (
    <button
      onClick={() => copy(text)}
      className={`
        px-3 py-1.5 rounded-md text-sm font-medium
        transition-all duration-200
        ${copied 
          ? 'bg-green-100 text-green-700' 
          : error 
            ? 'bg-red-100 text-red-700'
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
        }
      `}
    >
      {copied ? '✓ Copied!' : error ? '✗ Failed' : 'Copy'}
    </button>
  );
}
```

---

## Fallback for Older Browsers

```typescript
async function copyWithFallback(text: string): Promise<boolean> {
  // Try modern API first
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to legacy method
    }
  }
  
  // Legacy fallback using execCommand
  return copyWithExecCommand(text);
}

function copyWithExecCommand(text: string): boolean {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  
  // Prevent scrolling
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  textarea.style.top = '0';
  
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  
  let success = false;
  try {
    success = document.execCommand('copy');
  } catch {
    success = false;
  }
  
  document.body.removeChild(textarea);
  return success;
}
```

> **Warning:** `document.execCommand('copy')` is deprecated. Use it only as a fallback for browsers without Clipboard API support.

---

## Message Copy Integration

### Complete Message Component

```tsx
function MessageWithCopy({ message }: { message: Message }) {
  const { copy, copied } = useCopy();
  
  const getMessageText = (): string => {
    if (message.parts) {
      return message.parts
        .map(part => {
          if (part.type === 'text') return part.text;
          if (part.type === 'tool-invocation') return `[Tool: ${part.toolName}]`;
          return '';
        })
        .join('\n');
    }
    return message.content || '';
  };
  
  return (
    <div className="group relative p-4">
      <div className="prose">
        <MessageContent message={message} />
      </div>
      
      {/* Copy button - visible on hover */}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => copy(getMessageText())}
          className="p-1.5 rounded bg-white shadow-sm hover:bg-gray-50"
          aria-label={copied ? 'Copied' : 'Copy message'}
        >
          {copied ? (
            <CheckIcon className="w-4 h-4 text-green-500" />
          ) : (
            <ClipboardIcon className="w-4 h-4 text-gray-400" />
          )}
        </button>
      </div>
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `navigator.clipboard.writeText` | Use deprecated `execCommand` as primary |
| Provide visual feedback (2s duration) | Leave user guessing if copy worked |
| Handle errors gracefully | Ignore clipboard failures |
| Support keyboard shortcut (Cmd+C) | Require mouse for all copying |
| Offer format options when relevant | Force single format only |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not checking API availability | Check `navigator.clipboard` exists |
| Missing user activation | Only call from click/keyboard handler |
| No HTTPS in production | Ensure secure context |
| Feedback too brief | Show "Copied" for 2+ seconds |
| No fallback | Provide `execCommand` fallback |

---

## Hands-on Exercise

### Your Task

Build a copy system for chat messages with:
1. One-click copy button
2. Format selector (plain/markdown)
3. Animated feedback (icon change + color)
4. Toast notification for confirmation

### Requirements

1. Use `useCopy` hook pattern
2. Support both plain text and markdown
3. Show success state for 2 seconds
4. Handle errors with user feedback

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `useState` to track copied state
- Use `setTimeout` to reset after 2 seconds
- Check `navigator.clipboard` availability
- Use CSS transitions for smooth animations

</details>

---

## Summary

✅ **Clipboard API** is the modern standard  
✅ **writeText** for plain text copying  
✅ **ClipboardItem** for rich content (HTML)  
✅ **Visual feedback** confirms action  
✅ **Fallback** for older browsers  
✅ **Error handling** for edge cases

---

## Further Reading

- [Clipboard API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API)
- [Unblocking Clipboard Access](https://web.dev/articles/async-clipboard)
- [ClipboardItem - MDN](https://developer.mozilla.org/en-US/docs/Web/API/ClipboardItem)

---

**Previous:** [Message Actions Overview](./00-message-actions.md)  
**Next:** [Copy Code Blocks](./02-copy-code-blocks.md)

<!-- 
Sources Consulted:
- MDN Clipboard API: https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API
- web.dev async clipboard: https://web.dev/articles/async-clipboard
-->
