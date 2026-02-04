---
title: "Text Input Design"
---

# Text Input Design

## Introduction

The text input is the foundation of any chat interface—where users compose their messages. Proper input design encompasses styling, placeholder text, focus states, disabled states, and accessibility. A well-designed input feels natural and responsive across all interaction states.

In this lesson, we'll build a polished text input component with proper styling and state management.

### What We'll Cover

- Input field structure (input vs textarea)
- Placeholder text patterns
- Focus and blur states
- Disabled and readonly states
- Dark mode support
- Accessibility best practices

### Prerequisites

- [Input Handling Overview](./00-input-handling.md)
- CSS/Tailwind basics
- React controlled components

---

## Input vs Textarea

### When to Use Each

| Element | Use Case | Behavior |
|---------|----------|----------|
| `<input type="text">` | Single-line prompts, search | No line breaks |
| `<textarea>` | Multi-line messages, chat | Supports Enter for newlines |

For chat interfaces, **textarea** is preferred because users often write multi-line messages.

---

## Basic Styled Textarea

```tsx
interface ChatTextareaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  maxLength?: number;
}

function ChatTextarea({
  value,
  onChange,
  placeholder = 'Type a message...',
  disabled = false,
  maxLength = 4000
}: ChatTextareaProps) {
  return (
    <textarea
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      maxLength={maxLength}
      rows={1}
      className={`
        w-full px-4 py-3 rounded-xl border
        resize-none overflow-hidden
        transition-all duration-200
        
        /* Default state */
        border-gray-200 bg-white text-gray-900
        placeholder:text-gray-400
        
        /* Focus state */
        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
        
        /* Disabled state */
        disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed
        
        /* Dark mode */
        dark:bg-gray-800 dark:border-gray-700 dark:text-white
        dark:placeholder:text-gray-500
        dark:focus:ring-blue-400
      `}
      aria-label="Message input"
    />
  );
}
```

---

## Placeholder Text Patterns

### Dynamic Placeholders

```tsx
function useDynamicPlaceholder(status: string, error: Error | null) {
  if (error) {
    return 'Something went wrong. Try again...';
  }
  
  switch (status) {
    case 'submitted':
      return 'Sending...';
    case 'streaming':
      return 'AI is responding...';
    case 'ready':
    default:
      return 'Type a message...';
  }
}

// Usage
function ChatInput() {
  const { status, error } = useChat();
  const placeholder = useDynamicPlaceholder(status, error);
  
  return (
    <textarea placeholder={placeholder} />
  );
}
```

### Contextual Placeholders

```tsx
const PLACEHOLDERS = {
  default: 'Type a message...',
  newConversation: 'Start a new conversation...',
  followUp: 'Ask a follow-up question...',
  codeHelp: 'Describe the code you need help with...',
  imageDescription: 'Describe the image you want to generate...'
};

function getContextualPlaceholder(
  messageCount: number,
  mode: 'chat' | 'code' | 'image'
): string {
  if (mode === 'code') return PLACEHOLDERS.codeHelp;
  if (mode === 'image') return PLACEHOLDERS.imageDescription;
  if (messageCount === 0) return PLACEHOLDERS.newConversation;
  return PLACEHOLDERS.followUp;
}
```

---

## Focus States

### Visual Focus Indication

```css
/* CSS for clear focus states */
.chat-input {
  /* Default */
  border: 2px solid #e5e7eb;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.chat-input:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

/* High contrast for accessibility */
@media (prefers-contrast: high) {
  .chat-input:focus {
    border-color: #1d4ed8;
    box-shadow: 0 0 0 4px rgba(29, 78, 216, 0.5);
  }
}
```

### Focus Management

```tsx
function ChatInputWithFocus() {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { status } = useChat();
  
  // Auto-focus on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);
  
  // Re-focus after response completes
  useEffect(() => {
    if (status === 'ready') {
      textareaRef.current?.focus();
    }
  }, [status]);
  
  // Focus on keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + / to focus input
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  return <textarea ref={textareaRef} />;
}
```

---

## Disabled States

### Visual Disabled State

```tsx
interface DisabledTextareaProps {
  disabled: boolean;
  disabledReason?: string;
}

function DisabledAwareTextarea({ 
  disabled, 
  disabledReason,
  ...props 
}: DisabledTextareaProps & React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <div className="relative">
      <textarea
        {...props}
        disabled={disabled}
        className={`
          w-full px-4 py-3 rounded-lg border
          ${disabled 
            ? 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed' 
            : 'bg-white border-gray-300 text-gray-900'
          }
        `}
        aria-disabled={disabled}
      />
      
      {/* Reason tooltip */}
      {disabled && disabledReason && (
        <div className="absolute bottom-full mb-1 left-0 text-xs text-gray-500 bg-white px-2 py-1 rounded shadow">
          {disabledReason}
        </div>
      )}
    </div>
  );
}

// Usage
<DisabledAwareTextarea
  disabled={status === 'streaming'}
  disabledReason="Wait for response to complete"
/>
```

### Conditional Disabling

```tsx
function useInputDisabled(status: string, error: Error | null) {
  // Disable during processing
  if (status === 'submitted' || status === 'streaming') {
    return {
      disabled: true,
      reason: 'Waiting for response...'
    };
  }
  
  // Enable even on error (allow retry)
  return {
    disabled: false,
    reason: null
  };
}
```

---

## Complete Styled Input Component

```tsx
interface StyledChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  status: 'ready' | 'submitted' | 'streaming' | 'error';
  error?: Error | null;
  maxLength?: number;
  className?: string;
}

function StyledChatInput({
  value,
  onChange,
  onSubmit,
  status,
  error,
  maxLength = 4000,
  className = ''
}: StyledChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const isDisabled = status === 'submitted' || status === 'streaming';
  const isNearLimit = value.length > maxLength * 0.9;
  const isOverLimit = value.length > maxLength;
  
  const placeholder = useDynamicPlaceholder(status, error ?? null);
  
  // Auto-resize
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);
  
  // Handle keyboard
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isDisabled && value.trim()) {
      e.preventDefault();
      onSubmit();
    }
  };
  
  return (
    <div className={`relative ${className}`}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isDisabled}
        maxLength={maxLength}
        rows={1}
        className={`
          w-full px-4 py-3 pr-12
          rounded-2xl border-2
          resize-none
          transition-all duration-200
          
          /* Base styles */
          text-base leading-relaxed
          
          /* State-based colors */
          ${isDisabled 
            ? 'bg-gray-50 border-gray-200 text-gray-400' 
            : isOverLimit
            ? 'bg-red-50 border-red-300 text-gray-900'
            : error
            ? 'bg-red-50 border-red-300 text-gray-900'
            : 'bg-white border-gray-200 text-gray-900 hover:border-gray-300'
          }
          
          /* Focus state */
          focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100
          
          /* Placeholder */
          placeholder:text-gray-400
          
          /* Dark mode */
          dark:bg-gray-900 dark:border-gray-700 dark:text-white
          dark:placeholder:text-gray-500
          dark:focus:border-blue-400 dark:focus:ring-blue-900
        `}
        style={{ maxHeight: '200px' }}
        aria-label="Message input"
        aria-invalid={isOverLimit || !!error}
        aria-describedby={isOverLimit ? 'length-error' : undefined}
      />
      
      {/* Character counter */}
      <span 
        className={`
          absolute bottom-2 right-3 text-xs
          ${isOverLimit ? 'text-red-500 font-medium' : 
            isNearLimit ? 'text-yellow-600' : 'text-gray-400'}
        `}
        aria-live="polite"
      >
        {value.length.toLocaleString()} / {maxLength.toLocaleString()}
      </span>
      
      {/* Error message */}
      {isOverLimit && (
        <p id="length-error" className="mt-1 text-sm text-red-500">
          Message exceeds maximum length
        </p>
      )}
    </div>
  );
}
```

---

## Dark Mode Support

```tsx
function DarkModeTextarea({ className = '', ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={`
        /* Light mode */
        bg-white border-gray-300 text-gray-900
        placeholder:text-gray-400
        focus:ring-blue-500 focus:border-blue-500
        
        /* Dark mode */
        dark:bg-gray-800 dark:border-gray-600 dark:text-gray-100
        dark:placeholder:text-gray-400
        dark:focus:ring-blue-400 dark:focus:border-blue-400
        
        /* Disabled in both modes */
        disabled:bg-gray-100 disabled:dark:bg-gray-900
        disabled:text-gray-400 disabled:dark:text-gray-500
        
        ${className}
      `}
    />
  );
}
```

---

## Accessibility Checklist

| Requirement | Implementation |
|-------------|----------------|
| Label | `aria-label="Message input"` or visible `<label>` |
| Error state | `aria-invalid="true"` when invalid |
| Error description | `aria-describedby="error-id"` |
| Disabled state | `disabled` attribute + `aria-disabled` |
| Focus visible | Clear focus ring (2-4px) |
| Color contrast | 4.5:1 for text, 3:1 for placeholders |
| Reduced motion | Respect `prefers-reduced-motion` |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use visible focus rings | Remove focus outlines |
| Show placeholder in proper contrast | Use light gray on white |
| Indicate disabled reason | Disable without explanation |
| Support dark mode | Hard-code light colors |
| Re-focus after submit | Leave focus elsewhere |
| Use `aria-label` for inputs | Rely on placeholder for meaning |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Placeholder as label | Add proper label or aria-label |
| Low contrast placeholder | Use 4.5:1 contrast minimum |
| No focus indication | Add visible focus ring |
| Disabled looks like enabled | Use distinct disabled styling |
| Forgetting dark mode | Test both color schemes |

---

## Hands-on Exercise

### Your Task

Build a styled chat textarea with:
1. Custom border and focus styles
2. Dynamic placeholder based on status
3. Disabled state with visual feedback
4. Character counter
5. Dark mode support

### Requirements

1. Focus ring visible on keyboard focus
2. Disabled state clearly distinguishable
3. Counter changes color near limit
4. Works in light and dark modes

<details>
<summary>💡 Hints (click to expand)</summary>

- Use Tailwind's `dark:` prefix for dark mode
- Add `focus:ring-2 focus:ring-offset-2` for visible focus
- Use `aria-label` for accessibility
- Track character count with `value.length`

</details>

---

## Summary

✅ **Textarea** preferred for multi-line chat  
✅ **Placeholder** provides context, not labels  
✅ **Focus states** must be clearly visible  
✅ **Disabled states** need visual distinction  
✅ **Dark mode** requires explicit styling  
✅ **Accessibility** via ARIA attributes

---

## Further Reading

- [MDN: textarea element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/textarea)
- [WCAG Focus Visible](https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html)
- [Tailwind Forms Plugin](https://github.com/tailwindlabs/tailwindcss-forms)

---

**Previous:** [Input Handling Overview](./00-input-handling.md)  
**Next:** [Auto-Expanding Textarea](./02-auto-expanding-textarea.md)

<!-- 
Sources Consulted:
- MDN textarea: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/textarea
- WCAG Focus: https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html
- Tailwind Forms: https://github.com/tailwindlabs/tailwindcss-forms
-->
