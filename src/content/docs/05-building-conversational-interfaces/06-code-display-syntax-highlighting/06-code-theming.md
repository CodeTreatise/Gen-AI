---
title: "Code Theming"
---

# Code Theming

## Introduction

Consistent theming across code blocks creates a polished experience. Users expect code to adapt to light/dark mode preferences, and custom themes can match your application's brand. AI chat interfaces often need themes that work well in message bubbles.

In this lesson, we'll implement flexible theming for code blocks.

### What We'll Cover

- Built-in theme libraries
- Light and dark mode support
- System preference detection
- Custom theme creation
- Theme switching without flicker

### Prerequisites

- [Copy Functionality](./05-copy-functionality.md)
- CSS custom properties (variables)
- React context

---

## Theme Libraries

### Popular Theme Collections

| Library | Themes Included | Format |
|---------|-----------------|--------|
| **Prism.js** | ~10 official | CSS files |
| **highlight.js** | ~200+ | CSS files |
| **Shiki** | All VS Code themes | JSON |
| **react-syntax-highlighter** | Prism + hljs | JS objects |

### Importing Built-in Themes

```javascript
// Prism.js themes
import 'prismjs/themes/prism.css';              // Light
import 'prismjs/themes/prism-dark.css';         // Dark
import 'prismjs/themes/prism-tomorrow.css';     // Tomorrow Night
import 'prismjs/themes/prism-okaidia.css';      // Monokai-like

// highlight.js themes
import 'highlight.js/styles/github.css';        // GitHub Light
import 'highlight.js/styles/github-dark.css';   // GitHub Dark
import 'highlight.js/styles/atom-one-dark.css'; // Atom One Dark
import 'highlight.js/styles/vs2015.css';        // VS Dark

// react-syntax-highlighter (JS objects)
import { 
  oneDark, 
  oneLight,
  vs,
  vsDark,
  atomOneDark,
  github,
  dracula
} from 'react-syntax-highlighter/dist/esm/styles/prism';
```

---

## Light and Dark Mode

### Theme Context

```jsx
import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext({
  theme: 'dark',
  setTheme: () => {}
});

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('dark');
  
  // Persist preference
  useEffect(() => {
    const saved = localStorage.getItem('code-theme');
    if (saved) setTheme(saved);
  }, []);
  
  useEffect(() => {
    localStorage.setItem('code-theme', theme);
  }, [theme]);
  
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
```

### Theme-Aware Code Block

```jsx
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

function ThemedCodeBlock({ code, language }) {
  const { theme } = useTheme();
  const style = theme === 'dark' ? oneDark : oneLight;
  
  return (
    <SyntaxHighlighter 
      language={language}
      style={style}
      customStyle={{
        margin: 0,
        borderRadius: '8px'
      }}
    >
      {code}
    </SyntaxHighlighter>
  );
}
```

### Theme Toggle

```jsx
function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  
  return (
    <button 
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      className="theme-toggle"
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}
```

---

## System Preference Detection

### Matching OS Theme

```jsx
function useSystemTheme() {
  const [systemTheme, setSystemTheme] = useState(() => {
    if (typeof window === 'undefined') return 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches 
      ? 'dark' 
      : 'light';
  });
  
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);
  
  return systemTheme;
}
```

### Three-Way Toggle

```jsx
function ThemeProviderWithSystem({ children }) {
  const [preference, setPreference] = useState('system'); // 'light' | 'dark' | 'system'
  const systemTheme = useSystemTheme();
  
  const effectiveTheme = preference === 'system' ? systemTheme : preference;
  
  return (
    <ThemeContext.Provider value={{ 
      theme: effectiveTheme, 
      preference,
      setPreference 
    }}>
      {children}
    </ThemeContext.Provider>
  );
}

function ThemeSelector() {
  const { preference, setPreference } = useTheme();
  
  return (
    <div className="theme-selector">
      <button 
        onClick={() => setPreference('light')}
        className={preference === 'light' ? 'active' : ''}
      >
        <SunIcon /> Light
      </button>
      <button 
        onClick={() => setPreference('dark')}
        className={preference === 'dark' ? 'active' : ''}
      >
        <MoonIcon /> Dark
      </button>
      <button 
        onClick={() => setPreference('system')}
        className={preference === 'system' ? 'active' : ''}
      >
        <ComputerIcon /> System
      </button>
    </div>
  );
}
```

---

## CSS Custom Properties Approach

### Token-Based Theming

```css
:root {
  /* Light theme (default) */
  --code-bg: #f6f8fa;
  --code-text: #24292e;
  --code-keyword: #d73a49;
  --code-string: #032f62;
  --code-number: #005cc5;
  --code-comment: #6a737d;
  --code-function: #6f42c1;
  --code-operator: #d73a49;
  --code-class: #22863a;
  --code-line-number: #959da5;
  --code-selection: rgba(3, 102, 214, 0.2);
}

[data-theme="dark"] {
  --code-bg: #1e1e1e;
  --code-text: #d4d4d4;
  --code-keyword: #569cd6;
  --code-string: #ce9178;
  --code-number: #b5cea8;
  --code-comment: #6a9955;
  --code-function: #dcdcaa;
  --code-operator: #d4d4d4;
  --code-class: #4ec9b0;
  --code-line-number: #858585;
  --code-selection: rgba(51, 153, 255, 0.2);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --code-bg: #1e1e1e;
    --code-text: #d4d4d4;
    /* ... dark values */
  }
}
```

### Applying CSS Variables

```css
.code-block {
  background: var(--code-bg);
  color: var(--code-text);
}

.token.keyword { color: var(--code-keyword); }
.token.string { color: var(--code-string); }
.token.number { color: var(--code-number); }
.token.comment { color: var(--code-comment); font-style: italic; }
.token.function { color: var(--code-function); }
.token.operator { color: var(--code-operator); }
.token.class-name { color: var(--code-class); }

.line-number { color: var(--code-line-number); }

::selection {
  background: var(--code-selection);
}
```

### Setting Theme Attribute

```jsx
function ThemeProvider({ children }) {
  const { theme } = useTheme();
  
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);
  
  return children;
}
```

---

## Custom Theme Creation

### Theme Object Structure

```javascript
const myCustomTheme = {
  'code[class*="language-"]': {
    color: '#e6e6e6',
    background: 'none',
    fontFamily: '"Fira Code", monospace',
    fontSize: '14px',
    lineHeight: '1.5'
  },
  'pre[class*="language-"]': {
    color: '#e6e6e6',
    background: '#1a1a2e',
    padding: '1em',
    margin: '0',
    borderRadius: '8px',
    overflow: 'auto'
  },
  comment: { color: '#6272a4', fontStyle: 'italic' },
  prolog: { color: '#6272a4' },
  doctype: { color: '#6272a4' },
  cdata: { color: '#6272a4' },
  punctuation: { color: '#f8f8f2' },
  property: { color: '#ff79c6' },
  tag: { color: '#ff79c6' },
  constant: { color: '#bd93f9' },
  symbol: { color: '#ffb86c' },
  deleted: { color: '#ff5555' },
  boolean: { color: '#bd93f9' },
  number: { color: '#bd93f9' },
  selector: { color: '#50fa7b' },
  'attr-name': { color: '#50fa7b' },
  string: { color: '#f1fa8c' },
  char: { color: '#f1fa8c' },
  builtin: { color: '#8be9fd' },
  inserted: { color: '#50fa7b' },
  variable: { color: '#f8f8f2' },
  operator: { color: '#ff79c6' },
  entity: { color: '#f8f8f2' },
  url: { color: '#8be9fd' },
  '.language-css .token.string': { color: '#f1fa8c' },
  '.style .token.string': { color: '#f1fa8c' },
  atrule: { color: '#ff79c6' },
  'attr-value': { color: '#f1fa8c' },
  keyword: { color: '#ff79c6' },
  function: { color: '#50fa7b' },
  'class-name': { color: '#8be9fd' },
  regex: { color: '#ffb86c' },
  important: { color: '#ffb86c', fontWeight: 'bold' }
};
```

### Using Custom Theme

```jsx
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

function CodeBlock({ code, language }) {
  return (
    <SyntaxHighlighter 
      language={language}
      style={myCustomTheme}
    >
      {code}
    </SyntaxHighlighter>
  );
}
```

### Theme Generator

```javascript
function createTheme({
  background,
  foreground,
  keyword,
  string,
  number,
  comment,
  function: fn,
  className,
  fontFamily = '"Fira Code", monospace'
}) {
  return {
    'code[class*="language-"]': {
      color: foreground,
      background: 'none',
      fontFamily
    },
    'pre[class*="language-"]': {
      color: foreground,
      background,
      padding: '1em',
      margin: '0',
      borderRadius: '8px',
      overflow: 'auto'
    },
    comment: { color: comment, fontStyle: 'italic' },
    keyword: { color: keyword },
    string: { color: string },
    number: { color: number },
    function: { color: fn },
    'class-name': { color: className },
    boolean: { color: number },
    operator: { color: foreground }
  };
}

// Usage
const githubDark = createTheme({
  background: '#24292e',
  foreground: '#e1e4e8',
  keyword: '#f97583',
  string: '#9ecbff',
  number: '#79b8ff',
  comment: '#6a737d',
  function: '#b392f0',
  className: '#79b8ff'
});
```

---

## Preventing Theme Flash

### Critical CSS

```html
<!-- In <head> before React loads -->
<script>
  (function() {
    const theme = localStorage.getItem('code-theme') || 
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>

<style>
  /* Critical code theme CSS */
  [data-theme="dark"] .code-block {
    background: #1e1e1e;
    color: #d4d4d4;
  }
  [data-theme="light"] .code-block {
    background: #f6f8fa;
    color: #24292e;
  }
</style>
```

### SSR Considerations

```jsx
// For Next.js or similar
function CodeBlockSSR({ code, language }) {
  const [mounted, setMounted] = useState(false);
  const { theme } = useTheme();
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Render placeholder during SSR
  if (!mounted) {
    return (
      <pre className="code-block placeholder">
        <code>{code}</code>
      </pre>
    );
  }
  
  return (
    <SyntaxHighlighter 
      language={language}
      style={theme === 'dark' ? oneDark : oneLight}
    >
      {code}
    </SyntaxHighlighter>
  );
}
```

---

## Chat Interface Theming

### Message-Aware Code Blocks

```jsx
function ChatCodeBlock({ code, language, sender }) {
  const { theme } = useTheme();
  
  // Adjust style based on message sender
  const getStyle = () => {
    if (sender === 'user') {
      // User messages might have different bubble color
      return theme === 'dark' ? userDarkTheme : userLightTheme;
    }
    return theme === 'dark' ? oneDark : oneLight;
  };
  
  return (
    <div className={`code-in-message ${sender}`}>
      <SyntaxHighlighter 
        language={language}
        style={getStyle()}
        customStyle={{
          borderRadius: '8px',
          margin: '0.5rem 0'
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
```

### Contrast with Bubbles

```css
/* Ensure code blocks contrast with message bubbles */
.message.user .code-block {
  background: rgba(0, 0, 0, 0.2);  /* Darker than user bubble */
}

.message.assistant .code-block {
  background: rgba(0, 0, 0, 0.1);  /* Slight contrast */
}

/* Dark mode adjustments */
[data-theme="dark"] .message.user .code-block {
  background: rgba(0, 0, 0, 0.3);
}

[data-theme="dark"] .message.assistant .code-block {
  background: rgba(255, 255, 255, 0.05);
}
```

---

## Accessibility Considerations

### Contrast Requirements

```javascript
// Check color contrast for WCAG compliance
function checkContrast(foreground, background) {
  const getLuminance = (hex) => {
    const rgb = parseInt(hex.slice(1), 16);
    const r = (rgb >> 16) & 0xff;
    const g = (rgb >> 8) & 0xff;
    const b = rgb & 0xff;
    
    const toLinear = (c) => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    
    return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
  };
  
  const l1 = getLuminance(foreground);
  const l2 = getLuminance(background);
  const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  
  return {
    ratio: ratio.toFixed(2),
    passesAA: ratio >= 4.5,
    passesAAA: ratio >= 7
  };
}

// Validate theme colors
const themeColors = {
  background: '#1e1e1e',
  text: '#d4d4d4',
  comment: '#6a9955'
};

console.log('Text contrast:', checkContrast(themeColors.text, themeColors.background));
// { ratio: "10.36", passesAA: true, passesAAA: true }

console.log('Comment contrast:', checkContrast(themeColors.comment, themeColors.background));
// { ratio: "4.93", passesAA: true, passesAAA: false }
```

### High Contrast Mode

```css
@media (prefers-contrast: high) {
  .code-block {
    border: 2px solid currentColor;
  }
  
  .token.keyword,
  .token.function,
  .token.string {
    font-weight: bold;
  }
  
  .token.comment {
    /* Boost contrast for comments */
    color: #8bc34a !important;
  }
}

/* Forced colors (Windows High Contrast) */
@media (forced-colors: active) {
  .code-block {
    border: 1px solid CanvasText;
  }
  
  .token {
    color: CanvasText;
  }
  
  .token.keyword {
    color: LinkText;
  }
  
  .token.string {
    color: Highlight;
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Respect system preferences | Force single theme |
| Test contrast ratios | Use low-contrast colors |
| Load theme before render | Flash wrong theme |
| Provide theme picker | Assume all users prefer dark |
| Persist user preference | Reset theme on refresh |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Theme flashes on load | Set theme in `<head>` script |
| Code unreadable in light mode | Test both themes thoroughly |
| Comments too dim | Ensure 4.5:1 contrast ratio |
| SSR hydration mismatch | Defer theme-specific rendering |
| Overriding all styles | Use CSS custom properties |

---

## Hands-on Exercise

### Your Task

Create a `ThemeableCodeBlock` component that:
1. Detects system preference
2. Allows manual override
3. Persists choice in localStorage
4. Applies theme without flash

### Requirements

1. Implement `useSystemTheme` hook
2. Create context for theme state
3. Add CSS custom properties
4. Handle SSR (no hydration mismatch)

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `matchMedia` for system preference
- Set theme attribute in `document.documentElement`
- Add inline script in `<head>` for SSR
- Check `typeof window !== 'undefined'`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
// Theme context and hooks
const ThemeContext = createContext();

function useSystemTheme() {
  const [theme, setTheme] = useState('dark');
  
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    setTheme(mq.matches ? 'dark' : 'light');
    
    const handler = (e) => setTheme(e.matches ? 'dark' : 'light');
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  
  return theme;
}

function ThemeProvider({ children }) {
  const [preference, setPreference] = useState('system');
  const systemTheme = useSystemTheme();
  
  useEffect(() => {
    const saved = localStorage.getItem('theme-pref');
    if (saved) setPreference(saved);
  }, []);
  
  useEffect(() => {
    localStorage.setItem('theme-pref', preference);
    const theme = preference === 'system' ? systemTheme : preference;
    document.documentElement.setAttribute('data-theme', theme);
  }, [preference, systemTheme]);
  
  const theme = preference === 'system' ? systemTheme : preference;
  
  return (
    <ThemeContext.Provider value={{ theme, preference, setPreference }}>
      {children}
    </ThemeContext.Provider>
  );
}
```

</details>

---

## Summary

✅ **Theme libraries** provide ready-to-use styles  
✅ **System preference** detection respects user settings  
✅ **CSS custom properties** enable dynamic theming  
✅ **Custom themes** match application branding  
✅ **Flash prevention** requires head scripts  
✅ **Accessibility** demands adequate contrast

---

## Further Reading

- [prefers-color-scheme - MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme)
- [CSS Custom Properties Guide](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
- [WCAG Contrast Requirements](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [highlight.js Theme Gallery](https://highlightjs.org/static/demo/)

---

**Previous:** [Copy Functionality](./05-copy-functionality.md)  
**Back to Lesson Overview:** [Code Display & Syntax Highlighting](./00-code-display-syntax-highlighting.md)

<!-- 
Sources Consulted:
- MDN prefers-color-scheme: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme
- WCAG Contrast: https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
- highlight.js themes: https://highlightjs.org/static/demo/
- react-syntax-highlighter: https://github.com/react-syntax-highlighter/react-syntax-highlighter
-->
