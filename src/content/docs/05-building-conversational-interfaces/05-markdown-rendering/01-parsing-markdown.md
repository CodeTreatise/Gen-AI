---
title: "Parsing Markdown in Responses"
---

# Parsing Markdown in Responses

## Introduction

Choosing the right markdown parser shapes your entire rendering pipeline. For AI chat, we need parsers that handle streaming text, recover from incomplete syntax, and support extensibility for custom elements like code blocks and citations.

In this lesson, we'll explore parser selection and streaming-compatible parsing strategies.

### What We'll Cover

- Parser library comparison
- Streaming-compatible parsing
- Incremental rendering techniques
- Performance optimization
- Error recovery patterns

### Prerequisites

- [Markdown Rendering Overview](./00-markdown-rendering.md)
- Basic understanding of AST (Abstract Syntax Tree)
- React component basics

---

## Parser Library Comparison

### marked

The fastest pure JavaScript markdown parser.

```bash
npm install marked
```

```javascript
import { marked } from 'marked';

const html = marked.parse('# Hello **World**');
// <h1>Hello <strong>World</strong></h1>
```

**Pros:**
- Extremely fast (~2-3x faster than alternatives)
- Small bundle size (~32KB)
- Synchronous parsing

**Cons:**
- Limited extensibility
- HTML string output (not ideal for React)
- Partial streaming support

### remark / unified

A powerful plugin ecosystem for transforming markdown.

```bash
npm install remark remark-parse remark-html
```

```javascript
import { remark } from 'remark';
import html from 'remark-html';

const result = await remark()
  .use(html)
  .process('# Hello **World**');

console.log(String(result));
```

**Pros:**
- Extensive plugin ecosystem
- AST-based (full control)
- Great for React with `react-markdown`

**Cons:**
- Larger bundle (~85KB+)
- Async by default
- Steeper learning curve

### react-markdown

React-native markdown rendering using remark.

```bash
npm install react-markdown
```

```jsx
import ReactMarkdown from 'react-markdown';

function Message({ content }) {
  return <ReactMarkdown>{content}</ReactMarkdown>;
}
```

**Pros:**
- Direct React component output
- Built-in sanitization
- Component customization
- Best streaming compatibility

**Cons:**
- React-specific
- Heavier than marked

### markdown-it

Highly configurable with plugin support.

```bash
npm install markdown-it
```

```javascript
import MarkdownIt from 'markdown-it';

const md = new MarkdownIt();
const html = md.render('# Hello **World**');
```

**Pros:**
- CommonMark compliant
- Good plugin ecosystem
- Configurable

**Cons:**
- HTML string output
- Medium bundle size

---

## Streaming-Compatible Parsing

### The Challenge

During streaming, markdown arrives incomplete:

```
Stream chunk 1: "Here is some **bold"
Stream chunk 2: " text** and a `code"
Stream chunk 3: " block`."
```

A naive parser would fail on unclosed syntax.

### Solution: Graceful Degradation

```jsx
import ReactMarkdown from 'react-markdown';
import { useMemo } from 'react';

function StreamingMarkdown({ content, isStreaming }) {
  // Clean up incomplete markdown during streaming
  const processedContent = useMemo(() => {
    if (!isStreaming) return content;
    
    return cleanIncompleteMarkdown(content);
  }, [content, isStreaming]);
  
  return <ReactMarkdown>{processedContent}</ReactMarkdown>;
}

function cleanIncompleteMarkdown(text) {
  let result = text;
  
  // Close unclosed bold
  const boldCount = (result.match(/\*\*/g) || []).length;
  if (boldCount % 2 !== 0) {
    result += '**';
  }
  
  // Close unclosed italic (single asterisk)
  const italicCount = (result.match(/(?<!\*)\*(?!\*)/g) || []).length;
  if (italicCount % 2 !== 0) {
    result += '*';
  }
  
  // Close unclosed inline code
  const backtickCount = (result.match(/`/g) || []).length;
  if (backtickCount % 2 !== 0) {
    result += '`';
  }
  
  // Handle unclosed code blocks
  const codeBlockMatches = result.match(/```/g) || [];
  if (codeBlockMatches.length % 2 !== 0) {
    result += '\n```';
  }
  
  return result;
}
```

### Robust Incomplete Handler

```jsx
function useStreamingMarkdown(rawContent, isStreaming) {
  return useMemo(() => {
    if (!isStreaming) return rawContent;
    
    let content = rawContent;
    
    // Track open formatting
    const stack = [];
    
    // Scan for formatting markers
    let i = 0;
    while (i < content.length) {
      if (content.slice(i, i + 2) === '**') {
        if (stack[stack.length - 1] === '**') {
          stack.pop();
        } else {
          stack.push('**');
        }
        i += 2;
      } else if (content[i] === '*' && content[i - 1] !== '*' && content[i + 1] !== '*') {
        if (stack[stack.length - 1] === '*') {
          stack.pop();
        } else {
          stack.push('*');
        }
        i++;
      } else if (content[i] === '`') {
        if (stack[stack.length - 1] === '`') {
          stack.pop();
        } else {
          stack.push('`');
        }
        i++;
      } else {
        i++;
      }
    }
    
    // Close any open formatting (reverse order)
    while (stack.length > 0) {
      content += stack.pop();
    }
    
    return content;
  }, [rawContent, isStreaming]);
}
```

---

## Incremental Rendering

### Chunk-Based Updates

Instead of re-parsing everything on each chunk:

```jsx
function useIncrementalMarkdown(content) {
  const [paragraphs, setParagraphs] = useState([]);
  const lastParsedRef = useRef('');
  
  useEffect(() => {
    // Only parse new content
    const newContent = content.slice(lastParsedRef.current.length);
    
    if (newContent.includes('\n\n')) {
      // Complete paragraph received - parse it
      const newParagraphs = newContent.split('\n\n').filter(Boolean);
      setParagraphs(prev => [...prev, ...newParagraphs.slice(0, -1)]);
      lastParsedRef.current = content;
    }
  }, [content]);
  
  return {
    completedParagraphs: paragraphs,
    currentParagraph: content.slice(lastParsedRef.current.length)
  };
}
```

### Block-Level Streaming

```jsx
function StreamingContent({ content, isStreaming }) {
  // Split into complete blocks and current streaming block
  const blocks = useMemo(() => {
    const lines = content.split('\n');
    const complete = [];
    let current = [];
    
    for (const line of lines) {
      // Detect block boundaries
      if (line === '' && current.length > 0) {
        complete.push(current.join('\n'));
        current = [];
      } else {
        current.push(line);
      }
    }
    
    return {
      complete,
      current: current.join('\n')
    };
  }, [content]);
  
  return (
    <div className="streaming-content">
      {/* Render complete blocks */}
      {blocks.complete.map((block, i) => (
        <ReactMarkdown key={i}>{block}</ReactMarkdown>
      ))}
      
      {/* Render current streaming block */}
      {blocks.current && (
        <StreamingMarkdown 
          content={blocks.current}
          isStreaming={isStreaming}
        />
      )}
    </div>
  );
}
```

---

## Performance Optimization

### Memoization

```jsx
const MemoizedMarkdown = memo(function MemoizedMarkdown({ content }) {
  return <ReactMarkdown>{content}</ReactMarkdown>;
}, (prev, next) => prev.content === next.content);
```

### Virtualization for Long Content

```jsx
import { FixedSizeList as List } from 'react-window';

function VirtualizedContent({ blocks }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <ReactMarkdown>{blocks[index]}</ReactMarkdown>
    </div>
  );
  
  return (
    <List
      height={600}
      itemCount={blocks.length}
      itemSize={100}
      width="100%"
    >
      {Row}
    </List>
  );
}
```

### Lazy Parsing

```jsx
function LazyMarkdown({ content }) {
  const [parsed, setParsed] = useState(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !parsed) {
          setParsed(content);
        }
      },
      { threshold: 0.1 }
    );
    
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    
    return () => observer.disconnect();
  }, [content, parsed]);
  
  return (
    <div ref={containerRef}>
      {parsed ? (
        <ReactMarkdown>{parsed}</ReactMarkdown>
      ) : (
        <div className="markdown-placeholder" />
      )}
    </div>
  );
}
```

---

## Custom Components

### react-markdown Component Overrides

```jsx
import ReactMarkdown from 'react-markdown';

function CustomMarkdown({ content }) {
  return (
    <ReactMarkdown
      components={{
        // Custom heading with anchor
        h1: ({ children }) => (
          <h1 id={slugify(children)}>
            {children}
            <a href={`#${slugify(children)}`} className="anchor">
              #
            </a>
          </h1>
        ),
        
        // Custom link with external indicator
        a: ({ href, children }) => {
          const isExternal = href?.startsWith('http');
          return (
            <a 
              href={href}
              target={isExternal ? '_blank' : undefined}
              rel={isExternal ? 'noopener noreferrer' : undefined}
            >
              {children}
              {isExternal && <ExternalIcon />}
            </a>
          );
        },
        
        // Custom code block
        code: ({ inline, className, children }) => {
          if (inline) {
            return <code className="inline-code">{children}</code>;
          }
          
          const language = className?.replace('language-', '');
          return (
            <CodeBlock language={language}>
              {String(children)}
            </CodeBlock>
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

---

## Error Recovery

### Safe Parsing Wrapper

```jsx
function SafeMarkdown({ content, fallback }) {
  const [error, setError] = useState(null);
  
  if (error) {
    return fallback || <pre>{content}</pre>;
  }
  
  return (
    <ErrorBoundary onError={setError}>
      <ReactMarkdown>{content}</ReactMarkdown>
    </ErrorBoundary>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error) {
    this.props.onError?.(error);
  }
  
  render() {
    if (this.state.hasError) {
      return null;
    }
    return this.props.children;
  }
}
```

### Sanitization

```jsx
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

function SanitizedMarkdown({ content }) {
  return (
    <ReactMarkdown
      rehypePlugins={[rehypeSanitize]}
    >
      {content}
    </ReactMarkdown>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use react-markdown for React apps | Parse to HTML strings in React |
| Handle incomplete markdown during streaming | Let parser crash on partial input |
| Memoize parsed output | Re-parse on every render |
| Sanitize user-generated content | Trust all markdown input |
| Provide fallback for parse errors | Let errors crash the UI |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Parser crashes on unclosed code blocks | Close incomplete syntax before parsing |
| Performance degrades with long content | Use memoization and virtualization |
| XSS vulnerabilities from raw HTML | Use rehype-sanitize |
| Flickering during streaming | Batch updates, use transitions |
| Broken links in markdown | Validate and handle gracefully |

---

## Hands-on Exercise

### Your Task

Build a streaming-compatible markdown renderer that:
1. Handles incomplete markdown gracefully
2. Uses custom components for headings and links
3. Shows a fallback on parse errors

### Requirements

1. Close unclosed formatting markers
2. External links open in new tab
3. Headings have anchor links
4. Error boundary with fallback

<details>
<summary>💡 Hints (click to expand)</summary>

- Count formatting markers to detect unclosed ones
- Use `startsWith('http')` for external link detection
- Create `slugify` function for anchor IDs
- Wrap in ErrorBoundary component

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```jsx
function StreamingMarkdownRenderer({ content, isStreaming }) {
  const processedContent = useStreamingMarkdown(content, isStreaming);
  
  return (
    <SafeMarkdown content={processedContent}>
      <CustomMarkdown content={processedContent} />
    </SafeMarkdown>
  );
}
```

See the complete implementations in the sections above.

</details>

---

## Summary

✅ **react-markdown** is ideal for React chat interfaces  
✅ **Streaming requires** closing incomplete syntax  
✅ **Custom components** enable rich rendering  
✅ **Memoization** prevents performance issues  
✅ **Error boundaries** prevent crashes  
✅ **Sanitization** protects against XSS

---

## Further Reading

- [react-markdown Documentation](https://github.com/remarkjs/react-markdown)
- [remark Plugin Ecosystem](https://github.com/remarkjs/remark/blob/main/doc/plugins.md)
- [rehype-sanitize](https://github.com/rehypejs/rehype-sanitize)

---

**Previous:** [Markdown Rendering Overview](./00-markdown-rendering.md)  
**Next:** [Formatted Text Rendering](./02-formatted-text.md)

<!-- 
Sources Consulted:
- react-markdown: https://github.com/remarkjs/react-markdown
- remark ecosystem: https://github.com/remarkjs/remark
- marked.js: https://marked.js.org/
- markdown-it: https://github.com/markdown-it/markdown-it
-->
