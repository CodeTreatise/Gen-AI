---
title: "Detecting Code Blocks in Responses"
---

# Detecting Code Blocks in Responses

## Introduction

Before highlighting code, we must find it. AI responses mix prose with fenced code blocks, inline code snippets, and sometimes malformed markdown. Robust detection handles edge cases gracefully—especially during streaming when blocks are incomplete.

In this lesson, we'll implement reliable code block detection.

### What We'll Cover

- Fenced code block detection (```)
- Language tag extraction
- Inline code vs block code
- Handling malformed blocks
- Streaming considerations

### Prerequisites

- [Code Display Overview](./00-code-display-syntax-highlighting.md)
- Regular expressions basics
- Markdown syntax

---

## Fenced Code Block Detection

### Basic Pattern

Fenced code blocks use triple backticks:

~~~markdown
```javascript
const x = 42;
```
~~~

### Regex Detection

```javascript
function extractCodeBlocks(markdown) {
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  const blocks = [];
  let match;
  
  while ((match = codeBlockRegex.exec(markdown)) !== null) {
    blocks.push({
      language: match[1] || 'text',
      code: match[2],
      start: match.index,
      end: match.index + match[0].length
    });
  }
  
  return blocks;
}
```

**Output:**
```javascript
extractCodeBlocks('```python\nprint("hello")\n```');
// [{ language: 'python', code: 'print("hello")\n', start: 0, end: 26 }]
```

### Handling Indented Blocks

Some AI responses indent code blocks:

```javascript
function extractCodeBlocksRobust(markdown) {
  // Match both standard and indented code blocks
  const codeBlockRegex = /^[ \t]*```(\w*)\n([\s\S]*?)^[ \t]*```/gm;
  const blocks = [];
  let match;
  
  while ((match = codeBlockRegex.exec(markdown)) !== null) {
    blocks.push({
      language: match[1] || 'text',
      code: dedent(match[2]),  // Remove common indentation
      start: match.index,
      end: match.index + match[0].length
    });
  }
  
  return blocks;
}

function dedent(code) {
  const lines = code.split('\n');
  const minIndent = lines
    .filter(line => line.trim())
    .reduce((min, line) => {
      const indent = line.match(/^[ \t]*/)[0].length;
      return Math.min(min, indent);
    }, Infinity);
  
  if (minIndent === Infinity) return code;
  
  return lines
    .map(line => line.slice(minIndent))
    .join('\n');
}
```

---

## Language Tag Extraction

### Common Language Tags

```javascript
const LANGUAGE_ALIASES = {
  // JavaScript
  'js': 'javascript',
  'jsx': 'jsx',
  'ts': 'typescript',
  'tsx': 'tsx',
  
  // Python
  'py': 'python',
  'python3': 'python',
  
  // Shell
  'sh': 'bash',
  'shell': 'bash',
  'zsh': 'bash',
  
  // Web
  'htm': 'html',
  'css': 'css',
  'scss': 'scss',
  
  // Data
  'json': 'json',
  'yml': 'yaml',
  'yaml': 'yaml',
  
  // Others
  'rb': 'ruby',
  'rs': 'rust',
  'go': 'go',
  'c++': 'cpp',
  'c#': 'csharp'
};

function normalizeLanguage(lang) {
  if (!lang) return 'text';
  const lower = lang.toLowerCase().trim();
  return LANGUAGE_ALIASES[lower] || lower;
}
```

### Extracting Language with Metadata

Some code blocks include filename or metadata:

~~~markdown
```javascript title="app.js"
const app = express();
```
~~~

```javascript
function parseCodeFence(fenceLine) {
  // Pattern: ```language metadata
  const match = fenceLine.match(/^```(\w+)?(?:\s+(.*))?$/);
  
  if (!match) return { language: 'text', metadata: {} };
  
  const language = normalizeLanguage(match[1]);
  const metaString = match[2] || '';
  
  // Parse key="value" or key=value pairs
  const metadata = {};
  const metaRegex = /(\w+)=["']?([^"'\s]+)["']?/g;
  let metaMatch;
  
  while ((metaMatch = metaRegex.exec(metaString)) !== null) {
    metadata[metaMatch[1]] = metaMatch[2];
  }
  
  return { language, metadata };
}
```

**Usage:**
```javascript
parseCodeFence('```typescript title="utils.ts" highlight={1,3}');
// { 
//   language: 'typescript', 
//   metadata: { title: 'utils.ts', highlight: '{1,3}' } 
// }
```

---

## Inline Code vs Block Code

### Detection Rules

| Pattern | Type | Example |
|---------|------|---------|
| Single backticks | Inline | \`const x\` |
| Triple backticks | Block | \`\`\`js...``` |
| 4+ space indent | Block (old style) | (indented code) |

```javascript
function detectCodeType(text) {
  // Check for fenced block
  if (/^```/m.test(text)) {
    return 'fenced-block';
  }
  
  // Check for indented block (4 spaces or tab)
  if (/^(    |\t)/m.test(text)) {
    return 'indented-block';
  }
  
  // Check for inline
  if (/`[^`]+`/.test(text)) {
    return 'inline';
  }
  
  return 'none';
}
```

### React-Markdown Integration

```jsx
import ReactMarkdown from 'react-markdown';

function MarkdownWithCode({ content }) {
  return (
    <ReactMarkdown
      components={{
        code: ({ inline, className, children, ...props }) => {
          if (inline) {
            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          }
          
          // Block code
          const language = className?.replace('language-', '') || 'text';
          return (
            <CodeBlock language={language} code={String(children)} />
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

### Inline Code Styling

```css
.inline-code {
  padding: 2px 6px;
  background: var(--code-bg, #f4f4f5);
  border-radius: 4px;
  font-family: 'Fira Code', 'Monaco', monospace;
  font-size: 0.875em;
  color: var(--code-color, #e11d48);
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .inline-code {
    background: var(--code-bg-dark, #27272a);
    color: var(--code-color-dark, #fb7185);
  }
}
```

---

## Handling Malformed Blocks

### Common Issues

1. **Missing closing fence**
2. **Nested backticks in content**
3. **Wrong number of backticks**
4. **Mixed fence styles (``` vs ~~~)**

### Robust Parser

```javascript
function parseCodeBlocks(text) {
  const blocks = [];
  let current = null;
  let lineNumber = 0;
  
  const lines = text.split('\n');
  
  for (const line of lines) {
    lineNumber++;
    
    // Check for opening fence
    const openMatch = line.match(/^(`{3,}|~{3,})(\w*)/);
    
    if (!current && openMatch) {
      // Start of code block
      current = {
        fence: openMatch[1],
        language: openMatch[2] || 'text',
        lines: [],
        startLine: lineNumber
      };
      continue;
    }
    
    if (current) {
      // Check for matching closing fence
      if (line.startsWith(current.fence.charAt(0).repeat(current.fence.length))) {
        // End of code block
        blocks.push({
          language: current.language,
          code: current.lines.join('\n'),
          startLine: current.startLine,
          endLine: lineNumber
        });
        current = null;
        continue;
      }
      
      // Add line to current block
      current.lines.push(line);
    }
  }
  
  // Handle unclosed block
  if (current) {
    blocks.push({
      language: current.language,
      code: current.lines.join('\n'),
      startLine: current.startLine,
      endLine: lineNumber,
      incomplete: true
    });
  }
  
  return blocks;
}
```

### Graceful Incomplete Handling

```jsx
function CodeBlock({ language, code, incomplete }) {
  return (
    <div className={`code-block ${incomplete ? 'incomplete' : ''}`}>
      <pre>
        <code className={`language-${language}`}>
          {code}
        </code>
      </pre>
      {incomplete && (
        <div className="incomplete-indicator">
          Code block incomplete...
        </div>
      )}
    </div>
  );
}
```

```css
.code-block.incomplete {
  border-right: 3px solid var(--warning-color, #f59e0b);
}

.incomplete-indicator {
  padding: 8px;
  background: var(--warning-bg, #fffbeb);
  color: var(--warning-text, #92400e);
  font-size: 0.75rem;
  font-style: italic;
}
```

---

## Streaming Considerations

### Detecting Incomplete Blocks During Stream

```javascript
function hasIncompleteCodeBlock(text) {
  const fences = text.match(/```/g) || [];
  return fences.length % 2 !== 0;
}

function closeIncompleteBlock(text) {
  if (hasIncompleteCodeBlock(text)) {
    return text + '\n```';
  }
  return text;
}
```

### Streaming Code Block Component

```jsx
function StreamingCodeBlock({ content, isStreaming }) {
  const processedContent = useMemo(() => {
    if (!isStreaming) return content;
    
    // Close any incomplete blocks for rendering
    return closeIncompleteBlock(content);
  }, [content, isStreaming]);
  
  const blocks = useMemo(() => {
    return parseCodeBlocks(processedContent);
  }, [processedContent]);
  
  return (
    <div className="code-blocks">
      {blocks.map((block, i) => (
        <CodeBlock
          key={i}
          language={block.language}
          code={block.code}
          incomplete={isStreaming && block.incomplete}
        />
      ))}
    </div>
  );
}
```

### Debounced Parsing

```javascript
function useDebouncedCodeBlocks(content, delay = 100) {
  const [blocks, setBlocks] = useState([]);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setBlocks(parseCodeBlocks(content));
    }, delay);
    
    return () => clearTimeout(timer);
  }, [content, delay]);
  
  return blocks;
}
```

---

## Complete Detection System

```javascript
class CodeBlockDetector {
  constructor(options = {}) {
    this.languageAliases = options.aliases || LANGUAGE_ALIASES;
    this.supportTildes = options.supportTildes !== false;
  }
  
  detect(markdown) {
    const results = {
      blocks: [],
      inlineCode: [],
      hasIncomplete: false
    };
    
    // Extract fenced blocks
    results.blocks = this.extractFencedBlocks(markdown);
    
    // Extract inline code
    results.inlineCode = this.extractInlineCode(markdown);
    
    // Check for incomplete
    results.hasIncomplete = this.hasIncompleteBlock(markdown);
    
    return results;
  }
  
  extractFencedBlocks(text) {
    const pattern = this.supportTildes
      ? /^(`{3,}|~{3,})(\w*)(.*?)\n([\s\S]*?)^\1/gm
      : /^`{3,}(\w*)(.*?)\n([\s\S]*?)^`{3,}/gm;
    
    const blocks = [];
    let match;
    
    while ((match = pattern.exec(text)) !== null) {
      const language = this.normalizeLanguage(match[2] || match[1]);
      const metadata = this.parseMetadata(match[3] || match[2]);
      const code = match[4] || match[3];
      
      blocks.push({
        language,
        metadata,
        code: code.trimEnd(),
        position: {
          start: match.index,
          end: match.index + match[0].length
        }
      });
    }
    
    return blocks;
  }
  
  extractInlineCode(text) {
    const pattern = /`([^`\n]+)`/g;
    const inline = [];
    let match;
    
    while ((match = pattern.exec(text)) !== null) {
      inline.push({
        code: match[1],
        position: {
          start: match.index,
          end: match.index + match[0].length
        }
      });
    }
    
    return inline;
  }
  
  normalizeLanguage(lang) {
    if (!lang) return 'text';
    const lower = lang.toLowerCase().trim();
    return this.languageAliases[lower] || lower;
  }
  
  parseMetadata(metaString) {
    if (!metaString) return {};
    
    const metadata = {};
    const pattern = /(\w+)=["']?([^"'\s]+)["']?/g;
    let match;
    
    while ((match = pattern.exec(metaString)) !== null) {
      metadata[match[1]] = match[2];
    }
    
    return metadata;
  }
  
  hasIncompleteBlock(text) {
    const backticks = (text.match(/```/g) || []).length;
    const tildes = this.supportTildes 
      ? (text.match(/~~~/g) || []).length 
      : 0;
    return (backticks % 2 !== 0) || (tildes % 2 !== 0);
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Handle both ``` and ~~~ fences | Only support backticks |
| Normalize language aliases | Use raw language strings |
| Detect incomplete blocks | Let parser crash on malformed |
| Debounce during streaming | Re-parse on every character |
| Preserve original code whitespace | Trim aggressively |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Regex misses nested backticks | Use line-by-line parser |
| Language not normalized | Map common aliases |
| Streaming causes flicker | Debounce + close incomplete |
| Indented blocks not detected | Support 4-space indent |
| Inline code inside blocks | Parse blocks first, then inline |

---

## Hands-on Exercise

### Your Task

Build a code block detector that:
1. Extracts fenced code blocks with language
2. Handles incomplete blocks during streaming
3. Normalizes language aliases
4. Separates inline from block code

### Requirements

1. Support both ``` and ~~~ fences
2. Map `js` → `javascript`, `py` → `python`, etc.
3. Return `incomplete: true` for unclosed blocks
4. Parse metadata like `title="file.js"`

<details>
<summary>💡 Hints (click to expand)</summary>

- Count fences to detect incomplete
- Use multiline regex flag (`/m`)
- Handle fence inside code (count matching pairs)
- Trim trailing whitespace from code

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `CodeBlockDetector` class in the Complete Detection System section above.

</details>

---

## Summary

✅ **Fenced blocks** use ``` or ~~~ with optional language  
✅ **Language tags** need normalization for consistency  
✅ **Inline vs block** requires different handling  
✅ **Malformed blocks** should degrade gracefully  
✅ **Streaming** needs debouncing and incomplete detection  
✅ **Metadata** can include title, highlights, etc.

---

## Further Reading

- [CommonMark Fenced Code Blocks](https://spec.commonmark.org/0.30/#fenced-code-blocks)
- [GitHub Flavored Markdown](https://github.github.com/gfm/#fenced-code-blocks)
- [react-markdown Code Handling](https://github.com/remarkjs/react-markdown#use-custom-components-syntax-highlight)

---

**Previous:** [Code Display Overview](./00-code-display-syntax-highlighting.md)  
**Next:** [Language Detection](./02-language-detection.md)

<!-- 
Sources Consulted:
- CommonMark spec: https://spec.commonmark.org/0.30/
- GFM spec: https://github.github.com/gfm/
- react-markdown: https://github.com/remarkjs/react-markdown
-->
