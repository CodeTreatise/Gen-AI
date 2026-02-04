---
title: "Line Numbers and Line Highlighting"
---

# Line Numbers and Line Highlighting

## Introduction

Line numbers help users reference specific parts of code, while line highlighting draws attention to important sections. These features are essential when AI explains code by referring to "line 5" or when marking changes in generated code.

In this lesson, we'll implement line numbers, line highlighting, and range selection for code blocks.

### What We'll Cover

- Adding line numbers to code blocks
- Highlighting specific lines
- Line range selection
- Handling long lines and wrapping
- Accessibility considerations

### Prerequisites

- [Syntax Highlighting Implementation](./03-syntax-highlighting.md)
- CSS positioning and flexbox
- Event handling in React

---

## Basic Line Numbers

### CSS-Based Line Numbers

```jsx
function CodeWithLineNumbers({ code, language }) {
  const lines = code.split('\n');
  
  return (
    <pre className="code-block">
      <code>
        {lines.map((line, index) => (
          <div key={index} className="code-line">
            <span className="line-number">{index + 1}</span>
            <span className="line-content">
              <HighlightedCode code={line} language={language} />
            </span>
          </div>
        ))}
      </code>
    </pre>
  );
}
```

```css
.code-block {
  background: #1e1e1e;
  border-radius: 8px;
  overflow-x: auto;
  margin: 0;
  padding: 0;
}

.code-line {
  display: flex;
  min-height: 1.5em;
}

.line-number {
  flex-shrink: 0;
  width: 3em;
  padding: 0 0.75em;
  text-align: right;
  color: #858585;
  background: #252526;
  user-select: none;  /* Prevent copying line numbers */
  border-right: 1px solid #333;
}

.line-content {
  flex: 1;
  padding: 0 1em;
  white-space: pre;
}
```

### Dynamic Width for Large Files

```jsx
function CodeWithDynamicLineNumbers({ code }) {
  const lines = code.split('\n');
  const lineCount = lines.length;
  const digitCount = String(lineCount).length;
  
  // Calculate width based on digit count
  const lineNumberWidth = `${digitCount + 1}em`;
  
  return (
    <pre className="code-block">
      <code>
        {lines.map((line, index) => (
          <div key={index} className="code-line">
            <span 
              className="line-number"
              style={{ width: lineNumberWidth }}
            >
              {index + 1}
            </span>
            <span className="line-content">{line || ' '}</span>
          </div>
        ))}
      </code>
    </pre>
  );
}
```

---

## Highlighting Specific Lines

### Props-Based Highlighting

```jsx
function HighlightedCodeBlock({ 
  code, 
  language, 
  highlightLines = [],
  highlightRanges = [] 
}) {
  const lines = code.split('\n');
  
  const isHighlighted = (lineNum) => {
    // Check individual lines
    if (highlightLines.includes(lineNum)) return true;
    
    // Check ranges like [3, 7] for lines 3-7
    return highlightRanges.some(
      ([start, end]) => lineNum >= start && lineNum <= end
    );
  };
  
  return (
    <pre className="code-block">
      <code>
        {lines.map((line, index) => {
          const lineNum = index + 1;
          const highlighted = isHighlighted(lineNum);
          
          return (
            <div 
              key={index} 
              className={`code-line ${highlighted ? 'highlighted' : ''}`}
            >
              <span className="line-number">{lineNum}</span>
              <span className="line-content">
                <HighlightedCode code={line} language={language} />
              </span>
            </div>
          );
        })}
      </code>
    </pre>
  );
}
```

### Highlight Styles

```css
.code-line.highlighted {
  background: rgba(255, 255, 0, 0.1);
}

.code-line.highlighted .line-number {
  background: rgba(255, 255, 0, 0.2);
  color: #ffd700;
}

/* Different highlight types */
.code-line.added {
  background: rgba(0, 255, 0, 0.1);
}

.code-line.added .line-number {
  background: rgba(0, 255, 0, 0.2);
}

.code-line.removed {
  background: rgba(255, 0, 0, 0.1);
}

.code-line.removed .line-number {
  background: rgba(255, 0, 0, 0.2);
}

.code-line.focus {
  background: rgba(0, 150, 255, 0.15);
  box-shadow: inset 3px 0 0 #0096ff;
}
```

### Parsing Highlight Syntax

Support syntax like `{1,3-5,8}` for highlighting:

```javascript
function parseHighlightSpec(spec) {
  if (!spec) return { lines: [], ranges: [] };
  
  // Remove braces and split by comma
  const parts = spec.replace(/[{}]/g, '').split(',');
  const lines = [];
  const ranges = [];
  
  for (const part of parts) {
    const trimmed = part.trim();
    
    if (trimmed.includes('-')) {
      // Range like "3-5"
      const [start, end] = trimmed.split('-').map(Number);
      ranges.push([start, end]);
    } else {
      // Single line
      lines.push(Number(trimmed));
    }
  }
  
  return { lines, ranges };
}

// Usage
const { lines, ranges } = parseHighlightSpec('{1,3-5,8}');
// lines: [1, 8]
// ranges: [[3, 5]]
```

---

## Interactive Line Selection

### Click to Select

```jsx
function SelectableCodeBlock({ code, language, onLineSelect }) {
  const [selectedLines, setSelectedLines] = useState(new Set());
  const lines = code.split('\n');
  
  const handleLineClick = (lineNum, event) => {
    setSelectedLines(prev => {
      const next = new Set(prev);
      
      if (event.shiftKey && prev.size > 0) {
        // Shift-click: select range
        const lastSelected = Math.max(...prev);
        const start = Math.min(lastSelected, lineNum);
        const end = Math.max(lastSelected, lineNum);
        for (let i = start; i <= end; i++) {
          next.add(i);
        }
      } else if (event.ctrlKey || event.metaKey) {
        // Ctrl/Cmd-click: toggle single line
        if (next.has(lineNum)) {
          next.delete(lineNum);
        } else {
          next.add(lineNum);
        }
      } else {
        // Plain click: select only this line
        next.clear();
        next.add(lineNum);
      }
      
      return next;
    });
    
    onLineSelect?.(Array.from(selectedLines));
  };
  
  return (
    <pre className="code-block selectable">
      <code>
        {lines.map((line, index) => {
          const lineNum = index + 1;
          const isSelected = selectedLines.has(lineNum);
          
          return (
            <div 
              key={index}
              className={`code-line ${isSelected ? 'selected' : ''}`}
              onClick={(e) => handleLineClick(lineNum, e)}
            >
              <span className="line-number">{lineNum}</span>
              <span className="line-content">{line || ' '}</span>
            </div>
          );
        })}
      </code>
    </pre>
  );
}
```

```css
.code-block.selectable .code-line {
  cursor: pointer;
}

.code-block.selectable .code-line:hover {
  background: rgba(255, 255, 255, 0.05);
}

.code-line.selected {
  background: rgba(0, 120, 255, 0.2);
}

.code-line.selected .line-number {
  background: rgba(0, 120, 255, 0.3);
  color: #5cb3ff;
}
```

### Generate Shareable Link

```jsx
function CodeBlockWithPermalink({ code, language }) {
  const [selectedLines, setSelectedLines] = useState([]);
  
  const handleCopyPermalink = () => {
    const sorted = [...selectedLines].sort((a, b) => a - b);
    const lineSpec = compressLineSpec(sorted);
    
    const url = new URL(window.location.href);
    url.hash = `L${lineSpec}`;
    
    navigator.clipboard.writeText(url.toString());
  };
  
  return (
    <div className="code-block-container">
      {selectedLines.length > 0 && (
        <button onClick={handleCopyPermalink}>
          Copy link to lines {selectedLines.join(', ')}
        </button>
      )}
      
      <SelectableCodeBlock
        code={code}
        language={language}
        onLineSelect={setSelectedLines}
      />
    </div>
  );
}

// Compress [1,2,3,5,6,8] to "1-3,5-6,8"
function compressLineSpec(lines) {
  if (lines.length === 0) return '';
  
  const ranges = [];
  let start = lines[0];
  let end = lines[0];
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i] === end + 1) {
      end = lines[i];
    } else {
      ranges.push(start === end ? `${start}` : `${start}-${end}`);
      start = lines[i];
      end = lines[i];
    }
  }
  
  ranges.push(start === end ? `${start}` : `${start}-${end}`);
  return ranges.join(',');
}
```

---

## Line Wrapping Options

### No Wrap (Horizontal Scroll)

```css
.code-block.no-wrap {
  overflow-x: auto;
}

.code-block.no-wrap .line-content {
  white-space: pre;
  overflow: visible;
}
```

### Soft Wrap

```css
.code-block.soft-wrap .line-content {
  white-space: pre-wrap;
  word-break: break-all;
}

/* Indent wrapped lines */
.code-block.soft-wrap .code-line {
  position: relative;
}

.code-block.soft-wrap .line-content {
  padding-left: 2em;
  text-indent: -2em;
}
```

### Toggle Component

```jsx
function CodeBlockWithWrapToggle({ code, language }) {
  const [wrap, setWrap] = useState(false);
  
  return (
    <div className="code-container">
      <div className="code-toolbar">
        <button 
          onClick={() => setWrap(!wrap)}
          className={wrap ? 'active' : ''}
          title={wrap ? 'Disable word wrap' : 'Enable word wrap'}
        >
          <WrapIcon />
        </button>
      </div>
      
      <pre className={`code-block ${wrap ? 'soft-wrap' : 'no-wrap'}`}>
        <code>
          {/* ... lines */}
        </code>
      </pre>
    </div>
  );
}
```

---

## Gutter Features

### Collapsible Regions

```jsx
function CollapsibleCodeBlock({ code, language }) {
  const [collapsedRanges, setCollapsedRanges] = useState(new Set());
  const lines = code.split('\n');
  
  // Simple brace-based folding detection
  const foldingPoints = detectFoldingPoints(code);
  
  const toggleFold = (startLine) => {
    setCollapsedRanges(prev => {
      const next = new Set(prev);
      if (next.has(startLine)) {
        next.delete(startLine);
      } else {
        next.add(startLine);
      }
      return next;
    });
  };
  
  const isLineVisible = (lineNum) => {
    for (const startLine of collapsedRanges) {
      const fold = foldingPoints.find(f => f.start === startLine);
      if (fold && lineNum > fold.start && lineNum <= fold.end) {
        return false;
      }
    }
    return true;
  };
  
  return (
    <pre className="code-block">
      <code>
        {lines.map((line, index) => {
          const lineNum = index + 1;
          
          if (!isLineVisible(lineNum)) return null;
          
          const foldStart = foldingPoints.find(f => f.start === lineNum);
          const isCollapsed = collapsedRanges.has(lineNum);
          
          return (
            <div key={index} className="code-line">
              <span className="fold-gutter">
                {foldStart && (
                  <button 
                    onClick={() => toggleFold(lineNum)}
                    className="fold-toggle"
                  >
                    {isCollapsed ? '▶' : '▼'}
                  </button>
                )}
              </span>
              <span className="line-number">{lineNum}</span>
              <span className="line-content">
                {line}
                {isCollapsed && (
                  <span className="fold-placeholder">
                    {' '}... {foldStart.end - foldStart.start} lines
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </code>
    </pre>
  );
}
```

### Diff-Style Markers

```jsx
function DiffCodeBlock({ code }) {
  const lines = code.split('\n');
  
  const getLineType = (line) => {
    if (line.startsWith('+')) return 'added';
    if (line.startsWith('-')) return 'removed';
    if (line.startsWith('@')) return 'meta';
    return 'unchanged';
  };
  
  return (
    <pre className="code-block diff">
      <code>
        {lines.map((line, index) => {
          const type = getLineType(line);
          
          return (
            <div key={index} className={`code-line ${type}`}>
              <span className="diff-marker">
                {type === 'added' && '+'}
                {type === 'removed' && '-'}
                {type === 'unchanged' && ' '}
              </span>
              <span className="line-number">{index + 1}</span>
              <span className="line-content">{line.slice(1) || ' '}</span>
            </div>
          );
        })}
      </code>
    </pre>
  );
}
```

```css
.diff .code-line.added {
  background: rgba(46, 160, 67, 0.15);
}

.diff .code-line.removed {
  background: rgba(248, 81, 73, 0.15);
}

.diff .code-line.meta {
  background: rgba(56, 139, 253, 0.15);
  color: #58a6ff;
}

.diff-marker {
  width: 1.5em;
  text-align: center;
  color: inherit;
}

.diff .added .diff-marker { color: #3fb950; }
.diff .removed .diff-marker { color: #f85149; }
```

---

## Accessibility

### Screen Reader Support

```jsx
function AccessibleCodeBlock({ code, language, highlightLines = [] }) {
  const lines = code.split('\n');
  const highlightCount = highlightLines.length;
  
  return (
    <div 
      className="code-block"
      role="region"
      aria-label={`${language} code block with ${lines.length} lines`}
    >
      {highlightCount > 0 && (
        <div className="sr-only">
          {highlightCount} line{highlightCount > 1 ? 's' : ''} highlighted: 
          {highlightLines.join(', ')}
        </div>
      )}
      
      <pre>
        <code>
          {lines.map((line, index) => {
            const lineNum = index + 1;
            const isHighlighted = highlightLines.includes(lineNum);
            
            return (
              <div 
                key={index}
                className={`code-line ${isHighlighted ? 'highlighted' : ''}`}
                aria-label={`Line ${lineNum}${isHighlighted ? ', highlighted' : ''}`}
              >
                <span 
                  className="line-number" 
                  aria-hidden="true"
                >
                  {lineNum}
                </span>
                <span className="line-content">{line || ' '}</span>
              </div>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
```

### Keyboard Navigation

```jsx
function KeyboardNavigableCode({ code, language }) {
  const [focusedLine, setFocusedLine] = useState(null);
  const lines = code.split('\n');
  const lineRefs = useRef([]);
  
  const handleKeyDown = (e) => {
    if (focusedLine === null) return;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (focusedLine < lines.length) {
          setFocusedLine(focusedLine + 1);
          lineRefs.current[focusedLine]?.focus();
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (focusedLine > 1) {
          setFocusedLine(focusedLine - 1);
          lineRefs.current[focusedLine - 2]?.focus();
        }
        break;
      case 'Home':
        e.preventDefault();
        setFocusedLine(1);
        lineRefs.current[0]?.focus();
        break;
      case 'End':
        e.preventDefault();
        setFocusedLine(lines.length);
        lineRefs.current[lines.length - 1]?.focus();
        break;
    }
  };
  
  return (
    <pre className="code-block" onKeyDown={handleKeyDown}>
      <code>
        {lines.map((line, index) => (
          <div
            key={index}
            ref={el => lineRefs.current[index] = el}
            className={`code-line ${focusedLine === index + 1 ? 'focused' : ''}`}
            tabIndex={index === 0 ? 0 : -1}
            onFocus={() => setFocusedLine(index + 1)}
          >
            <span className="line-number">{index + 1}</span>
            <span className="line-content">{line || ' '}</span>
          </div>
        ))}
      </code>
    </pre>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `user-select: none` on line numbers | Let users copy line numbers |
| Show dynamic width for large files | Use fixed width that clips numbers |
| Provide keyboard navigation | Require mouse for all interactions |
| Add ARIA labels for highlights | Rely only on visual highlighting |
| Support shift+click for ranges | Force individual clicks |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Line numbers included in copy | Use `user-select: none` |
| Highlights not visible in light mode | Adjust highlight colors per theme |
| Wrapped lines confuse numbering | Indent wrapped continuation |
| Horizontal scroll hides content | Add scroll indicator |
| Focus trap in code block | Allow Tab to exit |

---

## Hands-on Exercise

### Your Task

Build a `LineHighlightedCode` component that:
1. Displays line numbers
2. Accepts a `highlight` prop like `{1,3-5,8}`
3. Parses the spec and highlights those lines
4. Supports click-to-select lines

### Requirements

1. Parse highlight string into lines and ranges
2. Apply different styles for highlighted vs selected
3. Prevent line number selection
4. Support shift-click for range selection

<details>
<summary>💡 Hints (click to expand)</summary>

- Split spec by comma, then check for `-`
- Track selected lines in state
- Use CSS `user-select: none` for line numbers
- Compare clicked line with last selected for shift-click

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

Combine the `parseHighlightSpec` function with `SelectableCodeBlock` from above, merging `highlightLines` with `selectedLines` in the className logic.

</details>

---

## Summary

✅ **Line numbers** improve code reference and navigation  
✅ **Dynamic width** handles large file line counts  
✅ **Highlight parsing** supports `{1,3-5}` syntax  
✅ **Click selection** enables user-driven highlights  
✅ **Permalinks** share specific line references  
✅ **Accessibility** requires ARIA and keyboard support

---

## Further Reading

- [GitHub Line Linking](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-a-permanent-link-to-a-code-snippet)
- [VS Code Line Numbers](https://code.visualstudio.com/docs/getstarted/userinterface#_line-numbers)
- [ARIA for Code Editors](https://www.w3.org/WAI/ARIA/apg/patterns/)

---

**Previous:** [Syntax Highlighting Implementation](./03-syntax-highlighting.md)  
**Next:** [Copy Functionality](./05-copy-functionality.md)

<!-- 
Sources Consulted:
- CSS user-select: https://developer.mozilla.org/en-US/docs/Web/CSS/user-select
- GitHub code linking: https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files
- ARIA practices: https://www.w3.org/WAI/ARIA/apg/
-->
