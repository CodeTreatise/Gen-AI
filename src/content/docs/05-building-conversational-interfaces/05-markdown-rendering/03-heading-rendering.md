---
title: "Heading Rendering"
---

# Heading Rendering

## Introduction

Headings structure AI responses, making long content scannable and navigable. But headings in chat messages differ from document headings—they need anchor links for sharing, consistent styling that fits the chat context, and sometimes dynamic table of contents generation.

In this lesson, we'll implement heading rendering with navigation features.

### What We'll Cover

- Heading levels (h1-h6)
- Anchor links for headings
- Table of contents generation
- Styling consistency
- Accessibility considerations

### Prerequisites

- [Formatted Text Rendering](./02-formatted-text.md)
- CSS positioning
- React refs and IDs

---

## Heading Levels

### Basic Heading Components

```jsx
import ReactMarkdown from 'react-markdown';

function HeadingMarkdown({ content }) {
  return (
    <ReactMarkdown
      components={{
        h1: (props) => <Heading level={1} {...props} />,
        h2: (props) => <Heading level={2} {...props} />,
        h3: (props) => <Heading level={3} {...props} />,
        h4: (props) => <Heading level={4} {...props} />,
        h5: (props) => <Heading level={5} {...props} />,
        h6: (props) => <Heading level={6} {...props} />
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function Heading({ level, children, ...props }) {
  const Tag = `h${level}`;
  const id = slugify(getTextContent(children));
  
  return (
    <Tag id={id} className={`md-heading md-h${level}`} {...props}>
      {children}
    </Tag>
  );
}

// Extract text content from React children
function getTextContent(children) {
  return React.Children.toArray(children)
    .map(child => {
      if (typeof child === 'string') return child;
      if (child?.props?.children) return getTextContent(child.props.children);
      return '';
    })
    .join('');
}

// Create URL-safe slug
function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
}
```

### Heading Styles

```css
.md-heading {
  font-family: var(--heading-font, inherit);
  font-weight: 600;
  line-height: 1.3;
  color: var(--heading-color, #111827);
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}

.md-heading:first-child {
  margin-top: 0;
}

.md-h1 {
  font-size: 1.875rem; /* 30px */
  font-weight: 700;
  border-bottom: 1px solid var(--border-color, #e5e7eb);
  padding-bottom: 0.3em;
}

.md-h2 {
  font-size: 1.5rem; /* 24px */
  font-weight: 600;
}

.md-h3 {
  font-size: 1.25rem; /* 20px */
}

.md-h4 {
  font-size: 1.125rem; /* 18px */
}

.md-h5 {
  font-size: 1rem; /* 16px */
}

.md-h6 {
  font-size: 0.875rem; /* 14px */
  color: var(--text-secondary, #6b7280);
}
```

### Chat-Context Headings

In chat, h1 might be too large. Shift levels down:

```jsx
function ChatHeading({ level, children, ...props }) {
  // Shift headings: h1->h3, h2->h4, etc.
  const adjustedLevel = Math.min(level + 2, 6);
  const Tag = `h${adjustedLevel}`;
  const id = slugify(getTextContent(children));
  
  return (
    <Tag 
      id={id} 
      className={`md-heading md-h${adjustedLevel}`}
      // Keep original level for semantics via aria
      aria-level={level}
      {...props}
    >
      {children}
    </Tag>
  );
}
```

```css
/* Chat-specific heading sizes */
.chat-message .md-h3 { font-size: 1.25rem; }
.chat-message .md-h4 { font-size: 1.125rem; }
.chat-message .md-h5 { font-size: 1rem; }
.chat-message .md-h6 { font-size: 0.875rem; }
```

---

## Anchor Links

### Heading with Anchor

```jsx
function HeadingWithAnchor({ level, children }) {
  const Tag = `h${level}`;
  const text = getTextContent(children);
  const id = slugify(text);
  
  return (
    <Tag id={id} className={`md-heading md-h${level} md-heading-anchor`}>
      {children}
      <a 
        href={`#${id}`}
        className="heading-link"
        aria-label={`Link to ${text}`}
      >
        <LinkIcon />
      </a>
    </Tag>
  );
}

function LinkIcon() {
  return (
    <svg 
      width="16" 
      height="16" 
      viewBox="0 0 16 16" 
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25z"/>
      <path d="M8.225 12.725a.75.75 0 01-1.06-1.06l-1.25 1.25a2 2 0 11-2.83-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25z"/>
    </svg>
  );
}
```

```css
.md-heading-anchor {
  position: relative;
}

.heading-link {
  position: absolute;
  left: -24px;
  top: 50%;
  transform: translateY(-50%);
  opacity: 0;
  color: var(--text-muted, #9ca3af);
  transition: opacity 0.2s;
  padding: 4px;
}

.md-heading-anchor:hover .heading-link,
.heading-link:focus {
  opacity: 1;
}

.heading-link:hover {
  color: var(--primary-color, #3b82f6);
}

/* Scroll margin for anchor navigation */
.md-heading[id] {
  scroll-margin-top: 80px; /* Account for fixed header */
}
```

### Copy Link on Click

```jsx
function HeadingWithCopyLink({ level, children }) {
  const [copied, setCopied] = useState(false);
  const text = getTextContent(children);
  const id = slugify(text);
  
  const handleCopyLink = async () => {
    const url = `${window.location.origin}${window.location.pathname}#${id}`;
    await navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const Tag = `h${level}`;
  
  return (
    <Tag id={id} className={`md-heading md-h${level} md-heading-anchor`}>
      {children}
      <button 
        onClick={handleCopyLink}
        className="heading-link-btn"
        aria-label={copied ? 'Copied!' : `Copy link to ${text}`}
      >
        {copied ? <CheckIcon /> : <LinkIcon />}
      </button>
    </Tag>
  );
}
```

```css
.heading-link-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  margin-left: 8px;
  opacity: 0;
  transition: opacity 0.2s, color 0.2s;
  color: var(--text-muted, #9ca3af);
}

.md-heading-anchor:hover .heading-link-btn,
.heading-link-btn:focus {
  opacity: 1;
}

.heading-link-btn[aria-label="Copied!"] {
  color: var(--success-color, #10b981);
  opacity: 1;
}
```

---

## Table of Contents

### Extract Headings

```jsx
function useTableOfContents(content) {
  return useMemo(() => {
    const headingRegex = /^(#{1,6})\s+(.+)$/gm;
    const headings = [];
    let match;
    
    while ((match = headingRegex.exec(content)) !== null) {
      const level = match[1].length;
      const text = match[2].trim();
      const id = slugify(text);
      
      headings.push({ level, text, id });
    }
    
    return headings;
  }, [content]);
}
```

### TOC Component

```jsx
function TableOfContents({ headings, activeId }) {
  if (headings.length < 2) return null;
  
  return (
    <nav className="toc" aria-label="Table of contents">
      <h4 className="toc-title">Contents</h4>
      <ul className="toc-list">
        {headings.map((heading) => (
          <li 
            key={heading.id}
            className={`toc-item toc-level-${heading.level}`}
            style={{ '--indent': (heading.level - 1) * 12 }}
          >
            <a 
              href={`#${heading.id}`}
              className={activeId === heading.id ? 'active' : ''}
            >
              {heading.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

```css
.toc {
  position: sticky;
  top: 80px;
  max-height: calc(100vh - 100px);
  overflow-y: auto;
  padding: 16px;
  background: var(--toc-bg, #f9fafb);
  border-radius: 8px;
  border: 1px solid var(--border-color, #e5e7eb);
}

.toc-title {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted, #6b7280);
  margin: 0 0 12px 0;
}

.toc-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.toc-item {
  padding-left: calc(var(--indent, 0) * 1px);
}

.toc-item a {
  display: block;
  padding: 6px 8px;
  font-size: 0.875rem;
  color: var(--text-secondary, #4b5563);
  text-decoration: none;
  border-radius: 4px;
  transition: background 0.2s, color 0.2s;
}

.toc-item a:hover {
  background: var(--toc-hover, #e5e7eb);
}

.toc-item a.active {
  background: var(--primary-light, #dbeafe);
  color: var(--primary-color, #2563eb);
  font-weight: 500;
}
```

### Active Heading Tracking

```jsx
function useActiveHeading(headingIds) {
  const [activeId, setActiveId] = useState(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      {
        rootMargin: '-80px 0px -80% 0px',
        threshold: 0
      }
    );
    
    headingIds.forEach((id) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
    
    return () => observer.disconnect();
  }, [headingIds]);
  
  return activeId;
}
```

### Complete TOC Integration

```jsx
function ContentWithTOC({ content }) {
  const headings = useTableOfContents(content);
  const headingIds = headings.map(h => h.id);
  const activeId = useActiveHeading(headingIds);
  
  return (
    <div className="content-with-toc">
      {headings.length >= 3 && (
        <aside className="toc-sidebar">
          <TableOfContents headings={headings} activeId={activeId} />
        </aside>
      )}
      
      <article className="content">
        <HeadingMarkdown content={content} />
      </article>
    </div>
  );
}
```

```css
.content-with-toc {
  display: grid;
  grid-template-columns: 1fr 220px;
  gap: 32px;
  align-items: start;
}

.toc-sidebar {
  order: 2;
}

.content {
  order: 1;
  min-width: 0;
}

@media (max-width: 1024px) {
  .content-with-toc {
    grid-template-columns: 1fr;
  }
  
  .toc-sidebar {
    display: none;
  }
}
```

---

## Collapsible TOC

### Mobile-Friendly TOC

```jsx
function CollapsibleTOC({ headings, activeId }) {
  const [isOpen, setIsOpen] = useState(false);
  
  if (headings.length < 2) return null;
  
  return (
    <div className="collapsible-toc">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="toc-toggle"
        aria-expanded={isOpen}
      >
        <span>Contents</span>
        <ChevronIcon direction={isOpen ? 'up' : 'down'} />
      </button>
      
      {isOpen && (
        <nav className="toc-dropdown" aria-label="Table of contents">
          <ul className="toc-list">
            {headings.map((heading) => (
              <li key={heading.id} className={`toc-level-${heading.level}`}>
                <a 
                  href={`#${heading.id}`}
                  onClick={() => setIsOpen(false)}
                >
                  {heading.text}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </div>
  );
}
```

```css
.collapsible-toc {
  position: relative;
  margin-bottom: 16px;
}

.toc-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--toc-bg, #f3f4f6);
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
}

.toc-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  margin-top: 4px;
  background: white;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  z-index: 50;
  max-height: 300px;
  overflow-y: auto;
  animation: dropdown-in 0.2s ease-out;
}

@keyframes dropdown-in {
  from {
    opacity: 0;
    transform: translateY(-8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

---

## Accessibility

### Screen Reader Considerations

```jsx
function AccessibleHeading({ level, children, ...props }) {
  const Tag = `h${level}`;
  const text = getTextContent(children);
  const id = slugify(text);
  
  return (
    <Tag 
      id={id} 
      className={`md-heading md-h${level}`}
      tabIndex="-1"  // Allow programmatic focus
      {...props}
    >
      {children}
      
      {/* Hidden link for screen readers */}
      <a 
        href={`#${id}`}
        className="sr-only-focusable"
        aria-label={`Link to section: ${text}`}
      >
        §
      </a>
    </Tag>
  );
}
```

```css
.sr-only-focusable {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0 0 0 8px;
  overflow: visible;
  clip: auto;
}
```

### Skip to Content

```jsx
function SkipToHeading({ headingId, label }) {
  return (
    <a 
      href={`#${headingId}`}
      className="skip-link"
    >
      Skip to {label}
    </a>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use unique IDs for headings | Duplicate IDs break anchors |
| Add scroll-margin for fixed headers | Let anchors hide behind headers |
| Track active heading for long content | Leave TOC without active indicator |
| Provide accessible anchor links | Hide all navigation from screen readers |
| Scale headings for chat context | Use document-sized h1 in messages |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Duplicate slugs from same text | Append numbers: `slug`, `slug-1`, `slug-2` |
| TOC shows during streaming | Wait for content to stabilize |
| Anchor links break on special chars | Sanitize slugify function |
| Active heading wrong on scroll up | Use proper IntersectionObserver margins |
| Heading IDs conflict with page IDs | Prefix with message ID |

### Unique Slug Generation

```jsx
function useUniqueSlugify() {
  const usedSlugs = useRef(new Set());
  
  return useCallback((text) => {
    let slug = slugify(text);
    let counter = 0;
    let uniqueSlug = slug;
    
    while (usedSlugs.current.has(uniqueSlug)) {
      counter++;
      uniqueSlug = `${slug}-${counter}`;
    }
    
    usedSlugs.current.add(uniqueSlug);
    return uniqueSlug;
  }, []);
}
```

---

## Hands-on Exercise

### Your Task

Build a heading system that:
1. Renders h1-h6 with consistent styling
2. Adds anchor links on hover
3. Generates a table of contents
4. Tracks active heading on scroll

### Requirements

1. Slugified heading IDs
2. Copy link button with feedback
3. TOC with indentation by level
4. Active heading highlight

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `IntersectionObserver` for active tracking
- Store copied state with setTimeout reset
- CSS `--indent` custom property for levels
- Scroll-margin-top for anchor positioning

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `ContentWithTOC` component with `useActiveHeading` and `TableOfContents` components above.

</details>

---

## Summary

✅ **Heading levels** need scaling for chat context  
✅ **Anchor links** enable content sharing  
✅ **Table of contents** improves navigation  
✅ **Active tracking** shows reading position  
✅ **Unique IDs** prevent anchor conflicts  
✅ **Accessibility** requires proper labeling

---

## Further Reading

- [MDN Heading Elements](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements)
- [IntersectionObserver API](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API)
- [WCAG Headings](https://www.w3.org/WAI/tutorials/page-structure/headings/)

---

**Previous:** [Formatted Text Rendering](./02-formatted-text.md)  
**Next:** [Link Handling](./04-link-handling.md)

<!-- 
Sources Consulted:
- MDN Heading Elements: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements
- MDN IntersectionObserver: https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API
- WCAG Headings: https://www.w3.org/WAI/tutorials/page-structure/headings/
-->
