---
title: "Table Rendering"
---

# Table Rendering

## Introduction

Tables in AI responses present structured data—comparisons, specifications, or formatted results. But markdown tables need special handling in chat interfaces: responsive design for mobile, scrollable containers for wide tables, and consistent styling that matches your design system.

In this lesson, we'll implement beautiful, responsive table rendering.

### What We'll Cover

- Basic table styling
- Responsive table patterns
- Scrollable table containers
- Header styling and sticky headers
- Accessibility considerations

### Prerequisites

- [Image Embedding](./05-image-embedding.md)
- CSS table properties
- Flexbox/Grid basics

---

## Basic Table Styling

### Default Table Component

```jsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function MarkdownWithTables({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}  // Required for tables
      components={{
        table: ({ children }) => (
          <div className="table-wrapper">
            <table className="md-table">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="md-thead">{children}</thead>
        ),
        tbody: ({ children }) => (
          <tbody className="md-tbody">{children}</tbody>
        ),
        tr: ({ children }) => (
          <tr className="md-tr">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="md-th">{children}</th>
        ),
        td: ({ children }) => (
          <td className="md-td">{children}</td>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
```

### Base Table Styles

```css
.table-wrapper {
  margin: 16px 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.md-table {
  width: 100%;
  border-collapse: collapse;
  border-spacing: 0;
  font-size: 0.875rem;
}

.md-th,
.md-td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-color, #e5e7eb);
}

.md-th {
  font-weight: 600;
  background: var(--header-bg, #f9fafb);
  color: var(--text-primary, #111827);
}

.md-td {
  color: var(--text-secondary, #374151);
}

.md-tbody .md-tr:hover {
  background: var(--hover-bg, #f3f4f6);
}

/* Last row no border */
.md-tbody .md-tr:last-child .md-td {
  border-bottom: none;
}
```

### Bordered Table Variant

```css
.md-table.bordered {
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 8px;
  overflow: hidden;
}

.md-table.bordered .md-th,
.md-table.bordered .md-td {
  border: 1px solid var(--border-color, #e5e7eb);
}
```

### Striped Rows

```css
.md-table.striped .md-tbody .md-tr:nth-child(even) {
  background: var(--stripe-bg, #f9fafb);
}

.md-table.striped .md-tbody .md-tr:nth-child(even):hover {
  background: var(--stripe-hover, #f3f4f6);
}
```

---

## Responsive Tables

### Horizontal Scroll

```jsx
function ResponsiveTable({ children }) {
  return (
    <div className="table-scroll-container">
      <div className="table-scroll-wrapper">
        <table className="md-table">{children}</table>
      </div>
      <div className="scroll-indicator" aria-hidden="true" />
    </div>
  );
}
```

```css
.table-scroll-container {
  position: relative;
  margin: 16px 0;
}

.table-scroll-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
}

/* Custom scrollbar */
.table-scroll-wrapper::-webkit-scrollbar {
  height: 6px;
}

.table-scroll-wrapper::-webkit-scrollbar-track {
  background: var(--scrollbar-track, #f1f1f1);
  border-radius: 3px;
}

.table-scroll-wrapper::-webkit-scrollbar-thumb {
  background: var(--scrollbar-thumb, #c1c1c1);
  border-radius: 3px;
}

/* Scroll shadow indicator */
.table-scroll-container::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 6px;  /* Above scrollbar */
  width: 40px;
  background: linear-gradient(to right, transparent, rgba(0,0,0,0.05));
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
}

.table-scroll-container.can-scroll-right::after {
  opacity: 1;
}
```

### Scroll State Detection

```jsx
function ScrollableTable({ children }) {
  const wrapperRef = useRef(null);
  const [canScrollRight, setCanScrollRight] = useState(false);
  
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    
    const checkScroll = () => {
      const { scrollLeft, scrollWidth, clientWidth } = wrapper;
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 5);
    };
    
    checkScroll();
    wrapper.addEventListener('scroll', checkScroll);
    window.addEventListener('resize', checkScroll);
    
    return () => {
      wrapper.removeEventListener('scroll', checkScroll);
      window.removeEventListener('resize', checkScroll);
    };
  }, []);
  
  return (
    <div className={`table-scroll-container ${canScrollRight ? 'can-scroll-right' : ''}`}>
      <div ref={wrapperRef} className="table-scroll-wrapper">
        <table className="md-table">{children}</table>
      </div>
    </div>
  );
}
```

### Card Layout on Mobile

```jsx
function ResponsiveCardTable({ headers, rows }) {
  return (
    <>
      {/* Standard table for desktop */}
      <table className="md-table desktop-only">
        <thead>
          <tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => <td key={j}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Card layout for mobile */}
      <div className="table-cards mobile-only">
        {rows.map((row, i) => (
          <div key={i} className="table-card">
            {row.map((cell, j) => (
              <div key={j} className="card-row">
                <span className="card-label">{headers[j]}</span>
                <span className="card-value">{cell}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}
```

```css
.desktop-only {
  display: table;
}

.mobile-only {
  display: none;
}

@media (max-width: 640px) {
  .desktop-only {
    display: none;
  }
  
  .mobile-only {
    display: block;
  }
}

.table-cards {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.table-card {
  background: var(--card-bg, white);
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 8px;
  padding: 12px;
}

.card-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-light, #f3f4f6);
}

.card-row:last-child {
  border-bottom: none;
}

.card-label {
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.75rem;
  text-transform: uppercase;
}

.card-value {
  text-align: right;
}
```

---

## Sticky Headers

### Sticky Header for Long Tables

```jsx
function StickyHeaderTable({ children, maxHeight = '400px' }) {
  return (
    <div 
      className="table-scroll-container sticky-header"
      style={{ maxHeight }}
    >
      <table className="md-table">{children}</table>
    </div>
  );
}
```

```css
.table-scroll-container.sticky-header {
  overflow: auto;
}

.sticky-header .md-thead {
  position: sticky;
  top: 0;
  z-index: 1;
}

.sticky-header .md-th {
  background: var(--header-bg, #f9fafb);
  /* Shadow to indicate sticky state */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
```

### Sticky First Column

```css
.md-table.sticky-col .md-th:first-child,
.md-table.sticky-col .md-td:first-child {
  position: sticky;
  left: 0;
  z-index: 1;
  background: var(--cell-bg, white);
  box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05);
}

.md-table.sticky-col .md-th:first-child {
  background: var(--header-bg, #f9fafb);
  z-index: 2;  /* Above other sticky elements */
}
```

---

## Header Styling

### Sortable Headers

```jsx
function SortableTable({ columns, data, onSort }) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  
  const handleSort = (key) => {
    const direction = sortConfig.key === key && sortConfig.direction === 'asc' 
      ? 'desc' 
      : 'asc';
    setSortConfig({ key, direction });
    onSort?.(key, direction);
  };
  
  return (
    <table className="md-table sortable">
      <thead>
        <tr>
          {columns.map((col) => (
            <th 
              key={col.key}
              onClick={() => handleSort(col.key)}
              className={`md-th sortable ${sortConfig.key === col.key ? 'sorted' : ''}`}
              aria-sort={
                sortConfig.key === col.key 
                  ? sortConfig.direction === 'asc' ? 'ascending' : 'descending'
                  : 'none'
              }
            >
              <span className="th-content">
                {col.label}
                <SortIcon 
                  active={sortConfig.key === col.key}
                  direction={sortConfig.direction}
                />
              </span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            {columns.map((col) => (
              <td key={col.key} className="md-td">{row[col.key]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SortIcon({ active, direction }) {
  return (
    <span className={`sort-icon ${active ? 'active' : ''}`} aria-hidden="true">
      {direction === 'asc' ? '↑' : '↓'}
    </span>
  );
}
```

```css
.md-th.sortable {
  cursor: pointer;
  user-select: none;
}

.md-th.sortable:hover {
  background: var(--header-hover, #f3f4f6);
}

.th-content {
  display: flex;
  align-items: center;
  gap: 4px;
}

.sort-icon {
  opacity: 0.3;
  transition: opacity 0.2s;
}

.md-th.sortable:hover .sort-icon,
.sort-icon.active {
  opacity: 1;
}
```

### Column Alignment

```jsx
function AlignedCell({ align = 'left', children }) {
  return (
    <td className={`md-td align-${align}`}>{children}</td>
  );
}
```

```css
.md-td.align-left { text-align: left; }
.md-td.align-center { text-align: center; }
.md-td.align-right { text-align: right; }

/* Numeric columns */
.md-td.numeric {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
```

---

## Accessibility

### Accessible Table Structure

```jsx
function AccessibleTable({ caption, headers, rows }) {
  return (
    <table className="md-table" role="table">
      {caption && <caption className="md-caption">{caption}</caption>}
      
      <thead role="rowgroup">
        <tr role="row">
          {headers.map((header, i) => (
            <th key={i} role="columnheader" scope="col">
              {header}
            </th>
          ))}
        </tr>
      </thead>
      
      <tbody role="rowgroup">
        {rows.map((row, i) => (
          <tr key={i} role="row">
            {row.map((cell, j) => (
              <td key={j} role="cell">{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

```css
.md-caption {
  caption-side: top;
  padding: 8px;
  font-weight: 600;
  text-align: left;
  color: var(--text-primary);
}

/* Visible focus for keyboard navigation */
.md-table:focus-within {
  outline: 2px solid var(--focus-color, #3b82f6);
  outline-offset: 2px;
}

.md-td:focus,
.md-th:focus {
  outline: 2px solid var(--focus-color, #3b82f6);
  outline-offset: -2px;
}
```

### Screen Reader Enhancements

```jsx
function TableWithSummary({ summary, children }) {
  return (
    <div className="table-container">
      {/* Screen reader only summary */}
      <p className="sr-only">{summary}</p>
      
      <table className="md-table">
        {children}
      </table>
    </div>
  );
}
```

---

## Complete Table Component

```jsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function MarkdownTable({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        table: ({ children }) => (
          <ScrollableTable>
            {children}
          </ScrollableTable>
        ),
        thead: ({ children }) => (
          <thead className="md-thead">{children}</thead>
        ),
        tbody: ({ children }) => (
          <tbody className="md-tbody">{children}</tbody>
        ),
        tr: ({ children, isHeader }) => (
          <tr className={`md-tr ${isHeader ? 'header-row' : ''}`}>
            {children}
          </tr>
        ),
        th: ({ children, style }) => (
          <th 
            className="md-th"
            style={{ textAlign: style?.textAlign }}
          >
            {children}
          </th>
        ),
        td: ({ children, style }) => (
          <td 
            className="md-td"
            style={{ textAlign: style?.textAlign }}
          >
            {children}
          </td>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function ScrollableTable({ children }) {
  const wrapperRef = useRef(null);
  const [showRightShadow, setShowRightShadow] = useState(false);
  
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    
    const checkScroll = () => {
      const { scrollLeft, scrollWidth, clientWidth } = wrapper;
      setShowRightShadow(scrollLeft + clientWidth < scrollWidth - 5);
    };
    
    checkScroll();
    wrapper.addEventListener('scroll', checkScroll);
    
    // Check on resize
    const resizeObserver = new ResizeObserver(checkScroll);
    resizeObserver.observe(wrapper);
    
    return () => {
      wrapper.removeEventListener('scroll', checkScroll);
      resizeObserver.disconnect();
    };
  }, []);
  
  return (
    <div className={`table-container ${showRightShadow ? 'has-overflow' : ''}`}>
      <div ref={wrapperRef} className="table-scroll-wrapper">
        <table className="md-table">{children}</table>
      </div>
    </div>
  );
}
```

### Complete Styles

```css
/* Table container */
.table-container {
  position: relative;
  margin: 16px 0;
}

.table-container.has-overflow::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 40px;
  background: linear-gradient(to right, transparent, rgba(0,0,0,0.05));
  pointer-events: none;
}

/* Scroll wrapper */
.table-scroll-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

/* Table base */
.md-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  line-height: 1.5;
}

/* Header */
.md-thead {
  border-bottom: 2px solid var(--border-color, #e5e7eb);
}

.md-th {
  padding: 12px 16px;
  font-weight: 600;
  text-align: left;
  background: var(--header-bg, #f9fafb);
  color: var(--text-primary, #111827);
  white-space: nowrap;
}

/* Body */
.md-td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color, #e5e7eb);
  color: var(--text-secondary, #374151);
}

.md-tbody .md-tr:last-child .md-td {
  border-bottom: none;
}

/* Hover state */
.md-tbody .md-tr:hover {
  background: var(--hover-bg, #f9fafb);
}

/* Responsive */
@media (max-width: 640px) {
  .md-th,
  .md-td {
    padding: 8px 12px;
    font-size: 0.8125rem;
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Wrap tables in scrollable container | Let tables overflow the viewport |
| Use sticky headers for long tables | Force users to scroll up for context |
| Provide table caption/summary | Leave table without context |
| Use consistent column alignment | Mix alignments randomly |
| Test on mobile devices | Assume desktop-only usage |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Table breaks mobile layout | Use horizontal scroll wrapper |
| Headers scroll out of view | Implement sticky headers |
| Can't tell table scrolls | Add scroll shadow indicator |
| Numbers misaligned | Use `font-variant-numeric: tabular-nums` |
| No hover feedback | Add row hover state |

---

## Hands-on Exercise

### Your Task

Build a responsive table component that:
1. Scrolls horizontally on overflow
2. Shows scroll indicator shadow
3. Has sticky header on long content
4. Styles headers distinctly

### Requirements

1. Detect overflow and show shadow
2. Sticky header with max-height container
3. Hover state on rows
4. Proper alignment for columns

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `ResizeObserver` to detect overflow
- `position: sticky` for header
- `::after` pseudo-element for shadow
- `scrollWidth > clientWidth` for overflow detection

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `ScrollableTable` component and complete styles in the Complete Table Component section above.

</details>

---

## Summary

✅ **Scrollable containers** prevent layout breaking  
✅ **Sticky headers** maintain context while scrolling  
✅ **Scroll indicators** signal interactive overflow  
✅ **Responsive patterns** work on all devices  
✅ **Proper semantics** improve accessibility  
✅ **Consistent styling** integrates with design

---

## Further Reading

- [MDN table element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/table)
- [Responsive Tables](https://css-tricks.com/responsive-data-tables/)
- [WCAG Tables](https://www.w3.org/WAI/tutorials/tables/)

---

**Previous:** [Image Embedding](./05-image-embedding.md)  
**Back to:** [Markdown Rendering Overview](./00-markdown-rendering.md)

<!-- 
Sources Consulted:
- MDN table element: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/table
- CSS-Tricks responsive tables: https://css-tricks.com/responsive-data-tables/
- WCAG tables: https://www.w3.org/WAI/tutorials/tables/
-->
