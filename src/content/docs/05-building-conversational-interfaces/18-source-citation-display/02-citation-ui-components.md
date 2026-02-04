---
title: "Citation UI Components"
---

# Citation UI Components

## Introduction

Beyond simply listing sources, effective citation display integrates references directly into the response text. Inline citation markers like [1] and [2] allow readers to quickly identify which claims are backed by sources, while a footer list provides full details.

This lesson covers building inline citation markers, footer citation lists, clickable source links, and favicon display for visual recognition.

### What We'll Cover

- Inline citation markers [1], [2], [3]
- Footer citation lists with full details
- Clickable links with proper attributes
- Favicon display for source recognition
- Numbered vs linked citation patterns

### Prerequisites

- [Rendering Source Parts](./01-rendering-source-parts.md)
- CSS styling fundamentals
- React state management basics

---

## Citation Pattern Overview

### Academic-Style Citations

```
AI is transforming healthcare [1] and education [2].
Machine learning enables personalized treatment [1][3].

──────────────────────────────────────
Sources:
[1] nature.com - AI in Healthcare 2024
[2] edtech.org - Future of Education
[3] pubmed.gov - Personalized Medicine
```

### Wikipedia-Style Citations

```
The study found significant improvements [¹] compared
to previous methods [²].

──────────────────────────────────────
References:
1. ^ Smith et al., 2024
2. ^ Johnson, Nature 2023
```

---

## Inline Citation Markers

### Basic Citation Badge

```tsx
interface CitationMarkerProps {
  index: number;
  url: string;
  title?: string;
}

export function CitationMarker({ index, url, title }: CitationMarkerProps) {
  return (
    <a
      className="citation-marker"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      title={title ?? url}
      aria-label={`Source ${index + 1}: ${title ?? url}`}
    >
      [{index + 1}]
    </a>
  );
}
```

```css
.citation-marker {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: #3b82f6;
  text-decoration: none;
  vertical-align: super;
  line-height: 1;
  margin: 0 1px;
  transition: all 0.15s;
}

.citation-marker:hover {
  color: #1d4ed8;
  text-decoration: underline;
}

/* Superscript style variant */
.citation-marker.superscript {
  font-size: 0.65rem;
  vertical-align: super;
  margin-left: 1px;
}

/* Badge style variant */
.citation-marker.badge {
  padding: 1px 4px;
  background: #dbeafe;
  border-radius: 4px;
  vertical-align: baseline;
}

.citation-marker.badge:hover {
  background: #bfdbfe;
}
```

### Interactive Citation with Tooltip

```tsx
import { useState } from 'react';

interface TooltipCitationProps {
  index: number;
  url: string;
  title?: string;
}

export function TooltipCitation({ index, url, title }: TooltipCitationProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const hostname = new URL(url).hostname;
  
  return (
    <span className="citation-wrapper">
      <a
        className="citation-marker badge"
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
      >
        [{index + 1}]
      </a>
      
      {showTooltip && (
        <div className="citation-tooltip" role="tooltip">
          <div className="tooltip-title">{title ?? 'Source'}</div>
          <div className="tooltip-url">{hostname}</div>
        </div>
      )}
    </span>
  );
}
```

```css
.citation-wrapper {
  position: relative;
  display: inline;
}

.citation-tooltip {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-bottom: 8px;
  padding: 8px 12px;
  background: #1e293b;
  color: white;
  border-radius: 6px;
  font-size: 0.75rem;
  white-space: nowrap;
  z-index: 1000;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.citation-tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #1e293b;
}

.tooltip-title {
  font-weight: 500;
  margin-bottom: 2px;
}

.tooltip-url {
  color: #94a3b8;
  font-size: 0.7rem;
}
```

---

## Footer Citation List

### Basic Footer List

```tsx
interface Source {
  id: string;
  url: string;
  title?: string;
  type: 'source-url' | 'source-document';
}

interface CitationFooterProps {
  sources: Source[];
}

export function CitationFooter({ sources }: CitationFooterProps) {
  if (sources.length === 0) return null;
  
  return (
    <footer className="citation-footer">
      <div className="footer-divider" />
      <h4 className="footer-heading">Sources</h4>
      <ol className="citation-list">
        {sources.map((source, index) => (
          <li key={source.id} className="citation-item">
            <CitationEntry source={source} index={index} />
          </li>
        ))}
      </ol>
    </footer>
  );
}

function CitationEntry({ source, index }: { source: Source; index: number }) {
  if (source.type === 'source-document') {
    return (
      <span className="doc-citation">
        <span className="citation-number">[{index + 1}]</span>
        <span className="citation-title">{source.title ?? 'Document'}</span>
      </span>
    );
  }
  
  const hostname = new URL(source.url).hostname.replace('www.', '');
  
  return (
    <a
      className="url-citation"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="citation-number">[{index + 1}]</span>
      <span className="citation-domain">{hostname}</span>
      <span className="citation-separator">—</span>
      <span className="citation-title">{source.title ?? 'Untitled'}</span>
    </a>
  );
}
```

```css
.citation-footer {
  margin-top: 24px;
}

.footer-divider {
  height: 1px;
  background: linear-gradient(to right, #e2e8f0, transparent);
  margin-bottom: 16px;
}

.footer-heading {
  margin: 0 0 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}

.citation-list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.citation-item {
  padding: 8px 0;
  border-bottom: 1px solid #f1f5f9;
}

.citation-item:last-child {
  border-bottom: none;
}

.url-citation,
.doc-citation {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: inherit;
  font-size: 0.875rem;
}

.url-citation:hover {
  color: #2563eb;
}

.citation-number {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 600;
  color: #3b82f6;
  min-width: 28px;
}

.citation-domain {
  color: #64748b;
  font-size: 0.8rem;
}

.citation-separator {
  color: #cbd5e1;
}

.citation-title {
  color: #334155;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

---

## Favicon Display

### Favicon Component

```tsx
interface FaviconProps {
  url: string;
  size?: number;
}

export function Favicon({ url, size = 16 }: FaviconProps) {
  const hostname = new URL(url).hostname;
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=${size * 2}`;
  
  return (
    <img
      src={faviconUrl}
      alt=""
      className="source-favicon"
      width={size}
      height={size}
      loading="lazy"
      onError={(e) => {
        // Fallback to generic icon
        (e.target as HTMLImageElement).style.display = 'none';
      }}
    />
  );
}
```

### Citation with Favicon

```tsx
export function FaviconCitation({ source, index }: { source: Source; index: number }) {
  if (source.type !== 'source-url') return null;
  
  const hostname = new URL(source.url).hostname.replace('www.', '');
  
  return (
    <a
      className="favicon-citation"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="citation-index">{index + 1}</span>
      <Favicon url={source.url} size={16} />
      <div className="citation-content">
        <span className="citation-title">
          {source.title ?? 'Untitled'}
        </span>
        <span className="citation-domain">{hostname}</span>
      </div>
    </a>
  );
}
```

```css
.favicon-citation {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;
}

.favicon-citation:hover {
  border-color: #3b82f6;
  background: #f8fafc;
}

.citation-index {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  background: #dbeafe;
  color: #2563eb;
  font-size: 0.7rem;
  font-weight: 600;
  border-radius: 4px;
}

.source-favicon {
  width: 16px;
  height: 16px;
  border-radius: 2px;
  flex-shrink: 0;
}

.citation-content {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1;
}

.citation-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.citation-domain {
  font-size: 0.75rem;
  color: #94a3b8;
}
```

---

## Grid Layout for Multiple Sources

### Sources Grid Component

```tsx
interface SourcesGridProps {
  sources: Source[];
  columns?: 1 | 2 | 3;
}

export function SourcesGrid({ sources, columns = 2 }: SourcesGridProps) {
  if (sources.length === 0) return null;
  
  return (
    <div className="sources-section">
      <div className="sources-header">
        <span className="sources-icon">📚</span>
        <span className="sources-label">Sources</span>
        <span className="sources-count">{sources.length}</span>
      </div>
      
      <div 
        className="sources-grid"
        style={{ '--columns': columns } as React.CSSProperties}
      >
        {sources.map((source, index) => (
          <FaviconCitation 
            key={source.id} 
            source={source} 
            index={index} 
          />
        ))}
      </div>
    </div>
  );
}
```

```css
.sources-section {
  margin-top: 20px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.sources-icon {
  font-size: 1rem;
}

.sources-label {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}

.sources-count {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  background: #e2e8f0;
  color: #475569;
  font-size: 0.7rem;
  font-weight: 600;
  border-radius: 10px;
}

.sources-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(var(--columns, 2), 1fr);
}

@media (max-width: 640px) {
  .sources-grid {
    grid-template-columns: 1fr;
  }
}
```

---

## Compact Citation Pills

### Pill-Style Citations

```tsx
export function CitationPills({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  
  return (
    <div className="citation-pills">
      {sources.map((source, index) => (
        <a
          key={source.id}
          className="citation-pill"
          href={source.type === 'source-url' ? source.url : '#'}
          target={source.type === 'source-url' ? '_blank' : undefined}
          rel={source.type === 'source-url' ? 'noopener noreferrer' : undefined}
        >
          {source.type === 'source-url' && (
            <Favicon url={source.url} size={12} />
          )}
          <span className="pill-text">
            {source.type === 'source-url' 
              ? new URL(source.url).hostname.replace('www.', '')
              : source.title ?? 'Doc'}
          </span>
        </a>
      ))}
    </div>
  );
}
```

```css
.citation-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.citation-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 9999px;
  font-size: 0.75rem;
  color: #475569;
  text-decoration: none;
  transition: all 0.15s;
}

.citation-pill:hover {
  background: #dbeafe;
  border-color: #93c5fd;
  color: #1e40af;
}

.citation-pill .source-favicon {
  width: 12px;
  height: 12px;
}

.pill-text {
  max-width: 120px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

---

## Accessibility Considerations

### ARIA Labels and Keyboard Navigation

```tsx
export function AccessibleCitationList({ sources }: { sources: Source[] }) {
  return (
    <nav 
      className="citation-nav" 
      aria-label="Source citations"
    >
      <h4 id="sources-heading" className="visually-hidden">
        Sources ({sources.length})
      </h4>
      
      <ol 
        className="citation-list"
        aria-labelledby="sources-heading"
      >
        {sources.map((source, index) => (
          <li key={source.id}>
            <a
              href={source.type === 'source-url' ? source.url : '#'}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`Source ${index + 1}: ${source.title ?? 'Untitled'}`}
            >
              <span aria-hidden="true">[{index + 1}]</span>
              <span>{source.title ?? 'Source'}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
```

```css
.visually-hidden {
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

.citation-nav a:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
  border-radius: 4px;
}

.citation-nav a:focus-visible {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}
```

---

## Summary

✅ Inline markers [1][2] link claims to sources within text

✅ Footer lists provide full source details with favicons

✅ Use `target="_blank"` with `rel="noopener noreferrer"` for security

✅ Google Favicon API provides site icons: `https://www.google.com/s2/favicons?domain=...`

✅ Grid and pill layouts adapt to different UI needs

✅ ARIA labels make citations accessible to screen readers

**Next:** [Source Preview Patterns](./03-source-preview-patterns.md)

---

## Further Reading

- [Web Content Accessibility Guidelines](https://www.w3.org/WAI/WCAG22/quickref/) — Accessibility standards
- [Google Favicon Service](https://www.google.com/s2/favicons) — Favicon API

---

<!-- 
Sources Consulted:
- AI SDK Chatbot Sources: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- WCAG Quick Reference: https://www.w3.org/WAI/WCAG22/quickref/
-->
