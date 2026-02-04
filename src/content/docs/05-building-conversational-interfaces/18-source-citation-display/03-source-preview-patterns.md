---
title: "Source Preview Patterns"
---

# Source Preview Patterns

## Introduction

When users hover over or click on a citation, showing a preview card with additional context improves the experience. Rather than immediately leaving the chat to visit a source, users can quickly assess relevance before deciding to click through.

This lesson covers hover preview cards, source metadata extraction, domain display patterns, and secure link handling.

### What We'll Cover

- Hover and click preview cards
- Source metadata extraction (title, description, image)
- Domain extraction and display
- Link security with `target="_blank"` and `rel` attributes
- Loading states for preview fetching

### Prerequisites

- [Citation UI Components](./02-citation-ui-components.md)
- CSS positioning (absolute/relative)
- React event handling

---

## Hover Preview Cards

### Basic Hover Card

```tsx
import { useState } from 'react';

interface SourcePreviewProps {
  url: string;
  title?: string;
  children: React.ReactNode;
}

export function SourcePreview({ url, title, children }: SourcePreviewProps) {
  const [isHovered, setIsHovered] = useState(false);
  const hostname = new URL(url).hostname.replace('www.', '');
  
  return (
    <span 
      className="source-preview-wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
      
      {isHovered && (
        <div className="source-preview-card">
          <div className="preview-header">
            <img
              src={`https://www.google.com/s2/favicons?domain=${hostname}&sz=32`}
              alt=""
              className="preview-favicon"
            />
            <span className="preview-domain">{hostname}</span>
          </div>
          
          <div className="preview-title">
            {title ?? 'Click to visit source'}
          </div>
          
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="preview-link"
          >
            Open in new tab →
          </a>
        </div>
      )}
    </span>
  );
}
```

```css
.source-preview-wrapper {
  position: relative;
  display: inline;
}

.source-preview-card {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-bottom: 12px;
  width: 280px;
  padding: 14px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12);
  z-index: 1000;
  animation: fadeIn 0.15s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}

.source-preview-card::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 8px solid transparent;
  border-top-color: white;
}

.source-preview-card::before {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 9px solid transparent;
  border-top-color: #e2e8f0;
}

.preview-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.preview-favicon {
  width: 16px;
  height: 16px;
  border-radius: 3px;
}

.preview-domain {
  font-size: 0.75rem;
  color: #64748b;
}

.preview-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  line-height: 1.4;
  margin-bottom: 10px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #3b82f6;
  text-decoration: none;
}

.preview-link:hover {
  text-decoration: underline;
}
```

---

## Rich Preview with Metadata

### Enhanced Preview Card

```tsx
interface RichSourceData {
  url: string;
  title?: string;
  description?: string;
  image?: string;
  siteName?: string;
  publishedDate?: string;
}

interface RichPreviewCardProps {
  source: RichSourceData;
  onClose: () => void;
}

export function RichPreviewCard({ source, onClose }: RichPreviewCardProps) {
  const hostname = new URL(source.url).hostname.replace('www.', '');
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;
  
  return (
    <div className="rich-preview-card">
      {/* Close button */}
      <button 
        className="preview-close"
        onClick={onClose}
        aria-label="Close preview"
      >
        ✕
      </button>
      
      {/* Preview image */}
      {source.image && (
        <div className="preview-image">
          <img src={source.image} alt="" />
        </div>
      )}
      
      {/* Content */}
      <div className="preview-content">
        {/* Site info */}
        <div className="preview-site">
          <img src={faviconUrl} alt="" className="site-favicon" />
          <span className="site-name">
            {source.siteName ?? hostname}
          </span>
          {source.publishedDate && (
            <span className="publish-date">
              • {formatDate(source.publishedDate)}
            </span>
          )}
        </div>
        
        {/* Title */}
        <h4 className="preview-title">
          {source.title ?? 'Untitled'}
        </h4>
        
        {/* Description */}
        {source.description && (
          <p className="preview-description">
            {source.description}
          </p>
        )}
        
        {/* Action */}
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="preview-action"
        >
          Visit Source
        </a>
      </div>
    </div>
  );
}

function formatDate(dateString: string): string {
  try {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return dateString;
  }
}
```

```css
.rich-preview-card {
  position: relative;
  width: 320px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
}

.preview-close {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 0.75rem;
  cursor: pointer;
  z-index: 10;
  transition: background 0.2s;
}

.preview-close:hover {
  background: rgba(0, 0, 0, 0.7);
}

.preview-image {
  width: 100%;
  height: 160px;
  overflow: hidden;
  background: #f1f5f9;
}

.preview-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.preview-content {
  padding: 14px;
}

.preview-site {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.site-favicon {
  width: 14px;
  height: 14px;
  border-radius: 2px;
}

.site-name {
  font-size: 0.75rem;
  font-weight: 500;
  color: #64748b;
}

.publish-date {
  font-size: 0.7rem;
  color: #94a3b8;
}

.preview-title {
  margin: 0 0 6px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #1e293b;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-description {
  margin: 0 0 12px;
  font-size: 0.8rem;
  color: #64748b;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-action {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 8px 14px;
  background: #3b82f6;
  color: white;
  font-size: 0.8rem;
  font-weight: 500;
  text-decoration: none;
  border-radius: 6px;
  transition: background 0.2s;
}

.preview-action:hover {
  background: #2563eb;
}
```

---

## Domain Extraction Utilities

### URL Parsing Helpers

```typescript
/**
 * Extract clean hostname from URL
 */
export function extractDomain(url: string): string {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

/**
 * Get root domain (e.g., "google.com" from "docs.google.com")
 */
export function getRootDomain(url: string): string {
  const hostname = extractDomain(url);
  const parts = hostname.split('.');
  
  // Handle special cases like co.uk, com.au
  const specialTLDs = ['co.uk', 'com.au', 'co.jp', 'org.uk'];
  for (const tld of specialTLDs) {
    if (hostname.endsWith(tld)) {
      return parts.slice(-3).join('.');
    }
  }
  
  return parts.slice(-2).join('.');
}

/**
 * Format URL for display (truncate long paths)
 */
export function formatDisplayUrl(url: string, maxLength = 50): string {
  try {
    const parsed = new URL(url);
    const display = parsed.hostname + parsed.pathname;
    
    if (display.length <= maxLength) {
      return display;
    }
    
    return display.substring(0, maxLength - 3) + '...';
  } catch {
    return url.substring(0, maxLength);
  }
}

/**
 * Check if URL is from a known trustworthy domain
 */
export function isTrustedDomain(url: string): boolean {
  const trustedDomains = [
    'github.com',
    'stackoverflow.com',
    'developer.mozilla.org',
    'docs.python.org',
    'wikipedia.org',
    'arxiv.org',
    'nature.com',
    'science.org',
  ];
  
  const domain = extractDomain(url);
  return trustedDomains.some(trusted => 
    domain === trusted || domain.endsWith('.' + trusted)
  );
}
```

### Domain Badge Component

```tsx
interface DomainBadgeProps {
  url: string;
  showTrustIndicator?: boolean;
}

export function DomainBadge({ url, showTrustIndicator = true }: DomainBadgeProps) {
  const domain = extractDomain(url);
  const isTrusted = isTrustedDomain(url);
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
  
  return (
    <div className={`domain-badge ${isTrusted ? 'trusted' : ''}`}>
      <img src={faviconUrl} alt="" className="domain-favicon" />
      <span className="domain-name">{domain}</span>
      {showTrustIndicator && isTrusted && (
        <span className="trust-icon" title="Trusted source">✓</span>
      )}
    </div>
  );
}
```

```css
.domain-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.8rem;
}

.domain-badge.trusted {
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.domain-favicon {
  width: 14px;
  height: 14px;
  border-radius: 2px;
}

.domain-name {
  color: #475569;
}

.trust-icon {
  color: #16a34a;
  font-size: 0.7rem;
}
```

---

## Link Security

### Secure Link Component

```tsx
interface SecureLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

export function SecureLink({ href, children, className }: SecureLinkProps) {
  // Validate URL
  let isValid = true;
  try {
    const url = new URL(href);
    // Only allow http and https protocols
    isValid = ['http:', 'https:'].includes(url.protocol);
  } catch {
    isValid = false;
  }
  
  if (!isValid) {
    return <span className={className}>{children}</span>;
  }
  
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
    >
      {children}
    </a>
  );
}
```

### Security Attributes Explained

| Attribute | Purpose |
|-----------|---------|
| `target="_blank"` | Opens link in new tab |
| `rel="noopener"` | Prevents new page from accessing `window.opener` |
| `rel="noreferrer"` | Prevents passing referrer header |

> **Warning:** Without `rel="noopener"`, a malicious page opened via `target="_blank"` could access and modify the opener page via `window.opener`.

### Link with External Indicator

```tsx
interface ExternalLinkProps {
  href: string;
  children: React.ReactNode;
  showIcon?: boolean;
}

export function ExternalLink({ 
  href, 
  children, 
  showIcon = true 
}: ExternalLinkProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="external-link"
    >
      {children}
      {showIcon && (
        <svg 
          className="external-icon" 
          viewBox="0 0 24 24" 
          fill="none"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path 
            d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </a>
  );
}
```

```css
.external-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #2563eb;
  text-decoration: none;
}

.external-link:hover {
  text-decoration: underline;
}

.external-icon {
  width: 12px;
  height: 12px;
  flex-shrink: 0;
}
```

---

## Click-to-Preview Modal

### Modal Preview Component

```tsx
import { useState } from 'react';

interface Source {
  id: string;
  url: string;
  title?: string;
}

interface ModalPreviewProps {
  sources: Source[];
}

export function ModalPreview({ sources }: ModalPreviewProps) {
  const [activeSource, setActiveSource] = useState<Source | null>(null);
  
  return (
    <>
      {/* Citation markers */}
      <span className="citation-markers">
        {sources.map((source, index) => (
          <button
            key={source.id}
            className="citation-marker-btn"
            onClick={() => setActiveSource(source)}
            aria-label={`View source ${index + 1}`}
          >
            [{index + 1}]
          </button>
        ))}
      </span>
      
      {/* Modal */}
      {activeSource && (
        <div 
          className="preview-modal-overlay"
          onClick={() => setActiveSource(null)}
        >
          <div 
            className="preview-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              onClick={() => setActiveSource(null)}
            >
              ✕
            </button>
            
            <div className="modal-content">
              <DomainBadge url={activeSource.url} />
              
              <h3 className="modal-title">
                {activeSource.title ?? 'Source'}
              </h3>
              
              <p className="modal-url">
                {formatDisplayUrl(activeSource.url, 60)}
              </p>
              
              <div className="modal-actions">
                <button
                  className="modal-btn secondary"
                  onClick={() => setActiveSource(null)}
                >
                  Close
                </button>
                <a
                  href={activeSource.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="modal-btn primary"
                >
                  Visit Source
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
```

```css
.citation-marker-btn {
  background: none;
  border: none;
  padding: 0 2px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: #3b82f6;
  cursor: pointer;
  vertical-align: super;
}

.citation-marker-btn:hover {
  text-decoration: underline;
}

.preview-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  animation: fadeIn 0.2s ease;
}

.preview-modal {
  position: relative;
  width: 90%;
  max-width: 400px;
  background: white;
  border-radius: 16px;
  padding: 24px;
  animation: scaleIn 0.2s ease;
}

@keyframes scaleIn {
  from {
    transform: scale(0.95);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.modal-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 32px;
  height: 32px;
  background: #f1f5f9;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 0.875rem;
  color: #64748b;
  transition: all 0.2s;
}

.modal-close:hover {
  background: #e2e8f0;
  color: #334155;
}

.modal-content {
  text-align: center;
}

.modal-title {
  margin: 16px 0 8px;
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
}

.modal-url {
  margin: 0 0 20px;
  font-size: 0.8rem;
  color: #64748b;
  word-break: break-all;
}

.modal-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.modal-btn {
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.2s;
}

.modal-btn.secondary {
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  color: #64748b;
}

.modal-btn.secondary:hover {
  background: #e2e8f0;
}

.modal-btn.primary {
  background: #3b82f6;
  border: none;
  color: white;
}

.modal-btn.primary:hover {
  background: #2563eb;
}
```

---

## Summary

✅ Hover cards show source preview without leaving the page

✅ Rich previews can include images, descriptions, and publish dates

✅ Extract domains with `new URL(url).hostname` for clean display

✅ Always use `rel="noopener noreferrer"` with `target="_blank"` for security

✅ Modal previews work better on touch devices than hover

**Next:** [Provider-Specific Sources](./04-provider-specific-sources.md)

---

## Further Reading

- [OWASP: Target Blank Vulnerability](https://owasp.org/www-community/attacks/Reverse_Tabnabbing) — Security considerations
- [MDN: URL API](https://developer.mozilla.org/en-US/docs/Web/API/URL) — URL parsing

---

<!-- 
Sources Consulted:
- OWASP Reverse Tabnabbing: https://owasp.org/www-community/attacks/Reverse_Tabnabbing
- MDN URL API: https://developer.mozilla.org/en-US/docs/Web/API/URL
-->
