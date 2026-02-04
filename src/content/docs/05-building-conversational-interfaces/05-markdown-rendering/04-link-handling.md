---
title: "Link Handling"
---

# Link Handling

## Introduction

Links in AI responses connect users to resources, documentation, and related content. But links need careful handling—external links should open safely in new tabs, internal links should navigate smoothly, and all links need proper accessibility. Security considerations like `rel="noopener"` are essential.

In this lesson, we'll implement secure, user-friendly link handling.

### What We'll Cover

- External vs internal link detection
- Security with `target="_blank"`
- External link indicators
- Link preview on hover
- Click tracking and analytics

### Prerequisites

- [Heading Rendering](./03-heading-rendering.md)
- HTML anchor elements
- React event handling

---

## External vs Internal Links

### Link Detection

```jsx
function isExternalLink(href) {
  if (!href) return false;
  
  // Protocol-relative or absolute URLs with different origin
  if (href.startsWith('//')) return true;
  
  // Full URLs
  if (href.startsWith('http://') || href.startsWith('https://')) {
    try {
      const url = new URL(href);
      return url.origin !== window.location.origin;
    } catch {
      return false;
    }
  }
  
  // Relative URLs are internal
  return false;
}

function isAnchorLink(href) {
  return href?.startsWith('#');
}

function isMailtoOrTel(href) {
  return href?.startsWith('mailto:') || href?.startsWith('tel:');
}
```

### Smart Link Component

```jsx
function SmartLink({ href, children, ...props }) {
  if (isAnchorLink(href)) {
    return (
      <AnchorLink href={href} {...props}>
        {children}
      </AnchorLink>
    );
  }
  
  if (isMailtoOrTel(href)) {
    return (
      <a href={href} className="md-link md-link-contact" {...props}>
        {children}
      </a>
    );
  }
  
  if (isExternalLink(href)) {
    return (
      <ExternalLink href={href} {...props}>
        {children}
      </ExternalLink>
    );
  }
  
  return (
    <InternalLink href={href} {...props}>
      {children}
    </InternalLink>
  );
}
```

---

## External Link Security

### Secure External Links

```jsx
function ExternalLink({ href, children, showIcon = true, ...props }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="md-link md-link-external"
      {...props}
    >
      {children}
      {showIcon && (
        <ExternalLinkIcon 
          className="external-icon" 
          aria-hidden="true" 
        />
      )}
      <span className="sr-only">(opens in new tab)</span>
    </a>
  );
}

function ExternalLinkIcon({ className }) {
  return (
    <svg 
      className={className}
      width="12" 
      height="12" 
      viewBox="0 0 12 12" 
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path d="M3.5 3.5h5v5" />
      <path d="M8.5 3.5L3.5 8.5" />
    </svg>
  );
}
```

```css
.md-link {
  color: var(--link-color, #2563eb);
  text-decoration: underline;
  text-decoration-color: transparent;
  text-underline-offset: 2px;
  transition: text-decoration-color 0.2s, color 0.2s;
}

.md-link:hover {
  text-decoration-color: currentColor;
}

.md-link-external {
  display: inline-flex;
  align-items: baseline;
  gap: 3px;
}

.external-icon {
  flex-shrink: 0;
  margin-left: 2px;
  opacity: 0.6;
  transform: translateY(-1px);
}

.md-link-external:hover .external-icon {
  opacity: 1;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

### Why `rel="noopener noreferrer"`?

| Attribute | Purpose |
|-----------|---------|
| `noopener` | Prevents `window.opener` access (security) |
| `noreferrer` | Prevents sending referrer header (privacy) |

> **Warning:** Without `noopener`, malicious pages can redirect your page via `window.opener.location`.

---

## Internal Links

### Smooth Scroll for Anchors

```jsx
function AnchorLink({ href, children, ...props }) {
  const handleClick = (e) => {
    e.preventDefault();
    const id = href.replace('#', '');
    const element = document.getElementById(id);
    
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      // Update URL without reload
      window.history.pushState(null, '', href);
      // Focus for accessibility
      element.focus({ preventScroll: true });
    }
  };
  
  return (
    <a 
      href={href}
      onClick={handleClick}
      className="md-link md-link-anchor"
      {...props}
    >
      {children}
    </a>
  );
}
```

### Internal Navigation

```jsx
import { useRouter } from 'next/router';  // or your router

function InternalLink({ href, children, ...props }) {
  const router = useRouter();
  
  const handleClick = (e) => {
    e.preventDefault();
    router.push(href);
  };
  
  return (
    <a 
      href={href}
      onClick={handleClick}
      className="md-link md-link-internal"
      {...props}
    >
      {children}
    </a>
  );
}
```

---

## Link Indicators

### Domain Badge

```jsx
function ExternalLinkWithDomain({ href, children }) {
  const domain = useMemo(() => {
    try {
      const url = new URL(href);
      return url.hostname.replace('www.', '');
    } catch {
      return null;
    }
  }, [href]);
  
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="md-link md-link-external"
    >
      {children}
      {domain && (
        <span className="link-domain" aria-hidden="true">
          {domain}
        </span>
      )}
    </a>
  );
}
```

```css
.link-domain {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  font-size: 0.7em;
  background: var(--domain-bg, #f1f5f9);
  color: var(--text-secondary, #64748b);
  border-radius: 4px;
  vertical-align: middle;
}
```

### Favicon Display

```jsx
function LinkWithFavicon({ href, children }) {
  const faviconUrl = useMemo(() => {
    try {
      const url = new URL(href);
      // Use Google's favicon service
      return `https://www.google.com/s2/favicons?domain=${url.hostname}&sz=16`;
    } catch {
      return null;
    }
  }, [href]);
  
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="md-link md-link-external with-favicon"
    >
      {faviconUrl && (
        <img 
          src={faviconUrl} 
          alt="" 
          className="link-favicon"
          loading="lazy"
        />
      )}
      {children}
      <ExternalLinkIcon />
    </a>
  );
}
```

```css
.with-favicon {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.link-favicon {
  width: 14px;
  height: 14px;
  border-radius: 2px;
}
```

---

## Link Preview

### Hover Preview Card

```jsx
function LinkWithPreview({ href, children }) {
  const [showPreview, setShowPreview] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [loading, setLoading] = useState(false);
  const timeoutRef = useRef(null);
  
  const handleMouseEnter = async () => {
    timeoutRef.current = setTimeout(async () => {
      if (!previewData && !loading) {
        setLoading(true);
        try {
          // Fetch preview data from your API
          const data = await fetchLinkPreview(href);
          setPreviewData(data);
        } catch (err) {
          console.error('Failed to fetch preview:', err);
        }
        setLoading(false);
      }
      setShowPreview(true);
    }, 500);  // Delay before showing
  };
  
  const handleMouseLeave = () => {
    clearTimeout(timeoutRef.current);
    setShowPreview(false);
  };
  
  return (
    <span className="link-preview-container">
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="md-link md-link-external"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleMouseEnter}
        onBlur={handleMouseLeave}
      >
        {children}
        <ExternalLinkIcon />
      </a>
      
      {showPreview && (
        <LinkPreviewCard 
          data={previewData} 
          loading={loading}
        />
      )}
    </span>
  );
}
```

### Preview Card Component

```jsx
function LinkPreviewCard({ data, loading }) {
  if (loading) {
    return (
      <div className="link-preview-card loading">
        <div className="preview-skeleton title" />
        <div className="preview-skeleton description" />
      </div>
    );
  }
  
  if (!data) return null;
  
  return (
    <div className="link-preview-card" role="tooltip">
      {data.image && (
        <img 
          src={data.image} 
          alt="" 
          className="preview-image"
        />
      )}
      <div className="preview-content">
        <div className="preview-title">{data.title}</div>
        <div className="preview-description">{data.description}</div>
        <div className="preview-domain">{data.domain}</div>
      </div>
    </div>
  );
}
```

```css
.link-preview-container {
  position: relative;
  display: inline;
}

.link-preview-card {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-bottom: 8px;
  width: 320px;
  background: white;
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.15);
  overflow: hidden;
  z-index: 100;
  animation: preview-in 0.2s ease-out;
}

@keyframes preview-in {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}

.preview-image {
  width: 100%;
  height: 160px;
  object-fit: cover;
}

.preview-content {
  padding: 12px;
}

.preview-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.preview-description {
  font-size: 13px;
  color: var(--text-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-bottom: 8px;
}

.preview-domain {
  font-size: 12px;
  color: var(--text-muted);
}
```

---

## Click Tracking

### Analytics Integration

```jsx
function TrackedLink({ href, children, category = 'content' }) {
  const handleClick = () => {
    // Send to analytics
    if (window.gtag) {
      window.gtag('event', 'click', {
        event_category: 'outbound_link',
        event_label: href,
        link_category: category
      });
    }
    
    // Or custom analytics
    trackEvent('link_click', {
      url: href,
      category,
      timestamp: Date.now()
    });
  };
  
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={handleClick}
      className="md-link md-link-external"
    >
      {children}
      <ExternalLinkIcon />
    </a>
  );
}
```

### Centralized Link Handler

```jsx
function useTrackLinks(containerRef) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const handleClick = (e) => {
      const link = e.target.closest('a');
      if (!link) return;
      
      const href = link.getAttribute('href');
      if (!href) return;
      
      // Track all link clicks
      trackEvent('link_click', {
        url: href,
        text: link.textContent,
        isExternal: isExternalLink(href),
        messageId: container.dataset.messageId
      });
    };
    
    container.addEventListener('click', handleClick);
    return () => container.removeEventListener('click', handleClick);
  }, [containerRef]);
}
```

---

## Link Validation

### Broken Link Detection

```jsx
function useValidateLinks(links) {
  const [validationResults, setResults] = useState({});
  
  useEffect(() => {
    const validateLinks = async () => {
      const results = {};
      
      for (const link of links) {
        try {
          // HEAD request is faster than GET
          const response = await fetch(link, { 
            method: 'HEAD',
            mode: 'no-cors'  // Avoid CORS issues
          });
          results[link] = { valid: true };
        } catch {
          results[link] = { valid: false, error: 'unreachable' };
        }
      }
      
      setResults(results);
    };
    
    validateLinks();
  }, [links]);
  
  return validationResults;
}
```

### Visual Broken Link Indicator

```jsx
function ValidatedLink({ href, children, isValid }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`md-link ${isValid === false ? 'md-link-broken' : ''}`}
      title={isValid === false ? 'This link may be broken' : undefined}
    >
      {children}
      {isValid === false && (
        <span className="broken-indicator" aria-label="Possibly broken link">
          ⚠️
        </span>
      )}
    </a>
  );
}
```

```css
.md-link-broken {
  color: var(--warning-color, #f59e0b);
  text-decoration-style: wavy;
}

.broken-indicator {
  margin-left: 4px;
  font-size: 0.8em;
}
```

---

## Complete Link Component

```jsx
import ReactMarkdown from 'react-markdown';

function MarkdownWithLinks({ content }) {
  return (
    <ReactMarkdown
      components={{
        a: ({ href, children }) => (
          <SmartLink href={href}>{children}</SmartLink>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function SmartLink({ href, children }) {
  // Handle missing href
  if (!href) {
    return <span className="md-link-placeholder">{children}</span>;
  }
  
  // Anchor links
  if (isAnchorLink(href)) {
    return <AnchorLink href={href}>{children}</AnchorLink>;
  }
  
  // mailto/tel
  if (isMailtoOrTel(href)) {
    return (
      <a href={href} className="md-link md-link-contact">
        {href.startsWith('mailto:') ? '📧' : '📞'} {children}
      </a>
    );
  }
  
  // External links
  if (isExternalLink(href)) {
    return (
      <ExternalLink href={href}>
        {children}
      </ExternalLink>
    );
  }
  
  // Internal links
  return (
    <InternalLink href={href}>
      {children}
    </InternalLink>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Add `rel="noopener noreferrer"` to external links | Open external links without security attributes |
| Show indicator for external links | Leave users confused about destination |
| Use smooth scroll for anchor links | Jump harshly to anchors |
| Track link clicks for analytics | Ignore user navigation patterns |
| Validate links when possible | Show broken links without warning |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting `target="_blank"` security | Always add `rel="noopener noreferrer"` |
| External icon on every link | Only show for actual external links |
| Preview blocks interaction | Show after delay, dismiss on blur |
| CORS errors on validation | Use `mode: 'no-cors'` or server-side |
| Link styles override inline code | Be specific with selectors |

---

## Hands-on Exercise

### Your Task

Build a complete link handling system that:
1. Distinguishes external, internal, and anchor links
2. Shows external indicator icon
3. Opens external links safely
4. Displays domain on hover

### Requirements

1. Security attributes on external links
2. Smooth scroll for anchor links
3. External link icon
4. Domain badge on hover

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `new URL()` to parse external links
- `scrollIntoView({ behavior: 'smooth' })` for anchors
- CSS `opacity` transition for hover effects
- `hostname.replace('www.', '')` for clean domains

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `SmartLink` component and supporting utilities above.

</details>

---

## Summary

✅ **Detect link types** (external, internal, anchor, mailto)  
✅ **Security attributes** prevent `window.opener` attacks  
✅ **External indicators** clarify navigation intent  
✅ **Link previews** provide context before clicking  
✅ **Click tracking** enables analytics insights  
✅ **Validation** warns about broken links

---

## Further Reading

- [MDN rel attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel)
- [Link Security Best Practices](https://web.dev/external-anchors-use-rel-noopener/)
- [WCAG Links](https://www.w3.org/WAI/WCAG21/Understanding/link-purpose-in-context.html)

---

**Previous:** [Heading Rendering](./03-heading-rendering.md)  
**Next:** [Image Embedding](./05-image-embedding.md)

<!-- 
Sources Consulted:
- MDN rel attribute: https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel
- web.dev external links: https://web.dev/external-anchors-use-rel-noopener/
- WCAG link purpose: https://www.w3.org/WAI/WCAG21/Understanding/link-purpose-in-context.html
-->
