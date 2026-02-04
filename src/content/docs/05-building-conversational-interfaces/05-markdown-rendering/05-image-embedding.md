---
title: "Image Embedding"
---

# Image Embedding

## Introduction

AI responses may include images—diagrams, charts, screenshots, or generated visuals. Rendering them requires lazy loading for performance, proper alt text for accessibility, placeholder states during loading, and expansion options for detailed viewing.

In this lesson, we'll implement robust image embedding in chat interfaces.

### What We'll Cover

- Inline image display
- Alt text handling
- Lazy loading strategies
- Loading states and placeholders
- Lightbox expansion
- Error handling

### Prerequisites

- [Link Handling](./04-link-handling.md)
- HTML img element
- CSS object-fit

---

## Inline Image Display

### Basic Image Component

```jsx
import ReactMarkdown from 'react-markdown';

function MarkdownWithImages({ content }) {
  return (
    <ReactMarkdown
      components={{
        img: ({ src, alt, title }) => (
          <ChatImage src={src} alt={alt} title={title} />
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function ChatImage({ src, alt, title }) {
  return (
    <figure className="md-image-figure">
      <img
        src={src}
        alt={alt || 'Image'}
        title={title}
        className="md-image"
        loading="lazy"
      />
      {(alt || title) && (
        <figcaption className="md-image-caption">
          {title || alt}
        </figcaption>
      )}
    </figure>
  );
}
```

```css
.md-image-figure {
  margin: 16px 0;
}

.md-image {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  display: block;
}

.md-image-caption {
  margin-top: 8px;
  font-size: 0.875rem;
  color: var(--text-secondary, #6b7280);
  text-align: center;
}
```

### Responsive Images

```css
.md-image-figure {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.md-image {
  max-width: 100%;
  max-height: 400px;
  width: auto;
  object-fit: contain;
}

/* Full-width images for diagrams */
.md-image-figure.full-width .md-image {
  max-height: none;
  width: 100%;
}

/* Thumbnail size */
.md-image-figure.thumbnail .md-image {
  max-width: 200px;
  max-height: 200px;
  object-fit: cover;
}
```

---

## Alt Text Handling

### Accessible Image Component

```jsx
function AccessibleImage({ src, alt, title, isDecorative = false }) {
  // Decorative images get empty alt
  if (isDecorative) {
    return (
      <img 
        src={src} 
        alt="" 
        role="presentation"
        className="md-image decorative"
      />
    );
  }
  
  // Missing alt text warning (development only)
  if (!alt && process.env.NODE_ENV === 'development') {
    console.warn(`Image missing alt text: ${src}`);
  }
  
  return (
    <figure className="md-image-figure">
      <img
        src={src}
        alt={alt || 'Image'}
        title={title}
        className="md-image"
        loading="lazy"
      />
      
      {/* Screen reader enhancement for complex images */}
      {alt && alt.length > 125 && (
        <figcaption className="sr-only">
          {alt}
        </figcaption>
      )}
    </figure>
  );
}
```

### Alt Text Extraction

```jsx
// When AI provides description in title
function parseImageMeta(alt, title) {
  // Pattern: ![Short alt](url "Long description")
  return {
    shortAlt: alt || 'Image',
    longDesc: title || alt,
    hasDescription: !!title
  };
}

function EnhancedImage({ src, alt, title }) {
  const meta = parseImageMeta(alt, title);
  
  return (
    <figure className="md-image-figure">
      <img
        src={src}
        alt={meta.shortAlt}
        className="md-image"
        loading="lazy"
      />
      {meta.hasDescription && (
        <figcaption className="md-image-caption">
          {meta.longDesc}
        </figcaption>
      )}
    </figure>
  );
}
```

---

## Lazy Loading

### Native Lazy Loading

```jsx
function LazyImage({ src, alt }) {
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      decoding="async"
      className="md-image"
    />
  );
}
```

### Intersection Observer Lazy Loading

```jsx
function AdvancedLazyImage({ src, alt, placeholder }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: '200px' }  // Load 200px before visible
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div 
      ref={imgRef}
      className={`lazy-image-container ${isLoaded ? 'loaded' : ''}`}
    >
      {!isLoaded && (
        <div className="image-placeholder">
          {placeholder || <ImagePlaceholder />}
        </div>
      )}
      
      {isInView && (
        <img
          src={src}
          alt={alt}
          className="md-image"
          onLoad={() => setIsLoaded(true)}
          style={{ opacity: isLoaded ? 1 : 0 }}
        />
      )}
    </div>
  );
}
```

```css
.lazy-image-container {
  position: relative;
  min-height: 200px;
  background: var(--placeholder-bg, #f3f4f6);
  border-radius: 8px;
  overflow: hidden;
}

.lazy-image-container .md-image {
  transition: opacity 0.3s ease;
}

.image-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

---

## Loading States

### Skeleton Placeholder

```jsx
function ImagePlaceholder() {
  return (
    <div className="image-skeleton">
      <div className="skeleton-icon">
        <ImageIcon />
      </div>
    </div>
  );
}

function ImageIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5.04-6.71l-2.75 3.54-1.96-2.36L6.5 17h11l-3.54-4.71z"/>
    </svg>
  );
}
```

```css
.image-skeleton {
  width: 100%;
  height: 200px;
  background: linear-gradient(
    90deg,
    var(--skeleton-base, #e5e7eb) 0%,
    var(--skeleton-highlight, #f3f4f6) 50%,
    var(--skeleton-base, #e5e7eb) 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  display: flex;
  align-items: center;
  justify-content: center;
}

.skeleton-icon {
  color: var(--text-muted, #9ca3af);
  opacity: 0.5;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Progressive Loading with Blur

```jsx
function ProgressiveImage({ src, alt, thumbnailSrc }) {
  const [currentSrc, setCurrentSrc] = useState(thumbnailSrc || src);
  const [isFullLoaded, setIsFullLoaded] = useState(false);
  
  useEffect(() => {
    if (thumbnailSrc) {
      // Preload full image
      const img = new Image();
      img.src = src;
      img.onload = () => {
        setCurrentSrc(src);
        setIsFullLoaded(true);
      };
    }
  }, [src, thumbnailSrc]);
  
  return (
    <img
      src={currentSrc}
      alt={alt}
      className={`md-image ${isFullLoaded ? '' : 'blurred'}`}
    />
  );
}
```

```css
.md-image.blurred {
  filter: blur(10px);
  transform: scale(1.1);
  transition: filter 0.3s, transform 0.3s;
}

.md-image:not(.blurred) {
  filter: blur(0);
  transform: scale(1);
}
```

---

## Lightbox Expansion

### Simple Lightbox

```jsx
function ImageWithLightbox({ src, alt, title }) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <>
      <figure className="md-image-figure">
        <button 
          onClick={() => setIsOpen(true)}
          className="image-expand-btn"
          aria-label={`View ${alt} in full size`}
        >
          <img
            src={src}
            alt={alt}
            title={title}
            className="md-image"
            loading="lazy"
          />
          <span className="expand-icon" aria-hidden="true">
            ⛶
          </span>
        </button>
      </figure>
      
      {isOpen && (
        <Lightbox
          src={src}
          alt={alt}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
```

### Lightbox Component

```jsx
function Lightbox({ src, alt, onClose }) {
  // Close on Escape
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);
  
  // Prevent body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);
  
  return (
    <div 
      className="lightbox-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Expanded view of ${alt}`}
    >
      <button 
        className="lightbox-close"
        onClick={onClose}
        aria-label="Close"
      >
        ×
      </button>
      
      <img
        src={src}
        alt={alt}
        className="lightbox-image"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
```

```css
.image-expand-btn {
  position: relative;
  display: block;
  border: none;
  background: none;
  padding: 0;
  cursor: zoom-in;
}

.expand-icon {
  position: absolute;
  bottom: 8px;
  right: 8px;
  width: 32px;
  height: 32px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
}

.image-expand-btn:hover .expand-icon {
  opacity: 1;
}

.lightbox-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fade-in 0.2s;
}

.lightbox-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 44px;
  height: 44px;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 24px;
  cursor: pointer;
  transition: background 0.2s;
}

.lightbox-close:hover {
  background: rgba(255, 255, 255, 0.2);
}

.lightbox-image {
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
  border-radius: 4px;
}

@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

### Zoom Controls

```jsx
function ZoomableLightbox({ src, alt, onClose }) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  const handleZoomIn = () => setScale(s => Math.min(s * 1.5, 5));
  const handleZoomOut = () => setScale(s => Math.max(s / 1.5, 0.5));
  const handleReset = () => { setScale(1); setPosition({ x: 0, y: 0 }); };
  
  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-controls">
        <button onClick={handleZoomOut} aria-label="Zoom out">−</button>
        <button onClick={handleReset} aria-label="Reset zoom">⟲</button>
        <button onClick={handleZoomIn} aria-label="Zoom in">+</button>
      </div>
      
      <img
        src={src}
        alt={alt}
        className="lightbox-image"
        style={{
          transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`
        }}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
```

---

## Error Handling

### Image Error Fallback

```jsx
function ImageWithFallback({ src, alt, fallback }) {
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  if (hasError) {
    return (
      <div className="image-error">
        {fallback || (
          <>
            <BrokenImageIcon />
            <span>Failed to load image</span>
          </>
        )}
      </div>
    );
  }
  
  return (
    <div className="image-wrapper">
      {isLoading && <ImagePlaceholder />}
      <img
        src={src}
        alt={alt}
        className="md-image"
        onLoad={() => setIsLoading(false)}
        onError={() => { setHasError(true); setIsLoading(false); }}
        style={{ display: isLoading ? 'none' : 'block' }}
      />
    </div>
  );
}
```

```css
.image-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px;
  background: var(--error-bg, #fef2f2);
  border: 1px dashed var(--error-border, #fecaca);
  border-radius: 8px;
  color: var(--error-text, #991b1b);
}
```

### Retry on Error

```jsx
function ImageWithRetry({ src, alt, maxRetries = 3 }) {
  const [retryCount, setRetryCount] = useState(0);
  const [hasError, setHasError] = useState(false);
  
  const handleError = () => {
    if (retryCount < maxRetries) {
      setTimeout(() => {
        setRetryCount(c => c + 1);
        setHasError(false);
      }, 1000 * (retryCount + 1));  // Exponential backoff
    } else {
      setHasError(true);
    }
  };
  
  if (hasError) {
    return (
      <div className="image-error">
        <span>Failed to load after {maxRetries} attempts</span>
        <button onClick={() => { setRetryCount(0); setHasError(false); }}>
          Try again
        </button>
      </div>
    );
  }
  
  return (
    <img
      key={retryCount}  // Force remount on retry
      src={src}
      alt={alt}
      className="md-image"
      onError={handleError}
    />
  );
}
```

---

## Complete Image Component

```jsx
function ChatImage({ src, alt, title }) {
  const [state, setState] = useState('loading');
  const [isLightboxOpen, setLightboxOpen] = useState(false);
  
  return (
    <figure className="md-image-figure">
      <div className="image-container">
        {state === 'loading' && <ImagePlaceholder />}
        
        {state !== 'error' && (
          <button
            className="image-expand-btn"
            onClick={() => setLightboxOpen(true)}
            aria-label={`View ${alt} in full size`}
          >
            <img
              src={src}
              alt={alt}
              title={title}
              className={`md-image ${state}`}
              loading="lazy"
              onLoad={() => setState('loaded')}
              onError={() => setState('error')}
            />
            <span className="expand-icon">⛶</span>
          </button>
        )}
        
        {state === 'error' && (
          <div className="image-error">
            <BrokenImageIcon />
            <span>Image unavailable</span>
          </div>
        )}
      </div>
      
      {title && <figcaption className="md-image-caption">{title}</figcaption>}
      
      {isLightboxOpen && (
        <Lightbox
          src={src}
          alt={alt}
          onClose={() => setLightboxOpen(false)}
        />
      )}
    </figure>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Always include alt text | Leave alt empty without reason |
| Use lazy loading for images | Load all images immediately |
| Show loading placeholder | Let layout jump on load |
| Provide zoom/expand option | Force small fixed size |
| Handle loading errors gracefully | Show broken image icon only |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Images cause layout shift | Set explicit dimensions or aspect-ratio |
| Large images slow page | Lazy load + serve appropriate sizes |
| Lightbox traps focus | Manage focus, close on Escape |
| No error state | Show fallback with retry option |
| Alt text is URL or filename | Provide meaningful description |

---

## Hands-on Exercise

### Your Task

Build a complete image component that:
1. Shows skeleton while loading
2. Handles errors with fallback
3. Opens in lightbox on click
4. Closes on Escape key

### Requirements

1. Lazy loading with placeholder
2. Error state with retry button
3. Lightbox with close button
4. Keyboard accessible

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `onLoad` and `onError` events
- `document.addEventListener('keydown')` for Escape
- Set `body.style.overflow = 'hidden'` when lightbox open
- Use `role="dialog"` for lightbox

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `ChatImage` component with `Lightbox` in the Complete Image Component section above.

</details>

---

## Summary

✅ **Lazy loading** improves initial page performance  
✅ **Alt text** is essential for accessibility  
✅ **Loading states** prevent layout shift  
✅ **Lightbox** enables detailed viewing  
✅ **Error handling** provides graceful fallbacks  
✅ **Progressive loading** smooths the experience

---

## Further Reading

- [MDN img element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img)
- [Lazy Loading Images](https://web.dev/lazy-loading-images/)
- [WCAG Images](https://www.w3.org/WAI/tutorials/images/)

---

**Previous:** [Link Handling](./04-link-handling.md)  
**Next:** [Table Rendering](./06-table-rendering.md)

<!-- 
Sources Consulted:
- MDN img element: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img
- web.dev lazy loading: https://web.dev/lazy-loading-images/
- WCAG images: https://www.w3.org/WAI/tutorials/images/
-->
