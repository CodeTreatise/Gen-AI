---
title: "Meta Tags & SEO"
---

# Meta Tags & SEO

## Introduction

Meta tags are invisible instructions in your HTML `<head>` that tell browsers, search engines, and social platforms how to handle your page. For AI-powered applications, proper meta tags improve discoverability, enable rich previews when shared, and help search engines understand your content.

This lesson covers essential meta tags, Open Graph for social sharing, structured data for rich results, and SEO best practices.

### What We'll Cover

- Essential meta tags every page needs
- Open Graph and Twitter Card tags
- Structured data (JSON-LD)
- SEO fundamentals for AI applications
- Performance-related meta tags

### Prerequisites

- HTML document structure
- Understanding of the `<head>` element

---

## Essential Meta Tags

### Character Encoding

Always declare encoding first:

```html
<meta charset="UTF-8">
```

This must appear within the first 1024 bytes of the document.

### Viewport for Responsive Design

Required for mobile-friendly sites:

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

| Property | Purpose |
|----------|---------|
| `width=device-width` | Match screen width |
| `initial-scale=1` | No initial zoom |
| `maximum-scale=1` | Prevent zoom (avoid for accessibility) |

### Page Title

Not a meta tag, but critical for SEO:

```html
<title>AI Image Generator - Create Art with AI | YourBrand</title>
```

Best practices:
- 50-60 characters
- Unique per page
- Primary keyword first
- Brand name last

### Description

The snippet shown in search results:

```html
<meta name="description" content="Generate stunning AI artwork in seconds. Free to try, no signup required. Create realistic images, digital art, and illustrations with our advanced AI.">
```

Best practices:
- 150-160 characters
- Compelling call to action
- Include primary keyword
- Unique per page

---

## Complete Head Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Character encoding -->
  <meta charset="UTF-8">
  
  <!-- Viewport -->
  <meta name="viewport" content="width=device-width, initial-scale=1">
  
  <!-- Primary Meta Tags -->
  <title>AI Chat Assistant - Intelligent Conversations | AIBrand</title>
  <meta name="description" content="Chat with our advanced AI assistant. Get instant answers, creative ideas, and helpful solutions. Free to use, no account required.">
  <meta name="keywords" content="AI chat, chatbot, AI assistant, GPT, conversational AI">
  <meta name="author" content="AIBrand Team">
  
  <!-- Favicon -->
  <link rel="icon" href="/favicon.ico" sizes="32x32">
  <link rel="icon" href="/icon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  
  <!-- Open Graph -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://aibrand.com/chat">
  <meta property="og:title" content="AI Chat Assistant - Intelligent Conversations">
  <meta property="og:description" content="Chat with our advanced AI assistant. Get instant answers, creative ideas, and helpful solutions.">
  <meta property="og:image" content="https://aibrand.com/og-image.png">
  
  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@aibrand">
  <meta name="twitter:title" content="AI Chat Assistant">
  <meta name="twitter:description" content="Chat with our advanced AI assistant.">
  <meta name="twitter:image" content="https://aibrand.com/twitter-image.png">
  
  <!-- Canonical URL -->
  <link rel="canonical" href="https://aibrand.com/chat">
  
  <!-- Styles -->
  <link rel="stylesheet" href="/styles.css">
</head>
```

---

## Open Graph Protocol

Open Graph tags control how your page appears when shared on Facebook, LinkedIn, Discord, Slack, and most social platforms.

### Required Tags

```html
<meta property="og:title" content="AI Image Generator">
<meta property="og:type" content="website">
<meta property="og:url" content="https://example.com/generate">
<meta property="og:image" content="https://example.com/og-image.png">
```

### Recommended Tags

```html
<meta property="og:description" content="Create stunning AI art in seconds.">
<meta property="og:site_name" content="AIBrand">
<meta property="og:locale" content="en_US">
```

### Image Guidelines

| Requirement | Specification |
|-------------|---------------|
| Minimum size | 1200 × 630 pixels |
| Aspect ratio | 1.91:1 |
| Format | PNG, JPG, or WebP |
| Max file size | 8MB |
| Use HTTPS | Required for most platforms |

```html
<meta property="og:image" content="https://example.com/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="AI-generated artwork example">
```

### Content Types

| Type | Use For |
|------|---------|
| `website` | Homepage, general pages |
| `article` | Blog posts, news |
| `product` | E-commerce products |
| `profile` | User profiles |

```html
<!-- For blog posts -->
<meta property="og:type" content="article">
<meta property="article:published_time" content="2025-01-15T10:00:00Z">
<meta property="article:author" content="Jane Doe">
<meta property="article:section" content="AI Technology">
```

---

## Twitter Cards

Twitter has its own card system (now X, but tags still work):

### Summary Card

```html
<meta name="twitter:card" content="summary">
<meta name="twitter:site" content="@yourbrand">
<meta name="twitter:title" content="AI Chat Assistant">
<meta name="twitter:description" content="Intelligent conversations powered by AI.">
<meta name="twitter:image" content="https://example.com/card.png">
```

### Large Image Card

```html
<meta name="twitter:card" content="summary_large_image">
```

Best for visual content—shows a large preview image.

### Card Types

| Type | Image Size | Best For |
|------|-----------|----------|
| `summary` | 120×120 min | Articles, general content |
| `summary_large_image` | 300×157 min | Visual content, products |
| `player` | Video player | Video content |
| `app` | App card | Mobile apps |

---

## Structured Data (JSON-LD)

Structured data helps search engines understand your content and can enable rich results (stars, prices, FAQs in search).

### JSON-LD Format

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebApplication",
  "name": "AI Image Generator",
  "description": "Generate AI-powered artwork in seconds",
  "url": "https://example.com/generate",
  "applicationCategory": "MultimediaApplication",
  "operatingSystem": "Web Browser",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  }
}
</script>
```

### Common Schema Types

#### Article

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Understanding Large Language Models",
  "author": {
    "@type": "Person",
    "name": "Jane Doe"
  },
  "datePublished": "2025-01-15",
  "dateModified": "2025-01-20",
  "image": "https://example.com/article-image.jpg",
  "publisher": {
    "@type": "Organization",
    "name": "AI Blog",
    "logo": {
      "@type": "ImageObject",
      "url": "https://example.com/logo.png"
    }
  }
}
</script>
```

#### FAQ Page

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is an AI chatbot?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "An AI chatbot is a software application that uses artificial intelligence to simulate human conversation."
      }
    },
    {
      "@type": "Question",
      "name": "How do I use the AI image generator?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Simply type a description of the image you want and click Generate. The AI will create an image based on your prompt."
      }
    }
  ]
}
</script>
```

#### Software Application

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "AI Writing Assistant",
  "applicationCategory": "BusinessApplication",
  "operatingSystem": "Web",
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.8",
    "ratingCount": "1250"
  },
  "offers": {
    "@type": "Offer",
    "price": "9.99",
    "priceCurrency": "USD"
  }
}
</script>
```

---

## SEO Fundamentals

### Canonical URLs

Prevent duplicate content issues:

```html
<link rel="canonical" href="https://example.com/page">
```

Use when:
- Same content accessible via multiple URLs
- HTTP vs HTTPS versions
- www vs non-www
- URL parameters create duplicates

### Robots Meta Tag

Control search engine behavior:

```html
<!-- Default: allow all -->
<meta name="robots" content="index, follow">

<!-- Block indexing -->
<meta name="robots" content="noindex, nofollow">

<!-- Index but don't follow links -->
<meta name="robots" content="index, nofollow">

<!-- Allow indexing but hide from snippets -->
<meta name="robots" content="index, nosnippet">
```

### Language and Localization

```html
<html lang="en">

<!-- For multilingual sites -->
<link rel="alternate" hreflang="en" href="https://example.com/en/">
<link rel="alternate" hreflang="es" href="https://example.com/es/">
<link rel="alternate" hreflang="x-default" href="https://example.com/">
```

### Heading Structure

```html
<h1>AI Image Generator</h1>  <!-- One per page -->
<h2>How It Works</h2>
<h3>Step 1: Enter Your Prompt</h3>
<h3>Step 2: Choose a Style</h3>
<h2>Gallery</h2>
<h3>Recent Creations</h3>
```

Rules:
- One `<h1>` per page
- Don't skip levels (h1 → h3)
- Use for structure, not styling

---

## Performance Meta Tags

### Preconnect

Establish early connections to external domains:

```html
<link rel="preconnect" href="https://api.openai.com">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
```

### DNS Prefetch

Resolve DNS early:

```html
<link rel="dns-prefetch" href="https://analytics.example.com">
```

### Preload Critical Resources

```html
<link rel="preload" href="/fonts/custom.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/hero-image.webp" as="image">
<link rel="preload" href="/critical.css" as="style">
```

### Theme Color

For mobile browser UI:

```html
<meta name="theme-color" content="#6366f1">
<meta name="theme-color" content="#1a1a2e" media="(prefers-color-scheme: dark)">
```

---

## AI Application SEO Considerations

### Dynamic Content

Search engines can render JavaScript, but:

```html
<!-- Provide essential content in HTML -->
<main id="app">
  <h1>AI Chat Assistant</h1>
  <p>Start a conversation with our intelligent AI...</p>
  <!-- JS app loads here -->
</main>

<!-- Consider server-side rendering for AI apps -->
```

### User-Generated Content

For AI-generated content pages:

```html
<!-- Mark AI-generated content -->
<article data-ai-generated="true">
  <p>This content was generated by AI...</p>
</article>

<!-- Consider noindex for low-quality AI output -->
<meta name="robots" content="noindex">
```

### API Documentation Pages

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "AI API Reference",
  "description": "Complete API documentation for the AI service",
  "proficiencyLevel": "Expert"
}
</script>
```

---

## Testing Your Meta Tags

### Tools

| Tool | Purpose |
|------|---------|
| [Google Rich Results Test](https://search.google.com/test/rich-results) | Test structured data |
| [Facebook Sharing Debugger](https://developers.facebook.com/tools/debug/) | Test Open Graph |
| [Twitter Card Validator](https://cards-dev.twitter.com/validator) | Test Twitter Cards |
| [Schema.org Validator](https://validator.schema.org/) | Validate JSON-LD |

### Chrome DevTools

1. Open DevTools → Elements
2. Check `<head>` section
3. Use Lighthouse for SEO audit

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Unique titles per page | Improves click-through rate |
| Descriptive meta descriptions | Shows in search results |
| Proper Open Graph images | Better social sharing |
| Canonical URLs | Prevents duplicate content |
| Structured data | Enables rich results |
| Fast-loading pages | SEO ranking factor |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Same title on every page | Write unique, descriptive titles |
| Missing Open Graph image | Add 1200×630 image |
| HTTP image URLs | Use HTTPS for all assets |
| Duplicate content | Add canonical tags |
| Missing viewport meta | Always include for mobile |
| Incorrect JSON-LD syntax | Validate with testing tools |

---

## Hands-on Exercise

### Your Task

Create a complete `<head>` section for an AI writing assistant landing page with:

1. All essential meta tags
2. Open Graph tags (title, description, image)
3. Twitter Card (large image)
4. JSON-LD for a SoftwareApplication
5. Favicon links

<details>
<summary>✅ Solution</summary>

```html
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  
  <title>AI Writing Assistant - Write Better, Faster | WriteAI</title>
  <meta name="description" content="Transform your writing with AI. Get instant suggestions, fix grammar, and generate content. Trusted by 100,000+ writers.">
  <meta name="keywords" content="AI writing, writing assistant, grammar checker, content generator">
  
  <!-- Favicon -->
  <link rel="icon" href="/favicon.ico" sizes="32x32">
  <link rel="icon" href="/icon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  
  <!-- Open Graph -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://writeai.com/">
  <meta property="og:title" content="AI Writing Assistant - Write Better, Faster">
  <meta property="og:description" content="Transform your writing with AI. Get instant suggestions, fix grammar, and generate content.">
  <meta property="og:image" content="https://writeai.com/og-image.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:site_name" content="WriteAI">
  
  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@writeai">
  <meta name="twitter:title" content="AI Writing Assistant">
  <meta name="twitter:description" content="Transform your writing with AI.">
  <meta name="twitter:image" content="https://writeai.com/twitter-card.png">
  
  <!-- Canonical -->
  <link rel="canonical" href="https://writeai.com/">
  
  <!-- Theme Color -->
  <meta name="theme-color" content="#6366f1">
  
  <!-- Structured Data -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "WriteAI",
    "description": "AI-powered writing assistant for better, faster writing",
    "url": "https://writeai.com",
    "applicationCategory": "BusinessApplication",
    "operatingSystem": "Web Browser",
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD"
    },
    "aggregateRating": {
      "@type": "AggregateRating",
      "ratingValue": "4.9",
      "ratingCount": "15000"
    }
  }
  </script>
</head>
```
</details>

---

## Summary

✅ Essential meta tags: charset, viewport, title, description

✅ Open Graph controls social media previews—always include og:image (1200×630)

✅ Twitter Cards need separate tags for optimal display

✅ JSON-LD structured data enables rich search results

✅ Canonical URLs prevent duplicate content penalties

✅ Test meta tags with platform-specific validation tools

---

**Previous:** [Accessibility & ARIA](./04-accessibility-aria.md)

**Next:** [Modern HTML Elements](./06-modern-html-elements.md)
