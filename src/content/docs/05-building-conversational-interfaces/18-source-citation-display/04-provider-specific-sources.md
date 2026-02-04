---
title: "Provider-Specific Sources"
---

# Provider-Specific Sources

## Introduction

Different AI providers format their source citations in unique ways. Perplexity returns sources as part of the standard AI SDK response, Google provides rich grounding metadata with confidence scores, and OpenAI web search returns citations through browsing results. Understanding these differences is essential for building robust citation displays.

This lesson covers provider-specific source formats, extracting grounding metadata, and creating unified source handlers.

### What We'll Cover

- Perplexity Sonar sources and images
- Google grounding metadata and chunks
- OpenAI web search citations
- Unified source normalization
- Custom source handlers

### Prerequisites

- [Source Preview Patterns](./03-source-preview-patterns.md)
- AI SDK provider configuration
- TypeScript type narrowing

---

## Perplexity Sources

### Basic Perplexity Integration

Perplexity's Sonar models return sources directly in the response:

```typescript
// app/api/chat/route.ts
import { perplexity } from '@ai-sdk/perplexity';
import { streamText, UIMessage, convertToModelMessages } from 'ai';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: perplexity('sonar-pro'),
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse({
    sendSources: true,
  });
}
```

### Accessing Sources from generateText

```typescript
import { perplexity } from '@ai-sdk/perplexity';
import { generateText } from 'ai';

const { text, sources } = await generateText({
  model: perplexity('sonar-pro'),
  prompt: 'What are the latest developments in quantum computing?',
});

// sources is an array of source objects
console.log(sources);
// [
//   { url: 'https://example.com/article1', title: 'Quantum Breakthrough' },
//   { url: 'https://example.com/article2', title: 'Computing Advances' }
// ]
```

### Perplexity Provider Metadata

```typescript
const result = await generateText({
  model: perplexity('sonar-pro'),
  prompt: 'Latest AI news',
  providerOptions: {
    perplexity: {
      return_images: true, // Tier-2 users only
    },
  },
});

// Access provider-specific metadata
const metadata = result.providerMetadata?.perplexity;

console.log(metadata);
// {
//   usage: { citationTokens: 5286, numSearchQueries: 1 },
//   images: [
//     { imageUrl: '...', originUrl: '...', height: 1280, width: 720 }
//   ]
// }
```

### Perplexity Source Component

```tsx
interface PerplexitySourcesProps {
  sources: Array<{ url: string; title?: string }>;
  images?: Array<{ imageUrl: string; originUrl: string }>;
}

export function PerplexitySources({ sources, images }: PerplexitySourcesProps) {
  return (
    <div className="perplexity-sources">
      {/* Image results */}
      {images && images.length > 0 && (
        <div className="source-images">
          {images.slice(0, 3).map((img, index) => (
            <a
              key={index}
              href={img.originUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="source-image-link"
            >
              <img src={img.imageUrl} alt="" />
            </a>
          ))}
        </div>
      )}
      
      {/* Text sources */}
      <div className="source-list">
        <h4>Sources</h4>
        <ol>
          {sources.map((source, index) => (
            <li key={index}>
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                {source.title ?? new URL(source.url).hostname}
              </a>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
```

---

## Google Grounding Sources

### Google Search Grounding Setup

```typescript
// app/api/chat/route.ts
import { google } from '@ai-sdk/google';
import { streamText, UIMessage, convertToModelMessages } from 'ai';

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: google('gemini-2.5-flash'),
    messages: await convertToModelMessages(messages),
    tools: {
      google_search: google.tools.googleSearch({}),
    },
  });

  return result.toUIMessageStreamResponse({
    sendSources: true,
  });
}
```

### Google Grounding Metadata

```typescript
import { google, GoogleGenerativeAIProviderMetadata } from '@ai-sdk/google';
import { generateText } from 'ai';

const { text, sources, providerMetadata } = await generateText({
  model: google('gemini-2.5-flash'),
  tools: {
    google_search: google.tools.googleSearch({}),
  },
  prompt: 'What are the top tech news today?',
});

// Access grounding metadata
const metadata = providerMetadata?.google as GoogleGenerativeAIProviderMetadata | undefined;
const groundingMetadata = metadata?.groundingMetadata;

// Structure of groundingMetadata:
// {
//   webSearchQueries: ['tech news today'],
//   searchEntryPoint: { renderedContent: '...' },
//   groundingChunks: [
//     { web: { uri: '...', title: '...' } }
//   ],
//   groundingSupports: [
//     {
//       segment: { startIndex: 0, endIndex: 65, text: '...' },
//       groundingChunkIndices: [0],
//       confidenceScores: [0.99]
//     }
//   ]
// }
```

### Google Grounding Component

```tsx
interface GroundingChunk {
  web?: { uri: string; title?: string };
  maps?: { uri: string; title?: string; placeId?: string };
  retrievedContext?: { uri: string; title?: string; text?: string };
}

interface GroundingSupport {
  segment: { startIndex: number; endIndex: number; text: string };
  groundingChunkIndices: number[];
  confidenceScores: number[];
}

interface GoogleGroundingProps {
  groundingChunks?: GroundingChunk[];
  groundingSupports?: GroundingSupport[];
  searchQueries?: string[];
}

export function GoogleGroundingSources({
  groundingChunks,
  groundingSupports,
  searchQueries,
}: GoogleGroundingProps) {
  if (!groundingChunks || groundingChunks.length === 0) {
    return null;
  }
  
  return (
    <div className="google-grounding">
      {/* Search queries used */}
      {searchQueries && searchQueries.length > 0 && (
        <div className="search-queries">
          <span className="queries-label">Searched:</span>
          {searchQueries.map((query, i) => (
            <span key={i} className="query-tag">{query}</span>
          ))}
        </div>
      )}
      
      {/* Grounding sources */}
      <div className="grounding-sources">
        <h4>Sources</h4>
        {groundingChunks.map((chunk, index) => (
          <GroundingChunkCard
            key={index}
            chunk={chunk}
            index={index}
            support={groundingSupports?.find(s => 
              s.groundingChunkIndices.includes(index)
            )}
          />
        ))}
      </div>
    </div>
  );
}

function GroundingChunkCard({
  chunk,
  index,
  support,
}: {
  chunk: GroundingChunk;
  index: number;
  support?: GroundingSupport;
}) {
  const source = chunk.web || chunk.maps || chunk.retrievedContext;
  if (!source) return null;
  
  const confidence = support?.confidenceScores[0];
  
  return (
    <a
      href={source.uri}
      target="_blank"
      rel="noopener noreferrer"
      className="grounding-card"
    >
      <div className="card-header">
        <span className="source-index">[{index + 1}]</span>
        {chunk.maps && <span className="source-type">📍 Maps</span>}
        {chunk.retrievedContext && <span className="source-type">📄 Document</span>}
      </div>
      
      <div className="card-title">
        {source.title ?? new URL(source.uri).hostname}
      </div>
      
      {confidence !== undefined && (
        <div className="confidence-bar">
          <div 
            className="confidence-fill"
            style={{ width: `${confidence * 100}%` }}
          />
          <span className="confidence-label">
            {Math.round(confidence * 100)}% confidence
          </span>
        </div>
      )}
    </a>
  );
}
```

```css
.google-grounding {
  margin-top: 16px;
  padding: 16px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 12px;
}

.search-queries {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.queries-label {
  font-size: 0.75rem;
  color: #64748b;
}

.query-tag {
  padding: 4px 10px;
  background: white;
  border: 1px solid #d1fae5;
  border-radius: 6px;
  font-size: 0.75rem;
  color: #166534;
}

.grounding-sources h4 {
  margin: 0 0 12px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #166534;
}

.grounding-card {
  display: block;
  padding: 12px;
  background: white;
  border: 1px solid #d1fae5;
  border-radius: 8px;
  text-decoration: none;
  color: inherit;
  margin-bottom: 8px;
  transition: all 0.2s;
}

.grounding-card:hover {
  border-color: #4ade80;
  box-shadow: 0 2px 8px rgba(34, 197, 94, 0.15);
}

.grounding-card:last-child {
  margin-bottom: 0;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.source-index {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: #16a34a;
}

.source-type {
  font-size: 0.7rem;
  color: #64748b;
}

.card-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #166534;
}

.confidence-bar {
  position: relative;
  margin-top: 8px;
  height: 4px;
  background: #d1fae5;
  border-radius: 2px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: #22c55e;
  transition: width 0.3s ease;
}

.confidence-label {
  position: absolute;
  right: 0;
  top: 8px;
  font-size: 0.65rem;
  color: #64748b;
}
```

---

## OpenAI Web Search

### OpenAI with Web Browsing

OpenAI's web browsing capability returns citations through the response:

```typescript
// Using OpenAI Responses API with web search
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

const { text, sources } = await generateText({
  model: openai('gpt-4o'),
  prompt: 'What are today\'s top tech stories?',
  // Web search is automatically enabled for certain prompts
});

// Sources returned as array
console.log(sources);
```

### OpenAI Citation Format

```tsx
interface OpenAICitation {
  url: string;
  title?: string;
  snippet?: string;
  publishedDate?: string;
}

interface OpenAISourcesProps {
  citations: OpenAICitation[];
}

export function OpenAISources({ citations }: OpenAISourcesProps) {
  if (citations.length === 0) return null;
  
  return (
    <div className="openai-sources">
      <h4>Web Sources</h4>
      <div className="citation-cards">
        {citations.map((citation, index) => (
          <a
            key={index}
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="citation-card"
          >
            <div className="card-index">{index + 1}</div>
            <div className="card-content">
              <div className="card-title">
                {citation.title ?? 'Source'}
              </div>
              {citation.snippet && (
                <div className="card-snippet">
                  {citation.snippet}
                </div>
              )}
              <div className="card-meta">
                <span className="card-domain">
                  {new URL(citation.url).hostname}
                </span>
                {citation.publishedDate && (
                  <span className="card-date">
                    {formatDate(citation.publishedDate)}
                  </span>
                )}
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function formatDate(date: string): string {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}
```

---

## Unified Source Normalization

### Normalized Source Interface

```typescript
interface NormalizedSource {
  id: string;
  type: 'url' | 'document' | 'map';
  url: string;
  title: string;
  domain: string;
  confidence?: number;
  snippet?: string;
  publishedDate?: string;
  imageUrl?: string;
  provider: 'perplexity' | 'google' | 'openai' | 'custom';
}
```

### Source Normalizer

```typescript
export function normalizePerplexitySources(
  sources: Array<{ url: string; title?: string }>
): NormalizedSource[] {
  return sources.map((source, index) => ({
    id: `perplexity-${index}`,
    type: 'url' as const,
    url: source.url,
    title: source.title ?? 'Untitled',
    domain: new URL(source.url).hostname.replace('www.', ''),
    provider: 'perplexity' as const,
  }));
}

export function normalizeGoogleSources(
  chunks?: Array<{
    web?: { uri: string; title?: string };
    maps?: { uri: string; title?: string };
    retrievedContext?: { uri: string; title?: string };
  }>,
  supports?: Array<{
    groundingChunkIndices: number[];
    confidenceScores: number[];
  }>
): NormalizedSource[] {
  if (!chunks) return [];
  
  return chunks.map((chunk, index) => {
    const source = chunk.web || chunk.maps || chunk.retrievedContext;
    if (!source) return null;
    
    const support = supports?.find(s => 
      s.groundingChunkIndices.includes(index)
    );
    
    return {
      id: `google-${index}`,
      type: chunk.maps ? 'map' : chunk.retrievedContext ? 'document' : 'url',
      url: source.uri,
      title: source.title ?? 'Untitled',
      domain: new URL(source.uri).hostname.replace('www.', ''),
      confidence: support?.confidenceScores[0],
      provider: 'google' as const,
    };
  }).filter((s): s is NormalizedSource => s !== null);
}

export function normalizeOpenAISources(
  citations: Array<{
    url: string;
    title?: string;
    snippet?: string;
    publishedDate?: string;
  }>
): NormalizedSource[] {
  return citations.map((citation, index) => ({
    id: `openai-${index}`,
    type: 'url' as const,
    url: citation.url,
    title: citation.title ?? 'Untitled',
    domain: new URL(citation.url).hostname.replace('www.', ''),
    snippet: citation.snippet,
    publishedDate: citation.publishedDate,
    provider: 'openai' as const,
  }));
}
```

### Universal Source Component

```tsx
interface UniversalSourcesProps {
  sources: NormalizedSource[];
}

export function UniversalSources({ sources }: UniversalSourcesProps) {
  if (sources.length === 0) return null;
  
  // Group by provider for visual distinction
  const grouped = groupBy(sources, 'provider');
  
  return (
    <div className="universal-sources">
      <h4>Sources ({sources.length})</h4>
      
      <div className="sources-grid">
        {sources.map((source, index) => (
          <UniversalSourceCard
            key={source.id}
            source={source}
            index={index}
          />
        ))}
      </div>
    </div>
  );
}

function UniversalSourceCard({
  source,
  index,
}: {
  source: NormalizedSource;
  index: number;
}) {
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${source.domain}&sz=32`;
  
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`universal-card provider-${source.provider}`}
    >
      <div className="card-left">
        <span className="card-index">{index + 1}</span>
        <img 
          src={faviconUrl} 
          alt="" 
          className="card-favicon"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = 'none';
          }}
        />
      </div>
      
      <div className="card-content">
        <div className="card-title">{source.title}</div>
        <div className="card-domain">{source.domain}</div>
        
        {source.snippet && (
          <div className="card-snippet">{source.snippet}</div>
        )}
        
        {source.confidence !== undefined && (
          <div className="card-confidence">
            {Math.round(source.confidence * 100)}% match
          </div>
        )}
      </div>
      
      <div className={`provider-badge ${source.provider}`}>
        {getProviderIcon(source.provider)}
      </div>
    </a>
  );
}

function getProviderIcon(provider: string): string {
  const icons: Record<string, string> = {
    perplexity: '🔍',
    google: '🌐',
    openai: '🤖',
    custom: '📎',
  };
  return icons[provider] || '📄';
}

function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((acc, item) => {
    const group = String(item[key]);
    acc[group] = acc[group] || [];
    acc[group].push(item);
    return acc;
  }, {} as Record<string, T[]>);
}
```

```css
.universal-sources {
  margin-top: 20px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.universal-sources h4 {
  margin: 0 0 12px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}

.sources-grid {
  display: grid;
  gap: 8px;
}

.universal-card {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;
}

.universal-card:hover {
  border-color: #3b82f6;
}

/* Provider-specific accent colors */
.universal-card.provider-perplexity {
  border-left: 3px solid #8b5cf6;
}

.universal-card.provider-google {
  border-left: 3px solid #22c55e;
}

.universal-card.provider-openai {
  border-left: 3px solid #10b981;
}

.card-left {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.card-index {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: #3b82f6;
}

.card-favicon {
  width: 16px;
  height: 16px;
  border-radius: 2px;
}

.card-content {
  flex: 1;
  min-width: 0;
}

.card-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card-domain {
  font-size: 0.75rem;
  color: #94a3b8;
  margin-top: 2px;
}

.card-snippet {
  font-size: 0.75rem;
  color: #64748b;
  margin-top: 6px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-confidence {
  font-size: 0.7rem;
  color: #16a34a;
  margin-top: 4px;
}

.provider-badge {
  font-size: 0.875rem;
  opacity: 0.7;
}
```

---

## Summary

✅ Perplexity returns sources via `sources` array with optional images

✅ Google provides `groundingMetadata` with chunks, supports, and confidence scores

✅ OpenAI web search returns citations through browsing results

✅ Normalize sources to a common interface for unified rendering

✅ Use provider-specific accent colors for visual distinction

✅ Extract confidence scores to show source reliability

**Previous:** [Source Preview Patterns](./03-source-preview-patterns.md)

---

## Further Reading

- [Perplexity Provider](https://ai-sdk.dev/providers/ai-sdk-providers/perplexity) — Perplexity sources
- [Google Search Grounding](https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai#google-search) — Google sources
- [AI SDK Chatbot Sources](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot#sources) — Sources documentation

---

<!-- 
Sources Consulted:
- Perplexity Provider: https://ai-sdk.dev/providers/ai-sdk-providers/perplexity
- Google Provider: https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai
- AI SDK Chatbot: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
-->
