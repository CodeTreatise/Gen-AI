---
title: "Rendering Source Parts"
---

# Rendering Source Parts

## Introduction

AI SDK streams source information as distinct message parts that you can render alongside text content. Understanding the structure of these parts and how to enable source streaming is the foundation for building citation-aware interfaces.

This lesson covers the `source-url` and `source-document` part types, server-side configuration with `sendSources`, and basic rendering patterns.

### What We'll Cover

- Source part types and their properties
- Enabling source streaming on the server
- Detecting and filtering source parts
- Basic source rendering components
- Handling source IDs for deduplication

### Prerequisites

- [Source & Citation Display Overview](./00-source-citation-display.md)
- AI SDK streaming basics
- React component fundamentals

---

## Source Part Types

### source-url Part

URL sources reference web pages used to ground the response:

```typescript
interface SourceURLPart {
  type: 'source-url';
  id: string;           // Unique identifier
  sourceId: string;     // Provider's source ID
  url: string;          // Full URL of the source
  title?: string;       // Page title (if available)
}
```

**Example from stream:**
```json
{
  "type": "source-url",
  "sourceId": "https://example.com/article",
  "url": "https://example.com/article",
  "title": "Understanding AI Citations"
}
```

### source-document Part

Document sources reference files or internal documents:

```typescript
interface SourceDocumentPart {
  type: 'source-document';
  id: string;           // Unique identifier
  sourceId: string;     // Provider's source ID
  mediaType: string;    // MIME type (e.g., 'application/pdf')
  title?: string;       // Document title
}
```

**Example from stream:**
```json
{
  "type": "source-document",
  "sourceId": "doc-123",
  "mediaType": "application/pdf",
  "title": "Research Paper 2024"
}
```

---

## Server Configuration

### Enabling Source Streaming

Add `sendSources: true` to your streaming response:

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
    sendSources: true,  // Streams source parts to client
  });
}
```

### Combined Streaming Options

You can combine source streaming with other options:

```typescript
return result.toUIMessageStreamResponse({
  sendSources: true,      // Include sources
  sendReasoning: true,    // Include reasoning (if available)
  messageMetadata: async ({ usage }) => ({
    tokens: usage?.totalTokens ?? 0,
  }),
});
```

---

## Detecting Source Parts

### Basic Part Filtering

```tsx
import type { UIMessage } from 'ai';

function getSourceParts(message: UIMessage) {
  const urlSources = message.parts.filter(
    part => part.type === 'source-url'
  );
  
  const docSources = message.parts.filter(
    part => part.type === 'source-document'
  );
  
  return { urlSources, docSources };
}
```

### Type-Safe Source Detection

```tsx
type SourcePart = 
  | { type: 'source-url'; id: string; sourceId: string; url: string; title?: string }
  | { type: 'source-document'; id: string; sourceId: string; mediaType: string; title?: string };

function isSourcePart(part: { type: string }): part is SourcePart {
  return part.type === 'source-url' || part.type === 'source-document';
}

function extractSources(message: UIMessage): SourcePart[] {
  return message.parts.filter(isSourcePart);
}
```

---

## Basic Source Rendering

### Simple Source List

```tsx
interface SourceListProps {
  message: UIMessage;
}

export function SourceList({ message }: SourceListProps) {
  const sources = message.parts.filter(
    part => part.type === 'source-url' || part.type === 'source-document'
  );

  if (sources.length === 0) return null;

  return (
    <div className="source-list">
      <h4>Sources</h4>
      <ul>
        {sources.map((source, index) => (
          <li key={source.id || index}>
            {source.type === 'source-url' ? (
              <a 
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                {source.title ?? source.url}
              </a>
            ) : (
              <span>{source.title ?? `Document ${index + 1}`}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

```css
.source-list {
  margin-top: 16px;
  padding: 12px 16px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.source-list h4 {
  margin: 0 0 8px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}

.source-list ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.source-list li {
  padding: 6px 0;
  border-bottom: 1px solid #e2e8f0;
}

.source-list li:last-child {
  border-bottom: none;
}

.source-list a {
  color: #2563eb;
  text-decoration: none;
  font-size: 0.875rem;
}

.source-list a:hover {
  text-decoration: underline;
}
```

---

## URL Source Component

### Detailed URL Source

```tsx
interface URLSourceProps {
  source: {
    id: string;
    url: string;
    title?: string;
  };
  index: number;
}

export function URLSource({ source, index }: URLSourceProps) {
  const hostname = new URL(source.url).hostname;
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;
  
  return (
    <a
      className="url-source"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="source-index">[{index + 1}]</span>
      <img 
        src={faviconUrl}
        alt=""
        className="source-favicon"
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = 'none';
        }}
      />
      <div className="source-info">
        <span className="source-title">
          {source.title ?? hostname}
        </span>
        <span className="source-domain">{hostname}</span>
      </div>
    </a>
  );
}
```

```css
.url-source {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;
}

.url-source:hover {
  border-color: #3b82f6;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.source-index {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 600;
  color: #3b82f6;
  min-width: 24px;
}

.source-favicon {
  width: 16px;
  height: 16px;
  border-radius: 2px;
}

.source-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.source-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.source-domain {
  font-size: 0.75rem;
  color: #94a3b8;
}
```

---

## Document Source Component

### Document Source with Icon

```tsx
interface DocumentSourceProps {
  source: {
    id: string;
    mediaType: string;
    title?: string;
  };
  index: number;
}

export function DocumentSource({ source, index }: DocumentSourceProps) {
  const icon = getDocumentIcon(source.mediaType);
  const typeLabel = getTypeLabel(source.mediaType);
  
  return (
    <div className="doc-source">
      <span className="source-index">[{index + 1}]</span>
      <span className="doc-icon">{icon}</span>
      <div className="source-info">
        <span className="source-title">
          {source.title ?? `Document ${index + 1}`}
        </span>
        <span className="source-type">{typeLabel}</span>
      </div>
    </div>
  );
}

function getDocumentIcon(mediaType: string): string {
  const icons: Record<string, string> = {
    'application/pdf': '📄',
    'text/plain': '📝',
    'text/markdown': '📑',
    'application/json': '📋',
    'text/csv': '📊',
  };
  return icons[mediaType] || '📎';
}

function getTypeLabel(mediaType: string): string {
  const labels: Record<string, string> = {
    'application/pdf': 'PDF Document',
    'text/plain': 'Text File',
    'text/markdown': 'Markdown',
    'application/json': 'JSON',
    'text/csv': 'CSV Data',
  };
  return labels[mediaType] || mediaType;
}
```

```css
.doc-source {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: #fefce8;
  border: 1px solid #fef08a;
  border-radius: 8px;
}

.doc-icon {
  font-size: 1.25rem;
}

.source-type {
  font-size: 0.75rem;
  color: #a16207;
}
```

---

## Deduplicating Sources

### Using sourceId for Deduplication

Sources can appear multiple times in a response. Use `sourceId` to deduplicate:

```tsx
function deduplicateSources(parts: UIMessage['parts']) {
  const seen = new Set<string>();
  const unique: SourcePart[] = [];
  
  for (const part of parts) {
    if (part.type === 'source-url' || part.type === 'source-document') {
      if (!seen.has(part.sourceId)) {
        seen.add(part.sourceId);
        unique.push(part);
      }
    }
  }
  
  return unique;
}
```

### Component with Deduplication

```tsx
export function DeduplicatedSourceList({ message }: { message: UIMessage }) {
  const sources = deduplicateSources(message.parts);
  
  if (sources.length === 0) return null;
  
  return (
    <div className="sources-container">
      <h4>Sources ({sources.length})</h4>
      <div className="sources-grid">
        {sources.map((source, index) => (
          source.type === 'source-url' ? (
            <URLSource key={source.sourceId} source={source} index={index} />
          ) : (
            <DocumentSource key={source.sourceId} source={source} index={index} />
          )
        ))}
      </div>
    </div>
  );
}
```

```css
.sources-container {
  margin-top: 20px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.sources-container h4 {
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

@media (min-width: 640px) {
  .sources-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
```

---

## Complete Message Renderer

### Full Message with Sources

```tsx
import type { UIMessage } from 'ai';

interface MessageWithSourcesProps {
  message: UIMessage;
}

export function MessageWithSources({ message }: MessageWithSourcesProps) {
  // Separate text from sources
  const textParts = message.parts.filter(part => part.type === 'text');
  const sources = deduplicateSources(message.parts);
  
  return (
    <div className={`message ${message.role}`}>
      {/* Message content */}
      <div className="message-content">
        {textParts.map((part, index) => (
          <p key={index}>{part.text}</p>
        ))}
      </div>
      
      {/* Sources section */}
      {sources.length > 0 && (
        <div className="message-sources">
          <div className="sources-header">
            <span className="sources-icon">🔗</span>
            <span>Sources ({sources.length})</span>
          </div>
          <div className="sources-list">
            {sources.map((source, index) => (
              source.type === 'source-url' ? (
                <URLSource 
                  key={source.sourceId} 
                  source={source} 
                  index={index} 
                />
              ) : (
                <DocumentSource 
                  key={source.sourceId} 
                  source={source} 
                  index={index} 
                />
              )
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Summary

✅ AI SDK provides `source-url` and `source-document` part types

✅ Enable source streaming with `sendSources: true` on the server

✅ Filter message parts by type to extract sources

✅ Use `sourceId` to deduplicate repeated sources

✅ Create specialized components for URL vs document sources

**Next:** [Citation UI Components](./02-citation-ui-components.md)

---

## Further Reading

- [AI SDK Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) — Source part format
- [AI SDK Chatbot Sources](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot#sources) — Official example

---

<!-- 
Sources Consulted:
- AI SDK Stream Protocol: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
- AI SDK Chatbot Documentation: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
-->
