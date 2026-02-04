---
title: "Source & Citation Display"
---

# Source & Citation Display

## Introduction

Modern AI models with web search capabilities return not just answers, but also the sources used to generate those answers. Displaying these citations properly is essential for building trustworthy AI interfaces—users need to verify information and explore original sources.

This lesson covers rendering source parts from AI SDK, creating citation UI components, and handling provider-specific source formats from Perplexity, Google, and OpenAI.

### Why Source Display Matters

- **Trust & Transparency**: Users can verify AI-generated claims
- **Academic Integrity**: Research assistants need proper citations
- **Legal Compliance**: Some domains require source attribution
- **User Experience**: Enables deeper exploration of topics

### What We'll Cover

- Source part types: `source-url` and `source-document`
- Inline citation markers and footer lists
- Hover preview cards and metadata display
- Provider-specific source handling (Perplexity, Google, OpenAI)
- Server-side configuration with `sendSources: true`

### Prerequisites

- [Reasoning & Thinking Display](../17-reasoning-thinking-display/00-reasoning-thinking-display.md)
- AI SDK useChat hook basics
- Understanding of streaming responses

---

## Source-Enabled Models

### Models with Web Search

| Provider | Model | Source Type |
|----------|-------|-------------|
| Perplexity | `sonar-pro`, `sonar` | Web search citations |
| Google | `gemini-2.5-flash` + Google Search | Grounding metadata |
| OpenAI | GPT models + web browsing | Web search citations |
| Custom RAG | Any model | Document sources |

### Source Part Types

AI SDK provides two source part types in the stream protocol:

```typescript
// URL-based sources (web pages)
type SourceURLPart = {
  type: 'source-url';
  sourceId: string;
  url: string;
  title?: string;
};

// Document-based sources (files, PDFs)
type SourceDocumentPart = {
  type: 'source-document';
  sourceId: string;
  mediaType: string;
  title?: string;
};
```

---

## Quick Start Example

### Server: Enable Source Streaming

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
    sendSources: true,  // Enable source streaming
  });
}
```

### Client: Render Sources

```tsx
import { useChat } from '@ai-sdk/react';

export function ChatWithSources() {
  const { messages, sendMessage, status } = useChat();

  return (
    <div>
      {messages.map(message => (
        <div key={message.id}>
          {/* Render text content */}
          {message.parts
            .filter(part => part.type === 'text')
            .map((part, i) => (
              <p key={i}>{part.text}</p>
            ))}

          {/* Render URL sources */}
          {message.parts
            .filter(part => part.type === 'source-url')
            .map(part => (
              <a 
                key={part.id} 
                href={part.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                [{part.title ?? new URL(part.url).hostname}]
              </a>
            ))}
        </div>
      ))}
    </div>
  );
}
```

---

## Lesson Structure

This lesson is organized into focused sub-lessons:

| Lesson | Topic | Description |
|--------|-------|-------------|
| [01](./01-rendering-source-parts.md) | Rendering Source Parts | `source-url`, `source-document`, server config |
| [02](./02-citation-ui-components.md) | Citation UI Components | Inline markers, footer lists, favicons |
| [03](./03-source-preview-patterns.md) | Source Preview Patterns | Hover cards, metadata, link security |
| [04](./04-provider-specific-sources.md) | Provider-Specific Sources | Perplexity, Google grounding, OpenAI |

---

## Key Concepts Preview

### Inline vs Footer Citations

```
Text with inline citations [1] and more sources [2].

──────────────────────────────────────
Sources:
[1] example.com - Article Title
[2] docs.site.com - Documentation
```

### Source Metadata

Sources can include rich metadata:
- **URL**: The source web address
- **Title**: Page or document title
- **Favicon**: Site icon for visual recognition
- **Domain**: Extracted hostname
- **Snippet**: Relevant text excerpt

### Provider Differences

Each provider formats sources differently:
- **Perplexity**: Web search with `sources` array
- **Google**: Grounding metadata with `groundingChunks`
- **OpenAI**: Web browsing citations

---

## Summary

✅ AI models with search return sources alongside responses

✅ AI SDK provides `source-url` and `source-document` part types

✅ Enable streaming with `sendSources: true` in server response

✅ Proper citation display builds user trust

**Next:** [Rendering Source Parts](./01-rendering-source-parts.md)

---

## Further Reading

- [AI SDK Chatbot Sources](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot#sources) — Official documentation
- [Perplexity Provider](https://ai-sdk.dev/providers/ai-sdk-providers/perplexity) — Perplexity sources
- [Google Search Grounding](https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai#google-search) — Google sources

---

<!-- 
Sources Consulted:
- AI SDK Chatbot Sources: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot
- AI SDK Stream Protocol: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
- Perplexity Provider: https://ai-sdk.dev/providers/ai-sdk-providers/perplexity
- Google Provider: https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai
-->
