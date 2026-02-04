---
title: "Message Components"
---

# Message Components

## Introduction

Messages are the building blocks of any chat interface. Each message—whether from a user, the AI, or the system—needs a consistent, well-structured component that handles content display, metadata, actions, and various states.

In this lesson series, we'll build a comprehensive message component system that handles everything from simple text to complex AI-generated content including code, files, and media.

---

## What We'll Cover

This lesson is divided into 7 focused sub-lessons:

| # | Lesson | Description |
|---|--------|-------------|
| 1 | [Message Container Structure](./01-message-container-structure.md) | Wrapper hierarchy, content areas, action placement |
| 2 | [User Message Styling](./02-user-message-styling.md) | Input echo, edit states, attachments, pending states |
| 3 | [AI Response Styling](./03-ai-response-styling.md) | Progressive reveal, citations, confidence indicators |
| 4 | [System Message Handling](./04-system-message-handling.md) | Visual distinction, context info, status updates |
| 5 | [Error Message Display](./05-error-message-display.md) | Error states, retry actions, recovery suggestions |
| 6 | [Message Grouping Strategies](./06-message-grouping-strategies.md) | Sender grouping, time-based, visual separators |
| 7 | [AI-Generated File Display](./07-ai-generated-file-display.md) | Images, media, file downloads from AI responses |

---

## Prerequisites

- HTML/CSS fundamentals ([Unit 1](../../01-web-development-fundamentals/00-overview.md))
- JavaScript DOM manipulation
- Understanding of React components (helpful but not required)
- Completion of [Lesson 1: Chat Interface Design Principles](../01-chat-interface-design-principles/00-chat-interface-design-principles.md)

---

## Component Architecture

A well-designed message component follows this hierarchy:

```
MessageWrapper
├── Avatar (optional)
├── MessageContent
│   ├── MessageHeader (sender, timestamp, model)
│   ├── MessageBody (text, code, media)
│   └── MessageFooter (metadata, token count)
├── MessageActions (copy, edit, regenerate)
└── MessageStatus (pending, sent, error)
```

---

## Message Types

| Type | Description | Visual Treatment |
|------|-------------|------------------|
| **User** | Human input | Right-aligned, colored bubble |
| **AI/Assistant** | Model response | Left-aligned, neutral background |
| **System** | Status/context | Centered, subtle styling |
| **Error** | Failures | Distinct color, retry action |
| **Tool** | Function results | Collapsible, code-style |

---

## Real-World Examples

### ChatGPT
- Clean message containers with copy buttons
- Markdown rendering with syntax highlighting
- Regenerate and edit actions on messages

### Claude
- Thinking indicators for reasoning models
- Artifact panels for generated content
- Source citations for web search results

### GitHub Copilot Chat
- File context badges on messages
- Inline code suggestions with diff view
- Tool invocation displays

---

## Topics Overview

### 1. Message Container Structure
- Semantic wrapper element hierarchy
- Content area layout patterns
- Action button positioning strategies
- Metadata placement options

### 2. User Message Styling
- Input echo display patterns
- Edit mode indicators
- Attachment preview thumbnails
- Pending/sent/failed states

### 3. AI Response Styling
- Progressive streaming reveal
- Completed response indicators
- Source citation formatting
- Confidence/uncertainty display

### 4. System Message Handling
- Visual distinction from chat messages
- Context and status information
- Error summary formatting
- Conversation state updates

### 5. Error Message Display
- Error state visual styling
- Retry action buttons
- Error detail expansion
- Recovery suggestion patterns

### 6. Message Grouping Strategies
- Consecutive same-sender grouping
- Time-based message clustering
- Thread/topic visual grouping
- Date and section dividers

### 7. AI-Generated File Display (2024-2025)
- Handling `part.type === 'file'` in responses
- Image display from `part.url`
- Media type detection via `part.mediaType`
- Lightbox and preview patterns
- Audio/video player embedding
- File download link formatting

---

## Key Takeaways

After completing this lesson series, you will be able to:

✅ Build reusable message components for any chat interface  
✅ Handle all message types (user, AI, system, error)  
✅ Implement message grouping for cleaner conversation flow  
✅ Display AI-generated files and media content  
✅ Create accessible, responsive message layouts

---

## Further Reading

- [Vercel AI SDK UI Components](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot)
- [OpenAI Streaming Responses](https://platform.openai.com/docs/guides/streaming-responses)
- [React Markdown](https://github.com/remarkjs/react-markdown)

---

**Previous:** [Chat Interface Design Principles](../01-chat-interface-design-principles/00-chat-interface-design-principles.md)  
**Next:** [Message Container Structure](./01-message-container-structure.md)
