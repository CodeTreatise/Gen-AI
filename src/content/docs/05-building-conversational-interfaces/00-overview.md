---
title: "Unit 5: Building Conversational Interfaces"
---

# Unit 5: Building Conversational Interfaces

## Overview & Importance

Conversational interfaces are the primary way users interact with AI. This unit covers the design and implementation of chat-based interfaces — from basic message displays to sophisticated interfaces with streaming, markdown rendering, and interactive elements.

Chat interfaces must handle unique challenges:
- Displaying incrementally streamed content
- Rendering formatted AI outputs (code, markdown, tables)
- Managing conversation history efficiently
- Providing intuitive user experiences during AI "thinking" time

## Prerequisites

- HTML/CSS proficiency (Unit 1)
- JavaScript DOM manipulation skills
- AI API integration knowledge (Unit 3)
- Understanding of streaming responses

## Learning Objectives

By the end of this unit, you will be able to:
- Design accessible and intuitive chat interfaces
- Implement real-time message streaming display
- Render markdown and code with syntax highlighting
- Manage conversation state and history
- Create responsive loading and typing indicators
- Handle long conversations with virtualization
- Implement message actions (copy, regenerate, edit)
- Use AI SDK useChat hook for React chatbots
- Integrate OpenAI Conversations API for persistent state
- Display reasoning tokens and thinking sections
- Render source citations and web search results
- Build voice-enabled chat interfaces with Realtime API
- Display tool invocations and results in chat
- Implement generative UI with streamed components

## Real-world Applications

- Customer support chatbots
- AI writing assistants
- Code generation interfaces
- Educational tutoring systems
- Healthcare symptom checkers
- Legal document assistants
- Personal AI companions
- Voice-first AI assistants (Realtime API)
- Research assistants with source citations
- Reasoning-enabled problem solvers (o1/o3 interfaces)
- Multi-modal chat with image/file attachments
- Enterprise chatbots with persistent conversations
- Generative UI applications (dynamic forms, booking widgets)
- AI-powered data visualization dashboards

## Market Demand & Relevance

- Chat interfaces are the dominant AI interaction pattern
- Every major tech company has a chat-based AI product
- UI/UX skills for AI interfaces command premium rates
- Growing demand for accessible AI interfaces
- Mobile chat interface skills increasingly valuable
- Specialized role emerging: AI UX Designer
- Voice chat interfaces growing rapidly (2024-2025)
- Reasoning model UIs require specialized thinking display
- AI SDK (Vercel) is industry standard for React chatbots

## Resources

### AI SDK & Frameworks
- [Vercel AI SDK Documentation](https://ai-sdk.dev/docs) — Official AI SDK docs
- [useChat Hook Reference](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot) — Chatbot hook API
- [AI SDK UI Overview](https://ai-sdk.dev/docs/ai-sdk-ui/overview) — UI components guide
- [Transport Configuration](https://ai-sdk.dev/docs/ai-sdk-ui/transport) — Custom transports
- [Message Metadata](https://ai-sdk.dev/docs/ai-sdk-ui/message-metadata) — Token usage tracking
- [Chatbot Tool Usage](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) — Function calling in UI

### OpenAI Conversation APIs
- [Conversation State Guide](https://platform.openai.com/docs/guides/conversation-state) — Multi-turn patterns
- [Conversations API](https://platform.openai.com/docs/api-reference/conversations) — Persistent state
- [Streaming Responses](https://platform.openai.com/docs/guides/streaming-responses) — SSE handling
- [Realtime API WebRTC](https://platform.openai.com/docs/guides/realtime-webrtc) — Voice chat

### Markdown & Code Rendering
- [react-markdown](https://github.com/remarkjs/react-markdown) — Markdown component
- [Shiki](https://shiki.style/) — Syntax highlighting
- [Highlight.js](https://highlightjs.org/) — Code highlighting
- [Prism.js](https://prismjs.com/) — Lightweight syntax highlighting
- [remark-gfm](https://github.com/remarkjs/remark-gfm) — GitHub Flavored Markdown

### Accessibility
- [WAI-ARIA Chat Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/) — ARIA patterns
- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/) — Accessibility standards
- [Inclusive Design Patterns](https://inclusive-components.design/) — Accessible components
- [Screen Reader Testing](https://webaim.org/articles/screenreader_testing/) — Testing guide

### Voice & Realtime
- [OpenAI Agents SDK (TypeScript)](https://openai.github.io/openai-agents-js/) — Voice agents
- [WebRTC Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) — Browser real-time
- [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) — Speech recognition
- [Realtime Console Example](https://github.com/openai/openai-realtime-console/) — Reference implementation

### State Management
- [React Query](https://tanstack.com/query) — Server state management
- [Zustand](https://zustand.docs.pmnd.rs/) — Lightweight state
- [IndexedDB API](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) — Client storage
- [Dexie.js](https://dexie.org/) — IndexedDB wrapper

### Mobile & PWA
- [Web Share API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Share_API) — Native sharing
- [Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API) — Offline support
- [Workbox](https://developer.chrome.com/docs/workbox/) — PWA toolkit
- [Capacitor](https://capacitorjs.com/) — Native mobile wrapper

### Design Resources
- [ChatGPT UI Patterns](https://www.nngroup.com/articles/chatgpt-ux/) — UX research
- [Conversational Design](https://designguidelines.withgoogle.com/conversation/) — Google guidelines
- [AI Interface Patterns](https://www.nngroup.com/articles/ai-paradigm/) — Nielsen Norman Group

### Generative UI
- [AI SDK Generative UI](https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces) — Component streaming
- [AI SDK Chatbot Tool Usage](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) — Tool rendering
- [v0 by Vercel](https://v0.dev/) — AI UI generation example
- [shadcn/ui](https://ui.shadcn.com/) — Component library for AI UIs
