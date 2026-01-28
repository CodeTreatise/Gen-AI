---
title: "Realtime API & Voice"
---

# Realtime API & Voice

## Overview

This lesson covers real-time voice and audio APIs for building conversational AI applications. You'll learn to implement low-latency speech interactions, voice agents, and multi-modal audio experiences.

### What You'll Learn

1. **Realtime API Overview** — Low-latency multimodal communication
2. **Connection Methods** — WebRTC, WebSocket, SIP, and ephemeral keys
3. **Voice Agent Building** — Agents SDK and RealtimeSession
4. **Audio in Responses API** — Audio input/output modalities
5. **Session Management** — Lifecycle, turn-taking, interruptions
6. **Realtime API Events** — Client and server event patterns
7. **Voice Agent Best Practices** — Latency, flow, and cost optimization
8. **Google Gemini 2.0 Live API** — WebSocket bidirectional streaming
9. **Anthropic Voice Capabilities** — Audio input and integration patterns

### Prerequisites

- Completion of previous lessons in this unit
- Understanding of WebSocket and streaming concepts
- Familiarity with audio formats
- Basic TypeScript/JavaScript knowledge

---

## Lesson Structure

| # | Topic | Focus |
|---|-------|-------|
| 01 | [Realtime API Overview](./01-realtime-api-overview.md) | Low-latency multimodal communication |
| 02 | [Connection Methods](./02-connection-methods.md) | WebRTC, WebSocket, SIP, ephemeral keys |
| 03 | [Voice Agent Building](./03-voice-agent-building.md) | Agents SDK and RealtimeSession |
| 04 | [Audio in Responses API](./04-audio-responses-api.md) | Audio modalities and configuration |
| 05 | [Session Management](./05-session-management.md) | Lifecycle, turn-taking, interruptions |
| 06 | [Realtime API Events](./06-realtime-events.md) | Client and server event handling |
| 07 | [Voice Best Practices](./07-voice-best-practices.md) | Latency, flow, and cost optimization |
| 08 | [Gemini Live API](./08-gemini-live-api.md) | Google's bidirectional streaming |
| 09 | [Anthropic Voice](./09-anthropic-voice.md) | Audio input and integration |

---

## Key Concepts

### Real-Time Audio Communication

Real-time voice AI enables natural conversations:

| Aspect | Traditional API | Realtime API |
|--------|-----------------|--------------|
| Latency | 1-3 seconds | 200-500ms |
| Flow | Request-response | Continuous stream |
| Audio | Pre-recorded | Live capture |
| Interruption | Not supported | Natural handling |

### Connection Options

```
┌─────────────────────────────────────────────────────────────────┐
│                    Realtime Connection Methods                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Browser Apps           Server Apps           Telephony         │
│   ┌─────────┐           ┌─────────┐          ┌─────────┐        │
│   │ WebRTC  │           │WebSocket│          │   SIP   │        │
│   │         │           │         │          │         │        │
│   │ - P2P   │           │ - Full  │          │ - VoIP  │        │
│   │ - Low   │           │   control         │ - PSTN  │        │
│   │   latency│          │ - Server│          │ - PBX   │        │
│   │ - Ephemeral│        │   -side │          │         │        │
│   │   keys  │           │         │          │         │        │
│   └─────────┘           └─────────┘          └─────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Voice Selection

OpenAI Realtime API voices:

| Voice | Style | Best For |
|-------|-------|----------|
| alloy | Neutral | General purpose |
| ash | Warm | Customer service |
| ballad | Expressive | Storytelling |
| coral | Clear | Instructions |
| echo | Authoritative | Announcements |
| sage | Calm | Support |
| shimmer | Friendly | Casual chat |
| verse | Dynamic | Entertainment |

---

## Learning Path

```
Realtime API Overview
        │
        ▼
Connection Methods ──► WebRTC vs WebSocket
        │
        ▼
Voice Agent Building ──► Agents SDK
        │
        ▼
Audio Responses API ──► Modalities & Formats
        │
        ▼
Session Management ──► Lifecycle Events
        │
        ▼
Event Patterns ──► Client/Server Events
        │
        ▼
Best Practices ──► Latency & Flow
        │
        ├──► Gemini Live API
        │
        └──► Anthropic Voice
```

---

## What's Next?

Start with [Realtime API Overview](./01-realtime-api-overview.md) to understand low-latency voice communication fundamentals.

---

## Further Reading

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Official documentation
- [OpenAI Audio Guide](https://platform.openai.com/docs/guides/audio) — Audio capabilities
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) — Google's realtime streaming
