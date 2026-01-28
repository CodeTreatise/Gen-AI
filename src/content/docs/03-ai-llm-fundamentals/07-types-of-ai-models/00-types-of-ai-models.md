---
title: "Types of AI Models"
---

# Types of AI Models

## Overview

The AI landscape includes many specialized model types, each designed for specific tasks. Understanding these categories helps you select the right model for your application needs.

This comprehensive lesson covers all major AI model types—from text generation to emerging 3D capabilities.

## What You'll Learn

This lesson covers fifteen model categories:

### Text & Code
1. **[Text Generation Models](./01-text-generation-models.md)** — GPT-4, Claude, Gemini, LLaMA
2. **[Code Generation Models](./02-code-generation-models.md)** — Codex, CodeLlama, DeepSeek
3. **[Classification Models](./05-classification-models.md)** — Zero-shot, fine-tuned, intent detection

### Search & Safety
4. **[Embedding Models](./03-embedding-models.md)** — Text-to-vector conversion
5. **[Reranking Models](./04-reranking-models.md)** — Search result refinement
6. **[Moderation Models](./06-moderation-models.md)** — Content safety

### Visual & Media
7. **[Image Generation Models](./07-image-generation-models.md)** — DALL-E, Stable Diffusion, Flux
8. **[Image Understanding Models](./08-image-understanding-models.md)** — Vision capabilities
9. **[Audio Models](./09-audio-models.md)** — Speech-to-text, TTS
10. **[Video Models](./10-video-models.md)** — Text-to-video generation

### Advanced & Emerging
11. **[Multimodal Models](./11-multimodal-models.md)** — Multiple input/output types
12. **[Document Understanding](./12-document-understanding-models.md)** — PDF, layout analysis
13. **[Agent & Tool-Use Models](./13-agent-tool-use-models.md)** — Planning, execution
14. **[Computer Use Models](./14-computer-use-models.md)** — Screen control, automation
15. **[3D & Spatial Models](./15-3d-spatial-models.md)** — Emerging capabilities

## Prerequisites

Before starting this lesson, you should have:

- Completed [Streaming & Response Modes](../06-streaming-response-modes/00-streaming-response-modes.md)
- Basic understanding of machine learning concepts
- Familiarity with API integration

## Model Landscape Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI MODEL TYPES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TEXT                    VISUAL                  AUDIO           │
│  ├─ Generation           ├─ Image Generation     ├─ STT          │
│  ├─ Code                 ├─ Image Understanding  ├─ TTS          │
│  ├─ Classification       ├─ Video Generation     └─ Music        │
│  └─ Embeddings           └─ 3D Generation                        │
│                                                                  │
│  SEARCH & SAFETY         ADVANCED                               │
│  ├─ Embeddings           ├─ Multimodal                          │
│  ├─ Reranking            ├─ Document Understanding               │
│  └─ Moderation           ├─ Agents & Tools                      │
│                          └─ Computer Use                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Reference

| Category | Primary Use | Key Models |
|----------|-------------|------------|
| Text Generation | Chat, content | GPT-4, Claude, Gemini |
| Code Generation | Programming | Codex, DeepSeek, CodeLlama |
| Embeddings | Semantic search | text-embedding-3, Cohere |
| Reranking | Search refinement | Cohere Rerank, Voyage |
| Classification | Categorization | Zero-shot, fine-tuned |
| Moderation | Safety | OpenAI Moderation |
| Image Generation | Art, design | DALL-E 3, Flux, SD |
| Vision | Image analysis | GPT-4V, Claude Vision |
| Audio | Speech | Whisper, ElevenLabs |
| Video | Video creation | Sora, Runway |
| Multimodal | Multiple types | GPT-4o, Gemini |
| Documents | PDF processing | Claude, Gemini |
| Agents | Automation | Claude, GPT-4 |
| Computer Use | GUI control | Claude Computer Use |

## Learning Path

Complete topics in order for comprehensive understanding:

```
01-text-generation → 02-code-generation → 03-embeddings
                                              ↓
06-moderation ← 05-classification ← 04-reranking
      ↓
07-image-generation → 08-image-understanding → 09-audio
                                                  ↓
12-document-understanding ← 11-multimodal ← 10-video
      ↓
13-agent-tool-use → 14-computer-use → 15-3d-spatial
```

---

## Summary

This lesson provides a comprehensive overview of AI model types. Each topic covers specific capabilities, use cases, and implementation guidance.

**Next:** [Text Generation Models](./01-text-generation-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Streaming Response Modes](../06-streaming-response-modes/00-streaming-response-modes.md) | [AI/LLM Fundamentals](../00-overview.md) | [Text Generation Models](./01-text-generation-models.md) |

