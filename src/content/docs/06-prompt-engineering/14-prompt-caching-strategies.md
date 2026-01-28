---
title: "Prompt Caching Strategies"
---

# Prompt Caching Strategies

- Identifying cacheable prompt patterns
  - Static prompt portions
  - Repeated prompt structures
  - System prompt stability
  - Common conversation starters
- Static vs. dynamic prompt portions
  - Separating static and dynamic
  - Template-based caching
  - Cache boundaries
  - Update frequency analysis
- Cache key design
  - Key components selection
  - Hash-based keys
  - Version inclusion
  - User/session considerations
- Provider prompt caching features
  - Anthropic prompt caching (explicit cache_control)
  - OpenAI automatic caching (50%+ prefix match)
  - Gemini context caching API
  - Cost savings: up to 90% on cached tokens
  - Cache hit optimization
- **Gemini Context Caching**
  - `client.caches.create()` for persistent cache
  - Specify TTL (time-to-live)
  - Reference cache in subsequent requests
  - Ideal for large system prompts or documents
  - Separate billing for cached vs fresh tokens
- Cache invalidation triggers
  - Prompt updates
  - Model version changes
  - Time-based expiration
  - Manual invalidation
