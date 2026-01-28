---
title: "OpenAI Conversations API"
---

# OpenAI Conversations API

- Conversations API overview
  - Persistent conversation objects
  - Durable conversation IDs
  - Cross-session state preservation
  - Cross-device continuity
- Creating conversations
  - `openai.conversations.create()`
  - Conversation items (messages, tool calls)
  - Conversation metadata
  - Listing conversations
- Automatic context chaining
  - `previous_response_id` parameter
  - Stateless multi-turn conversations
  - No manual message array management
  - Server-side context preservation
- Context window management
  - Token limit awareness
  - `/responses/compact` endpoint
  - Encrypted compaction items
  - ZDR-compatible compression
  - Compaction instructions parameter
- Response storage control
  - `store: true` for persistence
  - `store: false` for ephemeral
  - Data retention implications
  - Compliance considerations
