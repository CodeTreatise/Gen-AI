---
title: "Long-Running Operations"
---

# Long-Running Operations

- Handling operations exceeding timeout limits
  - Timeout detection
  - Background job offloading
  - Serverless function limits
  - Edge function considerations
- Job queue patterns
  - Queue selection (Redis, SQS, Bull)
  - Job creation and tracking
  - Worker implementation
  - Concurrency control
- Progress tracking
  - Progress events
  - Percentage completion
  - Stage-based progress
  - UI progress indicators
- Cancellation support
  - User-initiated cancellation
  - Graceful job termination
  - Cleanup on cancel
  - Partial result handling
- Result storage and retrieval
  - Temporary result storage
  - Result expiration
  - Result notification
  - Polling for results
- Background mode (2024-2025)
  - `background: true` parameter
  - Async response retrieval
  - Webhook notifications on completion
  - Deep Research API for long queries
  - Encrypted reasoning for ZDR compliance
  - Background mode timeout limits
