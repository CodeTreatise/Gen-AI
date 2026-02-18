---
title: "Helicone Integration"
---

# Helicone Integration

## Introduction

Helicone takes a different approach to LLM observability: instead of SDK decorators or trace processors, it operates as an **AI Gateway** — an OpenAI-compatible proxy that sits between your application and any LLM provider. Route your API calls through Helicone's gateway and you get **automatic logging, cost monitoring, rate limit tracking, caching, and user analytics** with zero code changes beyond updating the base URL.

This proxy-based approach means Helicone works with any language, any framework, and any LLM provider. It supports 100+ models across OpenAI, Anthropic, Google, Groq, Mistral, and more — all through a single unified endpoint.

### What we'll cover

- Setting up Helicone as an AI gateway
- Request logging and trace exploration
- Cost monitoring and budgets
- Rate limit tracking and retry logic
- User-level analytics
- Caching for cost reduction
- Custom properties and filtering

### Prerequisites

- An OpenAI or Anthropic API key
- A free Helicone account at [helicone.ai](https://helicone.ai/signup)
- Basic LLM API usage knowledge (Unit 4)

---

## Setting up the AI gateway

Helicone works by proxying your LLM requests. Change the base URL in your OpenAI client and pass your Helicone API key as a header.

### Python setup

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ai-gateway.helicone.ai",
    api_key="your-helicone-api-key",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is an AI gateway?"}],
)
print(response.choices[0].message.content)
```

**Output:**
```
An AI gateway is a proxy server that sits between your application and
LLM providers, adding logging, caching, rate limiting, and monitoring.
```

### TypeScript setup

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://ai-gateway.helicone.ai",
  apiKey: process.env.HELICONE_API_KEY,
});

const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello from TypeScript!" }],
});
console.log(response.choices[0].message.content);
```

### cURL

```bash
curl -X POST https://ai-gateway.helicone.ai/v1/chat/completions \
  -H "Authorization: Bearer $HELICONE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

After any of these calls, navigate to the [Requests tab](https://us.helicone.ai/requests) in the Helicone dashboard. Your request appears within seconds with full details.

> **🔑 Key insight:** Because Helicone is OpenAI-compatible, you can use it with **any model** — just change the `model` field. Helicone routes the request to the correct provider automatically.

---

## Request logging and exploration

Every request through the gateway is logged automatically. The dashboard shows:

| Field | Description |
|-------|-------------|
| **Model** | Which model handled the request |
| **Prompt** | Full input messages |
| **Response** | Complete model output |
| **Tokens** | Input + output token counts |
| **Cost** | Calculated cost based on provider pricing |
| **Latency** | Time from request to last byte |
| **Status** | HTTP status code (200, 429, 500, etc.) |
| **Timestamp** | When the request was made |

### Adding custom properties

Attach metadata to requests via headers for filtering and grouping:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this document"}],
    extra_headers={
        "Helicone-Property-UserId": "user_42",
        "Helicone-Property-Feature": "document-summary",
        "Helicone-Property-Environment": "production",
        "Helicone-Property-Version": "v2.1",
    },
)
```

In the dashboard, you can filter requests by any custom property — e.g., "Show me all requests from user_42 in production."

---

## Cost monitoring

Helicone tracks costs across all providers in one dashboard. For agent workflows that make multiple LLM calls, this is essential for budget management.

### Dashboard metrics

- **Total spend**: Aggregate cost over any time period
- **Cost per request**: Average cost per API call
- **Cost by model**: Breakdown across gpt-4o, gpt-4o-mini, Claude, etc.
- **Cost by user**: Per-user spending with custom properties
- **Cost trends**: Daily/weekly/monthly spend graphs

### Setting up budget alerts

In the Helicone dashboard:
1. Navigate to **Settings → Alerts**
2. Create a new alert: "Notify when daily spend exceeds $50"
3. Choose notification method (email, Slack, webhook)

### Tracking agent costs

For multi-step agents, tag each step with a property to track cost per workflow:

```python
# Step 1: Research
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Research topic X"}],
    extra_headers={
        "Helicone-Property-Workflow": "research-agent",
        "Helicone-Property-Step": "research",
    },
)

# Step 2: Synthesize
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Synthesize: {response1.choices[0].message.content}"}],
    extra_headers={
        "Helicone-Property-Workflow": "research-agent",
        "Helicone-Property-Step": "synthesize",
    },
)
```

Filter by `Workflow=research-agent` to see the total cost of the research agent pipeline.

---

## Rate limit tracking

Helicone tracks rate limit headers from LLM providers and provides visibility into throttling events.

### What Helicone captures

| Header | Meaning |
|--------|---------|
| `x-ratelimit-limit-requests` | Max requests per minute |
| `x-ratelimit-remaining-requests` | Requests remaining in window |
| `x-ratelimit-limit-tokens` | Max tokens per minute |
| `x-ratelimit-remaining-tokens` | Tokens remaining in window |
| `retry-after` | Seconds until rate limit resets |

### Automatic retries

Configure Helicone to automatically retry rate-limited requests:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "Helicone-Retry-Enabled": "true",
        "Helicone-Retry-Num": "3",
        "Helicone-Retry-Factor": "2",  # Exponential backoff factor
    },
)
```

| Header | Default | Description |
|--------|---------|-------------|
| `Helicone-Retry-Enabled` | `false` | Enable automatic retries |
| `Helicone-Retry-Num` | `3` | Maximum retry attempts |
| `Helicone-Retry-Factor` | `2` | Exponential backoff multiplier |

---

## User analytics

With custom properties, Helicone provides per-user analytics out of the box.

```python
# Tag every request with the user ID
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": query}],
    extra_headers={
        "Helicone-Property-UserId": current_user.id,
    },
)
```

### What you get per user

- **Request count**: How many API calls this user generated
- **Total cost**: How much this user costs you
- **Average latency**: Response time experience
- **Error rate**: How often their requests fail
- **Model distribution**: Which models they use most

This data is critical for **usage-based billing**, identifying power users, and detecting abuse.

---

## Caching for cost reduction

Helicone can cache identical requests, reducing costs and latency for repeated queries.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_headers={
        "Helicone-Cache-Enabled": "true",
    },
)
```

The first request hits the LLM; subsequent identical requests return the cached response instantly at zero cost.

### Cache behavior

| Scenario | Result |
|----------|--------|
| Same model + same messages | Cache **hit** (instant, free) |
| Same messages + different model | Cache **miss** (new request) |
| Same messages + different temperature | Cache **miss** (new request) |
| Cache TTL expired | Cache **miss** (new request) |

> **Note:** Caching works best for deterministic queries (temperature=0). For creative tasks, it's less useful since you typically want varied responses.

---

## Model fallbacks

Helicone can automatically fall back to a different model if the primary model fails:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze this data"}],
    extra_headers={
        "Helicone-Fallback-Model": "gpt-4o-mini",
        "Helicone-Fallback-OnCodes": "429,500,503",
    },
)
```

If `gpt-4o` returns a 429 (rate limited), 500, or 503, Helicone automatically retries with `gpt-4o-mini`.

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Tag every request with custom properties | Enables filtering, user tracking, and cost attribution |
| Use caching for deterministic queries | Reduces costs and latency significantly |
| Set up budget alerts early | Prevent surprise bills from runaway agents |
| Track costs per workflow/feature | Understand which agent pipelines cost the most |
| Enable retries for rate-limited calls | Automatic recovery without custom retry logic |
| Use model fallbacks for reliability | Graceful degradation when primary model is overloaded |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Not tagging requests with user IDs | Add `Helicone-Property-UserId` header to every request |
| Enabling cache for creative/varied responses | Only cache deterministic queries (temperature=0 or factual lookups) |
| Ignoring rate limit data | Monitor the dashboard for throttling patterns before they impact users |
| Using Helicone only for logging | Leverage caching, retries, and fallbacks for production reliability |
| Not setting budget alerts | Configure alerts before your first production deployment |
| Hardcoding the Helicone API key | Use environment variables: `process.env.HELICONE_API_KEY` |

---

## Hands-on exercise

### Your task

Set up Helicone as a gateway for a multi-step agent and track costs per workflow step.

### Requirements

1. Create a Helicone account and get an API key
2. Configure the OpenAI client to use `https://ai-gateway.helicone.ai`
3. Build a 2-step agent (research → summarize) with custom property headers
4. Tag each step with `Helicone-Property-Step` and `Helicone-Property-Workflow`
5. View the requests and costs in the Helicone dashboard

### Expected result

The Helicone dashboard shows both requests grouped under the same workflow, with per-step costs, token counts, and latency.

<details>
<summary>💡 Hints (click to expand)</summary>

- Set `base_url="https://ai-gateway.helicone.ai"` in the OpenAI client
- Use `extra_headers={}` to pass Helicone-specific headers
- Property headers follow the format `Helicone-Property-{Name}: value`
- Check the Requests tab in the dashboard after running

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ai-gateway.helicone.ai",
    api_key="your-helicone-api-key",
)

# Step 1: Research
research = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": "Research the benefits of AI observability"},
    ],
    extra_headers={
        "Helicone-Property-Workflow": "research-pipeline",
        "Helicone-Property-Step": "research",
        "Helicone-Property-UserId": "student_1",
    },
)
research_text = research.choices[0].message.content

# Step 2: Summarize
summary = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Summarize in 2 sentences."},
        {"role": "user", "content": research_text},
    ],
    extra_headers={
        "Helicone-Property-Workflow": "research-pipeline",
        "Helicone-Property-Step": "summarize",
        "Helicone-Property-UserId": "student_1",
    },
)
print(summary.choices[0].message.content)
```

</details>

### Bonus challenges

- [ ] Enable caching for the summarization step and call it twice to see a cache hit
- [ ] Set up automatic retries with exponential backoff via headers
- [ ] Configure a model fallback from `gpt-4o` to `gpt-4o-mini`

---

## Summary

✅ **AI gateway approach** provides zero-code observability by proxying API calls  
✅ **Custom property headers** enable filtering by user, workflow, feature, and environment  
✅ **Cost monitoring** tracks per-request, per-user, and per-model spend  
✅ **Automatic retries and fallbacks** improve reliability without custom retry logic  
✅ **Caching** reduces costs and latency for repeated deterministic queries  

**Previous:** [Langfuse Open-Source](./03-langfuse-open-source.md)  
**Next:** [Custom Observability Setup](./05-custom-observability-setup.md)

---

## Further Reading

- [Helicone Documentation](https://docs.helicone.ai/) — Full platform docs
- [Helicone Quick Start](https://docs.helicone.ai/getting-started/quick-start) — 2-minute setup
- [Helicone GitHub](https://github.com/Helicone/helicone) — Open-source repository
- [Helicone Platform Overview](https://docs.helicone.ai/getting-started/platform-overview) — Feature walkthrough

<!--
Sources Consulted:
- Helicone quickstart: https://docs.helicone.ai/getting-started/quick-start
- Helicone platform overview: https://docs.helicone.ai/getting-started/platform-overview
-->
