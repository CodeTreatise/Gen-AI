---
title: "Unit 4: AI API Integration"
---

# Unit 4: AI API Integration

## Overview & Importance

API integration is where theory meets practice. This unit teaches how to connect web applications to AI services through their APIs. You'll learn to make requests, handle responses, manage streaming data, and build robust integrations that handle errors gracefully.

Most AI capabilities are accessed through APIs. Understanding API integration enables you to:
- Add AI features to any web application
- Work with multiple AI providers
- Handle the unique challenges of AI APIs (streaming, long responses)
- Build production-ready AI integrations

## Prerequisites

- JavaScript proficiency (Unit 1)
- Understanding of HTTP and REST APIs
- AI fundamentals knowledge (Unit 2)
- Familiarity with async/await

## Learning Objectives

By the end of this unit, you will be able to:
- Set up authentication with major AI API providers
- Make basic completion requests to LLM APIs
- Handle both synchronous and streaming responses
- Parse and process AI-generated content
- Implement proper error handling and retry logic
- Manage API keys securely
- Monitor and control API usage and costs
- Use OpenAI Responses API for modern integrations
- Implement prompt caching for cost optimization
- Integrate built-in tools and MCP servers
- Build voice-enabled applications with Realtime API
- Generate structured outputs with JSON schemas
- Work with reasoning models and their parameters

## Real-world Applications

- Chat applications with real-time AI responses
- Content generation tools (blogs, emails, documentation)
- Code assistants integrated into development tools
- Customer support automation
- Document summarization services
- Translation services
- AI agents with built-in tools (web search, code interpreter)
- Voice assistants using Realtime API
- Data extraction pipelines with structured outputs
- Multi-modal applications (text + audio + vision)
- MCP-connected enterprise applications
- Cost-optimized high-volume applications with caching

## Market Demand & Relevance

- API integration is the most practical AI skill for web developers
- 80%+ of AI features in products use external APIs
- Companies prefer API integration over building custom models
- Skills directly transferable across all AI providers
- Entry point for most "AI Engineer" positions
- Consulting rates for AI integration: $100-300/hour

## Resources

### Official API Documentation
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — Complete API documentation
- [OpenAI Responses API Guide](https://platform.openai.com/docs/guides/responses-vs-chat-completions) — Migration and usage guide
- [Anthropic Claude API](https://docs.anthropic.com/en/api) — Claude API documentation
- [Google Gemini API](https://ai.google.dev/gemini-api/docs) — Gemini API documentation
- [Cohere API](https://docs.cohere.com/) — Cohere API reference
- [Mistral API](https://docs.mistral.ai/) — Mistral AI documentation

### Streaming & Real-time
- [OpenAI Streaming Guide](https://platform.openai.com/docs/guides/streaming-responses) — SSE streaming implementation
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Voice and real-time communication
- [MDN Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — SSE protocol reference
- [WebRTC Documentation](https://webrtc.org/getting-started/overview) — Browser real-time communication

### Tools & Function Calling
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — Tool use guide
- [OpenAI Built-in Tools](https://platform.openai.com/docs/guides/tools) — Web search, code interpreter, etc.
- [OpenAI MCP Integration](https://platform.openai.com/docs/guides/tools-connectors-mcp) — Model Context Protocol
- [MCP Specification](https://modelcontextprotocol.io/) — Official MCP documentation
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude tool calling

### Structured Outputs
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — JSON schema enforcement
- [Gemini Structured Output](https://ai.google.dev/gemini-api/docs/structured-output) — Google's implementation
- [Pydantic Documentation](https://docs.pydantic.dev/) — Python schema validation
- [Zod Documentation](https://zod.dev/) — TypeScript schema validation

### Cost & Optimization
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — Caching strategies
- [OpenAI Pricing](https://openai.com/api/pricing/) — Token pricing
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — Batch processing discounts
- [tiktoken Library](https://github.com/openai/tiktoken) — Token counting

### SDKs & Libraries
- [OpenAI Python SDK](https://github.com/openai/openai-python) — Official Python client
- [OpenAI Node.js SDK](https://github.com/openai/openai-node) — Official JavaScript client
- [OpenAI Agents SDK (JS)](https://openai.github.io/openai-agents-js/) — Agent building toolkit
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude Python client
- [Google GenAI SDK](https://github.com/google/generative-ai-python) — Gemini Python client

### Local AI & Self-Hosting
- [Ollama](https://ollama.ai/) — Local model running
- [LM Studio](https://lmstudio.ai/) — GUI for local models
- [vLLM](https://docs.vllm.ai/) — High-throughput serving
- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference) — Hugging Face TGI
- [LocalAI](https://localai.io/) — OpenAI-compatible local server
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Efficient CPU/GPU inference

### Error Handling & Resilience
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits) — Rate limit documentation
- [Retry-After Header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After) — HTTP retry handling
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) — Resilience pattern

### Tutorials & Cookbooks
- [OpenAI Cookbook](https://cookbook.openai.com/) — Official examples and tutorials
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) — Claude examples
- [Google Gemini Cookbook](https://github.com/google-gemini/cookbook) — Gemini examples
- [LangChain Documentation](https://python.langchain.com/docs/) — Multi-provider framework
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) — Data framework for LLMs

### Security & Best Practices
- [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices) — Security guidelines
- [OWASP API Security](https://owasp.org/API-Security/) — API security standards
- [dotenv Documentation](https://github.com/motdotla/dotenv) — Environment variable management
