---
title: "17.5 Microsoft Foundry (formerly Azure OpenAI Service)"
---

# 17.5 Microsoft Foundry (formerly Azure OpenAI Service)

## Introduction

Microsoft Foundry (rebranded from Azure AI Foundry in 2025) is the unified Azure platform-as-a-service for enterprise AI operations, model building, and application development. It combines production-grade infrastructure with developer-friendly interfaces for building AI applications.

## What is Microsoft Foundry

- Unified Azure AI platform (rebranded 2025)
- OpenAI models on Azure with enterprise security
- Multi-provider models (Claude, DeepSeek, xAI via Foundry Models)
- Enterprise security and compliance certifications
- Seamless Azure service integration
- Regional availability and data residency

## Platform Evolution

- 2023: Azure OpenAI Service launched
- 2024: Azure AI Foundry introduced (unified platform)
- 2025: Rebranded to Microsoft Foundry
  - New Foundry Projects
  - Foundry SDK and API
  - Foundry Agent Service (GA)
  - 11,000+ models in catalog

## Available Models (2025)

- OpenAI models
  - GPT-4o and GPT-5 models
  - o1 reasoning models
  - DALL-E 3 for images
  - Whisper for speech
- Third-party via Foundry Models
  - Anthropic Claude 3.5/4
  - DeepSeek models
  - xAI Grok models
- Embedding models
  - text-embedding-3 variants
  - Multilingual embeddings

## Foundry Projects

- New Foundry project type (recommended)
  - Full Foundry SDK and API support
  - Foundry Agent Service (GA)
  - Project-level isolation
  - Built-in storage and indexing
- Hub-based projects (legacy)
  - Azure ML integration
  - Prompt Flow support
  - Limited Foundry SDK support
- Migration path from hub-based to Foundry projects

## Foundry SDK and API

- Consistent API contract across model providers
- Python SDK (production ready)
- C# SDK (production ready)
- JavaScript/TypeScript SDK (preview)
- Java SDK (preview)
- VS Code Extension for development

## Using the API

- REST API access
- Chat completions endpoint
- Embeddings endpoint
- Function calling (tools)
- Structured outputs (JSON schema)
- Streaming responses
- Vision capabilities

## Enterprise Features

- Virtual network integration
- Private endpoints
- Managed identity authentication
- Entra ID (Azure AD) authentication
- Content filtering (built-in)
- Custom content policies
- Regional deployment options

## Model Router

- Automatic model selection
- Quality optimization strategy
- Cost optimization strategy
- Latency optimization strategy
- Configurable allowed models
- Request routing based on criteria

## Deployment Patterns

- Standard deployment (pay-per-token)
- Provisioned Throughput Units (PTU) for guaranteed capacity
- Global deployment with load balancing
- Multi-region for latency optimization

## Best Practices

- Use managed identity over API keys
- Implement retry logic with exponential backoff
- Monitor token usage for cost control
- Enable content filtering for safety
- Use structured outputs for reliability
- Implement proper error handling

## Hands-on Exercises

1. Build a basic chat application with Foundry SDK
2. Compare responses across GPT-4o, Claude, and DeepSeek
3. Implement function calling for database queries
4. Set up secure deployment with private endpoints

## Summary

Microsoft Foundry provides:
- Unified SDK across model providers
- 11,000+ models in catalog
- Enterprise-grade security
- Production-ready agent service
- Automatic model routing
- Comprehensive deployment options

## Next Steps

- [Microsoft Foundry Agent Service](06-foundry-agent-service.md)
- [Microsoft Foundry AI Services](07-foundry-ai-services.md)
- [Security & Compliance](13-security-compliance.md)
