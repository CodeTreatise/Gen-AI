---
title: "17.8 Vertex AI with Gemini"
---

# 17.8 Vertex AI with Gemini

## Introduction

Vertex AI is Google Cloud's unified AI platform, now deeply integrated with the Gemini model family. Gemini 2.5 Pro and Flash represent state-of-the-art multimodal AI with native tool use, extended context windows, and built-in thinking capabilities.

## Vertex AI Platform Overview

- Unified AI development platform
- Native Gemini integration
- 200+ models in Model Garden
- MLOps capabilities
- Enterprise security and governance

## Gemini Model Family (2025)

### Gemini 2.5 Pro

- Most advanced Gemini model
- 1M token context window (2M in preview)
- Native multimodal (text, image, audio, video)
- Built-in thinking/reasoning
- Improved code generation
- Complex task handling

### Gemini 2.5 Flash

- Speed-optimized variant
- Cost-effective for high-volume
- Thinking mode toggle
- 1M token context
- Multimodal support

### Gemini 2.0 Series

- Gemini 2.0 Flash
  - Low latency, high throughput
  - Agentic capabilities
  - Native tool use
- Gemini 2.0 Flash-Lite
  - Most cost-effective
  - Text-focused optimization

### Gemini Pro Vision

- Image understanding
- Video analysis
- Document processing
- Chart/diagram interpretation

## Model Garden

- 200+ models available
- Open source models
  - Llama 3.x
  - Gemma 2
  - Mistral models
  - CodeGemma
- Specialized models
  - Embedding models
  - Code generation
  - Image generation
- One-click deployment
- Fine-tuning support

## Vertex AI SDK

- Unified Python SDK
- Generative AI API
- Core capabilities
  - Text generation
  - Multimodal input
  - Function calling
  - Grounding
  - Safety settings

## Function Calling

- Native tool use
  - Declare functions
  - Model selects function
  - Execute and return
- Automatic mode
  - Model decides when to call
  - Multi-turn with tools
- Manual mode
  - Control execution
  - Custom logic
- Parallel calling
  - Multiple functions simultaneously
  - Efficiency optimization

## Grounding

- Google Search grounding
  - Real-time information
  - Citation support
  - Fact verification
- Vertex AI Search grounding
  - Enterprise data
  - Private knowledge
  - Structured data
- Dynamic retrieval
  - Automatic threshold
  - Cost optimization

## Context Caching

- Cost optimization feature
- Cache repeated context
- Use cases
  - System instructions
  - Few-shot examples
  - Large documents
  - Code repositories
- Cache duration control
- Significant cost savings

## Code Execution

- Gemini native code execution
- Supported languages
  - Python
  - JavaScript (preview)
- Use cases
  - Data analysis
  - Calculations
  - Visualization
- Sandboxed environment
- Result interpretation

## Safety Settings

- Harm categories
  - Hate speech
  - Dangerous content
  - Sexually explicit
  - Harassment
- Threshold configuration
  - Block none
  - Block few
  - Block some
  - Block most
- Safety feedback
  - Block reasons
  - Safety ratings

## Multimodal Capabilities

- Image input
  - Analysis
  - OCR
  - Object detection
- Video input
  - Scene understanding
  - Transcription
  - Event detection
- Audio input
  - Transcription
  - Speaker identification
  - Sound analysis
- PDF processing
  - Document understanding
  - Table extraction
  - Form processing

## Batch Prediction

- Large-scale processing
- Cost optimization (50% discount)
- Async processing
- BigQuery integration
- Cloud Storage I/O

## Fine-Tuning

- Supervised fine-tuning
  - Custom datasets
  - Domain adaptation
  - Task specialization
- RLHF (preview)
  - Human feedback
  - Preference learning
- Distillation
  - Model compression
  - Efficiency optimization

## Deployment Options

- Managed API
  - Serverless
  - Pay per token
  - Auto-scaling
- Dedicated endpoints
  - Reserved capacity
  - Predictable latency
  - Volume discounts
- Model Garden deployment
  - Custom containers
  - Open source models

## Integration Patterns

- Cloud Functions
  - Serverless inference
  - Event triggers
- Cloud Run
  - Container-based
  - Custom logic
- GKE
  - Kubernetes orchestration
  - Complex pipelines
- BigQuery ML
  - SQL-based ML
  - Direct integration

## Best Practices

- Model selection
  - Pro for complex tasks
  - Flash for speed/cost
  - Flash-Lite for volume
- Prompt engineering
  - Clear instructions
  - Structured output
  - Few-shot examples
- Cost optimization
  - Context caching
  - Batch prediction
  - Right model selection

## Hands-on Exercises

1. Build multimodal application with Gemini 2.5 Pro
2. Implement function calling for data retrieval
3. Set up context caching for cost optimization
4. Create grounded chatbot with Google Search

## Comparison with Other Platforms

| Feature | Vertex AI | Bedrock | Foundry |
|---------|-----------|---------|---------|
| Primary Model | Gemini 2.5 | Claude, Nova | GPT-4o, Claude |
| Context Window | 1M-2M | 200K | 128K-200K |
| Grounding | Google Search | S3, KB | Azure AI Search |
| Code Execution | Native | Lambda | Functions |

## Summary

| Component | Description |
|-----------|-------------|
| Gemini 2.5 | State-of-the-art multimodal |
| Model Garden | 200+ models |
| Grounding | Search + enterprise data |
| Context Cache | Cost optimization |
| Safety | Configurable harm filters |

## Next Steps

- [Vertex AI Agent Builder](09-vertex-agent-builder.md)
- [Multi-Cloud API Management](10-api-management.md)
- [MCP Integration](11-mcp-integration.md)
