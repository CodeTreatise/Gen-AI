---
title: "24.17 LLM Inference & Serving"
---

# 24.17 LLM Inference & Serving

## Overview
- Challenges of LLM inference
- Latency vs throughput trade-offs
- Memory management for large models
- Batching strategies
- Choosing inference framework

## vLLM (2025)
- 68K+ GitHub stars
- PagedAttention for memory efficiency
- Key features:
  - Disaggregated prefill & decode
  - Chunked prefill
  - Speculative decoding
  - Multi-LoRA serving
  - Prefix caching
  - FP8 KV cache
- Performance: 24x throughput vs HF
- OpenAI-compatible API
- Tensor parallelism support
- MoE model support
- Installation and deployment

## SGLang (2025)
- Stanford/Berkeley project
- RadixAttention for prefix sharing
- Key features:
  - Hybrid SSM + Attention model support
  - Elastic memory management
  - Fast structured output (JSON)
  - Grammar-constrained decoding
  - Speculative execution
- Programming language for LLM programs
- When to choose SGLang over vLLM

## PagedAttention Deep Dive
- Virtual memory for KV cache
- Block-based memory management
- Memory fragmentation elimination
- Sharing KV cache across requests
- Copy-on-write for beam search

## Continuous Batching
- Dynamic request batching
- Maximizing GPU utilization
- Handling variable-length sequences
- Preemption strategies
- Iteration-level scheduling

## Speculative Decoding
- Draft model + target model
- Parallel token verification
- Speed improvements (2-3x typical)
- Draft model selection
- Rejection sampling

## Prefix Caching
- Caching KV cache for common prefixes
- System prompt optimization
- RadixAttention tree structure
- Memory vs latency trade-off

## Multi-LoRA Serving
- Serving multiple LoRA adapters
- Dynamic adapter loading
- vLLM multi-LoRA support
- Efficient adapter switching
- Use cases (personalization, multi-tenant)

## Text Generation Inference (TGI)
- Hugging Face inference solution
- Production-ready serving
- Flash Attention integration
- Tensor parallelism
- Quantization support
- Docker deployment

## Deployment Patterns
- Single model serving
- Multi-model deployment
- A/B testing infrastructure
- Canary deployments
- Load balancing strategies

## API Design
- OpenAI-compatible endpoints
- Streaming responses
- Token usage tracking
- Rate limiting
- Error handling

## Monitoring & Observability
- Request latency tracking
- Token throughput metrics
- Queue depth monitoring
- GPU utilization
- Memory usage alerts
- Prometheus/Grafana integration

## Cost Optimization
- Batch size tuning
- Caching strategies
- Spot instance usage
- Auto-scaling configuration
- Right-sizing GPU selection
