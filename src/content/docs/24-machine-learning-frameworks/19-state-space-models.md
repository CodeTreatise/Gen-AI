---
title: "24.19 State Space Models"
---

# 24.19 State Space Models

## Introduction to SSMs
- Alternative to Transformers
- Linear complexity with sequence length
- Selective state spaces
- Long-range dependency handling
- Hardware-efficient design

## Mamba Architecture
- Selective State Space Model (S4 evolution)
- Selection mechanism
- Hardware-aware algorithm
- Linear scaling vs quadratic attention
- Memory efficiency

## Mamba-2
- Improved performance
- Better hardware utilization
- Simplified implementation
- Tensor parallelism support
- Integration with attention layers

## Key Concepts
- State space equations
- Discretization methods
- Parallel scan algorithm
- Selective mechanism
- Input-dependent dynamics

## SSM vs Transformer Trade-offs
- SSM advantages:
  - Linear time complexity
  - Efficient long sequences
  - Lower memory for inference
- Transformer advantages:
  - Better in-context learning
  - More mature tooling
  - Easier parallelization training
- Hybrid approaches emerging

## Hybrid Architectures (2025)
- Combining SSM + Attention
- Jamba (AI21 Labs)
- Mamba-Attention hybrids
- Best of both worlds approach
- SGLang hybrid model support

## Training SSMs
- Custom CUDA kernels
- Mamba-ssm library
- Integration with transformers
- FSDP compatibility
- Mixed precision training

## Inference Considerations
- Constant memory per token
- No KV cache needed
- Streaming generation
- vLLM Mamba support
- SGLang Mamba support

## Current Limitations
- Smaller model ecosystem
- Less community tooling
- Training instability at scale
- Limited fine-tuning frameworks
- Fewer pretrained models

## Notable SSM Models
- Mamba (Albert Gu, Tri Dao)
- Jamba (AI21 Labs)
- RWKV (hybrid approach)
- Griffin (Google DeepMind)
- Falcon-Mamba

## Research Directions
- Longer context handling
- Multimodal SSMs
- Efficient attention alternatives
- Hardware co-design
- Scaling laws for SSMs

## When to Consider SSMs
- Very long sequence tasks
- Memory-constrained deployment
- Streaming applications
- Real-time processing
- Edge devices
