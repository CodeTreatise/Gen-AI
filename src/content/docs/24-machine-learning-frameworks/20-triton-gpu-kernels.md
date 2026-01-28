---
title: "24.20 Triton & GPU Kernels"
---

# 24.20 Triton & GPU Kernels

## What is Triton
- OpenAI's GPU programming language
- Python-like syntax
- Compiles to efficient GPU code
- Used by PyTorch TorchInductor
- Lower barrier than CUDA

## Why Triton Matters
- Powers torch.compile performance
- FlexAttention implementation
- Custom kernel development
- Research to production path
- Open source community

## Triton vs CUDA
- Triton: Higher abstraction, easier
- CUDA: Maximum control, complex
- Triton: Auto-tuning built-in
- CUDA: Manual optimization needed
- Triton for most use cases

## Core Concepts
- Blocked programming model
- Programs operate on tiles
- Automatic memory coalescing
- Implicit parallelization
- JIT compilation

## Writing Triton Kernels
- @triton.jit decorator
- Block and grid dimensions
- tl.load and tl.store
- tl.program_id for block index
- Atomic operations

## FlexAttention
- torch.nn.attention.flex_attention
- Custom attention patterns
- Block sparsity masks
- Score modification functions
- Compiled with Triton backend
- Examples:
  - Sliding window attention
  - Document attention masks
  - Relative position biases

## Memory Management
- Shared memory usage
- L2 cache optimization
- Memory access patterns
- Avoiding bank conflicts
- Tiling strategies

## Performance Optimization
- Auto-tuning with @triton.autotune
- Kernel fusion opportunities
- Occupancy considerations
- Register pressure
- Profiling with NSight

## Common Kernel Patterns
- Matrix multiplication (GEMM)
- Softmax implementations
- Layer normalization
- Fused attention
- Element-wise operations

## Integration with PyTorch
- TorchInductor backend
- Custom autograd functions
- torch.compile integration
- Debugging compiled kernels
- Performance comparison

## Debugging Triton
- triton.testing module
- Comparison with reference
- Print debugging in kernels
- Checking for correctness
- Common pitfalls

## Use Cases
- Custom attention mechanisms
- Novel activation functions
- Specialized layer implementations
- Research experiments
- Production optimizations

## Learning Path
- Start with element-wise kernels
- Progress to reductions
- Learn attention patterns
- Study FlexAttention examples
- Contribute to ecosystem
