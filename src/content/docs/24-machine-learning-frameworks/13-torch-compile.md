---
title: "24.13 torch.compile & TorchDynamo"
---

# 24.13 torch.compile & TorchDynamo

## What is torch.compile
- PyTorch 2.0+ native model compilation
- JIT compilation for performance
- Drop-in optimization for existing models
- One-line change for significant speedups
- Foundation of PyTorch 2.x performance story

## TorchDynamo
- Python bytecode analysis
- Graph capture from Python code
- Handling dynamic control flow
- Guard system for correctness
- FX graph extraction

## Compilation Modes
- "default": Balanced speed and compilation time
- "reduce-overhead": CUDA graph optimization
- "max-autotune": Extensive kernel search (slower compile, faster runtime)
- Choosing mode based on use case

## Backends
- TorchInductor (default, recommended)
- OpenXLA for TPU support
- ONNX Runtime backend
- Custom backend creation
- Backend selection strategies

## TorchInductor
- Default torch.compile backend
- Generates Triton kernels for GPU
- C++/OpenMP for CPU
- Operator fusion
- Memory planning
- Kernel autotuning

## Practical Usage Patterns
- torch.compile(model) basic usage
- Compiling specific functions
- Dynamic shapes handling
- Avoiding recompilation
- Warm-up considerations

## FlexAttention Integration
- torch.nn.attention.flex_attention
- Custom attention patterns
- Block sparsity support
- Score modification functions
- Memory efficiency

## Performance Expectations
- 1.3x - 2x speedup typical for transformers
- Larger gains for compute-bound models
- First call compilation overhead
- Caching compiled graphs
- Production deployment considerations

## Debugging torch.compile
- TORCH_LOGS environment variable
- Graph breaks identification
- torch._dynamo.explain()
- Common graph break causes
- Workarounds for unsupported operations

## Limitations and Gotchas
- Not all Python code compiles
- Dynamic control flow challenges
- In-place operations considerations
- Third-party library compatibility
- Memory overhead during compilation
