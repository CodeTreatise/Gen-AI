---
title: "24.2 PyTorch 2.x Fundamentals"
---

# 24.2 PyTorch 2.x Fundamentals

## What is PyTorch 2.x
- Deep learning framework evolution (2.5-2.9 era)
- Dynamic computation graphs with TorchDynamo
- Python-first design with JIT compilation
- Research to production pipeline
- Community and ecosystem (68K+ GitHub stars)
- PyTorch Foundation governance

## Installation and Setup (2025-2026)
- pip install torch (2.9+ recommended)
- CUDA 12.x/13.x support installation
- ROCm support for AMD GPUs
- Intel XPU support (Arc, Data Center GPUs)
- Verifying GPU availability
- Version compatibility matrix
- CPU vs GPU vs XPU tensors

## Tensors
- Creating tensors (torch.tensor, torch.zeros, torch.ones)
- Tensor data types (float32, bfloat16, float16, int64, etc.)
- Tensor shapes and dimensions
- Tensor operations
- NumPy interoperability (numpy(), from_numpy())
- Nested tensors for variable-length sequences

## Tensor Operations
- Element-wise operations
- Matrix multiplication (@ operator, torch.mm, torch.matmul)
- Broadcasting rules
- Reshaping (view, reshape, contiguous)
- Indexing and slicing
- Advanced indexing (gather, scatter)

## Device Management
- CPU, CUDA, and XPU devices
- Moving tensors between devices (.to(), .cuda(), .xpu())
- torch.device() usage patterns
- Multi-GPU basics (device indexing)
- Memory management (empty_cache, memory_stats)
- Symmetric Memory for multi-GPU (PyTorch 2.9+)

## Autograd
- Automatic differentiation
- requires_grad flag
- Backward pass (backward())
- Gradient accumulation patterns
- Detaching tensors (detach())
- Gradient clearing (zero_grad())
- Gradient checkpointing for memory efficiency

## PyTorch 2.5-2.9 New Features
- FlexAttention for custom attention patterns
- Symmetric Memory for efficient multi-GPU communication
- Expanded wheel variants (ROCm 6.3, XPU, CUDA 13)
- TorchInductor improvements
- Enhanced torch.compile performance
- Native support for Intel GPUs
- Improved memory efficiency

## NumPy Interoperability
- From NumPy to Tensor
- From Tensor to NumPy
- Shared memory considerations
- Device considerations
- Common patterns
