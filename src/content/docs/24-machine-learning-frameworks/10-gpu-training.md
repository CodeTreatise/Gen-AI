---
title: "24.10 GPU & Distributed Training"
---

# 24.10 GPU & Distributed Training

## GPU Basics for ML
- Why GPUs for deep learning (parallelism)
- CUDA and cuDNN fundamentals
- ROCm for AMD GPUs
- Intel XPU support (oneAPI)
- GPU memory management
- Batch size and memory relationship
- GPU utilization monitoring (nvidia-smi, rocm-smi)

## Mixed Precision Training
- FP16 vs BF16 vs FP32
- Automatic mixed precision (AMP) with torch.autocast
- Memory savings (typically 2x)
- Speed improvements (1.5-3x)
- torch.cuda.amp usage patterns
- BF16 for modern GPUs (Ampere+)
- Loss scaling for FP16 stability

## Multi-GPU Training Strategies
- DataParallel (simple, single-node)
- DistributedDataParallel (DDP, recommended)
- Fully Sharded Data Parallel (FSDP1 & FSDP2)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP) for long sequences
- Expert Parallelism (EP) for MoE models
- Choosing strategy based on model size

## FSDP2 (PyTorch 2.x)
- Sharding strategies (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
- CPU offloading for memory efficiency
- Activation checkpointing integration
- Mixed precision with FSDP
- FSDP vs DeepSpeed comparison
- State dict handling

## DeepSpeed Integration
- ZeRO optimization stages (1, 2, 3)
- ZeRO-Infinity for trillion-parameter models
- ZeRO++ communication optimization
- DeepSpeed configuration files
- Integration with Hugging Face Trainer
- Offloading to CPU/NVMe

## ND Parallelism (2025)
- Combining multiple parallelism strategies
- Context Parallel + Tensor Parallel + FSDP
- Hugging Face Accelerate ND Parallelism
- Single-node and multi-node composition
- Optimal strategy selection

## Gradient Accumulation
- Simulating larger batches
- Memory constraints workaround
- Implementation pattern
- Effective batch size calculation
- When to use (limited GPU memory)

## Hugging Face Accelerate
- Simplified distributed training
- Device agnostic code
- Mixed precision integration
- Gradient accumulation handling
- Launch configurations (accelerate launch)
- FSDP and DeepSpeed backends

## Cloud GPU Options (2025-2026)
- Google Colab (free tier, Pro, Pro+)
- AWS (EC2 P5, SageMaker)
- Azure ML compute (ND H100 v5)
- GCP (Vertex AI, A3 instances)
- Lambda Labs, RunPod, Vast.ai
- Modal, Anyscale for serverless GPU
