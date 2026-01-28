---
title: "24.15 Fine-Tuning Frameworks"
---

# 24.15 Fine-Tuning Frameworks

## Overview
- Landscape of LLM fine-tuning tools
- Choosing the right framework
- Trade-offs: ease of use vs flexibility
- Community and support considerations

## Unsloth (2025)
- 30x faster fine-tuning claims
- Zero accuracy loss optimization
- Memory efficiency (16GB VRAM for 7B models)
- Key features:
  - 500K context length support
  - FP8 GRPO training
  - Dynamic 4-bit quantization
  - Automatic RoPE scaling
- Supported models: Llama, Mistral, Qwen, Phi, Gemma
- Unsloth Studio (paid) vs Open Source
- Installation and quick start
- Limitations (model support, custom architectures)

## Axolotl
- YAML-based configuration
- Production-grade fine-tuning
- Version 0.13+ features (2025):
  - Quantization-Aware Training (QAT)
  - NVFP4 precision support
  - ND Parallelism (multi-strategy)
  - VLM training (LLaVA, Pixtral)
  - Modal cloud integration
- Supported training methods:
  - Full fine-tuning
  - LoRA/QLoRA
  - GPTQ, GGUF quantized training
  - DPO, ORPO, RLHF
- Configuration examples
- Multi-node training
- When to choose Axolotl

## LitGPT (Lightning AI)
- Simple, hackable LLM training
- No boilerplate code
- Supported models (Llama, Mistral, Falcon, etc.)
- Fine-tuning and pretraining support
- LoRA and adapter methods
- Evaluation integration
- Deployment options

## LLaMA-Factory
- Web UI for fine-tuning
- 100+ model support
- No-code fine-tuning option
- Supported methods:
  - Full, LoRA, QLoRA, RLHF, DPO
- Evaluation and export tools
- When to use LLaMA-Factory

## Comparison Matrix
- Speed: Unsloth > others for supported models
- Flexibility: Axolotl > LitGPT > Unsloth
- Ease of use: LLaMA-Factory > LitGPT > Axolotl
- Model support: Axolotl ≈ LLaMA-Factory > Unsloth
- RLHF/DPO: Axolotl = LLaMA-Factory > LitGPT

## Choosing a Framework
- For speed on supported models → Unsloth
- For production with complex configs → Axolotl
- For quick prototyping → LitGPT
- For non-programmers → LLaMA-Factory
- For research flexibility → Custom TRL setup

## Common Configuration Patterns
- Dataset preparation (ShareGPT, Alpaca format)
- Model selection and loading
- Training hyperparameters
- Logging and checkpointing
- Evaluation during training

## Cloud Deployment
- Modal integration (Axolotl)
- RunPod templates
- AWS SageMaker jobs
- Google Colab workflows
