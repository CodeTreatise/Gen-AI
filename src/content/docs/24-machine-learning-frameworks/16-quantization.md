---
title: "24.16 Quantization Techniques"
---

# 24.16 Quantization Techniques

## What is Quantization
- Reducing numerical precision of weights/activations
- Memory reduction (FP32 → INT8 = 4x smaller)
- Faster inference on compatible hardware
- Trade-off between size and accuracy
- Critical for LLM deployment

## Precision Formats
- FP32 (full precision, 4 bytes)
- BF16 (brain floating point, 2 bytes)
- FP16 (half precision, 2 bytes)
- FP8 (E4M3, E5M2 variants, 1 byte)
- INT8 (8-bit integer, 1 byte)
- INT4/NF4 (4-bit, 0.5 bytes)
- NVFP4 (NVIDIA 4-bit format)

## FP8 Training & Inference (2025)
- Native FP8 in PyTorch 2.x
- E4M3 for forward pass
- E5M2 for gradients
- Hardware support (H100, H200, B100)
- torch.float8_e4m3fn, torch.float8_e5m2
- FP8 with FSDP
- Unsloth FP8 GRPO support

## Post-Training Quantization (PTQ)
- Quantize after training
- No additional training needed
- Calibration with representative data
- Dynamic vs static quantization
- Per-tensor vs per-channel
- Accuracy considerations

## Quantization-Aware Training (QAT)
- Simulate quantization during training
- Better accuracy than PTQ
- Fake quantization nodes
- Axolotl QAT support (2025)
- When QAT is worth the effort

## GGUF Format
- llama.cpp ecosystem
- CPU inference optimization
- Multiple quantization levels:
  - Q8_0 (8-bit)
  - Q5_K_M (5-bit, medium quality)
  - Q4_K_M (4-bit, medium)
  - Q3_K_S (3-bit, small)
  - Q2_K (2-bit, aggressive)
- Memory vs quality trade-offs
- Converting to GGUF

## GPTQ Quantization
- 4-bit weight quantization
- Layer-by-layer quantization
- Optimal Brain Quantization based
- AutoGPTQ library
- Hugging Face integration
- When to use GPTQ

## AWQ (Activation-aware Weight Quantization)
- 4-bit quantization
- Protects salient weights
- Better quality than GPTQ often
- AutoAWQ library
- Faster than GPTQ for inference

## bitsandbytes Library
- 8-bit optimizers (Adam, AdamW)
- 4-bit NF4 quantization
- Integration with Hugging Face
- load_in_8bit, load_in_4bit
- bnb_4bit_compute_dtype
- Paged optimizers

## ONNX Quantization
- ONNX Runtime quantization tools
- Static and dynamic modes
- INT8 optimization
- Deployment to edge devices

## Choosing Quantization Strategy
- Training: FP8 or BF16
- Inference (GPU): FP8, INT8, AWQ
- Inference (CPU): GGUF, ONNX INT8
- Mobile: INT8, GGUF Q4
- Memory-constrained: 4-bit (AWQ, GPTQ, NF4)

## Accuracy Preservation
- Calibration dataset selection
- Per-channel for activations
- Mixed precision strategies
- Evaluation after quantization
- When accuracy loss is unacceptable
