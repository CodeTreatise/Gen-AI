---
title: "24.18 ExecuTorch & Edge Deployment"
---

# 24.18 ExecuTorch & Edge Deployment

## What is ExecuTorch
- PyTorch edge deployment solution
- Released as stable 1.0 (October 2025)
- On-device ML execution
- Replaces PyTorch Mobile
- Part of PyTorch ecosystem

## ExecuTorch 1.0 Features
- Stable API guarantees
- Broad hardware support
- Optimized for mobile and embedded
- Small binary size
- Low memory footprint
- Integration with Arm, Qualcomm, Apple

## Target Platforms
- iOS (CoreML backend)
- Android (XNNPACK, Vulkan)
- Arm processors (Arm backend)
- Qualcomm Snapdragon (QNN)
- MediaTek chips
- Microcontrollers (limited models)
- Wearables and IoT

## Export Workflow
- torch.export() from PyTorch 2.x
- ExecuTorch export API
- Operator support and lowering
- Delegation to accelerators
- Quantization for edge

## Backend Delegates
- XNNPACK (CPU, cross-platform)
- CoreML (Apple devices)
- QNN (Qualcomm Neural Network)
- MPS (Metal Performance Shaders)
- Vulkan (GPU, Android)
- Custom delegate creation

## Quantization for Edge
- Post-training quantization
- INT8 for edge inference
- Per-channel quantization
- Calibration with device data
- Size vs accuracy trade-offs

## On-Device LLMs
- Llama on mobile devices
- Model size considerations
- Token generation speed
- Memory management
- User experience implications

## iOS Deployment
- CoreML integration
- Swift API usage
- App Store considerations
- Performance optimization
- Background processing

## Android Deployment
- XNNPACK backend
- Vulkan GPU acceleration
- NDK integration
- APK size management
- Battery optimization

## Comparison with Alternatives
- ONNX Runtime Mobile
- TensorFlow Lite
- CoreML (Apple only)
- NCNN (Tencent)
- When to choose ExecuTorch

## Performance Optimization
- Model pruning for edge
- Knowledge distillation
- Operator fusion
- Memory planning
- Profiling on device

## Security Considerations
- Model protection
- Secure storage
- Inference privacy
- Federated learning integration
