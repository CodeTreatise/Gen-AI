---
title: "24.21 Model Export & Production"
---

# 24.21 Model Export & Production

## Export Formats Overview
- PyTorch native (.pt, .pth)
- TorchScript (.pt compiled)
- ONNX (.onnx)
- SafeTensors (.safetensors)
- GGUF (llama.cpp)
- ExecuTorch (.pte)
- TensorRT engines

## SafeTensors Format
- Hugging Face standard
- Secure (no arbitrary code execution)
- Fast loading (memory-mapped)
- Cross-framework compatibility
- Default for HF model uploads
- Converting from pickle format

## ONNX Export
- Open Neural Network Exchange
- Framework interoperability
- torch.onnx.export() usage
- Dynamic axes for variable shapes
- Operator support limitations
- Optimization with onnx-simplifier

## TorchScript
- torch.jit.script for full compilation
- torch.jit.trace for graph capture
- Hybrid approach (script + trace)
- Limitations with dynamic control flow
- Production deployment pattern

## torch.export (PyTorch 2.x)
- Successor to TorchScript
- Full graph capture
- Dynamic shape support
- Foundation for ExecuTorch
- AOTInductor for deployment

## Model Versioning
- Hugging Face Hub versioning
- DVC for model files
- MLflow model registry
- Weights & Biases artifacts
- Git LFS considerations

## Containerization
- Docker for ML models
- NVIDIA Container Toolkit
- Multi-stage builds
- Base image selection:
  - nvidia/cuda
  - pytorch/pytorch
  - huggingface/transformers
- Size optimization strategies

## Serving Infrastructure
- FastAPI + Uvicorn
- Ray Serve for scaling
- Triton Inference Server
- BentoML packaging
- SageMaker endpoints
- Vertex AI deployment

## API Design Patterns
- REST vs gRPC
- Streaming responses
- Batch inference endpoints
- Health check endpoints
- OpenAPI documentation

## Monitoring Production Models
- Prometheus metrics
- Grafana dashboards
- Request latency tracking
- Model performance monitoring
- Drift detection
- Alerting strategies

## Scaling Strategies
- Horizontal scaling (replicas)
- Vertical scaling (larger GPU)
- Auto-scaling configuration
- Load balancing
- Queue-based processing

## CI/CD for ML
- Model testing pipelines
- Automated evaluation
- Deployment automation
- Rollback strategies
- A/B testing infrastructure

## Cost Management
- Spot instances for inference
- Right-sizing GPU selection
- Request batching
- Caching strategies
- Cold start optimization

## Security Considerations
- Model access control
- Input validation
- Output filtering
- Rate limiting
- Audit logging
- Secrets management
