---
title: "24.11 Model Serialization & Checkpointing"
---

# 24.11 Model Serialization & Checkpointing

## PyTorch State Dict
- torch.save(model.state_dict(), path)
- torch.load with map_location
- Loading state dict into model
- Partial state dict loading
- Strict vs non-strict loading

## Full Model Saving
- torch.save(model, path) approach
- Limitations (code dependency)
- When to use full model save
- Portability considerations

## SafeTensors Format (2025 Standard)
- Hugging Face standard format
- Security advantages (no pickle)
- Fast memory-mapped loading
- save_pretrained() with safe_serialization=True
- Converting from .bin to .safetensors

## Checkpoint Management
- Saving during training
- ModelCheckpoint callback patterns
- Best model selection (by metric)
- Last N checkpoints retention
- Checkpoint naming conventions

## Optimizer State Saving
- Saving optimizer state_dict
- Learning rate scheduler state
- Complete training state
- Resume training correctly

## Distributed Checkpointing
- FSDP checkpoint strategies
- Sharded checkpointing
- Consolidating sharded checkpoints
- DeepSpeed checkpoint format

## Hugging Face Integration
- save_pretrained() method
- from_pretrained() loading
- Model cards and config.json
- Upload to Hugging Face Hub
- Private model repositories

## Large Model Considerations
- Memory-efficient loading
- Low CPU memory loading
- Sharded model files
- Lazy loading strategies

## Versioning and Tracking
- DVC for large files
- Git LFS integration
- MLflow model registry
- Weights & Biases artifacts
- Model provenance tracking
