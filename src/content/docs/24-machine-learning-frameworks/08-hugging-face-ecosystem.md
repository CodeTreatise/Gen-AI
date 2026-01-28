---
title: "24.8 Hugging Face Transformers v5"
---

# 24.8 Hugging Face Transformers v5

## What is Hugging Face (2025-2026)
- ML platform and community (800K+ models)
- Transformers library (v5.x era)
- Model Hub ecosystem
- Datasets library
- Spaces for demos and apps
- Inference Endpoints for deployment

## Transformers v5 Changes
- New tokenization system
- Model definition framework pivot
- Improved memory efficiency
- Enhanced torch.compile support
- Breaking changes from v4

## Model Hub (2025)
- Browsing 800K+ models
- Model cards and documentation
- Model search and filtering
- Gated models and licensing
- Model versioning (revisions)
- Organization structure

## Pipelines
- High-level inference API
- Task-specific pipelines
- Text generation pipeline (generate, streaming)
- Classification and NER pipelines
- Zero-shot classification
- Question answering
- Image and multimodal pipelines

## AutoClasses
- AutoModel for models
- AutoModelForCausalLM for LLMs
- AutoTokenizer for tokenizers
- AutoConfig for configuration
- From_pretrained loading patterns
- trust_remote_code considerations
- torch_dtype and device_map options

## Tokenizers
- Fast tokenizers (Rust-based)
- Encoding/decoding
- Special tokens handling
- Padding and truncation
- Batch encoding
- Chat templates

## Trainer API
- Trainer class usage
- TrainingArguments configuration
- Evaluation strategies
- Callbacks (EarlyStopping, TensorBoard)
- Distributed training integration
- PEFT integration
