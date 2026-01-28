---
title: "24.12 Parameter-Efficient Fine-Tuning (PEFT)"
---

# 24.12 Parameter-Efficient Fine-Tuning (PEFT)

## Why PEFT Matters
- Full fine-tuning vs parameter-efficient
- Memory reduction (often 10-100x fewer parameters)
- Faster training with comparable results
- Multiple adapters for one base model
- Democratizing LLM fine-tuning

## LoRA (Low-Rank Adaptation)
- Core concept: low-rank decomposition of weight updates
- Rank (r) parameter and selection
- Alpha scaling factor
- Target modules (q_proj, v_proj, k_proj, o_proj, gate_proj, etc.)
- LoRA for different architectures (Llama, Mistral, Qwen)
- Merging LoRA weights into base model

## QLoRA (Quantized LoRA)
- 4-bit quantization + LoRA
- NF4 (Normal Float 4-bit) datatype
- Double quantization for memory savings
- Paged optimizers (Adam, AdamW 8-bit)
- Training 65B+ models on consumer GPUs
- bitsandbytes integration

## DoRA (Weight-Decomposed LoRA)
- Decomposing weight into magnitude and direction
- Improved performance over LoRA
- Similar memory footprint
- When to prefer DoRA over LoRA

## LoRA+ and Variants
- Different learning rates for A and B matrices
- rsLoRA (rank-stabilized)
- LoHA (Hadamard product adaptation)
- LoKr (Kronecker product adaptation)

## Hugging Face PEFT Library
- Unified interface for all PEFT methods
- get_peft_model() usage
- LoraConfig and PeftConfig
- Saving and loading adapters
- Multiple adapter management
- Merging and unloading

## Adapter Fusion
- Combining multiple LoRA adapters
- AdapterFusion for multi-task learning
- Weighted adapter merging
- Task-specific adapter switching

## Prefix Tuning
- Continuous prompts as trainable parameters
- Only updating prefix tokens
- Use cases and comparison to LoRA

## Prompt Tuning
- Soft prompt tokens
- Google's prompt tuning approach
- Comparison with prefix tuning
- When to use prompt tuning

## Practical Considerations
- Choosing target modules
- Rank selection (typically 8-64)
- Learning rate for adapters (higher than full fine-tuning)
- Training data requirements
- Evaluation strategies for PEFT
