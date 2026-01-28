---
title: "24.14 TRL for RLHF Training"
---

# 24.14 TRL for RLHF Training

## What is TRL
- Transformer Reinforcement Learning library
- Hugging Face ecosystem component
- LLM alignment and fine-tuning
- RLHF implementation made accessible
- Active development (v0.27+ as of 2025)

## RLHF Overview
- Reinforcement Learning from Human Feedback
- Preference data collection
- Reward model training
- Policy optimization
- Why RLHF matters for AI alignment

## GRPOTrainer (2025)
- Group Relative Policy Optimization
- DeepSeek's GRPO algorithm
- Advantages over PPO (simpler, no critic)
- Token-level and outcome-level rewards
- Multi-GPU and FSDP support
- FP8 training support

## DPOTrainer (Direct Preference Optimization)
- Bypassing reward model training
- Direct optimization from preferences
- Simpler than PPO pipeline
- Reference model management
- Beta parameter tuning
- When to prefer DPO over PPO

## PPOTrainer
- Proximal Policy Optimization
- Classic RLHF approach
- Reward model integration
- Value function training
- KL divergence penalty
- Clipping parameters

## RLOOTrainer
- REINFORCE Leave-One-Out
- Variance reduction technique
- Batch-level baseline
- Simpler than PPO
- When to use RLOO

## SFTTrainer
- Supervised Fine-Tuning
- First step before RLHF
- Dataset formatting (conversational, instruction)
- PEFT integration (LoRA, QLoRA)
- Efficient data collation

## KTOTrainer
- Kahneman-Tversky Optimization
- Based on prospect theory
- Handles unpaired preference data
- Simpler data requirements
- When to prefer KTO

## RewardTrainer
- Training reward models
- Preference dataset format
- Bradley-Terry model
- Evaluation metrics
- Multi-objective rewards

## VLM Alignment (2025)
- Vision-Language Model alignment
- Image preference data
- VLM-specific trainers
- Multimodal RLHF challenges

## Practical TRL Workflow
- SFT → Reward Model → RLHF
- Or: SFT → DPO (simpler path)
- Dataset preparation (Argilla, Label Studio)
- Evaluation during training
- Hyperparameter recommendations

## Integration with Other Tools
- Works with PEFT/LoRA
- DeepSpeed and FSDP support
- Weights & Biases logging
- Hugging Face Hub integration
- Axolotl and Unsloth compatibility
