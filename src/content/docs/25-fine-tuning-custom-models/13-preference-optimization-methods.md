---
title: "25.13 Preference Optimization Methods"
---

# 25.13 Preference Optimization Methods

- Direct Preference Optimization (DPO)
  - Preference-based training without reward models
  - Chosen vs rejected response pairs
  - Simpler alternative to RLHF
  - OpenAI DPO support for GPT models
  - Dataset format requirements
- Odds Ratio Preference Optimization (ORPO)
  - Combined SFT and preference alignment
  - No reference model required
  - Single-stage training efficiency
  - Reduced computational cost
  - When to choose ORPO over DPO
- Kahneman-Tversky Optimization (KTO)
  - Works with unpaired preference data
  - Binary feedback (good/bad) sufficient
  - Based on prospect theory
  - Lower data requirements
  - Use cases for KTO
- Identity Preference Optimization (IPO)
  - Addresses DPO overfitting issues
  - Improved theoretical guarantees
  - Length normalization
  - Stable training dynamics
- Contrastive Preference Optimization (CPO)
  - Contrastive learning approach
  - Multiple negative samples
  - Improved preference margins
  - Batch construction strategies
- Choosing the right method
  - Data availability considerations
  - Paired vs unpaired preferences
  - Compute budget constraints
  - Quality vs efficiency trade-offs
  - Method comparison matrix
