---
title: "Prompt Chaining & Pipelines"
---

# Prompt Chaining & Pipelines

- **What is Prompt Chaining?**
  - Sequential LLM calls with output→input flow
  - Breaking complex tasks into stages
  - Specialized prompts for each stage
  - Quality gates between stages
- **Pipeline Design Patterns**
  - Linear chains (A → B → C)
  - Branching pipelines (A → [B, C] → D)
  - Iterative refinement loops
  - Validation checkpoints
- **Stage Handoff Strategies**
  - Structured output for clean handoff
  - Extracting relevant portions
  - Context summarization between stages
  - Error propagation handling
- **Common Pipeline Patterns**
  - Generate → Critique → Refine
  - Extract → Transform → Load (ETL)
  - Plan → Execute → Verify
  - Draft → Review → Finalize
- **Optimization Techniques**
  - Parallel stage execution where possible
  - Caching intermediate results
  - Early termination on failure
  - Stage-specific model selection (fast vs accurate)
- **Framework Integration**
  - LangChain LCEL pipelines
  - LlamaIndex query pipelines
  - Custom orchestration code
  - Streaming through pipelines
