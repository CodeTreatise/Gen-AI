---
title: "Unit 23: AI Observability & Monitoring Tools"
---

# Unit 23: AI Observability & Monitoring Tools

## Overview & Importance

AI observability provides visibility into AI system behavior, performance, and quality. Unlike traditional application monitoring, AI observability must track prompts, responses, token usage, latency, and semantic quality. This unit covers tools, patterns, and practices for AI-specific observability.

Observability matters because:
- AI systems fail in subtle ways (quality degradation)
- Costs can spiral without visibility
- Debugging AI issues requires detailed traces
- Continuous improvement needs data

## Prerequisites

- Production deployment knowledge (Unit 13)
- API integration experience (Unit 3)
- Understanding of logging concepts
- Basic analytics awareness

## Learning Objectives

By the end of this unit, you will be able to:
- Implement comprehensive AI system logging
- Set up distributed tracing for AI workflows
- Monitor AI quality and performance metrics
- Use specialized AI observability platforms
- Build dashboards for AI system health
- Debug AI issues effectively
- Implement alerting for AI-specific problems

## Real-world Applications

- Production AI system monitoring
- AI cost management
- Quality assurance dashboards
- Performance optimization
- Incident investigation
- Compliance reporting
- Continuous improvement programs

## Market Demand & Relevance

- AI observability is emerging category
- Critical for enterprise AI operations
- MLOps and AI Ops roles require these skills
- Platform knowledge highly valued
- Cost savings from better visibility
- Growing vendor landscape

## Resources & References

### Official Documentation
- **LangSmith Documentation**: https://docs.langchain.com/langsmith/
- **Langfuse Documentation**: https://langfuse.com/docs
- **Arize Phoenix Documentation**: https://docs.arize.com/phoenix
- **Arize AX Documentation**: https://arize.com/docs/ax
- **Helicone Documentation**: https://docs.helicone.ai/
- **Braintrust Documentation**: https://www.braintrust.dev/docs
- **Traceloop Documentation**: https://www.traceloop.com/docs
- **Weights & Biases Prompts**: https://docs.wandb.ai/guides/prompts
- **Datadog LLM Observability**: https://docs.datadoghq.com/llm_observability/

### OpenTelemetry for GenAI
- **OTel GenAI Semantic Conventions**: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- **OpenLLMetry**: https://github.com/traceloop/openllmetry
- **OTel Python SDK**: https://opentelemetry.io/docs/languages/python/
- **OTel JavaScript SDK**: https://opentelemetry.io/docs/languages/js/
- **OTel Collector**: https://opentelemetry.io/docs/collector/
- **Jaeger**: https://www.jaegertracing.io/

### Guardrails & Safety
- **Guardrails AI**: https://www.guardrailsai.com/
- **Guardrails Hub**: https://hub.guardrailsai.com/
- **NeMo Guardrails**: https://github.com/NVIDIA/NeMo-Guardrails
- **LlamaGuard**: https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/
- **Rebuff AI**: https://github.com/protectai/rebuff
- **Lakera Guard**: https://www.lakera.ai/

### Observability Platforms
- **LangSmith**: https://smith.langchain.com/
- **Langfuse**: https://langfuse.com/
- **Arize Phoenix**: https://phoenix.arize.com/
- **Helicone**: https://www.helicone.ai/
- **Braintrust**: https://www.braintrust.dev/
- **Traceloop**: https://www.traceloop.com/
- **Portkey**: https://portkey.ai/
- **Literal AI**: https://www.literalai.com/
- **Parea AI**: https://www.parea.ai/

### Evaluation Frameworks
- **Ragas**: https://docs.ragas.io/
- **DeepEval**: https://docs.confident-ai.com/
- **Promptfoo**: https://promptfoo.dev/
- **Giskard**: https://www.giskard.ai/
- **TruLens**: https://www.trulens.org/
- **Phoenix Evals**: https://docs.arize.com/phoenix/evaluation/llm-evals

### Enterprise APM Integration
- **Datadog LLM Observability**: https://www.datadoghq.com/product/llm-observability/
- **New Relic AI Monitoring**: https://newrelic.com/platform/ai-monitoring
- **Dynatrace AI Observability**: https://www.dynatrace.com/
- **Splunk Observability**: https://www.splunk.com/en_us/products/observability.html
- **Grafana**: https://grafana.com/

### Research Papers
- **Holistic Evaluation of Language Models (HELM)**: https://arxiv.org/abs/2211.09110
- **Judging LLM-as-a-Judge**: https://arxiv.org/abs/2306.05685
- **FActScore: Fine-grained Atomic Evaluation**: https://arxiv.org/abs/2305.14251
- **RAGAS: Automated Evaluation of RAG**: https://arxiv.org/abs/2309.15217
- **TruthfulQA**: https://arxiv.org/abs/2109.07958

### Blogs & Articles
- **LangSmith Blog**: https://blog.langchain.dev/tag/langsmith/
- **Langfuse Blog**: https://langfuse.com/blog
- **Arize AI Blog**: https://arize.com/blog/
- **Braintrust Blog**: https://www.braintrust.dev/blog
- **Helicone Blog**: https://www.helicone.ai/blog
- **Eugene Yan - LLM Observability**: https://eugeneyan.com/

### Video Tutorials
- **LangChain YouTube**: https://www.youtube.com/@LangChain
- **Arize AI YouTube**: https://www.youtube.com/@arizeai
- **Weights & Biases YouTube**: https://www.youtube.com/@WeightsBiases
- **AI Jason**: https://www.youtube.com/@AIJasonZ
- **James Briggs**: https://www.youtube.com/@jamesbriggs

### GitHub Repositories
- **Langfuse**: https://github.com/langfuse/langfuse
- **Arize Phoenix**: https://github.com/Arize-ai/phoenix
- **OpenLLMetry**: https://github.com/traceloop/openllmetry
- **Guardrails AI**: https://github.com/guardrails-ai/guardrails
- **Promptfoo**: https://github.com/promptfoo/promptfoo
- **Ragas**: https://github.com/explodinggradients/ragas
- **DeepEval**: https://github.com/confident-ai/deepeval

### Courses & Learning
- **DeepLearning.AI - Evaluating LLMs**: https://www.deeplearning.ai/short-courses/
- **LangChain Academy**: https://academy.langchain.com/
- **Arize AI University**: https://arize.com/resource-hub/
- **Weights & Biases Courses**: https://www.wandb.courses/

### Community & Forums
- **Langfuse GitHub Discussions**: https://github.com/langfuse/langfuse/discussions
- **Arize AI Slack**: https://arize.com/community/
- **LangChain Discord**: https://discord.gg/langchain
- **Traceloop Slack**: https://www.traceloop.com/slack
- **OpenTelemetry Community**: https://opentelemetry.io/community/

### LLM Gateways with Observability
- **Portkey**: https://portkey.ai/
- **LiteLLM**: https://github.com/BerriAI/litellm
- **AI Gateway (Cloudflare)**: https://developers.cloudflare.com/ai-gateway/
- **Martian**: https://withmartian.com/
- **Keywords AI**: https://keywordsai.co/

### Security & Compliance
- **OWASP LLM Top 10**: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **Protect AI**: https://protectai.com/
- **Lakera AI**: https://www.lakera.ai/
- **Robust Intelligence**: https://www.robustintelligence.com/
- **CalypsoAI**: https://calypsoai.com/
