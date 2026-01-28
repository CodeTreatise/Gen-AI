---
title: "Unit 18: Security, Privacy & Ethics"
---

# Unit 18: Security, Privacy & Ethics

## Overview & Importance

AI systems introduce unique security risks, privacy considerations, and ethical challenges. This unit covers protecting AI systems from attacks, handling user data responsibly, and building AI features that are safe and fair.

Security and ethics matter because:
- AI systems can be manipulated (prompt injection)
- AI handles sensitive user data
- AI outputs can cause real harm
- Regulations require responsible AI practices
- Trust is essential for AI adoption

## Prerequisites

- All previous units
- Understanding of web security basics
- Awareness of data privacy concepts

## Learning Objectives

By the end of this unit, you will be able to:
- Identify and prevent prompt injection attacks
- Implement input validation for AI systems
- Handle personal data appropriately
- Moderate AI outputs for safety
- Build transparent AI features
- Address bias and fairness concerns
- Comply with AI regulations and guidelines

## Real-world Applications

- Enterprise AI Governance (NEW 2025)
  - AI governance platforms
  - Policy enforcement automation
  - Compliance dashboards
  - Risk management frameworks
- Healthcare AI Compliance
  - HIPAA-compliant AI deployments
  - Clinical decision support safety
  - Patient data protection
  - FDA medical device regulations
- Financial Services AI Safety
  - Model risk management
  - Fair lending compliance
  - Fraud detection ethics
  - Explainability requirements
- Consumer Protection in AI Products
  - AI disclosure requirements
  - Automated decision appeals
  - Consumer rights (CCPA, GDPR)
  - Children's privacy (COPPA)
- Content Platform Moderation
  - Social media AI moderation
  - User-generated content safety
  - Misinformation detection
  - Platform liability considerations
- Child Safety in AI
  - CSAM detection (mandatory)
  - Age verification
  - Child-safe content filtering
  - Educational AI safety

## Market Demand & Relevance

- AI Safety Roles Emerging (NEW 2025)
  - AI Safety Engineer
  - AI Red Team Specialist
  - AI Ethics Officer
  - AI Governance Lead
  - Trust and Safety Manager
  - AI Compliance Analyst
- Market Trends
  - AI safety is board-level concern
  - Regulations increasing globally (EU AI Act, state laws)
  - Trust and Safety teams expanding 2x-3x
  - High liability risk without proper controls
  - Competitive advantage through responsible AI
  - Growing demand for AI ethics expertise
- Key Certifications (NEW 2025)
  - Certified AI Governance Professional
  - AI Ethics certification programs
  - Privacy certifications (CIPP, CIPM)
  - Security certifications (CISSP + AI focus)
- India-Specific Considerations (NEW 2025)
  - Digital India AI ethics guidelines
  - DPDP Act (Digital Personal Data Protection)
  - MeitY AI governance recommendations
  - Sector-specific regulations emerging

---

## Resources & References

### Official Documentation

- **OWASP GenAI Security Project**
  - LLM Top 10 2025: https://genai.owasp.org/llm-top-10/
  - Resources: https://genai.owasp.org/resources/
  - Contribute: https://genai.owasp.org/contribute/
  - GitHub: https://github.com/OWASP/www-project-top-10-for-large-language-model-applications (1k+ stars)

- **EU AI Act**
  - Official Text: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689
  - EC AI Act Overview: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
  - AI Act Explorer: https://artificialintelligenceact.eu/ai-act-explorer/
  - Compliance Checker: https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/
  - High-Level Summary: https://artificialintelligenceact.eu/high-level-summary/

- **Azure AI Content Safety**
  - Documentation: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/
  - Prompt Shields: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection
  - Groundedness Detection: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness
  - Protected Material: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/protected-material
  - Content Safety Studio: https://contentsafety.cognitive.azure.com/

- **OpenAI Safety Documentation**
  - Moderation API: https://platform.openai.com/docs/guides/moderation
  - Safety Best Practices: https://platform.openai.com/docs/guides/safety-best-practices
  - Usage Policies: https://openai.com/policies/usage-policies
  - Data Privacy: https://openai.com/enterprise-privacy/

### Guardrails Frameworks (GitHub)

- **NVIDIA NeMo Guardrails** (5.5k+ stars)
  - Repository: https://github.com/NVIDIA/NeMo-Guardrails
  - Documentation: https://docs.nvidia.com/nemo/guardrails/
  - License: Apache-2.0
  - Features: Colang language, topical/safety/security rails, LangChain integration

- **Guardrails AI** (6.3k+ stars)
  - Repository: https://github.com/guardrails-ai/guardrails
  - Documentation: https://www.guardrailsai.com/docs
  - Guardrails Hub: https://hub.guardrailsai.com/
  - Guardrails Index: https://index.guardrailsai.com/
  - License: Apache-2.0
  - Features: Validators, structural/semantic validation, Guard class

- **Llama Guard 3** (Meta)
  - Model Card: https://huggingface.co/meta-llama/Llama-Guard-3-8B
  - INT8 Version: https://huggingface.co/meta-llama/Llama-Guard-3-8B-INT8
  - 1B On-Device: https://huggingface.co/meta-llama/Llama-Guard-3-1B
  - Paper: https://arxiv.org/abs/2312.06674
  - Features: 14 safety categories (S1-S14), input/output filtering, 8 languages

### AI Red Teaming Tools (GitHub)

- **Microsoft PyRIT** (3.3k+ stars)
  - Repository: https://github.com/Azure/PyRIT
  - Documentation: https://azure.github.io/PyRIT/
  - Discord: https://discord.gg/9fMpq3tc8u
  - License: MIT
  - Paper: https://arxiv.org/abs/2410.02828
  - Features: Red teaming framework, attack orchestration, scorers, converters

- **NVIDIA Garak** (6.8k+ stars)
  - Repository: https://github.com/NVIDIA/garak
  - Documentation: https://docs.garak.ai/
  - Discord: https://discord.gg/uVch4puUCs
  - License: Apache-2.0
  - Features: LLM vulnerability scanner, 50+ probes, multi-provider support

- **MITRE ATLAS**
  - Website: https://atlas.mitre.org/
  - GitHub: https://github.com/mitre-atlas/
  - Data Repository: https://github.com/mitre-atlas/atlas-data
  - Navigator: https://github.com/mitre-atlas/atlas-navigator
  - Features: 16 tactics, 147 techniques, 35 mitigations, 45 case studies

### Content Moderation APIs

- **OpenAI Moderation API**
  - Documentation: https://platform.openai.com/docs/guides/moderation
  - Model: omni-moderation-latest
  - Free to use, text + image moderation

- **Azure AI Content Safety**
  - Portal: https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety
  - Studio: https://contentsafety.cognitive.azure.com/
  - 4 severity levels, custom categories, 100+ compliance certifications

- **Google Perspective API**
  - Documentation: https://developers.perspectiveapi.com/
  - Toxicity, severe toxicity, identity attack, insult, profanity, threat

- **Anthropic Claude Safety**
  - Constitutional AI: https://www.anthropic.com/research/constitutional-ai
  - Responsible Scaling Policy: https://www.anthropic.com/news/anthropics-responsible-scaling-policy
  - Safety Levels (ASL-1 to ASL-4)

### Regulatory Resources

- **EU AI Act Resources**
  - Future of Life Institute Guide: https://artificialintelligenceact.eu/
  - EC AI Pact: https://digital-strategy.ec.europa.eu/en/policies/ai-pact
  - AI Act Service Desk: https://ai-act-service-desk.ec.europa.eu/en
  - Newsletter: https://artificialintelligenceact.substack.com/

- **NIST AI Risk Management Framework**
  - Framework: https://www.nist.gov/itl/ai-risk-management-framework
  - Playbook: https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook
  - Trustworthy AI: https://www.nist.gov/trustworthy-ai

- **Colorado AI Act (SB24-205)**
  - Text: https://leg.colorado.gov/bills/sb24-205
  - Effective: February 1, 2026
  - First comprehensive US state AI law

### Privacy & Compliance

- **OpenAI Enterprise Privacy**
  - Trust Portal: https://trust.openai.com/
  - Data Privacy FAQ: https://help.openai.com/en/collections/3686619-trust-safety-and-security
  - Zero Data Retention (ZDR) documentation
  - Regional Endpoints: us.api.openai.com, eu.api.openai.com

- **LangChain Trust Center**
  - Portal: https://trust.langchain.com/
  - HIPAA, SOC 2 Type 2, GDPR compliance

- **Data Protection Regulations**
  - GDPR Official: https://gdpr.eu/
  - CCPA Official: https://oag.ca.gov/privacy/ccpa
  - India DPDP Act: https://www.meity.gov.in/data-protection-framework

### Learning Resources

- **Microsoft AI Red Teaming**
  - Planning Red Teaming: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming
  - PyRIT Cookbooks: https://azure.github.io/PyRIT/cookbooks/README.html
  - Responsible AI Toolbox: https://responsibleaitoolbox.ai/

- **OWASP Learning**
  - LLM Security Resources: https://genai.owasp.org/resources/
  - AI Security Solutions Landscape: https://genai.owasp.org/ai-security-solutions-landscape/
  - Meetings: https://genai.owasp.org/meetings/

- **Garak Scanning**
  - Getting Started: https://docs.garak.ai/garak/llm-scanning-basics/setting-up
  - Probe Reference: https://reference.garak.ai/

### Research Papers

- "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations" (Meta, 2023)
- "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications" (NVIDIA, EMNLP 2023)
- "PyRIT: A Framework for Security Risk Identification and Red Teaming in GenAI Systems" (Microsoft, 2024)
- "Garak: A Framework for Security Probing Large Language Models" (NVIDIA, 2024)
- "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023)
- "Tree of Attacks: Jailbreaking Black-Box LLMs with Auto-Generated Prompts" (2023)
- "MITRE ATLAS: Adversarial Threat Landscape for AI Systems" (MITRE, ongoing)

### Community & Support

- **OWASP GenAI Community**
  - Slack: #project-top10-for-llm (owasp.org/slack/invite)
  - LinkedIn: https://www.linkedin.com/company/owasp-top-10-for-large-language-model-applications/
  - Newsletter: https://llmtop10.beehiiv.com/subscribe

- **NeMo Guardrails Community**
  - GitHub Discussions: https://github.com/NVIDIA/NeMo-Guardrails/discussions

- **PyRIT Community**
  - Discord: https://discord.gg/9fMpq3tc8u
  - GitHub Issues: https://github.com/Azure/PyRIT/issues

- **Garak Community**
  - Discord: https://discord.gg/uVch4puUCs
  - Email: garak@nvidia.com

- **Guardrails AI Community**
  - Discord: https://discord.gg/gw4cR9QvYE
  - Twitter: https://twitter.com/guardrails_ai

### Standards & Certifications

- IEEE 7000 Series (Ethical AI)
- ISO/IEC 42001 (AI Management Systems)
- OECD AI Principles: https://oecd.ai/en/ai-principles
- UNESCO AI Ethics Recommendation
- Partnership on AI: https://partnershiponai.org/
- MLCommons AI Safety: https://mlcommons.org/ai-safety/
