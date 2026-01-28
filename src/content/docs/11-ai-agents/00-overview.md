---
title: "Unit 11: AI Agents"
---

# Unit 11: AI Agents

## Overview & Importance

AI agents are autonomous systems that can plan, reason, and take actions to achieve goals. Unlike simple chatbots that respond to single queries, agents can handle complex multi-step tasks, use tools, maintain memory, and adapt their approach based on results.

Agents represent the evolution from:
- Single query → Goal-oriented tasks
- Predefined flows → Dynamic planning
- Human-directed → Autonomous operation

### 2025-2026 Agent Evolution

**Paradigm Shifts:**
- ReAct pattern now standard across all frameworks
- MCP (Model Context Protocol) as universal tool integration standard
- Multi-agent orchestration for complex workflows
- Computer/browser use as native agent capability
- Agent-to-agent (A2A) communication protocols
- Built-in observability and tracing

**Framework Landscape (2025-2026):**
| Framework | Focus | Key Feature |
|-----------|-------|-------------|
| OpenAI Agents SDK | Production agents | Handoffs, guardrails, tracing |
| LangGraph | Stateful workflows | Cycles, persistence, human-in-loop |
| CrewAI | Multi-agent crews | Flows, role-based agents |
| AutoGen 0.4 | Enterprise/research | Teams, GraphFlow, Magentic-One |
| Google ADK | Cloud-native | Vertex AI, Gemini integration |

**Industry Adoption:**
- 85%+ of enterprise AI projects now include agent capabilities
- Agent frameworks grew 340% in adoption (2024-2025)
- Average production agent uses 12+ tools
- Multi-agent systems handling 60% of complex workflows

## Prerequisites

- Function calling mastery (Unit 8)
- MCP understanding (Unit 8)
- Strong prompt engineering (Unit 5)
- Understanding of RAG (Unit 7)
- Solid programming fundamentals
- Async/await proficiency
- Basic graph concepts (helpful for LangGraph)

## Learning Objectives

By the end of this unit, you will be able to:
- Explain agent architectures and design patterns
- Implement reasoning and planning loops
- Build agents with tool-use capabilities
- Design memory systems for agent context
- Handle agent failures and recovery
- Implement human-in-the-loop controls
- Evaluate agent performance
- **NEW 2025**: Build production agents with OpenAI Agents SDK
- **NEW 2025**: Create stateful workflows with LangGraph
- **NEW 2025**: Design multi-agent crews with CrewAI Flows
- **NEW 2025**: Implement agent handoffs and guardrails
- **NEW 2025**: Use AutoGen 0.4 Teams and GraphFlow
- **NEW 2026**: Integrate MCP servers across frameworks
- **NEW 2026**: Build computer/browser use agents
- **NEW 2026**: Implement agent-to-agent communication

## Real-world Applications

- Research assistants that gather and synthesize information
- Code generation agents that write, test, and fix code (Devin, Cursor, GitHub Copilot Agent Mode)
- Customer service agents handling complex multi-turn issues
- Data analysis agents that explore, visualize, and report
- Personal assistants managing schedules and tasks
- DevOps agents handling deployments and incident response
- Content creation agents (research → draft → edit → publish)
- Sales agents qualifying and nurturing leads
- Agentic RAG for autonomous document analysis
- Computer use agents for browser automation (NEW 2025)
- Voice agents for real-time conversations (NEW 2025)
- Multi-agent systems for complex enterprise workflows

## Market Demand & Relevance

- Agents represent the next evolution of AI applications (2025 focus)
- Highest growth area in AI development job postings
- Companies investing heavily in agent capabilities (OpenAI, Google, Microsoft, Anthropic)
- Premium salaries for agent development skills ($150K-$300K+)
- Early adoption advantage in emerging agent ecosystem
- Foundational for future autonomous systems
- Agent frameworks maturing rapidly (Q1 2025 landscape shift)
- MCP becoming industry standard for tool integration
- Multi-agent orchestration as key differentiator
- Enterprise adoption accelerating (2025-2026 wave)

---

## Resources & References

### Official Documentation
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - Official Python SDK for building agents
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Stateful agent orchestration
- [CrewAI Documentation](https://docs.crewai.com/) - Multi-agent crew framework
- [AutoGen Documentation](https://microsoft.github.io/autogen/stable/) - Microsoft's agent framework
- [Google Agent Development Kit](https://google.github.io/adk-docs/) - Google's ADK for agents
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification and guides

### Framework GitHub Repositories
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CrewAI](https://github.com/crewAIInc/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)
- [Google ADK Python](https://github.com/google/adk-python)
- [MCP Servers](https://github.com/modelcontextprotocol/servers)

### Learning Resources
- [LangChain Academy](https://academy.langchain.com/) - Free LangGraph courses
- [CrewAI University](https://www.crewai.com/university) - CrewAI certification
- [DeepLearning.AI Agents Course](https://www.deeplearning.ai/) - Agent fundamentals
- [AutoGen Notebooks](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/) - Hands-on examples

### Agent Platforms & Tools
- [LangSmith](https://smith.langchain.com/) - Agent observability and tracing
- [LangGraph Platform](https://www.langchain.com/langgraph-platform) - Production deployment
- [CrewAI Enterprise](https://www.crewai.com/enterprise) - Enterprise agent platform
- [AutoGen Studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/) - No-code agent builder
- [Vertex AI Agent Engine](https://cloud.google.com/vertex-ai/docs/agents) - Google Cloud agent deployment

### Observability & Tracing
- [Langfuse](https://langfuse.com/) - Open-source LLM observability
- [Arize Phoenix](https://phoenix.arize.com/) - Agent tracing and evaluation
- [Weights & Biases Weave](https://wandb.ai/weave) - ML experiment tracking
- [OpenTelemetry](https://opentelemetry.io/) - Distributed tracing standard

### Computer Use & Browser Automation
- [Anthropic Computer Use](https://docs.anthropic.com/en/docs/computer-use) - Claude computer control
- [Playwright](https://playwright.dev/) - Browser automation framework
- [Puppeteer](https://pptr.dev/) - Chrome DevTools Protocol
- [Browserbase](https://www.browserbase.com/) - Cloud browser infrastructure

### Research Papers & Articles
- "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- "AutoGPT: An Autonomous GPT-4 Experiment" (2023)
- "The Landscape of Emerging AI Agent Architectures" (2024)
- "Multi-Agent Collaboration Patterns for Enterprise AI" (2025)

### Community & Support
- [LangChain Discord](https://discord.gg/langchain) - LangGraph community
- [CrewAI Discord](https://discord.gg/crewai) - CrewAI discussions
- [AutoGen Discord](https://discord.gg/autogen) - Microsoft AutoGen community
- [r/LangChain](https://reddit.com/r/LangChain) - Reddit community
- [AI Agents Slack](https://aiagents.slack.com/) - Cross-framework discussions
