---
title: "Additional Frameworks & Tools"
---

# Additional Frameworks & Tools

> **Last Updated:** January 2026 - Added emerging frameworks with community/industry adoption metrics

## Established Frameworks

- Semantic Kernel (Microsoft)
  - Plugin architecture
  - Planner abstraction
  - .NET and Python support
  - Azure AI integration
  - Enterprise focus
- Haystack by deepset
  - Pipeline-based design
  - RAG-focused framework
  - Component-based architecture
  - Production evaluation tools
- DSPy (Stanford)
  - Declarative LLM programming
  - Automatic prompt optimization
  - Modular program design
  - Teleprompters for tuning
- Pydantic AI (NEW 2025)
  - Type-safe agent development
  - Pydantic-first design
  - Validation integration
  - Python-native patterns
- Smolagents (Hugging Face)
  - Lightweight agent framework
  - Code-first agents
  - Hub integration
  - Model agnostic

## Emerging Frameworks (January 2026)

- Deep Agents (LangChain) - NEW January 2026
  - Official LangChain deep agent harness
  - Version: v0.3.6+ (released Jan 15, 2026)
  - GitHub: 8.4k stars, 1.3k forks, 46 contributors
  - Community/Industry adoption:
    - Official backing by LangChain team
    - 19 releases in 6 months → rapid iteration
    - Active development (commits daily)
    - Growing ecosystem adoption
  - Core architecture:
    - Built on LangGraph StateGraph
    - Middleware-based extensibility
    - Inspired by Claude Code patterns
  - Built-in middleware:
    - TodoListMiddleware - planning & task decomposition (`write_todos`, `read_todos`)
    - FilesystemMiddleware - context management (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`)
    - SubAgentMiddleware - subagent spawning (`task` tool)
    - SummarizationMiddleware - auto-summarize at 170k tokens
    - HumanInTheLoopMiddleware - `interrupt_on` for approvals
    - AnthropicPromptCachingMiddleware - cost reduction
  - Backend options:
    - StateBackend (default) - ephemeral in-memory
    - FilesystemBackend - real disk operations
    - StoreBackend - persistent via LangGraph Store
    - CompositeBackend - hybrid routing
  - Long-term memory:
    - Routes to durable storage via CompositeBackend
    - User preferences across sessions
    - Self-improving instructions
  - MCP integration: via langchain-mcp-adapters
  - HITL support: `interrupt_on` parameter with approve/edit/reject
  - API pattern: `create_deep_agent(model=, tools=, system_prompt=, subagents=, middleware=, backend=)`
  - Default model: claude-sonnet-4-5-20250929
  - Documentation: docs.langchain.com/oss/python/deepagents/overview
  - Install: `pip install deepagents`
  - Ideal for: Long-horizon tasks, Claude Code-like workflows, research agents

- Agno (agno-agi) - Multi-Agent Systems Platform
  - Version: v2.4.0 (released Jan 19, 2026)
  - GitHub: 37.1k stars, 4.9k forks, 383 contributors
  - Community/Industry adoption:
    - Very high community engagement
    - 156 releases → mature framework
    - Active enterprise adoption
    - Strong Discord community
  - Three-layer architecture:
    - Framework - agents, teams, workflows
    - AgentOS Runtime - FastAPI backend
    - Control Plane - UI for monitoring
  - Performance benchmarks:
    - 529× faster instantiation than LangGraph
    - 24× lower memory than LangGraph
    - Stateless horizontal scalability
  - Core features:
    - Model-agnostic (OpenAI, Anthropic, Google, local)
    - Type-safe I/O with input/output schemas
    - Async-first architecture
    - Natively multimodal (text, images, audio, video)
  - Memory & Knowledge:
    - Persistent session history
    - User memory across sessions
    - Agentic RAG with 20+ vector stores
    - "Culture" - shared long-term memory across agents
  - Orchestration:
    - Human-in-the-loop (confirmations, approvals)
    - Guardrails for validation/security
    - Pre/post lifecycle hooks
    - First-class MCP and A2A support
    - 100+ built-in toolkits
  - Production features:
    - Ready-to-use FastAPI runtime
    - Integrated control plane UI
    - Evals for accuracy, performance, latency
    - Durable execution for resumable workflows
    - RBAC and per-agent permissions
  - Install: `pip install agno`
  - Docs: docs.agno.com
  - Ideal for: Enterprise multi-agent systems, production deployments

- browser-use - AI Browser Automation
  - Version: v0.11.3 (Jan 2026)
  - GitHub: 75.9k stars, 9.1k forks, 283 contributors
  - Community/Industry adoption:
    - Extremely high adoption (75k+ stars)
    - Used by 2.3k+ projects
    - 105 releases → mature
    - Strong commercial offering (cloud)
  - Core capabilities:
    - Make websites accessible for AI agents
    - Playwright-based browser automation
    - Cloud option for stealth/scaling
  - API pattern:
    ```
    browser = Browser(use_cloud=True)
    agent = Agent(task="...", llm=ChatBrowserUse(), browser=browser)
    history = await agent.run()
    ```
  - Deployment options:
    - Local with Chromium
    - browser-use Cloud (stealth, scaling)
    - Sandbox decorator for production
  - Features:
    - Custom tool registry
    - Session persistence
    - Authentication handling
    - CAPTCHA solving integration
    - MCP integration
  - Templates: `uvx browser-use init --template default`
  - Install: `uv add browser-use`
  - Docs: docs.browser-use.com
  - Ideal for: Web automation, form filling, data extraction, browser-based agents

- Letta (formerly MemGPT) - Stateful Agents with Memory
  - Version: v0.16.2 (Jan 2026)
  - GitHub: 20.8k stars, 2.2k forks, 157 contributors
  - Community/Industry adoption:
    - Research-backed (academic origins)
    - Strong developer community
    - 172 releases → mature
    - Active Discord and forum
  - Core concept: AI with advanced memory that learns and self-improves
  - Two offerings:
    - Letta Code CLI - local terminal agents
    - Letta API - build agents into applications
  - Memory architecture:
    - memory_blocks with labels (human, persona, etc.)
    - Self-improving over time
    - Persistent across sessions
  - Features:
    - Skills and subagents support
    - Model-agnostic (recommends Opus 4.5, GPT-5.2)
    - Built-in tools (web_search, fetch_webpage)
    - Python and TypeScript SDKs
  - Install:
    - CLI: `npm install -g @letta-ai/letta-code`
    - Python: `pip install letta-client`
    - TypeScript: `npm install @letta-ai/letta-client`
  - Docs: docs.letta.com
  - Ideal for: Personalized assistants, self-improving agents, long-term memory applications

## Browser Automation Tools

- Playwright MCP integration
- Browserbase for cloud browsers
- Stagehand for AI browser control
- Anthropic Computer Use
- browser-use (see Emerging Frameworks above)

## Framework Selection Guide (Updated January 2026)

- Simple tasks → OpenAI Agents SDK
- Complex workflows → LangGraph
- Multi-agent teams → CrewAI or AutoGen
- Google ecosystem → Google ADK with A2A
- Enterprise .NET → Semantic Kernel
- Claude Code-like workflows → Deep Agents (LangChain)
- High-performance multi-agent → Agno
- Browser automation → browser-use
- Self-improving memory agents → Letta
- Model-agnostic + fast → Agno
- LangChain ecosystem → Deep Agents
