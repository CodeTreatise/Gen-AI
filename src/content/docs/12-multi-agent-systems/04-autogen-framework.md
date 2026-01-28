---
title: "12.4 AutoGen 0.4+ Framework"
---

# 12.4 AutoGen 0.4+ Framework

## What is AutoGen 0.4+
- Microsoft's completely rewritten multi-agent framework
- Async-first architecture (major change from 0.2)
- Event-driven message passing
- Modular package system
- Production-grade design

## Package Architecture
- autogen-agentchat: High-level agent abstractions
- autogen-core: Low-level messaging infrastructure
- autogen-ext: Extensions (tools, models, RAG)
- autogen-magentic-one: Magentic-One implementation
- Migration from pyautogen (0.2)

## AssistantAgent (0.4+ style)
- Async-first design
- Model client configuration
- Tool registration (function tools)
- System messages
- Response streaming

## Core Agent Types
- AssistantAgent: LLM-powered responses
- CodingAssistantAgent: Code execution
- UserProxyAgent: Human input handling
- ToolUseAssistantAgent: Tool-focused agent
- Custom agent patterns

## Agent Runtime
- SingleThreadedAgentRuntime
- DistributedAgentRuntime (for scaling)
- Message subscriptions
- Topic-based routing
- Event handlers

## Tool Integration
- @function_tool decorator
- Tool schemas
- Return type annotations
- Async tool functions
- Tool error handling

## Teams & Group Chat
- SelectorGroupChat: LLM selects next speaker
- RoundRobinGroupChat: Fixed order rotation
- Swarm: Agent handoffs
- MagenticOneGroupChat: Magentic-One pattern
- Custom team types

## Termination Conditions
- TextMentionTermination
- MaxMessageTermination
- TimeoutTermination
- Custom termination logic
- Combining conditions (AND/OR)

## GraphFlow
- Workflow graph definition
- Nodes as agents
- Conditional edges
- Parallel execution
- State transitions
