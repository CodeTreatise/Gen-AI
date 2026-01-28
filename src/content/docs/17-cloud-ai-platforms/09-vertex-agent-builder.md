---
title: "17.9 Vertex AI Agent Builder & Agent Development Kit"
---

# 17.9 Vertex AI Agent Builder & Agent Development Kit

## Introduction

Vertex AI Agent Builder is Google Cloud's production platform for creating AI agents, now enhanced with the Agent Development Kit (ADK) for flexible, extensible agent development. The platform integrates with Gemini models, supports multi-agent orchestration, and provides enterprise-ready deployment options.

## Agent Builder Overview

- Production-ready agent platform
- Gemini model integration
- Multi-agent orchestration
- Enterprise deployment options
- Grounding with Google Search

## Agent Development Kit (ADK)

- Open-source Python framework
- Modular architecture
- Multi-agent workflows
- Extensible tool system
- State management

### ADK Core Components

- Agent class
  - Base agent definition
  - Instruction configuration
  - Tool registration
- Workflow agents
  - Sequential execution
  - Parallel processing
  - Conditional routing
- Tool integration
  - Built-in tools
  - Custom functions
  - MCP tool servers
- Session management
  - In-memory state
  - Persistent storage
  - Checkpoint/resume

## ADK Agents

- LlmAgent
  - LLM-powered agent
  - Custom instructions
  - Tool configuration
  - Model selection
- SequentialAgent
  - Multi-step workflows
  - State passing
  - Ordered execution
- ParallelAgent
  - Concurrent tasks
  - Result aggregation
  - Fan-out/fan-in
- LoopAgent
  - Iterative processing
  - Condition-based looping
  - Max iteration limits

## ADK Tool Types

- Built-in tools
  - google_search
  - code_execution
  - load_web_page
  - vertex_ai_search
- Custom functions
  - Python functions as tools
  - Parameter schemas
  - Return types
- MCP Tool Servers
  - External tool integration
  - Standard protocol
  - Cross-platform tools

## Multi-Agent Patterns

- Agent-to-agent communication
  - Direct handoff
  - Result aggregation
  - Context sharing
- Supervisor pattern
  - Main agent delegates
  - Sub-agent execution
  - Result synthesis
- Workflow orchestration
  - DAG-based flows
  - Conditional branching
  - Error handling

## Knowledge Grounding

- Vertex AI Search
  - Structured data
  - Unstructured documents
  - Website crawling
- Google Search grounding
  - Real-time information
  - Citation support
  - Fact verification
- Custom RAG
  - Vector search
  - Embedding models
  - Hybrid retrieval

## State Management

- Session state
  - Conversation context
  - User preferences
  - Task progress
- Artifact storage
  - Generated files
  - Intermediate results
  - Outputs
- Memory systems
  - Short-term context
  - Long-term recall
  - User profiles

## Evaluation & Testing

- ADK evaluation framework
  - Trajectory evaluation
  - Response quality
  - Tool usage accuracy
- Custom evaluators
  - Domain-specific metrics
  - Business rules
  - Quality thresholds
- Test runners
  - Unit tests
  - Integration tests
  - End-to-end scenarios

## Deployment Options

- Local development
  - ADK dev server
  - Interactive testing
  - Quick iteration
- Cloud Run deployment
  - Container-based
  - Auto-scaling
  - Serverless
- Vertex AI deployment
  - Managed endpoints
  - Enterprise SLAs
  - Monitoring
- Agent Engine (Reasoning Engine)
  - Managed agent hosting
  - Built-in scaling
  - Monitoring and logging

## Enterprise Integration

- IAM integration
  - Role-based access
  - Service accounts
  - Cross-project sharing
- VPC networking
  - Private endpoints
  - Service perimeter
  - Firewall rules
- Audit logging
  - Agent interactions
  - Tool executions
  - Security events

## Best Practices

- Agent design
  - Clear instruction scope
  - Appropriate tool selection
  - Error handling
- Multi-agent systems
  - Task decomposition
  - Clear handoff criteria
  - Result verification
- Production deployment
  - Gradual rollout
  - Monitoring setup
  - Cost optimization

## Hands-on Exercises

1. Build a research agent with ADK
2. Create multi-agent workflow (sequential + parallel)
3. Implement custom MCP tool integration
4. Deploy agent to Cloud Run

## Comparison with Other Platforms

| Feature | Agent Builder | AWS Bedrock Agents | Foundry Agent Service |
|---------|---------------|--------------------|-----------------------|
| Framework | ADK (Python) | Bedrock SDK | Foundry SDK |
| Multi-agent | Native support | AgentCore | Multi-agent |
| Open Source | ADK is OSS | Proprietary | Proprietary |
| Grounding | Google Search | S3, KB | Azure AI Search |
| Deployment | Cloud Run, Vertex | Lambda, EC2 | Azure Functions |

## Summary

| Component | Description |
|-----------|-------------|
| Agent Builder | Production agent platform |
| ADK | Open-source Python framework |
| Agents | Llm, Sequential, Parallel, Loop |
| Tools | Built-in, custom, MCP |
| State | Session, artifacts, memory |
| Deployment | Local, Cloud Run, Vertex |

## Next Steps

- [Multi-Cloud API Management](10-api-management.md)
- [MCP Integration](11-mcp-integration.md)
- [Performance Optimization](16-performance-optimization.md)
