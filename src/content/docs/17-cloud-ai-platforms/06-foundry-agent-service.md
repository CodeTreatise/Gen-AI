---
title: "17.6 Microsoft Foundry Agent Service"
---

# 17.6 Microsoft Foundry Agent Service

## Introduction

Microsoft Foundry Agent Service is the production-ready platform for building, deploying, and managing AI agents at enterprise scale. In General Availability since 2025, it provides infrastructure for multi-agent applications with built-in enterprise security, monitoring, and governance.

## Foundry Agent Service Overview

- Production agent platform (GA)
- Enterprise-grade security and compliance
- Multi-agent support
- Full lifecycle management
- Azure ecosystem integration

## Architecture

- Agent Definition layer
  - System instructions
  - Model selection
  - Tool configuration
- Conversation Layer
  - Thread management
  - Message history
  - Context handling
- Tool Layer
  - Code Interpreter
  - File Search
  - Custom Functions
  - External APIs
- Infrastructure
  - Auto-scaling
  - Session management
  - Monitoring

## Creating Agents

- Agent definition basics
  - Model selection (GPT-4o, Claude, etc.)
  - System instructions
  - Tool configuration
  - Metadata and tags
- Advanced configuration
  - Memory settings
  - Tool resources (vector stores)
  - Custom functions
  - Response format

## Agent Tools

- Code Interpreter
  - Python code execution
  - Data analysis
  - Visualization generation
  - File processing
- File Search
  - Vector store integration
  - Document retrieval
  - Semantic search
  - Citation handling
- Custom Functions
  - Function definitions
  - Parameter schemas
  - External API calls
  - Database queries
- External APIs
  - REST API integration
  - Authentication handling
  - Response processing

## Conversation Management

- Thread lifecycle
  - Thread creation
  - Message addition
  - Run execution
  - Response retrieval
- Streaming responses
  - Real-time output
  - Delta processing
  - Event handling
- Context management
  - Message history
  - File attachments
  - Metadata tracking

## Agent Evaluation

- Quality metrics
  - Task completion rate
  - Response relevance
  - Factual accuracy
- Performance tracking
  - Latency metrics
  - Token usage
  - Cost per conversation
- A/B testing
  - Agent variants
  - Traffic splitting
  - Comparison analysis
- Continuous improvement
  - Feedback loops
  - Instruction refinement
  - Tool optimization

## Deployment Options

- Foundry portal deployment
  - Visual configuration
  - Playground testing
  - One-click deployment
- SDK deployment
  - Programmatic creation
  - Version management
  - Configuration updates
- CI/CD integration
  - Infrastructure as code
  - GitHub Actions
  - Azure DevOps pipelines

## Integration with Microsoft Ecosystem

- Copilot Studio integration
  - Use Foundry agents in Copilot actions
  - Access Foundry Models
  - Enterprise governance
- Microsoft 365 Copilot
  - Teams integration
  - Word, Excel plugins
  - Outlook actions
- Power Platform
  - Power Automate connectors
  - Power Apps integration
  - Logic Apps workflows
- Dynamics 365
  - Customer service agents
  - Sales assistants
  - Field service support

## Best Practices

- Clear, specific instructions
- Proper error handling and timeouts
- Performance monitoring
- Token usage tracking
- Security configuration
- Gradual rollout with testing

## Hands-on Exercises

1. Build a customer service agent with file search
2. Create a data analysis agent with code interpreter
3. Implement multi-agent orchestration
4. Deploy with CI/CD and monitoring

## Summary

| Feature | Description |
|---------|-------------|
| Agent Creation | SDK-based with flexible configuration |
| Tools | Code interpreter, file search, custom functions |
| Threads | Managed conversation state |
| Streaming | Real-time response delivery |
| Evaluation | Built-in quality metrics |
| Integration | M365, Power Platform, Copilot Studio |

## Next Steps

- [Microsoft Foundry AI Services](07-foundry-ai-services.md)
- [Vertex AI Agent Builder](09-vertex-agent-builder.md)
- [Responsible AI & Governance](14-responsible-ai.md)
