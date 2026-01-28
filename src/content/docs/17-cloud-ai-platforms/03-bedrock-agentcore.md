---
title: "17.3 AWS Bedrock AgentCore"
---

# 17.3 AWS Bedrock AgentCore

## Introduction

Amazon Bedrock AgentCore is AWS's production-grade infrastructure for deploying and managing AI agents at scale. Launched in 2025, AgentCore provides composable services that enable agents to take actions across tools and data with the right permissions and controls.

## What is AgentCore

- Production-grade agent infrastructure
- Composable services that work together or independently
- No infrastructure management required
- Secure at scale
- Unified platform for agent lifecycle

## AgentCore Runtime

- Serverless agent deployment
- Automatic scaling based on demand
- Session management
- Execution isolation
- No cold starts with warm pools
- Request queuing for bursts

## AgentCore Gateway

- Unified tool access
- Third-party service connections
- Pre-built connectors for popular services
- API management
- Rate limiting and retry logic
- OAuth integration for external APIs

## AgentCore Memory

- Intelligent context retention
- Cross-session memory for personalization
- Conversation history summarization
- Long-term context retention
- Memory management and cleanup
- Configurable retention policies

## AgentCore Identity

- AWS authentication integration
- Third-party service authentication
- OAuth 2.0 integration
- Secure credential management via Secrets Manager
- Token refresh handling
- Role assumption for actions

## AgentCore Tools

- Browser capability for web browsing
- Code Interpreter for code execution
- File operations (S3 integration)
- Custom tool development
- Tool chaining
- Sandbox execution environment

## AgentCore Observability

- Comprehensive monitoring
- Step-by-step trace visualization
- Debugging tools
- Performance metrics
- Cost tracking per agent/session
- CloudWatch integration

## AgentCore Evaluations (Preview)

- Continuous quality scoring
- Agent benchmarking
- Regression detection
- Quality gates for deployment
- A/B testing capabilities
- Comparison across agent versions

## AgentCore Policy (Preview)

- Fine-grained action control
- Safety boundaries
- Approval workflows for sensitive actions
- Audit trails
- Maximum actions per session
- Cost limits per session

## Best Practices

- Start with simple agents and iterate
- Implement proper error handling
- Enable observability from day one
- Use gradual rollout with aliases
- Set up quality gates before production
- Configure appropriate safety boundaries

## Hands-on Exercises

1. Build a data analysis agent with Code Interpreter
2. Create a multi-tool agent with browser and file operations
3. Implement cross-session memory for personalized agents
4. Configure comprehensive observability with CloudWatch

## Summary

AgentCore transforms agent development from prototype to production with:
- Runtime for serverless deployment
- Gateway for unified connections
- Memory for context retention
- Identity for secure authentication
- Observability for monitoring
- Evaluations for quality assurance
- Policy for governance and control

## Next Steps

- [AWS AI/ML Services](04-aws-ai-ml-services.md)
- [Model Deployment](10-model-deployment.md)
- [Responsible AI & Governance](14-responsible-ai.md)
