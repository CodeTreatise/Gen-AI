---
title: "12.10 OpenAI Agents SDK"
---

# 12.10 OpenAI Agents SDK

## What is OpenAI Agents SDK
- Production replacement for Swarm
- OpenAI's official agent framework
- Lightweight and opinionated
- Built-in tracing and debugging
- Seamless OpenAI integration

## Core Concepts
- Agent: LLM + instructions + tools
- Runner: Execution engine
- Handoffs: Agent-to-agent transfer
- Guardrails: Input/output validation
- Tracing: Built-in observability

## Agent Definition
- name and instructions
- tools list
- handoff_description
- output_type (structured outputs)
- model selection

## Tool Integration
- @function_tool decorator
- Automatic schema generation
- Return type handling
- Async tool support
- Error handling

## Handoff Patterns
- handoff() function
- Context preservation
- Multi-agent workflows
- Circular handoffs
- Handoff conditions

## Guardrails
- Input guardrails
- Output guardrails
- Content validation
- Safety filters
- Custom guardrail functions

## Runner Execution
- Runner.run() for single turn
- Runner.run_sync() for sync
- Streaming responses
- Session management
- Context accumulation

## Tracing and Debugging
- Built-in tracing UI
- OPENAI_AGENT_TRACING env
- Custom trace handlers
- Performance monitoring
- Error tracking

## Use Cases
- OpenAI-centric applications
- Simple multi-agent handoffs
- Production deployments
- API-first services
- Lightweight agent microservices

