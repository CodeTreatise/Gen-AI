---
title: "Unit 10: Function Calling & Tool Use"
---

# Unit 10: Function Calling & Tool Use

## Overview & Importance

Function calling enables AI to take actions in the real world — not just generate text. By defining tools that AI can invoke, you create systems that can search databases, call APIs, perform calculations, and interact with external systems autonomously.

Function calling transforms AI from:
- Text generator → Action taker
- Static responses → Dynamic interactions
- Isolated system → Integrated component

## Prerequisites

- API integration proficiency (Unit 3)
- Prompt engineering knowledge (Unit 5)
- Understanding of JSON and schemas
- Backend development concepts

## Learning Objectives

By the end of this unit, you will be able to:
- Define tools and functions for AI to use
- Write JSON schemas for function parameters
- Implement strict mode for guaranteed schema adherence
- Handle function call requests from AI across providers
- Execute functions and return results (including multimodal)
- Manage multi-step and parallel function calling conversations
- Handle errors and safety refusals in function execution
- Configure tool choice modes and control behavior
- Design safe and effective tool sets
- Work with thinking models and their special requirements
- Build and integrate MCP servers
- Combine function calling with built-in platform tools
- Implement real-time function calling for voice agents

## Real-world Applications

- AI assistants that book appointments
- Chatbots that query databases
- AI that controls smart home devices
- Automated customer service with order lookup
- AI-powered data analysis tools
- Code assistants that run tests
- AI that sends emails or messages
- Research assistants that search the web

## Market Demand & Relevance

- Function calling is key to production AI systems
- Enables automation of complex workflows
- High-value skill for enterprise AI projects
- Differentiates interactive AI from simple chatbots
- Growing demand as companies seek AI automation
- Essential for building AI agents
- Foundation for agentic workflows and autonomous systems
- Required for enterprise tool integrations

## References & Resources

### Official Documentation
- OpenAI Function Calling Guide
  - https://platform.openai.com/docs/guides/function-calling
- OpenAI Structured Outputs Guide
  - https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Tools Documentation
  - https://platform.openai.com/docs/guides/tools
- Anthropic Tool Use Documentation
  - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Google Gemini Function Calling
  - https://ai.google.dev/gemini-api/docs/function-calling
- Google Gemini Tools Overview
  - https://ai.google.dev/gemini-api/docs/tools

### Model Context Protocol (MCP)
- MCP Official Documentation
  - https://modelcontextprotocol.io/introduction
- MCP Specification
  - https://spec.modelcontextprotocol.io
- MCP TypeScript SDK
  - https://github.com/modelcontextprotocol/typescript-sdk
- MCP Python SDK
  - https://github.com/modelcontextprotocol/python-sdk
- MCP Server Examples
  - https://github.com/modelcontextprotocol/servers

### JSON Schema
- JSON Schema Official Documentation
  - https://json-schema.org/docs
- OpenAPI Schema Specification
  - https://spec.openapis.org/oas/v3.0.3

### Provider SDKs
- OpenAI Python SDK
  - https://github.com/openai/openai-python
- OpenAI Node.js SDK
  - https://github.com/openai/openai-node
- Google GenAI Python SDK
  - https://github.com/google/generative-ai-python
- Anthropic Python SDK
  - https://github.com/anthropics/anthropic-sdk-python

### Tutorials & Cookbooks
- OpenAI Cookbook - Function Calling
  - https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
- OpenAI Cookbook - Fine-tuning for Function Calling
  - https://cookbook.openai.com/examples/fine_tuning_for_function_calling
- Gemini Cookbook
  - https://github.com/google-gemini/cookbook
- Anthropic Cookbook
  - https://github.com/anthropics/anthropic-cookbook

### Advanced Topics
- LLGuidance (Grammar Constraints)
  - https://github.com/guidance-ai/llguidance
- Lark Parser (for Custom Grammars)
  - https://lark-parser.readthedocs.io
- OpenAI Agents SDK
  - https://platform.openai.com/docs/guides/agents

### Research & Background
- Gorilla: Large Language Model Connected with Massive APIs
  - https://arxiv.org/abs/2305.15334
- ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
  - https://arxiv.org/abs/2307.16789
- ReAct: Synergizing Reasoning and Acting in Language Models
  - https://arxiv.org/abs/2210.03629
