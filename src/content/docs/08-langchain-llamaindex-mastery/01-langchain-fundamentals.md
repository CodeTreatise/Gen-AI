---
title: "8.1 LangChain Fundamentals"
---

# 8.1 LangChain Fundamentals

## LangChain Architecture Overview
- Core components
- Modular design philosophy
- Separation of concerns
- Extensibility patterns
- Package structure: langchain, langchain-core, langchain-community

## Installation and Setup
- pip install langchain
- Provider-specific packages: langchain-openai, langchain-anthropic
- Version management
- Environment configuration
- API key setup

## LangChain Expression Language (LCEL)
- Pipe operator (|) for chaining
- Runnable protocol
- invoke, batch, stream methods
- Async support: ainvoke, astream
- Declarative composition

## Core Abstractions
- Runnables as building blocks
- RunnablePassthrough for data passing
- RunnableLambda for custom logic
- RunnableParallel for concurrent execution
- RunnableBranch for conditional logic

## Model Wrappers
- ChatOpenAI, ChatAnthropic, ChatGoogle
- Model configuration: temperature, max_tokens
- Streaming with callbacks
- Token counting
- Model fallbacks

## Debugging and Tracing
- LangSmith integration
- Verbose mode
- Callback handlers
- Tracing execution
- Performance monitoring
