---
title: "12.9 CrewAI Flows"
---

# 12.9 CrewAI Flows

## What are Flows
- Event-driven workflow orchestration
- Beyond simple crew execution
- State management across steps
- Complex conditional logic
- Production workflow patterns

## Flow Decorators
- @start() for entry points
- @listen() for event handlers
- @router() for conditional branching
- @persist() for state persistence
- @human_feedback() for HITL

## State Management
- Flow state classes
- Typed state with Pydantic
- State passing between methods
- State persistence strategies
- State versioning

## Flow Execution
- flow.kickoff() to start
- Event emission and handling
- Parallel step execution
- Error handling in flows
- Flow interruption and resume

## Conditional Routing
- Router return values
- Multiple execution paths
- Dynamic path selection
- Fallback routes
- Route validation

## Integration Patterns
- Crews within flows
- Multiple crew orchestration
- External API integration
- Database operations
- Async flow steps

## Advanced Features
- Nested flows
- Flow composition
- Long-running workflows
- Checkpoint and resume
- Flow monitoring

