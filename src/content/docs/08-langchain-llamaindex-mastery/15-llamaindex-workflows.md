---
title: "8.15 LlamaIndex Workflows"
---

# 8.15 LlamaIndex Workflows

## Introduction

LlamaIndex Workflows represent a paradigm shift from DAG-based pipelines to event-driven orchestration. Released in 2024, Workflows provide a flexible, Python-native approach to building complex AI applications with support for branching, loops, parallel execution, and human-in-the-loop patterns. This is now the recommended approach for production LlamaIndex applications.

## Learning Objectives

By the end of this section, you will be able to:
- Understand the event-driven workflow architecture
- Build workflows using @step decorators and Events
- Implement branching, loops, and parallel execution
- Manage state across workflow steps
- Handle human-in-the-loop patterns
- Deploy workflows to production with llama_deploy

## What are Workflows?

### From Query Pipelines to Workflows

**Old Approach (Query Pipelines)**:
- Import `QueryPipeline` from `llama_index.core.query_pipeline`
- Create pipeline by chaining components: `QueryPipeline(chain=[retriever, reranker, synthesizer])`
- DAG-based, limited control flow

**New Approach (Workflows)**:
- Import `Workflow`, `step`, `Event` from `llama_index.core.workflow`
- Create class extending `Workflow`
- Define async methods with `@step` decorator
- Steps accept and emit `Event` objects
- Supports arbitrary control flow including loops and branches

### Why Workflows?

| Feature | Query Pipelines | Workflows |
|---------|-----------------|-----------|
| Control Flow | DAG only | Arbitrary (loops, branches) |
| Execution | Synchronous | Async-first |
| Human-in-loop | Limited | Native support |
| State Management | Basic | Rich context |
| Debugging | Moderate | Excellent |
| Deployment | Manual | llama_deploy native |

### Core Components

1. **Workflow class**: Container for steps
2. **@step decorator**: Marks processing functions
3. **Events**: Data carriers between steps
4. **StartEvent / StopEvent**: Entry and exit points
5. **Context**: Shared state across steps

## Core Concepts

### Events

Events are Pydantic models that carry data between steps:

**Dependencies**:
- `Event` from `llama_index.core.workflow`

**Event Class Definition**:
- Create classes extending `Event`
- Define fields as typed attributes

**Example Events**:
- `QueryEvent`: Contains `query` (str) field
- `RetrieveEvent`: Contains `query` (str) and `nodes` (list[NodeWithScore]) fields
- `SynthesizeEvent`: Contains `query` (str) and `context` (str) fields

**Purpose**:
- Type-safe data carriers between workflow steps
- Enable Pydantic validation on data flow

### Steps

Steps are async functions decorated with `@step`:

**Dependencies**:
- `Workflow`, `step`, `StartEvent`, `StopEvent` from `llama_index.core.workflow`

**Step Definition**:
- Create a class extending `Workflow`
- Define async methods with `@step` decorator
- Accept an event type as the first parameter (after self)
- Return an event type (annotated in return type hint)

**Example Step**:
- `async def process(self, ev: StartEvent) -> StopEvent:`
- Extract data from input event: `query = ev.query`
- Process using instance resources: `result = await self.llm.acomplete(query)`
- Return output event: `return StopEvent(result=result.text)`

### StartEvent and StopEvent

**StartEvent**:
- Import from `llama_index.core.workflow`
- Entry point for workflow execution
- Keyword arguments passed to `workflow.run()` become event attributes
- Example: `workflow.run(query="What is AI?", context="...")` creates StartEvent with those fields

**StopEvent**:
- Import from `llama_index.core.workflow`
- Exit point that terminates workflow
- Must specify the `result` parameter with the final output
- Example: `return StopEvent(result={"answer": answer, "sources": sources})`

## Building Workflows

### Basic RAG Workflow

**Dependencies**:
- `Workflow`, `step`, `StartEvent`, `StopEvent`, `Event` from `llama_index.core.workflow`
- `VectorStoreIndex`, `Settings` from `llama_index.core`
- `OpenAI` from `llama_index.llms.openai`

**Custom Event (RetrieveEvent)**:
- `query`: String field for the user query
- `nodes`: List field for retrieved document nodes

**RAGWorkflow Class**:

*Constructor (__init__)*:
- Accept `index: VectorStoreIndex` parameter
- Call `super().__init__(**kwargs)`
- Store `self.index = index`
- Get LLM from `Settings.llm`

*retrieve Step*:
- Input: `StartEvent`, Output: `RetrieveEvent`
- Extract query: `query = ev.query`
- Create retriever: `self.index.as_retriever(similarity_top_k=5)`
- Await retrieval: `nodes = await retriever.aretrieve(query)`
- Return `RetrieveEvent(query=query, nodes=nodes)`

*synthesize Step*:
- Input: `RetrieveEvent`, Output: `StopEvent`
- Join node texts: `context = "\n\n".join([n.text for n in ev.nodes])`
- Build prompt with context and question
- Await LLM completion: `response = await self.llm.acomplete(prompt)`
- Return `StopEvent(result=response.text)`

**Usage**:
- Create instance: `workflow = RAGWorkflow(index=my_index)`
- Run: `result = await workflow.run(query="What is quantum computing?")`

### Running Workflows

**Dependencies**:
- `asyncio` module for running async code

**Workflow Initialization**:
- Create instance with required parameters: `workflow = RAGWorkflow(index=my_index, timeout=60, verbose=True)`
- `timeout`: Maximum execution time in seconds
- `verbose`: Enable detailed logging

**Async Execution**:
- Define async main function: `async def main():`
- Await workflow run: `result = await workflow.run(query="Explain neural networks")`
- Print or process result

**Running from Sync Context**:
- Use `asyncio.run(main())` to execute the async function

## Branches and Loops

### Conditional Branching

**Event Definitions**:
- `QueryTypeEvent`: Contains `query` (str) and `query_type` (str)
- `FactualEvent`: Contains `query` (str) for factual queries
- `CreativeEvent`: Contains `query` (str) for creative queries

**BranchingWorkflow Steps**:

*classify Step*:
- Input: `StartEvent`, Output: `QueryTypeEvent`
- Analyze query to determine type
- Simple example: Check for keywords like "explain" or "what is"
- Return `QueryTypeEvent` with classified `query_type`

*route Step*:
- Input: `QueryTypeEvent`, Output: `FactualEvent | CreativeEvent` (union type)
- Use return type union to indicate branching
- Based on `query_type`, return appropriate event
- If "factual": return `FactualEvent(query=ev.query)`
- Else: return `CreativeEvent(query=ev.query)`

*handle_factual Step*:
- Input: `FactualEvent`, Output: `StopEvent`
- Process factual queries with RAG or lookup
- Return `StopEvent(result=response)`

*handle_creative Step*:
- Input: `CreativeEvent`, Output: `StopEvent`
- Process creative queries with generation
- Return `StopEvent(result=response)`

### Loops for Refinement

**Event Definitions**:
- `RefineEvent`: Contains `query`, `answer`, `iteration` fields
- `QualityCheckEvent`: Contains `query`, `answer`, `iteration`, `quality_score` fields

**RefinementWorkflow Steps**:

*generate Step*:
- Input: `StartEvent | RefineEvent` (union type accepts multiple event types)
- Output: `QualityCheckEvent`
- Check event type with `isinstance(ev, StartEvent)`
- For StartEvent: Initialize `iteration = 0`, generate initial answer
- For RefineEvent: Get existing iteration, refine the answer
- Evaluate quality of the answer
- Return `QualityCheckEvent` with updated iteration count and quality score

*check_quality Step*:
- Input: `QualityCheckEvent`, Output: `RefineEvent | StopEvent`
- Exit conditions: `quality_score > 0.8` or `iteration >= 3`
- If conditions met: return `StopEvent(result=ev.answer)`
- Otherwise: return `RefineEvent` to loop back to generate step

**Loop Behavior**:
- The `generate` step accepts both initial and loop-back events
- The `check_quality` step controls whether to continue or exit
- Creates a refinement cycle until quality threshold is met

## Managing State

### Using Context

The Context object provides shared state across steps:

**Dependencies**:
- `Context` from `llama_index.core.workflow`

**Context as Parameter**:
- Add `ctx: Context` as parameter after `self` in step methods
- Example: `async def step1(self, ctx: Context, ev: StartEvent):`

**Storing Data (step1)**:
- Use `await ctx.set(key, value)` to store data
- Example: `await ctx.set("user_id", ev.user_id)`
- Example: `await ctx.set("start_time", time.time())`
- Data persists across all workflow steps

**Retrieving Data (step2)**:
- Use `await ctx.get(key)` to retrieve stored data
- Example: `user_id = await ctx.get("user_id")`
- Example: `start_time = await ctx.get("start_time")`

**Use Cases**:
- Tracking execution metadata (timing, user info)
- Sharing data between non-adjacent steps
- Accumulating results across the workflow

### Instance Attributes

**Constructor Pattern**:
- Override `__init__` with custom parameters (e.g., `config: dict`)
- Always call `super().__init__(**kwargs)`
- Initialize instance attributes:
  - `self.config = config` for configuration
  - `self.llm = OpenAI(model="gpt-4o")` for LLM client
  - `self.cache = {}` for caching results

**Accessing in Steps**:
- Instance attributes available via `self` in all steps
- Example: Check cache with `if ev.query in self.cache:`
- Return cached result if exists
- Otherwise, call LLM and update cache

**Benefits**:
- Share resources (LLM clients, indexes) across steps
- Implement cross-step caching
- Store configuration and state

## Concurrent Execution

### Parallel Steps

**Dependencies**:
- `Event` from `llama_index.core.workflow`
- `gather` from `asyncio` for parallel execution

**Event Definitions**:
- `SearchResultEvent`: Contains `source` (str) and `results` (list)
- `AggregatedEvent`: Contains `all_results` (dict)

**ParallelSearchWorkflow Steps**:

*dispatch Step*:
- Input: `StartEvent`, Output: `list[SearchResultEvent]`
- Extract query from event
- Create list of async tasks for parallel sources:
  - `self.search_web(query)`
  - `self.search_database(query)`
  - `self.search_vector_store(query)`
- Use `await gather(*tasks)` to run all in parallel
- Return list of `SearchResultEvent` objects, one per source
- Returning a list triggers parallel processing

*aggregate Step*:
- Input: `SearchResultEvent` (called multiple times), Output: `AggregatedEvent | None`
- Use `ctx.collect_events(ev, [SearchResultEvent] * 3)` to gather all events
- If `events is None`: Still waiting for more events, return `None`
- Once all collected: Build dict of all results by source
- Return `AggregatedEvent(all_results=all_results)`

*synthesize Step*:
- Input: `AggregatedEvent`, Output: `StopEvent`
- Merge results from all sources
- Return final combined result

### Rate Limiting

**Dependencies**:
- `asyncio` module for `Semaphore`

**Constructor Setup**:
- Accept `max_concurrent: int = 5` parameter
- Create semaphore: `self.semaphore = asyncio.Semaphore(max_concurrent)`

**Rate-Limited Step**:
- Use `async with self.semaphore:` context manager
- Limits concurrent executions to `max_concurrent`
- Perform expensive operation inside the context
- Other executions wait until semaphore is available

**Use Cases**:
- Respecting API rate limits
- Controlling resource usage (memory, connections)
- Preventing overwhelming external services

## Human-in-the-Loop

### Interactive Workflows

**Dependencies**:
- `InputRequiredEvent`, `HumanResponseEvent` from `llama_index.core.workflow`

**Event Definitions**:
- `ReviewEvent`: Contains `content` (str) and `requires_approval` (bool)

**InteractiveWorkflow Steps**:

*generate Step*:
- Input: `StartEvent`, Output: `ReviewEvent`
- Generate content using `self.generate_content(ev.query)`
- Determine if review needed with `self.needs_review(content)`
- Return `ReviewEvent(content=content, requires_approval=...)`

*request_review Step*:
- Input: `ReviewEvent`, Output: `InputRequiredEvent | StopEvent`
- If no approval needed: return `StopEvent(result=ev.content)` directly
- If approval needed: return `InputRequiredEvent(prefix="...prompt...")`
- The `prefix` contains the message shown to the human reviewer
- Returning `InputRequiredEvent` pauses workflow for human input

*process_review Step*:
- Input: `HumanResponseEvent`, Output: `StopEvent`
- Check human response (e.g., `ev.response.lower() == "yes"`)
- Return appropriate result with status and content

**Usage Pattern**:
- Create workflow and get handler: `handler = workflow.run(query=...)`
- Iterate with `async for event in handler.stream_events():`
- Check for `InputRequiredEvent` instances
- Collect human input and send response: `handler.ctx.send_event(HumanResponseEvent(response=...))`
- Await final result: `result = await handler`

## Deployment with llama_deploy

### Basic Deployment

**Dependencies**:
- `deploy_workflow`, `WorkflowServiceConfig` from `llama_deploy`

**Deployment Configuration**:
- Create workflow instance: `RAGWorkflow(index=my_index)`
- Configure service with `WorkflowServiceConfig`:
  - `host`: Server address (e.g., "0.0.0.0")
  - `port`: Service port (e.g., 8000)
  - `service_name`: Identifier for the service (e.g., "rag-workflow")

**Deployment Call**:
- Call `deploy_workflow(workflow=..., config=...)`
- Starts the workflow as a network service

### Multi-Service Deployment

**Dependencies**:
- `deploy_core`, `ControlPlaneConfig`, `SimpleMessageQueueConfig`, `WorkflowServiceConfig` from `llama_deploy`

**Control Plane Deployment**:
- Call `await deploy_core(...)` with:
  - `control_plane_config`: `ControlPlaneConfig()` for orchestration
  - `message_queue_config`: `SimpleMessageQueueConfig()` for inter-service messaging

**Deploying Multiple Workflows**:
- Deploy each workflow separately with `await deploy_workflow(...)`
- Each gets unique `service_name` (e.g., "rag", "summary")
- All workflows connect to the same control plane

**Benefits**:
- Microservices architecture for workflows
- Independent scaling per service
- Central orchestration via control plane

### Client Usage

**Dependencies**:
- `LlamaDeployClient` from `llama_deploy`

**Client Setup**:
- Create client instance: `client = LlamaDeployClient()`
- Client connects to the deployed control plane

**Calling Deployed Workflows**:
- Use `await client.run(service="service-name", query="...")`
- Specify which service to call by name
- Pass query parameters as keyword arguments
- Returns the workflow result

## Hands-On Exercise

### Build a Multi-Step Research Workflow

Create a workflow that:
1. Classifies the research topic
2. Retrieves from multiple sources (parallel)
3. Evaluates source quality
4. Requests human approval for low-quality sources
5. Synthesizes final report

**Exercise Requirements**:

*Dependencies*:
- `Workflow`, `step`, `StartEvent`, `StopEvent`, `Event`, `Context` from `llama_index.core.workflow`

*Event Definitions*:
- `TopicEvent`: Contains `topic` (str) and `category` (str)
- `SourceEvent`: Contains `source` (str), `content` (str), `quality` (float)

*ResearchWorkflow Class*:

*Steps to Implement*:
1. `classify_topic`: Accepts `StartEvent`, returns `TopicEvent` with category classification
2. `gather_sources`: Accepts `TopicEvent`, returns `list[SourceEvent]` for parallel source gathering
3. `evaluate_quality`: Accepts `SourceEvent` and `Context`, implements quality check with optional human-in-loop for low scores
4. `synthesize`: Aggregates all sources and returns `StopEvent` with final report

## Key Takeaways

1. **Workflows are event-driven**, not DAG-based
2. **@step decorator** marks processing functions
3. **Events carry data** between steps (Pydantic models)
4. **Context provides shared state** across steps
5. **Parallel execution** via returning multiple events
6. **Human-in-loop** via InputRequiredEvent
7. **llama_deploy** for production deployment

## Additional Resources

- [LlamaIndex Workflows Documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/)
- [Workflows Examples](https://docs.llamaindex.ai/en/stable/examples/workflow/)
- [llama_deploy Documentation](https://docs.llamaindex.ai/en/stable/module_guides/llama_deploy/)
- [Migrating from Query Pipelines](https://docs.llamaindex.ai/en/stable/understanding/workflows/migration/)

---

*Next: [8.16 Production & Observability](16-production-observability.md)*
