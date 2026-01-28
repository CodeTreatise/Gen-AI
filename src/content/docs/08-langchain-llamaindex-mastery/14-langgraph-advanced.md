---
title: "8.14 LangGraph Advanced Patterns"
---

# 8.14 LangGraph Advanced Patterns

## Introduction

Building on LangGraph fundamentals, this section covers advanced patterns essential for production agents: human-in-the-loop workflows, durable execution with checkpointing, subgraph composition, and sophisticated streaming patterns. These capabilities distinguish production-ready agents from prototypes.

> **Version Note (January 2026)**: This section reflects LangGraph v1.0.6, langgraph-checkpoint v4.0.0, and langgraph-sdk v0.3.3. Key updates include custom encryption at rest support and checkpointer type validation at compile time.

## Learning Objectives

By the end of this section, you will be able to:
- Implement human-in-the-loop patterns with interrupts
- Set up persistence for durable execution
- Design modular agents with subgraphs
- Build sophisticated streaming applications
- Handle tools and parallel execution
- Implement advanced control flow patterns

## Human-in-the-Loop

### The Interrupt Pattern

LangGraph's `interrupt` function pauses execution and waits for human input:

**Dependencies**:
- `interrupt`, `Command` from `langgraph.types`
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `MemorySaver` from `langgraph.checkpoint.memory`

**State Definition (ApprovalState)**:
- `request`: String containing the request to approve
- `approved`: Boolean indicating approval status
- `reviewer_notes`: String for human feedback

**request_approval Node**:
- Extract the `request` from state
- Call `interrupt()` with a dict containing:
  - `question`: The approval prompt message
  - `options`: List of choices like `["approve", "reject", "modify"]`
- The `interrupt()` call pauses graph execution and returns control
- Process the response: set `approved` based on `response["decision"] == "approve"`
- Capture optional `reviewer_notes` from `response.get("notes", "")`

**process_approved Node**:
- Check `state["approved"]` to determine outcome
- Return formatted request with "APPROVED:" or "REJECTED:" prefix

**Graph Building**:
- Create `StateGraph(ApprovalState)` and add both nodes
- Connect: `START` → "get_approval" → "process" → `END`

**Checkpointer Requirement**:
- Create `MemorySaver()` instance for in-memory persistence
- Pass `checkpointer=memory` to `workflow.compile()`
- Interrupts require a checkpointer to save/restore state

### Resuming After Interrupt

**Thread Configuration**:
- Create a config dict with `{"configurable": {"thread_id": "unique-id"}}`
- Thread ID identifies the conversation/session for checkpointing

**Starting Execution**:
- Call `app.invoke(initial_state, config)` to begin
- Execution pauses at the `interrupt()` call
- Returns the interrupt request to the caller

**Resuming with Human Decision**:
- Import `Command` from `langgraph.types`
- Create `Command(resume={"decision": "approve", "notes": "Looks good!"})`
- Call `app.invoke(command, config)` with the same thread config
- Execution continues from where it paused
- Returns the final result with updated state

### interrupt_before and interrupt_after

**Interrupt Before a Node**:
- Pass `interrupt_before=["node_name"]` to `workflow.compile()`
- Execution pauses *before* the specified node runs
- Allows review/modification of state before processing

**Interrupt After a Node**:
- Pass `interrupt_after=["node_name"]` to `workflow.compile()`
- Execution pauses *after* the specified node completes
- Allows review of node output before continuing

**Both Options**:
- Require a checkpointer to be configured
- Accept lists to interrupt at multiple nodes
- Can be combined in the same compile() call

## Checkpointing and Persistence

### Memory-Based (Development)

**Setup**:
- Import `MemorySaver` from `langgraph.checkpoint.memory`
- Create instance: `memory = MemorySaver()`
- Pass to compile: `app = workflow.compile(checkpointer=memory)`

**Thread Isolation**:
- Each `thread_id` maintains independent state
- Create separate configs for different users/sessions
- Example: `config1 = {"configurable": {"thread_id": "user-alice"}}`
- State is lost when the process terminates (in-memory only)

### SQLite Persistence

**Dependencies**:
- `SqliteSaver` from `langgraph.checkpoint.sqlite`
- `sqlite3` module for database connection

**Setup**:
- Create SQLite connection: `conn = sqlite3.connect("checkpoints.db", check_same_thread=False)`
- The `check_same_thread=False` allows multi-threaded access
- Create saver: `sqlite_saver = SqliteSaver(conn)`
- Compile with saver: `app = workflow.compile(checkpointer=sqlite_saver)`

**Benefits**:
- Persistent storage survives process restarts
- Simple setup for development and testing
- Good for single-server deployments

### PostgreSQL (Production)

**Dependencies**:
- `PostgresSaver` from `langgraph.checkpoint.postgres`
- `ConnectionPool` from `psycopg_pool`

**Setup**:
- Create connection pool with connection string:
  - `pool = ConnectionPool(conninfo="postgresql://user:pass@localhost:5432/langgraph")`
- Create saver: `postgres_saver = PostgresSaver(pool)`
- Initialize database tables: `postgres_saver.setup()`
- Compile: `app = workflow.compile(checkpointer=postgres_saver)`

**Benefits**:
- Production-grade persistence with connection pooling
- Supports horizontal scaling across multiple servers
- ACID-compliant transaction support

### Accessing Checkpoint History

**Getting Checkpoints**:
- Create config with thread_id: `config = {"configurable": {"thread_id": "my-thread"}}`
- Call `app.get_state_history(config)` to get checkpoint iterator
- Convert to list: `checkpoints = list(app.get_state_history(config))`

**Checkpoint Properties**:
- `checkpoint.metadata.get('step')` - The execution step number
- `checkpoint.values` - The state values at that checkpoint
- `checkpoint.config` - Configuration including checkpoint_id

**Use Cases**:
- Debug execution by examining state at each step
- Audit trail for compliance
- Understanding how state evolved over time

### Time Travel Debugging

**Getting a Specific Checkpoint ID**:
- Access checkpoint_id from history: `checkpoints[2].config["configurable"]["checkpoint_id"]`

**Creating Replay Config**:
- Build config with both `thread_id` and `checkpoint_id`:
  - `{"configurable": {"thread_id": "my-thread", "checkpoint_id": checkpoint_id}}`

**Replaying from Checkpoint**:
- Call `app.invoke(new_input, replay_config)`
- Execution resumes from the specified checkpoint state
- Allows "time travel" to explore different execution paths

## Subgraphs and Composition

### Creating Reusable Subgraphs

**Dependencies**:
- `StateGraph`, `START`, `END` from `langgraph.graph`

**Subgraph State Definition (ResearchState)**:
- `query`: String - the search query
- `findings`: List - accumulated research findings

**Subgraph Nodes**:

*search_web*:
- Returns dict with findings containing web search result
- Appends formatted result like `f"Web result for: {state['query']}"`

*search_database*:
- Gets existing findings from state (default empty list)
- Returns findings list with new database result appended

**Subgraph Construction**:
- Create `StateGraph(ResearchState)`
- Add both search nodes
- Connect: `START` → "web_search" → "db_search" → `END`
- Compile with `.compile()` to create reusable subgraph

### Embedding Subgraphs

**Main Graph State (MainState)**:
- `user_query`: String - the original user query
- `research_results`: List - results from subgraph
- `final_answer`: String - synthesized answer

**Transform Functions**:

*prepare_research (Input Transform)*:
- Takes `MainState`, returns `ResearchState`
- Maps `user_query` to subgraph's `query` field
- Initializes empty `findings` list

*process_research (Output Transform)*:
- Takes subgraph result `ResearchState`
- Returns dict mapping `findings` to `research_results`

**Adding Subgraph as Node**:
- Use `add_node()` with additional parameters:
  - First arg: node name ("research")
  - Second arg: the compiled subgraph
  - `input=prepare_research`: transform function for input
  - `output=process_research`: transform function for output

**Main Graph Flow**:
- Connect: `START` → "research" (subgraph) → "synthesize" → `END`

## Streaming Patterns

### Stream Node Outputs

**Basic Streaming**:
- Call `app.stream(initial_state)` to get event iterator
- Each event is a dict with node name as key

**Processing Events**:
- Extract node name: `node_name = list(event.keys())[0]`
- Get output: `output = event[node_name]`
- Print or process progressively as each node completes

### Stream LLM Tokens

**LLM Setup**:
- Create `ChatOpenAI(model="gpt-4o", streaming=True)` with streaming enabled

**Token-Level Streaming**:
- Use `async for event in app.astream(..., stream_mode="messages")`
- Events arrive as tuples of `(message, metadata)`
- Access token content: `msg.content`
- Print with `end=""` and `flush=True` for real-time display

**Use Case**:
- Real-time typing effect in chat interfaces
- Progressive response display

### Stream Mode Options

**Available Modes**:

*stream_mode="values"* (default):
- Returns full state after each node completes
- Best for tracking overall progress

*stream_mode="updates"*:
- Returns only the changes/delta from each node
- More efficient for large state objects

*stream_mode="debug"*:
- Returns detailed execution information
- Includes internal events and timing data
- Useful for debugging and profiling

### Real-time UI Updates

**Dependencies**:
- `asyncio` for async operations
- `FastAPI` from fastapi for web framework
- `StreamingResponse` from fastapi.responses
- `json` for serialization

**Server-Sent Events (SSE) Endpoint**:
- Create async generator function that:
  - Iterates over `app.astream(input)` events
  - Yields formatted SSE data: `f"data: {json.dumps(event)}\n\n"`

**FastAPI Streaming Response**:
- Define POST endpoint (e.g., `/chat/stream`)
- Return `StreamingResponse` with:
  - The async generator function
  - `media_type="text/event-stream"` for SSE protocol

**Use Case**:
- Real-time chat interfaces
- Progressive UI updates
- Live agent execution feedback

## Tool Integration

### Tool Nodes

**Dependencies**:
- `ToolNode` from `langgraph.prebuilt`
- `tool` decorator from `langchain_core.tools`

**Defining Tools**:
- Use `@tool` decorator on functions
- Add docstrings (used as tool descriptions for LLM)
- Example tools: `search(query: str)` and `calculator(expression: str)`
- Return string results

**Creating ToolNode**:
- Collect tools in a list: `tools = [search, calculator]`
- Create node: `tool_node = ToolNode(tools)`

**Integration with Graph**:
- Add to workflow: `workflow.add_node("tools", tool_node)`
- ToolNode automatically handles tool call routing and execution

### Handling ToolMessages

**Dependencies**:
- `ToolMessage` from `langchain_core.messages`

**Processing Tool Results**:
- Get messages from state: `messages = state["messages"]`
- Access last message: `last_message = messages[-1]`
- Check type: `isinstance(last_message, ToolMessage)`
- If true, extract content: `last_message.content`
- Return processed result in state update dict

### Parallel Tool Execution

**Dependencies**:
- `Send` from `langgraph.types`

**Fan-Out Pattern**:
- Define a router function that returns a list of `Send` objects
- Each `Send(node_name, state_dict)` dispatches to a parallel execution
- Get pending tools from state: `tool_calls = state["pending_tools"]`
- Return list comprehension: `[Send("execute_tool", {"tool": tc}) for tc in tool_calls]`

**Graph Integration**:
- Use `add_conditional_edges(source_node, router_function)`
- Each `Send` creates a parallel branch to the target node

## Advanced Control Flow

### Loops with Exit Conditions

**Loop Router Function**:
- Check multiple exit conditions:
  - `iterations >= 5` - Maximum iteration limit reached
  - `quality_score > 0.9` - Quality threshold met
- Return "exit" to terminate loop
- Return "continue" to loop back

**Conditional Edge Setup**:
- Use `add_conditional_edges(node, router, path_map)`
- Path map: `{"continue": "improve", "exit": END}`
- "continue" maps back to the same node (creates cycle)
- "exit" maps to `END` to terminate

### Dynamic Fan-Out with Send

**Dependencies**:
- `Send` from `langgraph.types`

**State Definition (ParallelState)**:
- `items`: List of strings to process
- `results`: List with `add` reducer to collect results from parallel workers

**Distribute Work Function**:
- Returns list of `Send` objects, one per item
- Each `Send("worker", {"item": item})` dispatches work
- Enables dynamic parallelism based on input data size

**Graph Setup**:
- Use `add_conditional_edges(START, distribute_work)`
- All `Send` objects execute in parallel

### Map-Reduce Pattern

**Map Phase (map_items)**:
- Returns list of `Send` objects with item and index
- `Send("process_item", {"item": item, "index": i})`
- Each item processed independently in parallel

**Process Phase (process_single_item)**:
- Individual node processing each item
- Results collected via reducer on results field

**Reduce Phase (reduce_results)**:
- Collects all results from state
- Aggregates into `final_output`
- Returns combined/summarized result

**Graph Wiring**:
- `add_conditional_edges("start", map_items)` - fan-out
- `add_node("process_item", process_single_item)` - parallel workers
- `add_edge("process_item", "reduce")` - all workers feed to reduce
- `add_node("reduce", reduce_results)` - aggregation

## Production Patterns

### Error Handling with Retries

**Dependencies**:
- `retry`, `stop_after_attempt`, `wait_exponential` from `tenacity`

**Retry Decorator Configuration**:
- `@retry(...)` decorator wraps the function
- `stop=stop_after_attempt(3)` - Maximum 3 retry attempts
- `wait=wait_exponential()` - Exponential backoff between retries

**Exception Handling Inside**:
- `try/except` block for different error types
- Raise `TransientError` to trigger retry
- Catch `PermanentError` and return error state (no retry)

**Use Case**:
- Network failures, rate limits, temporary API issues

### Timeout Management

**Recursion Limit**:
- Pass config to invoke: `app.invoke(state, {"recursion_limit": 50})`
- Limits maximum number of steps/cycles
- Prevents infinite loops in cyclic graphs

**Async Timeout**:
- Import `asyncio`
- Wrap invoke in `asyncio.wait_for(coroutine, timeout=30.0)`
- Catch `asyncio.TimeoutError` to handle timeout
- Useful for enforcing SLA response times

## Hands-On Exercise

### Build an Approval Workflow

Create a LangGraph application that:
1. Generates a document draft
2. Pauses for human review
3. Incorporates feedback if rejected
4. Persists state across sessions

**Exercise Requirements**:

*Dependencies*:
- `StateGraph` from `langgraph.graph`
- `interrupt` from `langgraph.types`
- `SqliteSaver` from `langgraph.checkpoint.sqlite`

*State Definition (DocumentState)*:
- `topic`: String - the document topic
- `draft`: String - the generated draft content
- `feedback`: String - human reviewer feedback
- `approved`: Boolean - approval status
- `version`: Integer - draft version number

*Nodes to Implement*:
1. `generate_draft` - Creates initial or revised draft based on topic/feedback
2. `review_draft` - Uses `interrupt()` to pause for human review input
3. `incorporate_feedback` - Revises draft based on feedback

*Features to Include*:
- Conditional routing based on approval status
- Loop back to draft generation if rejected
- SQLite persistence for durable state across sessions

## Key Takeaways

1. **Interrupts** enable true human-in-the-loop workflows
2. **Checkpointing** provides durability and time travel
3. **Subgraphs** enable modular, reusable agent design
4. **Streaming** supports real-time UI applications
5. **Send API** enables parallel execution patterns
6. **Tool nodes** simplify tool integration

## Additional Resources

- [LangGraph Human-in-the-Loop Guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Persistence Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Streaming Guide](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [LangGraph Templates](https://github.com/langchain-ai/langgraph-templates)

---

*Next: [8.15 LlamaIndex Workflows](15-llamaindex-workflows.md)*
