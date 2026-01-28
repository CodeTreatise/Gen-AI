---
title: "8.13 LangGraph Fundamentals"
---

# 8.13 LangGraph Fundamentals

## Introduction

LangGraph is the low-level orchestration framework that powers LangChain agents. Released in 2024, it provides fine-grained control over agent workflows with stateful execution, persistence, and streaming capabilities. Understanding LangGraph is essential for building production-ready AI agents.

## Learning Objectives

By the end of this section, you will be able to:
- Understand LangGraph's architecture and design philosophy
- Build stateful workflows using StateGraph
- Define nodes, edges, and conditional routing
- Implement state management with reducers
- Execute and stream graph outputs
- Visualize and debug LangGraph applications

## What is LangGraph

### The Evolution of LangChain Agents

**Traditional Approach (AgentExecutor)**:
- Import `AgentExecutor` and `create_openai_functions_agent` from `langchain.agents`
- This approach now uses LangGraph under the hood

**Modern Approach (Direct LangGraph)**:
- Import `StateGraph`, `START`, and `END` from `langgraph.graph`
- Import `create_react_agent` from `langgraph.prebuilt` for pre-built agent patterns
- Provides more fine-grained control over agent workflows

### Core Philosophy

LangGraph is inspired by Google's **Pregel** and **Apache Beam** - systems designed for:
- **Durable execution**: Survive failures and resume
- **Stateful processing**: Maintain context across steps
- **Cyclic workflows**: Support loops and complex control flow
- **Human-in-the-loop**: Pause and resume with human input

### When to Use LangGraph

| Use Case | LangGraph | Simple Chains |
|----------|-----------|---------------|
| Multi-step reasoning | ✅ | ❌ |
| Human approval workflows | ✅ | ❌ |
| Complex branching logic | ✅ | ❌ |
| State persistence | ✅ | ❌ |
| Simple Q&A | ❌ | ✅ |
| Linear pipelines | ❌ | ✅ |

## Core Concepts

### StateGraph

The fundamental building block - a graph where state flows between nodes:

**Dependencies**:
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `TypedDict`, `Annotated` from `typing`
- `add` from `operator` (used as a reducer)

**State Schema Definition**:
- Create a class extending `TypedDict` to define the state schema (e.g., `AgentState`)
- Use `Annotated[list, add]` to define a reducer that appends to lists (for `messages`)
- Add simple fields like `current_step` (str) and `results` (dict) without reducers

**Graph Creation**:
- Instantiate `StateGraph(AgentState)` passing the state schema class

### Nodes

Functions that process and update state:

**Node Function Structure**:
- Accept `state: AgentState` as input parameter
- Return a `dict` containing state updates (only fields to change)

**Research Node Example**:
- Extract the last message from `state["messages"][-1]` as the query
- Call `perform_research(query)` to get results
- Return dict with `results` containing research data and `current_step` set to "analyze"

**Analyze Node Example**:
- Retrieve research from `state["results"].get("research", "")`
- Call `analyze_data(research)` to process
- Return dict merging existing results with new analysis using `{**state["results"], "analysis": analysis}`
- Update `current_step` to "complete"

**Adding Nodes to Graph**:
- Call `graph.add_node("node_name", node_function)` for each node
- First argument is the node identifier, second is the function reference

### Edges

Define the flow between nodes:

**Simple Edge Definition**:
- Use `graph.add_edge(source, target)` to connect nodes
- `START` constant marks the entry point of the graph
- `END` constant marks the exit point of the graph

**Flow Example**:
- `graph.add_edge(START, "research")` - execution begins at the research node
- `graph.add_edge("research", "analyze")` - research flows to analyze
- `graph.add_edge("analyze", END)` - analyze completes the graph

### Conditional Edges

Route based on state:

**Router Function**:
- Define a function that accepts `state: AgentState` and returns a string
- The returned string determines which path to take
- Check state conditions (e.g., `state["current_step"]`) to decide routing
- Return values like "research", "analyze", or "end" based on logic

**Adding Conditional Edges**:
- Call `graph.add_conditional_edges(source_node, router_function, path_map)`
- First argument: the node after which routing occurs (e.g., "research")
- Second argument: the router function reference
- Third argument: dictionary mapping router return values to target nodes
  - Keys are the possible return values from the router function
  - Values are target node names or `END` constant
  - Example: `{"research": "research", "analyze": "analyze", "end": END}`

## Building State Graphs

### Complete Example: Research Agent

**Dependencies**:
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `add_messages` from `langgraph.graph.message` (built-in message reducer)
- `TypedDict`, `Annotated` from `typing`
- `ChatOpenAI` from `langchain_openai`
- `HumanMessage`, `AIMessage` from `langchain_core.messages`

**State Definition (ResearchState)**:
- `messages`: List with `add_messages` reducer for conversation history
- `research_topic`: String for the topic being researched
- `sources`: List to store gathered sources
- `final_report`: String for the final output

**LLM Initialization**:
- Create `ChatOpenAI(model="gpt-4o")` instance for LLM calls

**Node Functions**:

*gather_sources*:
- Extract `research_topic` from state
- Call LLM with a `HumanMessage` asking for 5 key sources
- Return dict with the response added to messages and sources parsed by splitting on newlines

*synthesize_report*:
- Retrieve `sources` and `research_topic` from state
- Call LLM to write a report based on sources and topic
- Return dict with response message and `final_report` set to response content

*quality_check* (Router Function):
- Get the current `final_report` from state (default empty string)
- Return "improve" if report length is less than 200 characters
- Return "complete" otherwise

**Graph Construction**:
- Create `StateGraph(ResearchState)` instance as `workflow`
- Add nodes: `workflow.add_node("gather", gather_sources)` and `workflow.add_node("synthesize", synthesize_report)`
- Add edge from `START` to "gather"
- Add edge from "gather" to "synthesize"
- Add conditional edges from "synthesize" using `quality_check` router:
  - "improve" → loops back to "gather" (enables refinement cycle)
  - "complete" → `END` (terminates workflow)

**Compilation**:
- Call `workflow.compile()` to create the executable `app`

## State Management

### Understanding Reducers

Reducers define how state updates merge with existing state:

**Dependencies**:
- `add` from `operator` (built-in list concatenation)
- `Annotated` from `typing`

**State Field Types**:

*With `add` Reducer*:
- Syntax: `messages: Annotated[list, add]`
- Behavior: New values are appended to the existing list

*Without Reducer*:
- Syntax: `current_status: str`
- Behavior: New values completely replace the old value

*With Custom Reducer*:
- Syntax: `counters: Annotated[dict, merge_dicts]`
- Define a custom function that takes `left` (existing) and `right` (update) values
- Example `merge_dicts`: Returns `{**left, **right}` to merge dictionaries

### MessagesState for Chat Applications

**Using MessagesState**:
- Import `MessagesState` from `langgraph.graph`
- Extend `MessagesState` to create your custom state class
- `MessagesState` includes a pre-configured `messages` field with `add_messages` reducer
- Add additional fields as needed (e.g., `user_name: str`, `session_id: str`)

### Nested State

**Nested TypedDict Structure**:
- Define inner TypedDict classes for complex data (e.g., `ToolResult`)
- `ToolResult` fields: `tool_name` (str), `output` (str), `success` (bool)

**Main State with Nested Types**:
- `messages`: List with `add_messages` reducer
- `tool_results`: `list[ToolResult]` for typed list of nested objects
- `metadata`: `dict[str, any]` for flexible key-value storage

## Running Graphs

### Basic Execution

**Compilation**:
- Call `workflow.compile()` to get an executable `app`

**Invocation**:
- Call `app.invoke(initial_state_dict)` with all required state fields
- Initial state example includes:
  - `research_topic`: The topic string (e.g., "Quantum Computing in 2025")
  - `messages`: Empty list `[]`
  - `sources`: Empty list `[]`
  - `final_report`: Empty string `""`

**Accessing Results**:
- The result is the final state dict
- Access specific fields like `result["final_report"]`

### Streaming Execution

**Using stream() Method**:
- Call `app.stream(initial_state_dict)` instead of `invoke()`
- Returns an iterator of events

**Processing Stream Events**:
- Each event is a dict with node name as key and output as value
- Iterate with `for event in app.stream(...):`
- Extract node outputs: `for node_name, output in event.items():`
- Enables real-time progress display as each node completes

### Async Execution

**Dependencies**:
- Import `asyncio` for async operations

**Async Invoke**:
- Define async function using `async def`
- Use `await app.ainvoke(initial_state_dict)` for async invocation
- Returns the complete result after all nodes finish

**Async Streaming**:
- Use `async for event in app.astream(...):`
- Enables non-blocking iteration over stream events

**Running Async Code**:
- Use `asyncio.run(async_function())` to execute from synchronous context

## Visualization and Debugging

### Get Graph Structure

**Inspecting Graph**:
- Call `app.get_graph()` to retrieve the graph structure object

**Accessing Components**:
- `graph_dict.nodes` - List of all node names in the graph
- `graph_dict.edges` - List of all edge connections between nodes
- Useful for debugging and understanding workflow structure

### Mermaid Diagram

**Generating Mermaid Code**:
- Call `app.get_graph().draw_mermaid()` to get Mermaid diagram syntax
- Output can be rendered in any Mermaid-compatible viewer

**Saving as PNG Image**:
- Requires additional dependencies (graphviz, etc.)
- Import `Image` from `IPython.display`
- Call `app.get_graph().draw_mermaid_png()` to generate PNG data
- Wrap in `Image()` for Jupyter notebook display

### Verbose Mode

**Enabling Debug Logging**:
- Import `logging` module
- Call `logging.basicConfig(level=logging.DEBUG)` for verbose output

**Execution with Options**:
- Pass a second config dict to `invoke()` with execution options
- `recursion_limit`: Integer to limit maximum cycles (e.g., 25)
- Prevents infinite loops in graphs with cycles

## Hands-On Exercise

### Build a Content Moderation Agent

Create a LangGraph workflow that:
1. Takes user content as input
2. Analyzes for policy violations
3. Routes to human review if uncertain
4. Returns moderation decision

**Exercise Requirements**:

*Dependencies*:
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `TypedDict`, `Literal` from `typing`

*State Definition (ModerationState)*:
- `content`: String - the user content to moderate
- `confidence`: Float - confidence score of the analysis
- `decision`: String - the moderation decision
- `requires_review`: Boolean - whether human review is needed

*Functions to Implement*:
- `analyze_content(state)`: Analyze content and return confidence/decision updates
- `route_decision(state)`: Return routing string based on confidence threshold

*Graph Structure*:
- Build workflow connecting analysis → routing → decision nodes
- Use conditional edges to route uncertain content to human review

## Key Takeaways

1. **LangGraph is the foundation** for modern LangChain agents
2. **StateGraph** provides stateful, durable execution
3. **Nodes process state**, edges control flow
4. **Reducers** define how state updates merge
5. **Conditional edges** enable complex routing
6. **Visualization** helps debug complex workflows

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangChain Academy - LangGraph Course](https://academy.langchain.com/)
- [LangGraph Examples Repository](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

*Next: [8.14 LangGraph Advanced Patterns](14-langgraph-advanced.md)*
