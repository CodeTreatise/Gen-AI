---
title: "Integration & Scenario Testing"
---

# Integration & Scenario Testing

## Introduction

Unit tests verify individual parts. Integration tests verify that parts work together — that the agent correctly orchestrates tool calls, manages state across turns, and produces coherent end-to-end results. Scenario tests go further by defining realistic user workflows and edge cases.

Together, these tests answer the question every agent developer needs answered: "Does this agent actually solve the user's problem?"

### What We'll Cover

- End-to-end agent flow testing patterns
- Multi-step workflow validation
- Mocking external services at integration boundaries
- State verification across conversation turns
- Scenario-based testing with common, edge, and failure cases
- Performance scenario testing

### Prerequisites

- Unit testing agent components (Lesson 01)
- Mocking AI responses (Lesson 02)
- Agent architecture with tools, memory, and state

---

## End-to-end agent flow testing

An end-to-end (E2E) test runs the full agent pipeline: receive input → reason → call tools → process results → produce output. We mock only the LLM, letting everything else run for real.

### The E2E testing pattern

```python
# agent_app/travel_agent.py
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext

@dataclass
class TravelDeps:
    budget: float
    currency: str = "USD"

@dataclass
class TravelResult:
    destination: str
    estimated_cost: float
    itinerary: list[str]

travel_agent = Agent(
    "openai:gpt-4o",
    deps_type=TravelDeps,
    output_type=TravelResult,
    instructions="You are a travel planning assistant.",
)

@travel_agent.tool
def search_flights(ctx: RunContext[TravelDeps], origin: str, destination: str) -> str:
    """Search for available flights."""
    return f"Flight {origin} → {destination}: ${ctx.deps.budget * 0.4:.0f}"

@travel_agent.tool
def search_hotels(ctx: RunContext[TravelDeps], city: str, nights: int) -> str:
    """Search for hotels in a city."""
    per_night = ctx.deps.budget * 0.1
    return f"Hotel in {city}: ${per_night:.0f}/night × {nights} nights = ${per_night * nights:.0f}"
```

```python
# tests/test_travel_e2e.py
import pytest
from pydantic_ai.models.test import TestModel
from agent_app.travel_agent import travel_agent, TravelDeps, TravelResult


@pytest.fixture
def test_deps():
    return TravelDeps(budget=2000.0, currency="USD")


class TestTravelAgentE2E:
    
    def test_full_planning_flow(self, test_deps):
        """Complete agent run produces valid TravelResult."""
        with travel_agent.override(model=TestModel()):
            result = travel_agent.run_sync(
                "Plan a trip from NYC to Paris for 5 nights",
                deps=test_deps,
            )
        
        # Verify structured output
        assert isinstance(result.output, TravelResult)
        assert result.output.destination  # Not empty
        assert result.output.estimated_cost >= 0
        assert len(result.output.itinerary) > 0
    
    def test_budget_flows_through_tools(self, test_deps):
        """Budget from deps is used in tool calculations."""
        with travel_agent.override(model=TestModel()):
            result = travel_agent.run_sync(
                "Find flights and hotels",
                deps=test_deps,
            )
        
        # The agent completed without errors, meaning tools
        # successfully accessed deps.budget
        assert result.output is not None
```

**Output:**
```
$ pytest tests/test_travel_e2e.py -v
tests/test_travel_e2e.py::TestTravelAgentE2E::test_full_planning_flow PASSED
tests/test_travel_e2e.py::TestTravelAgentE2E::test_budget_flows_through_tools PASSED
======================== 2 passed in 0.08s =========================
```

---

## Multi-step workflow validation

Agents often require multiple LLM calls with tool executions between them. We validate that the full sequence produces correct state at each step.

```python
# agent_app/research_agent.py
from dataclasses import dataclass, field

@dataclass
class ResearchState:
    query: str = ""
    sources: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    step: str = "init"

class ResearchAgent:
    """Multi-step research agent with explicit state tracking."""
    
    def __init__(self, llm_client, search_fn, summarize_fn):
        self.llm = llm_client
        self.search = search_fn
        self.summarize = summarize_fn
    
    def run(self, query: str) -> ResearchState:
        state = ResearchState(query=query)
        
        # Step 1: Search for sources
        state.step = "searching"
        state.sources = self.search(query)
        
        if not state.sources:
            state.step = "no_results"
            state.summary = "No sources found for the query."
            return state
        
        # Step 2: Extract findings from each source
        state.step = "analyzing"
        for source in state.sources:
            response = self.llm.chat(messages=[
                {"role": "user", "content": f"Extract key findings from: {source}"}
            ])
            state.findings.append(response.choices[0].message.content)
        
        # Step 3: Summarize all findings
        state.step = "summarizing"
        state.summary = self.summarize(state.findings)
        state.step = "complete"
        
        return state
```

```python
# tests/test_research_workflow.py
import pytest
from unittest.mock import MagicMock
from agent_app.research_agent import ResearchAgent, ResearchState


@pytest.fixture
def mock_search():
    """Mock search that returns two sources."""
    search = MagicMock()
    search.return_value = [
        "Source 1: AI in healthcare",
        "Source 2: ML for diagnostics",
    ]
    return search


@pytest.fixture
def mock_summarize():
    """Mock summarizer."""
    summarize = MagicMock()
    summarize.return_value = "AI and ML are transforming healthcare diagnostics."
    return summarize


@pytest.fixture
def mock_llm():
    """Mock LLM with sequential findings."""
    llm = MagicMock()
    llm.chat.side_effect = [
        MagicMock(choices=[MagicMock(
            message=MagicMock(content="Finding 1: AI improves accuracy")
        )]),
        MagicMock(choices=[MagicMock(
            message=MagicMock(content="Finding 2: ML reduces diagnosis time")
        )]),
    ]
    return llm


class TestResearchWorkflow:
    
    def test_complete_workflow(self, mock_llm, mock_search, mock_summarize):
        """Full workflow produces complete state."""
        agent = ResearchAgent(mock_llm, mock_search, mock_summarize)
        state = agent.run("AI in healthcare")
        
        assert state.step == "complete"
        assert len(state.sources) == 2
        assert len(state.findings) == 2
        assert "healthcare" in state.summary.lower()
    
    def test_no_sources_found(self, mock_llm, mock_summarize):
        """Empty search results short-circuit the workflow."""
        empty_search = MagicMock(return_value=[])
        agent = ResearchAgent(mock_llm, empty_search, mock_summarize)
        
        state = agent.run("nonexistent topic")
        
        assert state.step == "no_results"
        assert state.sources == []
        assert state.findings == []
        assert "No sources found" in state.summary
        
        # LLM should never be called
        mock_llm.chat.assert_not_called()
    
    def test_llm_called_per_source(self, mock_llm, mock_search, mock_summarize):
        """LLM is called once per source for finding extraction."""
        agent = ResearchAgent(mock_llm, mock_search, mock_summarize)
        agent.run("test query")
        
        assert mock_llm.chat.call_count == 2  # One per source
    
    def test_state_transitions(self, mock_llm, mock_search, mock_summarize):
        """Verify state progresses through expected steps."""
        # Track state.step at each LLM call
        steps_seen = []
        original_chat = mock_llm.chat.side_effect
        
        state_ref = {}
        
        agent = ResearchAgent(mock_llm, mock_search, mock_summarize)
        result = agent.run("test")
        
        # Final state should be complete
        assert result.step == "complete"
```

**Output:**
```
$ pytest tests/test_research_workflow.py -v
tests/test_research_workflow.py::TestResearchWorkflow::test_complete_workflow PASSED
tests/test_research_workflow.py::TestResearchWorkflow::test_no_sources_found PASSED
tests/test_research_workflow.py::TestResearchWorkflow::test_llm_called_per_source PASSED
tests/test_research_workflow.py::TestResearchWorkflow::test_state_transitions PASSED
======================== 4 passed in 0.03s =========================
```

---

## Mocking external services

Integration tests often involve external APIs (databases, search engines, vector stores). We mock at the service boundary, not deep inside the agent:

```python
# agent_app/rag_agent.py
class RAGAgent:
    """Agent with retrieval-augmented generation."""
    
    def __init__(self, llm_client, vector_store, document_store):
        self.llm = llm_client
        self.vector_store = vector_store
        self.doc_store = document_store
    
    def answer(self, question: str) -> dict:
        # Step 1: Embed the question and find similar docs
        results = self.vector_store.similarity_search(
            query=question,
            top_k=3
        )
        
        if not results:
            return {"answer": "I don't have enough information.", "sources": []}
        
        # Step 2: Fetch full documents
        docs = [
            self.doc_store.get(doc_id=r["id"])
            for r in results
        ]
        
        # Step 3: Generate answer with context
        context = "\n\n".join(d["content"] for d in docs)
        response = self.llm.chat(messages=[
            {"role": "system", "content": f"Answer based on:\n{context}"},
            {"role": "user", "content": question},
        ])
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [d["title"] for d in docs],
        }
```

```python
# tests/test_rag_integration.py
import pytest
from unittest.mock import MagicMock
from agent_app.rag_agent import RAGAgent


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        {"id": "doc-1", "score": 0.95},
        {"id": "doc-2", "score": 0.87},
    ]
    return store


@pytest.fixture
def mock_doc_store():
    docs = {
        "doc-1": {"title": "Python Basics", "content": "Python is a high-level language."},
        "doc-2": {"title": "Python OOP", "content": "Classes organize code into objects."},
    }
    store = MagicMock()
    store.get.side_effect = lambda doc_id: docs[doc_id]
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Python is a high-level language that uses classes.")
        )]
    )
    return llm


class TestRAGIntegration:
    
    def test_full_rag_pipeline(self, mock_llm, mock_vector_store, mock_doc_store):
        """Complete RAG pipeline retrieves docs and generates answer."""
        agent = RAGAgent(mock_llm, mock_vector_store, mock_doc_store)
        result = agent.answer("What is Python?")
        
        assert "Python" in result["answer"]
        assert len(result["sources"]) == 2
        assert "Python Basics" in result["sources"]
    
    def test_no_similar_docs(self, mock_llm, mock_doc_store):
        """No matching docs returns fallback answer."""
        empty_store = MagicMock()
        empty_store.similarity_search.return_value = []
        
        agent = RAGAgent(mock_llm, empty_store, mock_doc_store)
        result = agent.answer("Unrelated question")
        
        assert "don't have enough information" in result["answer"]
        assert result["sources"] == []
        mock_llm.chat.assert_not_called()
    
    def test_context_passed_to_llm(self, mock_llm, mock_vector_store, mock_doc_store):
        """Retrieved document content is included in the LLM prompt."""
        agent = RAGAgent(mock_llm, mock_vector_store, mock_doc_store)
        agent.answer("What is Python?")
        
        # Verify the system message contains document content
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        
        assert "high-level language" in system_msg
        assert "Classes organize" in system_msg
```

---

## Scenario-based testing

Scenarios model realistic usage patterns. We categorize them into four types:

### Common scenarios (happy path)

```python
# tests/test_scenarios_common.py
import pytest
from unittest.mock import MagicMock
from agent_app.simple_agent import SimpleAgent


class TestCommonScenarios:
    """Scenarios that happen most often in production."""
    
    def test_simple_question_answer(self):
        """User asks a direct question, agent answers."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="The capital of France is Paris."
            ))]
        )
        agent = SimpleAgent(mock_llm)
        result = agent.respond("What's the capital of France?")
        assert "Paris" in result
    
    def test_follow_up_question(self):
        """User asks a follow-up that requires conversation context."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content="Paris has a population of 2.1 million."
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content="Its most famous landmark is the Eiffel Tower."
            ))]),
        ]
        agent = SimpleAgent(mock_llm)
        agent.respond("Tell me about Paris")
        result = agent.respond("What's its most famous landmark?")
        
        # Verify context was sent with second call
        second_call = mock_llm.chat.call_args_list[1]
        messages = second_call.kwargs.get("messages") or second_call[0][0]
        assert len(messages) >= 3  # user1, assistant1, user2
```

### Edge case scenarios

```python
# tests/test_scenarios_edge.py
import pytest
from unittest.mock import MagicMock
from agent_app.simple_agent import SimpleAgent


class TestEdgeCaseScenarios:
    """Unusual but valid inputs."""
    
    def test_empty_message(self):
        """Agent handles empty user input gracefully."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="I didn't receive a message. How can I help?"
            ))]
        )
        agent = SimpleAgent(mock_llm)
        result = agent.respond("")
        assert result  # Returns something, doesn't crash
    
    def test_very_long_message(self):
        """Agent handles messages near context window limits."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Acknowledged."))
        ])
        agent = SimpleAgent(mock_llm)
        long_msg = "x " * 50000  # ~100k characters
        result = agent.respond(long_msg)
        assert result is not None
    
    def test_unicode_and_special_characters(self):
        """Agent processes unicode, emoji, and special characters."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="I see emoji in your message! 👍"
            ))]
        )
        agent = SimpleAgent(mock_llm)
        result = agent.respond("Hello! 🌍 café résumé naïve 日本語")
        assert result is not None
```

### Failure scenarios

```python
# tests/test_scenarios_failure.py
import pytest
from unittest.mock import MagicMock
from agent_app.simple_agent import SimpleAgent


class TestFailureScenarios:
    """What happens when things go wrong."""
    
    def test_llm_returns_empty_content(self):
        """Agent handles empty LLM response."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        agent = SimpleAgent(mock_llm)
        result = agent.respond("Hello")
        assert result == ""  # Empty but no crash
    
    def test_llm_raises_rate_limit(self):
        """Agent propagates rate limit errors."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("Rate limit exceeded")
        
        agent = SimpleAgent(mock_llm)
        with pytest.raises(Exception, match="Rate limit"):
            agent.respond("Hello")
    
    def test_llm_returns_none_content(self):
        """Agent handles None content from LLM."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        agent = SimpleAgent(mock_llm)
        result = agent.respond("Hello")
        # Depending on implementation: None, empty string, or error
        # The test documents the actual behavior
        assert result is None
```

### Performance scenarios

```python
# tests/test_scenarios_performance.py
import pytest
import time
from unittest.mock import MagicMock
from agent_app.research_agent import ResearchAgent


class TestPerformanceScenarios:
    """Verify agent performance under realistic loads."""
    
    def test_handles_many_sources(self):
        """Agent processes a large number of search results."""
        mock_search = MagicMock(return_value=[
            f"Source {i}: Content about topic {i}"
            for i in range(20)
        ])
        mock_summarize = MagicMock(return_value="Summary of 20 sources")
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Finding"))]
        )
        
        agent = ResearchAgent(mock_llm, mock_search, mock_summarize)
        
        start = time.time()
        state = agent.run("broad query")
        elapsed = time.time() - start
        
        assert state.step == "complete"
        assert len(state.findings) == 20
        assert elapsed < 1.0  # Should be fast with mocks
    
    def test_rapid_sequential_queries(self):
        """Agent handles many queries in succession."""
        mock_search = MagicMock(return_value=["Source 1"])
        mock_summarize = MagicMock(return_value="Summary")
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Finding"))]
        )
        
        agent = ResearchAgent(mock_llm, mock_search, mock_summarize)
        
        for i in range(50):
            state = agent.run(f"Query {i}")
            assert state.step == "complete"
```

---

## Organizing scenario tests

Structure scenario tests by category for clarity:

```
tests/
├── conftest.py                  # Shared fixtures
├── test_unit/                   # Unit tests (fast)
│   ├── test_tools.py
│   ├── test_memory.py
│   └── test_parsers.py
├── test_integration/            # Integration tests
│   ├── test_rag_pipeline.py
│   └── test_workflow.py
└── test_scenarios/              # Scenario tests
    ├── test_common.py           # Happy path scenarios
    ├── test_edge_cases.py       # Unusual inputs
    ├── test_failures.py         # Error handling
    └── test_performance.py      # Load and timing
```

Mark scenario categories with pytest markers:

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "scenario_common: common usage scenarios")
    config.addinivalue_line("markers", "scenario_edge: edge case scenarios")
    config.addinivalue_line("markers", "scenario_failure: failure scenarios")
    config.addinivalue_line("markers", "scenario_perf: performance scenarios")
```

```bash
# Run only common scenarios
pytest -m scenario_common

# Run everything except performance tests
pytest -m "not scenario_perf"
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Mock only the LLM in E2E tests | Validates real integration between agent components |
| Test state at each workflow step | Catches mid-flow bugs that final assertions miss |
| Group scenarios by type | Run fast scenarios in CI, slow ones in nightly builds |
| Use `side_effect` for multi-step mocking | Simulates realistic multi-turn agent conversations |
| Test both the result and the process | Assert on output AND verify which tools were called |
| Define scenario names as real user stories | "User asks follow-up" not "test_scenario_2" |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Mocking too many components in integration tests | Mock only external boundaries (LLM, APIs) |
| Testing only success paths | Add failure and edge case scenario categories |
| Ignoring state between turns | Assert on conversation history, not just final output |
| Fragile assertion on exact text | Assert on structure and key terms, not exact wording |
| No performance baselines | Time tests with mocks to catch algorithmic slowdowns |
| Duplicating setup across tests | Use `conftest.py` fixtures with scope management |

---

## Hands-on Exercise

### Your Task

Build a scenario test suite for a customer support agent that handles order lookups and refund requests.

### Requirements

1. Define the `SupportAgent` class with:
   - `lookup_order(order_id)` tool
   - `process_refund(order_id, reason)` tool
   - Conversation history tracking
2. Write at least 8 scenario tests across all four categories:
   - **Common** (3): Order lookup, refund request, follow-up question
   - **Edge** (2): Invalid order ID, very long reason text
   - **Failure** (2): LLM timeout, tool function raises exception
   - **Performance** (1): 20 sequential conversations
3. Use pytest markers to categorize scenarios
4. Use fixtures for shared mock setup

### Expected Result

All 8+ tests pass, organized by scenario category with descriptive names.

<details>
<summary>💡 Hints (click to expand)</summary>

- `side_effect` can also raise exceptions: `mock.side_effect = TimeoutError("Connection timeout")`
- For follow-up tests, call `agent.respond()` twice and check that the second call includes history
- Performance test: loop 20 times, each with fresh mock `side_effect` list
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# support_agent.py
class SupportAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.history = []
        self.tools = {
            "lookup_order": self.lookup_order,
            "process_refund": self.process_refund,
        }
    
    def lookup_order(self, order_id: str) -> str:
        if not order_id.startswith("ORD-"):
            return "Error: Invalid order ID"
        return f"Order {order_id}: Delivered on 2025-01-15"
    
    def process_refund(self, order_id: str, reason: str) -> str:
        return f"Refund initiated for {order_id}: {reason[:100]}"
    
    def respond(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        response = self.llm.chat(messages=self.history)
        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})
        return content
```

```python
# test_support_scenarios.py
import pytest
from unittest.mock import MagicMock
from support_agent import SupportAgent

@pytest.fixture
def agent():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Default response"))]
    )
    return SupportAgent(mock_llm)

# Common scenarios
@pytest.mark.scenario_common
def test_order_lookup(agent):
    result = agent.lookup_order("ORD-123")
    assert "Delivered" in result

@pytest.mark.scenario_common
def test_refund_request(agent):
    result = agent.process_refund("ORD-123", "Item damaged")
    assert "Refund initiated" in result

@pytest.mark.scenario_common
def test_follow_up_includes_history(agent):
    agent.respond("First message")
    agent.respond("Follow-up")
    assert len(agent.history) == 4  # 2 user + 2 assistant

# Edge cases
@pytest.mark.scenario_edge
def test_invalid_order_id(agent):
    result = agent.lookup_order("INVALID")
    assert "Error" in result

@pytest.mark.scenario_edge
def test_long_refund_reason(agent):
    reason = "x" * 10000
    result = agent.process_refund("ORD-1", reason)
    assert "Refund initiated" in result

# Failure scenarios
@pytest.mark.scenario_failure
def test_llm_timeout():
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = TimeoutError("Connection timeout")
    agent = SupportAgent(mock_llm)
    with pytest.raises(TimeoutError):
        agent.respond("Hello")

@pytest.mark.scenario_failure
def test_empty_llm_response():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=""))]
    )
    agent = SupportAgent(mock_llm)
    result = agent.respond("Hello")
    assert result == ""

# Performance
@pytest.mark.scenario_perf
def test_sequential_conversations():
    for _ in range(20):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Response"))]
        )
        agent = SupportAgent(mock_llm)
        agent.respond("Test")
        assert len(agent.history) == 2
```
</details>

### Bonus Challenges

- [ ] Add a scenario that tests tool chaining (lookup then refund)
- [ ] Create a `pytest.ini` with marker definitions and run categories selectively
- [ ] Add a scenario with 100 conversation turns to test memory limits

---

## Summary

✅ E2E tests run the full agent pipeline, mocking only the LLM at the boundary

✅ Multi-step workflow tests verify state transitions at each stage of agent processing

✅ External services (vector stores, databases, APIs) are mocked at the integration boundary

✅ Scenario tests organized by type (common, edge, failure, performance) ensure comprehensive coverage

✅ Pytest markers (`@pytest.mark.scenario_common`) enable selective test execution

**Next:** [Deterministic & Regression Testing](./04-deterministic-regression-testing.md)

---

## Further Reading

- [pytest Markers Documentation](https://docs.pytest.org/en/stable/how-to/mark.html) - Custom markers for test organization
- [Pydantic AI Testing](https://ai.pydantic.dev/testing/) - E2E testing with TestModel
- [unittest.mock — side_effect](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect) - Simulating sequences

<!-- 
Sources Consulted:
- pytest Documentation: https://docs.pytest.org/en/stable/
- Pydantic AI Testing: https://ai.pydantic.dev/testing/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
- LangSmith Evaluation Concepts: https://docs.langchain.com/langsmith/evaluation-concepts
-->
