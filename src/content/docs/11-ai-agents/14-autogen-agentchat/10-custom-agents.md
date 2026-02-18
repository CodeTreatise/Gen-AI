---
title: "Custom agents"
---

# Custom agents

## Introduction

Preset agents like `AssistantAgent` and `CodeExecutorAgent` cover many common workflows, but real-world applications often demand behavior that no off-the-shelf agent provides. Maybe you need an agent that queries a proprietary API, enforces domain-specific validation rules, or orchestrates a multi-step pipeline that doesn't map cleanly onto a model-plus-tools pattern. For these situations, AutoGen AgentChat exposes the `BaseChatAgent` abstract class — a minimal contract you implement to create agents that participate fully in any team or conversation.

This lesson walks you through the `BaseChatAgent` interface, builds progressively complex custom agents, and shows how to make them declarative and team-ready.

### What you'll learn

- When and why to build a custom agent instead of configuring a preset
- The `BaseChatAgent` abstract methods and the `Response` object
- Building a simple agent with streaming support
- Managing internal state across turns
- Integrating any LLM SDK as a custom model client
- Making custom agents declarative with the Component protocol
- Plugging custom agents into teams

### Prerequisites

- Familiarity with AutoGen AgentChat basics ([overview](./00-autogen-agentchat.md))
- Comfort with Python async/await and abstract base classes
- A working `autogen-agentchat` installation (`pip install autogen-agentchat`)

---

## When to build custom agents

Preset agents solve general problems. Custom agents solve *your* problems. Reach for a custom agent when:

| Scenario | Why a preset falls short |
|---|---|
| **Domain-specific logic** | You need validation, transformation, or routing that can't be expressed as a tool call. |
| **Non-OpenAI model integration** | You want to call Google Gemini, Anthropic Claude, or a self-hosted model through its native SDK rather than an OpenAI-compatible wrapper. |
| **Stateful multi-turn workflows** | The agent must track internal state (counters, accumulators, conversation history) across rounds in a group chat. |
| **Custom I/O formats** | The agent produces structured data, images, or side effects (database writes, webhook calls) that don't fit the default assistant pattern. |
| **Performance constraints** | You want a lightweight agent that skips model inference entirely — a rules-based responder, a lookup service, or a simple dispatcher. |

The guiding principle: if you find yourself fighting the configuration surface of a preset agent, it's time to subclass `BaseChatAgent`.

---

## The BaseChatAgent interface

Every agent in AutoGen AgentChat — preset or custom — inherits from `BaseChatAgent`. To create your own, you implement three things:

### Abstract methods

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage

class MyAgent(BaseChatAgent):

    @property
    def produced_message_types(self) -> list[type]:
        """Declare which message types this agent can produce."""
        ...

    async def on_messages(self, messages, cancellation_token) -> Response:
        """Handle incoming messages and return a Response."""
        ...

    async def on_reset(self, cancellation_token) -> None:
        """Reset the agent to its initial state."""
        ...
```

**`produced_message_types`** — A property that returns a tuple or list of message classes this agent can yield. Teams use this to understand what to expect. Most agents return `(TextMessage,)`.

**`on_messages(messages, cancellation_token)`** — The core method. It receives a list of messages from the team (or an empty list — more on that later) and returns a `Response` object.

**`on_reset(cancellation_token)`** — Called when the team or runtime resets. Use it to clear internal state — conversation history, counters, caches.

### The Response object

`Response` wraps the agent's final answer along with any intermediate messages produced during processing:

```python
Response(
    chat_message=TextMessage(content="Final answer", source=self.name),
    inner_messages=[
        TextMessage(content="Step 1 done", source=self.name),
        TextMessage(content="Step 2 done", source=self.name),
    ]
)
```

- **`chat_message`** — The message the team sees and routes. This is what other agents receive.
- **`inner_messages`** — Optional list of intermediate messages. These are logged and visible to observers, but not broadcast to the team as separate turns.

### Optional: streaming with on_messages_stream

If your agent produces output incrementally, override `on_messages_stream()`:

```python
async def on_messages_stream(self, messages, cancellation_token):
    # Yield intermediate messages as they're produced
    yield TextMessage(content="Working...", source=self.name)
    # Final yield must be a Response
    yield Response(chat_message=TextMessage(content="Done!", source=self.name))
```

If you don't implement `on_messages_stream()`, the default implementation wraps your `on_messages()` return value — so streaming is opt-in, not required.

---

## Building a simple custom agent

Let's build a `CountDownAgent` that counts down from a given number, yielding each step as a streamed message. This demonstrates both `on_messages` and `on_messages_stream`.

```python
import asyncio
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


class CountDownAgent(BaseChatAgent):
    """An agent that counts down from a given number."""

    def __init__(self, name: str, count: int = 3):
        super().__init__(name, description="A simple agent that counts down.")
        self._count = count

    @property
    def produced_message_types(self) -> tuple[type, ...]:
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token) -> Response:
        # Delegate to the streaming version and collect the final Response.
        response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(self, messages, cancellation_token):
        inner_messages: list[TextMessage] = []
        for i in range(self._count, 0, -1):
            msg = TextMessage(content=f"{i}...", source=self.name)
            inner_messages.append(msg)
            yield msg
        yield Response(
            chat_message=TextMessage(content="Done!", source=self.name),
            inner_messages=inner_messages,
        )

    async def on_reset(self, cancellation_token) -> None:
        pass  # No mutable state to reset.


async def main():
    agent = CountDownAgent("countdown", count=3)
    token = CancellationToken()

    # Non-streaming usage
    response = await agent.on_messages([], token)
    print(f"Chat message: {response.chat_message.content}")
    print(f"Inner messages: {[m.content for m in response.inner_messages]}")

    print("---")

    # Streaming usage
    async for msg in agent.on_messages_stream([], token):
        if isinstance(msg, Response):
            print(f"Final: {msg.chat_message.content}")
        else:
            print(f"Stream: {msg.content}")


asyncio.run(main())
```

**Output:**

```
Chat message: Done!
Inner messages: ['3...', '2...', '1...']
---
Stream: 3...
Stream: 2...
Stream: 1...
Final: Done!
```

Key takeaways:

- The constructor calls `super().__init__(name, description)`. The `description` parameter is critical — teams use it to decide when to select the agent.
- `on_messages` delegates to `on_messages_stream`, collecting the final `Response`. This is a common pattern that gives you streaming for free.
- `on_reset` is a no-op here because the agent has no mutable state that changes between calls.

---

## Custom agents with state

Real agents often accumulate state. Consider an `ArithmeticAgent` that performs running calculations in a `SelectorGroupChat`. It tracks its own message history because `on_messages` may be called with an **empty messages list** — this happens when the team selects the agent again before any other agent has spoken.

```python
import asyncio
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


class ArithmeticAgent(BaseChatAgent):
    """An agent that performs arithmetic operations on a running total."""

    def __init__(self, name: str, operation: str = "add", operand: int = 1):
        super().__init__(
            name,
            description=f"Performs '{operation}' with operand {operand} on the current value.",
        )
        self._operation = operation
        self._operand = operand
        self._history: list[TextMessage] = []
        self._current_value: int = 0

    @property
    def produced_message_types(self) -> tuple[type, ...]:
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token) -> Response:
        # Append any new messages to our internal history.
        self._history.extend(
            msg for msg in messages if isinstance(msg, TextMessage)
        )

        # Parse the latest value from history, if available.
        if self._history:
            try:
                self._current_value = int(self._history[-1].content)
            except ValueError:
                pass  # Keep the current value if parsing fails.

        # Apply the operation.
        if self._operation == "add":
            self._current_value += self._operand
        elif self._operation == "multiply":
            self._current_value *= self._operand
        elif self._operation == "subtract":
            self._current_value -= self._operand

        result_msg = TextMessage(
            content=str(self._current_value), source=self.name
        )
        self._history.append(result_msg)

        return Response(chat_message=result_msg)

    async def on_reset(self, cancellation_token) -> None:
        self._history.clear()
        self._current_value = 0


async def main():
    adder = ArithmeticAgent("adder", operation="add", operand=5)
    multiplier = ArithmeticAgent("multiplier", operation="multiply", operand=2)
    token = CancellationToken()

    # Simulate a multi-turn exchange.
    # Turn 1: adder receives starting value "10"
    start = [TextMessage(content="10", source="user")]
    r1 = await adder.on_messages(start, token)
    print(f"Adder: {r1.chat_message.content}")

    # Turn 2: multiplier receives adder's output
    r2 = await multiplier.on_messages([r1.chat_message], token)
    print(f"Multiplier: {r2.chat_message.content}")

    # Turn 3: adder called again with multiplier's output
    r3 = await adder.on_messages([r2.chat_message], token)
    print(f"Adder: {r3.chat_message.content}")

    # Turn 4: adder called with empty messages (selected again, no new input)
    r4 = await adder.on_messages([], token)
    print(f"Adder (empty input): {r4.chat_message.content}")


asyncio.run(main())
```

**Output:**

```
Adder: 15
Multiplier: 30
Adder: 35
Adder (empty input): 40
```

### The empty-messages edge case

Notice turn 4: `on_messages` receives an empty list. This is not a bug — it's by design. In a `SelectorGroupChat`, the selector may choose the same agent consecutively. When that happens, no new messages exist, so the list is empty. Your agent must handle this gracefully by relying on its internal state rather than assuming `messages` always has content.

The pattern is straightforward:

1. Always maintain your own `_history` list.
2. Extend it with incoming messages when they arrive.
3. Fall back to the last known state when `messages` is empty.

---

## Integrating custom model clients

One of the most powerful reasons to build a custom agent is to integrate an LLM SDK that AutoGen doesn't support natively. Here's a sketch of a `GeminiAssistantAgent` that calls Google's Gemini API directly:

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage

# pip install google-generativeai
import google.generativeai as genai


class GeminiAssistantAgent(BaseChatAgent):
    """An agent powered by Google Gemini."""

    def __init__(
        self,
        name: str,
        *,
        model: str = "gemini-1.5-flash-002",
        system_message: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(name, description="An assistant powered by Google Gemini.")
        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model,
            system_instruction=system_message,
        )
        self._chat = self._model.start_chat()

    @property
    def produced_message_types(self) -> tuple[type, ...]:
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token) -> Response:
        # Build prompt from incoming messages.
        if messages:
            prompt = "\n".join(
                msg.content for msg in messages if isinstance(msg, TextMessage)
            )
        else:
            prompt = "Continue."

        # Call Gemini.
        response = self._chat.send_message(prompt)
        return Response(
            chat_message=TextMessage(
                content=response.text, source=self.name
            )
        )

    async def on_reset(self, cancellation_token) -> None:
        self._chat = self._model.start_chat()
```

This pattern generalizes to any LLM or external service:

1. Initialize the client in `__init__`.
2. Translate AutoGen messages into the SDK's expected format in `on_messages`.
3. Translate the SDK's response back into a `TextMessage`.
4. Reset the client state in `on_reset`.

You can apply the same approach to Anthropic's Claude SDK, Cohere, Mistral, a local Ollama instance, or even a non-LLM service like a database or search engine.

---

## Making agents declarative

AutoGen's **Component protocol** lets you serialize and deserialize agents from configuration. This is essential for saving agent setups, sharing them across teams, or loading them from config files.

To make a custom agent declarative, you:

1. Define a Pydantic config schema.
2. Have the agent class inherit from both `BaseChatAgent` and `Component`.
3. Implement `_to_config()` and `_from_config()`.

```python
from pydantic import BaseModel
from autogen_core.components import Component


class GeminiAssistantAgentConfig(BaseModel):
    """Configuration schema for GeminiAssistantAgent."""
    name: str
    description: str = "An assistant powered by Google Gemini."
    model: str = "gemini-1.5-flash-002"
    system_message: str | None = None


class GeminiAssistantAgent(
    BaseChatAgent, Component[GeminiAssistantAgentConfig]
):
    component_config_schema = GeminiAssistantAgentConfig
    component_type = "agent"
    component_provider_override = "my_package.GeminiAssistantAgent"

    def __init__(self, name, *, model="gemini-1.5-flash-002",
                 system_message=None, description="An assistant powered by Google Gemini."):
        super().__init__(name, description=description)
        self._model_name = model
        self._system_message = system_message
        # ... initialize Gemini client ...

    def _to_config(self) -> GeminiAssistantAgentConfig:
        return GeminiAssistantAgentConfig(
            name=self.name,
            description=self.description,
            model=self._model_name,
            system_message=self._system_message,
        )

    @classmethod
    def _from_config(cls, config: GeminiAssistantAgentConfig):
        return cls(
            name=config.name,
            model=config.model,
            system_message=config.system_message,
            description=config.description,
        )

    # ... on_messages, on_reset, produced_message_types as before ...
```

Once declarative, your agent supports `dump_component()` and `load_component()`:

```python
# Serialize to a dictionary
config = agent.dump_component()
print(config)

# Reconstruct from the dictionary
restored_agent = GeminiAssistantAgent.load_component(config)
```

**Output:**

```json
{
  "provider": "my_package.GeminiAssistantAgent",
  "component_type": "agent",
  "config": {
    "name": "gemini_helper",
    "description": "An assistant powered by Google Gemini.",
    "model": "gemini-1.5-flash-002",
    "system_message": "You are a helpful assistant."
  }
}
```

This makes your agents first-class citizens in AutoGen's ecosystem — they can be stored in JSON files, loaded dynamically, and composed in declarative team definitions.

---

## Using custom agents in teams

Custom agents work in any team type — `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm` — as long as they inherit from `BaseChatAgent`. No special registration is needed.

```python
import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


async def main():
    # Create custom agents.
    agent_a = CountDownAgent("countdown_a", count=2)
    agent_b = CountDownAgent("countdown_b", count=3)

    # Build a team.
    termination = MaxMessageTermination(max_messages=6)
    team = RoundRobinGroupChat(
        participants=[agent_a, agent_b],
        termination_condition=termination,
    )

    # Run the team.
    result = await team.run(task="Start counting!")
    for msg in result.messages:
        print(f"[{msg.source}] {msg.content}")


asyncio.run(main())
```

**Output:**

```
[user] Start counting!
[countdown_a] Done!
[countdown_b] Done!
```

### Tips for team compatibility

- **Write a clear `description`**: In `SelectorGroupChat`, the LLM selector reads each agent's description to decide who speaks next. A vague description leads to poor routing. Be specific: *"Performs addition on the running total"* beats *"A math agent"*.
- **Declare `produced_message_types` accurately**: Teams may use this to validate message flow. If your agent produces `HandoffMessage` in addition to `TextMessage`, declare both.
- **Handle empty messages**: As discussed earlier, your agent may be called with an empty list. Never crash on `messages[0]` without checking.

---

## Best practices

1. **Start simple, iterate**. Begin with `on_messages` only. Add `on_messages_stream` once the core logic works.
2. **Keep `on_reset` honest**. If your agent caches conversation history, API sessions, or counters, reset *all* of them. Incomplete resets cause subtle bugs in multi-run scenarios.
3. **Use `cancellation_token`**. For long-running operations (model calls, HTTP requests), check `cancellation_token.is_cancelled()` periodically or pass it to awaitable calls. This keeps your agent responsive to team-level timeouts.
4. **Log inner messages**. Use `inner_messages` in the `Response` to surface intermediate reasoning. Observability tools and team logs consume these — they're invaluable for debugging.
5. **Separate concerns**. Keep the LLM client, business logic, and AutoGen interface in distinct layers. This makes the agent testable outside of AutoGen.

---

## Common pitfalls

| Pitfall | Consequence | Fix |
|---|---|---|
| Ignoring empty `messages` list | `IndexError` crashes during group chat | Always check `if messages:` before accessing elements |
| Forgetting `super().__init__()` | Agent missing `name` and `description` | Always call `super().__init__(name, description=...)` |
| Returning raw text instead of `Response` | Type errors at the team level | Wrap output in `Response(chat_message=TextMessage(...))` |
| Mutable default arguments in `__init__` | Shared state across instances | Use `None` as default, initialize inside the method |
| Not implementing `on_reset` properly | Stale state leaks between team runs | Clear all mutable instance variables |

---

## Hands-on exercise

Build a **`SentimentRouterAgent`** that:

1. Accepts a `TextMessage` containing user feedback.
2. Classifies the sentiment as *positive*, *neutral*, or *negative* using simple keyword matching (no LLM needed).
3. Returns a `TextMessage` with the classification and a brief routing instruction (e.g., "Route to support team" for negative).
4. Tracks how many messages of each sentiment it has seen (state).
5. Resets the counters in `on_reset`.

**Stretch goals:**

- Add `on_messages_stream` that yields the classification step, then the routing step.
- Make the agent declarative with a config schema that includes the keyword lists.
- Add the agent to a `RoundRobinGroupChat` with a `CountDownAgent` and observe the message flow.

<details>
<summary>Starter template</summary>

```python
class SentimentRouterAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, description="Routes messages based on sentiment.")
        self._counts = {"positive": 0, "neutral": 0, "negative": 0}

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token):
        if not messages:
            summary = ", ".join(f"{k}: {v}" for k, v in self._counts.items())
            return Response(
                chat_message=TextMessage(
                    content=f"No new input. Counts so far — {summary}",
                    source=self.name,
                )
            )
        text = messages[-1].content.lower()
        # TODO: classify, update counts, return routing instruction
        ...

    async def on_reset(self, cancellation_token):
        self._counts = {"positive": 0, "neutral": 0, "negative": 0}
```

</details>

---

## Summary

Custom agents are AutoGen AgentChat's escape hatch from convention into full control. You learned:

- **`BaseChatAgent`** requires three implementations: `produced_message_types`, `on_messages`, and `on_reset`.
- **`on_messages_stream`** is optional but valuable for incremental output.
- The **`Response` object** bundles a final chat message with optional inner messages.
- **Stateful agents** must manage their own history and handle empty message lists.
- **Any LLM SDK** can power a custom agent — just translate messages in and out.
- The **Component protocol** makes agents serializable and shareable.
- Custom agents **drop into any team** with no special configuration.

With custom agents, you can wrap any service, enforce any logic, and participate in any conversation — while staying fully compatible with AutoGen's runtime and team abstractions.

**Next lesson:** [State and Memory](./11-state-and-memory.md)

---

## Further reading

- [AutoGen AgentChat Custom Agents documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/custom-agents.html)
- [AutoGen Component Protocol reference](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components.html)
- [BaseChatAgent API reference](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat/autogen_agentchat.agents.html)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources:
- https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/custom-agents.html
- https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components.html
-->
