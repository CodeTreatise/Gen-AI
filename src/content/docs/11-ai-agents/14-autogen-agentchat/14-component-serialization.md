---
title: "Component Serialization"
---

# Component Serialization

## Introduction

At some point, every AI agent project hits the same wall: the agent works beautifully on your laptop, but you need to ship it somewhere else. Maybe you want a colleague to run the exact same team configuration, or you need to deploy an agent pipeline from a staging environment into production. Manually recreating agents, models, tools, and termination conditions in each environment is fragile and error-prone.

AutoGen solves this with **component serialization** — a declarative configuration system that lets you convert any AutoGen component (agents, model clients, tools, teams, termination conditions) into a portable JSON document and reconstruct it anywhere. This is the foundation that powers config-driven workflows in **AutoGen Studio**, and it gives you programmatic control over saving, sharing, and versioning your entire agent architecture.

This lesson teaches you the complete component serialization system: the `dump_component()` and `load_component()` APIs, the difference between config and state, how to build your own serializable components, how to handle secrets safely, and how to serialize full multi-agent teams.

### What you'll cover

- Understanding the difference between component **config** (blueprint) and **state** (snapshot)
- Serializing components to JSON with `dump_component()`
- Reconstructing components from JSON with `load_component()`
- Creating custom serializable component classes with Pydantic schemas
- Protecting API keys with `SecretStr`
- Serializing entire teams and understanding limitations
- Practical patterns: config files, version control, cross-environment sharing

### Prerequisites

- Familiarity with AutoGen agents and teams ([AgentChat high-level API](./02-agentchat-high-level-api.md))
- Understanding of agent state persistence ([State and Memory](./11-state-and-memory.md))
- Working knowledge of Pydantic models
- Python 3.10+ with `autogen-agentchat` and `autogen-ext[openai]` installed

---

## Config vs state

AutoGen has two distinct serialization mechanisms, and confusing them is one of the most common mistakes newcomers make. They serve different purposes.

**Component config** is a *blueprint*. It captures the parameters needed to create a new instance of a component — the model name, system message, tool definitions, team structure. You can stamp out as many identical instances as you want from a single config. Think of it as a class definition, not an object.

**Component state** is a *snapshot*. It captures everything about a specific running instance — its conversation history, accumulated tool results, internal counters. Restoring state gives you back the *exact same object* as it existed at the moment you saved it.

| Aspect | Config (blueprint) | State (snapshot) |
|---|---|---|
| **What it captures** | Construction parameters | Runtime data + history |
| **API** | `dump_component()` / `load_component()` | `save_state()` / `load_state()` |
| **Output format** | `ComponentConfig` (JSON with `provider` field) | Plain Python `dict` |
| **Creates** | A fresh, new instance | A restored instance with full history |
| **Portable** | Yes — share across environments | Yes, but tied to same component config |
| **Use case** | Deployment, sharing, version control | Checkpointing, session resumption |

Here is the key mental model: **config defines what an agent *is*; state captures what an agent *has done*.**

In practice, you use config serialization when you want to define your agent architecture declaratively — saving team layouts to JSON files, sharing setups with teammates, loading configurations in AutoGen Studio, or deploying the same agent pipeline across dev, staging, and production. You use state serialization (covered in [State and Memory](./11-state-and-memory.md)) when you need to pause and resume a specific conversation.

---

## Dumping component configs

Every AutoGen component that participates in the serialization system exposes a `dump_component()` method. Calling it returns a `ComponentConfig` object — a Pydantic model that you can convert to JSON.

### Basic usage

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = AssistantAgent(
    name="code_reviewer",
    model_client=model_client,
    system_message="You are an expert code reviewer. Review code for bugs and style.",
)

# Serialize to config
config = agent.dump_component()
print(config.model_dump_json(indent=2))
```

**Output:**

```json
{
  "provider": "autogen_agentchat.agents.AssistantAgent",
  "component_type": "agent",
  "version": 1,
  "config": {
    "name": "code_reviewer",
    "model_client": {
      "provider": "autogen_ext.models.openai.OpenAIChatCompletionClient",
      "component_type": "model",
      "version": 1,
      "config": {
        "model": "gpt-4o-mini"
      }
    },
    "system_message": "You are an expert code reviewer. Review code for bugs and style."
  }
}
```

The output has four top-level fields:

| Field | Purpose |
|---|---|
| `provider` | Fully qualified Python class path. Used by `load_component()` to find and instantiate the correct class. |
| `component_type` | Category label — `"agent"`, `"model"`, `"tool"`, `"termination"`, `"team"`, etc. |
| `version` | Schema version number. Allows future migrations if the config format changes. |
| `config` | The actual construction parameters, nested recursively for sub-components. |

Notice that the model client is serialized *inside* the agent's config. AutoGen serializes the full component tree recursively — agents contain model clients, teams contain agents, and so on. A single `dump_component()` call on a team captures everything.

### Saving to a file

Since `ComponentConfig` is a Pydantic model, converting it to a JSON file is straightforward:

```python
import json
from pathlib import Path

config = agent.dump_component()

# Save to file
config_path = Path("code_reviewer_config.json")
config_path.write_text(config.model_dump_json(indent=2))

print(f"Config saved to {config_path} ({config_path.stat().st_size} bytes)")
```

**Output:**

```
Config saved to code_reviewer_config.json (412 bytes)
```

---

## Loading components from config

The `load_component()` class method reconstructs a component from a config dictionary or `ComponentConfig` object. You call it on the **interface class** (the base type you expect back), not on the concrete implementation class.

### Loading from a dictionary

```python
from autogen_core.models import ChatCompletionClient

config = {
    "provider": "autogen_ext.models.openai.OpenAIChatCompletionClient",
    "component_type": "model",
    "version": 1,
    "config": {
        "model": "gpt-4o"
    },
}

# Load on the interface class, not the implementation
client = ChatCompletionClient.load_component(config)
print(type(client).__name__)
```

**Output:**

```
OpenAIChatCompletionClient
```

The `provider` field tells AutoGen which class to import and instantiate. You call `load_component()` on `ChatCompletionClient` (the interface), and AutoGen returns an `OpenAIChatCompletionClient` (the implementation). This design means your loading code does not need to know the concrete class in advance.

### Loading an agent from a file

```python
import json
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent

# Load from saved file
config_data = json.loads(Path("code_reviewer_config.json").read_text())

restored_agent = AssistantAgent.load_component(config_data)
print(f"Agent name: {restored_agent.name}")
print(f"System message: {restored_agent._system_messages[0].content[:60]}...")
```

**Output:**

```
Agent name: code_reviewer
System message: You are an expert code reviewer. Review code for bugs and s...
```

The restored agent is a brand-new instance with the same configuration as the original. It has no conversation history — remember, config is a blueprint, not a snapshot.

### Round-trip verification

A useful pattern during development is to verify that a component survives a full serialization round trip:

```python
# Original agent
original = AssistantAgent(
    name="analyst",
    model_client=model_client,
    system_message="Analyze data trends.",
)

# Round trip: dump → load
config = original.dump_component()
restored = AssistantAgent.load_component(config)

# Verify
assert restored.name == original.name
print("Round-trip serialization successful")
```

**Output:**

```
Round-trip serialization successful
```

---

## Creating custom component classes

When you build your own agents, tools, or other components, you can make them serializable by implementing the `Component` protocol. This requires a Pydantic config schema, two conversion methods, and a couple of class variables.

### Step 1: Define the config schema

Create a Pydantic `BaseModel` that captures every parameter needed to reconstruct your component:

```python
from pydantic import BaseModel, Field

class SentimentAnalyzerConfig(BaseModel):
    """Config schema for the SentimentAnalyzer component."""
    name: str = Field(description="Component name")
    threshold: float = Field(default=0.5, description="Sentiment threshold for positive/negative")
    language: str = Field(default="en", description="Language code for analysis")
```

Keep the schema focused on construction parameters. Do not include runtime state, cached results, or mutable data. Everything in the schema should be something you would pass to `__init__`.

### Step 2: Implement the Component protocol

```python
from autogen_core import Component, ComponentBase

class SentimentAnalyzer(ComponentBase[SentimentAnalyzerConfig], Component[SentimentAnalyzerConfig]):
    component_type = "custom"
    component_config_schema = SentimentAnalyzerConfig

    def __init__(self, name: str, threshold: float = 0.5, language: str = "en"):
        self.name = name
        self.threshold = threshold
        self.language = language

    def _to_config(self) -> SentimentAnalyzerConfig:
        """Serialize this instance to its config schema."""
        return SentimentAnalyzerConfig(
            name=self.name,
            threshold=self.threshold,
            language=self.language,
        )

    @classmethod
    def _from_config(cls, config: SentimentAnalyzerConfig) -> "SentimentAnalyzer":
        """Create a new instance from a config schema."""
        return cls(
            name=config.name,
            threshold=config.threshold,
            language=config.language,
        )

    def analyze(self, text: str) -> str:
        # Simplified sentiment logic for demonstration
        positive_words = {"good", "great", "excellent", "love", "happy"}
        words = set(text.lower().split())
        score = len(words & positive_words) / max(len(words), 1)
        label = "positive" if score >= self.threshold else "negative"
        return f"{label} (score: {score:.2f})"
```

The key pieces are:

| Element | Purpose |
|---|---|
| `ComponentBase[ConfigType]` | Base class that provides `dump_component()` and `load_component()` |
| `Component[ConfigType]` | Protocol that declares the serialization interface |
| `component_type` | String category label (e.g., `"custom"`, `"tool"`, `"agent"`) |
| `component_config_schema` | Class variable pointing to the Pydantic config model |
| `_to_config()` | Instance method that returns a populated config object |
| `_from_config(config)` | Class method that creates a new instance from a config object |

### Step 3: Test serialization

```python
analyzer = SentimentAnalyzer(name="review_analyzer", threshold=0.3, language="en")

# Dump
config = analyzer.dump_component()
print(config.model_dump_json(indent=2))

# Load
restored = SentimentAnalyzer.load_component(config)
print(f"\nRestored: name={restored.name}, threshold={restored.threshold}")
print(f"Analysis: {restored.analyze('This product is great and excellent')}")
```

**Output:**

```json
{
  "provider": "__main__.SentimentAnalyzer",
  "component_type": "custom",
  "version": 1,
  "config": {
    "name": "review_analyzer",
    "threshold": 0.3,
    "language": "en"
  }
}
```

```
Restored: name=review_analyzer, threshold=0.3
Analysis: positive (score: 0.29)
```

### Overriding the provider path

When `load_component()` processes a config, it uses the `provider` field to import the class. If your component lives in a module that is not importable by its default path (e.g., it was defined in a `__main__` script or a dynamically generated module), set `component_provider_override` to the full importable path:

```python
class SentimentAnalyzer(ComponentBase[SentimentAnalyzerConfig], Component[SentimentAnalyzerConfig]):
    component_type = "custom"
    component_config_schema = SentimentAnalyzerConfig
    component_provider_override = "mypackage.analyzers.SentimentAnalyzer"

    # ... rest of implementation
```

Now `dump_component()` writes `"provider": "mypackage.analyzers.SentimentAnalyzer"` instead of `"__main__.SentimentAnalyzer"`, ensuring the config works when loaded from a different module.

---

## Handling secrets

API keys and other credentials present a serialization challenge. You want configs to be shareable and version-controllable, but you do not want API keys appearing in plain text in JSON files committed to Git.

AutoGen integrates with Pydantic's `SecretStr` type to solve this. Fields declared as `SecretStr` are **excluded from serialized output** — they will not appear when you call `dump_component()`.

### How SecretStr works in practice

```python
from pydantic import BaseModel, SecretStr

class MyServiceConfig(BaseModel):
    endpoint: str
    api_key: SecretStr  # Protected — will not be serialized

config = MyServiceConfig(endpoint="https://api.example.com", api_key="sk-secret-123")

# In Python, you can access the value
print(config.api_key.get_secret_value())

# But serialization hides it
print(config.model_dump_json(indent=2))
```

**Output:**

```
sk-secret-123
{
  "endpoint": "https://api.example.com",
  "api_key": "**********"
}
```

AutoGen's model clients already use this pattern. When you serialize an `OpenAIChatCompletionClient`, the API key is not included in the output. When you load the config on another machine, the client picks up the key from the `OPENAI_API_KEY` environment variable automatically.

### Best practice for custom components

If your custom component uses any credentials, declare them as `SecretStr` in your config schema:

```python
from pydantic import BaseModel, SecretStr, Field

class ExternalServiceConfig(BaseModel):
    service_url: str = Field(description="API endpoint URL")
    api_key: SecretStr = Field(description="API authentication key")
    timeout: int = Field(default=30, description="Request timeout in seconds")
```

When loading a component with secret fields, provide the secrets through environment variables or pass them explicitly during reconstruction:

```python
import os

config_data = json.loads(Path("service_config.json").read_text())
# The api_key field will be masked in the JSON.
# Set it via environment variable before loading, or inject it manually:
os.environ["SERVICE_API_KEY"] = "sk-actual-key"
```

This separation of config (shareable) from secrets (environment-specific) is a production best practice that applies far beyond AutoGen.

---

## Serializing teams and workflows

The real power of component serialization emerges when you serialize entire teams. A single `dump_component()` call on a team captures every agent, their model clients, tools, and termination conditions — the complete multi-agent architecture as one JSON document.

### Serializing a complete team

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Define agents
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="You are a project planner. Break tasks into steps.",
)

coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="You are a Python developer. Implement the planned steps.",
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="You review code for quality. Say APPROVE when satisfied.",
)

# Define termination
termination = MaxMessageTermination(10) | TextMentionTermination("APPROVE")

# Create team
team = RoundRobinGroupChat(
    participants=[planner, coder, reviewer],
    termination_condition=termination,
)

# Serialize the entire team
team_config = team.dump_component()
print(f"Team config size: {len(team_config.model_dump_json())} bytes")

# Save to file
Path("team_config.json").write_text(team_config.model_dump_json(indent=2))
print("Full team config saved")
```

**Output:**

```
Team config size: 1847 bytes
Full team config saved
```

### Restoring a team

```python
from autogen_agentchat.teams import RoundRobinGroupChat

config_data = json.loads(Path("team_config.json").read_text())
restored_team = RoundRobinGroupChat.load_component(config_data)

print(f"Team participants: {[p.name for p in restored_team._participants]}")

# Run the restored team
result = await restored_team.run(task="Write a Python function to calculate fibonacci numbers")
print(f"Messages exchanged: {len(result.messages)}")
```

**Output:**

```
Team participants: ['planner', 'coder', 'reviewer']
Messages exchanged: 7
```

### Serialization limitations

Not everything survives serialization. Be aware of these constraints:

| What serializes | What does NOT serialize |
|---|---|
| Agent names and system messages | Custom `selector_func` (Python callables) |
| Model client configurations | Custom `candidate_func` |
| Tool definitions (built-in tools) | Runtime state and message history |
| Termination conditions | Closure-based or lambda tools |
| Team structure and participant order | In-memory references and connections |

The most important limitation: **custom Python functions passed to `SelectorGroupChat` or `SwarmGroupChat` are not serialized.** If your team relies on a `selector_func` or `candidate_func`, you need to reconstruct those manually after loading the config:

```python
from autogen_agentchat.teams import SelectorGroupChat

# Load the base team config
config_data = json.loads(Path("selector_team.json").read_text())
team = SelectorGroupChat.load_component(config_data)

# Manually reattach the selector function after loading
def my_selector(messages, available_agents, *args, **kwargs):
    # Custom selection logic
    if any("bug" in m.content.lower() for m in messages):
        return "debugger"
    return None

team._selector_func = my_selector  # Reattach after deserialization
```

---

## Practical applications

### 1. Environment-specific config files

Maintain separate configs for development, staging, and production:

```python
import json
from pathlib import Path

def load_team_for_environment(env: str) -> RoundRobinGroupChat:
    """Load a team config tailored to the target environment."""
    config_path = Path(f"configs/{env}_team.json")
    config_data = json.loads(config_path.read_text())
    return RoundRobinGroupChat.load_component(config_data)

# Usage
dev_team = load_team_for_environment("development")
prod_team = load_team_for_environment("production")
```

Your development config might use `gpt-4o-mini` for lower cost, while production uses `gpt-4o` for higher quality — same team structure, different model clients.

### 2. Version-controlled agent configurations

Store your agent configs in Git alongside your code:

```
project/
├── src/
│   └── app.py
├── configs/
│   ├── analyst_agent.json
│   ├── coder_agent.json
│   └── review_team.json
├── tests/
│   └── test_configs.py
└── pyproject.toml
```

This lets you track changes to agent configurations with the same rigor as code changes. Pull requests can include diffs of agent system messages, model selections, and team structures.

### 3. Config validation in tests

Write tests that verify your configs load correctly:

```python
import pytest
import json
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

CONFIG_DIR = Path("configs")

@pytest.mark.parametrize("config_file", CONFIG_DIR.glob("*_agent.json"))
def test_agent_config_loads(config_file):
    """Verify every agent config file produces a valid agent."""
    config_data = json.loads(config_file.read_text())
    agent = AssistantAgent.load_component(config_data)
    assert agent.name, f"Agent from {config_file.name} has no name"

def test_team_round_trip():
    """Verify team config survives dump → load → dump."""
    config_data = json.loads((CONFIG_DIR / "review_team.json").read_text())
    team = RoundRobinGroupChat.load_component(config_data)
    re_dumped = team.dump_component().model_dump()
    assert re_dumped["config"]["participants"] == config_data["config"]["participants"]
```

### 4. AutoGen Studio integration

AutoGen Studio — the visual builder for AutoGen workflows — uses this exact config format. Any config you create programmatically can be imported into AutoGen Studio, and any workflow you design in AutoGen Studio can be exported and loaded in your Python code:

```python
# Export from code → import into AutoGen Studio
team_config = team.dump_component()
Path("studio_import.json").write_text(team_config.model_dump_json(indent=2))
# Now drag-and-drop studio_import.json into AutoGen Studio

# Export from AutoGen Studio → use in code
studio_config = json.loads(Path("studio_export.json").read_text())
team = RoundRobinGroupChat.load_component(studio_config)
result = await team.run(task="Analyze this quarter's revenue data")
```

---

## Best practices

1. **Always use interface classes for loading.** Call `ChatCompletionClient.load_component()`, not `OpenAIChatCompletionClient.load_component()`. This keeps your loading code decoupled from specific implementations.

2. **Never commit secrets.** Use `SecretStr` for all credentials. Store API keys in environment variables or a secrets manager, never in config JSON files.

3. **Set `component_provider_override` for custom classes.** If your component is defined in a script or notebook (`__main__`), the default provider path will not work when loading from a different context.

4. **Validate configs on load.** Wrap `load_component()` in try/except to catch import errors, missing fields, or version mismatches:

    ```python
    try:
        agent = AssistantAgent.load_component(config_data)
    except Exception as e:
        print(f"Failed to load config: {e}")
        # Fall back to default configuration
    ```

5. **Version your config schemas.** When you change a custom component's config schema, increment the version and handle migration in `_from_config()`.

6. **Separate config from state.** Use `dump_component()` for deployment and sharing. Use `save_state()` / `load_state()` for session continuity. Do not conflate the two.

---

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Calling `load_component()` on the concrete class | `TypeError` or unexpected behavior | Call on the interface/base class instead |
| Serializing components defined in `__main__` | `ModuleNotFoundError` on load | Set `component_provider_override` to the installable module path |
| Expecting conversation history after loading config | Agent starts fresh with no memory | Use `save_state()` / `load_state()` for session resumption |
| Committing JSON configs with API keys | Credentials leaked in version control | Use `SecretStr` and environment variables |
| Assuming `selector_func` is serialized | Team loads but selection logic is missing | Reattach callable functions manually after `load_component()` |
| Changing config schema without versioning | Old configs fail to load | Increment version and add migration logic in `_from_config()` |

---

## Hands-on exercise

Build a configurable research team and verify it survives serialization:

1. **Create two agents** — a `researcher` that finds information and a `writer` that produces summaries — each with distinct system messages.
2. **Create a `RoundRobinGroupChat`** team with a `MaxMessageTermination(6)` condition.
3. **Serialize the team** to a JSON file called `research_team.json`.
4. **Load the team** from the JSON file in a new script (or a fresh cell) and verify that both agent names and the termination condition are intact.
5. **Extend the exercise**: Create a custom `TopicFilter` component (with a Pydantic schema containing `topics: list[str]` and `min_relevance: float`) that implements the `Component` protocol. Serialize it, verify the round trip, and confirm that the `provider` field is correct.

**Stretch goal:** Write a pytest test that loads every `.json` file in a `configs/` directory and asserts that each one produces a valid component.

---

## Summary

Component serialization transforms AutoGen agents from ephemeral Python objects into portable, shareable, version-controllable configurations. The system rests on a clean separation: `dump_component()` captures the *blueprint* for creating a component, while `save_state()` captures the *snapshot* of a running instance. Custom components join the system by implementing the `Component` protocol with a Pydantic config schema and two methods — `_to_config()` and `_from_config()`. Secrets stay safe through `SecretStr`, and entire teams serialize recursively with a single call. The same JSON format powers AutoGen Studio, CI/CD pipelines, and cross-environment deployments.

Master this system and you gain a clean separation between *defining* your agents and *running* them — a separation that pays dividends as your projects grow from prototypes to production systems.

**Next:** [Extensions and Ecosystem](./15-extensions-and-ecosystem.md)

---

## Further reading

- [AutoGen Component Config Documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/serialize-components.html)
- [Pydantic SecretStr Reference](https://docs.pydantic.dev/latest/concepts/types/#secret-types)
- [AutoGen Studio Documentation](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html)
- [State and Memory lesson](./11-state-and-memory.md) — for runtime state persistence with `save_state()` / `load_state()`

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources:
- AutoGen official documentation: Component Config and Serialization
- AutoGen GitHub repository: autogen-core/src/autogen_core/_component_config.py
- AutoGen AgentChat serialize-components user guide
-->
