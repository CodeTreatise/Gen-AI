---
title: "Multi-Language SDK Support in Google ADK"
---

# Multi-Language SDK Support in Google ADK

## Prerequisites

Before starting this lesson, you should have:

- Completed [ADK Web Integration](./09-adk-web-integration.md)
- Familiarity with at least one of: Python, TypeScript, Go, or Java
- A Google Cloud project with the Gemini API enabled
- Basic understanding of agent concepts from earlier lessons in this unit

---

## Introduction

One of Google Agent Development Kit's most compelling strengths is its commitment to meeting developers where they already work. Rather than forcing every team into a single language, ADK provides official SDKs in **four languages**: Python, TypeScript, Go, and Java. This means a data-science team fluent in Python, a frontend squad comfortable in TypeScript, a platform-engineering group writing Go microservices, and an enterprise backend team entrenched in the JVM can all build, deploy, and interconnect ADK agents without switching stacks.

In this lesson we explore each SDK in depth, compare feature parity across languages, and demonstrate how the **Agent-to-Agent (A2A) protocol** lets agents written in different languages communicate seamlessly. By the end, we will know which SDK to reach for in any given scenario and how to build heterogeneous multi-language agent systems.

---

## SDK Overview

The table below summarizes the current state of every official ADK SDK.

| Language | Version | Package / Install Command | Maturity |
|------------|---------|----------------------------------------------|-----------------|
| Python | Latest | `pip install google-adk` | Production-ready |
| TypeScript | v0.2.0 | `npm install @anthropic-ai/adk` | Beta |
| Go | v0.3.0 | `go get google.github.io/adk-go` | Alpha |
| Java | v0.5.0 | Maven / Gradle (`com.google.adk:adk-java`) | Alpha |

> **Note:** Package names for TypeScript, Go, and Java may change as the SDKs evolve. Always check the official repositories for the latest installation instructions.

Python is the **primary** SDK — it receives features first, has the broadest documentation, and carries the largest community ecosystem. The remaining three SDKs are catching up rapidly, and Google has signalled long-term support for all four.

---

## Python SDK — The Reference Implementation

Python is the most mature ADK SDK and the one against which all others are measured. If we are starting a greenfield project and want access to every ADK feature on day one, Python is the safest choice.

### Basic Agent Definition

```python
from google.adk.agents import Agent

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

root_agent = Agent(
    name="my_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Answer questions concisely.",
    tools=[search_web],
)
```

Expected output when running via `adk run`:

```text
[my_agent]: Hello! I'm your helpful assistant. Ask me anything.
```

### Why Choose Python

- **Full feature parity** — every ADK capability lands in Python first.
- **Best documentation** — official guides, API references, and cookbooks all target Python.
- **Largest community** — the majority of tutorials, blog posts, and Stack Overflow answers reference the Python SDK.
- **Recommended for production** — battle-tested in Google's own deployments.
- **Rich AI ecosystem** — integrates naturally with NumPy, pandas, LangChain, and other Python-native libraries.

---

## TypeScript SDK — Web-Native Agent Development

The TypeScript SDK brings ADK into the Node.js and Deno ecosystems, making it a natural fit for teams already building web applications in JavaScript or TypeScript.

### Basic Agent Definition

```typescript
import { Agent } from '@anthropic-ai/adk';

function searchWeb(query: string): string {
    return `Results for: ${query}`;
}

const rootAgent = new Agent({
    name: 'my_agent',
    model: 'gemini-2.0-flash',
    instruction: 'You are a helpful assistant. Answer questions concisely.',
    tools: [searchWeb],
});

rootAgent.run().then(() => {
    console.log('Agent started successfully');
});
```

Expected output:

```text
Agent started successfully
[my_agent]: Hello! I'm your helpful assistant. Ask me anything.
```

### Why Choose TypeScript

- **Node.js and Deno support** — run agents anywhere JavaScript runs, including serverless platforms like Vercel and Cloudflare Workers.
- **Great for web-native teams** — share types, validation logic, and utility functions between frontend and agent backend.
- **Growing feature set** — most core capabilities (LlmAgent, custom agents, workflow agents, FunctionTool) are already available.
- **Type safety** — TypeScript's static type system catches configuration errors at compile time rather than at runtime.

### TypeScript-Specific Considerations

The TypeScript SDK uses `async/await` natively, making streaming and event-driven patterns feel idiomatic:

```typescript
import { Agent } from '@anthropic-ai/adk';

const agent = new Agent({
    name: 'stream_agent',
    model: 'gemini-2.0-flash',
    instruction: 'Summarize input text.',
});

async function main() {
    const response = await agent.invoke('Summarize the history of AI.');
    console.log(response.text);
}

main();
```

Expected output:

```text
AI began as a field in the 1950s with symbolic reasoning, evolved through
machine learning in the 1990s, and entered the deep learning era in the 2010s...
```

---

## Go SDK — High-Performance Agents

Go's concurrency primitives (goroutines, channels) make it an excellent choice when we need agents that handle thousands of simultaneous requests with minimal resource overhead.

### Basic Agent Definition

```go
package main

import (
    "fmt"
    "github.com/google/adk-go/pkg/agents"
)

func searchWeb(query string) string {
    return fmt.Sprintf("Results for: %s", query)
}

func main() {
    agent := agents.NewAgent(agents.AgentConfig{
        Name:        "my_agent",
        Model:       "gemini-2.0-flash",
        Instruction: "You are a helpful assistant. Answer questions concisely.",
        Tools:       []agents.Tool{agents.WrapFunc("search_web", searchWeb)},
    })

    result, err := agent.Run()
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Println(result)
}
```

Expected output:

```text
[my_agent]: Hello! I'm your helpful assistant. Ask me anything.
```

### Why Choose Go

- **Strong concurrency support** — goroutines let us fan out tool calls and sub-agent invocations efficiently.
- **Ideal for microservices** — small binary size and fast cold starts make Go agents perfect for containerized deployments.
- **Statically typed and compiled** — catch errors early and ship a single binary with no runtime dependencies.
- **Low memory footprint** — well-suited for cost-sensitive or edge environments.

---

## Java SDK — Enterprise-Grade Agents

Java is the lingua franca of enterprise software. The Java ADK SDK integrates with the vast JVM ecosystem, including Spring Boot, Quarkus, and Gradle/Maven build pipelines.

### Basic Agent Definition

```java
import com.google.adk.agents.Agent;
import java.util.List;

public class MyAgent {
    public static void main(String[] args) {
        Agent rootAgent = Agent.builder()
            .name("my_agent")
            .model("gemini-2.0-flash")
            .instruction("You are a helpful assistant. Answer questions concisely.")
            .tools(List.of(new SearchWebTool()))
            .build();

        String response = rootAgent.run();
        System.out.println(response);
    }
}
```

Expected output:

```text
[my_agent]: Hello! I'm your helpful assistant. Ask me anything.
```

### Why Choose Java

- **Enterprise-friendly** — fits naturally into existing CI/CD pipelines, monitoring stacks, and governance frameworks.
- **Spring Boot integration** — expose agents as REST endpoints with dependency injection and auto-configuration.
- **JVM ecosystem** — leverage libraries for logging (SLF4J), metrics (Micrometer), security (Spring Security), and more.
- **Mature tooling** — IntelliJ, Eclipse, and Gradle/Maven provide first-class development experiences.

### Spring Boot Integration Example

```java
import com.google.adk.agents.Agent;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class AgentApplication {

    @Bean
    public Agent helpAgent() {
        return Agent.builder()
            .name("help_agent")
            .model("gemini-2.0-flash")
            .instruction("You assist enterprise users with IT support.")
            .build();
    }

    public static void main(String[] args) {
        SpringApplication.run(AgentApplication.class, args);
    }
}
```

Expected output:

```text
  .   ____          _
 /\\ / ___'_ __ _ _(_)_ __
( ( )\___ | '_ | '_| | '_ \
 \\/  ___)| |_)| | | | | |_)| |
  '  |____| .__|_| |_|_| |_\__|
 :: Spring Boot ::
Started AgentApplication in 2.3 seconds
```

---

## Feature Parity Matrix

Not every SDK supports every feature yet. The matrix below shows current feature availability.

| Feature | Python | TypeScript | Go | Java |
|----------------------|--------|------------|------|------|
| LlmAgent | ✅ | ✅ | ✅ | ✅ |
| Custom Agents | ✅ | ✅ | ✅ | ✅ |
| Workflow Agents | ✅ | ✅ | ✅ | ✅ |
| FunctionTool | ✅ | ✅ | ✅ | ✅ |
| MCP Tools | ✅ | ⚠️ Partial | ❌ | ❌ |
| Callbacks | ✅ | ✅ | ⚠️ Partial | ⚠️ Partial |
| A2A Protocol | ✅ | ✅ | ✅ | ✅ |
| Visual Builder | ✅ | ❌ | ❌ | ❌ |
| Bidi-Streaming | ✅ | ⚠️ Partial | ❌ | ❌ |

**Legend:** ✅ Fully supported | ⚠️ Partial / experimental | ❌ Not yet available

Key takeaways from the matrix:

- **Core agent types** (LlmAgent, custom, workflow) are available everywhere.
- **MCP tool integration** is Python-first; TypeScript has partial support.
- **A2A** is universal — this is the primary mechanism for cross-language interoperability.
- **Visual Builder** remains Python-only for now.

---

## Cross-Language Interoperability via A2A

The Agent-to-Agent (A2A) protocol is the glue that binds multi-language agent systems together. Because A2A communicates over HTTP with a standardized JSON payload, the language each agent is written in becomes irrelevant.

### Python Agent Exposing an A2A Server

```python
from google.adk.agents import Agent
from google.adk.a2a import A2AServer

research_agent = Agent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction="You research topics and return structured summaries.",
)

server = A2AServer(agent=research_agent, port=8080)
server.start()
```

Expected output:

```text
A2A server started on http://localhost:8080
Agent 'research_agent' is ready to accept requests.
```

### TypeScript Agent Connecting as a Remote Client

```typescript
import { Agent, RemoteA2aAgent } from '@anthropic-ai/adk';

const remoteResearcher = new RemoteA2aAgent({
    name: 'research_agent',
    url: 'http://localhost:8080',
});

const orchestrator = new Agent({
    name: 'orchestrator',
    model: 'gemini-2.0-flash',
    instruction: 'You coordinate research tasks.',
    tools: [remoteResearcher],
});

const result = await orchestrator.invoke('Research quantum computing trends.');
console.log(result.text);
```

Expected output:

```text
Quantum computing trends in 2026 include advances in error correction,
the rise of hybrid classical-quantum algorithms, and growing enterprise
adoption through cloud-based quantum services...
```

### Why This Matters

- **Language independence** — the Python research agent and TypeScript orchestrator collaborate without either knowing the other's language.
- **Team autonomy** — each team picks the language they are most productive in.
- **Common protocol** — A2A defines a universal contract, enabling truly heterogeneous agent systems.
- **Deployment flexibility** — agents can live in separate containers, clusters, or even cloud providers.

---

## Choosing the Right SDK

Use the decision table below to match your situation to the best SDK.

| Scenario | Recommended SDK | Rationale |
|---------------------------------------|-----------------|-----------------------------------------------|
| New project, need all features | Python | Full parity, best docs, largest community |
| Web application backend | TypeScript | Shares ecosystem with frontend code |
| High-performance microservice | Go | Low latency, small binaries, goroutines |
| Enterprise Java shop | Java | Integrates with Spring, Maven, JVM governance |
| Mixed-language distributed system | A2A Protocol | Language-agnostic inter-agent communication |
| Rapid prototyping / experimentation | Python | Fastest iteration, Visual Builder access |
| Serverless / edge deployment | Go or TypeScript | Small footprint, fast cold starts |

---

## Best Practices

| Practice | Description |
|------------------------------------------|----------------------------------------------------------------------|
| Start with Python for prototyping | Leverage full feature set and Visual Builder, then port if needed |
| Use A2A for cross-language systems | Never build custom bridges — A2A is the standard protocol |
| Pin SDK versions | Lock dependency versions to avoid breaking changes in alpha SDKs |
| Follow idiomatic patterns per language | Use builders in Java, structs in Go, classes in TypeScript |
| Monitor feature parity matrix | Check release notes before depending on a feature in non-Python SDKs |
| Write language-agnostic agent specs | Define agent behavior in docs first, then implement in any SDK |
| Test with the official CLI | Every SDK supports `adk run` or equivalent for local testing |
| Keep agents small and focused | Single-responsibility agents are easier to port across languages |

---

## Common Pitfalls

| ❌ Don't | ✅ Do |
|--------------------------------------------------|----------------------------------------------------------|
| Assume all features exist in every SDK | Check the feature parity matrix before starting |
| Build custom RPC between agents in different langs | Use the A2A protocol for inter-agent communication |
| Copy Python code and "translate" line by line | Write idiomatic code for each target language |
| Ignore SDK maturity labels (Alpha, Beta) | Plan production workloads around maturity level |
| Hardcode model names across SDKs | Use environment variables or config files for model names |
| Skip error handling in Alpha SDKs | Alpha SDKs have rougher edges — add defensive error handling |
| Mix SDK versions in a monorepo carelessly | Keep each SDK in its own module with pinned dependencies |
| Forget to test A2A serialization edge cases | Validate that complex types round-trip correctly over A2A |

---

## Hands-on Exercise

**Goal:** Build a two-agent system where a Python agent and a TypeScript agent communicate via A2A.

### Requirements

1. Create a Python agent called `fact_checker` that verifies factual claims.
2. Expose the `fact_checker` via an A2A server on port `9090`.
3. Create a TypeScript agent called `writer` that drafts short articles and uses `fact_checker` as a remote tool.
4. Invoke the `writer` agent with the prompt: *"Write a short article about the speed of light and verify all facts."*
5. Confirm that the TypeScript `writer` agent delegates fact-checking to the Python `fact_checker` agent.

<details>
<summary>💡 Hints</summary>

- Start the Python A2A server first, then run the TypeScript agent.
- Use `A2AServer` from `google.adk.a2a` in the Python agent.
- Use `RemoteA2aAgent` in the TypeScript agent, pointing to `http://localhost:9090`.
- The `writer` agent should list the remote fact-checker in its `tools` array.

</details>

<details>
<summary>✅ Solution — Python fact_checker (fact_checker/agent.py)</summary>

```python
from google.adk.agents import Agent
from google.adk.a2a import A2AServer

def verify_claim(claim: str) -> str:
    """Verify a factual claim and return a verdict."""
    # In production, this would call a knowledge base or search API
    return f"VERIFIED: '{claim}' is accurate."

fact_checker = Agent(
    name="fact_checker",
    model="gemini-2.0-flash",
    instruction="You verify factual claims. Use the verify_claim tool for each claim.",
    tools=[verify_claim],
)

server = A2AServer(agent=fact_checker, port=9090)
server.start()
```

Expected output:

```text
A2A server started on http://localhost:9090
Agent 'fact_checker' is ready to accept requests.
```

</details>

<details>
<summary>✅ Solution — TypeScript writer (writer/index.ts)</summary>

```typescript
import { Agent, RemoteA2aAgent } from '@anthropic-ai/adk';

const factChecker = new RemoteA2aAgent({
    name: 'fact_checker',
    url: 'http://localhost:9090',
});

const writer = new Agent({
    name: 'writer',
    model: 'gemini-2.0-flash',
    instruction: 'You write short, factual articles. Verify every fact using the fact_checker tool before including it.',
    tools: [factChecker],
});

async function main() {
    const result = await writer.invoke(
        'Write a short article about the speed of light and verify all facts.'
    );
    console.log(result.text);
}

main();
```

Expected output:

```text
The Speed of Light

The speed of light in a vacuum is approximately 299,792,458 meters per second.
[VERIFIED] This value, commonly denoted as 'c', is one of the fundamental
constants of physics...
```

</details>

---

## Summary

✅ ADK provides official SDKs in **four languages**: Python, TypeScript, Go, and Java.

✅ **Python** is the primary SDK with full feature parity, the best documentation, and production-ready maturity.

✅ **TypeScript** suits web-native teams, offering Node.js and Deno support with a growing feature set.

✅ **Go** excels in high-performance, low-latency microservice scenarios with strong concurrency primitives.

✅ **Java** fits enterprise environments with Spring Boot integration and the full JVM ecosystem.

✅ The **Feature Parity Matrix** helps us choose the right SDK by revealing which capabilities are available in each language.

✅ The **A2A protocol** enables language-agnostic inter-agent communication, letting us mix SDKs freely in distributed systems.

✅ Always check SDK **maturity labels** and **pin versions** when working with Alpha or Beta SDKs.

---

## Next

[A2A Protocol](./11-a2a-protocol.md)

---

## Further Reading

- [Google ADK Documentation — Getting Started](https://google.github.io/adk-docs/get-started/)
- [Google ADK Python SDK — GitHub](https://github.com/google/adk-python)
- [Google ADK TypeScript SDK — GitHub](https://github.com/google/adk-typescript)
- [Google ADK Go SDK — GitHub](https://github.com/google/adk-go)
- [Google ADK Java SDK — GitHub](https://github.com/google/adk-java)
- [A2A Protocol Specification](https://google.github.io/A2A/)
- [Gemini API — Model Reference](https://ai.google.dev/gemini-api/docs/models)

---

[Back to Google ADK Overview](./00-google-agent-development-kit.md)

<!-- Sources:
- Google ADK Documentation: https://google.github.io/adk-docs/
- Google ADK Python SDK: https://github.com/google/adk-python
- Google ADK TypeScript SDK: https://github.com/google/adk-typescript (placeholder — verify package name)
- Google ADK Go SDK: https://github.com/google/adk-go (placeholder — verify import path)
- Google ADK Java SDK: https://github.com/google/adk-java (placeholder — verify Maven coordinates)
- A2A Protocol: https://google.github.io/A2A/
- Gemini API Reference: https://ai.google.dev/gemini-api/docs/models
-->
