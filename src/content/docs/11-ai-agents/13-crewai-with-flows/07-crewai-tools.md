---
title: "CrewAI Tools"
---

# CrewAI Tools

## Introduction

Tools give CrewAI agents the ability to **interact with the real world** — searching the web, reading files, running code, querying databases, and more. Without tools, agents are limited to what the LLM knows from training data. With tools, they become capable of taking actions and accessing live information.

CrewAI provides 30+ built-in tools and a simple API for creating custom ones.

### What We'll Cover

- What tools are and how agents use them
- Creating custom tools with `@tool` decorator
- Creating custom tools with `BaseTool` subclass
- Built-in tools overview
- Async tools for non-blocking execution
- Tool caching for repeated queries

### Prerequisites

- Completed [Core Concepts](./01-core-concepts.md) (Agents, Tasks, Crews)
- `pip install crewai-tools` for built-in tools

---

## How Agents Use Tools

When an agent has tools assigned, the LLM decides **when and how** to use them based on the task description:

```python
from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool

researcher = Agent(
    role="Research Analyst",
    goal="Find current information about AI trends",
    backstory="Expert researcher who always verifies claims with real data.",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm="gpt-4o-mini",
)
```

The agent autonomously:
1. Reads the task description
2. Decides which tool to use (or none)
3. Formulates the tool input
4. Executes the tool
5. Uses the result to continue working

> **🤖 AI Context:** Under the hood, CrewAI uses function calling (tool use) — the same mechanism as OpenAI's function calling API. Tools are converted to JSON schemas that the LLM uses to decide when and how to invoke them.

---

## Creating Custom Tools

### Method 1: The @tool Decorator

The quickest way to create a tool:

```python
from crewai.tools import tool


@tool("Calculate Readability")
def calculate_readability(text: str) -> str:
    """Calculate the readability score of a given text.
    
    Args:
        text: The text to analyze for readability.
    
    Returns:
        A readability assessment with word count and average sentence length.
    """
    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")
    sentences = max(sentences, 1)
    
    word_count = len(words)
    avg_sentence_length = word_count / sentences
    
    if avg_sentence_length < 15:
        level = "Easy"
    elif avg_sentence_length < 25:
        level = "Moderate"
    else:
        level = "Difficult"
    
    return f"Words: {word_count}, Avg sentence length: {avg_sentence_length:.1f}, Level: {level}"
```

Key rules for `@tool`:
- The **function name** becomes the tool's internal identifier
- The **string argument** to `@tool()` is the display name
- The **docstring** is critical — the LLM reads it to decide when to use the tool
- **Type hints** define the input schema

### Method 2: BaseTool Subclass

For more control, subclass `BaseTool`:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileAnalyzerInput(BaseModel):
    file_path: str = Field(description="Path to the file to analyze")
    include_stats: bool = Field(default=True, description="Include word/line statistics")


class FileAnalyzerTool(BaseTool):
    name: str = "File Analyzer"
    description: str = "Analyzes a text file and returns its contents with optional statistics."
    args_schema: type[BaseModel] = FileAnalyzerInput
    
    def _run(self, file_path: str, include_stats: bool = True) -> str:
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            if include_stats:
                lines = content.count("\n") + 1
                words = len(content.split())
                return f"File: {file_path}\nLines: {lines}, Words: {words}\n\n{content[:500]}"
            
            return content[:1000]
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
```

### When to Use Each Approach

| Approach | Best For |
|----------|----------|
| `@tool` decorator | Simple tools, quick prototyping, straightforward logic |
| `BaseTool` subclass | Complex tools, custom input schemas, tools needing class state |

---

## Assigning Tools

Tools can be assigned at the **agent level** or the **task level**:

```python
from crewai import Agent, Task

# Agent-level: Available for all tasks this agent handles
agent = Agent(
    role="Analyst",
    goal="Analyze data",
    backstory="Data expert",
    tools=[calculate_readability, FileAnalyzerTool()],
)

# Task-level: Overrides agent tools for this specific task
task = Task(
    description="Analyze the readability of report.txt",
    expected_output="Readability assessment",
    agent=agent,
    tools=[calculate_readability],  # Only this tool available
)
```

> **Note:** When tools are set on a Task, they **replace** (not extend) the agent's tools for that task.

---

## Built-in Tools

Install the tools package:

```bash
pip install crewai-tools
```

### Most Common Tools

| Tool | Purpose | Example Use |
|------|---------|-------------|
| `SerperDevTool` | Web search via Serper API | Research current events |
| `WebsiteSearchTool` | Search within a specific website | Find documentation |
| `FileReadTool` | Read file contents | Process local files |
| `DirectoryReadTool` | List directory contents | Explore project structure |
| `PDFSearchTool` | Search within PDF documents | Analyze reports |
| `CodeInterpreterTool` | Execute Python code | Data analysis, calculations |
| `GithubSearchTool` | Search GitHub repositories | Find code examples |
| `YoutubeVideoSearchTool` | Search YouTube video content | Research video content |
| `ScrapeWebsiteTool` | Scrape webpage content | Extract web data |

### Example: Web Research Agent

```python
from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

web_researcher = Agent(
    role="Web Research Specialist",
    goal="Find and extract current information from the web",
    backstory="Expert at web research with 5 years finding reliable sources.",
    tools=[
        SerperDevTool(),        # Search the web
        ScrapeWebsiteTool(),    # Read webpage content
    ],
    llm="gpt-4o-mini",
)
```

### Example: Document Analysis Agent

```python
from crewai import Agent
from crewai_tools import FileReadTool, PDFSearchTool, DirectoryReadTool

doc_analyst = Agent(
    role="Document Analyst",
    goal="Analyze and summarize documents from the local filesystem",
    backstory="Experienced analyst specializing in document review.",
    tools=[
        DirectoryReadTool(directory="./documents"),
        FileReadTool(),
        PDFSearchTool(pdf="./reports/quarterly.pdf"),
    ],
    llm="gpt-4o-mini",
)
```

---

## Async Tools

For tools that perform I/O operations (API calls, file reads), async versions improve performance:

### Async with @tool

```python
import httpx
from crewai.tools import tool


@tool("Fetch URL")
async def fetch_url(url: str) -> str:
    """Fetch the content of a URL.
    
    Args:
        url: The URL to fetch content from.
    
    Returns:
        The text content of the webpage.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text[:2000]
```

### Async with BaseTool

```python
from crewai.tools import BaseTool


class AsyncAPITool(BaseTool):
    name: str = "API Fetcher"
    description: str = "Fetches data from an external API asynchronously."
    
    async def _run(self, endpoint: str) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.example.com/{endpoint}")
            return response.json()
```

---

## Tool Caching

CrewAI supports caching to avoid redundant tool calls with the same input:

```python
from crewai.tools import tool


@tool("Search Database")
def search_database(query: str) -> str:
    """Search the product database for matching items."""
    # Expensive database query
    results = db.search(query)
    return str(results)


def custom_cache_check(tool_call: dict, result: str) -> bool:
    """Cache results for 5 minutes."""
    # Return True to use cached result, False to re-execute
    return len(result) > 0  # Cache non-empty results


search_database.cache_function = custom_cache_check
```

You can also enable caching at the Crew level:

```python
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    cache=True,  # Enable global caching
)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Write detailed docstrings for tools | The LLM uses the docstring to decide when to use the tool |
| Add type hints to all tool parameters | Creates proper input schemas for function calling |
| Give tools focused, single responsibilities | "Search Web" not "Search Web and Analyze Results" |
| Use async tools for I/O-bound operations | Prevents blocking while waiting for network/disk |
| Enable caching for expensive, repeatable queries | Saves API calls and speeds up execution |
| Test tools independently before assigning to agents | Easier to debug tool logic in isolation |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Vague tool descriptions | Be specific: "Search the web for current news articles" not "Search stuff" |
| Missing docstring on `@tool` functions | Always include a docstring — the LLM can't use the tool without it |
| Giving agents too many tools | 3-5 tools per agent is optimal; too many cause decision paralysis |
| Not handling tool errors | Add try/except and return error messages as strings |
| Using synchronous HTTP calls in tools | Use `httpx.AsyncClient` or `aiohttp` for async network calls |
| Forgetting to install `crewai-tools` | Built-in tools require: `pip install crewai-tools` |

---

## Hands-on Exercise

### Your Task

Create two custom tools and an agent that uses them.

### Requirements

1. Create a `@tool` that counts words in a given text and returns statistics
2. Create a `BaseTool` subclass that formats text as a numbered list
3. Create an agent with both tools assigned
4. Create a task that asks the agent to analyze and format some text
5. Run the Crew and verify both tools are used

### Expected Result

```
Agent uses the word counter tool to analyze the text,
then uses the formatter tool to create a numbered list.
```

<details>
<summary>💡 Hints (click to expand)</summary>

- The word counter should return sentences count, word count, and average sentence length
- The formatter should split text by sentences or paragraphs and number them
- Set `verbose=True` on the agent to see tool usage in the logs

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field


@tool("Word Counter")
def word_counter(text: str) -> str:
    """Count words, sentences, and calculate readability metrics for text.
    
    Args:
        text: The text to analyze.
    
    Returns:
        Word count, sentence count, and average sentence length.
    """
    words = len(text.split())
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_length = words / sentences
    return f"Words: {words}, Sentences: {sentences}, Avg length: {avg_length:.1f}"


class FormatterInput(BaseModel):
    text: str = Field(description="The text to format as a numbered list")
    delimiter: str = Field(default=".", description="Character to split text on")


class NumberedListFormatter(BaseTool):
    name: str = "Numbered List Formatter"
    description: str = "Formats text into a numbered list by splitting on sentences."
    args_schema: type[BaseModel] = FormatterInput
    
    def _run(self, text: str, delimiter: str = ".") -> str:
        items = [s.strip() for s in text.split(delimiter) if s.strip()]
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


analyst = Agent(
    role="Text Analyst",
    goal="Analyze and format text content",
    backstory="Expert editor skilled at text analysis and formatting.",
    tools=[word_counter, NumberedListFormatter()],
    llm="gpt-4o-mini",
    verbose=True,
)

task = Task(
    description="""Analyze this text and format it as a numbered list:
    'AI agents can search the web. They can read files. They can execute code.
    They can interact with APIs. They improve with memory.'""",
    expected_output="Word count statistics followed by the text as a numbered list",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=True)
result = crew.kickoff()
print(result.raw)
```

</details>

### Bonus Challenges

- [ ] Add error handling to both tools (handle empty text, missing files)
- [ ] Create an async version of the word counter using `@tool`
- [ ] Add a `cache_function` that caches results for identical text inputs

---

## Summary

✅ Tools let agents interact with the real world — search, read, write, compute

✅ Create tools with `@tool` decorator (quick) or `BaseTool` subclass (flexible)

✅ Detailed docstrings and type hints are critical — the LLM uses them for decision-making

✅ 30+ built-in tools cover web search, file I/O, code execution, and more

✅ Use async tools for I/O operations and caching for expensive, repeatable queries

**Next:** [Production Deployment](./08-production-deployment.md)

---

## Further Reading

- [CrewAI Tools Documentation](https://docs.crewai.com/concepts/tools) — Full tools API and built-in tool list
- [CrewAI Tools Package](https://github.com/crewAIInc/crewAI-tools) — Source code and examples

*Back to [CrewAI with Flows Overview](./00-crewai-with-flows.md)*

<!-- 
Sources Consulted:
- CrewAI Tools: https://docs.crewai.com/concepts/tools
-->
