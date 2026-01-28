---
title: "OpenAI Built-in Tools"
---

# OpenAI Built-in Tools

## Introduction

OpenAI provides several built-in tools that execute server-side, extending model capabilities without requiring your own infrastructure. These tools are configured in the `tools` array and handle complex operations like web search, code execution, and file analysis.

### What We'll Cover

- Web search tool
- Code interpreter
- File search
- Image generation
- Computer use preview
- Shell and patch tools
- Combining multiple tools

### Prerequisites

- OpenAI API access
- Understanding of function calling
- Python development environment

---

## Web Search Tool

### Basic Configuration

```python
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional, List

client = OpenAI()

# Simple web search
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search"}],
    input="What happened in tech news today?"
)

print(response.output_text)

# Check for citations in output
for item in response.output:
    if hasattr(item, 'type') and item.type == 'web_search_result':
        print(f"Source: {item.url}")
```

### Web Search with Configuration

```python
@dataclass
class WebSearchConfig:
    """Configuration for web search tool."""
    
    enabled: bool = True
    search_context_size: str = "medium"  # low, medium, high
    
    def to_tool(self) -> dict:
        """Convert to tool definition."""
        if not self.enabled:
            return None
        
        return {
            "type": "web_search",
            "search_context_size": self.search_context_size
        }


class WebSearchClient:
    """Client with web search capabilities."""
    
    def __init__(self, config: WebSearchConfig = None):
        self.client = OpenAI()
        self.config = config or WebSearchConfig()
    
    def search_and_answer(
        self,
        query: str,
        model: str = "gpt-4o"
    ) -> dict:
        """Search the web and answer a question."""
        
        tools = []
        if self.config.enabled:
            tool = self.config.to_tool()
            if tool:
                tools.append(tool)
        
        response = self.client.responses.create(
            model=model,
            tools=tools,
            input=query
        )
        
        # Extract search results
        sources = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'web_search_result':
                sources.append({
                    "url": getattr(item, 'url', ''),
                    "title": getattr(item, 'title', ''),
                    "snippet": getattr(item, 'snippet', '')
                })
        
        return {
            "answer": response.output_text,
            "sources": sources,
            "model": response.model
        }


# Usage
search_client = WebSearchClient(WebSearchConfig(
    search_context_size="high"
))

result = search_client.search_and_answer(
    "What are the latest AI safety developments?"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

---

## Code Interpreter

### Basic Usage

```python
# Enable code interpreter
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Calculate the first 20 Fibonacci numbers and plot them"
)

# Check for code execution results
for item in response.output:
    if hasattr(item, 'type'):
        if item.type == 'code_interpreter_result':
            print(f"Code output: {item.output}")
        elif item.type == 'image':
            print(f"Generated image: {item.image_url}")
```

### Code Interpreter with Files

```python
from pathlib import Path

class CodeInterpreterSession:
    """Session with code interpreter."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.file_ids: List[str] = []
    
    def upload_file(self, file_path: str) -> str:
        """Upload file for code interpreter."""
        
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(
                file=f,
                purpose="assistants"  # Code interpreter uses assistants files
            )
        
        self.file_ids.append(file_obj.id)
        return file_obj.id
    
    def execute(
        self,
        instruction: str,
        include_files: bool = True
    ) -> dict:
        """Execute code with interpreter."""
        
        tools = [{
            "type": "code_interpreter"
        }]
        
        # Build input with file references
        input_content = instruction
        if include_files and self.file_ids:
            input_content += f"\n\nFiles available: {', '.join(self.file_ids)}"
        
        response = self.client.responses.create(
            model=self.model,
            tools=tools,
            input=input_content
        )
        
        # Parse results
        code_outputs = []
        images = []
        
        for item in response.output:
            if hasattr(item, 'type'):
                if item.type == 'code_interpreter_result':
                    code_outputs.append(getattr(item, 'output', ''))
                elif item.type == 'image':
                    images.append(getattr(item, 'image_url', ''))
        
        return {
            "text": response.output_text,
            "code_outputs": code_outputs,
            "images": images
        }
    
    def cleanup(self):
        """Clean up uploaded files."""
        for file_id in self.file_ids:
            try:
                self.client.files.delete(file_id)
            except Exception:
                pass
        self.file_ids = []


# Usage
session = CodeInterpreterSession()

try:
    # Upload a CSV file
    # session.upload_file("data.csv")
    
    result = session.execute(
        "Analyze the data and create visualizations"
    )
    
    print(f"Analysis: {result['text']}")
    print(f"Generated {len(result['images'])} images")
finally:
    session.cleanup()
```

---

## File Search Tool

### Basic File Search

```python
# File search requires a vector store
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }],
    input="Find information about authentication in the documentation"
)

# Results include file references
for item in response.output:
    if hasattr(item, 'type') and item.type == 'file_search_result':
        print(f"File: {item.file_name}, Score: {item.score}")
```

### Vector Store Management

```python
class VectorStoreManager:
    """Manage vector stores for file search."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def create_store(self, name: str) -> str:
        """Create a new vector store."""
        
        store = self.client.vector_stores.create(name=name)
        return store.id
    
    def add_files(
        self,
        store_id: str,
        file_paths: List[str]
    ) -> List[str]:
        """Add files to vector store."""
        
        file_ids = []
        
        for path in file_paths:
            with open(path, "rb") as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
                file_ids.append(file_obj.id)
        
        # Add to vector store
        self.client.vector_stores.files.create_batch(
            vector_store_id=store_id,
            file_ids=file_ids
        )
        
        return file_ids
    
    def search(
        self,
        store_id: str,
        query: str,
        max_results: int = 10
    ) -> dict:
        """Search files with query."""
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [store_id],
                "max_num_results": max_results
            }],
            input=query
        )
        
        results = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                results.append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0),
                    "content": getattr(item, 'content', '')
                })
        
        return {
            "answer": response.output_text,
            "results": results
        }
    
    def delete_store(self, store_id: str):
        """Delete a vector store."""
        self.client.vector_stores.delete(store_id)


# Usage
manager = VectorStoreManager()

# Create store and add files
# store_id = manager.create_store("documentation")
# manager.add_files(store_id, ["doc1.pdf", "doc2.pdf"])

# Search
# result = manager.search(store_id, "How to authenticate?")
```

---

## Image Generation

### DALL-E Integration

```python
# Generate image with DALL-E
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "image_generation"}],
    input="Create an image of a futuristic city with flying cars"
)

# Extract generated images
for item in response.output:
    if hasattr(item, 'type') and item.type == 'image':
        print(f"Image URL: {item.image_url}")
```

### Image Generation Helper

```python
@dataclass
class ImageGenConfig:
    """Image generation configuration."""
    
    size: str = "1024x1024"  # 256x256, 512x512, 1024x1024
    quality: str = "standard"  # standard, hd
    style: str = "natural"  # natural, vivid
    
    def to_tool(self) -> dict:
        return {
            "type": "image_generation",
            "size": self.size,
            "quality": self.quality,
            "style": self.style
        }


class ImageGenerator:
    """Generate images using DALL-E."""
    
    def __init__(self, config: ImageGenConfig = None):
        self.client = OpenAI()
        self.config = config or ImageGenConfig()
    
    def generate(
        self,
        prompt: str,
        enhance_prompt: bool = True
    ) -> dict:
        """Generate image from prompt."""
        
        input_prompt = prompt
        if enhance_prompt:
            input_prompt = f"Create a detailed, high-quality image: {prompt}"
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[self.config.to_tool()],
            input=input_prompt
        )
        
        images = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'image':
                images.append({
                    "url": getattr(item, 'image_url', ''),
                    "revised_prompt": getattr(item, 'revised_prompt', prompt)
                })
        
        return {
            "images": images,
            "description": response.output_text
        }
    
    def generate_variations(
        self,
        base_prompt: str,
        variations: List[str]
    ) -> List[dict]:
        """Generate multiple variations of an image."""
        
        results = []
        
        for variation in variations:
            full_prompt = f"{base_prompt}, {variation}"
            result = self.generate(full_prompt)
            results.append({
                "variation": variation,
                **result
            })
        
        return results


# Usage
generator = ImageGenerator(ImageGenConfig(
    size="1024x1024",
    quality="hd",
    style="vivid"
))

result = generator.generate(
    "A serene mountain landscape at sunset"
)

print(f"Generated {len(result['images'])} images")
```

---

## Computer Use Preview

### OpenAI Computer Use (Beta)

```python
# Computer use for browser automation
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1920,
        "display_height": 1080
    }],
    input="Navigate to example.com and take a screenshot"
)

# Handle computer actions
for item in response.output:
    if hasattr(item, 'type') and item.type == 'computer_action':
        action = item.action
        print(f"Action: {action.type}")
        
        if action.type == 'click':
            print(f"  Click at ({action.x}, {action.y})")
        elif action.type == 'type':
            print(f"  Type: {action.text}")
        elif action.type == 'screenshot':
            print("  Take screenshot")
```

### Computer Use Handler

```python
from enum import Enum
from typing import Callable

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    KEY = "key"


@dataclass
class ComputerAction:
    """Computer action to execute."""
    
    action_type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    key: Optional[str] = None
    scroll_x: Optional[int] = None
    scroll_y: Optional[int] = None


class ComputerUseHandler:
    """Handle computer use tool actions."""
    
    def __init__(
        self,
        display_width: int = 1920,
        display_height: int = 1080,
        executor: Callable[[ComputerAction], Optional[bytes]] = None
    ):
        self.display_width = display_width
        self.display_height = display_height
        self.executor = executor or self._default_executor
        self.action_history: List[ComputerAction] = []
    
    def _default_executor(self, action: ComputerAction) -> Optional[bytes]:
        """Default executor (logs actions)."""
        print(f"Would execute: {action.action_type.value}")
        return None  # No screenshot
    
    def execute_response(self, response) -> List[dict]:
        """Execute actions from response."""
        
        results = []
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'computer_action':
                action = self._parse_action(item)
                screenshot = self.executor(action)
                
                self.action_history.append(action)
                
                results.append({
                    "action": action,
                    "screenshot": screenshot is not None
                })
        
        return results
    
    def _parse_action(self, item) -> ComputerAction:
        """Parse action from response item."""
        
        action_data = item.action
        action_type = ActionType(action_data.type)
        
        return ComputerAction(
            action_type=action_type,
            x=getattr(action_data, 'x', None),
            y=getattr(action_data, 'y', None),
            text=getattr(action_data, 'text', None),
            key=getattr(action_data, 'key', None),
            scroll_x=getattr(action_data, 'scroll_x', None),
            scroll_y=getattr(action_data, 'scroll_y', None)
        )
    
    def get_tool_config(self) -> dict:
        """Get tool configuration."""
        return {
            "type": "computer_use_preview",
            "display_width": self.display_width,
            "display_height": self.display_height
        }


# Usage
handler = ComputerUseHandler(
    display_width=1920,
    display_height=1080
)

# response = client.responses.create(
#     model="gpt-4o",
#     tools=[handler.get_tool_config()],
#     input="Click on the search button"
# )
# 
# results = handler.execute_response(response)
```

---

## Shell and Patch Tools

### Shell Command Execution

```python
# Shell tool for command execution
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "shell"
    }],
    input="List files in the current directory"
)

# Handle shell commands
for item in response.output:
    if hasattr(item, 'type') and item.type == 'shell_command':
        print(f"Command: {item.command}")
```

### Apply Patch Tool

```python
# Apply patch for code modifications
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "apply_patch"
    }],
    input="Fix the bug in the authentication function"
)

# Handle patches
for item in response.output:
    if hasattr(item, 'type') and item.type == 'patch':
        print(f"File: {item.file_path}")
        print(f"Patch:\n{item.patch_content}")
```

---

## Combining Multiple Tools

### Multi-Tool Configuration

```python
class MultiToolClient:
    """Client with multiple tools enabled."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def create_request(
        self,
        query: str,
        enable_web_search: bool = False,
        enable_code_interpreter: bool = False,
        enable_image_generation: bool = False,
        vector_store_id: Optional[str] = None
    ) -> dict:
        """Create request with selected tools."""
        
        tools = []
        
        if enable_web_search:
            tools.append({"type": "web_search"})
        
        if enable_code_interpreter:
            tools.append({"type": "code_interpreter"})
        
        if enable_image_generation:
            tools.append({"type": "image_generation"})
        
        if vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [vector_store_id]
            })
        
        response = self.client.responses.create(
            model=self.model,
            tools=tools,
            input=query
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response) -> dict:
        """Parse multi-tool response."""
        
        result = {
            "text": response.output_text,
            "web_results": [],
            "code_outputs": [],
            "images": [],
            "file_results": []
        }
        
        for item in response.output:
            if not hasattr(item, 'type'):
                continue
            
            if item.type == 'web_search_result':
                result["web_results"].append({
                    "url": getattr(item, 'url', ''),
                    "title": getattr(item, 'title', '')
                })
            elif item.type == 'code_interpreter_result':
                result["code_outputs"].append(getattr(item, 'output', ''))
            elif item.type == 'image':
                result["images"].append(getattr(item, 'image_url', ''))
            elif item.type == 'file_search_result':
                result["file_results"].append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0)
                })
        
        return result


# Usage
multi_client = MultiToolClient()

result = multi_client.create_request(
    query="Research the latest AI papers, analyze trends, and create a visualization",
    enable_web_search=True,
    enable_code_interpreter=True
)

print(f"Found {len(result['web_results'])} web sources")
print(f"Generated {len(result['code_outputs'])} code outputs")
```

---

## Hands-on Exercise

### Your Task

Build a research assistant with multiple tools.

### Requirements

1. Enable web search for current information
2. Use code interpreter for analysis
3. Generate visualizations
4. Combine results into a report

<details>
<summary>💡 Hints</summary>

- Configure tools based on query type
- Parse different output types
- Combine text, code, and images
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ResearchPhase(Enum):
    SEARCH = "search"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    SUMMARIZE = "summarize"


@dataclass
class ResearchResult:
    """Result from a research phase."""
    
    phase: ResearchPhase
    content: str
    sources: List[str] = field(default_factory=list)
    code_outputs: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


class ResearchAssistant:
    """Multi-tool research assistant."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.results: List[ResearchResult] = []
    
    def research(self, topic: str) -> dict:
        """Conduct full research on a topic."""
        
        # Phase 1: Search for information
        search_result = self._search_phase(topic)
        self.results.append(search_result)
        
        # Phase 2: Analyze findings
        analyze_result = self._analyze_phase(topic, search_result)
        self.results.append(analyze_result)
        
        # Phase 3: Visualize data
        viz_result = self._visualize_phase(topic, analyze_result)
        self.results.append(viz_result)
        
        # Phase 4: Summarize
        summary = self._summarize_phase()
        
        return {
            "topic": topic,
            "phases": len(self.results),
            "summary": summary,
            "all_sources": self._collect_sources(),
            "all_images": self._collect_images()
        }
    
    def _search_phase(self, topic: str) -> ResearchResult:
        """Search for current information."""
        
        response = self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=f"Research the latest developments on: {topic}. "
                  f"Find recent articles, papers, and news."
        )
        
        sources = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'web_search_result':
                sources.append(getattr(item, 'url', ''))
        
        return ResearchResult(
            phase=ResearchPhase.SEARCH,
            content=response.output_text,
            sources=sources
        )
    
    def _analyze_phase(
        self,
        topic: str,
        search_result: ResearchResult
    ) -> ResearchResult:
        """Analyze the search findings."""
        
        context = f"""
        Topic: {topic}
        
        Research findings:
        {search_result.content}
        
        Sources consulted: {len(search_result.sources)}
        """
        
        response = self.client.responses.create(
            model=self.model,
            tools=[{"type": "code_interpreter"}],
            input=f"""
            Analyze the following research findings:
            
            {context}
            
            Tasks:
            1. Extract key statistics and data points
            2. Identify trends and patterns
            3. Calculate any relevant metrics
            4. Prepare data for visualization
            """
        )
        
        code_outputs = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'code_interpreter_result':
                code_outputs.append(getattr(item, 'output', ''))
        
        return ResearchResult(
            phase=ResearchPhase.ANALYZE,
            content=response.output_text,
            code_outputs=code_outputs
        )
    
    def _visualize_phase(
        self,
        topic: str,
        analyze_result: ResearchResult
    ) -> ResearchResult:
        """Create visualizations."""
        
        response = self.client.responses.create(
            model=self.model,
            tools=[
                {"type": "code_interpreter"},
                {"type": "image_generation"}
            ],
            input=f"""
            Create visualizations for the analysis:
            
            {analyze_result.content}
            
            Tasks:
            1. Create charts showing key trends
            2. Generate an infographic-style summary image
            3. Plot any time-series data if available
            """
        )
        
        images = []
        code_outputs = []
        
        for item in response.output:
            if hasattr(item, 'type'):
                if item.type == 'image':
                    images.append(getattr(item, 'image_url', ''))
                elif item.type == 'code_interpreter_result':
                    code_outputs.append(getattr(item, 'output', ''))
        
        return ResearchResult(
            phase=ResearchPhase.VISUALIZE,
            content=response.output_text,
            code_outputs=code_outputs,
            images=images
        )
    
    def _summarize_phase(self) -> str:
        """Create final summary."""
        
        # Compile all findings
        all_content = "\n\n".join([
            f"## {r.phase.value.title()}\n{r.content}"
            for r in self.results
        ])
        
        response = self.client.responses.create(
            model=self.model,
            input=f"""
            Create an executive summary of this research:
            
            {all_content}
            
            Include:
            1. Key findings (3-5 bullet points)
            2. Main insights
            3. Recommendations
            4. Areas for further research
            """
        )
        
        return response.output_text
    
    def _collect_sources(self) -> List[str]:
        """Collect all sources."""
        sources = []
        for result in self.results:
            sources.extend(result.sources)
        return list(set(sources))
    
    def _collect_images(self) -> List[str]:
        """Collect all generated images."""
        images = []
        for result in self.results:
            images.extend(result.images)
        return images
    
    def generate_report(self) -> str:
        """Generate markdown report."""
        
        report = f"# Research Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        
        for result in self.results:
            report += f"## {result.phase.value.title()}\n\n"
            report += f"{result.content}\n\n"
            
            if result.sources:
                report += "### Sources\n"
                for source in result.sources:
                    report += f"- {source}\n"
                report += "\n"
            
            if result.images:
                report += "### Visualizations\n"
                for img in result.images:
                    report += f"![Visualization]({img})\n"
                report += "\n"
        
        return report


# Usage
assistant = ResearchAssistant()

# Conduct research
result = assistant.research("AI agents and autonomous systems 2025")

print(f"Research complete!")
print(f"Phases: {result['phases']}")
print(f"Sources: {len(result['all_sources'])}")
print(f"Images: {len(result['all_images'])}")

print("\n=== Summary ===")
print(result['summary'])

# Generate full report
report = assistant.generate_report()
print("\n=== Report ===")
print(report[:2000] + "...")  # First 2000 chars
```

</details>

---

## Summary

✅ Web search provides real-time information with citations  
✅ Code interpreter executes Python for analysis  
✅ File search enables RAG over your documents  
✅ Image generation creates visuals from descriptions  
✅ Computer use enables browser automation  
✅ Multiple tools can be combined in one request

**Next:** [File Search Configuration](./02-file-search-configuration.md)

---

## Further Reading

- [OpenAI Tools Guide](https://platform.openai.com/docs/guides/tools) — Official documentation
- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses) — API details
- [Assistants Tools](https://platform.openai.com/docs/assistants/tools) — Assistant-specific tools
