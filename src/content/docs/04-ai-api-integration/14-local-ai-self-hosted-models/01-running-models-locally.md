---
title: "Running Models Locally"
---

# Running Models Locally

## Introduction

Local AI inference starts with getting models running on your machine. This lesson covers Ollama, LM Studio, model downloading, and management.

### What We'll Cover

- Ollama installation and setup
- LM Studio for GUI interface
- Model downloading
- Running inference locally
- Model management

### Prerequisites

- 8GB+ RAM (16GB+ recommended)
- SSD with 20GB+ free space
- macOS, Linux, or Windows

---

## Ollama

Ollama is the easiest way to run models locally with a CLI and API server.

### Installation

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell)
winget install Ollama.Ollama

# Or download from https://ollama.com/download
```

**Verify installation:**

```bash
ollama --version
```

**Output:**
```
ollama version 0.1.47
```

### Running Your First Model

```bash
# Pull and run Llama 3.1 8B
ollama run llama3.1
```

This:
1. Downloads the model (~4.7GB)
2. Loads it into memory
3. Starts an interactive chat

**Example session:**
```
>>> What is the capital of France?
The capital of France is Paris.

>>> /bye
```

### Ollama Commands

| Command | Description |
|---------|-------------|
| `ollama run <model>` | Run model interactively |
| `ollama pull <model>` | Download model |
| `ollama list` | List downloaded models |
| `ollama rm <model>` | Delete model |
| `ollama show <model>` | Show model details |
| `ollama serve` | Start API server |

### Available Models

```bash
# List available models
ollama list

# Popular models
ollama pull llama3.1        # Meta's Llama 3.1 8B
ollama pull llama3.1:70b    # Llama 3.1 70B (needs lots of VRAM)
ollama pull codellama       # Code-focused
ollama pull mistral         # Mistral 7B
ollama pull phi3            # Microsoft Phi-3
ollama pull gemma2          # Google Gemma 2
ollama pull qwen2           # Alibaba Qwen 2
```

### Ollama API Server

Ollama runs an API server on port 11434:

```bash
# Start server (runs automatically)
ollama serve
```

**Generate endpoint:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Explain recursion in one sentence.",
  "stream": false
}'
```

**Output:**
```json
{
  "model": "llama3.1",
  "response": "Recursion is when a function calls itself...",
  "done": true
}
```

**Chat endpoint:**

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}'
```

### Python Integration

```python
import requests
import json

def ollama_generate(prompt: str, model: str = "llama3.1") -> str:
    """Generate with Ollama."""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    return response.json()["response"]


def ollama_chat(messages: list, model: str = "llama3.1") -> str:
    """Chat with Ollama."""
    
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    
    return response.json()["message"]["content"]


# Usage
response = ollama_generate("What is 2 + 2?")
print(response)

chat_response = ollama_chat([
    {"role": "user", "content": "Write a haiku about programming"}
])
print(chat_response)
```

### Streaming Responses

```python
import requests
import json

def ollama_stream(prompt: str, model: str = "llama3.1"):
    """Stream response from Ollama."""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data.get("response", "")
            
            if data.get("done"):
                break


# Usage
for chunk in ollama_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## LM Studio

LM Studio provides a GUI for local AI, great for exploration and testing.

### Installation

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch
3. Browse and download models from the interface

### Features

- **Model Browser** — Search and download from Hugging Face
- **Chat Interface** — Interactive chat with models
- **Local Server** — OpenAI-compatible API
- **Model Management** — Easy model switching

### Starting the Local Server

1. Open LM Studio
2. Load a model
3. Click "Local Server" tab
4. Start server (default port 1234)

```bash
# Test the server
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Model Downloading

### From Hugging Face

```bash
# Ollama can import GGUF models
ollama create mymodel -f ./Modelfile

# Modelfile example
# FROM ./model.gguf
# PARAMETER temperature 0.7
# SYSTEM "You are a helpful assistant."
```

### GGUF Format

GGUF (GPT-Generated Unified Format) is the standard for quantized models:

```bash
# Download from Hugging Face
pip install huggingface-hub

huggingface-cli download \
  TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ./models
```

### Model Naming Convention

```
llama-2-7b-chat.Q4_K_M.gguf
│      │  │     │
│      │  │     └── Quantization type
│      │  └── Size (7 billion)
│      └── Version
└── Base model
```

**Quantization types:**

| Type | Size | Quality |
|------|------|---------|
| Q2_K | Smallest | Lowest |
| Q3_K_M | Small | Low |
| Q4_K_M | Medium | Good balance |
| Q5_K_M | Large | Better |
| Q6_K | Larger | Near-full |
| Q8_0 | Large | Near-lossless |
| F16 | Largest | Full precision |

---

## Model Management

### Ollama Model Storage

```bash
# Default locations
# macOS: ~/.ollama/models
# Linux: /usr/share/ollama/.ollama/models
# Windows: C:\Users\<user>\.ollama\models

# Check disk usage
du -sh ~/.ollama/models/*
```

### Creating Custom Models

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM llama3.1

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# Set system prompt
SYSTEM """
You are a helpful coding assistant. 
Provide clear, concise code examples.
"""
EOF

# Create the model
ollama create code-assistant -f Modelfile

# Run it
ollama run code-assistant
```

### Model Configuration

```python
from dataclasses import dataclass
from typing import Optional
import requests

@dataclass
class ModelConfig:
    """Configuration for local model."""
    
    name: str
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096  # Context window
    num_predict: int = -1  # Max tokens (-1 = unlimited)
    stop: Optional[list] = None


class OllamaClient:
    """Configurable Ollama client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        config: ModelConfig = None
    ):
        self.base_url = base_url
        self.config = config or ModelConfig(name="llama3.1")
    
    def generate(
        self,
        prompt: str,
        config: ModelConfig = None
    ) -> str:
        """Generate with configuration."""
        
        cfg = config or self.config
        
        payload = {
            "model": cfg.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_ctx": cfg.num_ctx,
                "num_predict": cfg.num_predict
            }
        }
        
        if cfg.stop:
            payload["options"]["stop"] = cfg.stop
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        
        return response.json()["response"]
    
    def list_models(self) -> list:
        """List available models."""
        
        response = requests.get(f"{self.base_url}/api/tags")
        return [m["name"] for m in response.json()["models"]]


# Usage
client = OllamaClient(config=ModelConfig(
    name="llama3.1",
    temperature=0.3,  # More deterministic
    num_ctx=8192  # Larger context
))

response = client.generate("Explain async/await in Python")
```

---

## Running Multiple Models

```python
from typing import Dict
import requests

class ModelPool:
    """Manage multiple local models."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.loaded: Dict[str, bool] = {}
    
    def ensure_loaded(self, model: str):
        """Ensure model is loaded."""
        
        if model not in self.loaded:
            # Pull if not present
            requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model}
            )
            self.loaded[model] = True
    
    def generate(
        self,
        model: str,
        prompt: str
    ) -> str:
        """Generate with specific model."""
        
        self.ensure_loaded(model)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        return response.json()["response"]
    
    def compare_models(
        self,
        models: list,
        prompt: str
    ) -> Dict[str, str]:
        """Compare responses across models."""
        
        results = {}
        
        for model in models:
            try:
                results[model] = self.generate(model, prompt)
            except Exception as e:
                results[model] = f"Error: {e}"
        
        return results


# Usage
pool = ModelPool()

responses = pool.compare_models(
    models=["llama3.1", "mistral", "phi3"],
    prompt="What is machine learning?"
)

for model, response in responses.items():
    print(f"\n=== {model} ===")
    print(response[:200] + "...")
```

---

## Hands-on Exercise

### Your Task

Build a local AI chat application with model switching.

### Requirements

1. Support multiple models
2. Maintain conversation history
3. Allow model switching mid-conversation
4. Handle errors gracefully

<details>
<summary>💡 Hints</summary>

- Store messages per conversation
- Check model availability before switching
- Use try/except for API errors
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import requests

@dataclass
class Message:
    role: str
    content: str


@dataclass  
class Conversation:
    id: str
    model: str
    messages: List[Message] = field(default_factory=list)


class LocalChatApp:
    """Local AI chat with model switching."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1"
    ):
        self.base_url = base_url
        self.default_model = default_model
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation: Optional[str] = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return [m["name"] for m in response.json()["models"]]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def new_conversation(
        self,
        conversation_id: str,
        model: str = None
    ) -> str:
        """Start a new conversation."""
        
        model = model or self.default_model
        
        self.conversations[conversation_id] = Conversation(
            id=conversation_id,
            model=model
        )
        
        self.current_conversation = conversation_id
        
        return conversation_id
    
    def switch_model(
        self,
        model: str,
        conversation_id: str = None
    ) -> bool:
        """Switch model for conversation."""
        
        conv_id = conversation_id or self.current_conversation
        
        if not conv_id or conv_id not in self.conversations:
            print("No active conversation")
            return False
        
        # Check model exists
        available = self.get_available_models()
        
        if model not in available:
            print(f"Model {model} not available. Available: {available}")
            return False
        
        self.conversations[conv_id].model = model
        print(f"Switched to {model}")
        
        return True
    
    def chat(
        self,
        message: str,
        conversation_id: str = None
    ) -> str:
        """Send message and get response."""
        
        conv_id = conversation_id or self.current_conversation
        
        if not conv_id:
            conv_id = self.new_conversation("default")
        
        conv = self.conversations[conv_id]
        
        # Add user message
        conv.messages.append(Message(role="user", content=message))
        
        # Build messages for API
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in conv.messages
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": conv.model,
                    "messages": api_messages,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code != 200:
                return f"Error: {response.text}"
            
            assistant_message = response.json()["message"]["content"]
            
            # Add assistant message
            conv.messages.append(Message(
                role="assistant",
                content=assistant_message
            ))
            
            return assistant_message
            
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Is it running?"
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        except Exception as e:
            return f"Error: {e}"
    
    def interactive_chat(self):
        """Run interactive chat session."""
        
        print("Local AI Chat")
        print("Commands: /models, /switch <model>, /clear, /exit")
        print("-" * 40)
        
        self.new_conversation("interactive")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/exit":
                    break
                
                if user_input == "/models":
                    models = self.get_available_models()
                    print(f"Available: {models}")
                    continue
                
                if user_input.startswith("/switch "):
                    model = user_input.split(" ", 1)[1]
                    self.switch_model(model)
                    continue
                
                response = self.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break


# Run it
if __name__ == "__main__":
    app = LocalChatApp()
    app.interactive_chat()
```

</details>

---

## Summary

✅ Ollama provides easy CLI and API access  
✅ LM Studio offers a GUI alternative  
✅ GGUF is the standard quantized format  
✅ Modelfiles customize model behavior  
✅ Multiple models can run from one server

**Next:** [Local API Compatibility](./02-local-api-compatibility.md)

---

## Further Reading

- [Ollama Documentation](https://ollama.com/) — Official docs
- [LM Studio](https://lmstudio.ai/) — GUI tool
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — File format spec

<!-- 
Sources Consulted:
- Ollama: https://ollama.com/
- LM Studio: https://lmstudio.ai/
-->
