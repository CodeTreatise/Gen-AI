---
title: "Local API Compatibility"
---

# Local API Compatibility

## Introduction

Local AI servers can expose OpenAI-compatible endpoints, letting you use the same code for local and cloud models. This lesson covers drop-in replacement patterns and configuration.

### What We'll Cover

- OpenAI-compatible servers
- Drop-in replacement patterns
- Endpoint configuration
- Feature compatibility matrix

### Prerequisites

- Running models locally (previous lesson)
- OpenAI SDK basics
- Python environment setup

---

## OpenAI SDK with Local Models

The OpenAI Python SDK works with any OpenAI-compatible endpoint:

```python
from openai import OpenAI

# Point to local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but not used
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

**Output:**
```
Hello! How can I help you today?
```

---

## Compatible Servers

### Ollama

```python
# Ollama provides OpenAI-compatible endpoints at /v1
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```

### LM Studio

```python
# LM Studio default port is 1234
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
```

### vLLM

```python
# vLLM for high-throughput serving
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm"
)
```

### LocalAI

```python
# LocalAI multi-model server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="localai"
)
```

---

## Drop-in Replacement Pattern

```python
import os
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIProvider:
    """AI provider configuration."""
    
    name: str
    base_url: str
    api_key: str
    default_model: str


# Define providers
PROVIDERS = {
    "openai": AIProvider(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="gpt-4o"
    ),
    "ollama": AIProvider(
        name="Ollama",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        default_model="llama3.1"
    ),
    "lmstudio": AIProvider(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        default_model="local-model"
    ),
    "vllm": AIProvider(
        name="vLLM",
        base_url="http://localhost:8000/v1",
        api_key="vllm",
        default_model="meta-llama/Llama-3.1-8B-Instruct"
    )
}


class UnifiedAIClient:
    """Unified client for any OpenAI-compatible provider."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = None
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.provider = PROVIDERS[provider]
        self.model = model or self.provider.default_model
        
        self.client = OpenAI(
            base_url=self.provider.base_url,
            api_key=self.provider.api_key
        )
    
    def chat(
        self,
        messages: list,
        **kwargs
    ) -> str:
        """Chat completion."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def stream(
        self,
        messages: list,
        **kwargs
    ):
        """Streaming chat completion."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Usage - same code, different provider
def generate_response(prompt: str, provider: str = "ollama") -> str:
    """Generate response using any provider."""
    
    client = UnifiedAIClient(provider=provider)
    
    return client.chat([
        {"role": "user", "content": prompt}
    ])


# Development: use local
dev_response = generate_response("Hello!", provider="ollama")

# Production: use cloud
# prod_response = generate_response("Hello!", provider="openai")
```

---

## Environment-Based Configuration

```python
import os
from openai import OpenAI
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def get_ai_client() -> OpenAI:
    """Get AI client based on environment."""
    
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        # Local model for development
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    
    elif env == "staging":
        # Smaller cloud model for staging
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    else:
        # Full cloud model for production
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )


def get_default_model() -> str:
    """Get default model for environment."""
    
    env = os.getenv("ENVIRONMENT", "development")
    
    models = {
        "development": "llama3.1",
        "staging": "gpt-4o-mini",
        "production": "gpt-4o"
    }
    
    return models.get(env, "llama3.1")


# Usage
client = get_ai_client()
model = get_default_model()

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Feature Compatibility Matrix

Not all features work with all providers:

| Feature | OpenAI | Ollama | LM Studio | vLLM |
|---------|--------|--------|-----------|------|
| Chat completions | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Function calling | ✅ | ⚠️ | ⚠️ | ⚠️ |
| JSON mode | ✅ | ✅ | ⚠️ | ✅ |
| Vision | ✅ | ✅ | ⚠️ | ⚠️ |
| Embeddings | ✅ | ✅ | ❌ | ✅ |
| Structured output | ✅ | ⚠️ | ❌ | ⚠️ |

⚠️ = Partial or model-dependent support

### Feature Detection

```python
from typing import Dict, Any

class FeatureDetector:
    """Detect supported features for a provider."""
    
    FEATURE_TESTS = {
        "streaming": lambda c, m: test_streaming(c, m),
        "json_mode": lambda c, m: test_json_mode(c, m),
        "function_calling": lambda c, m: test_function_calling(c, m),
        "embeddings": lambda c, m: test_embeddings(c, m),
    }
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self._cache: Dict[str, bool] = {}
    
    def supports(self, feature: str) -> bool:
        """Check if feature is supported."""
        
        if feature in self._cache:
            return self._cache[feature]
        
        if feature not in self.FEATURE_TESTS:
            return False
        
        try:
            result = self.FEATURE_TESTS[feature](self.client, self.model)
            self._cache[feature] = result
            return result
        except Exception:
            self._cache[feature] = False
            return False
    
    def get_all_features(self) -> Dict[str, bool]:
        """Test all features."""
        
        return {
            feature: self.supports(feature)
            for feature in self.FEATURE_TESTS
        }


def test_streaming(client: OpenAI, model: str) -> bool:
    """Test streaming support."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=5
        )
        
        for chunk in response:
            pass
        
        return True
    except Exception:
        return False


def test_json_mode(client: OpenAI, model: str) -> bool:
    """Test JSON mode support."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Return JSON: {\"test\": true}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=20
        )
        return True
    except Exception:
        return False


def test_function_calling(client: OpenAI, model: str) -> bool:
    """Test function calling support."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            max_tokens=20
        )
        return True
    except Exception:
        return False


def test_embeddings(client: OpenAI, model: str) -> bool:
    """Test embeddings support."""
    
    try:
        # Use a common embedding model name
        response = client.embeddings.create(
            model="nomic-embed-text",
            input="test"
        )
        return True
    except Exception:
        return False
```

---

## Fallback Patterns

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FallbackClient:
    """Client with automatic fallback."""
    
    def __init__(self, providers: List[str]):
        self.providers = providers
        self.clients = {}
        
        for provider in providers:
            try:
                self.clients[provider] = UnifiedAIClient(provider=provider)
            except Exception as e:
                logger.warning(f"Failed to init {provider}: {e}")
    
    def chat(
        self,
        messages: list,
        preferred_provider: str = None,
        **kwargs
    ) -> tuple[str, str]:
        """Chat with fallback. Returns (response, provider_used)."""
        
        # Try preferred first
        if preferred_provider and preferred_provider in self.clients:
            try:
                response = self.clients[preferred_provider].chat(
                    messages, **kwargs
                )
                return response, preferred_provider
            except Exception as e:
                logger.warning(f"{preferred_provider} failed: {e}")
        
        # Try others in order
        for provider in self.providers:
            if provider == preferred_provider:
                continue
            
            if provider not in self.clients:
                continue
            
            try:
                response = self.clients[provider].chat(messages, **kwargs)
                return response, provider
            except Exception as e:
                logger.warning(f"{provider} failed: {e}")
        
        raise RuntimeError("All providers failed")


# Usage
client = FallbackClient(["ollama", "openai"])

response, used = client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    preferred_provider="ollama"  # Try local first
)

print(f"Response from {used}: {response}")
```

---

## Endpoint Configuration

### Custom Base URLs

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EndpointConfig:
    """Endpoint configuration."""
    
    base_url: str
    api_key: str = "not-needed"
    timeout: float = 60.0
    max_retries: int = 2
    
    # Optional proxy
    proxy: Optional[str] = None
    
    # Custom headers
    headers: Optional[dict] = None


def create_client(config: EndpointConfig) -> OpenAI:
    """Create client from config."""
    
    kwargs = {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "timeout": config.timeout,
        "max_retries": config.max_retries
    }
    
    if config.headers:
        kwargs["default_headers"] = config.headers
    
    return OpenAI(**kwargs)


# Remote Ollama server
remote_config = EndpointConfig(
    base_url="http://192.168.1.100:11434/v1",
    api_key="remote-ollama"
)

# Behind nginx proxy
proxy_config = EndpointConfig(
    base_url="https://ai.internal.company.com/v1",
    api_key="internal-key",
    headers={"X-Internal-Auth": "token123"}
)
```

### Docker Network Configuration

```python
import os

def get_docker_endpoint() -> str:
    """Get endpoint for Docker environment."""
    
    # Check if running in Docker
    if os.path.exists("/.dockerenv"):
        # Use Docker network name
        return "http://ollama:11434/v1"
    
    # Local development
    return "http://localhost:11434/v1"


# docker-compose.yml
"""
services:
  app:
    build: .
    depends_on:
      - ollama
    environment:
      - AI_ENDPOINT=http://ollama:11434/v1
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
"""
```

---

## Hands-on Exercise

### Your Task

Build a provider-agnostic AI client with automatic feature detection.

### Requirements

1. Support multiple providers
2. Detect available features
3. Graceful degradation for unsupported features
4. Environment-based configuration

<details>
<summary>💡 Hints</summary>

- Cache feature detection results
- Use try/except for feature tests
- Environment variables for config
</details>

<details>
<summary>✅ Solution</summary>

```python
import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum

class Feature(Enum):
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    default_model: str
    known_features: List[Feature] = field(default_factory=list)


class SmartAIClient:
    """Provider-agnostic client with feature detection."""
    
    PROVIDERS = {
        "openai": ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            default_model="gpt-4o-mini",
            known_features=[
                Feature.STREAMING,
                Feature.JSON_MODE,
                Feature.FUNCTION_CALLING,
                Feature.EMBEDDINGS,
                Feature.VISION
            ]
        ),
        "ollama": ProviderConfig(
            name="Ollama",
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            default_model="llama3.1",
            known_features=[
                Feature.STREAMING,
                Feature.JSON_MODE,
                Feature.EMBEDDINGS
            ]
        ),
        "lmstudio": ProviderConfig(
            name="LM Studio",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            default_model="local-model",
            known_features=[
                Feature.STREAMING
            ]
        )
    }
    
    def __init__(
        self,
        provider: str = None,
        model: str = None
    ):
        # Auto-select provider based on environment
        if provider is None:
            provider = self._auto_select_provider()
        
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.config = self.PROVIDERS[provider]
        self.model = model or self.config.default_model
        self.provider_name = provider
        
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        
        self._feature_cache: Dict[Feature, bool] = {}
    
    def _auto_select_provider(self) -> str:
        """Auto-select provider based on environment."""
        
        env = os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            return "openai"
        
        # Try local providers
        for provider in ["ollama", "lmstudio"]:
            if self._is_available(provider):
                return provider
        
        # Fall back to OpenAI
        return "openai"
    
    def _is_available(self, provider: str) -> bool:
        """Check if provider is available."""
        
        import requests
        
        config = self.PROVIDERS.get(provider)
        if not config:
            return False
        
        try:
            response = requests.get(
                f"{config.base_url.rstrip('/v1')}/api/tags",
                timeout=1
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def supports(self, feature: Feature) -> bool:
        """Check if feature is supported."""
        
        if feature in self._feature_cache:
            return self._feature_cache[feature]
        
        # Check known features first
        if feature in self.config.known_features:
            self._feature_cache[feature] = True
            return True
        
        # Test dynamically
        result = self._test_feature(feature)
        self._feature_cache[feature] = result
        
        return result
    
    def _test_feature(self, feature: Feature) -> bool:
        """Test feature support."""
        
        try:
            if feature == Feature.STREAMING:
                return self._test_streaming()
            elif feature == Feature.JSON_MODE:
                return self._test_json_mode()
            elif feature == Feature.FUNCTION_CALLING:
                return self._test_function_calling()
            else:
                return False
        except Exception:
            return False
    
    def _test_streaming(self) -> bool:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=5
        )
        for _ in response:
            return True
        return False
    
    def _test_json_mode(self) -> bool:
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "{}"}],
            response_format={"type": "json_object"},
            max_tokens=10
        )
        return True
    
    def _test_function_calling(self) -> bool:
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Test"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            max_tokens=10
        )
        return True
    
    def chat(
        self,
        messages: list,
        json_mode: bool = False,
        stream: bool = False,
        **kwargs
    ):
        """Chat with graceful degradation."""
        
        params = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        # JSON mode with fallback
        if json_mode:
            if self.supports(Feature.JSON_MODE):
                params["response_format"] = {"type": "json_object"}
            else:
                # Add instruction to return JSON
                messages = messages.copy()
                messages[-1]["content"] += "\n\nRespond in valid JSON format."
                params["messages"] = messages
        
        # Streaming with fallback
        if stream:
            if self.supports(Feature.STREAMING):
                params["stream"] = True
                return self._stream_response(params)
            else:
                # Simulate streaming
                response = self.client.chat.completions.create(**params)
                return iter([response.choices[0].message.content])
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _stream_response(self, params):
        """Stream response."""
        
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def get_info(self) -> dict:
        """Get client info."""
        
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.config.base_url,
            "features": {
                f.value: self.supports(f)
                for f in Feature
            }
        }


# Test it
def test_smart_client():
    # Auto-select provider
    client = SmartAIClient()
    
    print(f"Using: {client.provider_name}")
    print(f"Model: {client.model}")
    
    # Check features
    info = client.get_info()
    print(f"\nFeatures:")
    for feature, supported in info["features"].items():
        status = "✅" if supported else "❌"
        print(f"  {status} {feature}")
    
    # Chat
    response = client.chat([
        {"role": "user", "content": "What is 2+2?"}
    ])
    print(f"\nResponse: {response}")
    
    # Streaming (with fallback)
    print("\nStreaming: ", end="")
    for chunk in client.chat(
        [{"role": "user", "content": "Count to 5"}],
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()


test_smart_client()
```

</details>

---

## Summary

✅ OpenAI SDK works with any compatible endpoint  
✅ Change `base_url` for different providers  
✅ Use environment variables for configuration  
✅ Implement fallbacks for reliability  
✅ Detect and adapt to feature support

**Next:** [Privacy Benefits](./03-privacy-benefits.md)

---

## Further Reading

- [OpenAI SDK](https://github.com/openai/openai-python) — Python client
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) — API documentation
- [LM Studio Server](https://lmstudio.ai/docs/local-server) — Server docs
