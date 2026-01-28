---
title: "Anthropic Caching"
---

# Anthropic Caching

## Introduction

Anthropic's Claude models use an explicit caching system with `cache_control` blocks. Unlike OpenAI's automatic caching, Anthropic requires you to specify exactly which content to cache, giving you fine-grained control over caching behavior.

### What We'll Cover

- `cache_control` block syntax
- Ephemeral cache type
- TTL options (5 minutes, 1 hour)
- Cache creation tracking
- Multi-block caching strategies

### Prerequisites

- Anthropic API access
- Understanding of cache fundamentals
- Python development environment

---

## Cache Control Blocks

### Basic Syntax

```python
from anthropic import Anthropic

client = Anthropic()

# Cache control block marks content for caching
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert Python tutor with deep knowledge...",  # Large content
            "cache_control": {"type": "ephemeral"}  # Mark for caching
        }
    ],
    messages=[
        {"role": "user", "content": "What is a decorator?"}
    ]
)

# Check cache usage
usage = response.usage
print(f"Input tokens: {usage.input_tokens}")
print(f"Cache creation: {usage.cache_creation_input_tokens}")
print(f"Cache read: {usage.cache_read_input_tokens}")
```

### Cache Control Structure

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class CacheControl:
    type: Literal["ephemeral"]
    ttl: Optional[int] = None  # TTL in seconds (if supported)


@dataclass  
class CachedTextBlock:
    type: Literal["text"]
    text: str
    cache_control: CacheControl


def create_cached_block(text: str, ttl: Optional[int] = None) -> dict:
    """Create a text block with cache control."""
    
    block = {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"}
    }
    
    if ttl is not None:
        block["cache_control"]["ttl"] = ttl
    
    return block


# Usage
system_block = create_cached_block(
    text="Comprehensive system instructions here...",
    ttl=3600  # 1 hour
)

print(system_block)
```

---

## Ephemeral Cache Type

### Understanding Ephemeral Caching

```python
from enum import Enum
from dataclasses import dataclass

class CacheType(Enum):
    EPHEMERAL = "ephemeral"  # Only type currently supported


@dataclass
class EphemeralCacheConfig:
    """Configuration for ephemeral cache."""
    
    cache_type: CacheType = CacheType.EPHEMERAL
    default_ttl_minutes: int = 5
    max_ttl_hours: int = 1
    min_tokens: int = 1024  # Minimum for caching
    
    def get_cache_control(self, ttl_minutes: int = 5) -> dict:
        """Generate cache_control block."""
        
        ttl_seconds = min(ttl_minutes * 60, self.max_ttl_hours * 3600)
        
        return {
            "type": self.cache_type.value,
            "ttl": ttl_seconds
        }


# Default configuration
EPHEMERAL_CONFIG = EphemeralCacheConfig()

# Usage
cache_control = EPHEMERAL_CONFIG.get_cache_control(ttl_minutes=60)
print(f"Cache control: {cache_control}")
```

### When to Use Ephemeral

```python
from typing import List

USE_CASES = {
    "long_context": {
        "description": "Large documents or knowledge bases",
        "recommended_ttl": 3600,  # 1 hour
        "example": "RAG with consistent document corpus"
    },
    "system_prompts": {
        "description": "Stable system instructions",
        "recommended_ttl": 3600,
        "example": "Chatbot personality and rules"
    },
    "few_shot": {
        "description": "Static examples for in-context learning",
        "recommended_ttl": 3600,
        "example": "Classification with labeled examples"
    },
    "tool_definitions": {
        "description": "Function/tool schemas",
        "recommended_ttl": 3600,
        "example": "Agent with consistent toolset"
    },
    "conversation_prefix": {
        "description": "Recent conversation history",
        "recommended_ttl": 300,  # 5 minutes
        "example": "Multi-turn chat sessions"
    }
}


def recommend_ttl(use_case: str) -> int:
    """Recommend TTL based on use case."""
    
    config = USE_CASES.get(use_case, {})
    return config.get("recommended_ttl", 300)


# Example
for case, config in USE_CASES.items():
    print(f"{case}: TTL={config['recommended_ttl']}s - {config['description']}")
```

---

## TTL Options

### 5-Minute TTL (Default)

```python
# Default 5-minute TTL
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "Your system prompt...",
            "cache_control": {"type": "ephemeral"}  # 5-minute default
        }
    ],
    messages=[{"role": "user", "content": "Query"}]
)
```

### Extended TTL (1 Hour)

```python
# Extended 1-hour TTL
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "Your system prompt...",
            "cache_control": {
                "type": "ephemeral",
                "ttl": 3600  # 1 hour in seconds
            }
        }
    ],
    messages=[{"role": "user", "content": "Query"}]
)
```

### TTL Strategy

```python
from dataclasses import dataclass
from enum import Enum

class TTLDuration(Enum):
    SHORT = 300      # 5 minutes
    MEDIUM = 1800    # 30 minutes
    LONG = 3600      # 1 hour


@dataclass
class TTLStrategy:
    duration: TTLDuration
    use_when: str
    cost_consideration: str


TTL_STRATEGIES = [
    TTLStrategy(
        duration=TTLDuration.SHORT,
        use_when="Bursty traffic, short sessions, frequently changing content",
        cost_consideration="Lower write costs, but more cache misses over time"
    ),
    TTLStrategy(
        duration=TTLDuration.MEDIUM,
        use_when="Moderate session length, semi-stable content",
        cost_consideration="Balanced between write costs and hit rate"
    ),
    TTLStrategy(
        duration=TTLDuration.LONG,
        use_when="Long sessions, stable content, high request volume",
        cost_consideration="Higher upfront write cost, but better long-term savings"
    ),
]


def select_ttl(
    requests_per_hour: float,
    content_stability_hours: float,
    session_duration_minutes: float
) -> TTLDuration:
    """Select optimal TTL based on usage pattern."""
    
    # Long TTL for stable, high-volume scenarios
    if content_stability_hours >= 24 and requests_per_hour >= 10:
        return TTLDuration.LONG
    
    # Medium TTL for moderate usage
    if session_duration_minutes >= 30 or requests_per_hour >= 5:
        return TTLDuration.MEDIUM
    
    # Short TTL for bursty or short sessions
    return TTLDuration.SHORT


# Example
ttl = select_ttl(
    requests_per_hour=20,
    content_stability_hours=48,
    session_duration_minutes=60
)

print(f"Selected TTL: {ttl.value} seconds ({ttl.name})")
```

---

## Cache Usage Tracking

### Understanding Usage Metrics

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    
    @property
    def total_input(self) -> int:
        """Total input including cache operations."""
        return self.input_tokens + self.cache_creation_input_tokens
    
    @property
    def cache_hit_rate(self) -> float:
        """Rate of tokens read from cache."""
        total = self.input_tokens + self.cache_read_input_tokens
        if total == 0:
            return 0.0
        return self.cache_read_input_tokens / total
    
    @property
    def is_cache_hit(self) -> bool:
        """Whether this request had a cache hit."""
        return self.cache_read_input_tokens > 0
    
    @property
    def is_cache_write(self) -> bool:
        """Whether this request wrote to cache."""
        return self.cache_creation_input_tokens > 0


def parse_cache_usage(response) -> CacheUsage:
    """Parse cache usage from response."""
    
    usage = response.usage
    
    return CacheUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=getattr(usage, 'cache_creation_input_tokens', 0),
        cache_read_input_tokens=getattr(usage, 'cache_read_input_tokens', 0)
    )


def analyze_cache_status(cache_usage: CacheUsage) -> str:
    """Analyze cache status from usage."""
    
    if cache_usage.is_cache_hit and cache_usage.is_cache_write:
        return "PARTIAL_HIT"  # Some cached, some new
    elif cache_usage.is_cache_hit:
        return "CACHE_HIT"
    elif cache_usage.is_cache_write:
        return "CACHE_WRITE"
    else:
        return "NO_CACHE"  # Content not marked for caching


# Usage example
# response = client.messages.create(...)
# usage = parse_cache_usage(response)
# print(f"Status: {analyze_cache_status(usage)}")
# print(f"Hit rate: {usage.cache_hit_rate:.1%}")
```

### Tracking Cache Performance

```python
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class CacheEvent:
    timestamp: datetime
    usage: CacheUsage
    content_hash: str


@dataclass
class CacheTracker:
    """Track cache performance over time."""
    
    events: List[CacheEvent] = field(default_factory=list)
    
    def record(self, usage: CacheUsage, content_hash: str = ""):
        """Record a cache event."""
        self.events.append(CacheEvent(
            timestamp=datetime.now(),
            usage=usage,
            content_hash=content_hash
        ))
    
    def get_stats(self) -> dict:
        """Calculate overall cache statistics."""
        
        if not self.events:
            return {"error": "No events recorded"}
        
        total_input = sum(e.usage.input_tokens for e in self.events)
        total_cached = sum(e.usage.cache_read_input_tokens for e in self.events)
        total_written = sum(e.usage.cache_creation_input_tokens for e in self.events)
        total_output = sum(e.usage.output_tokens for e in self.events)
        
        hits = sum(1 for e in self.events if e.usage.is_cache_hit)
        writes = sum(1 for e in self.events if e.usage.is_cache_write)
        
        return {
            "total_requests": len(self.events),
            "cache_hits": hits,
            "cache_writes": writes,
            "hit_rate": hits / len(self.events) if self.events else 0,
            "total_input_tokens": total_input,
            "total_cached_tokens": total_cached,
            "total_written_tokens": total_written,
            "total_output_tokens": total_output,
            "token_cache_rate": total_cached / (total_input + total_cached) if (total_input + total_cached) > 0 else 0
        }
    
    def estimate_savings(self, model: str = "claude-sonnet-4-20250514") -> dict:
        """Estimate cost savings from caching."""
        
        # Claude pricing (per 1M tokens)
        PRICING = {
            "claude-sonnet-4-20250514": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,  # 25% premium
                "cache_read": 0.30    # 90% discount
            },
            "claude-3-5-sonnet-20241022": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30
            }
        }
        
        rates = PRICING.get(model, PRICING["claude-sonnet-4-20250514"])
        stats = self.get_stats()
        
        # Calculate actual cost
        input_cost = (stats["total_input_tokens"] / 1_000_000) * rates["input"]
        output_cost = (stats["total_output_tokens"] / 1_000_000) * rates["output"]
        write_cost = (stats["total_written_tokens"] / 1_000_000) * rates["cache_write"]
        read_cost = (stats["total_cached_tokens"] / 1_000_000) * rates["cache_read"]
        
        actual_cost = input_cost + output_cost + write_cost + read_cost
        
        # Calculate cost without caching
        total_would_be_input = stats["total_input_tokens"] + stats["total_cached_tokens"]
        full_cost = (
            (total_would_be_input / 1_000_000) * rates["input"] +
            (stats["total_output_tokens"] / 1_000_000) * rates["output"]
        )
        
        savings = full_cost - actual_cost
        
        return {
            "model": model,
            "actual_cost": f"${actual_cost:.4f}",
            "without_cache": f"${full_cost:.4f}",
            "savings": f"${savings:.4f}",
            "savings_percent": f"{(savings / full_cost * 100):.1f}%" if full_cost > 0 else "0%"
        }


# Usage
tracker = CacheTracker()

# Simulate tracking
# for response in responses:
#     usage = parse_cache_usage(response)
#     tracker.record(usage)

# print(tracker.get_stats())
# print(tracker.estimate_savings())
```

---

## Multi-Block Caching

### Caching Multiple Content Blocks

```python
def create_multi_block_request(
    client,
    system_blocks: List[dict],
    messages: List[dict],
    max_tokens: int = 1024
):
    """Create request with multiple cached blocks."""
    
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system_blocks,
        messages=messages
    )


# Example with multiple cached blocks
system_blocks = [
    # Block 1: Core instructions (cached)
    {
        "type": "text",
        "text": "You are an expert assistant. " * 100,  # Large block
        "cache_control": {"type": "ephemeral", "ttl": 3600}
    },
    # Block 2: Knowledge base (cached)
    {
        "type": "text",
        "text": "Reference knowledge: " + "Important facts... " * 200,
        "cache_control": {"type": "ephemeral", "ttl": 3600}
    },
    # Block 3: Dynamic context (not cached)
    {
        "type": "text",
        "text": f"Today's date: {datetime.now().isoformat()}"
        # No cache_control - this changes frequently
    }
]

messages = [{"role": "user", "content": "Answer my question"}]

response = create_multi_block_request(client, system_blocks, messages)
```

### Cache Block Strategy

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContentBlock:
    content: str
    should_cache: bool
    ttl_seconds: Optional[int] = None
    description: str = ""
    
    def to_block(self) -> dict:
        """Convert to API block format."""
        block = {
            "type": "text",
            "text": self.content
        }
        
        if self.should_cache:
            cache_control = {"type": "ephemeral"}
            if self.ttl_seconds:
                cache_control["ttl"] = self.ttl_seconds
            block["cache_control"] = cache_control
        
        return block


class ContentBlockBuilder:
    """Build optimized content blocks for caching."""
    
    def __init__(self):
        self.blocks: List[ContentBlock] = []
    
    def add_static(
        self,
        content: str,
        description: str = "",
        ttl_hours: int = 1
    ) -> "ContentBlockBuilder":
        """Add static content (cached)."""
        self.blocks.append(ContentBlock(
            content=content,
            should_cache=True,
            ttl_seconds=ttl_hours * 3600,
            description=description
        ))
        return self
    
    def add_dynamic(
        self,
        content: str,
        description: str = ""
    ) -> "ContentBlockBuilder":
        """Add dynamic content (not cached)."""
        self.blocks.append(ContentBlock(
            content=content,
            should_cache=False,
            description=description
        ))
        return self
    
    def build(self) -> List[dict]:
        """Build the system blocks list."""
        return [block.to_block() for block in self.blocks]
    
    def summary(self) -> dict:
        """Get summary of cache configuration."""
        cached = [b for b in self.blocks if b.should_cache]
        uncached = [b for b in self.blocks if not b.should_cache]
        
        return {
            "total_blocks": len(self.blocks),
            "cached_blocks": len(cached),
            "uncached_blocks": len(uncached),
            "cached_chars": sum(len(b.content) for b in cached),
            "uncached_chars": sum(len(b.content) for b in uncached)
        }


# Usage
builder = ContentBlockBuilder()

system_blocks = (
    builder
    .add_static(
        "You are an expert Python tutor...",
        description="Core instructions",
        ttl_hours=1
    )
    .add_static(
        "Reference documentation: " + "..." * 1000,
        description="Knowledge base",
        ttl_hours=1
    )
    .add_dynamic(
        f"Current session: {datetime.now()}",
        description="Session context"
    )
    .build()
)

print("Cache Summary:", builder.summary())
```

---

## Hands-on Exercise

### Your Task

Build a Claude client with explicit cache management.

### Requirements

1. Create content blocks with appropriate caching
2. Track cache hits, writes, and misses
3. Calculate cost savings
4. Implement cache-aware conversation

<details>
<summary>💡 Hints</summary>

- Use ContentBlockBuilder for structured blocks
- Check cache_creation_input_tokens for writes
- Check cache_read_input_tokens for hits
- Calculate savings with 90% discount on cache reads
</details>

<details>
<summary>✅ Solution</summary>

```python
from anthropic import Anthropic
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class CacheStatus(Enum):
    HIT = "hit"
    WRITE = "write"
    MISS = "miss"
    PARTIAL = "partial"


@dataclass
class RequestResult:
    response: any
    cache_status: CacheStatus
    input_tokens: int
    cached_tokens: int
    written_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)


class ClaudeCacheClient:
    """Claude client with explicit cache management."""
    
    # Pricing per 1M tokens
    PRICING = {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.results: List[RequestResult] = []
        self._static_blocks: List[dict] = []
    
    def set_static_context(
        self,
        instructions: str,
        knowledge: Optional[str] = None,
        ttl_hours: int = 1
    ):
        """Set static context blocks for caching."""
        
        self._static_blocks = []
        ttl = ttl_hours * 3600
        
        # Instructions block
        self._static_blocks.append({
            "type": "text",
            "text": instructions,
            "cache_control": {"type": "ephemeral", "ttl": ttl}
        })
        
        # Knowledge block (if provided)
        if knowledge:
            self._static_blocks.append({
                "type": "text",
                "text": knowledge,
                "cache_control": {"type": "ephemeral", "ttl": ttl}
            })
    
    def create(
        self,
        message: str,
        dynamic_context: Optional[str] = None,
        max_tokens: int = 1024
    ) -> RequestResult:
        """Create message with cache tracking."""
        
        # Build system blocks
        system_blocks = self._static_blocks.copy()
        
        if dynamic_context:
            system_blocks.append({
                "type": "text",
                "text": dynamic_context
                # No cache_control - dynamic content
            })
        
        # Make request
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": message}]
        )
        
        # Parse usage
        usage = response.usage
        written = getattr(usage, 'cache_creation_input_tokens', 0)
        cached = getattr(usage, 'cache_read_input_tokens', 0)
        
        # Determine status
        if cached > 0 and written > 0:
            status = CacheStatus.PARTIAL
        elif cached > 0:
            status = CacheStatus.HIT
        elif written > 0:
            status = CacheStatus.WRITE
        else:
            status = CacheStatus.MISS
        
        result = RequestResult(
            response=response,
            cache_status=status,
            input_tokens=usage.input_tokens,
            cached_tokens=cached,
            written_tokens=written,
            output_tokens=usage.output_tokens
        )
        
        self.results.append(result)
        return result
    
    def get_stats(self) -> dict:
        """Get overall statistics."""
        
        if not self.results:
            return {"error": "No requests made"}
        
        hits = sum(1 for r in self.results if r.cache_status == CacheStatus.HIT)
        writes = sum(1 for r in self.results if r.cache_status == CacheStatus.WRITE)
        partials = sum(1 for r in self.results if r.cache_status == CacheStatus.PARTIAL)
        misses = sum(1 for r in self.results if r.cache_status == CacheStatus.MISS)
        
        total_input = sum(r.input_tokens for r in self.results)
        total_cached = sum(r.cached_tokens for r in self.results)
        total_written = sum(r.written_tokens for r in self.results)
        total_output = sum(r.output_tokens for r in self.results)
        
        return {
            "total_requests": len(self.results),
            "cache_hits": hits,
            "cache_writes": writes,
            "partial_hits": partials,
            "cache_misses": misses,
            "hit_rate": f"{hits / len(self.results):.1%}",
            "total_input_tokens": total_input,
            "total_cached_tokens": total_cached,
            "total_written_tokens": total_written,
            "total_output_tokens": total_output
        }
    
    def calculate_savings(self) -> dict:
        """Calculate cost savings from caching."""
        
        stats = self.get_stats()
        if "error" in stats:
            return stats
        
        # Actual costs
        input_cost = (stats["total_input_tokens"] / 1_000_000) * self.PRICING["input"]
        output_cost = (stats["total_output_tokens"] / 1_000_000) * self.PRICING["output"]
        write_cost = (stats["total_written_tokens"] / 1_000_000) * self.PRICING["cache_write"]
        read_cost = (stats["total_cached_tokens"] / 1_000_000) * self.PRICING["cache_read"]
        
        actual_total = input_cost + output_cost + write_cost + read_cost
        
        # Without caching
        all_input = stats["total_input_tokens"] + stats["total_cached_tokens"]
        full_cost = (
            (all_input / 1_000_000) * self.PRICING["input"] +
            (stats["total_output_tokens"] / 1_000_000) * self.PRICING["output"]
        )
        
        savings = full_cost - actual_total
        
        return {
            "actual_cost": f"${actual_total:.4f}",
            "without_cache": f"${full_cost:.4f}",
            "savings": f"${savings:.4f}",
            "savings_percent": f"{(savings / full_cost * 100):.1f}%" if full_cost > 0 else "0%",
            "breakdown": {
                "input": f"${input_cost:.4f}",
                "output": f"${output_cost:.4f}",
                "cache_write": f"${write_cost:.4f}",
                "cache_read": f"${read_cost:.4f}"
            }
        }


class CachedConversation:
    """Multi-turn conversation with caching."""
    
    def __init__(self, client: ClaudeCacheClient):
        self.client = client
        self.messages: List[dict] = []
    
    def send(self, user_message: str) -> str:
        """Send message and get response."""
        
        # Add to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Create context with conversation history
        history_context = self._format_history()
        
        result = self.client.create(
            message=user_message,
            dynamic_context=f"Conversation history:\n{history_context}"
        )
        
        # Extract text
        assistant_text = result.response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_text})
        
        return assistant_text
    
    def _format_history(self) -> str:
        """Format conversation history."""
        lines = []
        for msg in self.messages[:-1]:  # Exclude current message
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


# Usage
cache_client = ClaudeCacheClient()

# Set static context (cached)
cache_client.set_static_context(
    instructions="You are an expert Python tutor. Provide clear, detailed explanations " * 50,
    knowledge="Python documentation: " + "Reference content... " * 100,
    ttl_hours=1
)

# Make requests
for query in ["What is a decorator?", "Show an example", "How does functools.wraps work?"]:
    result = cache_client.create(query)
    print(f"Query: {query}")
    print(f"  Status: {result.cache_status.value}")
    print(f"  Cached: {result.cached_tokens}, Written: {result.written_tokens}")
    print()

# View stats
print("Statistics:")
for key, value in cache_client.get_stats().items():
    print(f"  {key}: {value}")

print("\nCost Analysis:")
for key, value in cache_client.calculate_savings().items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")
```

</details>

---

## Summary

✅ Anthropic uses explicit `cache_control` blocks  
✅ Ephemeral is the only cache type currently supported  
✅ TTL options: 5 minutes (default) or 1 hour  
✅ Track `cache_creation_input_tokens` and `cache_read_input_tokens`  
✅ 90% cost reduction on cached reads, 25% premium on writes

**Next:** [Cacheable Content](./04-cacheable-content.md)

---

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Official guide
- [Claude API Reference](https://docs.anthropic.com/en/api) — API documentation
- [Pricing](https://www.anthropic.com/pricing) — Current pricing
