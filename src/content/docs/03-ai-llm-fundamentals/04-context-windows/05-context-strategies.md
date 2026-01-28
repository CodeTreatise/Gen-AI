---
title: "Context Window Strategies"
---

# Context Window Strategies

## Introduction

There's no one-size-fits-all approach to managing context. Different applications need different strategies. This lesson covers the main approaches and helps you choose the right one for your use case.

### What We'll Cover

- Simple truncation
- Smart truncation
- Periodic summarization
- Hierarchical summarization
- Hybrid approaches

---

## Strategy Comparison

| Strategy | Complexity | Memory Cost | Context Quality | Best For |
|----------|-----------|-------------|-----------------|----------|
| Simple truncation | Low | Low | Poor | Stateless chats |
| Smart truncation | Medium | Low | Good | Support bots |
| Periodic summarization | Medium | Medium | Good | Long conversations |
| Hierarchical summarization | High | High | Excellent | Complex tasks |
| Hybrid (RAG + Window) | High | Medium | Excellent | Enterprise apps |

---

## Simple Truncation

The simplest approach: remove oldest messages when context fills up.

```python
def simple_truncate(messages: list, max_tokens: int, enc) -> list:
    """
    Drop oldest messages until context fits.
    System messages are always preserved.
    """
    
    # Separate system messages
    system = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    # Calculate system token usage
    system_tokens = sum(
        len(enc.encode(m["content"])) + 4 for m in system
    )
    
    available = max_tokens - system_tokens - 2000  # Reserve for response
    
    # Start from most recent, add until full
    kept = []
    current_tokens = 0
    
    for msg in reversed(conversation):
        msg_tokens = len(enc.encode(msg["content"])) + 4
        if current_tokens + msg_tokens <= available:
            kept.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break
    
    return system + kept
```

### Pros and Cons

```
✅ Simple to implement
✅ Predictable behavior
✅ No API calls needed
✅ Fast execution

❌ Loses all old context
❌ User references to old messages fail
❌ No memory of earlier decisions
❌ Abrupt information loss
```

---

## Smart Truncation

Keep important messages while removing less important ones:

```python
class SmartTruncator:
    """
    Remove messages based on importance, not just age.
    """
    
    def __init__(self, model: str, max_tokens: int):
        self.enc = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
    
    def truncate(self, messages: list) -> list:
        # Always keep: system, first exchange, recent N
        system = [m for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        
        if len(conversation) <= 6:
            return messages
        
        # Structure to keep
        first_exchange = conversation[:2]  # First Q&A
        recent = conversation[-4:]          # Last 4 messages
        middle = conversation[2:-4]         # Everything else
        
        # Score middle messages
        scored = [(m, self._score(m)) for m in middle]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate available space
        required = self._count_tokens(system + first_exchange + recent)
        available = self.max_tokens - required - 2000
        
        # Add highest-scored middle messages that fit
        kept_middle = []
        used = 0
        for msg, score in scored:
            msg_tokens = len(self.enc.encode(msg["content"])) + 4
            if used + msg_tokens <= available:
                kept_middle.append(msg)
                used += msg_tokens
        
        # Restore original order
        kept_middle.sort(key=lambda m: conversation.index(m))
        
        return system + first_exchange + kept_middle + recent
    
    def _score(self, message: dict) -> float:
        """Score message importance"""
        content = message["content"].lower()
        score = 0.0
        
        # User messages slightly more important
        if message["role"] == "user":
            score += 0.1
        
        # Contains explicit importance markers
        if any(word in content for word in ["important", "remember", "note"]):
            score += 0.3
        
        # Contains personal info
        if any(word in content for word in ["my name", "i am", "i prefer"]):
            score += 0.2
        
        # Contains decisions/confirmations
        if any(word in content for word in ["yes", "confirmed", "agreed", "decided"]):
            score += 0.15
        
        # Contains questions (often important for context)
        if "?" in content:
            score += 0.1
        
        # Length bonus (longer messages often contain more info)
        score += min(len(content) / 1000, 0.2)
        
        return score
    
    def _count_tokens(self, messages: list) -> int:
        return sum(len(self.enc.encode(m["content"])) + 4 for m in messages)
```

### When Smart Truncation Wins

```python
# Example: User references old context

conversation = [
    {"role": "user", "content": "My name is Alice, and I prefer metric units."},
    {"role": "assistant", "content": "Hello Alice! I'll remember to use metric."},
    # ... many messages later ...
    {"role": "user", "content": "How tall is Mount Everest?"},
]

# Simple truncation: might lose the "metric units" preference
# Smart truncation: keeps it because it contains "I prefer"
```

---

## Periodic Summarization

Regularly summarize old messages to compress context:

```python
class PeriodicSummarizer:
    """
    Summarize old messages on a schedule.
    """
    
    def __init__(
        self, 
        model: str,
        summarize_every: int = 10,  # Summarize every N messages
        keep_recent: int = 6
    ):
        self.model = model
        self.summarize_every = summarize_every
        self.keep_recent = keep_recent
        self.messages = []
        self.summary = None
        self.message_count = 0
    
    async def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.message_count += 1
        
        # Time to summarize?
        if self.message_count % self.summarize_every == 0:
            await self._summarize()
    
    async def _summarize(self):
        if len(self.messages) <= self.keep_recent:
            return
        
        old_messages = self.messages[:-self.keep_recent]
        
        # Include previous summary in what we're summarizing
        to_summarize = ""
        if self.summary:
            to_summarize += f"Previous summary: {self.summary}\n\n"
        to_summarize += self._format_messages(old_messages)
        
        # Generate new summary
        self.summary = await self._call_summarizer(to_summarize)
        
        # Keep only recent messages
        self.messages = self.messages[-self.keep_recent:]
    
    async def _call_summarizer(self, content: str) -> str:
        # Use a cheaper/faster model for summarization
        prompt = f"""
        Summarize this conversation, preserving:
        - Key facts about the user
        - Decisions and preferences expressed
        - Important context for future questions
        - Any commitments made
        
        Be concise but comprehensive.
        
        Content:
        {content}
        """
        
        # Call API (simplified)
        return await generate(model="gpt-4o-mini", prompt=prompt)
    
    def get_context(self) -> list:
        """Get messages ready for API call"""
        result = []
        
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[Conversation history summary: {self.summary}]"
            })
        
        result.extend(self.messages)
        return result
    
    def _format_messages(self, messages: list) -> str:
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" 
            for m in messages
        )
```

### Summarization Trade-offs

```
✅ Compresses context effectively (10:1 or better)
✅ Preserves key information
✅ Allows very long conversations
✅ Graceful degradation

❌ Loses some detail
❌ Requires additional API calls (cost)
❌ Summarization can miss nuances
❌ More complex implementation
```

---

## Hierarchical Summarization

Create summaries at multiple levels for very long conversations:

```python
class HierarchicalMemory:
    """
    Multi-level summarization for extended conversations.
    
    Levels:
    - Messages: Individual messages (most recent)
    - Chunks: Groups of ~10 messages summarized
    - Sessions: Groups of ~5 chunks summarized
    - Archive: Very old session summaries
    """
    
    def __init__(self, model: str):
        self.model = model
        self.levels = {
            "messages": [],        # Raw recent messages
            "chunks": [],          # Chunk summaries
            "sessions": [],        # Session summaries
            "archive": None,       # Archive summary
        }
        self.config = {
            "messages_per_chunk": 10,
            "chunks_per_session": 5,
            "sessions_per_archive": 3,
        }
    
    async def add_message(self, role: str, content: str):
        self.levels["messages"].append({"role": role, "content": content})
        await self._cascade_if_needed()
    
    async def _cascade_if_needed(self):
        """Check each level and cascade up if needed"""
        
        # Messages → Chunk
        if len(self.levels["messages"]) >= self.config["messages_per_chunk"]:
            chunk = await self._summarize_to_chunk()
            self.levels["chunks"].append(chunk)
            self.levels["messages"] = self.levels["messages"][-4:]  # Keep some
        
        # Chunks → Session
        if len(self.levels["chunks"]) >= self.config["chunks_per_session"]:
            session = await self._summarize_to_session()
            self.levels["sessions"].append(session)
            self.levels["chunks"] = self.levels["chunks"][-2:]  # Keep some
        
        # Sessions → Archive
        if len(self.levels["sessions"]) >= self.config["sessions_per_archive"]:
            await self._update_archive()
            self.levels["sessions"] = self.levels["sessions"][-1:]
    
    async def _summarize_to_chunk(self) -> str:
        messages = self.levels["messages"]
        return await summarize(messages, prompt="Summarize these messages briefly:")
    
    async def _summarize_to_session(self) -> str:
        chunks = self.levels["chunks"]
        return await summarize(chunks, prompt="Combine these summaries into a session overview:")
    
    async def _update_archive(self):
        sessions = self.levels["sessions"]
        existing = self.levels["archive"] or ""
        self.levels["archive"] = await summarize(
            sessions,
            prompt=f"Update this archive with new sessions:\n\nExisting: {existing}"
        )
    
    def get_context(self) -> str:
        """Build context from all levels"""
        parts = []
        
        if self.levels["archive"]:
            parts.append(f"[Long-term context: {self.levels['archive']}]")
        
        if self.levels["sessions"]:
            parts.append(f"[Session context: {' '.join(self.levels['sessions'])}]")
        
        if self.levels["chunks"]:
            parts.append(f"[Recent context: {' '.join(self.levels['chunks'])}]")
        
        for msg in self.levels["messages"]:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        
        return "\n\n".join(parts)
```

---

## Hybrid Approaches

Combine multiple strategies for best results:

```python
class HybridContextManager:
    """
    Combine sliding window + RAG + summarization.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        window_size: int = 10,
        max_tokens: int = 128000
    ):
        # Components
        self.window = SlidingWindow(window_size)
        self.vector_store = VectorStore()  # For RAG
        self.summarizer = PeriodicSummarizer(model)
        
        self.model = model
        self.max_tokens = max_tokens
    
    async def add_message(self, role: str, content: str):
        """Add message to all components"""
        # Add to sliding window
        self.window.add_message(role, content)
        
        # Index in vector store (for retrieval)
        self.vector_store.add(
            text=content,
            metadata={"role": role, "timestamp": time.time()}
        )
        
        # Add to summarizer
        await self.summarizer.add_message(role, content)
    
    async def get_context(self, current_query: str) -> list:
        """Build optimal context for current query"""
        context = []
        
        # 1. System prompt (always first)
        if self.system_prompt:
            context.append(self.system_prompt)
        
        # 2. Conversation summary (if exists)
        if self.summarizer.summary:
            context.append({
                "role": "system",
                "content": f"[Summary of earlier conversation: {self.summarizer.summary}]"
            })
        
        # 3. Retrieved relevant context (RAG)
        relevant = self.vector_store.search(current_query, limit=3)
        if relevant:
            retrieved_text = "\n".join([
                f"[Earlier: {r['text'][:200]}]" 
                for r in relevant
            ])
            context.append({
                "role": "system", 
                "content": f"Relevant earlier context:\n{retrieved_text}"
            })
        
        # 4. Recent messages (sliding window)
        context.extend(self.window.get_messages())
        
        # 5. Current message
        context.append({"role": "user", "content": current_query})
        
        return self._ensure_fits(context)
    
    def _ensure_fits(self, context: list) -> list:
        """Trim context if still too large"""
        # Implementation similar to smart truncation
        pass
```

### Decision Matrix

```python
def choose_strategy(requirements: dict) -> str:
    """
    Choose the best strategy based on requirements.
    """
    
    if requirements.get("conversation_length", 0) < 20:
        return "simple_truncation"  # Short conversations don't need complexity
    
    if requirements.get("stateless", False):
        return "sliding_window"  # No memory needed
    
    if requirements.get("needs_full_history", False):
        if requirements.get("conversation_length", 0) > 100:
            return "hierarchical"
        else:
            return "periodic_summarization"
    
    if requirements.get("reference_old_context", False):
        return "hybrid_rag"  # Need to retrieve old context
    
    return "smart_truncation"  # Good default

# Example usage
strategy = choose_strategy({
    "conversation_length": 150,
    "needs_full_history": True,
    "reference_old_context": True,
})
print(f"Recommended strategy: {strategy}")  # hybrid_rag
```

---

## Hands-on Exercise

### Your Task

Implement a configurable context manager:

```python
class ConfigurableContextManager:
    """
    Context manager that can use different strategies.
    """
    
    STRATEGIES = ["simple", "smart", "summarize", "hierarchical", "hybrid"]
    
    def __init__(self, strategy: str, **config):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.config = config
        # TODO: Initialize based on strategy
    
    async def add_message(self, role: str, content: str):
        """Add message using selected strategy"""
        # TODO: Implement for each strategy
        pass
    
    def get_messages(self) -> list:
        """Get messages ready for API"""
        # TODO: Implement for each strategy
        pass
    
    def get_stats(self) -> dict:
        """Return strategy-specific statistics"""
        # TODO: Implement
        pass

# Test all strategies
async def test_strategies():
    text = "Sample message content here. " * 50  # Long message
    
    for strategy in ConfigurableContextManager.STRATEGIES:
        manager = ConfigurableContextManager(strategy)
        
        for i in range(100):
            await manager.add_message("user", f"Message {i}: {text}")
            await manager.add_message("assistant", f"Response {i}")
        
        stats = manager.get_stats()
        print(f"{strategy}: {stats}")
```

---

## Summary

✅ **Simple truncation** — Fast but loses context completely

✅ **Smart truncation** — Keeps important messages based on scoring

✅ **Periodic summarization** — Compresses old messages regularly

✅ **Hierarchical summarization** — Multi-level for very long conversations

✅ **Hybrid approaches** — Combine strategies for best results

✅ **Choose based on needs** — Conversation length, memory requirements, cost

**Next:** [Impact on Application Design](./06-application-design.md)

---

## Further Reading

- [MemGPT](https://memgpt.ai) — Memory management for LLMs
- [LangChain Memory Types](https://python.langchain.com/docs/modules/memory/) — Various implementations
- [Conversation Memory Patterns](https://docs.anthropic.com/en/docs/build-with-claude/memory) — Anthropic's guide

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Managing Conversations](./04-managing-long-conversations.md) | [Context Windows](./00-context-windows.md) | [Application Design](./06-application-design.md) |

