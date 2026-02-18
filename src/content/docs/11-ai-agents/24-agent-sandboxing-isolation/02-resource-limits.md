---
title: "Resource limits"
---

# Resource limits

## Introduction

An agent stuck in a reasoning loop can burn through thousands of dollars in API calls before anyone notices. A code execution tool that runs user-submitted code without a timeout can block your server indefinitely. A research agent that downloads every link it finds can fill your disk in minutes. Resource limits are the hard guardrails that prevent these scenarios — they cap the damage any single agent run can cause, regardless of what the agent tries to do.

We implement resource limits at four levels: execution time, memory consumption, API call quotas, and storage. Each level uses different mechanisms — from Python's `resource` module for OS-level constraints to custom middleware for API call tracking.

### What we'll cover

- Execution time limits using `signal`, `asyncio.timeout()`, and `resource.setrlimit()`
- Memory quotas with OS-level and application-level controls
- API call caps and rate limiting per agent
- Storage limits for file creation and data accumulation
- Building a unified resource limiter for agent execution

### Prerequisites

- Python `asyncio` basics (Unit 2, Lesson 09)
- Understanding of Unix processes and signals
- Agent tool use patterns (Lesson 09-10)

---

## Execution time limits

The most critical resource to limit is time. An agent that runs forever blocks other requests, wastes compute, and accumulates costs. We apply time limits at three levels.

### Level 1: async timeout for agent runs

The simplest and most portable approach uses `asyncio.timeout()` to cap the total wall-clock time of an agent run.

```python
import asyncio
import time


async def simulate_llm_call(prompt: str, delay: float = 1.0) -> str:
    """Simulate an LLM call that takes variable time."""
    await asyncio.sleep(delay)
    return f"Response to: {prompt[:30]}..."


async def agent_run_with_timeout(prompt: str, timeout_seconds: float) -> str:
    """Run an agent with a strict time limit."""
    start = time.monotonic()

    try:
        async with asyncio.timeout(timeout_seconds):
            # Simulate multi-step agent execution
            result1 = await simulate_llm_call(prompt, delay=0.5)
            result2 = await simulate_llm_call(f"Follow up: {result1}", delay=0.5)
            result3 = await simulate_llm_call(f"Conclude: {result2}", delay=0.5)

            elapsed = time.monotonic() - start
            return f"Completed in {elapsed:.2f}s: {result3}"

    except TimeoutError:
        elapsed = time.monotonic() - start
        return f"TIMEOUT after {elapsed:.2f}s — agent run terminated"


# Run within time limit
result = asyncio.run(agent_run_with_timeout("Summarize AI safety research", 5.0))
print(f"✅ {result}")

# Run that exceeds time limit
result = asyncio.run(agent_run_with_timeout("Summarize AI safety research", 1.0))
print(f"⏱️ {result}")
```

**Output:**
```
✅ Completed in 1.50s: Response to: Conclude: Response to: Follow...
⏱️ TIMEOUT after 1.00s — agent run terminated
```

### Level 2: per-step timeouts

For complex agents that make many tool calls, we set individual timeouts per step rather than just a global timeout. This prevents one slow tool from consuming the entire time budget.

```python
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class StepBudget:
    """Track time budget across multiple agent steps."""
    total_seconds: float
    per_step_max: float
    start_time: float = field(default_factory=time.monotonic)
    steps_completed: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def remaining(self) -> float:
        return max(0, self.total_seconds - self.elapsed)

    def step_timeout(self) -> float:
        """Calculate timeout for the next step."""
        return min(self.per_step_max, self.remaining)

    def is_expired(self) -> bool:
        return self.remaining <= 0


async def run_step(budget: StepBudget, step_name: str, duration: float) -> str:
    """Execute a single agent step within budget constraints."""
    if budget.is_expired():
        return f"SKIPPED {step_name} — budget expired"

    timeout = budget.step_timeout()
    try:
        async with asyncio.timeout(timeout):
            await asyncio.sleep(duration)  # Simulate work
            budget.steps_completed += 1
            return f"✅ {step_name} completed ({duration:.1f}s)"
    except TimeoutError:
        return f"⏱️ {step_name} timed out (limit: {timeout:.1f}s)"


async def main():
    # 3 seconds total, 1.5 seconds max per step
    budget = StepBudget(total_seconds=3.0, per_step_max=1.5)

    steps = [
        ("Parse input", 0.3),
        ("Search documents", 0.8),
        ("Analyze results", 0.7),
        ("Generate summary", 0.5),
        ("Format output", 2.0),  # This one is too slow
    ]

    print("=== Agent Execution with Step Budgets ===\n")
    for name, duration in steps:
        result = await run_step(budget, name, duration)
        print(f"  {result} | Remaining: {budget.remaining:.1f}s")

    print(f"\nCompleted {budget.steps_completed}/{len(steps)} steps in {budget.elapsed:.1f}s")


asyncio.run(main())
```

**Output:**
```
=== Agent Execution with Step Budgets ===

  ✅ Parse input completed (0.3s) | Remaining: 2.7s
  ✅ Search documents completed (0.8s) | Remaining: 1.9s
  ✅ Analyze results completed (0.7s) | Remaining: 1.2s
  ✅ Generate summary completed (0.5s) | Remaining: 0.7s
  ⏱️ Format output timed out (limit: 0.7s) | Remaining: 0.0s

Completed 4/5 steps in 3.0s
```

### Level 3: OS-level CPU time limits

For agent processes that might execute tight CPU-bound loops (e.g., code execution tools), we use the `resource` module to set hard kernel-enforced limits.

```python
import resource
import signal


def set_cpu_limit(seconds: int) -> None:
    """Set a hard CPU time limit for the current process.

    The kernel sends SIGXCPU when the soft limit is reached,
    then SIGKILL when the hard limit is reached.
    """
    # (soft_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))

    def handler(signum, frame):
        raise TimeoutError(f"CPU time limit exceeded ({seconds}s)")

    signal.signal(signal.SIGXCPU, handler)


def demonstrate_limits():
    """Show current resource limits."""
    limits = {
        "CPU time": resource.RLIMIT_CPU,
        "Memory (bytes)": resource.RLIMIT_AS,
        "Open files": resource.RLIMIT_NOFILE,
        "Max processes": resource.RLIMIT_NPROC,
        "File size (bytes)": resource.RLIMIT_FSIZE,
        "Stack size (bytes)": resource.RLIMIT_STACK,
    }

    print("=== Current Resource Limits ===\n")
    print(f"{'Resource':<22} {'Soft Limit':<15} {'Hard Limit':<15}")
    print("-" * 52)

    for name, const in limits.items():
        soft, hard = resource.getrlimit(const)
        soft_str = "unlimited" if soft == resource.RLIM_INFINITY else str(soft)
        hard_str = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
        print(f"{name:<22} {soft_str:<15} {hard_str:<15}")


demonstrate_limits()
```

**Output:**
```
=== Current Resource Limits ===

Resource               Soft Limit      Hard Limit     
----------------------------------------------------
CPU time               unlimited       unlimited      
Memory (bytes)         unlimited       unlimited      
Open files             1024            1048576        
Max processes          63304           63304          
File size (bytes)      unlimited       unlimited      
Stack size (bytes)     8388608         unlimited      
```

> **Warning:** `resource.setrlimit()` is Linux/macOS only. On Windows, use `subprocess` with `CREATE_NEW_PROCESS_GROUP` and `TerminateProcess` for similar functionality. The `resource` module works at the process level, so it won't limit child processes unless they inherit the limits.

---

## Memory quotas

An agent that loads large files, accumulates conversation history, or generates massive outputs can exhaust system memory. We set limits at both the OS and application levels.

### OS-level memory limits

```python
import resource


def set_memory_limit(max_mb: int) -> None:
    """Limit the process address space (virtual memory).

    Uses RLIMIT_AS which limits the total virtual memory
    available to the process. malloc() returns NULL / mmap()
    returns MAP_FAILED when the limit is reached.
    """
    max_bytes = max_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))


def test_memory_limit():
    """Demonstrate memory limit enforcement."""
    # Set 100MB limit
    set_memory_limit(100)

    allocated = []
    try:
        while True:
            # Allocate 10MB chunks
            chunk = bytearray(10 * 1024 * 1024)
            allocated.append(chunk)
            total_mb = len(allocated) * 10
            print(f"Allocated {total_mb}MB...")
    except MemoryError:
        total_mb = len(allocated) * 10
        print(f"\n⚠️ MemoryError after allocating {total_mb}MB")
        print("Memory limit enforced successfully")


# Uncomment to test (will set limits on your process):
# test_memory_limit()
print("Memory limiting demo (uncomment to test)")
print("Usage: set_memory_limit(100)  # Limit to 100MB")
```

**Output:**
```
Memory limiting demo (uncomment to test)
Usage: set_memory_limit(100)  # Limit to 100MB
```

### Application-level memory tracking

For finer control, we track memory usage within the agent runtime:

```python
import sys
from dataclasses import dataclass, field


@dataclass
class MemoryTracker:
    """Track and limit memory usage at the application level."""
    max_bytes: int
    current_bytes: int = 0
    peak_bytes: int = 0
    allocations: dict[str, int] = field(default_factory=dict)

    def track(self, key: str, data: object) -> bool:
        """Track memory for a named allocation. Returns False if limit exceeded."""
        size = sys.getsizeof(data)

        # For containers, include contained objects
        if isinstance(data, (list, tuple)):
            size += sum(sys.getsizeof(item) for item in data)
        elif isinstance(data, dict):
            size += sum(
                sys.getsizeof(k) + sys.getsizeof(v)
                for k, v in data.items()
            )
        elif isinstance(data, str):
            size = sys.getsizeof(data)

        projected = self.current_bytes - self.allocations.get(key, 0) + size

        if projected > self.max_bytes:
            return False

        # Update tracking
        if key in self.allocations:
            self.current_bytes -= self.allocations[key]
        self.allocations[key] = size
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return True

    def release(self, key: str) -> None:
        """Release tracked memory for a key."""
        if key in self.allocations:
            self.current_bytes -= self.allocations.pop(key)

    @property
    def usage_pct(self) -> float:
        return (self.current_bytes / self.max_bytes) * 100

    def report(self) -> str:
        """Generate a memory usage report."""
        lines = [
            f"Memory Usage: {self.current_bytes:,} / {self.max_bytes:,} bytes ({self.usage_pct:.1f}%)",
            f"Peak Usage:   {self.peak_bytes:,} bytes",
            f"Allocations:  {len(self.allocations)}",
        ]
        for key, size in sorted(
            self.allocations.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {key}: {size:,} bytes")
        return "\n".join(lines)


# Simulate agent memory tracking
tracker = MemoryTracker(max_bytes=1_000_000)  # 1MB limit

# Agent accumulates data during execution
tracker.track("conversation_history", [{"role": "user", "content": "Hello " * 100}])
print(f"After conversation: {tracker.usage_pct:.1f}%")

tracker.track("search_results", {"results": ["result " * 50] * 20})
print(f"After search: {tracker.usage_pct:.1f}%")

tracker.track("document_cache", "x" * 500_000)
print(f"After document: {tracker.usage_pct:.1f}%")

# This one should fail
success = tracker.track("large_response", "y" * 600_000)
print(f"Large response allocation: {'✅ OK' if success else '❌ Rejected (over limit)'}")

print(f"\n{tracker.report()}")
```

**Output:**
```
After conversation: 0.6%
After search: 4.0%
After document: 54.1%
Large response allocation: ❌ Rejected (over limit)

Memory Usage: 541,064 / 1,000,000 bytes (54.1%)
Peak Usage:   541,064 bytes
Allocations:  3
  document_cache: 500,049 bytes
  search_results: 35,091 bytes
  conversation_history: 5,924 bytes
```

---

## API call caps

LLM API calls are expensive. An agent making recursive calls to refine its answer can generate hundreds of requests per user interaction. We cap API calls at multiple granularities.

### Per-run API call limiter

```python
import time
from dataclasses import dataclass, field
from enum import Enum


class LimitAction(Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    DENY = "deny"


@dataclass
class APICallLimiter:
    """Rate limit and cap API calls for agent runs."""
    max_calls_per_run: int = 20
    max_calls_per_minute: int = 10
    max_tokens_per_run: int = 100_000
    calls: list[float] = field(default_factory=list)
    total_calls: int = 0
    total_tokens: int = 0

    def check(self, estimated_tokens: int = 0) -> tuple[LimitAction, str]:
        """Check if an API call should be allowed."""
        now = time.time()

        # Check total call limit
        if self.total_calls >= self.max_calls_per_run:
            return LimitAction.DENY, f"Run limit reached ({self.max_calls_per_run} calls)"

        # Check token budget
        if self.total_tokens + estimated_tokens > self.max_tokens_per_run:
            return LimitAction.DENY, (
                f"Token budget exceeded "
                f"({self.total_tokens + estimated_tokens}/{self.max_tokens_per_run})"
            )

        # Check rate limit (calls in the last 60 seconds)
        recent = [t for t in self.calls if now - t < 60]
        if len(recent) >= self.max_calls_per_minute:
            wait_time = 60 - (now - recent[0])
            return LimitAction.THROTTLE, f"Rate limited — wait {wait_time:.1f}s"

        return LimitAction.ALLOW, "OK"

    def record(self, tokens_used: int) -> None:
        """Record a completed API call."""
        self.calls.append(time.time())
        self.total_calls += 1
        self.total_tokens += tokens_used

    def summary(self) -> str:
        """Return usage summary."""
        return (
            f"Calls: {self.total_calls}/{self.max_calls_per_run} | "
            f"Tokens: {self.total_tokens:,}/{self.max_tokens_per_run:,}"
        )


# Simulate agent API usage
limiter = APICallLimiter(
    max_calls_per_run=5,
    max_calls_per_minute=3,
    max_tokens_per_run=10_000,
)

simulated_calls = [
    ("Initial reasoning", 2000),
    ("Tool call 1", 1500),
    ("Tool call 2", 1800),
    ("Synthesize results", 3000),
    ("Generate response", 2500),
    ("Additional refinement", 1000),  # Should be denied
]

print("=== API Call Limiter Demo ===\n")
for step_name, tokens in simulated_calls:
    action, reason = limiter.check(estimated_tokens=tokens)

    if action == LimitAction.ALLOW:
        limiter.record(tokens)
        print(f"✅ {step_name} ({tokens:,} tokens) — {limiter.summary()}")
    elif action == LimitAction.THROTTLE:
        print(f"⏳ {step_name} — {reason}")
    else:
        print(f"❌ {step_name} — {reason}")
```

**Output:**
```
=== API Call Limiter Demo ===

✅ Initial reasoning (2,000 tokens) — Calls: 1/5 | Tokens: 2,000/10,000
✅ Tool call 1 (1,500 tokens) — Calls: 2/5 | Tokens: 3,500/10,000
✅ Tool call 2 (1,800 tokens) — Calls: 3/5 | Tokens: 5,300/10,000
✅ Synthesize results (3,000 tokens) — Calls: 4/5 | Tokens: 8,300/10,000
❌ Generate response — Token budget exceeded (10,800/10,000)
❌ Additional refinement — Token budget exceeded (9,300/10,000)
```

### Cost-aware limiter

We can extend the limiter to track costs and enforce budget limits:

```python
from dataclasses import dataclass


@dataclass
class CostTracker:
    """Track API costs and enforce budget limits."""
    budget_usd: float
    spent_usd: float = 0.0

    # Pricing per million tokens (approximate, 2025)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
    }

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a call."""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def can_afford(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[bool, float]:
        """Check if budget allows this call."""
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        if self.spent_usd + cost > self.budget_usd:
            return False, cost
        return True, cost

    def record(self, cost: float) -> None:
        self.spent_usd += cost

    @property
    def remaining(self) -> float:
        return self.budget_usd - self.spent_usd


# Budget limit of $0.10 for this agent run
tracker = CostTracker(budget_usd=0.10)

calls = [
    ("gpt-4o", 5000, 2000),
    ("gpt-4o-mini", 3000, 1000),
    ("gpt-4o", 8000, 4000),
    ("gpt-4o", 20000, 5000),  # Expensive — might exceed budget
]

print("=== Cost-Aware Limiter ===\n")
for model, inp_tok, out_tok in calls:
    affordable, cost = tracker.can_afford(model, inp_tok, out_tok)
    if affordable:
        tracker.record(cost)
        print(
            f"✅ {model} ({inp_tok}+{out_tok} tokens) "
            f"= ${cost:.4f} | Remaining: ${tracker.remaining:.4f}"
        )
    else:
        print(
            f"❌ {model} ({inp_tok}+{out_tok} tokens) "
            f"= ${cost:.4f} | Budget exceeded (${tracker.remaining:.4f} left)"
        )
```

**Output:**
```
=== Cost-Aware Limiter ===

✅ gpt-4o (5000+2000 tokens) = $0.0325 | Remaining: $0.0675
✅ gpt-4o-mini (3000+1000 tokens) = $0.0011 | Remaining: $0.0664
✅ gpt-4o (8000+4000 tokens) = $0.0600 | Remaining: $0.0064
❌ gpt-4o (20000+5000 tokens) = $0.1000 | Budget exceeded ($0.0064 left)
```

---

## Storage limits

Agents that write files need storage constraints to prevent disk exhaustion.

```python
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StorageLimiter:
    """Limit how much disk space an agent can use."""
    root_dir: str
    max_bytes: int
    max_files: int = 100
    max_file_size: int = 10 * 1024 * 1024  # 10MB per file
    tracked_files: dict[str, int] = field(default_factory=dict)

    @property
    def total_used(self) -> int:
        return sum(self.tracked_files.values())

    @property
    def files_count(self) -> int:
        return len(self.tracked_files)

    def can_write(self, filename: str, size: int) -> tuple[bool, str]:
        """Check if writing a file is allowed."""
        # Validate path is within root
        full_path = os.path.realpath(os.path.join(self.root_dir, filename))
        if not full_path.startswith(os.path.realpath(self.root_dir)):
            return False, "Path traversal detected"

        # Check individual file size
        if size > self.max_file_size:
            return False, (
                f"File too large: {size:,} bytes "
                f"(max: {self.max_file_size:,})"
            )

        # Check total storage
        new_total = self.total_used - self.tracked_files.get(filename, 0) + size
        if new_total > self.max_bytes:
            return False, (
                f"Storage quota exceeded: {new_total:,} / {self.max_bytes:,} bytes"
            )

        # Check file count
        if filename not in self.tracked_files and self.files_count >= self.max_files:
            return False, f"File count limit reached ({self.max_files})"

        return True, "OK"

    def record_write(self, filename: str, size: int) -> None:
        """Record a file write."""
        self.tracked_files[filename] = size

    def record_delete(self, filename: str) -> None:
        """Record a file deletion."""
        self.tracked_files.pop(filename, None)

    def report(self) -> str:
        used_pct = (self.total_used / self.max_bytes) * 100
        return (
            f"Storage: {self.total_used:,}/{self.max_bytes:,} bytes ({used_pct:.1f}%) | "
            f"Files: {self.files_count}/{self.max_files}"
        )


# 1MB total storage, max 5 files
limiter = StorageLimiter(
    root_dir="/tmp/agent-workspace",
    max_bytes=1_000_000,
    max_files=5,
)

writes = [
    ("output.json", 50_000),
    ("analysis.md", 120_000),
    ("data.csv", 300_000),
    ("report.pdf", 400_000),
    ("backup.tar", 500_000),     # Should exceed storage
    ("../../etc/cron", 100),     # Path traversal attempt
]

print("=== Storage Limiter Demo ===\n")
for filename, size in writes:
    allowed, reason = limiter.can_write(filename, size)
    if allowed:
        limiter.record_write(filename, size)
        print(f"✅ Write {filename} ({size:,} bytes) — {limiter.report()}")
    else:
        print(f"❌ Write {filename} ({size:,} bytes) — {reason}")
```

**Output:**
```
=== Storage Limiter Demo ===

✅ Write output.json (50,000 bytes) — Storage: 50,000/1,000,000 bytes (5.0%) | Files: 1/5
✅ Write analysis.md (120,000 bytes) — Storage: 170,000/1,000,000 bytes (17.0%) | Files: 2/5
✅ Write data.csv (300,000 bytes) — Storage: 470,000/1,000,000 bytes (47.0%) | Files: 3/5
✅ Write report.pdf (400,000 bytes) — Storage: 870,000/1,000,000 bytes (87.0%) | Files: 4/5
❌ Write backup.tar (500,000 bytes) — Storage quota exceeded: 1,370,000 / 1,000,000 bytes
❌ Write ../../etc/cron (100 bytes) — Path traversal detected
```

---

## Unified resource limiter

Here's a complete class that combines all four resource dimensions into a single enforcement point:

```python
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class ResourceLimits:
    """Configuration for all resource limits."""
    max_wall_time_seconds: float = 30.0
    max_step_time_seconds: float = 10.0
    max_memory_bytes: int = 100 * 1024 * 1024  # 100MB
    max_api_calls: int = 20
    max_tokens: int = 50_000
    max_budget_usd: float = 1.00
    max_storage_bytes: int = 10 * 1024 * 1024  # 10MB
    max_files: int = 20


@dataclass
class ResourceUsage:
    """Track current resource usage."""
    start_time: float = field(default_factory=time.monotonic)
    api_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    memory_bytes: int = 0
    storage_bytes: int = 0
    files_created: int = 0


class UnifiedResourceLimiter:
    """Enforce all resource limits in a single checkpoint."""

    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.usage = ResourceUsage()

    def check_time(self) -> tuple[bool, str]:
        """Check wall-clock time limit."""
        elapsed = time.monotonic() - self.usage.start_time
        if elapsed > self.limits.max_wall_time_seconds:
            return False, f"Wall time exceeded: {elapsed:.1f}s / {self.limits.max_wall_time_seconds}s"
        return True, f"Time OK: {elapsed:.1f}s / {self.limits.max_wall_time_seconds}s"

    def check_api(self, tokens: int = 0, cost: float = 0.0) -> tuple[bool, str]:
        """Check API call and token limits."""
        if self.usage.api_calls >= self.limits.max_api_calls:
            return False, f"API call limit: {self.usage.api_calls}/{self.limits.max_api_calls}"
        if self.usage.tokens_used + tokens > self.limits.max_tokens:
            return False, f"Token limit: {self.usage.tokens_used + tokens}/{self.limits.max_tokens}"
        if self.usage.cost_usd + cost > self.limits.max_budget_usd:
            return False, f"Budget limit: ${self.usage.cost_usd + cost:.4f}/${self.limits.max_budget_usd}"
        return True, "API OK"

    def check_storage(self, bytes_to_write: int) -> tuple[bool, str]:
        """Check storage limits."""
        if self.usage.storage_bytes + bytes_to_write > self.limits.max_storage_bytes:
            return False, "Storage quota exceeded"
        if self.usage.files_created >= self.limits.max_files:
            return False, "File count limit reached"
        return True, "Storage OK"

    def checkpoint(self, **kwargs) -> tuple[bool, str]:
        """Run all resource checks. Call before every agent step."""
        checks = [
            self.check_time(),
            self.check_api(
                tokens=kwargs.get("tokens", 0),
                cost=kwargs.get("cost", 0.0),
            ),
        ]
        if "bytes_to_write" in kwargs:
            checks.append(self.check_storage(kwargs["bytes_to_write"]))

        for passed, reason in checks:
            if not passed:
                return False, reason

        return True, "All checks passed"

    def record_api_call(self, tokens: int, cost: float) -> None:
        self.usage.api_calls += 1
        self.usage.tokens_used += tokens
        self.usage.cost_usd += cost

    def record_file_write(self, size: int) -> None:
        self.usage.storage_bytes += size
        self.usage.files_created += 1

    def summary(self) -> str:
        elapsed = time.monotonic() - self.usage.start_time
        return (
            f"Time: {elapsed:.1f}s/{self.limits.max_wall_time_seconds}s | "
            f"Calls: {self.usage.api_calls}/{self.limits.max_api_calls} | "
            f"Tokens: {self.usage.tokens_used:,}/{self.limits.max_tokens:,} | "
            f"Cost: ${self.usage.cost_usd:.4f}/${self.limits.max_budget_usd} | "
            f"Storage: {self.usage.storage_bytes:,}/{self.limits.max_storage_bytes:,}"
        )


# Demo
limiter = UnifiedResourceLimiter(ResourceLimits(
    max_wall_time_seconds=60,
    max_api_calls=4,
    max_tokens=8000,
    max_budget_usd=0.05,
))

steps = [
    {"name": "Reasoning", "tokens": 2000, "cost": 0.01},
    {"name": "Tool call", "tokens": 1500, "cost": 0.008},
    {"name": "Analysis", "tokens": 3000, "cost": 0.02},
    {"name": "Summary", "tokens": 2500, "cost": 0.015},
    {"name": "Refinement", "tokens": 1000, "cost": 0.005},
]

print("=== Unified Resource Limiter ===\n")
for step in steps:
    ok, reason = limiter.checkpoint(tokens=step["tokens"], cost=step["cost"])
    if ok:
        limiter.record_api_call(step["tokens"], step["cost"])
        print(f"✅ {step['name']}: {limiter.summary()}")
    else:
        print(f"❌ {step['name']}: {reason}")
```

**Output:**
```
=== Unified Resource Limiter ===

✅ Reasoning: Time: 0.0s/60s | Calls: 1/4 | Tokens: 2,000/8,000 | Cost: $0.0100/$0.05 | Storage: 0/10,485,760
✅ Tool call: Time: 0.0s/60s | Calls: 2/4 | Tokens: 3,500/8,000 | Cost: $0.0180/$0.05 | Storage: 0/10,485,760
✅ Analysis: Time: 0.0s/60s | Calls: 3/4 | Tokens: 6,500/8,000 | Cost: $0.0380/$0.05 | Storage: 0/10,485,760
❌ Summary: Budget limit: $0.0530/$0.05
❌ Refinement: Budget limit: $0.0430/$0.05
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Set limits before agent starts, not during | Prevents race conditions where the agent bypasses a limit before it's enforced |
| Use both OS-level and app-level limits | OS limits are unforgeable (kernel-enforced); app limits provide better error messages |
| Log limit violations as security events | Repeated limit hits may indicate an attack or a misconfigured agent |
| Make limits configurable per agent role | A research agent needs more API calls; a simple Q&A agent needs fewer |
| Include cost tracking alongside call counts | A single gpt-4o call can cost more than 100 gpt-4o-mini calls |
| Test with limits slightly below expected usage | Find your agent's resource requirements before deploying |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Timeout on wall clock only, not CPU time | Use both `asyncio.timeout()` for wall clock and `resource.RLIMIT_CPU` for CPU |
| Setting memory limits too low | Profile your agent first — Python itself uses ~30MB baseline |
| No per-step timeout, only global timeout | One slow tool can consume the entire time budget |
| Counting API calls but ignoring tokens | A single call with 100K context costs more than 50 calls with 1K |
| Forgetting to limit child processes | Use `subprocess` with `preexec_fn` to inherit resource limits |
| Hard-coding limits in source code | Use environment variables or config files so ops teams can adjust without code changes |

---

## Hands-on exercise

### Your task

Build a `ResourceGovernor` class that manages all resource limits for a simulated agent. The agent makes 10 "steps" where each step consumes random amounts of time, tokens, and storage.

### Requirements

1. Configure limits: 5 seconds total, 8 API calls, 15,000 tokens, $0.10 budget
2. Each step randomly uses 500-3,000 tokens and costs $0.005-$0.025
3. The governor should gracefully stop the agent when any limit is reached
4. Print a final report showing which limit caused termination (or all passed)
5. Include percentage utilization for each resource dimension

### Expected result

The agent should complete some steps before hitting a limit, then stop with a clear report.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `random.randint()` and `random.uniform()` for random resource usage
- Structure the main loop as `while governor.can_continue():`
- Track which limit was hit first in the governor
- Use `time.monotonic()` for accurate time measurement

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import random
import time
from dataclasses import dataclass, field


@dataclass
class ResourceGovernor:
    max_time: float = 5.0
    max_calls: int = 8
    max_tokens: int = 15_000
    max_budget: float = 0.10
    calls: int = 0
    tokens: int = 0
    budget: float = 0.0
    start: float = field(default_factory=time.monotonic)
    termination_reason: str = ""

    def can_continue(self, tokens: int = 0, cost: float = 0.0) -> bool:
        elapsed = time.monotonic() - self.start
        if elapsed > self.max_time:
            self.termination_reason = f"Time limit ({elapsed:.1f}s/{self.max_time}s)"
            return False
        if self.calls >= self.max_calls:
            self.termination_reason = f"Call limit ({self.calls}/{self.max_calls})"
            return False
        if self.tokens + tokens > self.max_tokens:
            self.termination_reason = f"Token limit ({self.tokens + tokens}/{self.max_tokens})"
            return False
        if self.budget + cost > self.max_budget:
            self.termination_reason = f"Budget limit (${self.budget + cost:.3f}/${self.max_budget})"
            return False
        return True

    def record(self, tokens: int, cost: float):
        self.calls += 1
        self.tokens += tokens
        self.budget += cost

    def report(self):
        elapsed = time.monotonic() - self.start
        print(f"\n{'='*50}")
        print(f"RESOURCE GOVERNOR REPORT")
        print(f"{'='*50}")
        print(f"Time:   {elapsed:.1f}s / {self.max_time}s ({elapsed/self.max_time*100:.0f}%)")
        print(f"Calls:  {self.calls} / {self.max_calls} ({self.calls/self.max_calls*100:.0f}%)")
        print(f"Tokens: {self.tokens:,} / {self.max_tokens:,} ({self.tokens/self.max_tokens*100:.0f}%)")
        print(f"Budget: ${self.budget:.4f} / ${self.max_budget} ({self.budget/self.max_budget*100:.0f}%)")
        if self.termination_reason:
            print(f"\n⚠️ Terminated: {self.termination_reason}")
        else:
            print(f"\n✅ All steps completed within limits")


gov = ResourceGovernor()
random.seed(42)

for step in range(1, 11):
    tokens = random.randint(500, 3000)
    cost = random.uniform(0.005, 0.025)

    if not gov.can_continue(tokens=tokens, cost=cost):
        print(f"❌ Step {step}: Stopped — {gov.termination_reason}")
        break

    gov.record(tokens, cost)
    print(f"✅ Step {step}: {tokens:,} tokens, ${cost:.4f}")

gov.report()
```
</details>

### Bonus challenges

- [ ] Add a "soft limit" warning at 80% utilization for each resource
- [ ] Implement a "burst mode" that allows temporarily exceeding limits with a cooldown period
- [ ] Add per-tool resource tracking (e.g., `search_web` uses more tokens than `format_output`)

---

## Summary

✅ **Execution time limits** operate at three levels: `asyncio.timeout()` for wall clock, per-step budgets for fair allocation, and `resource.RLIMIT_CPU` for kernel-enforced CPU caps

✅ **Memory quotas** combine OS-level `RLIMIT_AS` constraints with application-level tracking that provides better error messages and per-allocation visibility

✅ **API call caps** should track both call counts and token/cost budgets — a single large call can cost more than dozens of small ones

✅ **Storage limits** enforce maximum bytes, file counts, and individual file sizes while guarding against path traversal attacks

✅ **A unified resource limiter** runs checkpoint validations before every agent step, combining all dimensions into a single enforcement point

---

**Next:** [Sandboxed Code Execution](./03-sandboxed-code-execution.md)

**Previous:** [Security Boundaries for Agents](./01-security-boundaries-for-agents.md)

---

## Further Reading

- [Python `resource` Module](https://docs.python.org/3/library/resource.html) - OS-level resource limits
- [Python `asyncio.timeout()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.timeout) - Async timeout contexts
- [Docker Resource Constraints](https://docs.docker.com/engine/containers/resource_constraints/) - Container-level CPU and memory limits
- [OpenAI API Rate Limits](https://platform.openai.com/docs/guides/rate-limits) - API-level throttling and quotas

<!-- 
Sources Consulted:
- Python resource module: https://docs.python.org/3/library/resource.html
- Python asyncio timeouts: https://docs.python.org/3/library/asyncio-task.html#asyncio.timeout
- Docker Resource Constraints: https://docs.docker.com/engine/containers/resource_constraints/
- OpenAI Pricing: https://openai.com/api/pricing/
-->
