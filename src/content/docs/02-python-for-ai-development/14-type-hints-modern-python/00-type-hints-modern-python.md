---
title: "Type Hints & Modern Python"
---

# Type Hints & Modern Python

## Overview

Type hints make Python code more readable, maintainable, and catch errors before runtime. Combined with modern Python features, they enable safer, more expressive code.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-type-hints-basics.md) | Type Hints Basics | Annotations, returns, variables |
| [02](./02-common-types.md) | Common Types | List, Dict, Optional, Union |
| [03](./03-advanced-typing.md) | Advanced Typing | Generics, Protocol, TypedDict |
| [04](./04-pydantic.md) | Pydantic | Data validation, models |
| [05](./05-type-checking-tools.md) | Type Checking Tools | mypy, pyright, CI integration |
| [06](./06-modern-python-features.md) | Modern Python Features | Match, walrus, pipe union |

---

## Quick Start

```python
# Modern Python with type hints (3.10+)
from typing import Optional

def greet(name: str, times: int = 1) -> str:
    """Greet a person."""
    return f"Hello, {name}! " * times

def process(data: list[dict[str, int]]) -> Optional[int]:
    """Process data and return total or None."""
    if not data:
        return None
    return sum(item.get("value", 0) for item in data)

# Works with type checkers
result = greet("Alice", 3)
total = process([{"value": 10}, {"value": 20}])
```

---

## Why Type Hints?

| Benefit | Description |
|---------|-------------|
| **Documentation** | Types show expected inputs/outputs |
| **Error Detection** | Catch bugs before runtime |
| **IDE Support** | Better autocomplete, refactoring |
| **Maintainability** | Easier to understand and modify |

---

## Python Version Features

| Version | Feature |
|---------|---------|
| 3.5+ | Basic type hints |
| 3.9+ | `list[int]` instead of `List[int]` |
| 3.10+ | `X \| Y` instead of `Union[X, Y]` |
| 3.10+ | Match statements |
| 3.11+ | `Self` type |
| 3.12+ | Type parameter syntax |

---

## Prerequisites

Before starting this lesson:
- Python basics
- Functions and classes
- Basic OOP concepts

---

## Start Learning

Begin with [Type Hints Basics](./01-type-hints-basics.md) to understand annotation syntax.

---

## Further Reading

- [typing Module](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
