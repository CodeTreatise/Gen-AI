---
title: "Common Types"
---

# Common Types

## Introduction

The typing module provides types for collections, optional values, and more complex type expressions. Modern Python (3.9+) allows using built-in types directly.

### What We'll Cover

- Basic types
- Collection types
- Optional and Union
- Any type
- Type aliases

### Prerequisites

- Type hints basics

---

## Basic Types

### Built-in Types

```python
# No import needed
name: str = "Alice"
age: int = 30
height: float = 5.9
is_active: bool = True
data: bytes = b"hello"
```

### None Type

```python
# None is its own type
result: None = None

def log(msg: str) -> None:
    print(msg)
```

---

## Collection Types

### Lists

```python
# Python 3.9+: use list directly
names: list[str] = ["Alice", "Bob"]
numbers: list[int] = [1, 2, 3]
mixed: list[int | str] = [1, "two", 3]

# Nested lists
matrix: list[list[int]] = [[1, 2], [3, 4]]

# Python 3.8 and earlier
from typing import List
names: List[str] = ["Alice", "Bob"]
```

### Dictionaries

```python
# Python 3.9+
user: dict[str, str] = {"name": "Alice", "email": "a@b.com"}
scores: dict[str, int] = {"math": 90, "english": 85}
config: dict[str, bool | int] = {"debug": True, "port": 8080}

# Nested dicts
users: dict[int, dict[str, str]] = {
    1: {"name": "Alice"},
    2: {"name": "Bob"}
}
```

### Sets and Frozensets

```python
# Sets
tags: set[str] = {"python", "typing"}
ids: set[int] = {1, 2, 3}

# Frozensets (immutable)
constants: frozenset[str] = frozenset({"A", "B", "C"})
```

### Tuples

```python
# Fixed-length tuple with specific types
point: tuple[int, int] = (10, 20)
record: tuple[str, int, bool] = ("Alice", 30, True)

# Variable-length tuple (all same type)
values: tuple[int, ...] = (1, 2, 3, 4, 5)

# Empty tuple
empty: tuple[()] = ()
```

---

## Optional and Union

### Optional

```python
from typing import Optional

# Optional[X] means X | None
def find_user(id: int) -> Optional[dict]:
    """Returns user or None."""
    users = {1: {"name": "Alice"}}
    return users.get(id)

# Parameter with Optional
def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello!"
    return f"Hello, {name}!"
```

### Union

```python
from typing import Union

# Union[X, Y] means X or Y
def process(value: Union[int, str]) -> str:
    return str(value)

# Multiple types
def parse(data: Union[str, bytes, list]) -> dict:
    ...

# Python 3.10+ pipe syntax
def process(value: int | str) -> str:
    return str(value)

def parse(data: str | bytes | list) -> dict:
    ...
```

### Optional vs Union

```python
from typing import Optional, Union

# These are equivalent:
Optional[str]
Union[str, None]
str | None  # Python 3.10+
```

---

## Any Type

### When to Use Any

```python
from typing import Any

# Accept anything
def log(message: Any) -> None:
    print(str(message))

# Return anything
def get_value(key: str) -> Any:
    return cache.get(key)

# Mixed collections
data: list[Any] = [1, "two", 3.0, True]
```

### Any vs object

```python
from typing import Any

# Any: Opt out of type checking
def process_any(x: Any) -> Any:
    return x.anything()  # No type error

# object: Must work with all types
def process_object(x: object) -> str:
    return str(x)  # OK, str() works on all objects
    # x.specific_method()  # Type error! Not all objects have this
```

---

## Type Aliases

### Simple Aliases

```python
# Create readable aliases
UserId = int
Email = str
Score = float

def get_user_score(user_id: UserId) -> Score:
    return 85.5

# Complex type aliases
UserData = dict[str, str | int | bool]
Matrix = list[list[float]]
Callback = Callable[[str], None]
```

### TypeAlias (Python 3.10+)

```python
from typing import TypeAlias

# Explicit type alias declaration
Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]

def add_vectors(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]
```

### NewType for Distinct Types

```python
from typing import NewType

# Create distinct types (not just aliases)
UserId = NewType("UserId", int)
OrderId = NewType("OrderId", int)

def get_user(user_id: UserId) -> dict:
    ...

def get_order(order_id: OrderId) -> dict:
    ...

# Usage
user_id = UserId(123)
order_id = OrderId(456)

get_user(user_id)    # OK
get_user(order_id)   # Type error! OrderId is not UserId
get_user(123)        # Type error! int is not UserId
```

---

## Special Types

### Literal

```python
from typing import Literal

# Only specific values allowed
Mode = Literal["read", "write", "append"]

def open_file(path: str, mode: Mode) -> None:
    ...

open_file("test.txt", "read")   # OK
open_file("test.txt", "delete") # Type error!

# Multiple literals
Status = Literal["pending", "approved", "rejected"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
```

### Final

```python
from typing import Final

# Cannot be reassigned
MAX_SIZE: Final = 100
PI: Final[float] = 3.14159

MAX_SIZE = 200  # Type error! Cannot assign to Final

# Final class attributes
class Config:
    DEBUG: Final = False
```

### ClassVar

```python
from typing import ClassVar

class Counter:
    # Class variable (shared by all instances)
    count: ClassVar[int] = 0
    
    # Instance variable
    name: str
    
    def __init__(self, name: str) -> None:
        self.name = name
        Counter.count += 1
```

---

## Summary Table

| Type | Syntax | Example |
|------|--------|---------|
| List | `list[T]` | `list[int]` |
| Dict | `dict[K, V]` | `dict[str, int]` |
| Set | `set[T]` | `set[str]` |
| Tuple (fixed) | `tuple[T1, T2]` | `tuple[str, int]` |
| Tuple (variable) | `tuple[T, ...]` | `tuple[int, ...]` |
| Optional | `Optional[T]` or `T \| None` | `Optional[str]` |
| Union | `Union[T1, T2]` or `T1 \| T2` | `int \| str` |
| Any | `Any` | Anything |
| Literal | `Literal[...]` | `Literal["a", "b"]` |

---

## Hands-on Exercise

### Your Task

```python
# Add appropriate types to this code:

def create_report(data, title=None, format="json"):
    report = {
        "title": title or "Untitled",
        "data": data,
        "format": format
    }
    return report

def get_stats(numbers):
    if not numbers:
        return None
    return {
        "min": min(numbers),
        "max": max(numbers),
        "sum": sum(numbers)
    }
```

<details>
<summary>✅ Solution</summary>

```python
from typing import Literal, Optional, TypeAlias

# Type aliases for clarity
ReportFormat: TypeAlias = Literal["json", "csv", "xml"]
Stats: TypeAlias = dict[str, int | float]
Report: TypeAlias = dict[str, str | list | ReportFormat]

def create_report(
    data: list[dict[str, str | int]],
    title: Optional[str] = None,
    format: ReportFormat = "json"
) -> Report:
    """Create a report from data."""
    report: Report = {
        "title": title or "Untitled",
        "data": data,
        "format": format
    }
    return report

def get_stats(numbers: list[int | float]) -> Optional[Stats]:
    """Calculate statistics for a list of numbers."""
    if not numbers:
        return None
    return {
        "min": min(numbers),
        "max": max(numbers),
        "sum": sum(numbers)
    }

# Test
report = create_report(
    [{"name": "Alice", "score": 90}],
    title="Scores",
    format="json"
)
print(report)

stats = get_stats([1, 2, 3, 4, 5])
print(stats)  # {'min': 1, 'max': 5, 'sum': 15}

empty_stats = get_stats([])
print(empty_stats)  # None
```
</details>

---

## Summary

✅ **`list[T]`, `dict[K,V]`, `set[T]`** for collections
✅ **`Optional[T]`** for values that might be None
✅ **`Union[T1, T2]`** or **`T1 | T2`** for multiple types
✅ **`Any`** opts out of type checking
✅ **Type aliases** improve readability
✅ **`Literal`** restricts to specific values

**Next:** [Advanced Typing](./03-advanced-typing.md)

---

## Further Reading

- [typing Module](https://docs.python.org/3/library/typing.html)
- [PEP 585 - Type Hinting Generics](https://peps.python.org/pep-0585/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/typing.html
-->
