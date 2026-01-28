---
title: "Type Hints Basics"
---

# Type Hints Basics

## Introduction

Type hints add optional type information to Python code. They don't affect runtime behavior but enable static analysis, better IDE support, and self-documenting code.

### What We'll Cover

- Function annotations
- Variable annotations
- Return type hints
- None and Optional types
- Type hints are optional

### Prerequisites

- Python functions and variables

---

## Function Annotations

### Basic Syntax

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def calculate_average(numbers: list) -> float:
    return sum(numbers) / len(numbers)
```

### Parameters with Defaults

```python
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

def process(data: list, limit: int = 10, verbose: bool = False) -> list:
    if verbose:
        print(f"Processing {len(data)} items")
    return data[:limit]
```

### Multiple Parameters

```python
def create_user(
    username: str,
    email: str,
    age: int,
    active: bool = True
) -> dict:
    return {
        "username": username,
        "email": email,
        "age": age,
        "active": active
    }
```

---

## Variable Annotations

### Basic Variable Types

```python
# Variable annotations
name: str = "Alice"
age: int = 30
price: float = 19.99
is_active: bool = True

# Without initial value (declaration only)
user_id: int  # Declared but not assigned
```

### Class Attributes

```python
class User:
    # Class variable
    default_role: str = "user"
    
    # Instance variables (declared in __init__)
    name: str
    email: str
    age: int
    
    def __init__(self, name: str, email: str, age: int) -> None:
        self.name = name
        self.email = email
        self.age = age
```

---

## Return Type Hints

### Basic Returns

```python
def get_name() -> str:
    return "Alice"

def get_count() -> int:
    return 42

def is_valid() -> bool:
    return True
```

### Returning None

```python
def print_message(message: str) -> None:
    """Function that doesn't return anything."""
    print(message)
    # Implicit return None

def log_error(error: str) -> None:
    print(f"ERROR: {error}")
    return None  # Explicit return None
```

### Optional Returns

```python
from typing import Optional

def find_user(user_id: int) -> Optional[dict]:
    """Return user dict or None if not found."""
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)  # Returns None if not found

# Python 3.10+ alternative
def find_user_modern(user_id: int) -> dict | None:
    users = {1: {"name": "Alice"}}
    return users.get(user_id)
```

---

## None Type

### Explicit None

```python
from typing import Optional

# These are equivalent:
def func1() -> Optional[str]:
    return None

def func2() -> str | None:  # Python 3.10+
    return None

# Accepting None as parameter
def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, stranger!"
    return f"Hello, {name}!"
```

### None vs Optional

```python
from typing import Optional

# Optional[X] means X | None
# Use when value might not exist

def get_config_value(key: str) -> Optional[str]:
    """Returns value or None if key not found."""
    config = {"debug": "true"}
    return config.get(key)

# Not Optional - always returns str
def get_required_value(key: str) -> str:
    """Raises if key not found."""
    config = {"debug": "true"}
    if key not in config:
        raise KeyError(f"Missing required key: {key}")
    return config[key]
```

---

## Type Hints Are Optional

### Runtime Behavior

```python
# Type hints don't enforce types at runtime!
def add(a: int, b: int) -> int:
    return a + b

# These all work at runtime:
add(1, 2)       # Returns 3 (intended use)
add("a", "b")   # Returns "ab" (no runtime error!)
add([1], [2])   # Returns [1, 2] (no runtime error!)

# Type checkers catch the errors:
# mypy: error: Argument 1 to "add" has incompatible type "str"
```

### Gradual Typing

```python
# You can add types gradually
# Start without types:
def process(data):
    return data * 2

# Add parameter types:
def process(data: list):
    return data * 2

# Add return type:
def process(data: list) -> list:
    return data * 2

# Add more specific types:
def process(data: list[int]) -> list[int]:
    return [x * 2 for x in data]
```

### Using Any for Flexibility

```python
from typing import Any

def log(message: Any) -> None:
    """Accept any type."""
    print(str(message))

def passthrough(value: Any) -> Any:
    """Return value unchanged."""
    return value
```

---

## Accessing Type Hints

### \_\_annotations\_\_ Attribute

```python
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}!" * times

print(greet.__annotations__)
# {'name': <class 'str'>, 'times': <class 'int'>, 'return': <class 'str'>}
```

### get_type_hints()

```python
from typing import get_type_hints

class User:
    name: str
    age: int
    
    def greet(self) -> str:
        return f"Hi, I'm {self.name}"

print(get_type_hints(User))
# {'name': <class 'str'>, 'age': <class 'int'>}

print(get_type_hints(User.greet))
# {'return': <class 'str'>}
```

---

## Best Practices

### 1. Be Consistent

```python
# ✅ Good: All parameters and return typed
def calculate(a: int, b: int, operation: str) -> int:
    ...

# ❌ Inconsistent: Some typed, some not
def calculate(a: int, b, operation: str):
    ...
```

### 2. Start with Public APIs

```python
# Type public functions first
def public_function(data: list[str]) -> dict[str, int]:
    return _internal_process(data)

# Internal functions can be typed later
def _internal_process(data):
    ...
```

### 3. Use Modern Syntax

```python
# Python 3.10+ preferred syntax
def process(items: list[str]) -> dict[str, int]:
    ...

# Older syntax (still valid)
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...
```

---

## Hands-on Exercise

### Your Task

```python
# Add type hints to these functions:

def get_user_info(user_id, include_email=True):
    user = {"id": user_id, "name": "Alice"}
    if include_email:
        user["email"] = "alice@example.com"
    return user

def find_item(items, predicate):
    for item in items:
        if predicate(item):
            return item
    return None

def format_names(names):
    return [name.upper() for name in names]
```

<details>
<summary>✅ Solution</summary>

```python
from typing import Optional, Callable, TypeVar

T = TypeVar("T")

def get_user_info(
    user_id: int, 
    include_email: bool = True
) -> dict[str, str | int]:
    """Get user info by ID."""
    user: dict[str, str | int] = {"id": user_id, "name": "Alice"}
    if include_email:
        user["email"] = "alice@example.com"
    return user

def find_item(
    items: list[T], 
    predicate: Callable[[T], bool]
) -> Optional[T]:
    """Find first item matching predicate."""
    for item in items:
        if predicate(item):
            return item
    return None

def format_names(names: list[str]) -> list[str]:
    """Format names to uppercase."""
    return [name.upper() for name in names]

# Test the typed functions
user = get_user_info(1)
print(user)  # {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}

result = find_item([1, 2, 3, 4], lambda x: x > 2)
print(result)  # 3

formatted = format_names(["alice", "bob"])
print(formatted)  # ['ALICE', 'BOB']
```
</details>

---

## Summary

✅ **Function annotations** specify parameter and return types
✅ **Variable annotations** declare variable types
✅ **`-> None`** for functions that don't return values
✅ **`Optional[T]`** or **`T | None`** for nullable types
✅ **Type hints are optional** and don't affect runtime
✅ **Gradual typing** lets you add types incrementally

**Next:** [Common Types](./02-common-types.md)

---

## Further Reading

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/typing.html
-->
