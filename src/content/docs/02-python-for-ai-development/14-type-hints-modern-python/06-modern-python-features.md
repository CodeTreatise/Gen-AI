---
title: "Modern Python Features"
---

# Modern Python Features

## Introduction

Modern Python (3.10+) introduces features that make code more expressive and type-safe. These features improve readability and reduce boilerplate.

### What We'll Cover

- Match statements (3.10+)
- Walrus operator (:=)
- Union with pipe operator
- Exception groups (3.11+)
- Self type (3.11+)

### Prerequisites

- Python basics
- Type hints

---

## Match Statements (3.10+)

### Basic Pattern Matching

```python
def handle_command(command: str) -> str:
    match command:
        case "start":
            return "Starting..."
        case "stop":
            return "Stopping..."
        case "restart":
            return "Restarting..."
        case _:  # Default case
            return f"Unknown command: {command}"

print(handle_command("start"))    # Starting...
print(handle_command("unknown"))  # Unknown command: unknown
```

### Matching Values

```python
def http_status(code: int) -> str:
    match code:
        case 200:
            return "OK"
        case 201:
            return "Created"
        case 400:
            return "Bad Request"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:
            return f"Status {code}"
```

### Matching Patterns with Guards

```python
def classify_number(n: int) -> str:
    match n:
        case 0:
            return "zero"
        case n if n < 0:
            return "negative"
        case n if n % 2 == 0:
            return "positive even"
        case _:
            return "positive odd"

print(classify_number(-5))  # negative
print(classify_number(4))   # positive even
print(classify_number(7))   # positive odd
```

### Matching Sequences

```python
def process_point(point: tuple) -> str:
    match point:
        case (0, 0):
            return "Origin"
        case (x, 0):
            return f"On X-axis at {x}"
        case (0, y):
            return f"On Y-axis at {y}"
        case (x, y):
            return f"Point at ({x}, {y})"
        case _:
            return "Not a point"

print(process_point((0, 0)))   # Origin
print(process_point((5, 0)))   # On X-axis at 5
print(process_point((3, 4)))   # Point at (3, 4)
```

### Matching Dictionaries

```python
def handle_event(event: dict) -> str:
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": "scroll", "direction": d}:
            return f"Scroll {d}"
        case _:
            return "Unknown event"

print(handle_event({"type": "click", "x": 100, "y": 200}))
# Click at (100, 200)
```

### Matching Classes

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

@dataclass
class Rectangle:
    corner: Point
    width: float
    height: float

def area(shape) -> float:
    match shape:
        case Circle(radius=r):
            return 3.14159 * r * r
        case Rectangle(width=w, height=h):
            return w * h
        case _:
            raise ValueError("Unknown shape")

print(area(Circle(Point(0, 0), 5)))  # 78.53975
print(area(Rectangle(Point(0, 0), 10, 5)))  # 50
```

---

## Walrus Operator (:=)

### Assignment Expressions

```python
# Traditional way
data = get_data()
if data:
    process(data)

# With walrus operator
if data := get_data():
    process(data)
```

### In While Loops

```python
# Traditional
line = file.readline()
while line:
    process(line)
    line = file.readline()

# With walrus operator
while line := file.readline():
    process(line)
```

### In List Comprehensions

```python
# Filter and transform in one pass
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Traditional (calls expensive twice)
results = [expensive(n) for n in numbers if expensive(n) > 10]

# With walrus (calls expensive once)
results = [y for n in numbers if (y := expensive(n)) > 10]
```

### Regex Matching

```python
import re

text = "Contact: john@example.com"

# Traditional
match = re.search(r"[\w.-]+@[\w.-]+", text)
if match:
    email = match.group()
    print(f"Found: {email}")

# With walrus
if match := re.search(r"[\w.-]+@[\w.-]+", text):
    print(f"Found: {match.group()}")
```

---

## Union with Pipe Operator (3.10+)

### Type Unions

```python
# Old syntax (still valid)
from typing import Union
def process(value: Union[int, str]) -> str:
    return str(value)

# New syntax (3.10+)
def process(value: int | str) -> str:
    return str(value)

# Optional shorthand
from typing import Optional
def old_optional(x: Optional[str]) -> str: ...
def new_optional(x: str | None) -> str: ...  # Equivalent
```

### Multiple Types

```python
# Clean syntax for multiple types
def parse(data: str | bytes | list) -> dict:
    ...

# Works with generics
def get_items() -> list[int | str]:
    return [1, "two", 3]
```

### isinstance() with Union

```python
# Works with isinstance (3.10+)
def process(value: int | str | float) -> None:
    if isinstance(value, int | float):
        print(f"Number: {value * 2}")
    else:
        print(f"String: {value.upper()}")
```

---

## Exception Groups (3.11+)

### ExceptionGroup

```python
# Raise multiple exceptions at once
def validate(data: dict) -> None:
    errors = []
    
    if "name" not in data:
        errors.append(ValueError("Missing name"))
    if "email" not in data:
        errors.append(ValueError("Missing email"))
    if "age" in data and data["age"] < 0:
        errors.append(ValueError("Age cannot be negative"))
    
    if errors:
        raise ExceptionGroup("Validation failed", errors)

try:
    validate({"age": -5})
except ExceptionGroup as eg:
    for error in eg.exceptions:
        print(f"Error: {error}")
```

### except* Syntax

```python
try:
    raise ExceptionGroup("errors", [
        ValueError("bad value"),
        TypeError("wrong type"),
        KeyError("missing key")
    ])
except* ValueError as eg:
    print(f"Value errors: {eg.exceptions}")
except* TypeError as eg:
    print(f"Type errors: {eg.exceptions}")
except* KeyError as eg:
    print(f"Key errors: {eg.exceptions}")
```

### With asyncio

```python
import asyncio

async def task1():
    raise ValueError("Task 1 failed")

async def task2():
    raise TypeError("Task 2 failed")

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task1())
        tg.create_task(task2())
    # Raises ExceptionGroup with both exceptions

try:
    asyncio.run(main())
except* ValueError as eg:
    print("ValueError occurred")
except* TypeError as eg:
    print("TypeError occurred")
```

---

## Self Type (3.11+)

### Method Chaining

```python
from typing import Self

class Builder:
    def __init__(self) -> None:
        self._value = ""
    
    def add(self, text: str) -> Self:
        self._value += text
        return self
    
    def add_line(self, text: str) -> Self:
        self._value += text + "\n"
        return self
    
    def build(self) -> str:
        return self._value

result = (
    Builder()
    .add_line("Hello")
    .add_line("World")
    .build()
)
```

### Factory Methods

```python
from typing import Self

class User:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email
    
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(data["name"], data["email"])

class AdminUser(User):
    def __init__(self, name: str, email: str) -> None:
        super().__init__(name, email)
        self.role = "admin"

# Works correctly with subclasses
admin = AdminUser.from_dict({"name": "Alice", "email": "a@b.com"})
# admin is AdminUser, not User
```

---

## Python 3.12+ Features

### Type Parameter Syntax

```python
# Old syntax
from typing import TypeVar, Generic
T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

# New syntax (3.12+)
class Box[T]:
    def __init__(self, item: T) -> None:
        self.item = item

# Generic functions
def first[T](items: list[T]) -> T:
    return items[0]
```

---

## Hands-on Exercise

### Your Task

```python
# Refactor this code using modern Python features:
# 1. Use match statement
# 2. Use walrus operator
# 3. Use pipe union syntax

from typing import Union, Optional

def process_response(response: Union[dict, str, None]) -> str:
    if response is None:
        return "No response"
    elif isinstance(response, str):
        return response.upper()
    elif isinstance(response, dict):
        status = response.get("status")
        if status is not None:
            if status == "success":
                return "OK"
            elif status == "error":
                message = response.get("message")
                if message is not None:
                    return f"Error: {message}"
                return "Unknown error"
    return "Invalid response"
```

<details>
<summary>✅ Solution</summary>

```python
# Modern Python version (3.10+)

def process_response(response: dict | str | None) -> str:
    match response:
        case None:
            return "No response"
        case str() as s:
            return s.upper()
        case {"status": "success"}:
            return "OK"
        case {"status": "error", "message": msg}:
            return f"Error: {msg}"
        case {"status": "error"}:
            return "Unknown error"
        case _:
            return "Invalid response"

# Test cases
print(process_response(None))
# No response

print(process_response("hello"))
# HELLO

print(process_response({"status": "success"}))
# OK

print(process_response({"status": "error", "message": "Not found"}))
# Error: Not found

print(process_response({"status": "error"}))
# Unknown error

print(process_response({"invalid": "data"}))
# Invalid response
```
</details>

---

## Summary

✅ **Match statements** provide powerful pattern matching
✅ **Walrus operator** `:=` assigns and tests in one expression
✅ **Pipe `|`** creates cleaner union types
✅ **ExceptionGroup** handles multiple exceptions
✅ **Self type** enables correct subclass typing
✅ **Python 3.12** adds inline type parameters

**Back to:** [Type Hints & Modern Python Overview](./00-type-hints-modern-python.md)

---

## Further Reading

- [PEP 634 - Match Statement](https://peps.python.org/pep-0634/)
- [PEP 572 - Assignment Expressions](https://peps.python.org/pep-0572/)
- [PEP 604 - Union with Pipe](https://peps.python.org/pep-0604/)
- [PEP 654 - Exception Groups](https://peps.python.org/pep-0654/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/whatsnew/
-->
