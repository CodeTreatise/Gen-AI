---
title: "Advanced Typing"
---

# Advanced Typing

## Introduction

Advanced typing features enable generic programming, structural typing, and more precise type specifications for complex code patterns.

### What We'll Cover

- Generic types with TypeVar
- Callable for function types
- Protocol for structural typing
- TypedDict for dictionaries

### Prerequisites

- Common types
- Object-oriented Python

---

## Generic Types with TypeVar

### Basic TypeVar

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    """Return first item, preserving type."""
    return items[0]

# Type inference works!
name = first(["Alice", "Bob"])  # type: str
number = first([1, 2, 3])       # type: int
```

### Constrained TypeVar

```python
from typing import TypeVar

# Only allow specific types
Number = TypeVar("Number", int, float)

def add(a: Number, b: Number) -> Number:
    return a + b

add(1, 2)       # OK, returns int
add(1.0, 2.0)   # OK, returns float
add("a", "b")   # Type error! str not allowed
```

### Bound TypeVar

```python
from typing import TypeVar

class Animal:
    def speak(self) -> str:
        return "..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# Must be Animal or subclass
AnimalType = TypeVar("AnimalType", bound=Animal)

def make_speak(animal: AnimalType) -> str:
    return animal.speak()

make_speak(Dog())     # OK
make_speak("string")  # Type error!
```

### Generic Classes

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item
    
    def get(self) -> T:
        return self.item
    
    def replace(self, new_item: T) -> None:
        self.item = new_item

# Usage with type inference
int_box = Box(42)       # Box[int]
str_box = Box("hello")  # Box[str]

value: int = int_box.get()  # Type-safe!
```

---

## Callable

### Function Types

```python
from typing import Callable

# Function that takes str and returns int
Parser = Callable[[str], int]

def apply_parser(text: str, parser: Parser) -> int:
    return parser(text)

def my_parser(s: str) -> int:
    return len(s)

result = apply_parser("hello", my_parser)  # 5
```

### Multiple Parameters

```python
from typing import Callable

# (int, int) -> int
BinaryOp = Callable[[int, int], int]

def calculate(a: int, b: int, op: BinaryOp) -> int:
    return op(a, b)

calculate(10, 5, lambda x, y: x + y)  # 15
calculate(10, 5, lambda x, y: x * y)  # 50
```

### No Arguments

```python
from typing import Callable

# () -> str
Factory = Callable[[], str]

def create_message(factory: Factory) -> str:
    return factory()

create_message(lambda: "Hello!")
```

### Optional with Callable

```python
from typing import Callable, Optional

def process(
    data: str,
    callback: Optional[Callable[[str], None]] = None
) -> None:
    result = data.upper()
    if callback:
        callback(result)
```

---

## Protocol (Structural Typing)

### Basic Protocol

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

# No inheritance needed!
class Circle:
    def draw(self) -> str:
        return "○"

class Square:
    def draw(self) -> str:
        return "□"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # OK - has draw()
render(Square())  # OK - has draw()
```

### Protocol with Attributes

```python
from typing import Protocol

class Named(Protocol):
    name: str

class User:
    def __init__(self, name: str):
        self.name = name

class Product:
    def __init__(self, name: str):
        self.name = name

def greet(entity: Named) -> str:
    return f"Hello, {entity.name}!"

greet(User("Alice"))      # OK
greet(Product("Widget"))  # OK
```

### Combining Protocols

```python
from typing import Protocol

class Readable(Protocol):
    def read(self) -> str: ...

class Writable(Protocol):
    def write(self, data: str) -> None: ...

class ReadWritable(Readable, Writable, Protocol):
    pass

def copy(src: Readable, dest: Writable) -> None:
    data = src.read()
    dest.write(data)
```

---

## TypedDict

### Basic TypedDict

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    email: str
    age: int

# Must have all keys with correct types
user: User = {
    "name": "Alice",
    "email": "alice@example.com",
    "age": 30
}

# Type errors:
# user["name"] = 123      # Wrong type
# user["extra"] = "value" # Unknown key
```

### Optional Keys

```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    name: str
    email: str
    age: NotRequired[int]  # Optional key

# Valid without age
user: User = {
    "name": "Alice",
    "email": "alice@example.com"
}

# Also valid with age
user_with_age: User = {
    "name": "Bob",
    "email": "bob@example.com",
    "age": 25
}
```

### total=False

```python
from typing import TypedDict

class PartialUser(TypedDict, total=False):
    name: str
    email: str
    age: int

# All keys are optional
user1: PartialUser = {}
user2: PartialUser = {"name": "Alice"}
user3: PartialUser = {"name": "Bob", "age": 30}
```

### Nested TypedDict

```python
from typing import TypedDict

class Address(TypedDict):
    street: str
    city: str
    country: str

class User(TypedDict):
    name: str
    address: Address

user: User = {
    "name": "Alice",
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "country": "USA"
    }
}
```

---

## ParamSpec and Concatenate

### Preserving Function Signatures

```python
from typing import ParamSpec, Callable, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def with_logging(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that preserves function signature."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@with_logging
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}!" * times

# Type checker knows: greet(str, int) -> str
```

---

## Self Type

### Python 3.11+ Self

```python
from typing import Self

class Builder:
    def __init__(self) -> None:
        self.value = ""
    
    def add(self, text: str) -> Self:
        self.value += text
        return self
    
    def build(self) -> str:
        return self.value

# Method chaining with correct types
result = Builder().add("Hello").add(" World").build()
```

### Before Self (TypeVar approach)

```python
from typing import TypeVar

T = TypeVar("T", bound="Builder")

class Builder:
    def add(self: T, text: str) -> T:
        ...
```

---

## Hands-on Exercise

### Your Task

```python
# Create types for this pattern:
# 1. A Protocol for cacheable items
# 2. A generic cache class
# 3. TypedDict for cache entries
```

<details>
<summary>✅ Solution</summary>

```python
from typing import Protocol, TypeVar, Generic, TypedDict
from datetime import datetime

# 1. Protocol for cacheable items
class Cacheable(Protocol):
    @property
    def cache_key(self) -> str:
        ...

# 2. TypedDict for cache entries
class CacheEntry(TypedDict):
    value: object
    created_at: str
    expires_at: str | None

# 3. Generic cache class
T = TypeVar("T", bound=Cacheable)

class Cache(Generic[T]):
    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
    
    def get(self, item: T) -> object | None:
        key = item.cache_key
        if key in self._store:
            return self._store[key]["value"]
        return None
    
    def set(self, item: T, value: object, ttl: int | None = None) -> None:
        now = datetime.now().isoformat()
        entry: CacheEntry = {
            "value": value,
            "created_at": now,
            "expires_at": None
        }
        self._store[item.cache_key] = entry

# Usage
class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
    
    @property
    def cache_key(self) -> str:
        return f"user:{self.id}"

cache: Cache[User] = Cache()
user = User(1, "Alice")
cache.set(user, {"name": "Alice", "score": 100})
result = cache.get(user)
print(result)  # {'name': 'Alice', 'score': 100}
```
</details>

---

## Summary

✅ **TypeVar** creates generic type parameters
✅ **Callable** types functions and callbacks
✅ **Protocol** enables structural (duck) typing
✅ **TypedDict** types dictionaries with known keys
✅ **ParamSpec** preserves function signatures in decorators
✅ **Self** (3.11+) types method chaining

**Next:** [Pydantic](./04-pydantic.md)

---

## Further Reading

- [Generic Types](https://docs.python.org/3/library/typing.html#generics)
- [Protocols](https://docs.python.org/3/library/typing.html#protocols)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/typing.html
-->
