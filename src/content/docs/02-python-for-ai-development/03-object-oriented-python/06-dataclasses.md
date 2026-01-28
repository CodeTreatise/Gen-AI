---
title: "Dataclasses"
---

# Dataclasses

## Introduction

Dataclasses are a modern Python feature that automatically generate boilerplate code for classes that primarily store data. They reduce code while adding powerful features like automatic `__init__`, `__repr__`, comparisons, and more.

### What We'll Cover

- Basic @dataclass usage
- Field options
- Default values
- Frozen (immutable) dataclasses
- Post-init processing
- Inheritance

### Prerequisites

- Classes and objects
- Type hints

---

## Basic Dataclass

### Without Dataclass (Traditional)

```python
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
```

### With Dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Automatically generates:
# - __init__(self, x: float, y: float)
# - __repr__() -> "Point(x=..., y=...)"
# - __eq__() for comparison

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)        # Point(x=3, y=4)
print(p1 == p2)  # True
```

---

## Dataclass Options

### @dataclass Parameters

```python
from dataclasses import dataclass

@dataclass(
    init=True,       # Generate __init__ (default: True)
    repr=True,       # Generate __repr__ (default: True)
    eq=True,         # Generate __eq__ (default: True)
    order=False,     # Generate __lt__, __le__, etc. (default: False)
    frozen=False,    # Make immutable (default: False)
    slots=False,     # Use __slots__ for memory efficiency (3.10+)
    kw_only=False,   # All fields keyword-only (3.10+)
)
class Example:
    value: int
```

### Ordered Comparisons

```python
from dataclasses import dataclass

@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int

v1 = Version(1, 0, 0)
v2 = Version(2, 0, 0)
v3 = Version(1, 5, 0)

print(v1 < v2)    # True
print(v1 < v3)    # True (compares major, then minor, then patch)

# Can sort
versions = [v2, v1, v3]
print(sorted(versions))  # [Version(1, 0, 0), Version(1, 5, 0), Version(2, 0, 0)]
```

---

## Field Defaults

### Simple Defaults

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    active: bool = True      # Default value
    role: str = "user"       # Default value

u1 = User("Alice", "alice@example.com")
print(u1)  # User(name='Alice', email='alice@example.com', active=True, role='user')

u2 = User("Bob", "bob@example.com", active=False, role="admin")
print(u2)  # User(name='Bob', email='bob@example.com', active=False, role='admin')
```

### Mutable Default Values

```python
from dataclasses import dataclass, field

# ❌ Wrong - mutable default
@dataclass
class WrongWay:
    items: list = []  # Error! Mutable default

# ✅ Correct - use field with default_factory
@dataclass
class RightWay:
    items: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)

r1 = RightWay()
r2 = RightWay()

r1.items.append("a")
print(r1.items)  # ["a"]
print(r2.items)  # [] - independent!
```

### Field Options

```python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    
    # Not in __init__, calculated later
    id: str = field(init=False, default="")
    
    # Not in __repr__
    _internal: str = field(repr=False, default="hidden")
    
    # Not compared in __eq__
    metadata: dict = field(compare=False, default_factory=dict)
    
    # Custom default factory
    tags: list = field(default_factory=lambda: ["general"])
    
    def __post_init__(self):
        import uuid
        self.id = str(uuid.uuid4())[:8]

p = Product("Widget", 9.99)
print(p)  # Product(name='Widget', price=9.99, id='a1b2c3d4', tags=['general'])
```

---

## Frozen Dataclasses

### Immutable Objects

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(3, 4)
# p.x = 5  # FrozenInstanceError!

# Hashable - can use as dict key or in sets
points = {Point(1, 2), Point(3, 4)}
cache = {Point(0, 0): "origin"}
```

### Immutable with Mutable Fields

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    name: str
    # ⚠️ List is still mutable!
    values: tuple = field(default_factory=tuple)  # Use tuple instead

config = Config("settings", (1, 2, 3))
# Truly immutable
```

---

## Post-Init Processing

### `__post_init__` Method

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)

r = Rectangle(4, 5)
print(r.area)       # 20
print(r.perimeter)  # 18
```

### Validation in post_init

```python
from dataclasses import dataclass

@dataclass
class Email:
    address: str
    
    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError(f"Invalid email: {self.address}")
        self.address = self.address.lower()

email = Email("Alice@Example.COM")
print(email.address)  # "alice@example.com"

# Email("invalid")  # ValueError!
```

### InitVar for Initialization-Only Fields

```python
from dataclasses import dataclass, field, InitVar

@dataclass
class User:
    name: str
    password: InitVar[str]  # Not stored, only used in __post_init__
    password_hash: str = field(init=False)
    
    def __post_init__(self, password: str):
        import hashlib
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()[:16]

user = User("Alice", "secret123")
print(user)  # User(name='Alice', password_hash='2bb80d53...')
# user.password  # AttributeError - not stored
```

---

## Dataclass Inheritance

### Basic Inheritance

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: str
    department: str

emp = Employee("Alice", 30, "E001", "Engineering")
print(emp)  # Employee(name='Alice', age=30, employee_id='E001', department='Engineering')
```

### Default Values in Inheritance

```python
from dataclasses import dataclass, field

@dataclass
class Animal:
    name: str
    species: str = "Unknown"

@dataclass
class Dog(Animal):
    breed: str = "Mixed"
    species: str = "Canine"  # Override default

dog = Dog("Buddy")
print(dog)  # Dog(name='Buddy', species='Canine', breed='Mixed')
```

---

## Utility Functions

### asdict and astuple

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class Point:
    x: float
    y: float

p = Point(3, 4)

# Convert to dict
d = asdict(p)
print(d)  # {"x": 3, "y": 4}

# Convert to tuple
t = astuple(p)
print(t)  # (3, 4)

# Useful for JSON serialization
import json
print(json.dumps(asdict(p)))  # '{"x": 3, "y": 4}'
```

### fields() and replace()

```python
from dataclasses import dataclass, fields, replace

@dataclass
class Config:
    host: str
    port: int
    debug: bool = False

config = Config("localhost", 8080)

# Inspect fields
for f in fields(config):
    print(f"{f.name}: {f.type.__name__} = {getattr(config, f.name)}")
# host: str = localhost
# port: int = 8080
# debug: bool = False

# Create modified copy
new_config = replace(config, port=9090, debug=True)
print(new_config)  # Config(host='localhost', port=9090, debug=True)
print(config)      # Config(host='localhost', port=8080, debug=False) - unchanged
```

---

## Dataclass vs NamedTuple

| Feature | dataclass | NamedTuple |
|---------|-----------|------------|
| Mutable | Yes (default) | No |
| Default values | ✅ | ✅ |
| Methods | ✅ | ✅ |
| Inheritance | ✅ | Limited |
| Memory | Standard | Smaller |
| Iterable | No | Yes |
| Dict key | If frozen | Yes |

```python
from dataclasses import dataclass
from typing import NamedTuple

# NamedTuple - lighter, immutable, iterable
class PointNT(NamedTuple):
    x: float
    y: float

# Dataclass - more features, mutable by default
@dataclass
class PointDC:
    x: float
    y: float

pnt = PointNT(3, 4)
pdc = PointDC(3, 4)

# NamedTuple is iterable
x, y = pnt  # Works
# x, y = pdc  # TypeError

# NamedTuple is hashable
{pnt: "a"}  # Works
# {pdc: "a"}  # TypeError (unless frozen=True)
```

---

## Slots for Memory Efficiency

```python
from dataclasses import dataclass
import sys

@dataclass
class RegularPoint:
    x: float
    y: float

@dataclass(slots=True)  # Python 3.10+
class SlotPoint:
    x: float
    y: float

regular = RegularPoint(3, 4)
slot = SlotPoint(3, 4)

print(sys.getsizeof(regular))  # ~48 bytes
print(sys.getsizeof(slot))     # ~32 bytes

# Slots are faster for attribute access too
```

---

## Hands-on Exercise

### Your Task

Create a dataclass-based system for a library:

```python
# Requirements:
# 1. Book dataclass with isbn, title, author, year, available (default True)
# 2. Library dataclass with name and books list
# 3. Library methods: add_book, find_by_author, checkout, return_book
# 4. Use frozen Book if you want immutability

# Example:
book = Book("978-0-13-468599-1", "Clean Code", "Robert C. Martin", 2008)
library = Library("City Library")
library.add_book(book)
print(library.find_by_author("Robert"))  # [Book(...)]
```

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Book:
    isbn: str
    title: str
    author: str
    year: int
    available: bool = True
    
    def __hash__(self):
        return hash(self.isbn)

@dataclass
class Library:
    name: str
    books: list[Book] = field(default_factory=list)
    
    def add_book(self, book: Book) -> None:
        """Add a book to the library."""
        if any(b.isbn == book.isbn for b in self.books):
            raise ValueError(f"Book with ISBN {book.isbn} already exists")
        self.books.append(book)
    
    def find_by_author(self, author: str) -> list[Book]:
        """Find books by author (partial match)."""
        return [b for b in self.books if author.lower() in b.author.lower()]
    
    def find_by_isbn(self, isbn: str) -> Optional[Book]:
        """Find a book by ISBN."""
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None
    
    def checkout(self, isbn: str) -> bool:
        """Checkout a book by ISBN."""
        book = self.find_by_isbn(isbn)
        if book and book.available:
            book.available = False
            return True
        return False
    
    def return_book(self, isbn: str) -> bool:
        """Return a book by ISBN."""
        book = self.find_by_isbn(isbn)
        if book and not book.available:
            book.available = True
            return True
        return False
    
    @property
    def available_books(self) -> list[Book]:
        """Get all available books."""
        return [b for b in self.books if b.available]

# Test
book1 = Book("978-0-13-468599-1", "Clean Code", "Robert C. Martin", 2008)
book2 = Book("978-0-13-235088-4", "Clean Architecture", "Robert C. Martin", 2017)
book3 = Book("978-0-596-51774-8", "JavaScript: The Good Parts", "Douglas Crockford", 2008)

library = Library("City Library")
library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

print(library.find_by_author("Robert"))
# [Book(...Clean Code...), Book(...Clean Architecture...)]

library.checkout("978-0-13-468599-1")
print(book1.available)  # False
print(len(library.available_books))  # 2
```
</details>

---

## Summary

✅ **@dataclass** auto-generates `__init__`, `__repr__`, `__eq__`
✅ Use **field()** for mutable defaults and special options
✅ **frozen=True** creates immutable, hashable objects
✅ **`__post_init__`** for validation and computed fields
✅ **InitVar** for init-only parameters
✅ **asdict/astuple** for serialization
✅ **slots=True** for memory efficiency (Python 3.10+)

**Back to:** [OOP Overview](./00-object-oriented-python.md)

---

## Further Reading

- [dataclasses Module](https://docs.python.org/3/library/dataclasses.html)
- [PEP 557 - Data Classes](https://peps.python.org/pep-0557/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/dataclasses.html
-->
