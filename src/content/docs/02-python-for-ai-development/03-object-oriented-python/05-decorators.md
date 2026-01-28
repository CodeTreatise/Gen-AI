---
title: "Decorators"
---

# Decorators

## Introduction

Decorators are functions that modify the behavior of other functions or classes. They're a powerful way to add functionality without changing the original code—used extensively for logging, caching, authentication, and more.

### What We'll Cover

- Function decorators
- Decorator syntax
- Decorators with arguments
- functools.wraps
- Class decorators
- Common built-in decorators

### Prerequisites

- Functions
- Classes

---

## Function Decorators

### Understanding the Pattern

```python
# A decorator is a function that takes a function and returns a function
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

# Apply decorator manually
def greet(name):
    return f"Hello, {name}!"

greet = my_decorator(greet)  # Wrap the function

print(greet("Alice"))
# Before function call
# After function call
# Hello, Alice!
```

### The @ Syntax

```python
# Same as above, but cleaner
@my_decorator
def greet(name):
    return f"Hello, {name}!"

# This is syntactic sugar for:
# greet = my_decorator(greet)
```

---

## Practical Decorator Examples

### Timing Decorator

```python
import time
from functools import wraps

def timer(func):
    """Measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function took 1.0023 seconds
```

### Logging Decorator

```python
from functools import wraps

def log_calls(func):
    """Log function calls with arguments."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(3, 5)
# Calling add(3, 5)
# add returned 8
```

### Retry Decorator

```python
import time
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry a function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("API unavailable")
    return "Success!"
```

---

## functools.wraps

### Why Use wraps?

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greet someone by name."""
    return f"Hello, {name}!"

# Without @wraps, metadata is lost
print(greet.__name__)  # "wrapper" - wrong!
print(greet.__doc__)   # None - wrong!

# With @wraps
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserve metadata
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greet someone by name."""
    return f"Hello, {name}!"

print(greet.__name__)  # "greet" - correct!
print(greet.__doc__)   # "Greet someone by name." - correct!
```

---

## Decorators with Arguments

### Creating Parameterized Decorators

```python
from functools import wraps

def repeat(times: int):
    """Repeat a function call multiple times."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# ["Hello, Alice!", "Hello, Alice!", "Hello, Alice!"]
```

### Optional Arguments Pattern

```python
from functools import wraps

def cache(func=None, *, max_size: int = 128):
    """Cache function results with optional max size."""
    def decorator(fn):
        _cache = {}
        
        @wraps(fn)
        def wrapper(*args):
            if args not in _cache:
                if len(_cache) >= max_size:
                    _cache.pop(next(iter(_cache)))
                _cache[args] = fn(*args)
            return _cache[args]
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator

# Can use with or without arguments
@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@cache(max_size=50)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

---

## Stacking Decorators

```python
from functools import wraps

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # <b><i>Hello, Alice!</i></b>

# Order matters! Equivalent to:
# greet = bold(italic(greet))
```

---

## Class Decorators

### Decorating Classes

```python
def add_greeting(cls):
    """Add a greet method to a class."""
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    cls.greet = greet
    return cls

@add_greeting
class Person:
    def __init__(self, name: str):
        self.name = name

p = Person("Alice")
print(p.greet())  # "Hello, I'm Alice"
```

### Singleton Pattern

```python
def singleton(cls):
    """Make a class a singleton."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Connecting to database...")

db1 = Database()  # "Connecting to database..."
db2 = Database()  # (no output - same instance)
print(db1 is db2)  # True
```

---

## Built-in Decorators

### @property, @classmethod, @staticmethod

```python
class Circle:
    def __init__(self, radius: float):
        self._radius = radius
    
    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @classmethod
    def from_diameter(cls, diameter: float) -> "Circle":
        return cls(diameter / 2)
    
    @staticmethod
    def area_formula() -> str:
        return "π × r²"
```

### @functools.lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Instant! (cached)
print(fibonacci.cache_info())  # CacheInfo(hits=98, misses=101, ...)
```

### @dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(3, 4)
print(p)  # Point(x=3, y=4)
```

---

## Method Decorators

```python
from functools import wraps

def validate_positive(func):
    """Validate that first argument is positive."""
    @wraps(func)
    def wrapper(self, value, *args, **kwargs):
        if value <= 0:
            raise ValueError(f"{func.__name__} requires positive value")
        return func(self, value, *args, **kwargs)
    return wrapper

class BankAccount:
    def __init__(self, balance: float = 0):
        self.balance = balance
    
    @validate_positive
    def deposit(self, amount: float) -> float:
        self.balance += amount
        return self.balance
    
    @validate_positive
    def withdraw(self, amount: float) -> float:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance

account = BankAccount(100)
account.deposit(50)    # OK
# account.deposit(-10)  # ValueError!
```

---

## Hands-on Exercise

### Your Task

Create useful decorators:

```python
# 1. @memoize - cache function results
# 2. @debug - print function signature and return value
# 3. @validate_types - check argument types at runtime

# Example usage:
@memoize
def expensive_computation(n):
    # ... complex calculation
    return result

@debug
def add(a, b):
    return a + b

@validate_types
def greet(name: str, times: int) -> str:
    return (name + " ") * times
```

<details>
<summary>✅ Solution</summary>

```python
from functools import wraps
import inspect

def memoize(func):
    """Cache function results."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper

def debug(func):
    """Print debug info for function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"→ {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"← {func.__name__} returned {result!r}")
        return result
    return wrapper

def validate_types(func):
    """Validate argument types at runtime using annotations."""
    sig = inspect.signature(func)
    hints = func.__annotations__
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        for name, value in bound.arguments.items():
            if name in hints:
                expected = hints[name]
                if not isinstance(value, expected):
                    raise TypeError(
                        f"Argument '{name}' must be {expected.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        result = func(*args, **kwargs)
        
        if "return" in hints:
            expected = hints["return"]
            if not isinstance(result, expected):
                raise TypeError(
                    f"Return value must be {expected.__name__}, "
                    f"got {type(result).__name__}"
                )
        
        return result
    return wrapper

# Test
@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # Fast with memoization

@debug
def add(a, b):
    return a + b

add(3, 5)
# → add(3, 5)
# ← add returned 8

@validate_types
def greet(name: str, times: int) -> str:
    return (name + " ") * times

print(greet("Hello", 3))  # "Hello Hello Hello "
# greet(123, 3)  # TypeError!
```
</details>

---

## Summary

✅ **Decorators** modify function/class behavior
✅ Use **@syntax** for cleaner code
✅ Always use **@wraps** to preserve metadata
✅ **Parameterized decorators** need an extra wrapper layer
✅ Stack decorators: order matters (bottom to top)
✅ Built-in: `@property`, `@classmethod`, `@staticmethod`, `@lru_cache`

**Next:** [Dataclasses](./06-dataclasses.md)

---

## Further Reading

- [Decorators](https://docs.python.org/3/glossary.html#term-decorator)
- [functools Module](https://docs.python.org/3/library/functools.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/functools.html
-->
