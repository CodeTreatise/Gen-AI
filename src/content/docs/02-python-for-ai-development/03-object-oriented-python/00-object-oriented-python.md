---
title: "Object-Oriented Python"
---

# Object-Oriented Python

## Overview

Object-Oriented Programming (OOP) is a paradigm for organizing code around objects—data structures that combine data and behavior. Python's OOP features are powerful yet accessible, making it easy to write maintainable, reusable code.

This lesson covers Python's OOP capabilities from basic classes to advanced patterns like decorators and dataclasses.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-classes-objects.md) | Classes & Objects | `__init__`, attributes, instances |
| [02](./02-methods.md) | Methods | Instance, class, static methods, properties |
| [03](./03-inheritance.md) | Inheritance | Single/multiple inheritance, super(), ABC |
| [04](./04-magic-methods.md) | Magic Methods | `__str__`, `__repr__`, operators, protocols |
| [05](./05-decorators.md) | Decorators | Function/class decorators, custom decorators |
| [06](./06-dataclasses.md) | Dataclasses | @dataclass, fields, frozen classes |

---

## Why OOP?

| Benefit | Description |
|---------|-------------|
| **Encapsulation** | Bundle data and methods together |
| **Reusability** | Inherit and extend existing classes |
| **Maintainability** | Organized, modular code |
| **Abstraction** | Hide complexity behind simple interfaces |

---

## Quick Reference

```python
# Basic class
class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

# Create instance
user = User("Alice", 30)
print(user.greet())  # "Hello, I'm Alice"

# Dataclass (modern approach)
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(10, 20)
print(p)  # Point(x=10, y=20)
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals (functions, data types)
- Data structures (lists, dicts)

---

## Start Learning

Begin with [Classes & Objects](./01-classes-objects.md) to understand the foundation of OOP in Python.

---

## Further Reading

- [Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Data Model](https://docs.python.org/3/reference/datamodel.html)
- [dataclasses Module](https://docs.python.org/3/library/dataclasses.html)
