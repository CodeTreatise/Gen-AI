---
title: "Classes & Objects"
---

# Classes & Objects

## Introduction

Classes are blueprints for creating objects. They define the structure (attributes) and behavior (methods) that objects will have. Understanding classes is fundamental to Python programming.

### What We'll Cover

- Defining classes
- The `__init__` constructor
- Instance vs class attributes
- Creating and using objects
- The `self` parameter

### Prerequisites

- Functions
- Data structures

---

## Defining a Class

### Basic Syntax

```python
class Dog:
    """A simple Dog class."""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor
    def __init__(self, name: str, age: int):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def bark(self) -> str:
        return f"{self.name} says woof!"
```

### Creating Objects (Instances)

```python
# Create instances
dog1 = Dog("Buddy", 5)
dog2 = Dog("Max", 3)

# Access attributes
print(dog1.name)      # "Buddy"
print(dog2.age)       # 3
print(dog1.species)   # "Canis familiaris"

# Call methods
print(dog1.bark())    # "Buddy says woof!"
```

---

## The `__init__` Method

### Constructor Purpose

```python
class User:
    def __init__(self, name: str, email: str, age: int = 0):
        """Initialize a new User.
        
        Args:
            name: User's name
            email: User's email
            age: User's age (default 0)
        """
        self.name = name
        self.email = email
        self.age = age
        self.created_at = datetime.now()  # Computed attribute

# With positional args
user1 = User("Alice", "alice@example.com", 30)

# With keyword args
user2 = User(name="Bob", email="bob@example.com")
```

### Validation in `__init__`

```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        if balance < 0:
            raise ValueError("Initial balance cannot be negative")
        
        self.owner = owner
        self._balance = balance  # Convention: _prefix for "private"
    
    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
    
    def get_balance(self) -> float:
        return self._balance

account = BankAccount("Alice", 100)
account.deposit(50)
print(account.get_balance())  # 150
```

---

## The `self` Parameter

### Understanding self

```python
class Counter:
    def __init__(self):
        self.count = 0  # self refers to the instance
    
    def increment(self):
        self.count += 1  # Access instance attribute via self
    
    def get_count(self):
        return self.count

c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(c1.get_count())  # 2
print(c2.get_count())  # 1
```

### self is Just Convention

```python
# "self" is convention, not keyword
class Example:
    def greet(this):  # Works, but don't do this!
        return f"Hello from {this}"

# Stick with self for clarity
```

---

## Instance Attributes vs Class Attributes

### Instance Attributes

```python
class Person:
    def __init__(self, name: str):
        self.name = name  # Instance attribute - unique per instance

p1 = Person("Alice")
p2 = Person("Bob")

print(p1.name)  # "Alice"
print(p2.name)  # "Bob"

p1.name = "Alicia"  # Only affects p1
print(p1.name)  # "Alicia"
print(p2.name)  # "Bob" (unchanged)
```

### Class Attributes

```python
class Employee:
    # Class attribute - shared by all instances
    company = "TechCorp"
    employee_count = 0
    
    def __init__(self, name: str):
        self.name = name
        Employee.employee_count += 1  # Modify via class name

e1 = Employee("Alice")
e2 = Employee("Bob")

print(e1.company)  # "TechCorp"
print(e2.company)  # "TechCorp"
print(Employee.employee_count)  # 2
```

### Shadowing Class Attributes

```python
class Config:
    debug = False  # Class attribute

c1 = Config()
c2 = Config()

# Reading - both see class attribute
print(c1.debug)  # False
print(c2.debug)  # False

# Assigning creates instance attribute (shadows class attr)
c1.debug = True

print(c1.debug)  # True (instance attribute)
print(c2.debug)  # False (still class attribute)
print(Config.debug)  # False (class attribute unchanged)
```

### Mutable Class Attributes

```python
# ⚠️ Careful with mutable class attributes!
class WrongWay:
    items = []  # Shared by ALL instances!
    
    def add(self, item):
        self.items.append(item)

w1 = WrongWay()
w2 = WrongWay()

w1.add("a")
w2.add("b")

print(w1.items)  # ["a", "b"] - Both share same list!
print(w2.items)  # ["a", "b"]

# ✅ Correct: Initialize in __init__
class RightWay:
    def __init__(self):
        self.items = []  # Each instance gets its own list
    
    def add(self, item):
        self.items.append(item)
```

---

## Class vs Instance Namespaces

```python
class Demo:
    x = 10  # Class namespace
    
    def __init__(self):
        self.y = 20  # Instance namespace

d = Demo()

# Lookup order: instance → class → parent classes
print(d.y)  # 20 (found in instance)
print(d.x)  # 10 (found in class)

# Direct access
print(Demo.x)           # 10
print(d.__class__.x)    # 10

# Instance namespace
print(d.__dict__)       # {"y": 20}

# Class namespace
print(Demo.__dict__)    # {"x": 10, "__init__": ..., ...}
```

---

## Object Introspection

```python
class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, {self.name}"

user = User("Alice", 30)

# Type checking
print(type(user))                    # <class '__main__.User'>
print(isinstance(user, User))        # True

# Attribute inspection
print(hasattr(user, "name"))         # True
print(hasattr(user, "email"))        # False

print(getattr(user, "name"))         # "Alice"
print(getattr(user, "email", "N/A")) # "N/A" (default)

setattr(user, "email", "alice@example.com")
print(user.email)                    # "alice@example.com"

# All attributes
print(dir(user))  # Lists all attributes and methods
print(vars(user)) # {"name": "Alice", "age": 30, "email": "..."}
```

---

## Private Attributes (Convention)

```python
class Account:
    def __init__(self, balance: float):
        self._balance = balance      # "Protected" - convention only
        self.__secret = "hidden"     # "Private" - name mangling
    
    def get_balance(self):
        return self._balance

acc = Account(100)

# Single underscore - accessible but "don't touch"
print(acc._balance)  # 100 (works, but shouldn't access directly)

# Double underscore - name mangled
# print(acc.__secret)  # AttributeError
print(acc._Account__secret)  # "hidden" (mangled name)
```

---

## Hands-on Exercise

### Your Task

Create a `Rectangle` class:

```python
# Requirements:
# 1. Constructor takes width and height
# 2. Properties: area, perimeter
# 3. Method: is_square() returns True if width == height
# 4. Class attribute for counting rectangles created

# Example usage:
r1 = Rectangle(4, 5)
print(r1.area)        # 20
print(r1.perimeter)   # 18
print(r1.is_square()) # False

r2 = Rectangle(3, 3)
print(r2.is_square()) # True
print(Rectangle.count) # 2
```

<details>
<summary>✅ Solution</summary>

```python
class Rectangle:
    """A rectangle with width and height."""
    
    count = 0  # Class attribute
    
    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        
        self.width = width
        self.height = height
        Rectangle.count += 1
    
    @property
    def area(self) -> float:
        """Calculate area."""
        return self.width * self.height
    
    @property
    def perimeter(self) -> float:
        """Calculate perimeter."""
        return 2 * (self.width + self.height)
    
    def is_square(self) -> bool:
        """Check if rectangle is a square."""
        return self.width == self.height
    
    def __repr__(self) -> str:
        return f"Rectangle({self.width}, {self.height})"

# Test
r1 = Rectangle(4, 5)
print(r1.area)        # 20
print(r1.perimeter)   # 18
print(r1.is_square()) # False

r2 = Rectangle(3, 3)
print(r2.is_square()) # True
print(Rectangle.count) # 2
```
</details>

---

## Summary

✅ **Classes** define blueprints for objects
✅ **`__init__`** initializes instance attributes
✅ **`self`** refers to the current instance
✅ **Instance attributes** are unique per object
✅ **Class attributes** are shared by all instances
✅ Use `_prefix` for "protected" attributes (convention)
✅ Use `__prefix` for name mangling (rarely needed)

**Next:** [Methods](./02-methods.md)

---

## Further Reading

- [Classes Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Class Definitions](https://docs.python.org/3/reference/compound_stmts.html#class-definitions)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/classes.html
-->
