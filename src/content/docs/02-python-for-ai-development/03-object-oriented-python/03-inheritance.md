---
title: "Inheritance"
---

# Inheritance

## Introduction

Inheritance allows classes to inherit attributes and methods from parent classes, enabling code reuse and polymorphism. Python supports both single and multiple inheritance.

### What We'll Cover

- Single inheritance
- Method overriding
- The super() function
- Multiple inheritance and MRO
- Abstract base classes

### Prerequisites

- Classes and objects
- Methods

---

## Single Inheritance

### Basic Inheritance

```python
class Animal:
    """Base class for animals."""
    
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        return "Some sound"
    
    def describe(self) -> str:
        return f"{self.name} is an animal"

class Dog(Animal):
    """Dog inherits from Animal."""
    
    def speak(self) -> str:
        return f"{self.name} says woof!"

class Cat(Animal):
    """Cat inherits from Animal."""
    
    def speak(self) -> str:
        return f"{self.name} says meow!"

# Create instances
dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())     # "Buddy says woof!"
print(cat.speak())     # "Whiskers says meow!"
print(dog.describe())  # "Buddy is an animal" (inherited)
```

### isinstance() and issubclass()

```python
dog = Dog("Buddy")

# isinstance checks object type
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(isinstance(dog, Cat))     # False

# issubclass checks class hierarchy
print(issubclass(Dog, Animal))  # True
print(issubclass(Cat, Animal))  # True
print(issubclass(Dog, Cat))     # False
```

---

## Method Overriding

### Overriding Parent Methods

```python
class Vehicle:
    def __init__(self, brand: str, model: str):
        self.brand = brand
        self.model = model
    
    def start(self) -> str:
        return "Starting vehicle..."
    
    def info(self) -> str:
        return f"{self.brand} {self.model}"

class ElectricCar(Vehicle):
    def __init__(self, brand: str, model: str, battery_capacity: int):
        self.brand = brand
        self.model = model
        self.battery_capacity = battery_capacity
    
    # Override start method
    def start(self) -> str:
        return "Powering up electric motors..."
    
    # Override info to add battery info
    def info(self) -> str:
        return f"{self.brand} {self.model} ({self.battery_capacity}kWh)"

ev = ElectricCar("Tesla", "Model 3", 75)
print(ev.start())  # "Powering up electric motors..."
print(ev.info())   # "Tesla Model 3 (75kWh)"
```

---

## The super() Function

### Calling Parent Methods

```python
class Vehicle:
    def __init__(self, brand: str, model: str):
        self.brand = brand
        self.model = model
    
    def info(self) -> str:
        return f"{self.brand} {self.model}"

class ElectricCar(Vehicle):
    def __init__(self, brand: str, model: str, battery_capacity: int):
        super().__init__(brand, model)  # Call parent __init__
        self.battery_capacity = battery_capacity
    
    def info(self) -> str:
        base_info = super().info()  # Call parent method
        return f"{base_info} ({self.battery_capacity}kWh)"

ev = ElectricCar("Tesla", "Model 3", 75)
print(ev.info())  # "Tesla Model 3 (75kWh)"
```

### Why Use super()?

```python
# Without super - fragile to refactoring
class Child(Parent):
    def __init__(self, x, y):
        Parent.__init__(self, x)  # Hardcoded parent name
        self.y = y

# With super - recommended
class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)  # Works with multiple inheritance
        self.y = y
```

---

## Multiple Inheritance

### Basic Multiple Inheritance

```python
class Flyable:
    def fly(self) -> str:
        return "Flying..."

class Swimmable:
    def swim(self) -> str:
        return "Swimming..."

class Duck(Flyable, Swimmable):
    def __init__(self, name: str):
        self.name = name
    
    def quack(self) -> str:
        return f"{self.name} says quack!"

duck = Duck("Donald")
print(duck.fly())    # "Flying..."
print(duck.swim())   # "Swimming..."
print(duck.quack())  # "Donald says quack!"
```

### Method Resolution Order (MRO)

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # "B" - follows MRO

# View MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Or use mro() method
print(D.mro())
```

### The Diamond Problem

```python
class A:
    def __init__(self):
        print("A init")
        self.value = "A"

class B(A):
    def __init__(self):
        print("B init")
        super().__init__()
        self.value = "B"

class C(A):
    def __init__(self):
        print("C init")
        super().__init__()
        self.value = "C"

class D(B, C):
    def __init__(self):
        print("D init")
        super().__init__()

d = D()
# Output:
# D init
# B init
# C init
# A init

# super() follows MRO, so A.__init__ is only called once
```

---

## Mixins

```python
class JSONMixin:
    """Mixin to add JSON serialization."""
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_str: str):
        import json
        data = json.loads(json_str)
        return cls(**data)

class TimestampMixin:
    """Mixin to add timestamps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()

class User(JSONMixin, TimestampMixin):
    def __init__(self, name: str, email: str):
        super().__init__()
        self.name = name
        self.email = email

user = User("Alice", "alice@example.com")
print(user.to_json())      # {"name": "Alice", "email": "...", "created_at": "..."}
print(user.created_at)     # 2024-01-15 10:30:45.123456
```

---

## Abstract Base Classes

### Creating Abstract Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""
    
    @abstractmethod
    def area(self) -> float:
        """Calculate area. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter. Must be implemented by subclasses."""
        pass
    
    def describe(self) -> str:
        """Concrete method - can use abstract methods."""
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self) -> float:
        import math
        return 2 * math.pi * self.radius

# Cannot instantiate abstract class
# shape = Shape()  # TypeError!

# Can instantiate concrete subclasses
rect = Rectangle(4, 5)
circle = Circle(3)

print(rect.describe())    # "Area: 20, Perimeter: 18"
print(circle.describe())  # "Area: 28.27..., Perimeter: 18.85..."
```

### Abstract Properties

```python
from abc import ABC, abstractmethod

class DatabaseConnection(ABC):
    @property
    @abstractmethod
    def connection_string(self) -> str:
        """Must provide connection string."""
        pass
    
    @abstractmethod
    def connect(self) -> None:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass

class PostgresConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self._connected = False
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.host}:{self.port}/{self.database}"
    
    def connect(self) -> None:
        print(f"Connecting to {self.connection_string}")
        self._connected = True
    
    def disconnect(self) -> None:
        print("Disconnecting...")
        self._connected = False
```

---

## Composition vs Inheritance

```python
# Inheritance: "is-a" relationship
class Car(Vehicle):
    pass  # Car IS-A Vehicle

# Composition: "has-a" relationship
class Car:
    def __init__(self):
        self.engine = Engine()     # Car HAS-AN Engine
        self.wheels = [Wheel() for _ in range(4)]  # Car HAS Wheels

# Prefer composition when:
# - Components can exist independently
# - You need flexibility to swap components
# - "is-a" relationship doesn't make sense

class Logger:
    def log(self, message: str) -> None:
        print(f"LOG: {message}")

class Database:
    def __init__(self, logger: Logger):
        self.logger = logger  # Composition via dependency injection
    
    def query(self, sql: str):
        self.logger.log(f"Executing: {sql}")
        # ... execute query
```

---

## Hands-on Exercise

### Your Task

Create a shape hierarchy:

```python
# Requirements:
# 1. Abstract Shape class with area() and perimeter()
# 2. Rectangle(width, height) class
# 3. Square(side) class that inherits from Rectangle
# 4. Triangle(a, b, c) class for sides

# Example usage:
shapes = [
    Rectangle(4, 5),
    Square(3),
    Triangle(3, 4, 5)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: area={shape.area()}, perimeter={shape.perimeter()}")
```

<details>
<summary>✅ Solution</summary>

```python
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Square(Rectangle):
    def __init__(self, side: float):
        super().__init__(side, side)  # Square is a special Rectangle

class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
    
    def perimeter(self) -> float:
        return self.a + self.b + self.c
    
    def area(self) -> float:
        # Heron's formula
        s = self.perimeter() / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

# Test
shapes = [
    Rectangle(4, 5),
    Square(3),
    Triangle(3, 4, 5)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: area={shape.area()}, perimeter={shape.perimeter()}")
# Rectangle: area=20, perimeter=18
# Square: area=9, perimeter=12
# Triangle: area=6.0, perimeter=12
```
</details>

---

## Summary

✅ **Inheritance** enables code reuse via parent-child relationships
✅ Use **super()** to call parent class methods
✅ **MRO** determines method lookup order in multiple inheritance
✅ **Mixins** add functionality through composition-like inheritance
✅ **Abstract classes** define interfaces with required methods
✅ **Prefer composition** for "has-a" relationships

**Next:** [Magic Methods](./04-magic-methods.md)

---

## Further Reading

- [Inheritance](https://docs.python.org/3/tutorial/classes.html#inheritance)
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/classes.html
-->
