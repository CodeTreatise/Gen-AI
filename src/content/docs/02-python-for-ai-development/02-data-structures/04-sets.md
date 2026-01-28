---
title: "Sets"
---

# Sets

## Introduction

Sets are unordered collections of unique elements. They're ideal for membership testing, removing duplicates, and mathematical set operations like union and intersection.

### What We'll Cover

- Creating and using sets
- Set operations
- Set comprehensions
- Frozen sets
- When to use sets

### Prerequisites

- Lists and dictionaries
- Python fundamentals

---

## Creating Sets

### Basic Creation

```python
# Curly braces (but NOT empty!)
fruits = {"apple", "banana", "cherry"}

# Empty set (use set(), not {})
empty = set()       # ✅ Empty set
empty_dict = {}     # ❌ This is an empty dict!

# From iterable
chars = set("hello")  # {"h", "e", "l", "o"}
nums = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Duplicates are automatically removed
numbers = {1, 2, 2, 3, 3, 3}
print(numbers)  # {1, 2, 3}
```

### Set Requirements

```python
# Elements must be hashable (immutable)
valid = {1, "hello", (1, 2), 3.14}

# ❌ Mutable objects can't be in sets
# invalid = {[1, 2]}  # TypeError: unhashable type: 'list'
# invalid = {{"a": 1}}  # TypeError: unhashable type: 'dict'

# Convert list to tuple for set membership
data = [(1, 2), (3, 4), (1, 2)]
unique = set(data)  # {(1, 2), (3, 4)}
```

---

## Basic Set Operations

### Adding and Removing

```python
fruits = {"apple", "banana"}

# Add single element
fruits.add("cherry")
print(fruits)  # {"apple", "banana", "cherry"}

# Add multiple elements
fruits.update(["date", "elderberry"])
print(fruits)  # {"apple", "banana", "cherry", "date", "elderberry"}

# Remove (raises KeyError if missing)
fruits.remove("apple")

# Discard (no error if missing)
fruits.discard("not_there")  # No error

# Pop (remove and return arbitrary element)
item = fruits.pop()
print(f"Removed: {item}")

# Clear all
fruits.clear()
```

### Membership Testing

```python
fruits = {"apple", "banana", "cherry"}

# O(1) membership test (very fast!)
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# Compare to list (O(n))
fruit_list = ["apple", "banana", "cherry"]
print("apple" in fruit_list)  # True, but slower for large lists
```

---

## Mathematical Set Operations

### Union

```python
a = {1, 2, 3}
b = {3, 4, 5}

# Elements in either set
print(a | b)           # {1, 2, 3, 4, 5}
print(a.union(b))      # {1, 2, 3, 4, 5}

# Multiple sets
c = {5, 6, 7}
print(a | b | c)       # {1, 2, 3, 4, 5, 6, 7}
print(a.union(b, c))   # {1, 2, 3, 4, 5, 6, 7}
```

### Intersection

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Elements in both sets
print(a & b)                # {3, 4}
print(a.intersection(b))    # {3, 4}

# Multiple sets
c = {4, 5, 6, 7}
print(a & b & c)            # {4}
```

### Difference

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Elements in a but not in b
print(a - b)             # {1, 2}
print(a.difference(b))   # {1, 2}

# Elements in b but not in a
print(b - a)             # {5, 6}
```

### Symmetric Difference

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Elements in either but not both (XOR)
print(a ^ b)                        # {1, 2, 5, 6}
print(a.symmetric_difference(b))    # {1, 2, 5, 6}
```

### Visual Summary

```
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b  = {1, 2, 3, 4, 5, 6}   (union)
a & b  = {3, 4}               (intersection)
a - b  = {1, 2}               (difference)
a ^ b  = {1, 2, 5, 6}         (symmetric difference)
```

---

## Set Comparisons

```python
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}
c = {1, 2, 3}

# Subset
print(a <= b)           # True (a is subset of b)
print(a.issubset(b))    # True

# Proper subset (not equal)
print(a < b)            # True
print(a < c)            # False (equal, not proper subset)

# Superset
print(b >= a)           # True (b is superset of a)
print(b.issuperset(a))  # True

# Disjoint (no common elements)
d = {10, 20, 30}
print(a.isdisjoint(d))  # True
print(a.isdisjoint(b))  # False
```

---

## In-Place Operations

```python
a = {1, 2, 3}
b = {3, 4, 5}

# Update in place
a |= b          # Same as a.update(b)
print(a)        # {1, 2, 3, 4, 5}

a = {1, 2, 3}
a &= {2, 3, 4}  # Same as a.intersection_update({2, 3, 4})
print(a)        # {2, 3}

a = {1, 2, 3, 4}
a -= {3, 4, 5}  # Same as a.difference_update({3, 4, 5})
print(a)        # {1, 2}

a = {1, 2, 3}
a ^= {2, 3, 4}  # Same as a.symmetric_difference_update({2, 3, 4})
print(a)        # {1, 4}
```

---

## Set Comprehensions

```python
# Basic syntax: {expression for item in iterable}

# Squares
squares = {x ** 2 for x in range(10)}
print(squares)  # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# With condition
evens = {x for x in range(20) if x % 2 == 0}
print(evens)  # {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}

# From string
vowels = {c for c in "hello world" if c in "aeiou"}
print(vowels)  # {"e", "o"}
```

---

## Frozen Sets

```python
# Immutable sets - can be used as dict keys or in other sets
fs = frozenset([1, 2, 3])

# ❌ Cannot modify
# fs.add(4)  # AttributeError

# ✅ Can use in sets or as dict keys
set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}
print(set_of_sets)

mapping = {
    frozenset(["a", "b"]): "first",
    frozenset(["c", "d"]): "second"
}
print(mapping[frozenset(["a", "b"])])  # "first"

# All operations work (return new frozenset)
fs1 = frozenset([1, 2, 3])
fs2 = frozenset([3, 4, 5])
print(fs1 | fs2)  # frozenset({1, 2, 3, 4, 5})
```

---

## Practical Use Cases

### Remove Duplicates

```python
# From list
items = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(items))
print(unique)  # [1, 2, 3, 4] (order may vary)

# Preserve order (Python 3.7+)
unique_ordered = list(dict.fromkeys(items))
print(unique_ordered)  # [1, 2, 3, 4] (order preserved)
```

### Fast Membership Testing

```python
# Convert to set for repeated lookups
allowed_users = set(["alice", "bob", "charlie"])

def is_allowed(user):
    return user in allowed_users  # O(1)
```

### Find Common Elements

```python
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

common = set(list1) & set(list2)
print(common)  # {4, 5}
```

### Tag/Category Management

```python
post_tags = {"python", "programming", "tutorial"}
user_interests = {"python", "data-science", "machine-learning"}

# Matching tags
matching = post_tags & user_interests
print(f"Matching interests: {matching}")  # {"python"}

# All tags
all_tags = post_tags | user_interests
print(f"All tags: {all_tags}")
```

### Checking for Duplicates

```python
def has_duplicates(items):
    return len(items) != len(set(items))

print(has_duplicates([1, 2, 3]))     # False
print(has_duplicates([1, 2, 2, 3]))  # True
```

---

## Performance Comparison

| Operation | Set | List |
|-----------|-----|------|
| Membership (`in`) | O(1) | O(n) |
| Add element | O(1) | O(1) amortized |
| Remove element | O(1) | O(n) |
| Iteration | O(n) | O(n) |
| Union | O(n+m) | — |
| Intersection | O(min(n,m)) | — |

```python
import timeit

# Membership test comparison
large_list = list(range(100000))
large_set = set(large_list)

# List lookup
list_time = timeit.timeit("99999 in large_list", globals=globals(), number=1000)

# Set lookup  
set_time = timeit.timeit("99999 in large_set", globals=globals(), number=1000)

print(f"List: {list_time:.4f}s, Set: {set_time:.4f}s")
# Set is ~1000x faster for large collections
```

---

## Hands-on Exercise

### Your Task

Implement functions to analyze user permissions:

```python
# Given user roles and role permissions, determine:
# 1. All permissions a user has
# 2. Permissions common to all given users
# 3. Users who have a specific permission

roles = {
    "admin": {"read", "write", "delete", "manage_users"},
    "editor": {"read", "write"},
    "viewer": {"read"}
}

users = {
    "alice": {"admin", "editor"},
    "bob": {"editor"},
    "charlie": {"viewer", "editor"}
}
```

<details>
<summary>✅ Solution</summary>

```python
def get_user_permissions(user: str, users: dict, roles: dict) -> set:
    """Get all permissions for a user."""
    user_roles = users.get(user, set())
    permissions = set()
    for role in user_roles:
        permissions |= roles.get(role, set())
    return permissions

def common_permissions(user_list: list, users: dict, roles: dict) -> set:
    """Get permissions common to all users in list."""
    if not user_list:
        return set()
    
    result = get_user_permissions(user_list[0], users, roles)
    for user in user_list[1:]:
        result &= get_user_permissions(user, users, roles)
    return result

def users_with_permission(permission: str, users: dict, roles: dict) -> set:
    """Find all users with a specific permission."""
    result = set()
    for user in users:
        if permission in get_user_permissions(user, users, roles):
            result.add(user)
    return result

# Test
roles = {
    "admin": {"read", "write", "delete", "manage_users"},
    "editor": {"read", "write"},
    "viewer": {"read"}
}

users = {
    "alice": {"admin", "editor"},
    "bob": {"editor"},
    "charlie": {"viewer", "editor"}
}

print(get_user_permissions("alice", users, roles))
# {"read", "write", "delete", "manage_users"}

print(common_permissions(["alice", "bob", "charlie"], users, roles))
# {"read", "write"}

print(users_with_permission("delete", users, roles))
# {"alice"}
```
</details>

---

## Summary

✅ **Sets** store unique, unordered elements
✅ Use `set()` for empty set (not `{}`)
✅ **Union** (`|`), **intersection** (`&`), **difference** (`-`), **symmetric difference** (`^`)
✅ O(1) membership testing—much faster than lists
✅ **Frozen sets** are immutable, can be dict keys
✅ Use sets to remove duplicates and find commonalities

**Next:** [Comprehensions](./05-comprehensions.md)

---

## Further Reading

- [Set Types](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)
- [Set Operations](https://docs.python.org/3/tutorial/datastructures.html#sets)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/stdtypes.html#set
-->
