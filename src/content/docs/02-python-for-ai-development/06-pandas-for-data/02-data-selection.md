---
title: "Data Selection"
---

# Data Selection

## Introduction

Selecting data efficiently is fundamental to data analysis. Pandas provides multiple ways to access rows and columns—understanding when to use each method is key to writing clean, performant code.

### What We'll Cover

- Column selection
- Row selection with `.loc` and `.iloc`
- Boolean indexing
- The query method
- Selecting multiple columns

### Prerequisites

- DataFrames and Series basics

---

## Column Selection

### Single Column

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# Bracket notation (recommended)
ages = df['age']
print(type(ages))  # <class 'pandas.core.series.Series'>
print(ages)
# 0    25
# 1    30
# 2    35

# Dot notation (convenient but limited)
ages = df.age
print(ages)

# ⚠️ Dot notation fails when:
# - Column name has spaces: df['column name']
# - Column name matches method: df['count'] not df.count
# - Column name is a number: df[0]
```

### Multiple Columns

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'city': ['NYC', 'LA'],
    'salary': [50000, 60000]
})

# Select multiple columns (returns DataFrame)
subset = df[['name', 'age']]
print(type(subset))  # DataFrame
print(subset)
#     name  age
# 0  Alice   25
# 1    Bob   30

# Column list from variable
cols = ['name', 'salary']
subset = df[cols]
```

---

## Row Selection with .loc

### Label-Based Selection

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}, index=['a', 'b', 'c'])

# Single row by label
row = df.loc['a']
print(row)
# name    Alice
# age        25
# city      NYC

# Multiple rows by label
rows = df.loc[['a', 'c']]
print(rows)

# Range of labels (inclusive!)
rows = df.loc['a':'b']
print(rows)
```

### Row and Column Selection

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}, index=['a', 'b', 'c'])

# Single cell
value = df.loc['a', 'age']
print(value)  # 25

# Row subset, specific columns
subset = df.loc['a', ['name', 'city']]
print(subset)

# Multiple rows, multiple columns
subset = df.loc[['a', 'c'], ['name', 'age']]
print(subset)
#       name  age
# a    Alice   25
# c  Charlie   35

# All rows, specific columns
subset = df.loc[:, ['name', 'age']]
```

---

## Row Selection with .iloc

### Integer Position-Based Selection

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# Single row by position
row = df.iloc[0]  # First row
print(row)

# Multiple rows by position
rows = df.iloc[[0, 2]]  # First and third
print(rows)

# Range (exclusive end, like Python)
rows = df.iloc[0:2]  # First two rows
print(rows)

# Negative indexing
last_row = df.iloc[-1]
print(last_row)
```

### Row and Column by Position

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# Single cell
value = df.iloc[0, 1]  # Row 0, Column 1
print(value)  # 25

# Slice rows and columns
subset = df.iloc[0:2, 0:2]
print(subset)
#     name  age
# 0  Alice   25
# 1    Bob   30

# Specific positions
subset = df.iloc[[0, 2], [0, 2]]  # Rows 0,2 and Cols 0,2
print(subset)
```

---

## .loc vs .iloc Summary

| Feature | `.loc` | `.iloc` |
|---------|--------|---------|
| Selection | By label | By position |
| Slicing | Inclusive end | Exclusive end |
| Use when | Know label names | Know positions |

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
}, index=['x', 'y', 'z'])

# .loc uses labels
print(df.loc['x', 'a'])      # 1

# .iloc uses positions
print(df.iloc[0, 0])         # 1

# Slicing difference
print(df.loc['x':'y', 'a'])  # Includes 'y'!
print(df.iloc[0:1, 0])       # Excludes position 1
```

---

## Boolean Indexing

### Filtering Rows

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'NYC', 'Chicago']
})

# Boolean mask
mask = df['age'] > 28
print(mask)
# 0    False
# 1     True
# 2     True
# 3    False

# Apply mask
filtered = df[mask]
print(filtered)
#       name  age city
# 1      Bob   30   LA
# 2  Charlie   35  NYC

# One-liner
filtered = df[df['age'] > 28]
```

### Multiple Conditions

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'NYC', 'Chicago']
})

# AND: use & (with parentheses!)
filtered = df[(df['age'] > 25) & (df['city'] == 'NYC')]
print(filtered)
#       name  age city
# 2  Charlie   35  NYC

# OR: use |
filtered = df[(df['city'] == 'NYC') | (df['city'] == 'LA')]
print(filtered)

# NOT: use ~
filtered = df[~(df['city'] == 'NYC')]
print(filtered)
```

### String Conditions

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@gmail.com', 'bob@yahoo.com', 'charlie@gmail.com']
})

# String contains
gmail_users = df[df['email'].str.contains('gmail')]
print(gmail_users)

# String starts/ends with
a_names = df[df['name'].str.startswith('A')]
print(a_names)

# Case-insensitive
filtered = df[df['name'].str.lower() == 'alice']
```

### isin() Method

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC']
})

# Check if value is in list
cities = ['NYC', 'LA']
filtered = df[df['city'].isin(cities)]
print(filtered)
#     name city
# 0  Alice  NYC
# 1    Bob   LA
# 3  Diana  NYC

# Negate with ~
not_in = df[~df['city'].isin(cities)]
```

---

## Query Method

### String-Based Filtering

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

# Basic query
result = df.query('age > 28')
print(result)

# Multiple conditions
result = df.query('age > 25 and salary > 55000')
print(result)

# Using variables
min_age = 28
result = df.query('age > @min_age')
print(result)

# String columns
result = df.query('name == "Alice"')
```

### Query vs Boolean Indexing

```python
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Boolean indexing
result = df[(df['age'] > 25) & (df['salary'] > 55000)]

# Query (often more readable)
result = df.query('age > 25 and salary > 55000')

# Both produce same result
```

---

## Modifying Selected Data

### Setting Values

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'bonus': [0, 0, 0]
})

# Set single value
df.loc[0, 'bonus'] = 1000

# Set based on condition
df.loc[df['age'] > 28, 'bonus'] = 2000
print(df)
#       name  age  bonus
# 0    Alice   25   1000
# 1      Bob   30   2000
# 2  Charlie   35   2000

# Set entire column
df['bonus'] = 500
```

### Avoid Chained Indexing

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# ❌ Bad: Chained indexing (may not work)
df[df['a'] > 1]['b'] = 100  # SettingWithCopyWarning!

# ✅ Good: Use .loc
df.loc[df['a'] > 1, 'b'] = 100
```

---

## Hands-on Exercise

### Your Task

```python
# Given this sales data:
import pandas as pd

sales = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone'],
    'region': ['North', 'South', 'North', 'South', 'North'],
    'quantity': [10, 25, 15, 8, 30],
    'price': [1000, 500, 300, 1000, 500]
})

# 1. Select all Phone sales
# 2. Select sales where quantity > 15
# 3. Select North region sales with quantity > 10
# 4. Add 'revenue' column (quantity * price)
# 5. Set bonus=True for revenue > 10000
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd

sales = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone'],
    'region': ['North', 'South', 'North', 'South', 'North'],
    'quantity': [10, 25, 15, 8, 30],
    'price': [1000, 500, 300, 1000, 500]
})

# 1. Phone sales
phones = sales[sales['product'] == 'Phone']
print("Phone sales:")
print(phones)

# 2. Quantity > 15
high_qty = sales[sales['quantity'] > 15]
print("\nQuantity > 15:")
print(high_qty)

# 3. North region, quantity > 10
north_high = sales[(sales['region'] == 'North') & (sales['quantity'] > 10)]
print("\nNorth region, quantity > 10:")
print(north_high)

# 4. Add revenue
sales['revenue'] = sales['quantity'] * sales['price']
print("\nWith revenue:")
print(sales)

# 5. Set bonus for high revenue
sales['bonus'] = False
sales.loc[sales['revenue'] > 10000, 'bonus'] = True
print("\nWith bonus:")
print(sales)
```
</details>

---

## Summary

✅ Use **`df['col']`** for single column, **`df[['a', 'b']]`** for multiple
✅ **`.loc`** for label-based selection (inclusive slicing)
✅ **`.iloc`** for position-based selection (exclusive slicing)
✅ **Boolean indexing** for filtering: `df[df['col'] > 5]`
✅ Use **`&`**, **`|`**, **`~`** for combining conditions
✅ **`.query()`** for readable string-based filtering
✅ Always use **`.loc`** for assignment to avoid warnings

**Next:** [Data Manipulation](./03-data-manipulation.md)

---

## Further Reading

- [Indexing and Selecting](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [Query Method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/indexing.html
-->
