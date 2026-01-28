---
title: "Data Manipulation"
---

# Data Manipulation

## Introduction

Transforming data is a core part of any data pipeline. Pandas provides powerful tools for adding, removing, and modifying columns, sorting data, and applying custom transformations.

### What We'll Cover

- Adding and removing columns
- Renaming columns
- Sorting data
- Applying functions
- Value replacement

### Prerequisites

- DataFrames basics
- Data selection

---

## Adding Columns

### Direct Assignment

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'price': [100, 200, 150],
    'quantity': [2, 3, 1]
})

# Add new column
df['total'] = df['price'] * df['quantity']
print(df)
#       name  price  quantity  total
# 0    Alice    100         2    200
# 1      Bob    200         3    600
# 2  Charlie    150         1    150

# Add constant value
df['status'] = 'active'

# Add from Series
df['rank'] = pd.Series([1, 2, 3])
```

### Insert at Specific Position

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'c': [7, 8, 9]
})

# Insert column at position 1
df.insert(1, 'b', [4, 5, 6])
print(df)
#    a  b  c
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9
```

### Assign Method (Chaining)

```python
import pandas as pd

df = pd.DataFrame({
    'price': [100, 200, 150],
    'quantity': [2, 3, 1]
})

# Chain multiple operations
result = (df
    .assign(total=lambda x: x['price'] * x['quantity'])
    .assign(tax=lambda x: x['total'] * 0.1)
    .assign(grand_total=lambda x: x['total'] + x['tax'])
)
print(result)
```

---

## Removing Columns

### Drop Method

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
    'd': [10, 11, 12]
})

# Drop single column
df_new = df.drop('a', axis=1)
print(df_new)

# Drop multiple columns
df_new = df.drop(['a', 'b'], axis=1)

# Drop in place
df.drop('d', axis=1, inplace=True)

# Using columns parameter
df_new = df.drop(columns=['b', 'c'])
```

### Del Keyword

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Delete in place
del df['b']
print(df)
```

### Pop Method

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Remove and return column
col_b = df.pop('b')
print(col_b)  # Series
print(df)     # Only 'a' remains
```

---

## Renaming Columns

### Rename Method

```python
import pandas as pd

df = pd.DataFrame({
    'old_name': [1, 2, 3],
    'another_old': [4, 5, 6]
})

# Rename specific columns
df = df.rename(columns={
    'old_name': 'new_name',
    'another_old': 'better_name'
})
print(df)

# Using function
df = df.rename(columns=str.upper)
print(df.columns)  # ['NEW_NAME', 'BETTER_NAME']

# Lowercase
df = df.rename(columns=str.lower)
```

### Direct Assignment

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Replace all column names
df.columns = ['col1', 'col2']
print(df)
```

### Clean Column Names

```python
import pandas as pd

df = pd.DataFrame({
    'First Name': [1, 2],
    'Last Name': [3, 4],
    'EMAIL ADDRESS': [5, 6]
})

# Clean: lowercase, replace spaces
df.columns = (df.columns
    .str.lower()
    .str.replace(' ', '_')
)
print(df.columns)
# Index(['first_name', 'last_name', 'email_address'])
```

---

## Sorting

### Sort by Values

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob'],
    'age': [35, 25, 30],
    'salary': [70000, 50000, 60000]
})

# Sort by single column
df_sorted = df.sort_values('age')
print(df_sorted)
#       name  age  salary
# 1    Alice   25   50000
# 2      Bob   30   60000
# 0  Charlie   35   70000

# Descending order
df_sorted = df.sort_values('age', ascending=False)

# Sort by multiple columns
df_sorted = df.sort_values(['age', 'salary'], ascending=[True, False])
```

### Sort by Index

```python
import pandas as pd

df = pd.DataFrame({
    'value': [3, 1, 2]
}, index=['c', 'a', 'b'])

# Sort by index
df_sorted = df.sort_index()
print(df_sorted)
#    value
# a      1
# b      2
# c      3

# Descending
df_sorted = df.sort_index(ascending=False)
```

### Reset Index After Sort

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob'],
    'age': [35, 25, 30]
})

df_sorted = df.sort_values('age').reset_index(drop=True)
print(df_sorted)
#       name  age
# 0    Alice   25
# 1      Bob   30
# 2  Charlie   35
```

---

## Applying Functions

### Apply to Column

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['alice', 'bob', 'charlie'],
    'age': [25, 30, 35]
})

# Apply function to column
df['name'] = df['name'].apply(str.title)
print(df)
#       name  age
# 0    Alice   25
# 1      Bob   30
# 2  Charlie   35

# Custom function
def categorize_age(age):
    return 'young' if age < 30 else 'adult'

df['category'] = df['age'].apply(categorize_age)

# Lambda
df['age_group'] = df['age'].apply(lambda x: 'young' if x < 30 else 'adult')
```

### Apply to Row

```python
import pandas as pd

df = pd.DataFrame({
    'first': ['Alice', 'Bob'],
    'last': ['Smith', 'Jones'],
    'score1': [80, 90],
    'score2': [85, 88]
})

# Apply function to each row
df['full_name'] = df.apply(lambda row: f"{row['first']} {row['last']}", axis=1)

# Calculate row-wise
df['avg_score'] = df.apply(lambda row: (row['score1'] + row['score2']) / 2, axis=1)
print(df)
```

### Map Values

```python
import pandas as pd

df = pd.DataFrame({
    'grade': ['A', 'B', 'C', 'A', 'B']
})

# Map to new values
grade_map = {'A': 4.0, 'B': 3.0, 'C': 2.0}
df['gpa'] = df['grade'].map(grade_map)
print(df)
#   grade  gpa
# 0     A  4.0
# 1     B  3.0
# 2     C  2.0
# 3     A  4.0
# 4     B  3.0
```

### Vectorized Operations (Faster)

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5]
})

# ❌ Slow: apply with function
df['squared'] = df['value'].apply(lambda x: x ** 2)

# ✅ Fast: vectorized
df['squared'] = df['value'] ** 2

# ✅ Fast: NumPy functions
df['sqrt'] = np.sqrt(df['value'])

# ✅ Fast: conditional
df['category'] = np.where(df['value'] > 3, 'high', 'low')
```

---

## Value Replacement

### Replace Method

```python
import pandas as pd

df = pd.DataFrame({
    'status': ['active', 'inactive', 'active', 'pending'],
    'code': [1, 2, 1, 3]
})

# Replace single value
df['status'] = df['status'].replace('inactive', 'disabled')

# Replace multiple values
df['status'] = df['status'].replace({
    'active': 'on',
    'pending': 'waiting'
})
print(df)

# Replace in entire DataFrame
df = df.replace(1, 100)
```

### np.where for Conditional

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': [45, 75, 55, 90, 30]
})

# Binary condition
df['passed'] = np.where(df['score'] >= 60, 'Yes', 'No')
print(df)
#    score passed
# 0     45     No
# 1     75    Yes
# 2     55     No
# 3     90    Yes
# 4     30     No
```

### np.select for Multiple Conditions

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': [45, 75, 55, 90, 30]
})

conditions = [
    df['score'] >= 90,
    df['score'] >= 70,
    df['score'] >= 60,
    df['score'] >= 50
]
choices = ['A', 'B', 'C', 'D']

df['grade'] = np.select(conditions, choices, default='F')
print(df)
#    score grade
# 0     45     F
# 1     75     B
# 2     55     D
# 3     90     A
# 4     30     F
```

---

## Hands-on Exercise

### Your Task

```python
# Given this employee data:
import pandas as pd

employees = pd.DataFrame({
    'name': ['alice smith', 'BOB JONES', 'Charlie Brown'],
    'department': ['sales', 'engineering', 'sales'],
    'salary': [50000, 80000, 55000],
    'years': [2, 5, 3]
})

# 1. Clean the name column (title case)
# 2. Rename 'years' to 'experience'
# 3. Add 'bonus' column: 10% of salary if experience > 3, else 5%
# 4. Add 'level' column: Junior (<3), Mid (3-5), Senior (>5)
# 5. Sort by salary descending
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd
import numpy as np

employees = pd.DataFrame({
    'name': ['alice smith', 'BOB JONES', 'Charlie Brown'],
    'department': ['sales', 'engineering', 'sales'],
    'salary': [50000, 80000, 55000],
    'years': [2, 5, 3]
})

# 1. Clean name column
employees['name'] = employees['name'].str.title()
print("After title case:")
print(employees)

# 2. Rename years to experience
employees = employees.rename(columns={'years': 'experience'})

# 3. Add bonus column
employees['bonus'] = np.where(
    employees['experience'] > 3,
    employees['salary'] * 0.10,
    employees['salary'] * 0.05
)
print("\nWith bonus:")
print(employees)

# 4. Add level column
conditions = [
    employees['experience'] < 3,
    employees['experience'] <= 5,
    employees['experience'] > 5
]
choices = ['Junior', 'Mid', 'Senior']
employees['level'] = np.select(conditions, choices)
print("\nWith level:")
print(employees)

# 5. Sort by salary descending
employees = employees.sort_values('salary', ascending=False).reset_index(drop=True)
print("\nSorted by salary:")
print(employees)
```
</details>

---

## Summary

✅ Add columns with **direct assignment** or **`.assign()`** for chaining
✅ Remove with **`.drop(columns=[])`** or **`del df['col']`**
✅ **`.rename(columns={})`** for renaming
✅ **`.sort_values()`** for sorting, use `ascending=False` for descending
✅ **`.apply()`** for custom functions, but prefer **vectorized** operations
✅ **`np.where()`** for binary conditions, **`np.select()`** for multiple

**Next:** [Data Cleaning](./04-data-cleaning.md)

---

## Further Reading

- [Applying Functions](https://pandas.pydata.org/docs/user_guide/basics.html#function-application)
- [Sorting](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/basics.html
-->
