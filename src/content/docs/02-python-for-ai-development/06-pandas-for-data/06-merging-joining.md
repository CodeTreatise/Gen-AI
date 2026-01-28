---
title: "Merging and Joining"
---

# Merging and Joining

## Introduction

Combining data from multiple sources is a common task. Pandas provides SQL-like merge and join operations, plus concatenation for stacking DataFrames.

### What We'll Cover

- Merge for SQL-like joins
- Join types (inner, outer, left, right)
- Concatenating DataFrames
- Handling duplicates
- Multi-index operations

### Prerequisites

- DataFrames basics
- Data selection

---

## Merge Basics

### Simple Merge

```python
import pandas as pd

# Two related DataFrames
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana']
})

salaries = pd.DataFrame({
    'emp_id': [1, 2, 3, 5],
    'salary': [50000, 60000, 70000, 80000]
})

# Merge on common column
merged = pd.merge(employees, salaries, on='emp_id')
print(merged)
#    emp_id     name  salary
# 0       1    Alice   50000
# 1       2      Bob   60000
# 2       3  Charlie   70000
```

### Different Column Names

```python
import pandas as pd

employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

departments = pd.DataFrame({
    'employee_id': [1, 2, 3],
    'dept': ['Sales', 'Engineering', 'Marketing']
})

# Merge on different column names
merged = pd.merge(
    employees,
    departments,
    left_on='emp_id',
    right_on='employee_id'
)
print(merged)
```

### Merge on Multiple Columns

```python
import pandas as pd

df1 = pd.DataFrame({
    'year': [2023, 2023, 2024, 2024],
    'region': ['North', 'South', 'North', 'South'],
    'sales': [100, 200, 150, 250]
})

df2 = pd.DataFrame({
    'year': [2023, 2023, 2024],
    'region': ['North', 'South', 'North'],
    'target': [90, 180, 140]
})

# Merge on multiple columns
merged = pd.merge(df1, df2, on=['year', 'region'])
print(merged)
```

---

## Join Types

### Inner Join (Default)

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [2, 3, 4], 'B': ['b2', 'b3', 'b4']})

# Only matching keys
inner = pd.merge(left, right, on='key', how='inner')
print(inner)
#    key   A   B
# 0    2  a2  b2
# 1    3  a3  b3
```

### Left Join

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [2, 3, 4], 'B': ['b2', 'b3', 'b4']})

# All keys from left, matching from right
left_join = pd.merge(left, right, on='key', how='left')
print(left_join)
#    key   A    B
# 0    1  a1  NaN
# 1    2  a2   b2
# 2    3  a3   b3
```

### Right Join

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [2, 3, 4], 'B': ['b2', 'b3', 'b4']})

# All keys from right, matching from left
right_join = pd.merge(left, right, on='key', how='right')
print(right_join)
#    key    A   B
# 0    2   a2  b2
# 1    3   a3  b3
# 2    4  NaN  b4
```

### Outer (Full) Join

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [2, 3, 4], 'B': ['b2', 'b3', 'b4']})

# All keys from both
outer = pd.merge(left, right, on='key', how='outer')
print(outer)
#    key    A    B
# 0    1   a1  NaN
# 1    2   a2   b2
# 2    3   a3   b3
# 3    4  NaN   b4
```

### Join Types Summary

| Type | Description | SQL Equivalent |
|------|-------------|----------------|
| `inner` | Only matching keys | INNER JOIN |
| `left` | All from left + matching | LEFT OUTER JOIN |
| `right` | All from right + matching | RIGHT OUTER JOIN |
| `outer` | All from both | FULL OUTER JOIN |

---

## Handling Column Name Conflicts

### Suffixes

```python
import pandas as pd

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [100, 200, 300]
})

# Default suffixes
merged = pd.merge(df1, df2, on='id')
print(merged)
#    id  value_x  value_y
# 0   1       10      100
# 1   2       20      200
# 2   3       30      300

# Custom suffixes
merged = pd.merge(df1, df2, on='id', suffixes=('_2023', '_2024'))
print(merged)
#    id  value_2023  value_2024
```

---

## Concatenation

### Vertical Concatenation (Stacking Rows)

```python
import pandas as pd

df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

# Stack vertically
result = pd.concat([df1, df2])
print(result)
#    A  B
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8

# Reset index
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#    A  B
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8
```

### Horizontal Concatenation

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

# Stack horizontally
result = pd.concat([df1, df2], axis=1)
print(result)
#    A  B  C  D
# 0  1  3  5  7
# 1  2  4  6  8
```

### Handling Missing Columns

```python
import pandas as pd

df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'C': [7, 8]  # Different column
})

# Outer join (default): keeps all columns
result = pd.concat([df1, df2])
print(result)
#    A    B    C
# 0  1  3.0  NaN
# 1  2  4.0  NaN
# 0  5  NaN  7.0
# 1  6  NaN  8.0

# Inner join: only common columns
result = pd.concat([df1, df2], join='inner')
print(result)
#    A
# 0  1
# 1  2
# 0  5
# 1  6
```

### Adding Keys

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'A': [3, 4]})

# Add source identifier
result = pd.concat([df1, df2], keys=['source1', 'source2'])
print(result)
#            A
# source1 0  1
#         1  2
# source2 0  3
#         1  4

# Access by key
print(result.loc['source1'])
```

---

## Join Method

### Index-Based Join

```python
import pandas as pd

df1 = pd.DataFrame(
    {'A': [1, 2, 3]},
    index=['a', 'b', 'c']
)

df2 = pd.DataFrame(
    {'B': [4, 5, 6]},
    index=['b', 'c', 'd']
)

# Join on index
result = df1.join(df2, how='inner')
print(result)
#    A  B
# b  2  4
# c  3  5

result = df1.join(df2, how='outer')
print(result)
#      A    B
# a  1.0  NaN
# b  2.0  4.0
# c  3.0  5.0
# d  NaN  6.0
```

### Join with Suffix

```python
import pandas as pd

df1 = pd.DataFrame({'value': [1, 2]}, index=['a', 'b'])
df2 = pd.DataFrame({'value': [3, 4]}, index=['a', 'b'])

result = df1.join(df2, lsuffix='_left', rsuffix='_right')
print(result)
#    value_left  value_right
# a           1            3
# b           2            4
```

---

## Validating Merges

### Check for Duplicates

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 1, 2], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [1, 2, 2], 'B': ['b1', 'b2', 'b3']})

# Validate merge relationship
try:
    merged = pd.merge(left, right, on='key', validate='one_to_one')
except pd.errors.MergeError as e:
    print(f"Validation failed: {e}")

# Options: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
```

### Indicator Column

```python
import pandas as pd

left = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
right = pd.DataFrame({'key': [2, 3, 4], 'B': ['b2', 'b3', 'b4']})

# Add indicator showing merge source
merged = pd.merge(left, right, on='key', how='outer', indicator=True)
print(merged)
#    key    A    B      _merge
# 0    1   a1  NaN   left_only
# 1    2   a2   b2        both
# 2    3   a3   b3        both
# 3    4  NaN   b4  right_only

# Find unmatched rows
left_only = merged[merged['_merge'] == 'left_only']
```

---

## Practical Examples

### Multiple Table Join

```python
import pandas as pd

# Three related tables
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104],
    'customer_id': [1, 1, 2, 3],
    'product_id': [1, 2, 1, 3]
})

products = pd.DataFrame({
    'product_id': [1, 2, 3],
    'product_name': ['Laptop', 'Phone', 'Tablet'],
    'price': [1000, 500, 300]
})

# Chain merges
result = (orders
    .merge(customers, on='customer_id')
    .merge(products, on='product_id')
)
print(result)
```

### Combine CSVs

```python
import pandas as pd
from pathlib import Path

# Read and combine multiple CSVs
def combine_csvs(folder_path):
    dfs = []
    for file in Path(folder_path).glob('*.csv'):
        df = pd.read_csv(file)
        df['source_file'] = file.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# combined = combine_csvs('data/')
```

---

## Hands-on Exercise

### Your Task

```python
# Given these three DataFrames:
import pandas as pd

users = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'username': ['alice', 'bob', 'charlie', 'diana']
})

purchases = pd.DataFrame({
    'purchase_id': [101, 102, 103, 104, 105],
    'user_id': [1, 1, 2, 3, 5],  # Note: user 5 doesn't exist
    'product_id': [1, 2, 1, 3, 2],
    'quantity': [2, 1, 3, 1, 2]
})

products = pd.DataFrame({
    'product_id': [1, 2, 3],
    'product_name': ['Widget', 'Gadget', 'Gizmo'],
    'price': [10.00, 25.00, 15.00]
})

# 1. Join all three tables to show: username, product_name, quantity, total_price
# 2. Find users who haven't made any purchases
# 3. Find purchases from non-existent users
# 4. Calculate total revenue per user
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd

users = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'username': ['alice', 'bob', 'charlie', 'diana']
})

purchases = pd.DataFrame({
    'purchase_id': [101, 102, 103, 104, 105],
    'user_id': [1, 1, 2, 3, 5],
    'product_id': [1, 2, 1, 3, 2],
    'quantity': [2, 1, 3, 1, 2]
})

products = pd.DataFrame({
    'product_id': [1, 2, 3],
    'product_name': ['Widget', 'Gadget', 'Gizmo'],
    'price': [10.00, 25.00, 15.00]
})

# 1. Join all tables
full_data = (purchases
    .merge(users, on='user_id', how='left')
    .merge(products, on='product_id')
)
full_data['total_price'] = full_data['quantity'] * full_data['price']

print("1. Full joined data:")
print(full_data[['username', 'product_name', 'quantity', 'total_price']])

# 2. Users without purchases
merged = pd.merge(users, purchases, on='user_id', how='left', indicator=True)
no_purchases = merged[merged['_merge'] == 'left_only']['username'].unique()
print(f"\n2. Users without purchases: {list(no_purchases)}")

# 3. Purchases from non-existent users
merged = pd.merge(purchases, users, on='user_id', how='left', indicator=True)
invalid = merged[merged['_merge'] == 'left_only']
print(f"\n3. Purchases from non-existent users:")
print(invalid[['purchase_id', 'user_id']])

# 4. Revenue per user
revenue = full_data.groupby('username')['total_price'].sum()
print(f"\n4. Revenue per user:")
print(revenue)
```
</details>

---

## Summary

✅ **`pd.merge()`** for SQL-like joins on columns
✅ **`how`** parameter: 'inner', 'left', 'right', 'outer'
✅ Use **`left_on`/`right_on`** for different column names
✅ **`pd.concat()`** for stacking DataFrames (rows or columns)
✅ **`indicator=True`** to track merge sources
✅ **`validate`** to check relationship assumptions

**Back to:** [Pandas Overview](./00-pandas-for-data.md)

---

## Further Reading

- [Merge, Join, Concatenate](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Database-style Joins](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/merging.html
-->
