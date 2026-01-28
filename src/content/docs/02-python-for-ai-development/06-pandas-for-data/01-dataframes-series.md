---
title: "DataFrames and Series"
---

# DataFrames and Series

## Introduction

DataFrames and Series are the two primary data structures in Pandas. A DataFrame is a 2D labeled table (like a spreadsheet), while a Series is a single column. Understanding these is fundamental to all Pandas operations.

### What We'll Cover

- Creating DataFrames
- Understanding Series
- Index and columns
- Data types
- Reading/writing files

### Prerequisites

- Python basics
- NumPy helpful

---

## Series

### Creating a Series

```python
import pandas as pd

# From list
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# With custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s)
# a    1
# b    2
# c    3

# From dictionary
s = pd.Series({'x': 10, 'y': 20, 'z': 30})
print(s)
```

### Series Attributes

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='values')

print(s.values)  # array([10, 20, 30])
print(s.index)   # Index(['a', 'b', 'c'])
print(s.name)    # 'values'
print(s.dtype)   # int64
print(len(s))    # 3
```

### Series Operations

```python
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])

# Vectorized operations
print(s * 2)       # [2, 4, 6, 8, 10]
print(s + 10)      # [11, 12, 13, 14, 15]
print(s ** 2)      # [1, 4, 9, 16, 25]

# Aggregations
print(s.sum())     # 15
print(s.mean())    # 3.0
print(s.max())     # 5
print(s.describe())
```

---

## DataFrames

### Creating from Dictionary

```python
import pandas as pd

# Dictionary of lists
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}

df = pd.DataFrame(data)
print(df)
#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago
```

### Creating from Lists

```python
import pandas as pd

# List of lists
data = [
    ['Alice', 25, 'NYC'],
    ['Bob', 30, 'LA'],
    ['Charlie', 35, 'Chicago']
]

df = pd.DataFrame(data, columns=['name', 'age', 'city'])
print(df)

# List of dictionaries
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]

df = pd.DataFrame(data)
```

### Creating from NumPy

```python
import pandas as pd
import numpy as np

arr = np.random.randint(0, 100, (3, 4))

df = pd.DataFrame(
    arr,
    columns=['A', 'B', 'C', 'D'],
    index=['row1', 'row2', 'row3']
)
print(df)
```

---

## Index and Columns

### Working with Index

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})

# Default numeric index
print(df.index)  # RangeIndex(start=0, stop=3, step=1)

# Set custom index
df.index = ['a', 'b', 'c']
print(df)

# Set column as index
df = df.reset_index()  # Restore numeric index
df = df.set_index('name')
print(df)
#          age
# name        
# Alice     25
# Bob       30
# Charlie   35

# Reset index back to column
df = df.reset_index()
```

### Working with Columns

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'city': ['NYC', 'LA']
})

# Column names
print(df.columns)  # Index(['name', 'age', 'city'])

# Rename columns
df.columns = ['Name', 'Age', 'City']

# Or use rename
df = df.rename(columns={'Name': 'name', 'Age': 'age'})

# Reorder columns
df = df[['city', 'name', 'age']]
```

---

## DataFrame Attributes

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000.0, 60000.0, 70000.0]
})

print(df.shape)      # (3, 3) - rows, columns
print(df.columns)    # Column names
print(df.index)      # Row labels
print(df.dtypes)     # Data types per column
print(df.values)     # NumPy array
print(df.info())     # Summary info
print(df.describe()) # Statistics
```

**Output of describe():**
```
             age        salary
count   3.000000      3.000000
mean   30.000000  60000.000000
std     5.000000  10000.000000
min    25.000000  50000.000000
25%    27.500000  55000.000000
50%    30.000000  60000.000000
75%    32.500000  65000.000000
max    35.000000  70000.000000
```

---

## Data Types

### Common dtypes

| Pandas dtype | Python type | Description |
|--------------|-------------|-------------|
| `int64` | int | Integer |
| `float64` | float | Float |
| `object` | str | String |
| `bool` | bool | Boolean |
| `datetime64` | datetime | Date/time |
| `category` | - | Categorical |

```python
import pandas as pd

df = pd.DataFrame({
    'ints': [1, 2, 3],
    'floats': [1.0, 2.0, 3.0],
    'strings': ['a', 'b', 'c'],
    'bools': [True, False, True]
})

print(df.dtypes)
# ints        int64
# floats    float64
# strings    object
# bools        bool
```

### Converting Types

```python
import pandas as pd

df = pd.DataFrame({
    'values': ['1', '2', '3'],
    'prices': ['10.5', '20.3', '30.1']
})

# Convert types
df['values'] = df['values'].astype(int)
df['prices'] = df['prices'].astype(float)

print(df.dtypes)
# values      int64
# prices    float64
```

---

## Reading Files

### CSV Files

```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Common parameters
df = pd.read_csv(
    'data.csv',
    sep=',',              # Delimiter
    header=0,             # Row to use as header
    names=['a', 'b', 'c'], # Custom column names
    index_col='id',       # Column to use as index
    usecols=['a', 'b'],   # Only read these columns
    nrows=100,            # Read first 100 rows
    dtype={'a': int},     # Specify data types
    na_values=['NA', ''], # Values to treat as NaN
    parse_dates=['date']  # Parse as datetime
)

# Write CSV
df.to_csv('output.csv', index=False)
```

### JSON Files

```python
import pandas as pd

# Read JSON
df = pd.read_json('data.json')

# Different orientations
df = pd.read_json('data.json', orient='records')
df = pd.read_json('data.json', orient='columns')

# Write JSON
df.to_json('output.json', orient='records', indent=2)
```

### Excel Files

```python
import pandas as pd

# Read Excel (requires openpyxl)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read multiple sheets
sheets = pd.read_excel('data.xlsx', sheet_name=None)
# Returns dict: {'Sheet1': df1, 'Sheet2': df2}

# Write Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)
```

### Other Formats

```python
import pandas as pd

# Parquet (efficient for large data)
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM users', conn)
df.to_sql('users', conn, if_exists='replace')

# Clipboard
df = pd.read_clipboard()  # Paste from Excel
```

---

## Viewing Data

```python
import pandas as pd

df = pd.DataFrame({
    'a': range(100),
    'b': range(100, 200)
})

# First/last rows
print(df.head())      # First 5 rows
print(df.head(10))    # First 10 rows
print(df.tail())      # Last 5 rows

# Sample random rows
print(df.sample(5))   # 5 random rows

# Get info
print(df.info())      # Column types, memory
print(df.describe())  # Statistics
print(df.shape)       # (rows, columns)
```

---

## Hands-on Exercise

### Your Task

```python
# 1. Create a DataFrame with student data:
#    - names: ['Alice', 'Bob', 'Charlie', 'Diana']
#    - math_score: [85, 90, 78, 92]
#    - english_score: [88, 75, 95, 87]
#
# 2. Set 'names' as the index
# 3. Add a column 'average' with the mean of both scores
# 4. Find the student with the highest average
# 5. Save to CSV (without the index column in file)
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd

# 1. Create DataFrame
df = pd.DataFrame({
    'names': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'math_score': [85, 90, 78, 92],
    'english_score': [88, 75, 95, 87]
})
print("Original DataFrame:")
print(df)

# 2. Set names as index
df = df.set_index('names')
print("\nWith names as index:")
print(df)

# 3. Add average column
df['average'] = (df['math_score'] + df['english_score']) / 2
# Or: df['average'] = df[['math_score', 'english_score']].mean(axis=1)
print("\nWith average:")
print(df)

# 4. Student with highest average
best_student = df['average'].idxmax()
print(f"\nHighest average: {best_student} ({df.loc[best_student, 'average']})")

# 5. Save to CSV
df.to_csv('students.csv')  # index=True by default (names column)
print("\nSaved to students.csv")
```
</details>

---

## Summary

✅ **Series** is a 1D labeled array (single column)
✅ **DataFrame** is a 2D labeled table (spreadsheet-like)
✅ Create DataFrames from **dicts, lists, or NumPy arrays**
✅ Use **`.index`** and **`.columns`** to access labels
✅ **`read_csv()`**, **`read_json()`**, **`read_excel()`** for file I/O
✅ **`.head()`**, **`.info()`**, **`.describe()`** for quick exploration

**Next:** [Data Selection](./02-data-selection.md)

---

## Further Reading

- [DataFrame Documentation](https://pandas.pydata.org/docs/reference/frame.html)
- [IO Tools](https://pandas.pydata.org/docs/user_guide/io.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/dsintro.html
-->
