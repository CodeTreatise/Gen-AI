---
title: "Pandas for Data"
---

# Pandas for Data

## Overview

Pandas is Python's most popular library for data manipulation and analysis. Built on NumPy, it provides powerful, flexible data structures—DataFrames and Series—that make working with structured data intuitive and efficient.

This lesson covers Pandas fundamentals essential for AI/ML data preparation.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-dataframes-series.md) | DataFrames & Series | Creating, indexing, I/O |
| [02](./02-data-selection.md) | Data Selection | loc, iloc, boolean indexing |
| [03](./03-data-manipulation.md) | Data Manipulation | Transform, apply, sort |
| [04](./04-data-cleaning.md) | Data Cleaning | Missing values, duplicates, types |
| [05](./05-grouping-aggregation.md) | Grouping & Aggregation | groupby, pivot tables |
| [06](./06-merging-joining.md) | Merging & Joining | merge, concat, joins |

---

## Why Pandas?

| Task | Without Pandas | With Pandas |
|------|----------------|-------------|
| Load CSV | `csv.reader()` + loops | `pd.read_csv()` |
| Filter rows | Manual iteration | `df[df['col'] > 5]` |
| Group & aggregate | Complex dictionaries | `df.groupby().mean()` |
| Handle missing | Custom logic | `df.fillna()` |
| Merge datasets | Nested loops | `pd.merge()` |

---

## Quick Start

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

print(df)
#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago

# Basic operations
print(df['age'].mean())      # 30.0
print(df[df['age'] > 25])    # Filter
print(df.describe())          # Statistics
```

---

## Installation

```bash
pip install pandas
```

```python
import pandas as pd

print(pd.__version__)  # 2.x
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- NumPy basics (helpful)
- Basic understanding of tabular data

---

## Start Learning

Begin with [DataFrames and Series](./01-dataframes-series.md) to understand the core data structures.

---

## Further Reading

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
