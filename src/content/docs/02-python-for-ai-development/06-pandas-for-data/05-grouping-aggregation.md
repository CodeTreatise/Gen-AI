---
title: "Grouping and Aggregation"
---

# Grouping and Aggregation

## Introduction

Grouping data and computing aggregates is fundamental to data analysis. Pandas' `groupby` enables SQL-like operations—split data by categories, apply functions, and combine results.

### What We'll Cover

- GroupBy operations
- Aggregation functions
- Multiple aggregations
- Transform operations
- Pivot tables
- Cross-tabulation

### Prerequisites

- DataFrames basics
- Data manipulation

---

## GroupBy Basics

### Split-Apply-Combine

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering', 'Sales'],
    'employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'salary': [50000, 55000, 80000, 75000, 52000]
})

# Group by department
grouped = df.groupby('department')

# What is grouped?
print(type(grouped))  # DataFrameGroupBy

# See the groups
print(grouped.groups)
# {'Engineering': [2, 3], 'Sales': [0, 1, 4]}
```

### Basic Aggregation

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering'],
    'salary': [50000, 55000, 80000, 75000]
})

# Calculate mean salary per department
result = df.groupby('department')['salary'].mean()
print(result)
# department
# Engineering    77500.0
# Sales          52500.0

# Multiple aggregations
print(df.groupby('department')['salary'].sum())
print(df.groupby('department')['salary'].count())
print(df.groupby('department')['salary'].min())
print(df.groupby('department')['salary'].max())
```

### GroupBy on Multiple Columns

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering'],
    'region': ['North', 'South', 'North', 'South'],
    'salary': [50000, 55000, 80000, 75000]
})

# Group by multiple columns
result = df.groupby(['department', 'region'])['salary'].mean()
print(result)
# department   region
# Engineering  North     80000.0
#              South     75000.0
# Sales        North     50000.0
#              South     55000.0

# Reset index to get DataFrame
result = df.groupby(['department', 'region'])['salary'].mean().reset_index()
print(result)
```

---

## Aggregation Functions

### Built-in Aggregations

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

# Common aggregation functions
print(df.groupby('category')['value'].mean())   # Average
print(df.groupby('category')['value'].sum())    # Sum
print(df.groupby('category')['value'].count())  # Count
print(df.groupby('category')['value'].min())    # Minimum
print(df.groupby('category')['value'].max())    # Maximum
print(df.groupby('category')['value'].std())    # Standard deviation
print(df.groupby('category')['value'].median()) # Median
print(df.groupby('category')['value'].first())  # First value
print(df.groupby('category')['value'].last())   # Last value
```

### Multiple Aggregations with agg()

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50],
    'quantity': [1, 2, 3, 4, 5]
})

# Multiple functions on one column
result = df.groupby('category')['value'].agg(['mean', 'sum', 'count'])
print(result)
#           mean  sum  count
# category
# A         15.0   30      2
# B         40.0  120      3

# Different functions for different columns
result = df.groupby('category').agg({
    'value': ['mean', 'max'],
    'quantity': 'sum'
})
print(result)
```

### Named Aggregations

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'quantity': [1, 2, 3, 4]
})

# Named aggregations (cleaner output)
result = df.groupby('category').agg(
    avg_value=('value', 'mean'),
    total_value=('value', 'sum'),
    max_quantity=('quantity', 'max'),
    count=('value', 'count')
)
print(result)
#           avg_value  total_value  max_quantity  count
# category
# A              15.0           30             2      2
# B              35.0           70             4      2
```

### Custom Aggregation Functions

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

# Custom function
def range_func(x):
    return x.max() - x.min()

result = df.groupby('category')['value'].agg(range_func)
print(result)
# category
# A    10
# B    20

# Lambda function
result = df.groupby('category')['value'].agg(lambda x: x.max() - x.min())

# Multiple custom functions
result = df.groupby('category')['value'].agg([
    ('range', lambda x: x.max() - x.min()),
    ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
])
```

---

## Transform Operations

### Transform vs Aggregate

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering'],
    'salary': [50000, 55000, 80000, 75000]
})

# Aggregate: returns reduced DataFrame
agg_result = df.groupby('department')['salary'].mean()
print("Aggregate result:")
print(agg_result)
# department
# Engineering    77500.0
# Sales          52500.0

# Transform: returns same-size DataFrame
transform_result = df.groupby('department')['salary'].transform('mean')
print("\nTransform result:")
print(transform_result)
# 0    52500.0
# 1    52500.0
# 2    77500.0
# 3    77500.0
```

### Common Transform Use Cases

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering'],
    'salary': [50000, 55000, 80000, 75000]
})

# Add department mean as new column
df['dept_mean'] = df.groupby('department')['salary'].transform('mean')

# Calculate percentage of department total
df['pct_of_dept'] = df['salary'] / df.groupby('department')['salary'].transform('sum') * 100

# Normalize within group (z-score)
df['z_score'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df)
```

### Filling Missing by Group

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, np.nan, 30, np.nan]
})

# Fill NaN with group mean
df['value_filled'] = df.groupby('category')['value'].transform(
    lambda x: x.fillna(x.mean())
)
print(df)
```

---

## Pivot Tables

### Basic Pivot Table

```python
import pandas as pd

df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})

# Pivot table
pivot = df.pivot_table(
    values='sales',
    index='date',
    columns='product',
    aggfunc='sum'
)
print(pivot)
# product      A      B
# date
# 2024-01    100    200
# 2024-02    150    250
```

### Multiple Aggregations

```python
import pandas as pd

df = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250],
    'quantity': [10, 20, 15, 25]
})

# Multiple values
pivot = df.pivot_table(
    values=['sales', 'quantity'],
    index='region',
    columns='product',
    aggfunc='sum'
)
print(pivot)

# Multiple aggregation functions
pivot = df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc=['sum', 'mean']
)
print(pivot)
```

### Margins (Totals)

```python
import pandas as pd

df = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})

# Add row/column totals
pivot = df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)
print(pivot)
# product      A      B  Total
# region
# North      100    200    300
# South      150    250    400
# Total      250    450    700
```

---

## Cross-Tabulation

### Basic Crosstab

```python
import pandas as pd

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'product': ['A', 'A', 'B', 'B', 'A', 'B']
})

# Count occurrences
ct = pd.crosstab(df['gender'], df['product'])
print(ct)
# product  A  B
# gender
# F        1  2
# M        2  1

# With percentages
ct = pd.crosstab(df['gender'], df['product'], normalize='all')
print(ct * 100)  # Percentages
```

### Crosstab with Values

```python
import pandas as pd

df = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})

# Sum of sales
ct = pd.crosstab(
    df['region'],
    df['product'],
    values=df['sales'],
    aggfunc='sum'
)
print(ct)
```

---

## Hands-on Exercise

### Your Task

```python
# Given this sales data:
import pandas as pd

sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', 
             '2024-01-01', '2024-01-02'],
    'region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'product': ['A', 'A', 'A', 'B', 'B', 'B'],
    'quantity': [10, 15, 12, 20, 8, 18],
    'price': [100, 100, 100, 150, 150, 150]
})

# 1. Calculate total revenue (quantity * price) per region
# 2. Find average quantity per product per region
# 3. Add a column showing each row's % of regional total revenue
# 4. Create a pivot table: regions as rows, products as columns, sum of revenue
# 5. Which region-product combination has highest revenue?
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd

sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', 
             '2024-01-01', '2024-01-02'],
    'region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'product': ['A', 'A', 'A', 'B', 'B', 'B'],
    'quantity': [10, 15, 12, 20, 8, 18],
    'price': [100, 100, 100, 150, 150, 150]
})

# Add revenue column
sales['revenue'] = sales['quantity'] * sales['price']

# 1. Total revenue per region
revenue_by_region = sales.groupby('region')['revenue'].sum()
print("1. Revenue by region:")
print(revenue_by_region)

# 2. Average quantity per product per region
avg_qty = sales.groupby(['region', 'product'])['quantity'].mean()
print("\n2. Avg quantity by region & product:")
print(avg_qty)

# 3. Percentage of regional total
sales['pct_of_region'] = (
    sales['revenue'] / 
    sales.groupby('region')['revenue'].transform('sum') * 100
)
print("\n3. With percentage of regional total:")
print(sales[['region', 'product', 'revenue', 'pct_of_region']])

# 4. Pivot table
pivot = sales.pivot_table(
    values='revenue',
    index='region',
    columns='product',
    aggfunc='sum',
    margins=True
)
print("\n4. Pivot table:")
print(pivot)

# 5. Highest revenue combination
best = sales.groupby(['region', 'product'])['revenue'].sum().idxmax()
best_value = sales.groupby(['region', 'product'])['revenue'].sum().max()
print(f"\n5. Highest: {best} with ${best_value}")
```
</details>

---

## Summary

✅ **`groupby()`** splits data into groups for aggregation
✅ Use **`.agg()`** for multiple aggregations with named results
✅ **`transform()`** returns same-size output (add group stats to rows)
✅ **`pivot_table()`** for spreadsheet-style summaries
✅ **`pd.crosstab()`** for frequency tables
✅ Chain with **`.reset_index()`** to get clean DataFrames

**Next:** [Merging and Joining](./06-merging-joining.md)

---

## Further Reading

- [GroupBy Documentation](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pivot Tables](https://pandas.pydata.org/docs/user_guide/reshaping.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/groupby.html
-->
