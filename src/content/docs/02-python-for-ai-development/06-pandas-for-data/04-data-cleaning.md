---
title: "Data Cleaning"
---

# Data Cleaning

## Introduction

Real-world data is messy. Missing values, duplicates, incorrect types, and inconsistent formats are common. Pandas provides comprehensive tools for cleaning data before analysis or model training.

### What We'll Cover

- Handling missing values
- Removing duplicates
- Data type conversion
- String operations
- Date/time handling
- Outlier detection

### Prerequisites

- DataFrames basics
- Data selection and manipulation

---

## Handling Missing Values

### Detecting Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'Diana'],
    'age': [25, np.nan, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', None]
})

# Check for missing
print(df.isna())  # or df.isnull()
#     name    age   city
# 0  False  False  False
# 1  False   True  False
# 2   True  False  False
# 3  False  False   True

# Count missing per column
print(df.isna().sum())
# name    1
# age     1
# city    1

# Total missing
print(df.isna().sum().sum())  # 3

# Percentage missing
print(df.isna().mean() * 100)
```

### Dropping Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': [1, 2, np.nan, 4],
    'b': [5, np.nan, np.nan, 8],
    'c': [9, 10, 11, 12]
})

# Drop rows with any missing
df_clean = df.dropna()
print(df_clean)
#      a    b   c
# 0  1.0  5.0   9
# 3  4.0  8.0  12

# Drop rows only if all values missing
df_clean = df.dropna(how='all')

# Drop if missing in specific columns
df_clean = df.dropna(subset=['a'])

# Require minimum non-null values
df_clean = df.dropna(thresh=2)  # At least 2 non-null
```

### Filling Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [1, np.nan, 3, np.nan, 5],
    'category': ['A', 'B', None, 'A', 'B']
})

# Fill with constant
df['value'] = df['value'].fillna(0)
df['category'] = df['category'].fillna('Unknown')

# Fill with mean/median/mode
df = pd.DataFrame({'value': [1, np.nan, 3, np.nan, 5]})
df['value'] = df['value'].fillna(df['value'].mean())

# Forward fill (use previous value)
df['value'] = df['value'].fillna(method='ffill')

# Backward fill (use next value)
df['value'] = df['value'].fillna(method='bfill')

# Fill with interpolation
df['value'] = df['value'].interpolate()
```

### Missing Value Strategies

| Strategy | When to Use |
|----------|-------------|
| Drop row | Few missing, random |
| Fill with mean | Numeric, normally distributed |
| Fill with median | Numeric, with outliers |
| Fill with mode | Categorical |
| Forward/backward fill | Time series |
| Custom logic | Domain-specific |

---

## Removing Duplicates

### Detecting Duplicates

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'age': [25, 30, 25, 35, 30]
})

# Check for duplicates
print(df.duplicated())
# 0    False
# 1    False
# 2     True
# 3    False
# 4     True

# Count duplicates
print(df.duplicated().sum())  # 2

# Show duplicate rows
print(df[df.duplicated(keep=False)])
```

### Removing Duplicates

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'age': [25, 30, 25, 35, 31],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
})

# Remove exact duplicates
df_clean = df.drop_duplicates()
print(df_clean)

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['name'])
print(df_clean)
#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 3  Charlie   35  Chicago

# Keep last occurrence
df_clean = df.drop_duplicates(subset=['name'], keep='last')

# Keep no duplicates
df_clean = df.drop_duplicates(subset=['name'], keep=False)
```

---

## Data Type Conversion

### Converting Types

```python
import pandas as pd

df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'price': ['10.5', '20.3', '15.7'],
    'active': ['True', 'False', 'True']
})

print(df.dtypes)
# All 'object' (string)

# Convert to numeric
df['id'] = df['id'].astype(int)
df['price'] = df['price'].astype(float)

# Convert to boolean
df['active'] = df['active'].map({'True': True, 'False': False})

print(df.dtypes)
# id         int64
# price    float64
# active      bool
```

### Handling Conversion Errors

```python
import pandas as pd

df = pd.DataFrame({
    'value': ['1', '2', 'three', '4', 'N/A']
})

# Errors='coerce' replaces invalid with NaN
df['value'] = pd.to_numeric(df['value'], errors='coerce')
print(df)
#    value
# 0    1.0
# 1    2.0
# 2    NaN
# 3    4.0
# 4    NaN

# Then fill NaN
df['value'] = df['value'].fillna(0)
```

### Category Type

```python
import pandas as pd

df = pd.DataFrame({
    'status': ['active', 'inactive', 'active', 'pending'] * 1000
})

# Convert to category (memory efficient)
print(f"Before: {df['status'].memory_usage()} bytes")
df['status'] = df['status'].astype('category')
print(f"After: {df['status'].memory_usage()} bytes")

# Check categories
print(df['status'].cat.categories)
```

---

## String Operations

### String Accessor

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie', 'Diana Smith']
})

# Common string operations
df['name'] = df['name'].str.strip()     # Remove whitespace
df['name'] = df['name'].str.lower()     # Lowercase
df['name'] = df['name'].str.upper()     # Uppercase
df['name'] = df['name'].str.title()     # Title Case

# Replace
df['name'] = df['name'].str.replace(' ', '_')

# Contains
mask = df['name'].str.contains('Smith', case=False)
```

### Splitting Strings

```python
import pandas as pd

df = pd.DataFrame({
    'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown']
})

# Split into columns
df[['first', 'last']] = df['full_name'].str.split(' ', expand=True)
print(df)
#        full_name    first   last
# 0    Alice Smith    Alice  Smith
# 1      Bob Jones      Bob  Jones
# 2  Charlie Brown  Charlie  Brown
```

### Extracting with Regex

```python
import pandas as pd

df = pd.DataFrame({
    'email': ['alice@gmail.com', 'bob@yahoo.com', 'charlie@company.org']
})

# Extract domain
df['domain'] = df['email'].str.extract(r'@(.+)$')
print(df)
#                  email       domain
# 0      alice@gmail.com    gmail.com
# 1        bob@yahoo.com    yahoo.com
# 2  charlie@company.org  company.org

# Extract multiple groups
df[['user', 'domain']] = df['email'].str.extract(r'(.+)@(.+)')
```

---

## Date/Time Handling

### Converting to Datetime

```python
import pandas as pd

df = pd.DataFrame({
    'date_str': ['2024-01-15', '2024-02-20', '2024-03-25'],
    'time_str': ['14:30:00', '09:15:00', '16:45:00']
})

# Convert to datetime
df['date'] = pd.to_datetime(df['date_str'])
print(df['date'].dtype)  # datetime64[ns]

# Parse various formats
df = pd.DataFrame({
    'date': ['15/01/2024', '20-02-2024', 'March 25, 2024']
})
df['parsed'] = pd.to_datetime(df['date'], format='mixed')

# Handle errors
df['parsed'] = pd.to_datetime(df['date'], errors='coerce')
```

### Extracting Components

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-15 14:30:00', 
                                  '2024-06-20 09:15:00',
                                  '2024-12-25 16:45:00'])
})

# Extract components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()
df['quarter'] = df['timestamp'].dt.quarter

print(df)
```

### Date Arithmetic

```python
import pandas as pd
from datetime import timedelta

df = pd.DataFrame({
    'start_date': pd.to_datetime(['2024-01-01', '2024-02-15'])
})

# Add time delta
df['end_date'] = df['start_date'] + timedelta(days=30)

# Calculate difference
df['days_diff'] = (df['end_date'] - df['start_date']).dt.days

# Days from today
df['days_ago'] = (pd.Timestamp.now() - df['start_date']).dt.days
```

---

## Outlier Detection

### Statistical Methods

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [10, 12, 14, 15, 100, 11, 13, 12, 200, 14]
})

# Z-score method
mean = df['value'].mean()
std = df['value'].std()
df['z_score'] = (df['value'] - mean) / std
outliers = df[abs(df['z_score']) > 2]
print("Z-score outliers:")
print(outliers)

# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print("\nIQR outliers:")
print(outliers)
```

### Handling Outliers

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [10, 12, 14, 15, 100, 11, 13, 12, 200, 14]
})

# Option 1: Remove outliers
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5 * IQR) & 
              (df['value'] <= Q3 + 1.5 * IQR)]

# Option 2: Cap outliers (winsorize)
lower = df['value'].quantile(0.05)
upper = df['value'].quantile(0.95)
df['value_capped'] = df['value'].clip(lower, upper)

# Option 3: Replace with median
median = df['value'].median()
df['value_fixed'] = np.where(
    abs(df['value'] - median) > 2 * df['value'].std(),
    median,
    df['value']
)
```

---

## Hands-on Exercise

### Your Task

```python
# Clean this messy dataset:
import pandas as pd
import numpy as np

messy_data = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'alice', 'Charlie', None],
    'email': ['alice@gmail.com', 'bob@yahoo', 'alice@gmail.com', 'charlie@company.org', 'diana@test.com'],
    'age': ['25', '30', '25', 'thirty-five', '28'],
    'signup_date': ['2024-01-15', '15/02/2024', '2024-01-15', 'March 10, 2024', '2024-04-20'],
    'score': [85, 92, 85, 1000, 78]  # 1000 is an outlier
})

# 1. Clean and standardize names
# 2. Remove duplicate rows (based on email)
# 3. Convert age to numeric, handling errors
# 4. Parse all date formats
# 5. Detect and handle the outlier in score
```

<details>
<summary>✅ Solution</summary>

```python
import pandas as pd
import numpy as np

messy_data = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'alice', 'Charlie', None],
    'email': ['alice@gmail.com', 'bob@yahoo', 'alice@gmail.com', 'charlie@company.org', 'diana@test.com'],
    'age': ['25', '30', '25', 'thirty-five', '28'],
    'signup_date': ['2024-01-15', '15/02/2024', '2024-01-15', 'March 10, 2024', '2024-04-20'],
    'score': [85, 92, 85, 1000, 78]
})

print("Original data:")
print(messy_data)
print(f"\nMissing values: {messy_data.isna().sum().sum()}")

# 1. Clean names
messy_data['name'] = messy_data['name'].str.strip().str.title()
messy_data['name'] = messy_data['name'].fillna('Unknown')
print("\n1. After cleaning names:")
print(messy_data['name'])

# 2. Remove duplicates by email
messy_data = messy_data.drop_duplicates(subset=['email'], keep='first')
print(f"\n2. After removing duplicates: {len(messy_data)} rows")

# 3. Convert age to numeric
messy_data['age'] = pd.to_numeric(messy_data['age'], errors='coerce')
messy_data['age'] = messy_data['age'].fillna(messy_data['age'].median())
print("\n3. Age column (numeric):")
print(messy_data['age'])

# 4. Parse dates
messy_data['signup_date'] = pd.to_datetime(messy_data['signup_date'], format='mixed')
print("\n4. Parsed dates:")
print(messy_data['signup_date'])

# 5. Handle outlier in score
median_score = messy_data['score'].median()
std_score = messy_data['score'].std()
messy_data['score'] = np.where(
    messy_data['score'] > median_score + 2 * std_score,
    median_score,
    messy_data['score']
)
print("\n5. Score after outlier handling:")
print(messy_data['score'])

print("\n=== Final cleaned data ===")
print(messy_data)
```
</details>

---

## Summary

✅ Use **`.isna()`** to find missing, **`.fillna()`** or **`.dropna()`** to handle
✅ **`.drop_duplicates()`** with `subset` for specific columns
✅ **`.astype()`** and **`pd.to_numeric()`** for type conversion
✅ **`.str`** accessor for string operations
✅ **`pd.to_datetime()`** with `format='mixed'` for dates
✅ Use **IQR** or **Z-score** for outlier detection

**Next:** [Grouping and Aggregation](./05-grouping-aggregation.md)

---

## Further Reading

- [Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Working with Text](https://pandas.pydata.org/docs/user_guide/text.html)
- [Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

<!-- 
Sources Consulted:
- Pandas Docs: https://pandas.pydata.org/docs/user_guide/missing_data.html
-->
