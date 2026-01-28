---
title: "Seaborn for Statistical Visualization"
---

# Seaborn for Statistical Visualization

## Introduction

Seaborn is built on matplotlib and provides a high-level interface for statistical graphics. It integrates with pandas DataFrames and makes complex visualizations simple.

### What We'll Cover

- Themes and styling
- Distribution plots
- Categorical plots
- Relational plots
- Regression plots
- Pair plots and heatmaps

### Prerequisites

- Matplotlib fundamentals
- Pandas DataFrames

---

## Setup and Themes

### Basic Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set theme
sns.set_theme(style='whitegrid')

# Load sample dataset
tips = sns.load_dataset('tips')
print(tips.head())
```

### Style Options

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, style in zip(axes, styles):
    sns.set_style(style)
    ax.plot(np.random.randn(50).cumsum())
    ax.set_title(style)

plt.tight_layout()
plt.show()

# Reset to default
sns.set_theme()
```

### Color Palettes

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Built-in palettes
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

for palette in palettes:
    sns.palplot(sns.color_palette(palette))
    plt.title(palette)
    plt.show()

# Set default palette
sns.set_palette('colorblind')  # Accessible!
```

---

## Distribution Plots

### Histogram with KDE

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax)
ax.set_title('Distribution of Total Bill')
plt.show()
```

### KDE Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, ax=ax)
ax.set_title('Total Bill Distribution by Time')
plt.show()
```

### Displot (Figure-Level)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Figure-level function (creates own figure)
g = sns.displot(
    data=tips,
    x='total_bill',
    hue='day',
    kind='kde',
    height=6,
    aspect=1.5
)
g.set_axis_labels('Total Bill ($)', 'Density')
plt.show()
```

---

## Categorical Plots

### Box Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', ax=ax)
ax.set_title('Total Bill by Day and Sex')
plt.show()
```

### Violin Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, ax=ax)
ax.set_title('Total Bill Distribution by Day')
plt.show()
```

### Bar Plot (with Error Bars)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ax=ax)
ax.set_title('Average Total Bill by Day')
plt.show()
```

### Strip and Swarm Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strip plot (points with jitter)
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex', 
              dodge=True, ax=axes[0])
axes[0].set_title('Strip Plot')

# Swarm plot (non-overlapping points)
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex',
              dodge=True, ax=axes[1])
axes[1].set_title('Swarm Plot')

plt.tight_layout()
plt.show()
```

### Catplot (Figure-Level)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Create faceted categorical plot
g = sns.catplot(
    data=tips,
    x='day',
    y='total_bill',
    hue='sex',
    col='time',
    kind='box',
    height=5,
    aspect=1
)
g.set_axis_labels('Day', 'Total Bill ($)')
plt.show()
```

---

## Relational Plots

### Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='time',
    size='size',
    sizes=(20, 200),
    ax=ax
)
ax.set_title('Tip vs Total Bill')
plt.show()
```

### Line Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create time series data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
data = pd.DataFrame({
    'date': dates,
    'group': np.tile(['A', 'B'], 50),
    'value': np.random.randn(100).cumsum() + np.tile([0, 10], 50)
})

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='value', hue='group', ax=ax)
ax.set_title('Time Series by Group')
plt.xticks(rotation=45)
plt.show()
```

### Relplot (Figure-Level)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Faceted scatter plot
g = sns.relplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='smoker',
    col='time',
    row='sex',
    height=4,
    aspect=1.2
)
plt.show()
```

---

## Regression Plots

### Linear Regression

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip', ax=ax)
ax.set_title('Linear Regression: Tip vs Total Bill')
plt.show()
```

### Grouped Regression

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# lmplot: figure-level with grouping
g = sns.lmplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='smoker',
    col='time',
    height=5,
    aspect=1
)
plt.show()
```

---

## Pair Plots and Heatmaps

### Pair Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# All pairwise relationships
g = sns.pairplot(iris, hue='species', diag_kind='kde')
plt.show()
```

### Correlation Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Calculate correlation matrix
numeric_cols = tips.select_dtypes(include='number')
corr = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax
)
ax.set_title('Correlation Heatmap')
plt.show()
```

### Cluster Map

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
np.random.seed(42)
data = np.random.randn(10, 10)

# Clustered heatmap
g = sns.clustermap(
    data,
    cmap='viridis',
    figsize=(10, 10),
    annot=True,
    fmt='.1f'
)
plt.show()
```

---

## FacetGrid

### Custom Faceted Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Create FacetGrid
g = sns.FacetGrid(tips, col='time', row='sex', height=4, aspect=1.2)

# Map a plot to each facet
g.map(sns.histplot, 'total_bill', kde=True)

# Add titles
g.set_titles('{col_name} - {row_name}')
g.set_axis_labels('Total Bill ($)', 'Count')

plt.show()
```

---

## Hands-on Exercise

### Your Task

```python
# Using the 'tips' dataset:
# 1. Create a figure with 2x2 subplots:
#    - Top-left: Box plot of tips by day
#    - Top-right: Scatter of total_bill vs tip with regression line
#    - Bottom-left: Distribution of total_bill by time
#    - Bottom-right: Heatmap of correlations
#
# 2. Use 'colorblind' palette
# 3. Save the figure
```

<details>
<summary>✅ Solution</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set style and palette
sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')

tips = sns.load_dataset('tips')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Box plot of tips by day
sns.boxplot(data=tips, x='day', y='tip', ax=axes[0, 0])
axes[0, 0].set_title('Tips by Day')

# 2. Scatter with regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('Tip vs Total Bill (with regression)')

# 3. Distribution by time
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, ax=axes[1, 0])
axes[1, 0].set_title('Total Bill Distribution by Time')

# 4. Correlation heatmap
numeric_tips = tips.select_dtypes(include='number')
corr = numeric_tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
fig.savefig('seaborn_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved as seaborn_analysis.png")
```
</details>

---

## Summary

✅ **`sns.set_theme()`** for consistent styling
✅ Use **`hue`** parameter to add color grouping
✅ **Distribution**: `histplot`, `kdeplot`, `displot`
✅ **Categorical**: `boxplot`, `violinplot`, `barplot`, `catplot`
✅ **Relational**: `scatterplot`, `lineplot`, `relplot`
✅ **Statistical**: `regplot`, `lmplot`, `pairplot`, `heatmap`
✅ Use `colorblind` palette for accessibility

**Next:** [ML Visualization](./04-ml-visualization.md)

---

## Further Reading

- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Example Gallery](https://seaborn.pydata.org/examples/index.html)

<!-- 
Sources Consulted:
- Seaborn Docs: https://seaborn.pydata.org/tutorial.html
-->
