---
title: "Matplotlib Fundamentals"
---

# Matplotlib Fundamentals

## Introduction

Matplotlib is Python's foundational plotting library. Nearly every other visualization library builds on it. Understanding matplotlib gives you complete control over your visualizations.

### What We'll Cover

- Figure and Axes objects
- Line, scatter, bar, and histogram plots
- Customizing plots
- Adding titles, labels, legends
- Saving figures

### Prerequisites

- NumPy basics
- Pandas helpful

---

## Figure and Axes

### The Object-Oriented Interface

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot on the axes
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# Customize
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Sine Wave')

plt.show()
```

### Figure vs Axes

```python
import matplotlib.pyplot as plt

# Figure: the entire window/canvas
# Axes: the actual plot area (can have multiple per figure)

fig, ax = plt.subplots()
print(type(fig))  # <class 'matplotlib.figure.Figure'>
print(type(ax))   # <class 'matplotlib.axes._subplots.AxesSubplot'>
```

### The pyplot Interface

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple pyplot interface (good for quick plots)
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Note: Object-oriented is recommended for complex plots
```

---

## Line Plots

### Basic Line Plot

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
plt.show()
```

### Multiple Lines

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.plot(x, np.sin(x) + np.cos(x), label='sin(x) + cos(x)')

ax.legend()
ax.set_title('Trigonometric Functions')
plt.show()
```

### Line Styles and Colors

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)

fig, ax = plt.subplots(figsize=(10, 6))

# Different styles
ax.plot(x, x, 'r-', label='solid red')        # color-style shorthand
ax.plot(x, x + 2, 'b--', label='dashed blue')  
ax.plot(x, x + 4, 'g-.', label='dash-dot green')
ax.plot(x, x + 6, 'k:', label='dotted black')

# Explicit parameters
ax.plot(x, x + 8, color='purple', linestyle='-', linewidth=2, label='thick purple')

ax.legend()
plt.show()
```

### Common Line Style Options

| Parameter | Options |
|-----------|---------|
| `color` | 'r', 'g', 'b', 'k', '#FF5733', (0.1, 0.2, 0.5) |
| `linestyle` | '-', '--', '-.', ':', 'None' |
| `linewidth` | 0.5, 1, 2, 3... |
| `marker` | 'o', 's', '^', 'v', 'd', '*' |

---

## Scatter Plots

### Basic Scatter

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot')
plt.show()
```

### Customizing Points

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
colors = np.random.rand(n)  # Color by value
sizes = np.abs(np.random.randn(n)) * 200  # Size by value

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')

# Add colorbar
plt.colorbar(scatter, label='Color Value')
ax.set_title('Scatter with Colors and Sizes')
plt.show()
```

---

## Bar Charts

### Vertical Bar Chart

```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(categories, values, color='steelblue', edgecolor='black')

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Bar Chart')
plt.show()
```

### Horizontal Bar Chart

```python
import matplotlib.pyplot as plt

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 45, 56, 78]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(categories, values, color='coral')

ax.set_xlabel('Value')
ax.set_title('Horizontal Bar Chart')
plt.show()
```

### Grouped Bar Chart

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Q1', 'Q2', 'Q3', 'Q4']
sales_2023 = [100, 150, 130, 180]
sales_2024 = [120, 160, 140, 200]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, sales_2023, width, label='2023', color='steelblue')
ax.bar(x + width/2, sales_2024, width, label='2024', color='coral')

ax.set_xlabel('Quarter')
ax.set_ylabel('Sales')
ax.set_title('Quarterly Sales Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.show()
```

---

## Histograms

### Basic Histogram

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=30, edgecolor='black')

ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Normal Distribution')
plt.show()
```

### Multiple Histograms

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data1 = np.random.randn(1000)
data2 = np.random.randn(1000) + 2

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data1, bins=30, alpha=0.7, label='Group 1', edgecolor='black')
ax.hist(data2, bins=30, alpha=0.7, label='Group 2', edgecolor='black')

ax.legend()
ax.set_title('Comparing Distributions')
plt.show()
```

---

## Customizing Plots

### Titles, Labels, Legends

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# Title and labels
ax.set_title('Trigonometric Functions', fontsize=16, fontweight='bold')
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)

# Legend
ax.legend(loc='upper right', fontsize=10)

# Grid
ax.grid(True, linestyle='--', alpha=0.7)

plt.show()
```

### Axis Limits and Ticks

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x))

# Set axis limits
ax.set_xlim(0, 8)
ax.set_ylim(-1.5, 1.5)

# Custom ticks
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xticklabels(['zero', 'two', 'four', 'six', 'eight'])

plt.show()
```

---

## Saving Figures

### Save to File

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x))
ax.set_title('Sine Wave')

# Save as PNG (high resolution)
fig.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Save as SVG (vector format)
fig.savefig('sine_wave.svg', bbox_inches='tight')

# Save as PDF
fig.savefig('sine_wave.pdf', bbox_inches='tight')

plt.close()  # Close to free memory
```

### Save Options

| Parameter | Description |
|-----------|-------------|
| `dpi` | Resolution (300 for print) |
| `bbox_inches='tight'` | Remove whitespace |
| `facecolor` | Background color |
| `transparent=True` | Transparent background |

---

## Hands-on Exercise

### Your Task

```python
# Create a figure with:
# 1. Line plot of y = x^2 for x from -5 to 5
# 2. Add horizontal line at y = 10
# 3. Mark the intersection points
# 4. Add title, labels, legend, grid
# 5. Save as 'quadratic.png' at 300 dpi
```

<details>
<summary>✅ Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = x ** 2

fig, ax = plt.subplots(figsize=(10, 6))

# Line plot
ax.plot(x, y, 'b-', linewidth=2, label='y = x²')

# Horizontal line
ax.axhline(y=10, color='r', linestyle='--', label='y = 10')

# Intersection points (x² = 10, so x = ±√10)
intersection_x = np.sqrt(10)
ax.plot([intersection_x, -intersection_x], [10, 10], 'go', 
        markersize=10, label='Intersections')

# Customization
ax.set_title('Quadratic Function', fontsize=16, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(loc='upper center')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 30)

# Save
fig.savefig('quadratic.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved as quadratic.png")
```
</details>

---

## Summary

✅ Use **`fig, ax = plt.subplots()`** for the OO interface
✅ **`ax.plot()`** for lines, **`ax.scatter()`** for points
✅ **`ax.bar()`** for bars, **`ax.hist()`** for histograms
✅ Customize with **`set_title()`**, **`set_xlabel()`**, **`legend()`**
✅ Save with **`fig.savefig()`**, use `dpi=300` for print quality
✅ Close figures with **`plt.close()`** to free memory

**Next:** [Advanced Matplotlib](./02-advanced-matplotlib.md)

---

## Further Reading

- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Plot Types Gallery](https://matplotlib.org/stable/gallery/index.html)

<!-- 
Sources Consulted:
- Matplotlib Docs: https://matplotlib.org/stable/tutorials/introductory/usage.html
-->
