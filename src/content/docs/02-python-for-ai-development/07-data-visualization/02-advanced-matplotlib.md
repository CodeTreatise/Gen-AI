---
title: "Advanced Matplotlib"
---

# Advanced Matplotlib

## Introduction

Beyond basic plots, matplotlib offers powerful features for creating complex, publication-quality visualizations. This lesson covers subplots, annotations, styling, and advanced customization.

### What We'll Cover

- Multiple subplots
- Annotations and text
- Colormaps and colorbars
- Style sheets
- Twin axes
- 3D plotting basics

### Prerequisites

- Matplotlib fundamentals

---

## Subplots

### Basic Subplots

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('tan(x)')
axes[1, 0].set_ylim(-5, 5)

axes[1, 1].plot(x, x**2)
axes[1, 1].set_title('x²')

plt.tight_layout()
plt.show()
```

### Shared Axes

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# Share x-axis
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(x, np.sin(x))
axes[0].set_ylabel('sin(x)')

axes[1].plot(x, np.cos(x))
axes[1].set_ylabel('cos(x)')

axes[2].plot(x, np.sin(x) + np.cos(x))
axes[2].set_ylabel('sin + cos')
axes[2].set_xlabel('x')

plt.tight_layout()
plt.show()
```

### GridSpec for Complex Layouts

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Large plot spanning 2 columns
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(np.random.randn(100).cumsum())
ax1.set_title('Main Plot')

# Side plot
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(['A', 'B', 'C'], [3, 5, 2])
ax2.set_title('Side')

# Bottom row
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(np.random.randn(50), np.random.randn(50))

ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(np.random.randn(200), bins=20)

ax5 = fig.add_subplot(gs[1, 2])
ax5.pie([30, 40, 30], labels=['A', 'B', 'C'])

plt.tight_layout()
plt.show()
```

---

## Annotations and Text

### Adding Text

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

# Simple text
ax.text(5, 0.5, 'Peak Region', fontsize=12, ha='center')

# Text with box
ax.text(2, -0.5, 'Info Box', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.show()
```

### Annotations with Arrows

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

# Annotate maximum
max_idx = np.argmax(y)
ax.annotate('Maximum',
            xy=(x[max_idx], y[max_idx]),      # Point to annotate
            xytext=(x[max_idx] + 1, y[max_idx] + 0.3),  # Text position
            fontsize=12,
            arrowprops=dict(arrowstyle='->', color='red'))

# Annotate minimum
min_idx = np.argmin(y)
ax.annotate('Minimum',
            xy=(x[min_idx], y[min_idx]),
            xytext=(x[min_idx] + 1, y[min_idx] - 0.3),
            fontsize=12,
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.show()
```

### Horizontal and Vertical Lines

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

# Horizontal line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Vertical line
ax.axvline(x=np.pi, color='red', linestyle='--', label='x = π')

# Shaded region
ax.axvspan(0, np.pi/2, alpha=0.2, color='green', label='First quarter')

ax.legend()
plt.show()
```

---

## Colormaps and Colorbars

### Using Colormaps

```python
import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Different colormaps
cmaps = ['viridis', 'plasma', 'coolwarm']
for ax, cmap in zip(axes, cmaps):
    im = ax.imshow(Z, cmap=cmap, extent=[-5, 5, -5, 5])
    ax.set_title(f'cmap: {cmap}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### Common Colormaps

| Type | Colormaps |
|------|-----------|
| Sequential | viridis, plasma, inferno, magma |
| Diverging | coolwarm, RdBu, seismic |
| Qualitative | Set1, Set2, tab10 |
| Perceptually Uniform | viridis (default, best choice) |

### Heatmap with Colorbar

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(10, 10)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='RdBu', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value')

ax.set_title('Heatmap')
plt.show()
```

---

## Style Sheets

### Using Built-in Styles

```python
import matplotlib.pyplot as plt
import numpy as np

# List available styles
print(plt.style.available)

# Use a style
plt.style.use('seaborn-v0_8-darkgrid')  # or 'ggplot', 'dark_background', etc.

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title('With Style')
plt.show()

# Reset to default
plt.style.use('default')
```

### Common Styles

| Style | Description |
|-------|-------------|
| `default` | Matplotlib default |
| `seaborn-v0_8` | Clean, modern |
| `ggplot` | R ggplot2-like |
| `dark_background` | Dark theme |
| `fivethirtyeight` | FiveThirtyEight-style |

### Temporary Style Context

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# Use style only for this block
with plt.style.context('ggplot'):
    plt.figure()
    plt.plot(x, np.sin(x))
    plt.title('ggplot style')
    plt.show()

# Back to default outside the block
```

---

## Twin Axes

### Secondary Y-axis

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x / 10)  # Different scale

fig, ax1 = plt.subplots(figsize=(10, 6))

# First axis
color1 = 'tab:blue'
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)', color=color1)
ax1.plot(x, y1, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# Second axis (shared x)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('exp(x/10)', color=color2)
ax2.plot(x, y2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Dual Y-Axis')
plt.tight_layout()
plt.show()
```

---

## 3D Plotting

### Basic 3D Plot

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Surface plot
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface')
plt.show()
```

### 3D Scatter

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Random 3D points
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
colors = np.random.rand(n)

ax.scatter(x, y, z, c=colors, cmap='viridis', s=50)
ax.set_title('3D Scatter')
plt.show()
```

---

## Figure-Level Customization

### Figure Size and DPI

```python
import matplotlib.pyplot as plt
import numpy as np

# High-resolution figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

plt.show()
```

### Adjusting Spacing

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax in axes.flat:
    ax.plot(np.random.randn(100))

# Adjust spacing
plt.subplots_adjust(
    left=0.1,
    right=0.95,
    top=0.95,
    bottom=0.05,
    wspace=0.2,  # width space
    hspace=0.3   # height space
)

# Or use tight_layout
# plt.tight_layout()

plt.show()
```

---

## Hands-on Exercise

### Your Task

```python
# Create a 2x2 subplot figure showing:
# 1. Top-left: Line plot with annotation at peak
# 2. Top-right: Scatter with colorbar
# 3. Bottom-left: Bar chart
# 4. Bottom-right: Histogram with vertical mean line
#
# Use a style sheet and save at 300 DPI
```

<details>
<summary>✅ Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Line plot with annotation
x = np.linspace(0, 10, 100)
y = np.sin(x)
axes[0, 0].plot(x, y, 'b-', linewidth=2)
peak_idx = np.argmax(y)
axes[0, 0].annotate('Peak',
                     xy=(x[peak_idx], y[peak_idx]),
                     xytext=(x[peak_idx] + 1, y[peak_idx] + 0.2),
                     arrowprops=dict(arrowstyle='->'))
axes[0, 0].set_title('Line Plot with Annotation')

# 2. Scatter with colorbar
scatter_x = np.random.randn(50)
scatter_y = np.random.randn(50)
colors = np.random.rand(50)
sc = axes[0, 1].scatter(scatter_x, scatter_y, c=colors, cmap='viridis', s=100)
plt.colorbar(sc, ax=axes[0, 1])
axes[0, 1].set_title('Scatter with Colorbar')

# 3. Bar chart
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]
axes[1, 0].bar(categories, values, color='coral', edgecolor='black')
axes[1, 0].set_title('Bar Chart')

# 4. Histogram with mean line
data = np.random.randn(500)
axes[1, 1].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(data.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {data.mean():.2f}')
axes[1, 1].legend()
axes[1, 1].set_title('Histogram with Mean')

plt.tight_layout()
fig.savefig('advanced_subplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved as advanced_subplot.png")
```
</details>

---

## Summary

✅ Use **`plt.subplots(rows, cols)`** for multiple plots
✅ **`GridSpec`** for complex layouts
✅ **`ax.annotate()`** for arrows and annotations
✅ **Colormaps** visualize continuous data; use 'viridis' by default
✅ **`plt.style.use()`** for consistent styling
✅ **`ax.twinx()`** for secondary y-axis

**Next:** [Seaborn](./03-seaborn.md)

---

## Further Reading

- [Subplots](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
- [Annotations](https://matplotlib.org/stable/tutorials/text/annotations.html)
- [Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

<!-- 
Sources Consulted:
- Matplotlib Docs: https://matplotlib.org/stable/gallery/index.html
-->
