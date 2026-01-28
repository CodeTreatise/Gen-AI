---
title: "Visualization Best Practices"
---

# Visualization Best Practices

## Introduction

Great visualizations communicate clearly, are accessible to all viewers, and present data honestly. This lesson covers principles that make the difference between good and excellent data visualization.

### What We'll Cover

- Choosing the right chart type
- Color accessibility
- Publication-quality figures
- Common mistakes to avoid
- Export best practices

### Prerequisites

- Matplotlib, Seaborn, or Plotly basics

---

## Choosing the Right Chart

### Chart Selection Guide

| Data Type | Goal | Chart Type |
|-----------|------|------------|
| **Distribution** | Show spread | Histogram, KDE, Box, Violin |
| **Comparison** | Compare values | Bar, Grouped Bar |
| **Relationship** | Show correlation | Scatter, Heatmap |
| **Trend** | Show change over time | Line |
| **Composition** | Show parts of whole | Pie, Stacked Bar |
| **Ranking** | Order by value | Horizontal Bar |

### When to Use What

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate sample data
np.random.seed(42)
data = {
    'distribution': np.random.randn(1000),
    'categories': ['A', 'B', 'C', 'D'],
    'values': [25, 40, 30, 55],
    'x': np.random.randn(100),
    'y': np.random.randn(100)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution → Histogram
axes[0, 0].hist(data['distribution'], bins=30, edgecolor='black')
axes[0, 0].set_title('Distribution → Histogram')

# Comparison → Bar
axes[0, 1].bar(data['categories'], data['values'])
axes[0, 1].set_title('Comparison → Bar')

# Relationship → Scatter
axes[1, 0].scatter(data['x'], data['y'])
axes[1, 0].set_title('Relationship → Scatter')

# Composition → Pie
axes[1, 1].pie(data['values'], labels=data['categories'], autopct='%1.1f%%')
axes[1, 1].set_title('Composition → Pie')

plt.tight_layout()
plt.show()
```

### Bad Chart Choices

| ❌ Avoid | ✅ Use Instead |
|---------|----------------|
| Pie chart with >5 categories | Horizontal bar chart |
| 3D bar charts | 2D bar charts |
| Dual y-axis with different scales | Two separate charts |
| Truncated y-axis (misleading) | Start y-axis at zero |
| Too many lines (>5-6) | Faceted/small multiples |

---

## Color Accessibility

### Colorblind-Friendly Palettes

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Accessible palettes
fig, axes = plt.subplots(1, 3, figsize=(15, 2))

palettes = ['colorblind', 'deep', 'muted']
for ax, palette in zip(axes, palettes):
    sns.palplot(sns.color_palette(palette), ax=ax)
    ax.set_title(palette)

plt.tight_layout()
plt.show()
```

### Good Color Practices

```python
import matplotlib.pyplot as plt
import numpy as np

# ✅ Good: Use distinct colors + patterns/markers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 10, 50)

# With colors AND markers
ax = axes[0]
ax.plot(x, np.sin(x), 'o-', color='#0072B2', label='sin', markersize=4)
ax.plot(x, np.cos(x), 's--', color='#D55E00', label='cos', markersize=4)
ax.plot(x, np.sin(x) + 0.5, '^:', color='#009E73', label='sin+0.5', markersize=4)
ax.legend()
ax.set_title('✅ Colors + Markers (Accessible)')

# ❌ Bad: Similar colors only
ax = axes[1]
ax.plot(x, np.sin(x), color='#1f77b4', label='sin')
ax.plot(x, np.cos(x), color='#17becf', label='cos')
ax.plot(x, np.sin(x) + 0.5, color='#aec7e8', label='sin+0.5')
ax.legend()
ax.set_title('❌ Similar Colors Only (Hard to Distinguish)')

plt.tight_layout()
plt.show()
```

### Color Guidelines

| Rule | Reason |
|------|--------|
| Use 'colorblind' palette | ~8% of men are colorblind |
| Don't rely on color alone | Add patterns, markers, labels |
| Use sequential for ordered data | viridis, plasma |
| Use diverging for +/- values | coolwarm, RdBu |
| Limit to 5-7 colors | More becomes confusing |

---

## Publication-Quality Figures

### High-Resolution Export

```python
import matplotlib.pyplot as plt
import numpy as np

# Set publication defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150
})

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), linewidth=2)
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')
ax.set_title('Publication-Ready Figure')

# Save high-resolution
fig.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
fig.savefig('publication_figure.pdf', bbox_inches='tight')
fig.savefig('publication_figure.svg', bbox_inches='tight')

plt.show()
print("Saved in multiple formats")
```

### Sizing Guidelines

| Format | Recommended DPI |
|--------|-----------------|
| Screen/Web | 72-100 |
| Presentations | 150 |
| Print | 300+ |
| Publication | 300-600 |

### Figure Size for Journals

```python
import matplotlib.pyplot as plt

# Common journal column widths
single_column = (3.5, 2.5)   # inches
double_column = (7.0, 4.0)   # inches

fig, ax = plt.subplots(figsize=single_column)
# ... your plot
fig.savefig('single_column.pdf', bbox_inches='tight')
```

---

## Clear Labels and Annotations

### Effective Labeling

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 10, 50)
y = x ** 2

# ❌ Bad: Missing context
axes[0].plot(x, y)
axes[0].set_title('Bad: No Labels')

# ✅ Good: Clear labels with units
axes[1].plot(x, y)
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Distance (meters)')
axes[1].set_title('Good: Clear Labels with Units')
axes[1].annotate('Acceleration point', 
                  xy=(5, 25), xytext=(7, 15),
                  arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.show()
```

### Label Best Practices

| Element | Guideline |
|---------|-----------|
| **Title** | Descriptive, not just variable names |
| **Axis labels** | Include units in parentheses |
| **Legend** | Positioned outside data area |
| **Annotations** | Highlight key insights |
| **Font size** | Readable at final display size |

---

## Avoiding Common Mistakes

### Mistake 1: Misleading Axes

```python
import matplotlib.pyplot as plt
import numpy as np

data = [100, 102, 105, 103, 108]
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ❌ Bad: Truncated y-axis (misleading)
axes[0].bar(labels, data)
axes[0].set_ylim(95, 110)
axes[0].set_title('❌ Misleading: Truncated Y-axis')

# ✅ Good: Full y-axis
axes[1].bar(labels, data)
axes[1].set_ylim(0, 120)
axes[1].set_title('✅ Honest: Full Y-axis')

plt.tight_layout()
plt.show()
```

### Mistake 2: Chart Junk

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 45]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ❌ Bad: Too much decoration
ax = axes[0]
ax.bar(categories, values, color=['red', 'green', 'blue', 'orange'],
       edgecolor='black', linewidth=2)
ax.set_facecolor('lightgray')
ax.grid(True, color='white', linewidth=2)
for i, v in enumerate(values):
    ax.text(i, v + 1, str(v), ha='center', fontsize=14, fontweight='bold')
ax.set_title('❌ Chart Junk: Too Busy', fontsize=16)

# ✅ Good: Clean and minimal
ax = axes[1]
ax.bar(categories, values, color='steelblue')
ax.set_title('✅ Clean: Focus on Data')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### Common Mistakes Checklist

| ❌ Mistake | ✅ Fix |
|-----------|--------|
| 3D charts for 2D data | Use 2D charts |
| Rainbow color maps | Use perceptually uniform (viridis) |
| Pie charts with many slices | Horizontal bar chart |
| Missing legends | Add clear legends |
| Overlapping labels | Rotate or abbreviate |
| Low contrast | Increase contrast |

---

## Responsive Design

### Adjusting for Different Outputs

```python
import matplotlib.pyplot as plt
import numpy as np

def create_plot(figsize, fontscale=1.0):
    """Create plot adjusted for size."""
    plt.rcParams.update({
        'font.size': 10 * fontscale,
        'axes.labelsize': 10 * fontscale,
        'axes.titlesize': 12 * fontscale,
        'xtick.labelsize': 8 * fontscale,
        'ytick.labelsize': 8 * fontscale,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), linewidth=2)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Responsive Plot')
    return fig

# Different sizes
fig_small = create_plot((4, 3), fontscale=1.0)
fig_small.savefig('plot_small.png', dpi=150, bbox_inches='tight')

fig_large = create_plot((10, 6), fontscale=1.2)
fig_large.savefig('plot_large.png', dpi=100, bbox_inches='tight')

plt.show()
```

---

## Export Formats

### When to Use Each Format

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **PNG** | Web, presentations | Universal, supports transparency | Fixed resolution |
| **SVG** | Web, scaling | Vector, infinite scaling | Larger file size |
| **PDF** | Publications, print | Vector, high quality | Not for web |
| **EPS** | LaTeX documents | Vector, publication standard | Limited software |

### Export Code

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.random.randn(100).cumsum())
ax.set_title('Export Example')

# Web/Screen
fig.savefig('chart_web.png', dpi=100, bbox_inches='tight')

# Presentation
fig.savefig('chart_presentation.png', dpi=150, bbox_inches='tight')

# Publication
fig.savefig('chart_publication.png', dpi=300, bbox_inches='tight')
fig.savefig('chart_publication.pdf', bbox_inches='tight')
fig.savefig('chart_publication.svg', bbox_inches='tight')

plt.close()
print("Exported in multiple formats")
```

---

## Hands-on Exercise

### Your Task

```python
# Take this poorly designed chart and fix all the issues:

import matplotlib.pyplot as plt
import numpy as np

# Bad chart
data = [23, 45, 32, 67, 54, 38, 49]
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(labels, data, color=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
ax.set_ylim(20, 70)
ax.set_title('data')
plt.show()

# Fix: truncated y-axis, rainbow colors, no labels, unclear title
# Export at publication quality
```

<details>
<summary>✅ Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

data = [23, 45, 32, 67, 54, 38, 49]
labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Set publication defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Use single color, add value labels
bars = ax.bar(labels, data, color='steelblue', edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, value in zip(bars, data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(value), ha='center', va='bottom', fontsize=10)

# ✅ Full y-axis starting from 0
ax.set_ylim(0, 80)

# ✅ Clear labels
ax.set_xlabel('Day of Week')
ax.set_ylabel('Daily Sales (units)')
ax.set_title('Weekly Sales Performance')

# ✅ Clean design
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# ✅ Publication quality export
fig.savefig('sales_chart_fixed.png', dpi=300, bbox_inches='tight')
fig.savefig('sales_chart_fixed.pdf', bbox_inches='tight')

plt.show()
print("Saved publication-quality versions")
```
</details>

---

## Summary

✅ **Choose the right chart** for your data type and goal
✅ Use **colorblind-friendly palettes** (colorblind, viridis)
✅ **Don't rely on color alone**—add markers, patterns, labels
✅ **Start y-axis at zero** to avoid misleading viewers
✅ **Remove chart junk**—focus on the data
✅ Export at **300+ DPI** for print, use **vector formats** for scaling

**Back to:** [Data Visualization Overview](./00-data-visualization.md)

---

## Further Reading

- [Data Visualization Catalogue](https://datavizcatalogue.com/)
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/)
- [ColorBrewer](https://colorbrewer2.org/) - Color advice for maps

<!-- 
Sources Consulted:
- Matplotlib Style Guide: https://matplotlib.org/stable/tutorials/introductory/customizing.html
-->
