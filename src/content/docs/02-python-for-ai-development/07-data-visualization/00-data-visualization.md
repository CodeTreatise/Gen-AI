---
title: "Data Visualization"
---

# Data Visualization

## Overview

Data visualization is essential for AI/ML development—from exploratory data analysis (EDA) to model evaluation and results presentation. This lesson covers matplotlib, seaborn, and interactive visualization tools.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-matplotlib-fundamentals.md) | Matplotlib Fundamentals | Figures, axes, basic plots |
| [02](./02-advanced-matplotlib.md) | Advanced Matplotlib | Subplots, annotations, styling |
| [03](./03-seaborn.md) | Seaborn | Statistical graphics, themes |
| [04](./04-ml-visualization.md) | ML Visualization | Confusion matrices, ROC curves |
| [05](./05-interactive-plotly.md) | Interactive Plotly | Interactive charts, dashboards |
| [06](./06-best-practices.md) | Best Practices | Chart selection, accessibility |

---

## Why Visualization Matters

| Stage | Visualization Purpose |
|-------|----------------------|
| **EDA** | Understand distributions, relationships |
| **Feature Engineering** | Identify patterns, outliers |
| **Model Training** | Track loss curves, learning |
| **Evaluation** | Confusion matrices, ROC curves |
| **Communication** | Present results to stakeholders |

---

## Quick Start

```python
import matplotlib.pyplot as plt
import numpy as np

# Create simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()
```

---

## Installation

```bash
pip install matplotlib seaborn plotly
```

```python
import matplotlib
import seaborn as sns
import plotly

print(matplotlib.__version__)  # 3.x
print(sns.__version__)         # 0.13.x
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- NumPy and Pandas basics
- Understanding of statistical concepts helpful

---

## Start Learning

Begin with [Matplotlib Fundamentals](./01-matplotlib-fundamentals.md) to learn the foundation of Python visualization.

---

## Further Reading

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Plotly Python](https://plotly.com/python/)
