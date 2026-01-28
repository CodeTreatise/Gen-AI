---
title: "Interactive Visualization with Plotly"
---

# Interactive Visualization with Plotly

## Introduction

Plotly creates interactive, web-based visualizations. Users can zoom, pan, hover for details, and explore data dynamically—perfect for dashboards and Jupyter notebooks.

### What We'll Cover

- Plotly Express basics
- Interactive charts
- Hover information
- Animations
- Dashboard concepts

### Prerequisites

- Basic visualization concepts
- Pandas DataFrames

---

## Getting Started

### Installation

```bash
pip install plotly
```

### Plotly Express

```python
import plotly.express as px
import pandas as pd

# Sample data
df = px.data.tips()

# Simple interactive scatter
fig = px.scatter(df, x='total_bill', y='tip', color='day')
fig.show()
```

---

## Basic Charts

### Scatter Plot

```python
import plotly.express as px

df = px.data.iris()

fig = px.scatter(
    df,
    x='sepal_width',
    y='sepal_length',
    color='species',
    size='petal_length',
    hover_name='species',
    title='Iris Dataset'
)
fig.show()
```

### Line Chart

```python
import plotly.express as px
import pandas as pd
import numpy as np

# Create time series
dates = pd.date_range('2024-01-01', periods=100)
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100).cumsum()
})

fig = px.line(
    df,
    x='date',
    y='value',
    title='Time Series'
)
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Value'
)
fig.show()
```

### Bar Chart

```python
import plotly.express as px

df = px.data.tips()

# Grouped bar chart
fig = px.bar(
    df,
    x='day',
    y='total_bill',
    color='sex',
    barmode='group',
    title='Total Bill by Day and Gender'
)
fig.show()
```

### Histogram

```python
import plotly.express as px

df = px.data.tips()

fig = px.histogram(
    df,
    x='total_bill',
    color='time',
    nbins=30,
    title='Distribution of Total Bill',
    marginal='box'  # Add box plot on margin
)
fig.show()
```

---

## Advanced Charts

### Box Plot

```python
import plotly.express as px

df = px.data.tips()

fig = px.box(
    df,
    x='day',
    y='total_bill',
    color='smoker',
    title='Total Bill Distribution by Day'
)
fig.show()
```

### Heatmap

```python
import plotly.express as px
import pandas as pd
import numpy as np

# Create correlation matrix
df = px.data.iris()
numeric_cols = df.select_dtypes(include='number')
corr = numeric_cols.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='RdBu',
    title='Correlation Heatmap'
)
fig.show()
```

### Pie Chart

```python
import plotly.express as px

df = px.data.tips()
day_counts = df['day'].value_counts().reset_index()
day_counts.columns = ['day', 'count']

fig = px.pie(
    day_counts,
    values='count',
    names='day',
    title='Tips by Day of Week'
)
fig.show()
```

### 3D Scatter

```python
import plotly.express as px

df = px.data.iris()

fig = px.scatter_3d(
    df,
    x='sepal_length',
    y='sepal_width',
    z='petal_length',
    color='species',
    title='3D Iris Dataset'
)
fig.show()
```

---

## Customization

### Layout and Styling

```python
import plotly.express as px

df = px.data.tips()

fig = px.scatter(df, x='total_bill', y='tip', color='time')

fig.update_layout(
    title={
        'text': 'Tip vs Total Bill',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='Total Bill ($)',
    yaxis_title='Tip ($)',
    legend_title='Time of Day',
    template='plotly_white',  # Clean theme
    width=800,
    height=500
)

fig.show()
```

### Hover Information

```python
import plotly.express as px

df = px.data.gapminder().query("year == 2007")

fig = px.scatter(
    df,
    x='gdpPercap',
    y='lifeExp',
    size='pop',
    color='continent',
    hover_name='country',
    hover_data={
        'gdpPercap': ':.2f',
        'lifeExp': ':.1f',
        'pop': ':,.0f'
    },
    log_x=True,
    title='World Development 2007'
)

fig.show()
```

### Custom Colors

```python
import plotly.express as px

df = px.data.tips()

# Custom color sequence
fig = px.bar(
    df,
    x='day',
    y='total_bill',
    color='sex',
    color_discrete_map={
        'Male': '#1f77b4',
        'Female': '#ff7f0e'
    },
    title='Custom Colors'
)

fig.show()
```

---

## Subplots

### Multiple Charts

```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = px.data.tips()

fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))

# Add histogram
fig.add_trace(
    go.Histogram(x=df['total_bill'], name='Total Bill'),
    row=1, col=1
)

# Add box plot
fig.add_trace(
    go.Box(y=df['total_bill'], name='Total Bill'),
    row=1, col=2
)

fig.update_layout(title_text='Total Bill Analysis', showlegend=False)
fig.show()
```

### Faceted Plots

```python
import plotly.express as px

df = px.data.tips()

fig = px.scatter(
    df,
    x='total_bill',
    y='tip',
    color='smoker',
    facet_col='time',
    facet_row='sex',
    title='Faceted Scatter Plot'
)

fig.show()
```

---

## Animations

### Animated Scatter

```python
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter(
    df,
    x='gdpPercap',
    y='lifeExp',
    animation_frame='year',
    animation_group='country',
    size='pop',
    color='continent',
    hover_name='country',
    log_x=True,
    size_max=55,
    range_x=[100, 100000],
    range_y=[25, 90],
    title='Gapminder Animation'
)

fig.show()
```

### Animated Bar Chart

```python
import plotly.express as px

df = px.data.gapminder()
df_continent = df.groupby(['continent', 'year'])['pop'].sum().reset_index()

fig = px.bar(
    df_continent,
    x='continent',
    y='pop',
    animation_frame='year',
    color='continent',
    title='Population by Continent Over Time'
)

fig.update_layout(yaxis_range=[0, 5e9])
fig.show()
```

---

## Saving Figures

### Export Options

```python
import plotly.express as px

df = px.data.tips()
fig = px.scatter(df, x='total_bill', y='tip')

# Save as HTML (interactive)
fig.write_html('scatter.html')

# Save as static image (requires kaleido)
# pip install -U kaleido
fig.write_image('scatter.png', scale=2)
fig.write_image('scatter.svg')
fig.write_image('scatter.pdf')

print("Saved scatter.html and images")
```

---

## Dashboard Concepts

### Simple Dashboard Layout

```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = px.data.tips()

# Create dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribution', 'By Day', 'By Time', 'Scatter'),
    specs=[
        [{'type': 'histogram'}, {'type': 'bar'}],
        [{'type': 'pie'}, {'type': 'scatter'}]
    ]
)

# Histogram
fig.add_trace(
    go.Histogram(x=df['total_bill'], name='Bills'),
    row=1, col=1
)

# Bar by day
day_avg = df.groupby('day')['total_bill'].mean()
fig.add_trace(
    go.Bar(x=day_avg.index, y=day_avg.values, name='Avg Bill'),
    row=1, col=2
)

# Pie by time
time_counts = df['time'].value_counts()
fig.add_trace(
    go.Pie(labels=time_counts.index, values=time_counts.values, name='Time'),
    row=2, col=1
)

# Scatter
fig.add_trace(
    go.Scatter(x=df['total_bill'], y=df['tip'], mode='markers', name='Tips'),
    row=2, col=2
)

fig.update_layout(
    height=700,
    title_text='Tips Dataset Dashboard',
    showlegend=False
)

fig.show()
```

---

## Hands-on Exercise

### Your Task

```python
# Create an interactive visualization:
# 1. Load the gapminder dataset
# 2. Create a scatter plot of GDP vs Life Expectancy for 2007
# 3. Size by population, color by continent
# 4. Add custom hover info showing country, GDP, life expectancy, population
# 5. Add animation by year
# 6. Save as HTML
```

<details>
<summary>✅ Solution</summary>

```python
import plotly.express as px

# Load data
df = px.data.gapminder()

# Create animated scatter
fig = px.scatter(
    df,
    x='gdpPercap',
    y='lifeExp',
    size='pop',
    color='continent',
    hover_name='country',
    hover_data={
        'gdpPercap': ':$,.0f',
        'lifeExp': ':.1f years',
        'pop': ':,.0f',
        'continent': True
    },
    animation_frame='year',
    animation_group='country',
    log_x=True,
    size_max=60,
    range_x=[100, 150000],
    range_y=[20, 95],
    title='Global Development: GDP vs Life Expectancy'
)

# Customize layout
fig.update_layout(
    xaxis_title='GDP per Capita (log scale)',
    yaxis_title='Life Expectancy (years)',
    legend_title='Continent',
    template='plotly_white'
)

# Show
fig.show()

# Save as HTML
fig.write_html('gapminder_interactive.html')
print("Saved as gapminder_interactive.html")
```
</details>

---

## Summary

✅ **Plotly Express** for quick, high-level charts
✅ Charts are **interactive** by default (zoom, pan, hover)
✅ Use **`hover_data`** and **`hover_name`** for tooltips
✅ **`animation_frame`** creates animated visualizations
✅ **`make_subplots`** for multi-chart dashboards
✅ Export with **`write_html()`** or **`write_image()`**

**Next:** [Best Practices](./06-best-practices.md)

---

## Further Reading

- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Plotly Graph Objects](https://plotly.com/python/graph-objects/)
- [Dash for Dashboards](https://dash.plotly.com/)

<!-- 
Sources Consulted:
- Plotly Docs: https://plotly.com/python/
-->
