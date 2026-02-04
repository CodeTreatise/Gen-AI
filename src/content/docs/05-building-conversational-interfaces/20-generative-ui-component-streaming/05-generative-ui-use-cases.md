---
title: "Generative UI Use Cases"
---

# Generative UI Use Cases

## Introduction

Generative UI shines when the AI needs to display structured, interactive content. This lesson showcases practical use cases with complete tool definitions and component implementations: dynamic forms, interactive charts, product cards, booking widgets, and data tables.

### What We'll Cover

- Dynamic forms generated from AI context
- Interactive charts and data visualizations
- Product cards and e-commerce previews
- Booking and scheduling widgets
- Data tables with sorting and actions

### Prerequisites

- [Interactive Generated Components](./04-interactive-generated-components.md)
- React form handling
- Basic charting concepts

---

## Dynamic Forms

AI can generate forms based on conversation context:

### Form Tool Definition

```typescript
// ai/tools.ts
import { tool } from 'ai';
import { z } from 'zod';

const fieldSchema = z.object({
  name: z.string(),
  label: z.string(),
  type: z.enum(['text', 'email', 'tel', 'number', 'select', 'textarea', 'date']),
  placeholder: z.string().optional(),
  required: z.boolean().default(false),
  options: z.array(z.object({
    value: z.string(),
    label: z.string(),
  })).optional(),
  validation: z.object({
    min: z.number().optional(),
    max: z.number().optional(),
    pattern: z.string().optional(),
  }).optional(),
});

export const tools = {
  showForm: tool({
    description: 'Display a form for user input',
    inputSchema: z.object({
      title: z.string().describe('Form title'),
      description: z.string().optional(),
      fields: z.array(fieldSchema),
      submitLabel: z.string().default('Submit'),
    }),
    // Client-side tool - no execute
  }),
};
```

### Dynamic Form Component

```tsx
// components/dynamic-form.tsx
import { useState } from 'react';

interface Field {
  name: string;
  label: string;
  type: 'text' | 'email' | 'tel' | 'number' | 'select' | 'textarea' | 'date';
  placeholder?: string;
  required?: boolean;
  options?: { value: string; label: string }[];
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
  };
}

interface DynamicFormProps {
  title: string;
  description?: string;
  fields: Field[];
  submitLabel: string;
  onSubmit: (data: Record<string, any>) => void;
}

export function DynamicForm({
  title,
  description,
  fields,
  submitLabel,
  onSubmit,
}: DynamicFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (name: string, value: any) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
    setErrors((prev) => ({ ...prev, [name]: '' }));
  };

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    fields.forEach((field) => {
      const value = formData[field.name];

      if (field.required && !value) {
        newErrors[field.name] = `${field.label} is required`;
      }

      if (field.validation?.pattern && value) {
        const regex = new RegExp(field.validation.pattern);
        if (!regex.test(value)) {
          newErrors[field.name] = `Invalid ${field.label.toLowerCase()} format`;
        }
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      onSubmit(formData);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="dynamic-form">
      <div className="form-header">
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </div>

      <div className="form-fields">
        {fields.map((field) => (
          <FormField
            key={field.name}
            field={field}
            value={formData[field.name] || ''}
            error={errors[field.name]}
            onChange={(value) => handleChange(field.name, value)}
          />
        ))}
      </div>

      <button type="submit" className="form-submit">
        {submitLabel}
      </button>
    </form>
  );
}

function FormField({
  field,
  value,
  error,
  onChange,
}: {
  field: Field;
  value: any;
  error?: string;
  onChange: (value: any) => void;
}) {
  const renderInput = () => {
    switch (field.type) {
      case 'select':
        return (
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className={error ? 'error' : ''}
          >
            <option value="">Select...</option>
            {field.options?.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        );

      case 'textarea':
        return (
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={field.placeholder}
            className={error ? 'error' : ''}
            rows={4}
          />
        );

      default:
        return (
          <input
            type={field.type}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={field.placeholder}
            className={error ? 'error' : ''}
            min={field.validation?.min}
            max={field.validation?.max}
          />
        );
    }
  };

  return (
    <div className="form-field">
      <label>
        {field.label}
        {field.required && <span className="required">*</span>}
      </label>
      {renderInput()}
      {error && <span className="field-error">{error}</span>}
    </div>
  );
}
```

```css
.dynamic-form {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 24px;
  max-width: 500px;
}

.form-header h3 {
  margin: 0 0 8px;
  font-size: 1.25rem;
}

.form-header p {
  margin: 0 0 20px;
  color: #64748b;
  font-size: 0.875rem;
}

.form-fields {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-field label {
  font-weight: 500;
  font-size: 0.875rem;
  color: #334155;
}

.required {
  color: #dc2626;
  margin-left: 2px;
}

.form-field input,
.form-field select,
.form-field textarea {
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.875rem;
}

.form-field input.error,
.form-field select.error,
.form-field textarea.error {
  border-color: #dc2626;
}

.field-error {
  color: #dc2626;
  font-size: 0.75rem;
}

.form-submit {
  margin-top: 20px;
  width: 100%;
  padding: 12px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
}
```

---

## Interactive Charts

Visualize data the AI retrieves or generates:

### Chart Tool Definition

```typescript
export const tools = {
  showChart: tool({
    description: 'Display a chart visualization',
    inputSchema: z.object({
      type: z.enum(['line', 'bar', 'pie', 'area']),
      title: z.string(),
      data: z.array(z.object({
        label: z.string(),
        value: z.number(),
        color: z.string().optional(),
      })),
      xLabel: z.string().optional(),
      yLabel: z.string().optional(),
    }),
    execute: async ({ type, title, data }) => {
      // Could fetch real data here
      return { type, title, data };
    },
  }),
};
```

### Chart Component (with Recharts)

```tsx
// components/interactive-chart.tsx
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface ChartData {
  label: string;
  value: number;
  color?: string;
}

interface InteractiveChartProps {
  type: 'line' | 'bar' | 'pie' | 'area';
  title: string;
  data: ChartData[];
  xLabel?: string;
  yLabel?: string;
  onDataClick?: (item: ChartData) => void;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export function InteractiveChart({
  type,
  title,
  data,
  xLabel,
  yLabel,
  onDataClick,
}: InteractiveChartProps) {
  const chartData = data.map((item, i) => ({
    name: item.label,
    value: item.value,
    fill: item.color || COLORS[i % COLORS.length],
  }));

  const renderChart = () => {
    switch (type) {
      case 'line':
        return (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" label={{ value: xLabel, position: 'bottom' }} />
            <YAxis label={{ value: yLabel, angle: -90, position: 'left' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', cursor: 'pointer' }}
              activeDot={{ r: 8 }}
              onClick={(_, index) => onDataClick?.(data[index])}
            />
          </LineChart>
        );

      case 'bar':
        return (
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar
              dataKey="value"
              cursor="pointer"
              onClick={(item) => onDataClick?.(data.find(d => d.label === item.name)!)}
            >
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        );

      case 'pie':
        return (
          <PieChart>
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label
              onClick={(_, index) => onDataClick?.(data[index])}
            >
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.fill} cursor="pointer" />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        );

      case 'area':
        return (
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              fill="#93c5fd"
            />
          </AreaChart>
        );
    }
  };

  return (
    <div className="interactive-chart">
      <h4 className="chart-title">{title}</h4>
      <ResponsiveContainer width="100%" height={300}>
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
}
```

```css
.interactive-chart {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  margin: 12px 0;
}

.chart-title {
  margin: 0 0 16px;
  font-size: 1rem;
  font-weight: 600;
  color: #334155;
}
```

---

## Product Cards

E-commerce product displays with actions:

### Product Tool Definition

```typescript
export const tools = {
  showProduct: tool({
    description: 'Display a product card',
    inputSchema: z.object({
      id: z.string(),
      name: z.string(),
      description: z.string(),
      price: z.number(),
      originalPrice: z.number().optional(),
      image: z.string(),
      rating: z.number().min(0).max(5),
      reviewCount: z.number(),
      inStock: z.boolean(),
      features: z.array(z.string()).optional(),
    }),
    execute: async ({ id }) => {
      // Fetch from product database
      return await getProduct(id);
    },
  }),

  showProductGrid: tool({
    description: 'Display multiple products in a grid',
    inputSchema: z.object({
      products: z.array(z.object({
        id: z.string(),
        name: z.string(),
        price: z.number(),
        image: z.string(),
        rating: z.number(),
      })),
      columns: z.number().default(3),
    }),
  }),
};
```

### Product Card Component

```tsx
// components/product-card.tsx
interface ProductCardProps {
  id: string;
  name: string;
  description: string;
  price: number;
  originalPrice?: number;
  image: string;
  rating: number;
  reviewCount: number;
  inStock: boolean;
  features?: string[];
  onAddToCart?: (id: string) => void;
  onAskQuestion?: (question: string) => void;
}

export function ProductCard({
  id,
  name,
  description,
  price,
  originalPrice,
  image,
  rating,
  reviewCount,
  inStock,
  features,
  onAddToCart,
  onAskQuestion,
}: ProductCardProps) {
  const discount = originalPrice
    ? Math.round((1 - price / originalPrice) * 100)
    : 0;

  return (
    <div className="product-card">
      <div className="product-image">
        <img src={image} alt={name} />
        {discount > 0 && (
          <span className="discount-badge">-{discount}%</span>
        )}
      </div>

      <div className="product-info">
        <h3 className="product-name">{name}</h3>
        <p className="product-description">{description}</p>

        <div className="product-rating">
          <Stars rating={rating} />
          <span className="review-count">({reviewCount} reviews)</span>
        </div>

        <div className="product-price">
          <span className="current-price">${price.toFixed(2)}</span>
          {originalPrice && (
            <span className="original-price">${originalPrice.toFixed(2)}</span>
          )}
        </div>

        {features && features.length > 0 && (
          <ul className="product-features">
            {features.slice(0, 3).map((feature, i) => (
              <li key={i}>✓ {feature}</li>
            ))}
          </ul>
        )}

        <div className="product-actions">
          <button
            className={`add-to-cart ${!inStock ? 'disabled' : ''}`}
            onClick={() => inStock && onAddToCart?.(id)}
            disabled={!inStock}
          >
            {inStock ? 'Add to Cart' : 'Out of Stock'}
          </button>
          
          {onAskQuestion && (
            <button
              className="ask-question"
              onClick={() => onAskQuestion(`Tell me more about ${name}`)}
            >
              Ask AI
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Stars({ rating }: { rating: number }) {
  return (
    <div className="stars">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className={star <= rating ? 'filled' : 'empty'}
        >
          ★
        </span>
      ))}
    </div>
  );
}
```

```css
.product-card {
  display: flex;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  overflow: hidden;
  background: white;
  max-width: 600px;
}

.product-image {
  position: relative;
  width: 200px;
  flex-shrink: 0;
}

.product-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.discount-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  background: #dc2626;
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.product-info {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.product-name {
  margin: 0;
  font-size: 1.125rem;
  color: #1e293b;
}

.product-description {
  margin: 0;
  color: #64748b;
  font-size: 0.875rem;
  line-height: 1.4;
}

.product-rating {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stars {
  color: #fbbf24;
}

.stars .empty {
  color: #e2e8f0;
}

.review-count {
  color: #94a3b8;
  font-size: 0.75rem;
}

.product-price {
  display: flex;
  align-items: center;
  gap: 8px;
}

.current-price {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1e293b;
}

.original-price {
  font-size: 1rem;
  color: #94a3b8;
  text-decoration: line-through;
}

.product-features {
  margin: 8px 0;
  padding: 0;
  list-style: none;
}

.product-features li {
  font-size: 0.75rem;
  color: #16a34a;
  margin: 4px 0;
}

.product-actions {
  display: flex;
  gap: 8px;
  margin-top: auto;
}

.add-to-cart {
  flex: 1;
  padding: 10px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
}

.add-to-cart.disabled {
  background: #94a3b8;
  cursor: not-allowed;
}

.ask-question {
  padding: 10px 16px;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  cursor: pointer;
}
```

---

## Booking Widget

Scheduling and reservation interfaces:

### Booking Tool Definition

```typescript
export const tools = {
  showBookingWidget: tool({
    description: 'Display a booking/scheduling widget',
    inputSchema: z.object({
      service: z.string(),
      provider: z.string().optional(),
      availableSlots: z.array(z.object({
        date: z.string(),
        times: z.array(z.string()),
      })),
      duration: z.number().describe('Duration in minutes'),
      price: z.number().optional(),
    }),
    // Client handles booking submission
  }),
};
```

### Booking Component

```tsx
// components/booking-widget.tsx
import { useState } from 'react';

interface TimeSlot {
  date: string;
  times: string[];
}

interface BookingWidgetProps {
  service: string;
  provider?: string;
  availableSlots: TimeSlot[];
  duration: number;
  price?: number;
  onBook: (booking: { date: string; time: string }) => void;
}

export function BookingWidget({
  service,
  provider,
  availableSlots,
  duration,
  price,
  onBook,
}: BookingWidgetProps) {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [selectedTime, setSelectedTime] = useState<string | null>(null);

  const selectedSlot = availableSlots.find((s) => s.date === selectedDate);

  const handleBook = () => {
    if (selectedDate && selectedTime) {
      onBook({ date: selectedDate, time: selectedTime });
    }
  };

  return (
    <div className="booking-widget">
      <div className="booking-header">
        <h3>📅 Book {service}</h3>
        {provider && <p>with {provider}</p>}
      </div>

      <div className="booking-dates">
        <h4>Select a Date</h4>
        <div className="date-grid">
          {availableSlots.map((slot) => (
            <button
              key={slot.date}
              className={`date-btn ${selectedDate === slot.date ? 'selected' : ''}`}
              onClick={() => {
                setSelectedDate(slot.date);
                setSelectedTime(null);
              }}
            >
              <span className="date-day">
                {new Date(slot.date).toLocaleDateString('en-US', { weekday: 'short' })}
              </span>
              <span className="date-num">
                {new Date(slot.date).getDate()}
              </span>
            </button>
          ))}
        </div>
      </div>

      {selectedSlot && (
        <div className="booking-times">
          <h4>Select a Time</h4>
          <div className="time-grid">
            {selectedSlot.times.map((time) => (
              <button
                key={time}
                className={`time-btn ${selectedTime === time ? 'selected' : ''}`}
                onClick={() => setSelectedTime(time)}
              >
                {time}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="booking-summary">
        <div className="summary-row">
          <span>Duration</span>
          <span>{duration} minutes</span>
        </div>
        {price && (
          <div className="summary-row">
            <span>Price</span>
            <span className="price">${price}</span>
          </div>
        )}
      </div>

      <button
        className="book-btn"
        disabled={!selectedDate || !selectedTime}
        onClick={handleBook}
      >
        {selectedDate && selectedTime
          ? `Book for ${selectedDate} at ${selectedTime}`
          : 'Select date and time'}
      </button>
    </div>
  );
}
```

```css
.booking-widget {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  max-width: 400px;
}

.booking-header h3 {
  margin: 0;
  font-size: 1.25rem;
}

.booking-header p {
  margin: 4px 0 0;
  color: #64748b;
}

.booking-dates,
.booking-times {
  margin-top: 20px;
}

.booking-dates h4,
.booking-times h4 {
  margin: 0 0 12px;
  font-size: 0.875rem;
  color: #475569;
}

.date-grid {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 8px;
}

.date-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 16px;
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  min-width: 60px;
}

.date-btn.selected {
  border-color: #2563eb;
  background: #eff6ff;
}

.date-day {
  font-size: 0.75rem;
  color: #64748b;
}

.date-num {
  font-size: 1.25rem;
  font-weight: 600;
}

.time-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.time-btn {
  padding: 10px;
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.875rem;
}

.time-btn.selected {
  border-color: #2563eb;
  background: #2563eb;
  color: white;
}

.booking-summary {
  margin-top: 20px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
}

.summary-row {
  display: flex;
  justify-content: space-between;
  margin: 4px 0;
  font-size: 0.875rem;
}

.price {
  font-weight: 600;
}

.book-btn {
  width: 100%;
  margin-top: 16px;
  padding: 14px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
}

.book-btn:disabled {
  background: #94a3b8;
  cursor: not-allowed;
}
```

---

## Data Tables

Interactive tables with sorting and actions:

### Table Tool Definition

```typescript
export const tools = {
  showDataTable: tool({
    description: 'Display a data table with sorting and actions',
    inputSchema: z.object({
      title: z.string(),
      columns: z.array(z.object({
        key: z.string(),
        header: z.string(),
        sortable: z.boolean().default(true),
        type: z.enum(['text', 'number', 'date', 'status', 'actions']).default('text'),
      })),
      data: z.array(z.record(z.any())),
      actions: z.array(z.object({
        label: z.string(),
        action: z.string(),
        variant: z.enum(['primary', 'secondary', 'danger']).default('secondary'),
      })).optional(),
    }),
  }),
};
```

### Data Table Component

```tsx
// components/data-table.tsx
import { useState, useMemo } from 'react';

interface Column {
  key: string;
  header: string;
  sortable?: boolean;
  type?: 'text' | 'number' | 'date' | 'status' | 'actions';
}

interface Action {
  label: string;
  action: string;
  variant?: 'primary' | 'secondary' | 'danger';
}

interface DataTableProps {
  title: string;
  columns: Column[];
  data: Record<string, any>[];
  actions?: Action[];
  onAction?: (action: string, row: Record<string, any>) => void;
}

export function DataTable({
  title,
  columns,
  data,
  actions,
  onAction,
}: DataTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [search, setSearch] = useState('');

  const filteredData = useMemo(() => {
    let result = data;

    if (search) {
      const lower = search.toLowerCase();
      result = result.filter((row) =>
        Object.values(row).some(
          (val) => String(val).toLowerCase().includes(lower)
        )
      );
    }

    if (sortKey) {
      result = [...result].sort((a, b) => {
        const aVal = a[sortKey];
        const bVal = b[sortKey];
        const comparison = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        return sortDir === 'asc' ? comparison : -comparison;
      });
    }

    return result;
  }, [data, search, sortKey, sortDir]);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const renderCell = (column: Column, value: any, row: Record<string, any>) => {
    switch (column.type) {
      case 'status':
        return <StatusBadge status={value} />;
      case 'date':
        return new Date(value).toLocaleDateString();
      case 'number':
        return typeof value === 'number' ? value.toLocaleString() : value;
      case 'actions':
        return (
          <div className="table-actions">
            {actions?.map((action) => (
              <button
                key={action.action}
                className={`action-btn ${action.variant}`}
                onClick={() => onAction?.(action.action, row)}
              >
                {action.label}
              </button>
            ))}
          </div>
        );
      default:
        return value;
    }
  };

  return (
    <div className="data-table-container">
      <div className="table-header">
        <h3>{title}</h3>
        <input
          type="search"
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="table-search"
        />
      </div>

      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => col.sortable && handleSort(col.key)}
                  className={col.sortable ? 'sortable' : ''}
                >
                  {col.header}
                  {sortKey === col.key && (
                    <span className="sort-indicator">
                      {sortDir === 'asc' ? ' ↑' : ' ↓'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredData.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col.key}>{renderCell(col, row[col.key], row)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        <span>{filteredData.length} of {data.length} rows</span>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, string> = {
    active: 'success',
    pending: 'warning',
    inactive: 'error',
    completed: 'success',
  };

  return (
    <span className={`status-badge ${variants[status.toLowerCase()] || 'default'}`}>
      {status}
    </span>
  );
}
```

```css
.data-table-container {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  overflow: hidden;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #e2e8f0;
}

.table-header h3 {
  margin: 0;
}

.table-search {
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  width: 200px;
}

.table-wrapper {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
}

.data-table th {
  background: #f8fafc;
  font-weight: 500;
  font-size: 0.75rem;
  text-transform: uppercase;
  color: #64748b;
}

.data-table th.sortable {
  cursor: pointer;
}

.data-table th.sortable:hover {
  background: #f1f5f9;
}

.sort-indicator {
  color: #3b82f6;
}

.data-table tbody tr:hover {
  background: #f8fafc;
}

.status-badge {
  padding: 4px 8px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.status-badge.success {
  background: #dcfce7;
  color: #16a34a;
}

.status-badge.warning {
  background: #fef3c7;
  color: #d97706;
}

.status-badge.error {
  background: #fee2e2;
  color: #dc2626;
}

.table-actions {
  display: flex;
  gap: 4px;
}

.action-btn {
  padding: 4px 8px;
  font-size: 0.75rem;
  border-radius: 4px;
  cursor: pointer;
}

.action-btn.secondary {
  background: #f1f5f9;
  border: none;
}

.action-btn.danger {
  background: #fee2e2;
  border: none;
  color: #dc2626;
}

.table-footer {
  padding: 12px 16px;
  background: #f8fafc;
  font-size: 0.75rem;
  color: #64748b;
}
```

---

## Summary

✅ Dynamic forms can be generated based on conversation context

✅ Interactive charts enable data exploration with click callbacks

✅ Product cards integrate e-commerce actions with AI queries

✅ Booking widgets handle scheduling with date/time selection

✅ Data tables provide sorting, filtering, and row actions

✅ All components can trigger new AI interactions via callbacks

**Next:** [Implementation Patterns](./06-implementation-patterns.md)

---

## Further Reading

- [Recharts](https://recharts.org/) — React charting library
- [React Hook Form](https://react-hook-form.com/) — Advanced form handling
- [TanStack Table](https://tanstack.com/table) — Headless table library
- [AI SDK Generative UI](https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces) — Official guide

---

<!-- 
Sources Consulted:
- AI SDK Generative UI: https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces
- v0 by Vercel component patterns
- shadcn/ui component library
-->
