---
title: "Number Formatting"
---

# Number Formatting

## Introduction

Numbers look different around the world. Americans write "1,234.56" while Germans write "1.234,56" and Indians write "12,34,567." Currency symbols appear before, after, or even within the number depending on the locale. Percentages, units, and compact notations all have locale-specific conventions.

In chat interfaces, numbers appear constantly: token counts, file sizes, prices, percentages, and more. Using `Intl.NumberFormat` ensures these display correctly for every user without manual formatting logic.

### What We'll Cover

- `Intl.NumberFormat` basics and locale differences
- Currency formatting with proper symbols and placement
- Unit formatting (bytes, speeds, temperatures)
- Compact notation ("1.2K", "3M")
- Percentage and decimal handling
- Building a complete number formatting service

### Prerequisites

- JavaScript number types
- Basic understanding of locale codes
- Understanding of chat interface needs

---

## Intl.NumberFormat Fundamentals

### Basic Usage

```javascript
const number = 1234567.89;

new Intl.NumberFormat('en-US').format(number);    // "1,234,567.89"
new Intl.NumberFormat('de-DE').format(number);    // "1.234.567,89"
new Intl.NumberFormat('fr-FR').format(number);    // "1 234 567,89"
new Intl.NumberFormat('ar-EG').format(number);    // "١٬٢٣٤٬٥٦٧٫٨٩"
new Intl.NumberFormat('en-IN').format(number);    // "12,34,567.89"
new Intl.NumberFormat('ja-JP').format(number);    // "1,234,567.89"
```

Notice the differences:
- **Grouping separators**: comma, period, space, or Arabic comma
- **Decimal separators**: period or comma
- **Grouping patterns**: Indian uses lakhs (2-digit groups after thousands)
- **Numeral systems**: Arabic-Indic numerals for Arabic

### Style Options

The `style` option determines the type of formatting:

```javascript
const number = 1234.5;

// Decimal (default)
new Intl.NumberFormat('en-US', { style: 'decimal' }).format(number);
// "1,234.5"

// Currency
new Intl.NumberFormat('en-US', { 
  style: 'currency', 
  currency: 'USD' 
}).format(number);
// "$1,234.50"

// Percent
new Intl.NumberFormat('en-US', { style: 'percent' }).format(0.456);
// "46%"

// Unit
new Intl.NumberFormat('en-US', { 
  style: 'unit', 
  unit: 'kilometer' 
}).format(number);
// "1,234.5 km"
```

---

## Currency Formatting

Currency formatting handles symbols, placement, and decimal rules automatically.

### Basic Currency

```javascript
const amount = 1234.5;

// USD
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD'
}).format(amount);
// "$1,234.50"

// EUR (German)
new Intl.NumberFormat('de-DE', {
  style: 'currency',
  currency: 'EUR'
}).format(amount);
// "1.234,50 €"

// JPY (no decimals)
new Intl.NumberFormat('ja-JP', {
  style: 'currency',
  currency: 'JPY'
}).format(amount);
// "¥1,235"

// GBP
new Intl.NumberFormat('en-GB', {
  style: 'currency',
  currency: 'GBP'
}).format(amount);
// "£1,234.50"

// SAR (Arabic)
new Intl.NumberFormat('ar-SA', {
  style: 'currency',
  currency: 'SAR'
}).format(amount);
// "١٬٢٣٤٫٥٠ ر.س."
```

### Currency Display Options

```javascript
const amount = 1234.5;

// symbol (default) - $
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  currencyDisplay: 'symbol'
}).format(amount);
// "$1,234.50"

// narrowSymbol - compact symbol
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  currencyDisplay: 'narrowSymbol'
}).format(amount);
// "$1,234.50" (same for USD, but CA$ vs $ for CAD)

// code - ISO code
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  currencyDisplay: 'code'
}).format(amount);
// "USD 1,234.50"

// name - full name
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  currencyDisplay: 'name'
}).format(amount);
// "1,234.50 US dollars"
```

### Handling Different Currencies

```javascript
class CurrencyFormatter {
  constructor(locale = 'en-US') {
    this.locale = locale;
    this.cache = new Map();
  }
  
  format(amount, currency, options = {}) {
    const cacheKey = `${currency}-${JSON.stringify(options)}`;
    
    if (!this.cache.has(cacheKey)) {
      this.cache.set(cacheKey, new Intl.NumberFormat(this.locale, {
        style: 'currency',
        currency: currency,
        ...options
      }));
    }
    
    return this.cache.get(cacheKey).format(amount);
  }
  
  formatCompact(amount, currency) {
    return this.format(amount, currency, {
      notation: 'compact',
      maximumFractionDigits: 1
    });
  }
}

const cf = new CurrencyFormatter('en-US');
cf.format(1234.5, 'USD');           // "$1,234.50"
cf.format(1234.5, 'EUR');           // "€1,234.50"
cf.format(1234567, 'USD');          // "$1,234,567.00"
cf.formatCompact(1234567, 'USD');   // "$1.2M"
```

---

## Unit Formatting

Format values with units for measurements, data sizes, and more.

### Available Units

| Category | Units |
|----------|-------|
| Length | `meter`, `kilometer`, `mile`, `inch`, `foot`, `yard` |
| Mass | `gram`, `kilogram`, `ounce`, `pound` |
| Volume | `liter`, `milliliter`, `gallon` |
| Speed | `meter-per-second`, `kilometer-per-hour`, `mile-per-hour` |
| Temperature | `celsius`, `fahrenheit` |
| Digital | `bit`, `byte`, `kilobit`, `kilobyte`, `megabit`, `megabyte`, `gigabit`, `gigabyte`, `terabyte` |
| Duration | `second`, `minute`, `hour`, `day`, `week`, `month`, `year` |

### Basic Unit Formatting

```javascript
// Length
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'kilometer'
}).format(42);
// "42 km"

// Speed
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'mile-per-hour'
}).format(65);
// "65 mph"

// Temperature
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'celsius'
}).format(25);
// "25°C"

// Digital storage
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'megabyte'
}).format(256);
// "256 MB"
```

### Unit Display Options

```javascript
// short (default)
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'kilometer',
  unitDisplay: 'short'
}).format(10);
// "10 km"

// narrow
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'kilometer',
  unitDisplay: 'narrow'
}).format(10);
// "10km"

// long
new Intl.NumberFormat('en-US', {
  style: 'unit',
  unit: 'kilometer',
  unitDisplay: 'long'
}).format(10);
// "10 kilometers"
```

### Practical File Size Formatter

```javascript
function formatFileSize(bytes, locale = 'en-US') {
  const units = ['byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte'];
  let unitIndex = 0;
  let size = bytes;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return new Intl.NumberFormat(locale, {
    style: 'unit',
    unit: units[unitIndex],
    maximumFractionDigits: unitIndex === 0 ? 0 : 1
  }).format(size);
}

// Examples
formatFileSize(500);           // "500 byte"
formatFileSize(1024);          // "1 kB"
formatFileSize(1536);          // "1.5 kB"
formatFileSize(1048576);       // "1 MB"
formatFileSize(1073741824);    // "1 GB"

// German locale
formatFileSize(1536, 'de-DE'); // "1,5 kB"
```

---

## Compact Notation

Display large numbers in abbreviated form.

### Basic Compact Notation

```javascript
// Short (default)
new Intl.NumberFormat('en-US', {
  notation: 'compact'
}).format(1234567);
// "1.2M"

new Intl.NumberFormat('en-US', {
  notation: 'compact'
}).format(9876);
// "9.9K"

// Long
new Intl.NumberFormat('en-US', {
  notation: 'compact',
  compactDisplay: 'long'
}).format(1234567);
// "1.2 million"
```

### Locale Differences in Compact Notation

```javascript
const number = 12345678;

new Intl.NumberFormat('en-US', { notation: 'compact' }).format(number);
// "12M"

new Intl.NumberFormat('de-DE', { notation: 'compact' }).format(number);
// "12 Mio."

new Intl.NumberFormat('ja-JP', { notation: 'compact' }).format(number);
// "1235万"

new Intl.NumberFormat('zh-CN', { notation: 'compact' }).format(number);
// "1235万"

new Intl.NumberFormat('ko-KR', { notation: 'compact' }).format(number);
// "1235만"
```

### Compact Currency

```javascript
new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1
}).format(2500000);
// "$2.5M"

new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  compactDisplay: 'long',
  maximumFractionDigits: 1
}).format(2500000);
// "$2.5 million"
```

### Token Counter for Chat

```javascript
function formatTokenCount(tokens, locale = 'en-US') {
  if (tokens < 1000) {
    return new Intl.NumberFormat(locale).format(tokens);
  }
  
  return new Intl.NumberFormat(locale, {
    notation: 'compact',
    maximumFractionDigits: 1
  }).format(tokens);
}

formatTokenCount(500);      // "500"
formatTokenCount(1234);     // "1.2K"
formatTokenCount(45678);    // "46K"
formatTokenCount(1234567);  // "1.2M"
```

---

## Percentage Formatting

### Basic Percentage

```javascript
// Input is a fraction (0.456 = 45.6%)
new Intl.NumberFormat('en-US', {
  style: 'percent'
}).format(0.456);
// "46%"

new Intl.NumberFormat('en-US', {
  style: 'percent',
  maximumFractionDigits: 1
}).format(0.456);
// "45.6%"

// Locale differences
new Intl.NumberFormat('de-DE', {
  style: 'percent',
  maximumFractionDigits: 1
}).format(0.456);
// "45,6 %"

new Intl.NumberFormat('fr-FR', {
  style: 'percent',
  maximumFractionDigits: 1
}).format(0.456);
// "45,6 %"
```

### Progress Indicators

```javascript
function formatProgress(current, total, locale = 'en-US') {
  const fraction = current / total;
  
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    maximumFractionDigits: 0
  }).format(fraction);
}

formatProgress(75, 100);   // "75%"
formatProgress(1, 3);      // "33%"
formatProgress(2, 3);      // "67%"
```

---

## Sign Display Options

Control how positive/negative signs are displayed.

### Sign Display Values

```javascript
const positiveNumber = 42;
const negativeNumber = -42;
const zero = 0;

// auto (default) - sign only for negative
new Intl.NumberFormat('en-US', {
  signDisplay: 'auto'
}).format(positiveNumber);  // "42"

// always - sign for all non-zero
new Intl.NumberFormat('en-US', {
  signDisplay: 'always'
}).format(positiveNumber);  // "+42"

// exceptZero - sign for non-zero
new Intl.NumberFormat('en-US', {
  signDisplay: 'exceptZero'
}).format(positiveNumber);  // "+42"

// negative - sign only for negative (no sign for positive)
new Intl.NumberFormat('en-US', {
  signDisplay: 'negative'
}).format(negativeNumber);  // "-42"

// never - no sign ever
new Intl.NumberFormat('en-US', {
  signDisplay: 'never'
}).format(negativeNumber);  // "42"
```

### Change Indicators

```javascript
function formatChange(change, locale = 'en-US') {
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    signDisplay: 'exceptZero',
    maximumFractionDigits: 1
  }).format(change);
}

formatChange(0.15);   // "+15%"
formatChange(-0.05);  // "-5%"
formatChange(0);      // "0%"
```

---

## Precision Control

### Fraction Digits

```javascript
// Minimum fraction digits
new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2
}).format(42);
// "42.00"

// Maximum fraction digits
new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2
}).format(3.14159);
// "3.14"

// Both
new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4
}).format(3.1);
// "3.10"
```

### Significant Digits

```javascript
// Significant figures
new Intl.NumberFormat('en-US', {
  maximumSignificantDigits: 3
}).format(1234567);
// "1,230,000"

new Intl.NumberFormat('en-US', {
  maximumSignificantDigits: 3
}).format(0.001234567);
// "0.00123"
```

### Rounding Modes

```javascript
// Default is 'halfExpand' (standard rounding)
new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0
}).format(2.5);
// "3"

// Other rounding modes (requires newer browsers)
new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
  roundingMode: 'ceil'  // Always round up
}).format(2.1);
// "3"

new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
  roundingMode: 'floor'  // Always round down
}).format(2.9);
// "2"

new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
  roundingMode: 'trunc'  // Towards zero
}).format(-2.9);
// "-2"
```

---

## Complete Number Formatting Service

```javascript
class NumberFormattingService {
  constructor(locale = 'en-US') {
    this.locale = locale;
    this.formatters = {};
    this.initFormatters();
  }
  
  initFormatters() {
    this.formatters = {
      // Standard number
      decimal: new Intl.NumberFormat(this.locale),
      
      // Integer only
      integer: new Intl.NumberFormat(this.locale, {
        maximumFractionDigits: 0
      }),
      
      // Precise decimals
      precise: new Intl.NumberFormat(this.locale, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }),
      
      // Compact
      compact: new Intl.NumberFormat(this.locale, {
        notation: 'compact',
        maximumFractionDigits: 1
      }),
      
      // Compact long
      compactLong: new Intl.NumberFormat(this.locale, {
        notation: 'compact',
        compactDisplay: 'long',
        maximumFractionDigits: 1
      }),
      
      // Percent
      percent: new Intl.NumberFormat(this.locale, {
        style: 'percent',
        maximumFractionDigits: 0
      }),
      
      // Percent precise
      percentPrecise: new Intl.NumberFormat(this.locale, {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
      }),
      
      // Change indicator
      change: new Intl.NumberFormat(this.locale, {
        style: 'percent',
        signDisplay: 'exceptZero',
        maximumFractionDigits: 1
      })
    };
  }
  
  setLocale(locale) {
    this.locale = locale;
    this.initFormatters();
  }
  
  // Basic formatting
  format(number, type = 'decimal') {
    const formatter = this.formatters[type];
    return formatter ? formatter.format(number) : this.formatters.decimal.format(number);
  }
  
  // Currency (cached)
  currencyCache = new Map();
  
  formatCurrency(amount, currency, options = {}) {
    const key = `${currency}-${options.compact || false}`;
    
    if (!this.currencyCache.has(key)) {
      this.currencyCache.set(key, new Intl.NumberFormat(this.locale, {
        style: 'currency',
        currency: currency,
        ...(options.compact && {
          notation: 'compact',
          maximumFractionDigits: 1
        })
      }));
    }
    
    return this.currencyCache.get(key).format(amount);
  }
  
  // File size
  formatFileSize(bytes) {
    const units = ['byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte'];
    let unitIndex = 0;
    let size = bytes;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return new Intl.NumberFormat(this.locale, {
      style: 'unit',
      unit: units[unitIndex],
      maximumFractionDigits: unitIndex === 0 ? 0 : 1
    }).format(size);
  }
  
  // Token count (common in chat apps)
  formatTokens(tokens) {
    if (tokens < 1000) {
      return this.formatters.integer.format(tokens);
    }
    return this.formatters.compact.format(tokens);
  }
  
  // Duration in seconds to readable format
  formatDuration(seconds) {
    if (seconds < 60) {
      return new Intl.NumberFormat(this.locale, {
        style: 'unit',
        unit: 'second',
        unitDisplay: 'narrow'
      }).format(Math.round(seconds));
    }
    
    if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      const minStr = new Intl.NumberFormat(this.locale, {
        style: 'unit',
        unit: 'minute',
        unitDisplay: 'narrow'
      }).format(mins);
      
      if (secs === 0) return minStr;
      
      const secStr = new Intl.NumberFormat(this.locale, {
        style: 'unit',
        unit: 'second',
        unitDisplay: 'narrow'
      }).format(secs);
      
      return `${minStr} ${secStr}`;
    }
    
    const hours = Math.floor(seconds / 3600);
    const mins = Math.round((seconds % 3600) / 60);
    
    const hourStr = new Intl.NumberFormat(this.locale, {
      style: 'unit',
      unit: 'hour',
      unitDisplay: 'narrow'
    }).format(hours);
    
    if (mins === 0) return hourStr;
    
    const minStr = new Intl.NumberFormat(this.locale, {
      style: 'unit',
      unit: 'minute',
      unitDisplay: 'narrow'
    }).format(mins);
    
    return `${hourStr} ${minStr}`;
  }
  
  // Range formatting
  formatRange(start, end, type = 'decimal') {
    const formatter = this.formatters[type] || this.formatters.decimal;
    return formatter.formatRange(start, end);
  }
}

// Usage
const numFormat = new NumberFormattingService('en-US');

// Basic numbers
numFormat.format(1234567);                    // "1,234,567"
numFormat.format(1234567, 'compact');         // "1.2M"
numFormat.format(0.456, 'percent');           // "46%"

// Currency
numFormat.formatCurrency(1234.5, 'USD');      // "$1,234.50"
numFormat.formatCurrency(1234567, 'USD', { compact: true }); // "$1.2M"

// File sizes
numFormat.formatFileSize(1536);               // "1.5 kB"
numFormat.formatFileSize(1073741824);         // "1 GB"

// Tokens
numFormat.formatTokens(500);                  // "500"
numFormat.formatTokens(12345);                // "12K"

// Duration
numFormat.formatDuration(45);                 // "45s"
numFormat.formatDuration(125);                // "2m 5s"
numFormat.formatDuration(3725);               // "1h 2m"
```

---

## Chat-Specific Formatting Examples

### Usage Statistics Display

```javascript
function formatUsageStats(stats, locale = 'en-US') {
  const nf = new NumberFormattingService(locale);
  
  return {
    tokensUsed: nf.formatTokens(stats.tokensUsed),
    tokensRemaining: nf.formatTokens(stats.tokensRemaining),
    percentUsed: nf.format(stats.tokensUsed / stats.tokensTotal, 'percent'),
    cost: nf.formatCurrency(stats.cost, stats.currency),
    responseTime: nf.formatDuration(stats.avgResponseTime)
  };
}

// Example
const stats = {
  tokensUsed: 45678,
  tokensRemaining: 54322,
  tokensTotal: 100000,
  cost: 1.23,
  currency: 'USD',
  avgResponseTime: 2.5
};

formatUsageStats(stats, 'en-US');
// {
//   tokensUsed: "46K",
//   tokensRemaining: "54K",
//   percentUsed: "46%",
//   cost: "$1.23",
//   responseTime: "2s"
// }
```

### Message Metadata

```javascript
function formatMessageMeta(message, locale = 'en-US') {
  const nf = new NumberFormattingService(locale);
  
  return {
    tokens: nf.formatTokens(message.tokenCount),
    attachmentSize: message.attachmentSize 
      ? nf.formatFileSize(message.attachmentSize) 
      : null,
    processingTime: `${nf.format(message.processingMs / 1000, 'precise')}s`
  };
}
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `toFixed()` for display | Use `Intl.NumberFormat` |
| Hardcoding thousand separators | Let `Intl.NumberFormat` handle locale |
| Assuming 2 decimal places for currency | Use `currency` option (JPY has 0) |
| Creating formatters in loops | Cache formatter instances |
| Using `Math.round()` + string concat | Use `Intl.NumberFormat` with precision options |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Cache `Intl.NumberFormat` instances | Creating formatters is expensive |
| Use `notation: 'compact'` for large numbers | More readable in UI |
| Specify currency explicitly | Avoids ambiguity |
| Use unit formatting for measurements | Proper localization |
| Match precision to context | Financial = precise, stats = compact |

---

## Hands-on Exercise

### Your Task

Build a number formatting utility for a chat interface that displays token usage and costs.

### Requirements

1. Format token counts: compact for large numbers, full for small
2. Format costs in user's currency
3. Format percentages for usage meters
4. Format file sizes for attachments
5. Support at least English and German locales

### Expected Result

```javascript
const formatter = new ChatNumberFormatter('en-US');

formatter.formatTokens(12345);         // "12.3K"
formatter.formatCost(4.99, 'USD');     // "$4.99"
formatter.formatPercent(0.75);         // "75%"
formatter.formatFileSize(5242880);     // "5 MB"

// Switch locale
formatter.setLocale('de-DE');
formatter.formatTokens(12345);         // "12.300" or "12,3K"
formatter.formatCost(4.99, 'EUR');     // "4,99 €"
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Create separate formatters for tokens, currency, percent, and file size
- Use `notation: 'compact'` with `maximumFractionDigits: 1`
- Cache formatters to avoid recreation
- Use the `unit` style for file sizes

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class ChatNumberFormatter {
  constructor(locale = 'en-US') {
    this.locale = locale;
    this.initFormatters();
  }
  
  initFormatters() {
    this.compactFormatter = new Intl.NumberFormat(this.locale, {
      notation: 'compact',
      maximumFractionDigits: 1
    });
    
    this.integerFormatter = new Intl.NumberFormat(this.locale, {
      maximumFractionDigits: 0
    });
    
    this.percentFormatter = new Intl.NumberFormat(this.locale, {
      style: 'percent',
      maximumFractionDigits: 0
    });
    
    this.currencyCache = new Map();
  }
  
  setLocale(locale) {
    this.locale = locale;
    this.currencyCache.clear();
    this.initFormatters();
  }
  
  formatTokens(tokens) {
    if (tokens < 1000) {
      return this.integerFormatter.format(tokens);
    }
    return this.compactFormatter.format(tokens);
  }
  
  formatCost(amount, currency) {
    if (!this.currencyCache.has(currency)) {
      this.currencyCache.set(currency, new Intl.NumberFormat(this.locale, {
        style: 'currency',
        currency: currency
      }));
    }
    return this.currencyCache.get(currency).format(amount);
  }
  
  formatPercent(value) {
    return this.percentFormatter.format(value);
  }
  
  formatFileSize(bytes) {
    const units = ['byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte'];
    let unitIndex = 0;
    let size = bytes;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return new Intl.NumberFormat(this.locale, {
      style: 'unit',
      unit: units[unitIndex],
      maximumFractionDigits: unitIndex === 0 ? 0 : 1
    }).format(size);
  }
}

// Test
const fmt = new ChatNumberFormatter('en-US');
console.log(fmt.formatTokens(500));        // "500"
console.log(fmt.formatTokens(12345));      // "12.3K"
console.log(fmt.formatCost(4.99, 'USD'));  // "$4.99"
console.log(fmt.formatPercent(0.75));      // "75%"
console.log(fmt.formatFileSize(5242880));  // "5 MB"
```

</details>

---

## Summary

✅ Use `Intl.NumberFormat` for all number display—never manual formatting

✅ Cache formatter instances for performance

✅ Use `style: 'currency'` with explicit currency code for monetary values

✅ Use `notation: 'compact'` for large numbers in UI contexts

✅ Use `style: 'unit'` for measurements, file sizes, and durations

**Next:** [AI Response Language Handling](./05-ai-response-language.md)

---

## Further Reading

- [MDN: Intl.NumberFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat) - Complete API reference
- [ISO 4217 Currency Codes](https://www.iso.org/iso-4217-currency-codes.html) - Currency code list
- [Unicode CLDR](https://cldr.unicode.org/) - Locale data source

<!--
Sources Consulted:
- MDN Intl.NumberFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat
- ISO 4217: https://www.iso.org/iso-4217-currency-codes.html
- Unicode CLDR: https://cldr.unicode.org/
-->
