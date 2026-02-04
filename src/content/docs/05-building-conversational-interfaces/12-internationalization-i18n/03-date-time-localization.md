---
title: "Date and Time Localization"
---

# Date and Time Localization

## Introduction

"5 minutes ago" reads differently across the globe—in German it's "vor 5 Minuten," in Arabic it's "منذ 5 دقائق," and in Japanese it's "5分前." Beyond translations, date formats themselves vary wildly: Americans write "12/25/2024" while Europeans write "25/12/2024" and Japanese write "2024年12月25日."

Chat interfaces are particularly date-heavy. Every message has a timestamp. Relative times like "just now" and "2 hours ago" appear constantly. Getting this wrong confuses users; getting it right makes your interface feel native.

### What We'll Cover

- `Intl.DateTimeFormat` for locale-aware date/time formatting
- `Intl.RelativeTimeFormat` for "2 hours ago" style strings
- Timezone handling and conversion
- Calendar system support (Gregorian, Islamic, Hebrew, etc.)
- Chat-specific timestamp patterns

### Prerequisites

- JavaScript `Date` object basics
- Understanding of Unix timestamps
- Basic internationalization concepts from previous lessons

---

## Intl.DateTimeFormat Fundamentals

The `Intl.DateTimeFormat` API formats dates and times according to locale conventions.

### Basic Usage

```javascript
const date = new Date('2024-12-25T14:30:00');

// Default formatting for different locales
new Intl.DateTimeFormat('en-US').format(date);    // "12/25/2024"
new Intl.DateTimeFormat('en-GB').format(date);    // "25/12/2024"
new Intl.DateTimeFormat('de-DE').format(date);    // "25.12.2024"
new Intl.DateTimeFormat('ja-JP').format(date);    // "2024/12/25"
new Intl.DateTimeFormat('ar-SA').format(date);    // "٢٥‏/١٢‏/٢٠٢٤"
```

### Style Options

The `dateStyle` and `timeStyle` options provide predefined formatting levels:

```javascript
const date = new Date('2024-12-25T14:30:00');
const locale = 'en-US';

// Date styles
new Intl.DateTimeFormat(locale, { dateStyle: 'full' }).format(date);
// "Wednesday, December 25, 2024"

new Intl.DateTimeFormat(locale, { dateStyle: 'long' }).format(date);
// "December 25, 2024"

new Intl.DateTimeFormat(locale, { dateStyle: 'medium' }).format(date);
// "Dec 25, 2024"

new Intl.DateTimeFormat(locale, { dateStyle: 'short' }).format(date);
// "12/25/24"

// Time styles
new Intl.DateTimeFormat(locale, { timeStyle: 'full' }).format(date);
// "2:30:00 PM Eastern Standard Time"

new Intl.DateTimeFormat(locale, { timeStyle: 'long' }).format(date);
// "2:30:00 PM EST"

new Intl.DateTimeFormat(locale, { timeStyle: 'medium' }).format(date);
// "2:30:00 PM"

new Intl.DateTimeFormat(locale, { timeStyle: 'short' }).format(date);
// "2:30 PM"
```

### Combining Date and Time

```javascript
const formatter = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
  timeStyle: 'short'
});

formatter.format(new Date('2024-12-25T14:30:00'));
// "Dec 25, 2024, 2:30 PM"
```

### Granular Component Options

For precise control, specify individual components:

```javascript
const options = {
  year: 'numeric',       // '2024'
  month: 'long',         // 'December'
  day: 'numeric',        // '25'
  weekday: 'long',       // 'Wednesday'
  hour: 'numeric',       // '2'
  minute: '2-digit',     // '30'
  second: '2-digit',     // '00'
  hour12: true           // Use 12-hour format
};

new Intl.DateTimeFormat('en-US', options).format(date);
// "Wednesday, December 25, 2024 at 2:30:00 PM"
```

### Component Value Reference

| Component | Values |
|-----------|--------|
| `year` | `'numeric'`, `'2-digit'` |
| `month` | `'numeric'`, `'2-digit'`, `'narrow'`, `'short'`, `'long'` |
| `day` | `'numeric'`, `'2-digit'` |
| `weekday` | `'narrow'`, `'short'`, `'long'` |
| `hour` | `'numeric'`, `'2-digit'` |
| `minute` | `'numeric'`, `'2-digit'` |
| `second` | `'numeric'`, `'2-digit'` |
| `hour12` | `true`, `false` |

---

## Intl.RelativeTimeFormat for Chat

Chat interfaces rely heavily on relative time strings like "just now," "5 minutes ago," or "yesterday."

### Basic Usage

```javascript
const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

rtf.format(-1, 'day');     // "yesterday"
rtf.format(0, 'day');      // "today"
rtf.format(1, 'day');      // "tomorrow"
rtf.format(-5, 'minute');  // "5 minutes ago"
rtf.format(2, 'hour');     // "in 2 hours"
```

### Numeric vs Auto

```javascript
// numeric: 'always' (default) - Always uses numbers
const rtfNumeric = new Intl.RelativeTimeFormat('en', { numeric: 'always' });
rtfNumeric.format(-1, 'day');  // "1 day ago"

// numeric: 'auto' - Uses words when possible
const rtfAuto = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
rtfAuto.format(-1, 'day');     // "yesterday"
rtfAuto.format(-2, 'day');     // "2 days ago"
```

### Style Options

```javascript
// long (default)
new Intl.RelativeTimeFormat('en', { style: 'long' }).format(-5, 'minute');
// "5 minutes ago"

// short
new Intl.RelativeTimeFormat('en', { style: 'short' }).format(-5, 'minute');
// "5 min. ago"

// narrow
new Intl.RelativeTimeFormat('en', { style: 'narrow' }).format(-5, 'minute');
// "5m ago"
```

### Available Units

| Unit | Example |
|------|---------|
| `'second'` | "5 seconds ago" |
| `'minute'` | "5 minutes ago" |
| `'hour'` | "2 hours ago" |
| `'day'` | "yesterday" / "3 days ago" |
| `'week'` | "last week" / "2 weeks ago" |
| `'month'` | "last month" / "3 months ago" |
| `'quarter'` | "last quarter" |
| `'year'` | "last year" / "2 years ago" |

### Practical Relative Time Function

```javascript
function formatRelativeTime(date, locale = 'en') {
  const rtf = new Intl.RelativeTimeFormat(locale, { 
    numeric: 'auto',
    style: 'long'
  });
  
  const now = new Date();
  const diff = date - now;
  const diffInSeconds = Math.round(diff / 1000);
  const diffInMinutes = Math.round(diff / 60000);
  const diffInHours = Math.round(diff / 3600000);
  const diffInDays = Math.round(diff / 86400000);
  const diffInWeeks = Math.round(diff / 604800000);
  const diffInMonths = Math.round(diff / 2592000000);
  const diffInYears = Math.round(diff / 31536000000);
  
  if (Math.abs(diffInSeconds) < 60) {
    return rtf.format(diffInSeconds, 'second');
  } else if (Math.abs(diffInMinutes) < 60) {
    return rtf.format(diffInMinutes, 'minute');
  } else if (Math.abs(diffInHours) < 24) {
    return rtf.format(diffInHours, 'hour');
  } else if (Math.abs(diffInDays) < 7) {
    return rtf.format(diffInDays, 'day');
  } else if (Math.abs(diffInWeeks) < 4) {
    return rtf.format(diffInWeeks, 'week');
  } else if (Math.abs(diffInMonths) < 12) {
    return rtf.format(diffInMonths, 'month');
  } else {
    return rtf.format(diffInYears, 'year');
  }
}

// Examples
const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
formatRelativeTime(fiveMinutesAgo, 'en');    // "5 minutes ago"
formatRelativeTime(fiveMinutesAgo, 'de');    // "vor 5 Minuten"
formatRelativeTime(fiveMinutesAgo, 'ar');    // "قبل ٥ دقائق"
formatRelativeTime(fiveMinutesAgo, 'ja');    // "5分前"
```

---

## Timezone Handling

Chat messages may be sent from different timezones. Display them correctly for each user.

### Setting Timezone in DateTimeFormat

```javascript
const date = new Date('2024-12-25T14:30:00Z'); // UTC time

// Display in different timezones
new Intl.DateTimeFormat('en-US', {
  timeStyle: 'long',
  timeZone: 'America/New_York'
}).format(date);
// "9:30:00 AM EST"

new Intl.DateTimeFormat('en-US', {
  timeStyle: 'long',
  timeZone: 'Europe/London'
}).format(date);
// "2:30:00 PM GMT"

new Intl.DateTimeFormat('en-US', {
  timeStyle: 'long',
  timeZone: 'Asia/Tokyo'
}).format(date);
// "11:30:00 PM JST"
```

### Timezone Display Options

```javascript
const options = {
  dateStyle: 'medium',
  timeStyle: 'long',
  timeZone: 'America/New_York',
  timeZoneName: 'short'  // 'long', 'short', 'shortOffset', 'longOffset', 'shortGeneric', 'longGeneric'
};

// short: "EST"
// long: "Eastern Standard Time"
// shortOffset: "GMT-5"
// longOffset: "GMT-05:00"
// shortGeneric: "ET" (Eastern Time)
// longGeneric: "Eastern Time"
```

### Getting User's Timezone

```javascript
function getUserTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

console.log(getUserTimezone());  // e.g., "America/New_York"
```

### Timezone-Aware Chat Timestamps

```javascript
class ChatTimestampFormatter {
  constructor(locale, userTimezone = null) {
    this.locale = locale;
    this.timezone = userTimezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    
    this.initFormatters();
  }
  
  initFormatters() {
    // For today's messages
    this.timeOnly = new Intl.DateTimeFormat(this.locale, {
      timeStyle: 'short',
      timeZone: this.timezone
    });
    
    // For this week's messages
    this.weekday = new Intl.DateTimeFormat(this.locale, {
      weekday: 'short',
      hour: 'numeric',
      minute: '2-digit',
      timeZone: this.timezone
    });
    
    // For older messages
    this.fullDate = new Intl.DateTimeFormat(this.locale, {
      dateStyle: 'medium',
      timeStyle: 'short',
      timeZone: this.timezone
    });
    
    // Relative time
    this.relativeTime = new Intl.RelativeTimeFormat(this.locale, {
      numeric: 'auto',
      style: 'long'
    });
  }
  
  format(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    // Less than 1 minute: "just now"
    if (diffMins < 1) {
      return this.relativeTime.format(0, 'second');  // "now" or equivalent
    }
    
    // Less than 1 hour: "X minutes ago"
    if (diffMins < 60) {
      return this.relativeTime.format(-diffMins, 'minute');
    }
    
    // Less than 24 hours: "X hours ago" or time
    if (diffHours < 24 && this.isToday(date, now)) {
      return this.timeOnly.format(date);
    }
    
    // Yesterday
    if (this.isYesterday(date, now)) {
      return `${this.relativeTime.format(-1, 'day')} ${this.timeOnly.format(date)}`;
    }
    
    // This week
    if (diffDays < 7) {
      return this.weekday.format(date);
    }
    
    // Older: full date
    return this.fullDate.format(date);
  }
  
  isToday(date, now) {
    return date.toDateString() === now.toDateString();
  }
  
  isYesterday(date, now) {
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    return date.toDateString() === yesterday.toDateString();
  }
}

// Usage
const formatter = new ChatTimestampFormatter('en-US', 'America/New_York');

// Recent message
formatter.format(Date.now() - 30000);          // "now"
formatter.format(Date.now() - 5 * 60000);      // "5 minutes ago"
formatter.format(Date.now() - 3 * 3600000);    // "10:30 AM"
formatter.format(Date.now() - 86400000);       // "yesterday 2:30 PM"
formatter.format(Date.now() - 3 * 86400000);   // "Mon, 2:30 PM"
formatter.format(Date.now() - 30 * 86400000);  // "Nov 25, 2024, 2:30 PM"
```

---

## Calendar Systems

Some cultures use non-Gregorian calendars. The `Intl` API supports many:

### Available Calendars

| Calendar | Regions |
|----------|---------|
| `gregory` | Default (Western) |
| `islamic` | Middle East, Muslim regions |
| `islamic-umalqura` | Saudi Arabia |
| `hebrew` | Israel |
| `chinese` | China, East Asia |
| `japanese` | Japan |
| `buddhist` | Thailand |
| `persian` | Iran |
| `indian` | India |

### Using Non-Gregorian Calendars

```javascript
const date = new Date('2024-12-25');

// Gregorian (default)
new Intl.DateTimeFormat('en-US', {
  dateStyle: 'full',
  calendar: 'gregory'
}).format(date);
// "Wednesday, December 25, 2024"

// Islamic (Hijri)
new Intl.DateTimeFormat('ar-SA', {
  dateStyle: 'full',
  calendar: 'islamic-umalqura'
}).format(date);
// "الأربعاء، ٢٣ جمادى الآخرة ١٤٤٦ هـ"

// Hebrew
new Intl.DateTimeFormat('he-IL', {
  dateStyle: 'full',
  calendar: 'hebrew'
}).format(date);
// "יום רביעי, כ״ד בכסלו ה׳תשפ״ה"

// Japanese (with era)
new Intl.DateTimeFormat('ja-JP', {
  dateStyle: 'full',
  calendar: 'japanese'
}).format(date);
// "令和6年12月25日水曜日"

// Buddhist (Thailand)
new Intl.DateTimeFormat('th-TH', {
  dateStyle: 'full',
  calendar: 'buddhist'
}).format(date);
// "วันพุธที่ 25 ธันวาคม พ.ศ. 2567"
```

### Detecting User's Preferred Calendar

```javascript
function getUserCalendar(locale) {
  // Get the resolved calendar from a DateTimeFormat
  const dtf = new Intl.DateTimeFormat(locale);
  return dtf.resolvedOptions().calendar;
}

console.log(getUserCalendar('ar-SA'));  // "gregory" (default) or specified
console.log(getUserCalendar('th-TH'));  // "gregory" (default)
```

> **Note:** Browsers default to Gregorian even for locales that traditionally use other calendars. You may need to explicitly set the calendar based on user preferences.

---

## Complete Chat Date/Time Service

```javascript
class ChatDateTimeService {
  constructor(config = {}) {
    this.locale = config.locale || 'en';
    this.timezone = config.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    this.calendar = config.calendar || 'gregory';
    this.hour12 = config.hour12;  // undefined = locale default
    
    this.initFormatters();
  }
  
  initFormatters() {
    const baseOptions = {
      timeZone: this.timezone,
      calendar: this.calendar,
      ...(this.hour12 !== undefined && { hour12: this.hour12 })
    };
    
    // Various formatters
    this.formatters = {
      // Just time: "2:30 PM"
      time: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        timeStyle: 'short'
      }),
      
      // Time with seconds: "2:30:45 PM"
      timeWithSeconds: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        timeStyle: 'medium'
      }),
      
      // Short date: "12/25/24"
      dateShort: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        dateStyle: 'short'
      }),
      
      // Medium date: "Dec 25, 2024"
      dateMedium: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        dateStyle: 'medium'
      }),
      
      // Full date: "Wednesday, December 25, 2024"
      dateFull: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        dateStyle: 'full'
      }),
      
      // Date + time: "Dec 25, 2024, 2:30 PM"
      dateTime: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        dateStyle: 'medium',
        timeStyle: 'short'
      }),
      
      // Weekday + time: "Wed 2:30 PM"
      weekdayTime: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        weekday: 'short',
        hour: 'numeric',
        minute: '2-digit'
      }),
      
      // Month and day: "Dec 25"
      monthDay: new Intl.DateTimeFormat(this.locale, {
        ...baseOptions,
        month: 'short',
        day: 'numeric'
      }),
      
      // Relative time
      relative: new Intl.RelativeTimeFormat(this.locale, {
        numeric: 'auto',
        style: 'long'
      }),
      
      // Relative time (short)
      relativeShort: new Intl.RelativeTimeFormat(this.locale, {
        numeric: 'auto',
        style: 'short'
      })
    };
  }
  
  // Update settings dynamically
  setLocale(locale) {
    this.locale = locale;
    this.initFormatters();
  }
  
  setTimezone(timezone) {
    this.timezone = timezone;
    this.initFormatters();
  }
  
  // Format methods
  formatTime(date) {
    return this.formatters.time.format(new Date(date));
  }
  
  formatDate(date, style = 'medium') {
    const formatter = this.formatters[`date${style.charAt(0).toUpperCase() + style.slice(1)}`];
    return formatter ? formatter.format(new Date(date)) : this.formatters.dateMedium.format(new Date(date));
  }
  
  formatDateTime(date) {
    return this.formatters.dateTime.format(new Date(date));
  }
  
  // Smart timestamp for chat messages
  formatMessageTimestamp(timestamp, options = {}) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    
    // Very recent: relative time
    if (diffMins < 1) {
      return this.formatters.relative.format(0, 'minute');  // "this minute"
    }
    
    if (diffMins < 60) {
      return this.formatters.relative.format(-diffMins, 'minute');
    }
    
    // Today: just time
    if (this.isSameDay(date, now)) {
      return this.formatters.time.format(date);
    }
    
    // Yesterday
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (this.isSameDay(date, yesterday)) {
      const yesterdayLabel = this.formatters.relative.format(-1, 'day');
      return options.includeTime 
        ? `${yesterdayLabel}, ${this.formatters.time.format(date)}`
        : yesterdayLabel;
    }
    
    // This week: weekday + time
    const daysDiff = Math.floor(diffMs / 86400000);
    if (daysDiff < 7) {
      return this.formatters.weekdayTime.format(date);
    }
    
    // This year: month + day
    if (date.getFullYear() === now.getFullYear()) {
      return options.includeTime
        ? `${this.formatters.monthDay.format(date)}, ${this.formatters.time.format(date)}`
        : this.formatters.monthDay.format(date);
    }
    
    // Older: full date + time
    return this.formatters.dateTime.format(date);
  }
  
  // For message grouping (same day separator)
  formatDaySeparator(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    
    if (this.isSameDay(date, now)) {
      return this.formatters.relative.format(0, 'day');  // "today"
    }
    
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (this.isSameDay(date, yesterday)) {
      return this.formatters.relative.format(-1, 'day');  // "yesterday"
    }
    
    // Within this week
    const daysDiff = Math.floor((now - date) / 86400000);
    if (daysDiff < 7) {
      return new Intl.DateTimeFormat(this.locale, {
        weekday: 'long',
        timeZone: this.timezone
      }).format(date);  // "Monday"
    }
    
    // Older
    return this.formatters.dateMedium.format(date);
  }
  
  // Relative time formatting
  formatRelative(timestamp, style = 'long') {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = date - now;
    
    const seconds = Math.round(diff / 1000);
    const minutes = Math.round(diff / 60000);
    const hours = Math.round(diff / 3600000);
    const days = Math.round(diff / 86400000);
    const weeks = Math.round(diff / 604800000);
    const months = Math.round(diff / 2592000000);
    const years = Math.round(diff / 31536000000);
    
    const rtf = style === 'short' 
      ? this.formatters.relativeShort 
      : this.formatters.relative;
    
    if (Math.abs(seconds) < 60) {
      return rtf.format(seconds, 'second');
    } else if (Math.abs(minutes) < 60) {
      return rtf.format(minutes, 'minute');
    } else if (Math.abs(hours) < 24) {
      return rtf.format(hours, 'hour');
    } else if (Math.abs(days) < 7) {
      return rtf.format(days, 'day');
    } else if (Math.abs(weeks) < 4) {
      return rtf.format(weeks, 'week');
    } else if (Math.abs(months) < 12) {
      return rtf.format(months, 'month');
    } else {
      return rtf.format(years, 'year');
    }
  }
  
  // Helper methods
  isSameDay(date1, date2) {
    // Convert to timezone for accurate comparison
    const d1 = this.formatters.dateShort.format(date1);
    const d2 = this.formatters.dateShort.format(date2);
    return d1 === d2;
  }
  
  // Get current timezone info
  getTimezoneInfo() {
    const now = new Date();
    const formatter = new Intl.DateTimeFormat(this.locale, {
      timeZone: this.timezone,
      timeZoneName: 'long'
    });
    
    const parts = formatter.formatToParts(now);
    const timeZonePart = parts.find(p => p.type === 'timeZoneName');
    
    return {
      id: this.timezone,
      name: timeZonePart ? timeZonePart.value : this.timezone,
      offset: this.getTimezoneOffset(now)
    };
  }
  
  getTimezoneOffset(date) {
    const formatter = new Intl.DateTimeFormat(this.locale, {
      timeZone: this.timezone,
      timeZoneName: 'shortOffset'
    });
    const parts = formatter.formatToParts(date);
    const offsetPart = parts.find(p => p.type === 'timeZoneName');
    return offsetPart ? offsetPart.value : '';
  }
}

// Usage
const dateService = new ChatDateTimeService({
  locale: 'en-US',
  timezone: 'America/New_York'
});

// Chat message timestamps
dateService.formatMessageTimestamp(Date.now() - 30000);
// "0 minutes ago"

dateService.formatMessageTimestamp(Date.now() - 5 * 60000);
// "5 minutes ago"

dateService.formatMessageTimestamp(Date.now() - 3 * 3600000);
// "10:30 AM"

dateService.formatMessageTimestamp(Date.now() - 86400000);
// "yesterday"

// Day separators
dateService.formatDaySeparator(new Date());
// "today"

dateService.formatDaySeparator(Date.now() - 86400000);
// "yesterday"

dateService.formatDaySeparator(Date.now() - 5 * 86400000);
// "Monday"
```

---

## Chat Timestamp Patterns

### Message List with Day Separators

```javascript
function groupMessagesByDay(messages, dateService) {
  const groups = [];
  let currentDay = null;
  
  for (const message of messages) {
    const dayKey = dateService.formatDate(message.timestamp, 'short');
    
    if (dayKey !== currentDay) {
      currentDay = dayKey;
      groups.push({
        type: 'separator',
        label: dateService.formatDaySeparator(message.timestamp),
        date: dayKey
      });
    }
    
    groups.push({
      type: 'message',
      ...message,
      formattedTime: dateService.formatTime(message.timestamp)
    });
  }
  
  return groups;
}

// Render
function renderChat(messages, dateService) {
  const grouped = groupMessagesByDay(messages, dateService);
  
  return grouped.map(item => {
    if (item.type === 'separator') {
      return `<div class="day-separator">${item.label}</div>`;
    }
    
    return `
      <div class="message">
        <div class="content">${item.content}</div>
        <time class="timestamp">${item.formattedTime}</time>
      </div>
    `;
  }).join('');
}
```

### Live Updating Timestamps

```javascript
class LiveTimestamps {
  constructor(dateService) {
    this.dateService = dateService;
    this.elements = new Map();
    this.intervalId = null;
  }
  
  register(element, timestamp) {
    this.elements.set(element, timestamp);
    this.update(element, timestamp);
    this.startIfNeeded();
  }
  
  unregister(element) {
    this.elements.delete(element);
    this.stopIfEmpty();
  }
  
  update(element, timestamp) {
    element.textContent = this.dateService.formatMessageTimestamp(timestamp);
  }
  
  updateAll() {
    for (const [element, timestamp] of this.elements) {
      this.update(element, timestamp);
    }
  }
  
  startIfNeeded() {
    if (!this.intervalId && this.elements.size > 0) {
      // Update every minute
      this.intervalId = setInterval(() => this.updateAll(), 60000);
    }
  }
  
  stopIfEmpty() {
    if (this.elements.size === 0 && this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
  
  destroy() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    this.elements.clear();
  }
}

// Usage
const liveTimestamps = new LiveTimestamps(dateService);

// When rendering a message
const timeElement = document.querySelector('.message-time');
liveTimestamps.register(timeElement, message.timestamp);

// When removing a message
liveTimestamps.unregister(timeElement);
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `toLocaleString()` without options | Use `Intl.DateTimeFormat` with explicit options |
| Storing local times in database | Store UTC, convert on display |
| Hardcoding 12/24-hour format | Let `Intl.DateTimeFormat` handle based on locale |
| Not handling timezone changes | Re-render timestamps when timezone updates |
| Calculating relative time incorrectly | Use `Intl.RelativeTimeFormat` |
| Ignoring calendar systems | Support via `calendar` option when needed |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Store timestamps as UTC | Consistent storage, flexible display |
| Cache formatter instances | Creating formatters is expensive |
| Use `numeric: 'auto'` for relative times | More natural language ("yesterday" vs "1 day ago") |
| Update relative times periodically | "5 minutes ago" should update |
| Show full timestamp on hover | Provides precise time when needed |
| Group messages by day | Improves chat readability |

---

## Hands-on Exercise

### Your Task

Build a chat timestamp formatter that displays times appropriately based on how old the message is.

### Requirements

1. Messages under 1 minute: "just now"
2. Messages under 1 hour: "X minutes ago"
3. Messages today: time only (e.g., "2:30 PM")
4. Messages yesterday: "Yesterday, 2:30 PM"
5. Messages this week: weekday (e.g., "Monday, 2:30 PM")
6. Older messages: full date (e.g., "Dec 25, 2024")
7. Support multiple locales

### Expected Result

```javascript
const formatter = new MessageTimestampFormatter('en-US');

formatter.format(justNow);     // "just now"
formatter.format(fiveMinAgo);  // "5 minutes ago"
formatter.format(todayMorning); // "10:30 AM"
formatter.format(yesterday);   // "Yesterday, 3:15 PM"
formatter.format(monday);      // "Monday, 9:00 AM"
formatter.format(lastMonth);   // "Nov 15, 2024"
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Intl.RelativeTimeFormat` with `numeric: 'auto'`
- Compare dates using `toDateString()` for same-day checks
- Create multiple `Intl.DateTimeFormat` instances for different formats
- Calculate difference in milliseconds, then convert to appropriate units

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class MessageTimestampFormatter {
  constructor(locale = 'en', timezone) {
    this.locale = locale;
    this.timezone = timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    
    this.rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
    
    this.timeFormatter = new Intl.DateTimeFormat(locale, {
      timeStyle: 'short',
      timeZone: this.timezone
    });
    
    this.weekdayFormatter = new Intl.DateTimeFormat(locale, {
      weekday: 'long',
      hour: 'numeric',
      minute: '2-digit',
      timeZone: this.timezone
    });
    
    this.dateFormatter = new Intl.DateTimeFormat(locale, {
      dateStyle: 'medium',
      timeZone: this.timezone
    });
  }
  
  format(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    // Under 1 minute
    if (diffMins < 1) {
      return this.rtf.format(0, 'minute');  // "this minute" / locale equivalent
    }
    
    // Under 1 hour
    if (diffMins < 60) {
      return this.rtf.format(-diffMins, 'minute');
    }
    
    // Today
    if (date.toDateString() === now.toDateString()) {
      return this.timeFormatter.format(date);
    }
    
    // Yesterday
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) {
      const yest = this.rtf.format(-1, 'day');
      return `${yest.charAt(0).toUpperCase() + yest.slice(1)}, ${this.timeFormatter.format(date)}`;
    }
    
    // This week
    if (diffDays < 7) {
      return this.weekdayFormatter.format(date);
    }
    
    // Older
    return this.dateFormatter.format(date);
  }
}

// Test
const fmt = new MessageTimestampFormatter('en-US');
console.log(fmt.format(Date.now() - 30000));       // "this minute"
console.log(fmt.format(Date.now() - 5 * 60000));   // "5 minutes ago"
console.log(fmt.format(Date.now() - 86400000));    // "Yesterday, X:XX PM"
```

</details>

---

## Summary

✅ Use `Intl.DateTimeFormat` with explicit options for consistent locale-aware formatting

✅ Use `Intl.RelativeTimeFormat` with `numeric: 'auto'` for natural relative times

✅ Store timestamps as UTC, display in user's timezone via the `timeZone` option

✅ Cache formatter instances—creating them is expensive

✅ Support non-Gregorian calendars when serving users who use them

**Next:** [Number Formatting](./04-number-formatting.md)

---

## Further Reading

- [MDN: Intl.DateTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat) - Full API reference
- [MDN: Intl.RelativeTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat) - Relative time formatting
- [IANA Time Zone Database](https://www.iana.org/time-zones) - Official timezone identifiers
- [Unicode CLDR](https://cldr.unicode.org/) - Calendar and locale data

<!--
Sources Consulted:
- MDN Intl.DateTimeFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat
- MDN Intl.RelativeTimeFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat
- IANA Time Zone Database: https://www.iana.org/time-zones
-->
