---
title: "Development Tools"
---

# Development Tools

## Introduction

Your browser is more than a window to the web—it's a complete development environment. Browser DevTools let you inspect, debug, and profile web applications in real-time. Combined with version control (Git), you have everything needed for professional web development.

This lesson covers the essential tools every web developer must master: browser DevTools for debugging and performance analysis, and Git for tracking changes and collaboration.

### What We'll Cover

| Lesson | Topic |
|--------|-------|
| [01](./01-browser-developer-tools.md) | Browser developer tools |
| [02](./02-console-debugging.md) | Console debugging techniques |
| [03](./03-network-tab-analysis.md) | Network tab analysis |
| [04](./04-performance-profiling.md) | Performance profiling basics |
| [05](./05-advanced-devtools.md) | Advanced DevTools features |
| [06](./06-version-control-git.md) | Version control with Git |

### Prerequisites

- Basic HTML, CSS, and JavaScript knowledge
- A modern browser (Chrome, Firefox, or Edge)
- Git installed on your system

---

## Why DevTools Matter

Every professional web developer has DevTools open constantly. They're essential for:

| Task | DevTools Feature |
|------|------------------|
| Fixing layout issues | Elements panel |
| Debugging JavaScript | Console + Debugger |
| Optimizing load times | Network + Performance |
| Finding memory leaks | Memory profiler |
| Testing responsiveness | Device emulation |
| Inspecting API calls | Network tab |

### Opening DevTools

| Browser | Shortcut (Windows/Linux) | Shortcut (Mac) |
|---------|--------------------------|----------------|
| Chrome | `F12` or `Ctrl+Shift+I` | `Cmd+Option+I` |
| Firefox | `F12` or `Ctrl+Shift+I` | `Cmd+Option+I` |
| Edge | `F12` or `Ctrl+Shift+I` | `Cmd+Option+I` |
| Safari | Enable in Preferences | `Cmd+Option+I` |

Or: **Right-click → Inspect** on any element.

---

## DevTools for AI Development

When building AI-powered applications, DevTools become even more valuable:

> **🤖 AI Context:** AI applications often involve streaming responses, large payloads, and complex state management. You'll use DevTools to:
> - Debug streaming SSE/WebSocket connections
> - Profile memory usage with large embeddings
> - Inspect API request/response payloads
> - Measure time-to-first-token latency

```javascript
// Example: Debugging an AI chat request
console.time('AI Response');

const response = await fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ message: 'Hello' })
});

const data = await response.json();
console.timeEnd('AI Response');  // AI Response: 847.23ms

console.table(data.usage);  // Token usage breakdown
```

---

## Quick Reference

### Console Methods

```javascript
console.log('Basic output');
console.warn('Warning message');
console.error('Error message');
console.table([{ id: 1 }, { id: 2 }]);
console.time('label');
console.timeEnd('label');
console.trace();
```

### Common DevTools Panels

| Panel | Purpose |
|-------|---------|
| **Elements** | Inspect/modify DOM and CSS |
| **Console** | JavaScript REPL, logs, errors |
| **Sources** | Debug JavaScript, set breakpoints |
| **Network** | Monitor HTTP requests |
| **Performance** | Profile runtime performance |
| **Memory** | Analyze memory usage |
| **Application** | Storage, service workers, cache |

### Git Essentials

```bash
git init                    # Initialize repo
git add .                   # Stage all changes
git commit -m "message"     # Commit changes
git branch feature          # Create branch
git checkout feature        # Switch branch
git merge feature           # Merge branch
git push origin main        # Push to remote
git pull origin main        # Pull from remote
```

---

## Lesson Structure

Each lesson provides hands-on practice with real debugging scenarios:

1. **Tool overview** with key features
2. **Practical examples** you can follow along
3. **Common use cases** and workflows
4. **Tips and shortcuts** to work faster

**Start with:** [Browser Developer Tools](./01-browser-developer-tools.md)

---

## Further Reading

- [Chrome DevTools Documentation](https://developer.chrome.com/docs/devtools/)
- [Firefox Developer Tools](https://firefox-source-docs.mozilla.org/devtools-user/)
- [Git Documentation](https://git-scm.com/doc)
