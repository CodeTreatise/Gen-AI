---
title: "Conversation Layout Patterns"
---

# Conversation Layout Patterns

## Introduction

The layout of a chat interface determines how users perceive and interact with AI conversations. A well-designed layout guides the eye, establishes visual hierarchy, and creates a comfortable reading experience—even during long conversations.

In this lesson, we'll explore the most common layout patterns used in modern chat applications and learn when to apply each one.

### What We'll Cover

- Full-width vs centered container layouts
- Side-by-side comparison layouts for multi-model interfaces
- Split-screen designs with context panels
- Single-column mobile-first design patterns
- CSS implementation for each layout type

### Prerequisites

- CSS Flexbox and Grid fundamentals ([Unit 1](../../../01-web-development-fundamentals/02-css-fundamentals/00-css-fundamentals.md))
- Responsive design basics
- Understanding of viewport units

---

## Full-Width vs Centered Layouts

The first decision in chat interface design is whether messages should span the full viewport width or be constrained to a centered container.

### Full-Width Layout

Messages extend edge-to-edge, maximizing horizontal space:

```css
.chat-container {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.message {
  width: 100%;
  padding: 1rem;
  margin-bottom: 0.5rem;
}
```

```html
<div class="chat-container">
  <div class="message-list">
    <div class="message user">How do I center a div?</div>
    <div class="message ai">You can use Flexbox or Grid...</div>
  </div>
</div>
```

**When to use full-width:**
- Mobile devices where space is limited
- Code-heavy conversations needing horizontal room
- Embedded chat widgets in sidebars

### Centered Container Layout

Messages are constrained to a maximum width and centered:

```css
.chat-container {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.message-list {
  width: 100%;
  max-width: 48rem; /* 768px */
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.message {
  max-width: 85%;
  padding: 1rem;
  margin-bottom: 0.5rem;
}

.message.user {
  margin-left: auto; /* Push to right */
}

.message.ai {
  margin-right: auto; /* Keep on left */
}
```

**When to use centered layout:**
- Long-form text conversations (better readability)
- Desktop applications with wide screens
- When mimicking popular apps like ChatGPT

> **Note:** The optimal line length for reading is 50-75 characters. A `max-width` of 48rem (768px) typically achieves this with standard font sizes.

---

## Side-by-Side Comparison Layouts

When comparing responses from multiple AI models, a side-by-side layout allows direct comparison:

```css
.comparison-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  height: 100vh;
  padding: 1rem;
}

.chat-panel {
  display: flex;
  flex-direction: column;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
}

.panel-header {
  padding: 0.75rem 1rem;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
  font-weight: 600;
}

.panel-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

/* Responsive: Stack on mobile */
@media (max-width: 768px) {
  .comparison-container {
    grid-template-columns: 1fr;
  }
}
```

```html
<div class="comparison-container">
  <div class="chat-panel">
    <div class="panel-header">GPT-4o</div>
    <div class="panel-messages">
      <!-- Messages here -->
    </div>
  </div>
  <div class="chat-panel">
    <div class="panel-header">Claude 3.5</div>
    <div class="panel-messages">
      <!-- Messages here -->
    </div>
  </div>
</div>
```

**Use cases:**
- Model comparison tools (like Chatbot Arena)
- A/B testing different prompts
- Educational interfaces showing model differences

---

## Split-Screen with Context Panels

Many AI coding assistants use a split-screen layout with a context panel showing relevant files, code, or documents:

```css
.split-layout {
  display: grid;
  grid-template-columns: 1fr 400px;
  height: 100vh;
}

.chat-area {
  display: flex;
  flex-direction: column;
  border-right: 1px solid #e5e7eb;
}

.context-panel {
  display: flex;
  flex-direction: column;
  background: #f9fafb;
}

.context-header {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.context-content {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

/* Resizable panel */
.split-layout.resizable {
  grid-template-columns: 1fr var(--panel-width, 400px);
}

/* Collapsible panel */
@media (max-width: 1024px) {
  .split-layout {
    grid-template-columns: 1fr;
  }
  
  .context-panel {
    position: fixed;
    right: 0;
    top: 0;
    bottom: 0;
    width: 400px;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    z-index: 100;
  }
  
  .context-panel.open {
    transform: translateX(0);
  }
}
```

```html
<div class="split-layout">
  <div class="chat-area">
    <div class="message-list">
      <!-- Chat messages -->
    </div>
    <div class="input-area">
      <!-- Input form -->
    </div>
  </div>
  <div class="context-panel">
    <div class="context-header">
      <span>📄</span>
      <span>index.js</span>
    </div>
    <div class="context-content">
      <!-- Code preview or document -->
    </div>
  </div>
</div>
```

**Common context panel uses:**
- Code file previews (GitHub Copilot)
- Document references (RAG applications)
- Search results with sources
- Tool invocation outputs

---

## Single-Column Mobile-First Design

The mobile-first approach starts with a single-column layout and expands for larger screens:

```css
/* Base: Mobile-first single column */
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  height: 100dvh; /* Dynamic viewport height for mobile */
}

.chat-header {
  padding: 0.75rem 1rem;
  background: #fff;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  /* Safe area for notched phones */
  padding-top: max(0.75rem, env(safe-area-inset-top));
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  /* Prevent iOS bounce */
  overscroll-behavior: contain;
}

.input-area {
  padding: 0.75rem 1rem;
  background: #fff;
  border-top: 1px solid #e5e7eb;
  /* Safe area for home indicator */
  padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
}

/* Tablet: Add some constraints */
@media (min-width: 768px) {
  .message-list {
    padding: 1.5rem 2rem;
  }
  
  .message {
    max-width: 75%;
  }
}

/* Desktop: Center and constrain */
@media (min-width: 1024px) {
  .chat-app {
    max-width: 900px;
    margin: 0 auto;
    border-left: 1px solid #e5e7eb;
    border-right: 1px solid #e5e7eb;
  }
}
```

### Handling the Virtual Keyboard

On mobile, the virtual keyboard affects layout. Use the Visual Viewport API:

```javascript
// Adjust layout when virtual keyboard appears
if ('visualViewport' in window) {
  window.visualViewport.addEventListener('resize', () => {
    const viewport = window.visualViewport;
    const keyboardHeight = window.innerHeight - viewport.height;
    
    document.documentElement.style.setProperty(
      '--keyboard-height',
      `${keyboardHeight}px`
    );
  });
}
```

```css
.input-area {
  /* Adjust for keyboard */
  margin-bottom: var(--keyboard-height, 0);
  transition: margin-bottom 0.15s ease;
}
```

> **Warning:** On iOS Safari, the virtual keyboard pushes content up by default. Use `interactive-widget=resizes-content` in your viewport meta tag for predictable behavior.

---

## Layout Decision Framework

Use this framework to choose the right layout:

| Factor | Full-Width | Centered | Split-Screen | Comparison |
|--------|------------|----------|--------------|------------|
| **Screen size** | Mobile, small | Desktop, large | Desktop, large | Desktop only |
| **Content type** | Code, data | Text, chat | Code + chat | A/B testing |
| **Primary use** | Embedded widgets | General chat | IDE integrations | Model eval |
| **Complexity** | Low | Low | Medium | Medium |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use `100dvh` for mobile viewport height | Use `100vh` alone (breaks on mobile) |
| Add safe-area insets for notched devices | Ignore `env(safe-area-inset-*)` |
| Constrain line length for readability | Allow text to span full wide screens |
| Make layouts responsive with breakpoints | Build desktop-only layouts |
| Use CSS Grid for complex layouts | Nest multiple Flexbox containers |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Messages jump when keyboard opens | Use Visual Viewport API |
| Content hidden behind notch/home bar | Add `env(safe-area-inset-*)` padding |
| Horizontal scrolling on mobile | Use `max-width: 100%` on messages |
| Layout breaks at certain widths | Test at all common breakpoints |
| Scroll position lost on resize | Store/restore scroll position in JS |

---

## Hands-on Exercise

### Your Task

Build a responsive chat layout that:
1. Displays as a single centered column on mobile
2. Adds a collapsible context panel on tablet/desktop
3. Properly handles safe areas for mobile devices

### Requirements

1. Create HTML structure with chat area and context panel
2. Style with CSS Grid for the split layout
3. Add media queries for responsive breakpoints
4. Include safe-area handling for mobile

### Expected Result

- On mobile: Full-screen single column chat
- On tablet (768px+): Chat with toggleable side panel
- On desktop (1024px+): Persistent side panel

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `grid-template-columns` with media queries
- The context panel can use `position: fixed` on mobile
- Remember `100dvh` for dynamic viewport height
- Use CSS custom properties for the panel width

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, interactive-widget=resizes-content">
  <title>Responsive Chat Layout</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    :root {
      --panel-width: 350px;
    }
    
    .app {
      display: grid;
      grid-template-columns: 1fr;
      height: 100vh;
      height: 100dvh;
    }
    
    .chat-area {
      display: flex;
      flex-direction: column;
    }
    
    .chat-header {
      padding: 0.75rem 1rem;
      padding-top: max(0.75rem, env(safe-area-inset-top));
      background: #fff;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    
    .message-list {
      flex: 1;
      overflow-y: auto;
      padding: 1rem;
      overscroll-behavior: contain;
    }
    
    .message {
      max-width: 85%;
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
      border-radius: 1rem;
    }
    
    .message.user {
      background: #3b82f6;
      color: white;
      margin-left: auto;
      border-bottom-right-radius: 0.25rem;
    }
    
    .message.ai {
      background: #f3f4f6;
      border-bottom-left-radius: 0.25rem;
    }
    
    .input-area {
      padding: 0.75rem 1rem;
      padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
      background: #fff;
      border-top: 1px solid #e5e7eb;
    }
    
    .input-area input {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 1px solid #e5e7eb;
      border-radius: 1.5rem;
      font-size: 1rem;
    }
    
    .context-panel {
      display: none;
      flex-direction: column;
      background: #f9fafb;
      border-left: 1px solid #e5e7eb;
    }
    
    .toggle-panel {
      display: none;
      padding: 0.5rem;
      background: none;
      border: none;
      cursor: pointer;
      font-size: 1.25rem;
    }
    
    /* Tablet: Toggleable panel */
    @media (min-width: 768px) {
      .toggle-panel {
        display: block;
      }
      
      .context-panel {
        display: flex;
        position: fixed;
        right: 0;
        top: 0;
        bottom: 0;
        width: var(--panel-width);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        z-index: 100;
      }
      
      .context-panel.open {
        transform: translateX(0);
      }
      
      .message {
        max-width: 75%;
      }
    }
    
    /* Desktop: Persistent panel */
    @media (min-width: 1024px) {
      .app {
        grid-template-columns: 1fr var(--panel-width);
      }
      
      .context-panel {
        display: flex;
        position: static;
        transform: none;
      }
      
      .toggle-panel {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="chat-area">
      <header class="chat-header">
        <h1>AI Chat</h1>
        <button class="toggle-panel" onclick="togglePanel()">📄</button>
      </header>
      <div class="message-list">
        <div class="message user">How do I create a responsive layout?</div>
        <div class="message ai">You can use CSS Grid with media queries...</div>
      </div>
      <div class="input-area">
        <input type="text" placeholder="Type a message...">
      </div>
    </div>
    <aside class="context-panel">
      <div class="chat-header">
        <span>Context</span>
      </div>
      <div class="message-list">
        <p>Related files and references appear here.</p>
      </div>
    </aside>
  </div>
  
  <script>
    function togglePanel() {
      document.querySelector('.context-panel').classList.toggle('open');
    }
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ **Full-width layouts** maximize space for code and mobile devices  
✅ **Centered layouts** improve readability with constrained line lengths  
✅ **Split-screen layouts** combine chat with context panels for IDEs  
✅ **Comparison layouts** enable side-by-side model evaluation  
✅ **Mobile-first design** ensures layouts work across all devices

---

## Further Reading

- [CSS Grid Layout Guide](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Visual Viewport API](https://developer.mozilla.org/en-US/docs/Web/API/Visual_Viewport_API)
- [Safe Area Insets](https://developer.mozilla.org/en-US/docs/Web/CSS/env)
- [Dynamic Viewport Units](https://web.dev/blog/viewport-units)

---

**Previous:** [Chat Interface Design Principles](./00-chat-interface-design-principles.md)  
**Next:** [Message Bubble Design](./02-message-bubble-design.md)

<!-- 
Sources Consulted:
- MDN CSS Grid: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout
- web.dev Viewport Units: https://web.dev/blog/viewport-units
- CSS-Tricks Grid Guide: https://css-tricks.com/snippets/css/complete-guide-grid/
-->
