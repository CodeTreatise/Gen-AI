---
title: "Browser Developer Tools"
---

# Browser Developer Tools

## Introduction

Browser Developer Tools (DevTools) are built into every modern browser. They provide real-time access to a web page's DOM, styles, JavaScript, network requests, and performance metrics—all without modifying your source code.

This lesson covers the core DevTools panels: Elements for DOM inspection, Styles for CSS debugging, and responsive design tools for testing across devices.

### What We'll Cover

- Opening DevTools
- Elements panel (DOM inspection)
- Styles panel (CSS debugging)
- Computed styles
- Box model visualization
- Responsive design mode
- Device emulation

### Prerequisites

- Basic HTML and CSS knowledge
- A modern browser (Chrome recommended for this lesson)

---

## Opening DevTools

### Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| Open DevTools | `F12` or `Ctrl+Shift+I` | `Cmd+Option+I` |
| Inspect element | `Ctrl+Shift+C` | `Cmd+Shift+C` |
| Open Console | `Ctrl+Shift+J` | `Cmd+Option+J` |
| Open Command Menu | `Ctrl+Shift+P` | `Cmd+Shift+P` |

### Right-Click Menu

Right-click any element on a page and select **"Inspect"** to:
1. Open DevTools
2. Jump directly to that element in the Elements panel
3. See its applied styles

### Docking Options

DevTools can be docked in different positions:

| Position | Best For |
|----------|----------|
| **Right** | Wide monitors, horizontal layouts |
| **Bottom** | Narrow monitors, vertical layouts |
| **Left** | Right-to-left layouts |
| **Separate window** | Multi-monitor setups |

Toggle docking: Click the **⋮** menu → Dock side.

---

## Elements Panel

The Elements panel shows the live DOM tree. You can inspect, edit, and debug HTML in real-time.

### Navigating the DOM

```
Elements panel structure:
├── <html>
│   ├── <head>
│   │   ├── <title>
│   │   ├── <meta>
│   │   └── <link>
│   └── <body>
│       ├── <header>
│       ├── <main>
│       └── <footer>
```

**Navigation shortcuts:**
- `↑/↓` - Move between elements
- `←` - Collapse element
- `→` - Expand element
- `Enter` - Edit element

### Inspecting Elements

1. Click the **inspect icon** (cursor in box) or press `Ctrl+Shift+C`
2. Hover over the page—elements highlight with their box model
3. Click to select an element
4. The element appears in the Elements panel with its styles

### Live DOM Editing

Double-click to edit:

```html
<!-- Original -->
<h1>Welcome</h1>

<!-- After editing (changes apply instantly) -->
<h1 class="title">Welcome to My Site</h1>
```

**Edit options:**
- **Edit as HTML** - Right-click → Edit as HTML
- **Add attribute** - Double-click the tag
- **Delete element** - Press `Delete` or right-click → Delete element
- **Copy element** - Right-click → Copy → Copy element

### Element State

Force element states for debugging hover, focus, etc.:

1. Right-click element → **Force state**
2. Check `:hover`, `:active`, `:focus`, `:visited`, or `:focus-within`

```css
/* Now you can see and edit hover styles without hovering */
.button:hover {
  background: blue;
}
```

### DOM Breakpoints

Pause JavaScript when the DOM changes:

1. Right-click element → **Break on**
2. Choose:
   - **Subtree modifications** - Child elements change
   - **Attribute modifications** - Attributes change
   - **Node removal** - Element is removed

Useful for debugging dynamic UI changes.

---

## Styles Panel

The Styles panel shows all CSS affecting the selected element.

### Reading the Cascade

Styles are listed in order of specificity (highest first):

```
Styles panel:
├── element.style { }           ← Inline styles (highest priority)
├── .button.primary { }         ← More specific selector
├── .button { }                 ← Less specific selector
├── button { }                  ← Element selector
└── user agent stylesheet       ← Browser defaults (lowest)
```

Crossed-out properties are overridden by higher-specificity rules.

### Live CSS Editing

Click any value to edit:

```css
/* Click on "16px" to change it */
.container {
  font-size: 16px;  /* Change to 20px and see instant update */
  color: #333;
}
```

**Edit features:**
- **Color picker** - Click color swatches to open
- **Increment values** - Use `↑/↓` arrows to adjust numbers
- **Toggle properties** - Click checkbox to enable/disable
- **Add properties** - Click empty space in a rule

### Adding New Rules

1. Click the **+** button in Styles panel
2. A new rule is created with the selected element's selector
3. Add properties as needed

```css
/* DevTools creates this */
.my-element {
  /* Add your properties here */
}
```

### Filter Styles

Use the **Filter** box to search for specific properties:

- Type `margin` to show only margin-related properties
- Type `color` to find all color properties
- Helps find overridden styles quickly

---

## Computed Styles

The **Computed** tab shows the final, resolved values after the cascade.

### Final Values

```
Computed styles for .button:
├── background-color: rgb(37, 99, 235)
├── border-radius: 8px
├── color: rgb(255, 255, 255)
├── display: inline-flex
├── font-size: 14px
├── height: 40px
├── padding: 8px 16px
└── width: auto
```

### Trace Property Origin

Click the arrow next to any computed value to see:
1. Where the value comes from
2. The cascade of rules that were overridden
3. The exact file and line number

### Show All

Check **"Show all"** to see every CSS property (even inherited/default ones).

---

## Box Model Visualization

Every element is a box with content, padding, border, and margin.

### Box Model Diagram

```
┌─────────────────────────────────────┐
│              margin                 │
│   ┌─────────────────────────────┐   │
│   │         border              │   │
│   │   ┌─────────────────────┐   │   │
│   │   │     padding         │   │   │
│   │   │   ┌─────────────┐   │   │   │
│   │   │   │   content   │   │   │   │
│   │   │   │  400 × 200  │   │   │   │
│   │   │   └─────────────┘   │   │   │
│   │   │                     │   │   │
│   │   └─────────────────────┘   │   │
│   │                             │   │
│   └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

### Interactive Box Model

In the Computed tab, hover over each section:
- **Blue** - Content dimensions
- **Green** - Padding
- **Yellow/Orange** - Border
- **Orange/Tan** - Margin

Click values to edit them live!

### Highlight on Page

When hovering over the box model or any element:
- Page highlights the actual dimensions
- Shows grid/flexbox overlays if applicable
- Displays exact pixel measurements

---

## Responsive Design Mode

Test your site across different screen sizes without real devices.

### Activating Device Mode

- Click the **device icon** in DevTools toolbar
- Or press `Ctrl+Shift+M` (Windows/Linux) / `Cmd+Shift+M` (Mac)

### Viewport Controls

| Control | Purpose |
|---------|---------|
| Device dropdown | Preset device sizes (iPhone, iPad, etc.) |
| Dimensions | Custom width × height |
| Zoom | Viewport zoom level |
| Throttling | Simulate slower connections |
| Rotate | Switch portrait/landscape |

### Testing Responsive Breakpoints

1. Grab the viewport edge and drag
2. Watch your CSS breakpoints trigger
3. The current viewport size displays at the top

```css
/* See these kick in as you resize */
@media (max-width: 768px) {
  .sidebar { display: none; }
}

@media (max-width: 480px) {
  .header { flex-direction: column; }
}
```

### Responsive Ruler

A ruler along the top shows common breakpoints. Click to snap to:
- 320px (small mobile)
- 375px (iPhone)
- 768px (tablet)
- 1024px (laptop)
- 1440px (desktop)

---

## Device Emulation

Beyond viewport size, emulate full device characteristics.

### Touch Simulation

In device mode:
- Mouse clicks become touch events
- Touch gestures work (pinch, swipe via Shift+drag)
- Touch-specific CSS applies (`:hover` behavior changes)

### Device Presets

Popular presets include:

| Device | Resolution | Pixel Ratio |
|--------|------------|-------------|
| iPhone 14 Pro | 393 × 852 | 3x |
| iPhone SE | 375 × 667 | 2x |
| iPad Pro | 1024 × 1366 | 2x |
| Galaxy S20 | 360 × 800 | 3x |
| Pixel 7 | 412 × 915 | 2.625x |

### Custom Devices

Add your own device:
1. Open device dropdown → **Edit**
2. Click **Add custom device**
3. Set width, height, pixel ratio, user agent

### User Agent Emulation

Presets include device-specific user agents. Useful for testing:
- Mobile-specific redirects
- Feature detection based on UA
- Server-side rendering differences

---

## Practical Exercises

### Exercise 1: Debug a Layout Issue

1. Right-click on a misaligned element → **Inspect**
2. Look at the box model for unexpected margin/padding
3. Check the Styles panel for overriding rules
4. Use the checkbox to toggle properties on/off

### Exercise 2: Test Responsive Design

1. Open Device Mode (`Ctrl+Shift+M`)
2. Select "iPhone 14 Pro" from presets
3. Navigate your site and note layout issues
4. Drag the viewport to find your breakpoints

### Exercise 3: Edit Live CSS

1. Inspect a button element
2. In Styles panel, change:
   - `background-color` to something new
   - `padding` using the arrow keys
   - Add `transform: scale(1.1)` on hover (force `:hover` state)
3. Copy the final CSS to your source file

---

## Tips and Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` in Elements | Search the DOM |
| `H` | Hide selected element |
| `Ctrl+Z` | Undo DOM/style changes |
| `Ctrl+Shift+C` | Inspect element mode |
| Hold `Shift` + adjust colors | Cycle color formats |

### Command Menu

Press `Ctrl+Shift+P` to access everything:
- "Capture screenshot" - Full page or node screenshot
- "Show rendering" - Paint flashing, FPS meter
- "Disable JavaScript" - Test no-JS fallbacks

---

## Summary

✅ **Elements panel** shows the live DOM tree
✅ **Styles panel** displays the CSS cascade with live editing
✅ **Computed tab** shows final resolved values
✅ **Box model** visualizes content, padding, border, margin
✅ **Device mode** tests responsive layouts
✅ **Device emulation** simulates touch, screen density, and user agents

**Next:** [Console Debugging Techniques](./02-console-debugging.md)

---

## Further Reading

- [Chrome DevTools - Elements Panel](https://developer.chrome.com/docs/devtools/dom/)
- [Chrome DevTools - CSS](https://developer.chrome.com/docs/devtools/css/)
- [Chrome DevTools - Device Mode](https://developer.chrome.com/docs/devtools/device-mode/)

<!-- 
Sources Consulted:
- Chrome DevTools Documentation: https://developer.chrome.com/docs/devtools/
- MDN Developer Tools: https://developer.mozilla.org/en-US/docs/Learn/Common_questions/Tools_and_setup/What_are_browser_developer_tools
-->
