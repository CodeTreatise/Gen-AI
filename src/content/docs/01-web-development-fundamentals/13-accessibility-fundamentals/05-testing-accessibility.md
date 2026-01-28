---
title: "Testing Accessibility"
---

# Testing Accessibility

## Introduction

Accessibility testing combines automated tools, manual checks, and assistive technology testing. No single tool catches everything—you need a multi-layered approach.

This lesson covers browser tools, automated testing, screen reader basics, and a practical testing workflow.

### What We'll Cover

- Browser accessibility tools
- Screen reader testing basics
- Automated testing (axe, Lighthouse)
- Color contrast checkers
- Testing workflow

### Prerequisites

- Understanding of accessibility principles
- Basic browser DevTools experience
- Access to screen reader (built-in options available)

---

## Browser Developer Tools

### Chrome Accessibility Tools

**Accessibility Inspector:**
1. DevTools → Elements panel
2. Select an element
3. View Accessibility pane in sidebar

Shows:
- Role
- Name (accessible name)
- Description
- States/properties
- Keyboard focusable

**Full Accessibility Tree:**
1. DevTools → Elements
2. Right-click the `<html>` element
3. "Show in accessibility tree"

### Firefox Accessibility Tools

**Accessibility Inspector:**
1. DevTools → Accessibility tab
2. Click "Turn On Accessibility Features"
3. Browse the full accessibility tree

**Checks:**
- Check for Issues (button)
- Filters: contrast, keyboard, text labels

### Edge Accessibility Tools

Similar to Chrome, plus:
- **Issues panel** with accessibility category
- **What's New** accessibility tips

---

## Automated Testing Tools

### Lighthouse

Built into Chrome DevTools:

1. DevTools → Lighthouse tab
2. Check "Accessibility"
3. Click "Analyze page load"
4. Review score and issues

**Lighthouse catches:**
- Missing alt text
- Contrast issues
- Missing form labels
- Heading order problems
- ARIA issues

**Lighthouse misses:**
- Keyboard traps
- Focus management
- Logical reading order
- Complex widget behavior

### axe DevTools

Browser extension with deeper testing:

1. Install axe DevTools extension
2. DevTools → axe DevTools tab
3. "Scan ALL of my page"
4. Review issues by severity

**Features:**
- More rules than Lighthouse
- Intelligent scanning (fewer false positives)
- Issue grouping and guidance
- Can test selected elements

### axe-core in CI/CD

```bash
npm install --save-dev axe-core @axe-core/playwright
```

```javascript
// Playwright test with axe
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('homepage accessibility', async ({ page }) => {
  await page.goto('/');
  
  const results = await new AxeBuilder({ page }).analyze();
  
  expect(results.violations).toEqual([]);
});
```

### Comparing Tools

| Tool | Type | Best For |
|------|------|----------|
| **Lighthouse** | Built-in | Quick audits, metrics |
| **axe DevTools** | Extension | Detailed testing, guidance |
| **WAVE** | Extension | Visual error overlay |
| **axe-core** | Library | CI/CD automation |
| **Pa11y** | CLI | Automated testing |

---

## Screen Reader Testing

### Available Screen Readers

| Screen Reader | Platform | Cost |
|---------------|----------|------|
| **VoiceOver** | macOS, iOS | Free (built-in) |
| **Narrator** | Windows | Free (built-in) |
| **NVDA** | Windows | Free |
| **JAWS** | Windows | Paid |
| **TalkBack** | Android | Free (built-in) |

### VoiceOver Basics (macOS)

**Enable:** `Cmd + F5` or System Preferences → Accessibility → VoiceOver

**Essential commands:**
| Action | Shortcut |
|--------|----------|
| Start/Stop VoiceOver | Cmd + F5 |
| Read next | VO + Right Arrow |
| Read previous | VO + Left Arrow |
| Interact with element | VO + Shift + Down |
| Stop interaction | VO + Shift + Up |
| Read all | VO + A |
| Rotor (navigation) | VO + U |

*VO = Control + Option*

### NVDA Basics (Windows)

**Essential commands:**
| Action | Shortcut |
|--------|----------|
| Start NVDA | Ctrl + Alt + N |
| Stop speech | Ctrl |
| Read next | Down Arrow |
| Read previous | Up Arrow |
| Activate element | Enter |
| Browse mode | NVDA + Space |
| Element list | NVDA + F7 |

### What to Test

1. **Navigate by headings** (H key)
2. **Navigate by landmarks** (D key in NVDA)
3. **Read link text** - Does it make sense alone?
4. **Check forms** - Are labels announced?
5. **Test interactive widgets** - Do states announce?
6. **Check images** - Is alt text appropriate?
7. **Test dynamic content** - Do updates announce?

### Testing Checklist

```
☐ All content announced in logical order
☐ Headings provide outline of page
☐ Links and buttons have clear labels
☐ Forms announce labels, errors, required fields
☐ Images have appropriate alt text
☐ Dynamic updates announce via live regions
☐ Custom widgets announce roles and states
☐ No unexpected content (hidden items, repetition)
```

---

## Color Contrast Testing

### WCAG Requirements

| Content Type | Level AA | Level AAA |
|--------------|----------|-----------|
| Normal text (<18pt) | 4.5:1 | 7:1 |
| Large text (≥18pt or 14pt bold) | 3:1 | 4.5:1 |
| UI components, graphics | 3:1 | 3:1 |

### Tools

**Browser Extensions:**
- WAVE Evaluation Tool
- axe DevTools (includes contrast)
- Colour Contrast Analyser

**Online:**
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Coolors Contrast Checker](https://coolors.co/contrast-checker)

**DevTools:**
- Chrome: Inspect element → Color picker → Contrast ratio
- Firefox: Accessibility → Check for Issues → Contrast

### Testing Procedure

```css
/* Check these combinations */
.text {
  color: #666;      /* Foreground */
  background: #fff; /* Background */
  /* Contrast ratio: 5.74:1 ✓ */
}

.button {
  color: #fff;
  background: #3498db;
  /* Contrast ratio: 4.51:1 ✓ (barely) */
}

.error {
  color: #e74c3c;
  background: #fff;
  /* Contrast ratio: 4.0:1 ✗ (fails AA for normal text) */
}
```

### Color Blindness

Test with simulators:
- Chrome DevTools: Rendering → Emulate vision deficiencies
- Extension: NoCoffee Vision Simulator

**Common types:**
- Protanopia (red-blind)
- Deuteranopia (green-blind)
- Tritanopia (blue-blind)
- Achromatopsia (no color)

---

## Testing Workflow

### 1. During Development

```
Write code
    ↓
Run ESLint a11y rules ←─── Automated
    ↓
Tab through manually  ←─── Manual
    ↓
Check contrast        ←─── Manual/Automated
    ↓
Continue development
```

**ESLint a11y:**
```bash
npm install --save-dev eslint-plugin-jsx-a11y
```

```javascript
// .eslintrc.js
module.exports = {
  plugins: ['jsx-a11y'],
  extends: ['plugin:jsx-a11y/recommended']
};
```

### 2. Before Committing

```bash
# Run axe tests
npm run test:a11y

# Run Lighthouse CI
npm run lighthouse
```

### 3. Code Review

- Check new components for:
  - Semantic HTML
  - Keyboard support
  - ARIA when needed
  - Color contrast
  - Focus management

### 4. QA Testing

```
1. Run automated scan (axe/Lighthouse)
2. Keyboard-only navigation test
3. Screen reader spot checks
4. Mobile accessibility check
5. Zoom to 200% check
```

### 5. Regular Audits

- Monthly automated scans
- Quarterly manual audits
- Annual third-party review

---

## Common Issues and Fixes

| Issue | Detection | Fix |
|-------|-----------|-----|
| Missing alt text | Automated | Add descriptive alt or alt="" |
| Low contrast | Automated | Adjust colors to meet 4.5:1 |
| Missing form labels | Automated | Add `<label for="...">` |
| Skipped headings | Automated | Fix heading hierarchy |
| No focus indicator | Manual | Add `:focus` styles |
| Keyboard trap | Manual | Ensure Escape exits |
| Missing skip link | Manual | Add skip to main content |
| Inaccessible custom widget | Screen reader | Add ARIA roles/states |

---

## Testing Resources

### Browser Extensions

- **axe DevTools** - Comprehensive testing
- **WAVE** - Visual error overlay
- **Lighthouse** - Built into Chrome
- **HeadingsMap** - Visualize heading structure

### Automated Testing Libraries

```javascript
// Jest + jest-axe
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);

test('component is accessible', async () => {
  const { container } = render(<MyComponent />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### Checklists

- [WebAIM WCAG 2 Checklist](https://webaim.org/standards/wcag/checklist)
- [A11y Project Checklist](https://www.a11yproject.com/checklist/)
- [Vox Product Accessibility Guidelines](https://accessibility.voxmedia.com/)

---

## Hands-on Exercise

### Your Task

Audit any website for accessibility:

1. **Automated:** Run Lighthouse accessibility audit
2. **Keyboard:** Tab through the entire page
3. **Screen reader:** Use VoiceOver or NVDA to navigate
4. **Contrast:** Check text contrast in DevTools

### Report Template

```markdown
## Accessibility Audit: [Site Name]

### Automated Testing
- Tool: Lighthouse
- Score: __/100
- Critical issues: 

### Keyboard Navigation
- ☐ All links/buttons focusable
- ☐ Focus visible
- ☐ Logical tab order
- ☐ Skip link present
Issues:

### Screen Reader
- ☐ Headings provide structure
- ☐ Links/buttons labeled
- ☐ Forms accessible
- ☐ Images have alt text
Issues:

### Contrast
- ☐ Text passes 4.5:1
- ☐ Large text passes 3:1
Issues:

### Recommendations
1. 
2.
3.
```

---

## Summary

✅ **Automated tools** catch ~30% of issues
✅ **Manual testing** is essential for keyboard and interaction
✅ **Screen reader testing** reveals real user experience
✅ **Contrast checkers** ensure text is readable
✅ **Integrate testing** into development workflow
✅ **Regular audits** maintain accessibility over time

**Back to:** [Accessibility Fundamentals Overview](./00-accessibility-fundamentals.md)

---

## Further Reading

- [WebAIM Screen Reader Survey](https://webaim.org/projects/screenreadersurvey9/)
- [Deque University (Free)](https://dequeuniversity.com/)
- [axe DevTools](https://www.deque.com/axe/devtools/)
- [NVDA User Guide](https://www.nvaccess.org/files/nvda/documentation/userGuide.html)

<!-- 
Sources Consulted:
- WebAIM: https://webaim.org/
- Deque axe: https://www.deque.com/axe/
-->
