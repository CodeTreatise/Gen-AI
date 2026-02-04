---
title: "Mobile-Specific Patterns"
---

# Mobile-Specific Patterns

Mobile devices represent the primary way most users interact with chat applications and conversational AI. Building mobile-first chat interfaces requires understanding the unique constraints and opportunities that touch screens, virtual keyboards, and mobile operating systems provide.

This lesson explores the specialized patterns that make chat interfaces feel native and responsive on mobile devices—from touch-optimized interaction design to leveraging device capabilities like haptic feedback and native sharing.

---

## Learning Objectives

By the end of this lesson, you will be able to:

- ✅ Design touch-optimized interfaces with proper target sizes and gesture handling
- ✅ Handle virtual keyboard appearance and manage viewport changes
- ✅ Implement safe area handling for notched and edge-to-edge displays
- ✅ Add haptic feedback to enhance user interactions
- ✅ Integrate native sharing capabilities using the Web Share API
- ✅ Leverage mobile-specific input methods like voice and camera

---

## Prerequisites

Before starting this lesson, you should:

- Understand JavaScript event handling and the DOM
- Have experience building responsive web interfaces
- Be familiar with CSS viewport units and media queries
- Have completed the previous lessons on chat interface fundamentals

---

## What We'll Cover

### [1. Touch-Optimized Interfaces](./01-touch-optimized-interfaces.md)

Touch interaction is fundamentally different from mouse-based interfaces. We'll explore:

- Minimum touch target sizes (44×44px per Apple HIG, 48×48px per Material Design)
- Pointer events for unified touch and mouse handling
- Implementing swipe gestures for message actions
- Long-press menus for contextual options
- Visual and haptic touch feedback

### [2. Virtual Keyboard Handling](./02-virtual-keyboard-handling.md)

The virtual keyboard transforms the mobile viewport significantly. Learn to:

- Use the Visual Viewport API to track actual visible area
- Leverage the VirtualKeyboard API for fine-grained control
- Implement smooth input focus with auto-scrolling
- Handle keyboard dismiss patterns elegantly
- Use CSS environment variables for keyboard-aware layouts

### [3. Mobile Viewport Considerations](./03-mobile-viewport-considerations.md)

Modern phones have notches, dynamic islands, and edge-to-edge displays. Master:

- Safe area insets with `env()` CSS function
- Dynamic viewport height units (`dvh`, `svh`, `lvh`)
- Handling orientation changes gracefully
- Browser chrome considerations (address bar, bottom nav)

### [4. Haptic Feedback](./04-haptic-feedback.md)

Physical feedback enhances perceived responsiveness. Discover:

- The Vibration API and `navigator.vibrate()`
- Creating feedback patterns for different actions
- Platform-specific haptic behaviors
- When (and when not) to use haptic feedback

### [5. Native Share Integration](./05-native-share-integration.md)

Let users share conversations and AI responses natively:

- The Web Share API (`navigator.share()`)
- Feature detection with `navigator.canShare()`
- Preparing content for sharing
- Implementing fallback mechanisms for unsupported browsers

### [6. Mobile-Specific Input Methods](./06-mobile-specific-input.md)

Mobile devices offer unique input capabilities:

- Voice input with speech recognition
- Camera integration for image input
- Quick reply buttons and suggestion chips
- Working with predictive text and autocomplete

---

## Why Mobile-First Matters for Chat

> **🤖 AI Context:** Most AI assistant interactions happen on mobile devices. Users expect the same natural, responsive experience they get from native messaging apps—anything less creates friction that reduces engagement.

Mobile chat applications face unique challenges:

| Challenge | Desktop | Mobile |
|-----------|---------|--------|
| Input area | Large keyboard, precise mouse | Virtual keyboard, touch |
| Screen space | Abundant | Limited, keyboard takes ~40% |
| Interactions | Click, hover, right-click | Tap, swipe, long-press |
| Feedback | Visual (cursor changes) | Haptic + visual |
| Sharing | Copy/paste workflows | Native share sheets |

---

## Mobile Chat Interface Anatomy

A well-designed mobile chat interface considers every pixel:

```
┌─────────────────────────────────┐
│ ░░░░░ Status Bar ░░░░░░░░░░░░░ │ ← Safe area inset top
├─────────────────────────────────┤
│ ← Back   AI Assistant   ⋮ Menu │ ← Navigation bar
├─────────────────────────────────┤
│                                 │
│  ┌───────────────────┐          │
│  │ AI response here  │          │
│  └───────────────────┘          │
│                                 │
│          ┌───────────────────┐  │
│          │ User message      │  │
│          └───────────────────┘  │
│                                 │ ← Messages scroll area
│  ┌───────────────────┐          │
│  │ AI response with  │          │
│  │ longer content... │          │
│  └───────────────────┘          │
│                                 │
├─────────────────────────────────┤
│ [Quick Reply] [Another Reply]   │ ← Suggestion chips
├─────────────────────────────────┤
│ 🎤 │ Type message...     │ ➤   │ ← Input area (44px+ height)
├─────────────────────────────────┤
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← Safe area inset bottom
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   (home indicator on iOS)
└─────────────────────────────────┘
```

---

## Browser Support Overview

| Feature | Chrome Android | Safari iOS | Firefox Android |
|---------|----------------|------------|-----------------|
| Pointer Events | ✅ 55+ | ✅ 13+ | ✅ 59+ |
| Visual Viewport API | ✅ 61+ | ✅ 13+ | ✅ 91+ |
| VirtualKeyboard API | ✅ 94+ | ❌ | ❌ |
| `env()` safe-area-inset | ✅ 69+ | ✅ 11+ | ✅ 65+ |
| Dynamic viewport units | ✅ 108+ | ✅ 15.4+ | ✅ 101+ |
| Vibration API | ✅ 32+ | ❌ | ⚠️ Limited |
| Web Share API | ✅ 61+ | ✅ 12.2+ | ✅ 79+ |

> **Note:** VirtualKeyboard API and Vibration API have limited cross-platform support. Always implement fallbacks for Safari iOS.

---

## Development Setup

For testing mobile patterns, you'll want:

1. **Real Device Testing** — Simulators don't fully replicate touch, haptics, or keyboard behavior
2. **Remote Debugging**:
   - Chrome Android: `chrome://inspect` on desktop Chrome
   - Safari iOS: Safari > Develop > [Your Device]
3. **Responsive Design Mode** — Good for layout, but test interactions on real devices

---

## Summary

Mobile-specific patterns transform a chat interface from merely "working on mobile" to feeling native and responsive. In this lesson series, you'll learn:

✅ How touch targets and gestures differ from desktop interactions  
✅ Techniques for handling virtual keyboards without layout chaos  
✅ Safe area and viewport handling for modern edge-to-edge displays  
✅ When and how to add haptic feedback  
✅ Native sharing integration for seamless content distribution  
✅ Leveraging mobile-specific input capabilities like voice and camera

**Next:** [Touch-Optimized Interfaces](./01-touch-optimized-interfaces.md)

---

## Lesson Files

| # | Topic | Description |
|---|-------|-------------|
| 00 | [Overview](./00-mobile-patterns-overview.md) | This file — introduction and roadmap |
| 01 | [Touch-Optimized Interfaces](./01-touch-optimized-interfaces.md) | Touch targets, gestures, feedback |
| 02 | [Virtual Keyboard Handling](./02-virtual-keyboard-handling.md) | Viewport adjustment, focus management |
| 03 | [Mobile Viewport Considerations](./03-mobile-viewport-considerations.md) | Safe areas, dynamic units, orientation |
| 04 | [Haptic Feedback](./04-haptic-feedback.md) | Vibration API, feedback patterns |
| 05 | [Native Share Integration](./05-native-share-integration.md) | Web Share API, fallbacks |
| 06 | [Mobile-Specific Input](./06-mobile-specific-input.md) | Voice, camera, quick replies |

---

## Further Reading

- [Apple Human Interface Guidelines: Touch](https://developer.apple.com/design/human-interface-guidelines/inputs)
- [Material Design: Touch targets](https://m3.material.io/foundations/interaction/touch-targets)
- [MDN: Mobile web development](https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Frameworks_libraries/Mobile_development)
- [web.dev: Mobile-first design](https://web.dev/articles/responsive-web-design-basics)

---

<!-- 
Sources Consulted:
- MDN Pointer Events: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
- MDN Visual Viewport API: https://developer.mozilla.org/en-US/docs/Web/API/VisualViewport
- MDN VirtualKeyboard API: https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard_API
- MDN env() CSS function: https://developer.mozilla.org/en-US/docs/Web/CSS/env
- MDN CSS length units: https://developer.mozilla.org/en-US/docs/Web/CSS/length
- MDN Vibration API: https://developer.mozilla.org/en-US/docs/Web/API/Vibration_API
- MDN Web Share API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Share_API
-->
