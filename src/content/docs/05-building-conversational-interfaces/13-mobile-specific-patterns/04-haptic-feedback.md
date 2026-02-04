---
title: "Haptic Feedback"
---

# Haptic Feedback

## Introduction

Haptic feedback adds a tactile dimension to mobile interfaces, confirming user actions through physical sensation. When implemented thoughtfully, haptic patterns make chat applications feel more responsive and intuitive—a subtle vibration when sending a message, a light pulse when receiving one, or distinct feedback when actions succeed or fail.

This lesson covers the Vibration API and patterns for implementing meaningful haptic feedback in conversational interfaces.

### What We'll Cover

- Understanding the Vibration API
- Creating meaningful vibration patterns
- Appropriate use cases for haptic feedback
- Cross-platform considerations and fallbacks
- Building a reusable haptic feedback system

### Prerequisites

- JavaScript event handling
- Understanding of mobile touch interactions
- Familiarity with feature detection

---

## The Vibration API

The Vibration API provides a simple interface for triggering device vibration. Despite its simplicity, creating meaningful haptic experiences requires thoughtful pattern design.

### Basic API Usage

```javascript
// Single vibration (milliseconds)
navigator.vibrate(100);

// Vibration pattern: vibrate, pause, vibrate, pause...
navigator.vibrate([100, 50, 100]);

// Stop any ongoing vibration
navigator.vibrate(0);
// or
navigator.vibrate([]);
```

### Pattern Format

The `vibrate()` method accepts either a single number or an array:

| Input | Behavior |
|-------|----------|
| `200` | Vibrate for 200ms |
| `[200]` | Vibrate for 200ms |
| `[200, 100, 200]` | Vibrate 200ms, pause 100ms, vibrate 200ms |
| `[100, 50, 100, 50, 100]` | Three quick pulses with pauses |
| `0` or `[]` | Cancel current vibration |

> **Note:** Pattern arrays alternate between vibration and pause durations. The first value is always a vibration duration.

### Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome Android | ✅ Full | Works well |
| Firefox Android | ✅ Full | Works well |
| Samsung Internet | ✅ Full | Works well |
| Safari iOS | ❌ None | Not supported |
| Chrome iOS | ❌ None | Uses Safari engine |
| Desktop browsers | ⚠️ Limited | Requires vibration hardware |

> **Warning:** iOS does not support the Vibration API. On iOS, you must use alternative feedback mechanisms or accept graceful degradation.

### Feature Detection

Always check for API support before using:

```javascript
function canVibrate() {
  return 'vibrate' in navigator;
}

function vibrate(pattern) {
  if (canVibrate()) {
    navigator.vibrate(pattern);
    return true;
  }
  return false;
}
```

---

## Designing Haptic Patterns

Different actions warrant different haptic signatures. Effective patterns are:

- **Distinct** — Each pattern should feel different
- **Appropriate** — Intensity matches the action's significance
- **Brief** — Long vibrations are annoying
- **Consistent** — Same action = same feedback

### Standard Patterns

```javascript
const HapticPatterns = {
  // Single tap feedback - light, instant
  tap: 10,
  
  // Button press confirmation
  click: 25,
  
  // Success/completion
  success: [50, 30, 50],
  
  // Error/failure - stronger, attention-getting
  error: [100, 50, 100, 50, 150],
  
  // Warning - moderate alert
  warning: [75, 50, 75],
  
  // Message sent
  messageSent: [30, 20, 30],
  
  // Message received - subtle notification
  messageReceived: 40,
  
  // Long press activated
  longPress: 50,
  
  // Selection changed
  selection: 15,
  
  // Pull-to-refresh triggered
  refresh: [40, 30, 40, 30, 40]
};
```

### Pattern Design Principles

| Action Type | Pattern Characteristics | Example |
|-------------|------------------------|---------|
| **Confirmation** | Short, light, single pulse | `10-30ms` |
| **Success** | Double/triple pulse, moderate | `[50, 30, 50]` |
| **Error** | Longer, stronger, more pulses | `[100, 50, 100, 50, 150]` |
| **Notification** | Medium single pulse | `40-60ms` |
| **Warning** | Moderate, distinct rhythm | `[75, 50, 75]` |

### Intensity Guidelines

```
Light (10-25ms):     ·         Subtle feedback, frequent actions
                               (typing, scrolling, selections)

Medium (30-60ms):    ••        Standard confirmations
                               (button clicks, toggles)

Strong (75-150ms):   •••       Important events
                               (errors, warnings, completions)

Pattern:             ·-·-·     Complex feedback
                               (success, failure, special events)
```

---

## Implementing a Haptic System

Create a centralized haptic feedback manager:

### Basic Haptic Manager

```javascript
class HapticManager {
  constructor() {
    this.enabled = this.isSupported() && this.getUserPreference();
    
    // Standard patterns
    this.patterns = {
      tap: 10,
      click: 25,
      success: [50, 30, 50],
      error: [100, 50, 100, 50, 150],
      warning: [75, 50, 75],
      messageSent: [30, 20, 30],
      messageReceived: 40,
      longPress: 50,
      selection: 15
    };
  }
  
  isSupported() {
    return 'vibrate' in navigator;
  }
  
  getUserPreference() {
    // Check stored preference
    const stored = localStorage.getItem('haptic-enabled');
    if (stored !== null) {
      return stored === 'true';
    }
    // Default to enabled on supported devices
    return true;
  }
  
  setEnabled(enabled) {
    this.enabled = enabled && this.isSupported();
    localStorage.setItem('haptic-enabled', enabled.toString());
  }
  
  trigger(patternName) {
    if (!this.enabled) return false;
    
    const pattern = this.patterns[patternName];
    if (!pattern) {
      console.warn(`Unknown haptic pattern: ${patternName}`);
      return false;
    }
    
    return this.vibrate(pattern);
  }
  
  vibrate(pattern) {
    if (!this.enabled) return false;
    
    try {
      navigator.vibrate(pattern);
      return true;
    } catch (error) {
      console.warn('Vibration failed:', error);
      return false;
    }
  }
  
  stop() {
    if (this.isSupported()) {
      navigator.vibrate(0);
    }
  }
}

// Singleton instance
const haptics = new HapticManager();
```

### Usage Examples

```javascript
// Button click
document.querySelector('.send-button').addEventListener('click', () => {
  haptics.trigger('click');
  sendMessage();
});

// Message sent successfully
async function sendMessage() {
  try {
    await api.send(message);
    haptics.trigger('messageSent');
  } catch (error) {
    haptics.trigger('error');
    showError(error);
  }
}

// Message received
websocket.on('message', (message) => {
  haptics.trigger('messageReceived');
  displayMessage(message);
});

// Selection change
document.querySelectorAll('.option').forEach(option => {
  option.addEventListener('click', () => {
    haptics.trigger('selection');
    selectOption(option);
  });
});
```

---

## Chat-Specific Haptic Patterns

### Message Interactions

```javascript
class ChatHaptics extends HapticManager {
  constructor() {
    super();
    
    // Extend patterns for chat-specific actions
    this.patterns = {
      ...this.patterns,
      
      // Typing indicator appears
      typingStart: 15,
      
      // Message delivered
      delivered: [20, 30, 20],
      
      // Message read
      read: 20,
      
      // Reaction added
      reaction: [25, 15, 25],
      
      // Reply swipe triggered
      replySwipe: 30,
      
      // Message copied
      copied: [30, 20, 30],
      
      // Message deleted
      deleted: [40, 40, 40],
      
      // Voice message recording
      recordingStart: 50,
      recordingStop: [30, 20, 30]
    };
  }
  
  // Contextual haptic for message status
  messageStatus(status) {
    const statusPatterns = {
      sending: this.patterns.tap,
      sent: this.patterns.messageSent,
      delivered: this.patterns.delivered,
      read: this.patterns.read,
      failed: this.patterns.error
    };
    
    const pattern = statusPatterns[status];
    if (pattern) {
      this.vibrate(pattern);
    }
  }
}
```

### Gesture Feedback

```javascript
class GestureHaptics {
  constructor(hapticManager) {
    this.haptics = hapticManager;
    this.swipeThreshold = 100;
    this.hasTriggeredSwipe = false;
  }
  
  // Called during swipe gesture
  onSwipeProgress(distance) {
    // Trigger feedback when threshold is crossed
    if (Math.abs(distance) >= this.swipeThreshold && !this.hasTriggeredSwipe) {
      this.haptics.trigger('replySwipe');
      this.hasTriggeredSwipe = true;
    }
  }
  
  // Called when swipe ends
  onSwipeEnd() {
    this.hasTriggeredSwipe = false;
  }
  
  // Long press feedback with progressive intensity
  onLongPressProgress(progress) {
    // Progress is 0-1, trigger at completion
    if (progress >= 1) {
      this.haptics.trigger('longPress');
    }
  }
}
```

### Pull-to-Refresh Haptics

```javascript
class PullToRefreshHaptics {
  constructor(hapticManager, threshold = 80) {
    this.haptics = hapticManager;
    this.threshold = threshold;
    this.hasTriggeredThreshold = false;
    this.hasTriggeredRelease = false;
  }
  
  onPull(distance) {
    // Light feedback when threshold is reached
    if (distance >= this.threshold && !this.hasTriggeredThreshold) {
      this.haptics.vibrate(20);
      this.hasTriggeredThreshold = true;
    }
    
    // Reset if pulled back above threshold
    if (distance < this.threshold * 0.8) {
      this.hasTriggeredThreshold = false;
    }
  }
  
  onRelease(distance) {
    if (distance >= this.threshold && !this.hasTriggeredRelease) {
      this.haptics.trigger('refresh');
      this.hasTriggeredRelease = true;
    }
  }
  
  onRefreshComplete() {
    this.haptics.trigger('success');
    this.hasTriggeredRelease = false;
    this.hasTriggeredThreshold = false;
  }
}
```

---

## iOS Alternatives

Since iOS doesn't support the Vibration API, consider these alternatives:

### Audio Feedback

```javascript
class AudioHaptics {
  constructor() {
    this.sounds = {};
    this.enabled = true;
    this.volume = 0.3;
  }
  
  async preload() {
    // Preload short audio clips
    const soundFiles = {
      tap: '/sounds/tap.mp3',
      success: '/sounds/success.mp3',
      error: '/sounds/error.mp3'
    };
    
    for (const [name, url] of Object.entries(soundFiles)) {
      try {
        const audio = new Audio(url);
        audio.volume = this.volume;
        audio.preload = 'auto';
        this.sounds[name] = audio;
      } catch (error) {
        console.warn(`Failed to load sound: ${name}`);
      }
    }
  }
  
  play(soundName) {
    if (!this.enabled) return;
    
    const sound = this.sounds[soundName];
    if (sound) {
      // Reset and play
      sound.currentTime = 0;
      sound.play().catch(() => {});
    }
  }
}
```

### Combined Feedback System

```javascript
class FeedbackManager {
  constructor() {
    this.vibration = new HapticManager();
    this.audio = new AudioHaptics();
    
    // Determine best feedback method
    this.useVibration = this.vibration.isSupported();
    this.useAudio = !this.useVibration; // Fallback for iOS
    
    this.audio.preload();
  }
  
  trigger(feedbackType) {
    if (this.useVibration) {
      this.vibration.trigger(feedbackType);
    } else if (this.useAudio) {
      this.audio.play(feedbackType);
    }
  }
  
  // Allow users to choose
  setMode(mode) {
    switch (mode) {
      case 'vibration':
        this.useVibration = this.vibration.isSupported();
        this.useAudio = false;
        break;
      case 'audio':
        this.useVibration = false;
        this.useAudio = true;
        break;
      case 'both':
        this.useVibration = this.vibration.isSupported();
        this.useAudio = true;
        break;
      case 'none':
        this.useVibration = false;
        this.useAudio = false;
        break;
    }
  }
}
```

---

## User Preferences

Always respect user preferences for haptic feedback:

### Settings UI Integration

```javascript
class HapticSettings {
  constructor(hapticManager) {
    this.haptics = hapticManager;
  }
  
  createSettingsUI() {
    const container = document.createElement('div');
    container.className = 'haptic-settings';
    
    container.innerHTML = `
      <div class="setting-row">
        <label for="haptic-toggle">Haptic Feedback</label>
        <input type="checkbox" id="haptic-toggle" 
               ${this.haptics.enabled ? 'checked' : ''}>
      </div>
      
      <div class="setting-row">
        <label>Test Patterns</label>
        <div class="test-buttons">
          <button data-pattern="tap">Tap</button>
          <button data-pattern="success">Success</button>
          <button data-pattern="error">Error</button>
        </div>
      </div>
    `;
    
    // Toggle handler
    container.querySelector('#haptic-toggle')
      .addEventListener('change', (e) => {
        this.haptics.setEnabled(e.target.checked);
      });
    
    // Test button handlers
    container.querySelectorAll('[data-pattern]').forEach(btn => {
      btn.addEventListener('click', () => {
        const wasEnabled = this.haptics.enabled;
        this.haptics.enabled = true; // Temporarily enable for test
        this.haptics.trigger(btn.dataset.pattern);
        this.haptics.enabled = wasEnabled;
      });
    });
    
    return container;
  }
}
```

### CSS for Settings

```css
.haptic-settings {
  padding: 16px;
}

.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #e0e0e0;
}

.setting-row label {
  font-size: 16px;
  font-weight: 500;
}

.test-buttons {
  display: flex;
  gap: 8px;
}

.test-buttons button {
  padding: 8px 16px;
  border: 1px solid #007AFF;
  border-radius: 8px;
  background: white;
  color: #007AFF;
  font-size: 14px;
}

.test-buttons button:active {
  background: #007AFF;
  color: white;
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Check for API support before using | Assume vibration is available |
| Keep patterns short (< 500ms total) | Create long, annoying vibrations |
| Provide user controls to disable | Force haptic feedback on users |
| Use distinct patterns for different actions | Use the same pattern for everything |
| Test on actual devices | Only test in browser simulators |
| Provide fallbacks for iOS | Ignore platforms without support |

### Accessibility Considerations

| Consideration | Implementation |
|---------------|----------------|
| **User control** | Always let users disable haptics |
| **Not essential** | Never rely solely on haptic feedback |
| **Visual confirmation** | Pair haptics with visual feedback |
| **Vestibular sensitivity** | Strong vibrations can trigger issues |
| **Battery life** | Excessive vibration drains battery |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Vibrating on every keystroke | Reserve for meaningful events |
| Long vibration patterns | Keep total duration under 500ms |
| Not testing on real devices | Simulators can't replicate haptics |
| No fallback for iOS | Provide audio or visual alternative |
| Same feedback for all actions | Use distinct patterns |
| Ignoring user preferences | Store and respect haptic settings |

---

## Hands-on Exercise

### Your Task

Create a complete haptic feedback system for a chat interface:

1. **Pattern library** with distinct patterns for:
   - Message sent/received
   - Success/error states  
   - Long press activation
   - Selection changes

2. **Feature detection** with graceful fallback

3. **User preference** storage and settings toggle

4. **Test interface** to preview patterns

### Requirements

1. Works on Android devices with vibration
2. Degrades gracefully on iOS (no errors)
3. Stores user preference in localStorage
4. Provides test buttons for each pattern

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `'vibrate' in navigator` for feature detection
- Store preference as string in localStorage
- Create pattern arrays for complex feedback
- Wrap vibrate calls in try-catch for safety

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
// Complete Haptic Feedback System
class ChatHapticSystem {
  constructor() {
    this.isSupported = 'vibrate' in navigator;
    this.enabled = this.loadPreference();
    
    this.patterns = {
      // Light feedback
      tap: 10,
      selection: 15,
      
      // Medium feedback  
      click: 25,
      messageReceived: 40,
      longPress: 50,
      
      // Complex patterns
      messageSent: [30, 20, 30],
      success: [50, 30, 50],
      warning: [75, 50, 75],
      error: [100, 50, 100, 50, 150],
      
      // Chat-specific
      reaction: [25, 15, 25],
      copied: [30, 20, 30],
      deleted: [40, 40, 40]
    };
  }
  
  loadPreference() {
    if (!this.isSupported) return false;
    const stored = localStorage.getItem('haptic-feedback');
    return stored === null ? true : stored === 'true';
  }
  
  savePreference() {
    localStorage.setItem('haptic-feedback', String(this.enabled));
  }
  
  setEnabled(enabled) {
    this.enabled = enabled && this.isSupported;
    this.savePreference();
    return this.enabled;
  }
  
  toggle() {
    return this.setEnabled(!this.enabled);
  }
  
  trigger(patternName) {
    if (!this.enabled || !this.isSupported) return false;
    
    const pattern = this.patterns[patternName];
    if (!pattern) {
      console.warn(`Unknown pattern: ${patternName}`);
      return false;
    }
    
    return this.vibrate(pattern);
  }
  
  vibrate(pattern) {
    if (!this.enabled || !this.isSupported) return false;
    
    try {
      navigator.vibrate(pattern);
      return true;
    } catch (e) {
      console.warn('Vibration error:', e);
      return false;
    }
  }
  
  stop() {
    if (this.isSupported) {
      navigator.vibrate(0);
    }
  }
  
  // Create settings UI
  createSettingsUI(container) {
    const html = `
      <div class="haptic-settings">
        <div class="setting-header">
          <span>Haptic Feedback</span>
          <span class="support-badge ${this.isSupported ? 'supported' : 'unsupported'}">
            ${this.isSupported ? 'Supported' : 'Not Supported'}
          </span>
        </div>
        
        <label class="toggle-setting">
          <span>Enable haptic feedback</span>
          <input type="checkbox" id="haptic-toggle" 
                 ${this.enabled ? 'checked' : ''}
                 ${!this.isSupported ? 'disabled' : ''}>
        </label>
        
        <div class="pattern-tests">
          <p>Test patterns:</p>
          <div class="pattern-grid">
            ${Object.keys(this.patterns).map(name => `
              <button class="pattern-btn" data-pattern="${name}">
                ${name}
              </button>
            `).join('')}
          </div>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
    
    // Event listeners
    container.querySelector('#haptic-toggle')
      ?.addEventListener('change', (e) => {
        this.setEnabled(e.target.checked);
      });
    
    container.querySelectorAll('.pattern-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        // Force enable for test
        const wasEnabled = this.enabled;
        this.enabled = this.isSupported;
        this.trigger(btn.dataset.pattern);
        this.enabled = wasEnabled;
      });
    });
  }
}

// Styles
const styles = `
  .haptic-settings {
    padding: 16px;
    font-family: -apple-system, sans-serif;
  }
  
  .setting-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    font-size: 18px;
    font-weight: 600;
  }
  
  .support-badge {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
  }
  
  .support-badge.supported {
    background: #e8f5e9;
    color: #2e7d32;
  }
  
  .support-badge.unsupported {
    background: #ffebee;
    color: #c62828;
  }
  
  .toggle-setting {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid #eee;
  }
  
  .pattern-tests {
    margin-top: 16px;
  }
  
  .pattern-tests p {
    color: #666;
    margin-bottom: 12px;
  }
  
  .pattern-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }
  
  .pattern-btn {
    padding: 10px;
    border: 1px solid #007AFF;
    border-radius: 8px;
    background: white;
    color: #007AFF;
    font-size: 12px;
    text-transform: capitalize;
    cursor: pointer;
  }
  
  .pattern-btn:active {
    background: #007AFF;
    color: white;
  }
`;

// Initialize
const haptics = new ChatHapticSystem();

// Add styles
const styleSheet = document.createElement('style');
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

// Create UI in a container
const settingsContainer = document.querySelector('#settings');
if (settingsContainer) {
  haptics.createSettingsUI(settingsContainer);
}

// Example usage
document.querySelector('.send-btn')?.addEventListener('click', () => {
  haptics.trigger('messageSent');
});
```

</details>

---

## Summary

✅ The Vibration API provides simple device vibration control with `navigator.vibrate()`  
✅ Pattern arrays alternate between vibration and pause durations  
✅ iOS does not support the Vibration API—provide fallbacks  
✅ Keep haptic patterns brief and distinct for different actions  
✅ Always check for API support and respect user preferences  
✅ Test on real devices—simulators cannot replicate haptic feedback  

**Next:** [Native Share Integration](./05-native-share-integration.md)

---

<!-- 
Sources Consulted:
- MDN Vibration API: https://developer.mozilla.org/en-US/docs/Web/API/Vibration_API
- Can I Use Vibration API: https://caniuse.com/vibration
- Apple Human Interface Guidelines: Haptics
-->
