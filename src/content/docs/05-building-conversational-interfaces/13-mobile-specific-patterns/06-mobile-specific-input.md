---
title: "Mobile-Specific Input"
---

# Mobile-Specific Input

## Introduction

Mobile devices offer input methods that go far beyond the keyboard—voice dictation, camera capture, quick reply suggestions, and predictive text. These inputs feel natural to mobile users and can dramatically improve the chat experience, especially for AI-powered conversational interfaces where voice and image inputs are increasingly common.

This lesson explores mobile-specific input methods that make chat interfaces feel native and effortless.

### What We'll Cover

- Voice input and speech recognition
- Camera and image input integration
- Quick replies and suggested responses
- Predictive text and autocomplete
- Input mode optimization

### Prerequisites

- Understanding of HTML form inputs
- JavaScript event handling
- Basic media API concepts

---

## Voice Input

Voice input transforms spoken words into text, enabling hands-free messaging.

### Web Speech API

The Web Speech API provides speech recognition capabilities:

```javascript
class VoiceInput {
  constructor(options = {}) {
    this.onResult = options.onResult || (() => {});
    this.onError = options.onError || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
    
    this.isSupported = 'webkitSpeechRecognition' in window || 
                       'SpeechRecognition' in window;
    
    if (this.isSupported) {
      this.initRecognition();
    }
  }
  
  initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || 
                              window.webkitSpeechRecognition;
    
    this.recognition = new SpeechRecognition();
    
    // Configuration
    this.recognition.continuous = false;  // Stop after user pauses
    this.recognition.interimResults = true;  // Get partial results
    this.recognition.lang = navigator.language || 'en-US';
    this.recognition.maxAlternatives = 1;
    
    // Event handlers
    this.recognition.onstart = () => {
      this.isListening = true;
      this.onStateChange({ listening: true });
    };
    
    this.recognition.onend = () => {
      this.isListening = false;
      this.onStateChange({ listening: false });
    };
    
    this.recognition.onresult = (event) => {
      const result = event.results[event.results.length - 1];
      const transcript = result[0].transcript;
      const isFinal = result.isFinal;
      
      this.onResult({
        text: transcript,
        confidence: result[0].confidence,
        isFinal
      });
    };
    
    this.recognition.onerror = (event) => {
      this.onError({
        error: event.error,
        message: this.getErrorMessage(event.error)
      });
    };
  }
  
  getErrorMessage(error) {
    const messages = {
      'no-speech': 'No speech detected',
      'audio-capture': 'Microphone not available',
      'not-allowed': 'Microphone access denied',
      'network': 'Network error during recognition',
      'aborted': 'Recognition was aborted'
    };
    return messages[error] || 'Recognition error';
  }
  
  start() {
    if (!this.isSupported) {
      this.onError({ error: 'not-supported', message: 'Speech recognition not supported' });
      return false;
    }
    
    if (this.isListening) return false;
    
    try {
      this.recognition.start();
      return true;
    } catch (error) {
      this.onError({ error: 'start-failed', message: error.message });
      return false;
    }
  }
  
  stop() {
    if (this.isListening) {
      this.recognition.stop();
    }
  }
  
  abort() {
    if (this.isListening) {
      this.recognition.abort();
    }
  }
}
```

### Voice Input UI

```javascript
class VoiceInputButton {
  constructor(textInput, options = {}) {
    this.textInput = textInput;
    this.container = options.container || textInput.parentElement;
    
    this.voice = new VoiceInput({
      onResult: (result) => this.handleResult(result),
      onError: (error) => this.handleError(error),
      onStateChange: (state) => this.updateUI(state)
    });
    
    if (this.voice.isSupported) {
      this.createElement();
    }
  }
  
  createElement() {
    this.button = document.createElement('button');
    this.button.className = 'voice-input-btn';
    this.button.type = 'button';
    this.button.innerHTML = `
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
        <path fill="currentColor" d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
      </svg>
    `;
    this.button.title = 'Voice input';
    
    this.button.addEventListener('click', () => this.toggleListening());
    
    this.container.appendChild(this.button);
  }
  
  toggleListening() {
    if (this.voice.isListening) {
      this.voice.stop();
    } else {
      // Request permission on first use
      this.voice.start();
    }
  }
  
  handleResult(result) {
    if (result.isFinal) {
      // Append to existing text
      const currentText = this.textInput.value;
      const separator = currentText && !currentText.endsWith(' ') ? ' ' : '';
      this.textInput.value = currentText + separator + result.text;
      this.textInput.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      // Show interim results as placeholder or overlay
      this.showInterim(result.text);
    }
  }
  
  handleError(error) {
    console.error('Voice input error:', error);
    this.showToast(error.message);
  }
  
  updateUI(state) {
    this.button.classList.toggle('listening', state.listening);
    if (state.listening) {
      this.button.innerHTML = `
        <div class="voice-pulse"></div>
      `;
    } else {
      this.button.innerHTML = `
        <svg viewBox="0 0 24 24" width="24" height="24">
          <path fill="currentColor" d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
          <path fill="currentColor" d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
        </svg>
      `;
    }
  }
  
  showInterim(text) {
    // Could show floating preview
    console.log('Interim:', text);
  }
  
  showToast(message) {
    // Toast notification implementation
  }
}
```

### Voice Input Styles

```css
.voice-input-btn {
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: #f0f0f0;
  color: #666;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.voice-input-btn:hover {
  background: #e0e0e0;
}

.voice-input-btn.listening {
  background: #ff4444;
  color: white;
}

.voice-pulse {
  width: 20px;
  height: 20px;
  background: white;
  border-radius: 50%;
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(0.8); opacity: 0.8; }
  50% { transform: scale(1.2); opacity: 1; }
}
```

### Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome Android | ✅ Full | Best support |
| Chrome Desktop | ✅ Full | Good support |
| Safari | ⚠️ Partial | iOS 14.5+, macOS 14.1+ |
| Firefox | ❌ None | Not implemented |
| Edge | ✅ Full | Chromium-based |

---

## Camera and Image Input

Camera access enables photo sharing and image-based AI interactions.

### Basic Camera Input

```html
<input type="file" 
       accept="image/*" 
       capture="environment"
       id="camera-input">
```

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `accept` | `image/*` | Only accept images |
| `capture` | `user` | Front camera (selfie) |
| `capture` | `environment` | Rear camera |
| (omit capture) | — | Let user choose camera or gallery |

### Enhanced Camera Input

```javascript
class CameraInput {
  constructor(options = {}) {
    this.onCapture = options.onCapture || (() => {});
    this.onError = options.onError || (() => {});
    this.maxSize = options.maxSize || 1024 * 1024; // 1MB
    this.quality = options.quality || 0.8;
    
    this.createElement();
  }
  
  createElement() {
    this.input = document.createElement('input');
    this.input.type = 'file';
    this.input.accept = 'image/*';
    this.input.capture = 'environment';
    this.input.style.display = 'none';
    
    this.input.addEventListener('change', (e) => this.handleCapture(e));
    
    document.body.appendChild(this.input);
  }
  
  open(mode = 'environment') {
    this.input.capture = mode;
    this.input.click();
  }
  
  openGallery() {
    this.input.removeAttribute('capture');
    this.input.click();
  }
  
  async handleCapture(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    
    try {
      const processed = await this.processImage(file);
      this.onCapture(processed);
    } catch (error) {
      this.onError(error);
    }
    
    // Reset input for same-file selection
    this.input.value = '';
  }
  
  async processImage(file) {
    // Load image
    const img = await this.loadImage(file);
    
    // Resize if too large
    const resized = await this.resizeImage(img, 1920, 1920);
    
    // Convert to blob
    const blob = await this.canvasToBlob(resized, this.quality);
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(blob);
    
    return {
      blob,
      previewUrl,
      width: resized.width,
      height: resized.height,
      size: blob.size,
      type: blob.type,
      originalName: file.name
    };
  }
  
  loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
  
  resizeImage(img, maxWidth, maxHeight) {
    const canvas = document.createElement('canvas');
    let { width, height } = img;
    
    // Calculate new dimensions
    if (width > maxWidth) {
      height = (height * maxWidth) / width;
      width = maxWidth;
    }
    if (height > maxHeight) {
      width = (width * maxHeight) / height;
      height = maxHeight;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    
    return canvas;
  }
  
  canvasToBlob(canvas, quality) {
    return new Promise((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', quality);
    });
  }
  
  destroy() {
    this.input.remove();
  }
}
```

### Camera Button UI

```javascript
class CameraButton {
  constructor(options = {}) {
    this.onImage = options.onImage || (() => {});
    this.container = options.container;
    
    this.camera = new CameraInput({
      onCapture: (image) => this.handleImage(image),
      onError: (error) => this.handleError(error)
    });
    
    this.createElement();
  }
  
  createElement() {
    this.button = document.createElement('button');
    this.button.className = 'camera-btn';
    this.button.type = 'button';
    this.button.innerHTML = `
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4z"/>
        <path fill="currentColor" d="M9 2L7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>
      </svg>
    `;
    
    // Long press for options
    this.setupPressHandling();
    
    this.container?.appendChild(this.button);
  }
  
  setupPressHandling() {
    let pressTimer;
    
    this.button.addEventListener('pointerdown', () => {
      pressTimer = setTimeout(() => {
        this.showOptions();
      }, 500);
    });
    
    this.button.addEventListener('pointerup', () => {
      if (pressTimer) {
        clearTimeout(pressTimer);
        // Short press - open camera
        this.camera.open();
      }
    });
    
    this.button.addEventListener('pointercancel', () => {
      clearTimeout(pressTimer);
    });
  }
  
  showOptions() {
    const menu = document.createElement('div');
    menu.className = 'camera-options';
    menu.innerHTML = `
      <button data-action="camera">📷 Take Photo</button>
      <button data-action="selfie">🤳 Selfie</button>
      <button data-action="gallery">🖼️ Gallery</button>
    `;
    
    menu.addEventListener('click', (e) => {
      const action = e.target.dataset.action;
      switch (action) {
        case 'camera':
          this.camera.open('environment');
          break;
        case 'selfie':
          this.camera.open('user');
          break;
        case 'gallery':
          this.camera.openGallery();
          break;
      }
      menu.remove();
    });
    
    // Position near button
    this.button.parentElement.appendChild(menu);
  }
  
  handleImage(image) {
    this.onImage(image);
  }
  
  handleError(error) {
    console.error('Camera error:', error);
  }
}
```

### Camera Options Menu Styles

```css
.camera-btn {
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: #f0f0f0;
  color: #666;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.camera-options {
  position: absolute;
  bottom: 100%;
  left: 0;
  margin-bottom: 8px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  overflow: hidden;
}

.camera-options button {
  display: block;
  width: 100%;
  padding: 12px 16px;
  border: none;
  background: none;
  text-align: left;
  font-size: 14px;
  cursor: pointer;
}

.camera-options button:hover {
  background: #f5f5f5;
}

.camera-options button:not(:last-child) {
  border-bottom: 1px solid #eee;
}
```

---

## Quick Replies

Quick replies provide one-tap responses for common messages:

### Quick Reply System

```javascript
class QuickReplies {
  constructor(options = {}) {
    this.container = options.container;
    this.onSelect = options.onSelect || (() => {});
    
    this.createElement();
  }
  
  createElement() {
    this.element = document.createElement('div');
    this.element.className = 'quick-replies';
    this.container?.appendChild(this.element);
  }
  
  setReplies(replies) {
    if (!replies?.length) {
      this.hide();
      return;
    }
    
    this.element.innerHTML = replies.map((reply, i) => `
      <button class="quick-reply" data-index="${i}">
        ${reply.icon ? `<span class="reply-icon">${reply.icon}</span>` : ''}
        <span class="reply-text">${reply.text}</span>
      </button>
    `).join('');
    
    // Store replies for reference
    this.replies = replies;
    
    // Attach handlers
    this.element.querySelectorAll('.quick-reply').forEach(btn => {
      btn.addEventListener('click', () => {
        const index = parseInt(btn.dataset.index);
        this.selectReply(index);
      });
    });
    
    this.show();
  }
  
  selectReply(index) {
    const reply = this.replies[index];
    if (reply) {
      this.onSelect(reply);
      this.hide();
    }
  }
  
  show() {
    this.element.classList.add('visible');
  }
  
  hide() {
    this.element.classList.remove('visible');
  }
}
```

### Context-Aware Quick Replies

```javascript
class ContextualQuickReplies extends QuickReplies {
  constructor(options) {
    super(options);
    this.contextRules = options.rules || this.getDefaultRules();
  }
  
  getDefaultRules() {
    return [
      {
        trigger: /how are you|what's up/i,
        replies: [
          { text: "I'm doing great, thanks!", icon: '😊' },
          { text: "Pretty good! How about you?", icon: '👍' },
          { text: "Could be better", icon: '😕' }
        ]
      },
      {
        trigger: /thanks|thank you/i,
        replies: [
          { text: "You're welcome!", icon: '🙂' },
          { text: "Happy to help!", icon: '😄' },
          { text: "Anytime!", icon: '👍' }
        ]
      },
      {
        trigger: /yes or no|would you like|do you want/i,
        replies: [
          { text: "Yes", icon: '✅' },
          { text: "No", icon: '❌' },
          { text: "Maybe later", icon: '🤔' }
        ]
      },
      {
        trigger: /what time|when/i,
        replies: [
          { text: "Now works for me", icon: '⏰' },
          { text: "In an hour", icon: '🕐' },
          { text: "Tomorrow", icon: '📅' },
          { text: "Let me check my schedule", icon: '📋' }
        ]
      }
    ];
  }
  
  suggestForMessage(message) {
    for (const rule of this.contextRules) {
      if (rule.trigger.test(message)) {
        this.setReplies(rule.replies);
        return true;
      }
    }
    
    // Default replies
    this.setReplies([
      { text: "Got it!", icon: '👍' },
      { text: "Thanks!", icon: '🙏' },
      { text: "Tell me more", icon: '💬' }
    ]);
    return true;
  }
  
  // For AI assistants, suggest based on response type
  suggestForAIResponse(response) {
    const suggestions = [];
    
    // Check for questions in response
    if (response.includes('?')) {
      suggestions.push(
        { text: "Yes", icon: '✅' },
        { text: "No", icon: '❌' }
      );
    }
    
    // Check for explanations
    if (response.length > 200) {
      suggestions.push(
        { text: "Can you simplify that?", icon: '🔄' },
        { text: "Tell me more", icon: '➕' },
        { text: "Give me an example", icon: '💡' }
      );
    }
    
    // Check for code
    if (response.includes('```')) {
      suggestions.push(
        { text: "Explain this code", icon: '🔍' },
        { text: "Run this code", icon: '▶️' },
        { text: "Modify it", icon: '✏️' }
      );
    }
    
    if (suggestions.length) {
      this.setReplies(suggestions);
    }
  }
}
```

### Quick Reply Styles

```css
.quick-replies {
  display: flex;
  gap: 8px;
  padding: 8px 16px;
  overflow-x: auto;
  scrollbar-width: none;
  -webkit-overflow-scrolling: touch;
  opacity: 0;
  transform: translateY(10px);
  transition: all 0.2s;
}

.quick-replies::-webkit-scrollbar {
  display: none;
}

.quick-replies.visible {
  opacity: 1;
  transform: translateY(0);
}

.quick-reply {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: 1px solid #007AFF;
  border-radius: 18px;
  background: white;
  color: #007AFF;
  font-size: 14px;
  white-space: nowrap;
  cursor: pointer;
  transition: all 0.15s;
}

.quick-reply:hover {
  background: #007AFF;
  color: white;
}

.quick-reply:active {
  transform: scale(0.95);
}

.reply-icon {
  font-size: 16px;
}
```

---

## Input Mode Optimization

HTML5 input modes optimize the virtual keyboard for expected input types:

### Input Modes

| Mode | Keyboard | Use Case |
|------|----------|----------|
| `text` | Standard text | Default for chat |
| `numeric` | Number pad | Phone numbers, codes |
| `decimal` | Numbers with decimal | Prices, measurements |
| `tel` | Phone keypad | Phone numbers |
| `email` | Text + @ + .com | Email addresses |
| `url` | Text + / + .com | Web URLs |
| `search` | Text + search button | Search queries |
| `none` | No keyboard | Custom input |

### Smart Input Detection

```javascript
class SmartInput {
  constructor(input) {
    this.input = input;
    this.defaultMode = 'text';
    
    this.patterns = [
      { pattern: /email|e-mail/i, mode: 'email' },
      { pattern: /phone|tel|mobile|call/i, mode: 'tel' },
      { pattern: /url|website|link|http/i, mode: 'url' },
      { pattern: /search|find|look/i, mode: 'search' },
      { pattern: /number|amount|quantity|price/i, mode: 'decimal' },
      { pattern: /code|pin|otp|verification/i, mode: 'numeric' }
    ];
  }
  
  setModeForContext(context) {
    for (const { pattern, mode } of this.patterns) {
      if (pattern.test(context)) {
        this.setInputMode(mode);
        return mode;
      }
    }
    
    this.setInputMode(this.defaultMode);
    return this.defaultMode;
  }
  
  setInputMode(mode) {
    this.input.inputMode = mode;
    
    // Also set type for better semantics
    switch (mode) {
      case 'email':
        this.input.type = 'email';
        break;
      case 'tel':
        this.input.type = 'tel';
        break;
      case 'url':
        this.input.type = 'url';
        break;
      default:
        this.input.type = 'text';
    }
  }
  
  // Temporarily change mode for specific question
  setTemporaryMode(mode, duration = 60000) {
    const originalMode = this.input.inputMode;
    this.setInputMode(mode);
    
    setTimeout(() => {
      this.setInputMode(originalMode);
    }, duration);
  }
}
```

### Contextual Keyboard in Chat

```javascript
class ChatInput {
  constructor(element) {
    this.element = element;
    this.smartInput = new SmartInput(element);
  }
  
  // Called when AI asks a question
  prepareForQuestion(question) {
    // Detect expected input type from question
    if (question.match(/email|e-mail address/i)) {
      this.smartInput.setInputMode('email');
      this.element.placeholder = 'Enter your email...';
    } else if (question.match(/phone|number to call|mobile/i)) {
      this.smartInput.setInputMode('tel');
      this.element.placeholder = 'Enter phone number...';
    } else if (question.match(/how many|how much|quantity|amount/i)) {
      this.smartInput.setInputMode('decimal');
      this.element.placeholder = 'Enter a number...';
    } else if (question.match(/website|url|link/i)) {
      this.smartInput.setInputMode('url');
      this.element.placeholder = 'Enter URL...';
    } else {
      this.smartInput.setInputMode('text');
      this.element.placeholder = 'Type a message...';
    }
  }
  
  reset() {
    this.smartInput.setInputMode('text');
    this.element.placeholder = 'Type a message...';
  }
}
```

---

## Autocomplete and Suggestions

### Text Autocomplete

```javascript
class AutocompleteInput {
  constructor(input, options = {}) {
    this.input = input;
    this.suggestions = options.suggestions || [];
    this.minChars = options.minChars || 2;
    this.maxSuggestions = options.maxSuggestions || 5;
    
    this.createElement();
    this.attachEvents();
  }
  
  createElement() {
    this.dropdown = document.createElement('div');
    this.dropdown.className = 'autocomplete-dropdown';
    this.input.parentElement.appendChild(this.dropdown);
  }
  
  attachEvents() {
    this.input.addEventListener('input', () => this.handleInput());
    this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
    this.input.addEventListener('blur', () => {
      setTimeout(() => this.hide(), 150);
    });
  }
  
  handleInput() {
    const value = this.input.value.trim();
    
    if (value.length < this.minChars) {
      this.hide();
      return;
    }
    
    const matches = this.findMatches(value);
    this.showSuggestions(matches);
  }
  
  findMatches(query) {
    const lowerQuery = query.toLowerCase();
    return this.suggestions
      .filter(s => s.toLowerCase().includes(lowerQuery))
      .slice(0, this.maxSuggestions);
  }
  
  showSuggestions(matches) {
    if (!matches.length) {
      this.hide();
      return;
    }
    
    this.dropdown.innerHTML = matches.map((match, i) => `
      <div class="autocomplete-item" data-index="${i}">
        ${this.highlightMatch(match, this.input.value)}
      </div>
    `).join('');
    
    this.matches = matches;
    this.selectedIndex = -1;
    
    this.dropdown.querySelectorAll('.autocomplete-item').forEach(item => {
      item.addEventListener('click', () => {
        this.selectItem(parseInt(item.dataset.index));
      });
    });
    
    this.show();
  }
  
  highlightMatch(text, query) {
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
  }
  
  handleKeydown(e) {
    if (!this.matches?.length) return;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        this.navigate(1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        this.navigate(-1);
        break;
      case 'Enter':
        if (this.selectedIndex >= 0) {
          e.preventDefault();
          this.selectItem(this.selectedIndex);
        }
        break;
      case 'Escape':
        this.hide();
        break;
    }
  }
  
  navigate(direction) {
    const items = this.dropdown.querySelectorAll('.autocomplete-item');
    
    if (this.selectedIndex >= 0) {
      items[this.selectedIndex].classList.remove('selected');
    }
    
    this.selectedIndex += direction;
    
    if (this.selectedIndex < 0) this.selectedIndex = items.length - 1;
    if (this.selectedIndex >= items.length) this.selectedIndex = 0;
    
    items[this.selectedIndex].classList.add('selected');
  }
  
  selectItem(index) {
    this.input.value = this.matches[index];
    this.input.dispatchEvent(new Event('input', { bubbles: true }));
    this.hide();
  }
  
  show() {
    this.dropdown.classList.add('visible');
  }
  
  hide() {
    this.dropdown.classList.remove('visible');
    this.selectedIndex = -1;
  }
  
  setSuggestions(suggestions) {
    this.suggestions = suggestions;
  }
}
```

### Autocomplete Styles

```css
.autocomplete-dropdown {
  position: absolute;
  bottom: 100%;
  left: 0;
  right: 0;
  margin-bottom: 4px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  max-height: 200px;
  overflow-y: auto;
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s;
}

.autocomplete-dropdown.visible {
  opacity: 1;
  visibility: visible;
}

.autocomplete-item {
  padding: 12px 16px;
  cursor: pointer;
}

.autocomplete-item:hover,
.autocomplete-item.selected {
  background: #f5f5f5;
}

.autocomplete-item strong {
  color: #007AFF;
}
```

---

## Complete Input System

```javascript
class MobileInputSystem {
  constructor(options = {}) {
    this.container = options.container;
    this.textInput = options.textInput;
    this.onSend = options.onSend || (() => {});
    
    this.setupComponents();
  }
  
  setupComponents() {
    // Voice input
    this.voiceButton = new VoiceInputButton(this.textInput, {
      container: this.container.querySelector('.input-actions')
    });
    
    // Camera input
    this.cameraButton = new CameraButton({
      container: this.container.querySelector('.input-actions'),
      onImage: (image) => this.handleImage(image)
    });
    
    // Quick replies
    this.quickReplies = new ContextualQuickReplies({
      container: this.container,
      onSelect: (reply) => this.sendMessage(reply.text)
    });
    
    // Smart input mode
    this.smartInput = new SmartInput(this.textInput);
    
    // Autocomplete (for common phrases)
    this.autocomplete = new AutocompleteInput(this.textInput, {
      suggestions: [
        "Hello!",
        "Thank you",
        "Could you help me with...",
        "I have a question about...",
        "Can you explain..."
      ]
    });
  }
  
  handleImage(image) {
    this.onSend({
      type: 'image',
      data: image
    });
  }
  
  sendMessage(text) {
    this.onSend({
      type: 'text',
      data: text
    });
  }
  
  showQuickRepliesFor(message) {
    this.quickReplies.suggestForMessage(message);
  }
  
  prepareForQuestion(question) {
    this.smartInput.setModeForContext(question);
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Check for API support before showing UI | Assume all devices support all APIs |
| Request permissions at point of use | Request all permissions on load |
| Show clear feedback during voice input | Leave users wondering if it's working |
| Provide fallback for unsupported features | Hide functionality without explanation |
| Use appropriate `inputMode` for context | Always use default text keyboard |
| Make quick replies contextually relevant | Show random suggestions |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Speech recognition on Firefox | Show alternative or disable feature |
| Camera not working in WebView | Check for permission and protocol (HTTPS) |
| Quick replies blocking input | Position above keyboard, not over input |
| Wrong keyboard for input type | Set `inputMode` based on expected input |
| Autocomplete covering content | Position dropdown above input |
| Voice permission denied | Show explanation and manual input fallback |

---

## Hands-on Exercise

### Your Task

Build a mobile-optimized input system with:

1. **Voice input button** — Toggle speech recognition
2. **Camera button** — Capture or select images
3. **Quick replies** — Context-aware suggestions
4. **Smart keyboard** — Optimized for expected input

### Requirements

1. Feature detection for voice and camera
2. Graceful degradation on unsupported browsers
3. Clear visual feedback for all states
4. At least 3 contextual quick reply sets

<details>
<summary>💡 Hints (click to expand)</summary>

- Check `webkitSpeechRecognition` for Chrome support
- Use `capture="environment"` for rear camera by default
- Show quick replies after receiving messages
- Change `inputMode` when AI asks specific questions

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mobile Input Demo</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: -apple-system, sans-serif;
      background: #f5f5f5;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
    }
    
    .message {
      background: white;
      padding: 10px 14px;
      border-radius: 12px;
      margin-bottom: 8px;
      max-width: 80%;
    }
    
    .message.ai {
      background: #007AFF;
      color: white;
      margin-left: auto;
    }
    
    .quick-replies {
      display: flex;
      gap: 8px;
      padding: 8px 16px;
      overflow-x: auto;
    }
    
    .quick-reply {
      padding: 8px 16px;
      border: 1px solid #007AFF;
      border-radius: 18px;
      background: white;
      color: #007AFF;
      white-space: nowrap;
      cursor: pointer;
    }
    
    .input-area {
      display: flex;
      gap: 8px;
      padding: 12px;
      background: white;
      border-top: 1px solid #eee;
    }
    
    .input-field {
      flex: 1;
      padding: 10px 16px;
      border: 1px solid #ddd;
      border-radius: 20px;
      font-size: 16px;
    }
    
    .action-btn {
      width: 44px;
      height: 44px;
      border: none;
      border-radius: 50%;
      background: #f0f0f0;
      cursor: pointer;
      font-size: 20px;
    }
    
    .action-btn.listening {
      background: #ff4444;
      color: white;
    }
    
    .action-btn:disabled {
      opacity: 0.5;
    }
  </style>
</head>
<body>
  <div class="messages" id="messages">
    <div class="message ai">Hi! How can I help you today?</div>
  </div>
  
  <div class="quick-replies" id="quickReplies">
    <button class="quick-reply" data-text="Tell me about the weather">🌤️ Weather</button>
    <button class="quick-reply" data-text="I have a question">❓ Question</button>
    <button class="quick-reply" data-text="Help me with code">💻 Code help</button>
  </div>
  
  <div class="input-area">
    <button class="action-btn" id="cameraBtn">📷</button>
    <input type="text" class="input-field" id="textInput" placeholder="Type a message...">
    <button class="action-btn" id="voiceBtn">🎤</button>
  </div>
  
  <input type="file" id="cameraInput" accept="image/*" capture="environment" style="display:none">

  <script>
    const messagesEl = document.getElementById('messages');
    const textInput = document.getElementById('textInput');
    const voiceBtn = document.getElementById('voiceBtn');
    const cameraBtn = document.getElementById('cameraBtn');
    const cameraInput = document.getElementById('cameraInput');
    const quickReplies = document.getElementById('quickReplies');
    
    // Voice input
    const hasVoice = 'webkitSpeechRecognition' in window;
    let recognition, isListening = false;
    
    if (hasVoice) {
      recognition = new webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      
      recognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        textInput.value = text;
        sendMessage(text);
      };
      
      recognition.onend = () => {
        isListening = false;
        voiceBtn.classList.remove('listening');
        voiceBtn.textContent = '🎤';
      };
    } else {
      voiceBtn.disabled = true;
      voiceBtn.title = 'Voice not supported';
    }
    
    voiceBtn.onclick = () => {
      if (!hasVoice) return;
      
      if (isListening) {
        recognition.stop();
      } else {
        recognition.start();
        isListening = true;
        voiceBtn.classList.add('listening');
        voiceBtn.textContent = '⏹️';
      }
    };
    
    // Camera input
    cameraBtn.onclick = () => cameraInput.click();
    cameraInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        addMessage(`📷 [Image: ${file.name}]`, 'user');
        cameraInput.value = '';
      }
    };
    
    // Quick replies
    quickReplies.onclick = (e) => {
      if (e.target.classList.contains('quick-reply')) {
        sendMessage(e.target.dataset.text);
      }
    };
    
    // Text input
    textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && textInput.value.trim()) {
        sendMessage(textInput.value);
      }
    });
    
    function addMessage(text, type = 'user') {
      const div = document.createElement('div');
      div.className = `message ${type}`;
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    
    function sendMessage(text) {
      addMessage(text, 'user');
      textInput.value = '';
      
      // Simulate AI response
      setTimeout(() => {
        addMessage('Thanks for your message!', 'ai');
      }, 500);
    }
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ Voice input uses the Web Speech API—check for `webkitSpeechRecognition` support  
✅ Camera input uses `<input type="file" accept="image/*" capture="environment">`  
✅ Quick replies provide one-tap contextual responses  
✅ Use `inputMode` to optimize the virtual keyboard for expected input types  
✅ Always provide fallbacks for unsupported features  
✅ Test on real mobile devices—emulators miss many input nuances  

**Previous:** [Native Share Integration](./05-native-share-integration.md)

---

<!-- 
Sources Consulted:
- MDN Web Speech API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
- MDN SpeechRecognition: https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition
- MDN input element: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
- MDN inputmode attribute: https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/inputmode
-->
