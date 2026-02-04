---
title: "Native Share Integration"
---

# Native Share Integration

## Introduction

Mobile users expect to share content using their device's native sharing interface—the familiar sheet that displays their installed apps and recent contacts. The Web Share API bridges web applications and native sharing, letting users share messages, links, and files from your chat interface directly to any app on their device.

This lesson covers implementing native share functionality, handling different content types, and building fallback experiences for unsupported browsers.

### What We'll Cover

- Understanding the Web Share API
- Sharing text, URLs, and files
- Checking share capability with `canShare()`
- Building fallback sharing mechanisms
- Share target registration (receiving shares)

### Prerequisites

- JavaScript Promises and async/await
- Understanding of HTTPS requirements
- Basic file handling concepts

---

## The Web Share API

The Web Share API invokes the device's native share dialog, providing a familiar and efficient sharing experience.

### Core Methods

| Method | Purpose |
|--------|---------|
| `navigator.share()` | Trigger share dialog with content |
| `navigator.canShare()` | Check if content can be shared |

### Basic Usage

```javascript
async function shareMessage(text) {
  if (!navigator.share) {
    console.log('Web Share API not supported');
    return false;
  }
  
  try {
    await navigator.share({
      title: 'Chat Message',
      text: text,
      url: window.location.href
    });
    console.log('Shared successfully');
    return true;
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Share cancelled by user');
    } else {
      console.error('Share failed:', error);
    }
    return false;
  }
}
```

### Share Data Properties

| Property | Type | Description |
|----------|------|-------------|
| `title` | string | Title of shared content (may be ignored) |
| `text` | string | Body text to share |
| `url` | string | URL to share |
| `files` | File[] | Array of files to share |

> **Note:** At least one of `text`, `url`, or `files` must be provided. The `title` is optional and often ignored by share targets.

### Requirements

| Requirement | Details |
|-------------|---------|
| **HTTPS** | Required (except localhost) |
| **User activation** | Must be triggered by user gesture |
| **Transient activation** | Call must happen during gesture |

> **Warning:** You cannot call `navigator.share()` automatically or on page load. It must be in response to a user action like a click or tap.

---

## Browser Support

| Browser | `share()` | `canShare()` | Files |
|---------|-----------|--------------|-------|
| Safari iOS | ✅ 12.2+ | ✅ 14+ | ✅ 14+ |
| Chrome Android | ✅ 61+ | ✅ 89+ | ✅ 75+ |
| Samsung Internet | ✅ | ✅ | ✅ |
| Firefox Android | ✅ 79+ | ✅ 79+ | ❌ |
| Safari macOS | ✅ 12.1+ | ✅ 14+ | ✅ |
| Chrome Desktop | ⚠️ Windows/ChromeOS | ⚠️ | ⚠️ |
| Edge | ⚠️ Windows 10+ | ⚠️ | ⚠️ |

---

## Feature Detection

Always check for support before offering share functionality:

```javascript
function canShare() {
  return 'share' in navigator;
}

function canShareFiles() {
  return 'canShare' in navigator && navigator.canShare({ files: [] });
}

// Comprehensive check
const shareSupport = {
  basic: 'share' in navigator,
  canShare: 'canShare' in navigator,
  files: false
};

// Test file sharing support
if (shareSupport.canShare) {
  try {
    const testFile = new File([''], 'test.txt', { type: 'text/plain' });
    shareSupport.files = navigator.canShare({ files: [testFile] });
  } catch {
    shareSupport.files = false;
  }
}
```

---

## Sharing Text and URLs

### Share a Chat Message

```javascript
async function shareMessageContent(message) {
  const shareData = {
    title: 'Chat Message',
    text: message.text
  };
  
  // Include link if message has one
  if (message.link) {
    shareData.url = message.link;
  }
  
  if (!navigator.share) {
    return fallbackShare(shareData);
  }
  
  try {
    await navigator.share(shareData);
    return { success: true };
  } catch (error) {
    if (error.name === 'AbortError') {
      return { success: false, cancelled: true };
    }
    return { success: false, error };
  }
}
```

### Share Conversation Link

```javascript
async function shareConversation(conversationId) {
  const shareUrl = `${window.location.origin}/chat/${conversationId}`;
  
  const shareData = {
    title: 'Check out this conversation',
    text: 'I wanted to share this chat with you',
    url: shareUrl
  };
  
  try {
    await navigator.share(shareData);
    trackEvent('conversation_shared', { conversationId });
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.error('Share failed:', error);
    }
  }
}
```

### Share with Rich Preview Data

```javascript
async function shareWithPreview(content) {
  // Structure data for best preview across platforms
  const shareData = {
    title: content.title || 'Shared from AI Chat',
    text: content.summary || content.text.substring(0, 200),
    url: content.url || window.location.href
  };
  
  // Some platforms use text + url combination
  // Others only show one. Test on target platforms.
  
  try {
    await navigator.share(shareData);
  } catch (error) {
    handleShareError(error);
  }
}
```

---

## Sharing Files

File sharing enables sharing images, documents, and other media from chat.

### Using canShare() for Files

Always verify file sharing is supported before attempting:

```javascript
function canShareFiles(files) {
  if (!navigator.canShare) {
    return false;
  }
  
  try {
    return navigator.canShare({ files });
  } catch {
    return false;
  }
}
```

### Share an Image

```javascript
async function shareImage(imageUrl, caption = '') {
  // First, fetch the image as a blob
  const response = await fetch(imageUrl);
  const blob = await response.blob();
  
  // Create a File object
  const fileName = imageUrl.split('/').pop() || 'image.png';
  const file = new File([blob], fileName, { type: blob.type });
  
  const shareData = {
    files: [file],
    title: 'Shared Image',
    text: caption
  };
  
  // Check if we can share this file
  if (!navigator.canShare || !navigator.canShare(shareData)) {
    // Fallback: open in new tab or download
    return fallbackImageShare(imageUrl);
  }
  
  try {
    await navigator.share(shareData);
    return { success: true };
  } catch (error) {
    if (error.name === 'AbortError') {
      return { success: false, cancelled: true };
    }
    return { success: false, error };
  }
}
```

### Share Multiple Files

```javascript
async function shareMultipleFiles(fileUrls) {
  const files = await Promise.all(
    fileUrls.map(async (url) => {
      const response = await fetch(url);
      const blob = await response.blob();
      const fileName = url.split('/').pop();
      return new File([blob], fileName, { type: blob.type });
    })
  );
  
  const shareData = { files };
  
  if (!navigator.canShare?.(shareData)) {
    throw new Error('Cannot share these files');
  }
  
  await navigator.share(shareData);
}
```

### Supported File Types

Common shareable MIME types:

| Category | MIME Types |
|----------|------------|
| Images | `image/png`, `image/jpeg`, `image/gif`, `image/webp` |
| Documents | `application/pdf`, `text/plain` |
| Audio | `audio/mp3`, `audio/wav`, `audio/ogg` |
| Video | `video/mp4`, `video/webm` |

> **Note:** Supported types vary by platform and installed apps. Always use `canShare()` to verify.

---

## Complete Share Implementation

### ShareManager Class

```javascript
class ShareManager {
  constructor() {
    this.support = this.detectSupport();
  }
  
  detectSupport() {
    const support = {
      basic: 'share' in navigator,
      canShare: 'canShare' in navigator,
      files: false,
      types: []
    };
    
    if (support.canShare) {
      // Test file types
      const testTypes = [
        { type: 'image/png', ext: 'png' },
        { type: 'image/jpeg', ext: 'jpg' },
        { type: 'application/pdf', ext: 'pdf' },
        { type: 'text/plain', ext: 'txt' }
      ];
      
      testTypes.forEach(({ type, ext }) => {
        try {
          const testFile = new File([''], `test.${ext}`, { type });
          if (navigator.canShare({ files: [testFile] })) {
            support.files = true;
            support.types.push(type);
          }
        } catch {}
      });
    }
    
    return support;
  }
  
  isSupported() {
    return this.support.basic;
  }
  
  canShareFiles() {
    return this.support.files;
  }
  
  canShareType(mimeType) {
    return this.support.types.includes(mimeType);
  }
  
  async share(data) {
    if (!this.isSupported()) {
      throw new Error('Web Share API not supported');
    }
    
    // Validate data
    if (!data.text && !data.url && !data.files?.length) {
      throw new Error('Share data must include text, url, or files');
    }
    
    // Verify file sharing if files included
    if (data.files?.length && !navigator.canShare?.(data)) {
      throw new Error('Cannot share provided files');
    }
    
    try {
      await navigator.share(data);
      return { success: true };
    } catch (error) {
      return this.handleError(error);
    }
  }
  
  handleError(error) {
    switch (error.name) {
      case 'AbortError':
        return { success: false, reason: 'cancelled' };
      case 'NotAllowedError':
        return { success: false, reason: 'not_allowed' };
      case 'TypeError':
        return { success: false, reason: 'invalid_data' };
      default:
        return { success: false, reason: 'unknown', error };
    }
  }
  
  // Convenience methods
  async shareText(text, title = '') {
    return this.share({ text, title });
  }
  
  async shareUrl(url, title = '', text = '') {
    return this.share({ url, title, text });
  }
  
  async shareFile(file, title = '') {
    return this.share({ files: [file], title });
  }
}

// Usage
const shareManager = new ShareManager();
```

### Chat-Specific Share Actions

```javascript
class ChatShareActions {
  constructor(shareManager) {
    this.share = shareManager;
  }
  
  async shareMessage(message) {
    const data = {
      text: message.content,
      title: `Message from ${message.sender}`
    };
    
    // If message has attachments, share those too
    if (message.attachments?.length && this.share.canShareFiles()) {
      data.files = await this.fetchAttachments(message.attachments);
    }
    
    return this.share.share(data);
  }
  
  async shareConversationSummary(messages) {
    const summary = messages
      .slice(-10)
      .map(m => `${m.sender}: ${m.content}`)
      .join('\n');
    
    return this.share.shareText(summary, 'Conversation Summary');
  }
  
  async shareAIResponse(response, question) {
    const text = `Q: ${question}\n\nA: ${response}`;
    return this.share.shareText(text, 'AI Chat Response');
  }
  
  async fetchAttachments(attachments) {
    return Promise.all(
      attachments.map(async (att) => {
        const response = await fetch(att.url);
        const blob = await response.blob();
        return new File([blob], att.filename, { type: att.mimeType });
      })
    );
  }
}
```

---

## Fallback Sharing

For browsers without Web Share API support, provide alternative sharing methods:

### Fallback Share Menu

```javascript
class FallbackShare {
  constructor() {
    this.services = [
      {
        name: 'Copy Link',
        icon: '📋',
        action: (data) => this.copyToClipboard(data)
      },
      {
        name: 'Email',
        icon: '✉️',
        action: (data) => this.shareViaEmail(data)
      },
      {
        name: 'Twitter',
        icon: '🐦',
        action: (data) => this.shareToTwitter(data)
      },
      {
        name: 'WhatsApp',
        icon: '💬',
        action: (data) => this.shareToWhatsApp(data)
      },
      {
        name: 'Telegram',
        icon: '✈️',
        action: (data) => this.shareToTelegram(data)
      }
    ];
  }
  
  async copyToClipboard(data) {
    const text = data.url || data.text;
    try {
      await navigator.clipboard.writeText(text);
      return { success: true, message: 'Copied to clipboard' };
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      return { success: true, message: 'Copied to clipboard' };
    }
  }
  
  shareViaEmail(data) {
    const subject = encodeURIComponent(data.title || 'Shared Content');
    const body = encodeURIComponent(`${data.text || ''}\n\n${data.url || ''}`);
    window.location.href = `mailto:?subject=${subject}&body=${body}`;
    return { success: true };
  }
  
  shareToTwitter(data) {
    const text = encodeURIComponent(data.text || data.title || '');
    const url = encodeURIComponent(data.url || '');
    window.open(
      `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      '_blank',
      'width=550,height=420'
    );
    return { success: true };
  }
  
  shareToWhatsApp(data) {
    const text = encodeURIComponent(`${data.text || ''} ${data.url || ''}`);
    window.open(`https://wa.me/?text=${text}`, '_blank');
    return { success: true };
  }
  
  shareToTelegram(data) {
    const text = encodeURIComponent(data.text || '');
    const url = encodeURIComponent(data.url || '');
    window.open(
      `https://t.me/share/url?url=${url}&text=${text}`,
      '_blank'
    );
    return { success: true };
  }
  
  showMenu(data, container) {
    const menu = document.createElement('div');
    menu.className = 'share-menu';
    
    menu.innerHTML = `
      <div class="share-menu-backdrop"></div>
      <div class="share-menu-content">
        <div class="share-menu-header">
          <span>Share</span>
          <button class="share-menu-close">✕</button>
        </div>
        <div class="share-menu-options">
          ${this.services.map((service, i) => `
            <button class="share-option" data-index="${i}">
              <span class="share-icon">${service.icon}</span>
              <span class="share-name">${service.name}</span>
            </button>
          `).join('')}
        </div>
      </div>
    `;
    
    // Event handlers
    menu.querySelector('.share-menu-backdrop').onclick = () => menu.remove();
    menu.querySelector('.share-menu-close').onclick = () => menu.remove();
    
    menu.querySelectorAll('.share-option').forEach(btn => {
      btn.onclick = async () => {
        const index = parseInt(btn.dataset.index);
        const result = await this.services[index].action(data);
        if (result.message) {
          this.showToast(result.message);
        }
        menu.remove();
      };
    });
    
    container.appendChild(menu);
  }
  
  showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'share-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.classList.add('visible'), 10);
    setTimeout(() => {
      toast.classList.remove('visible');
      setTimeout(() => toast.remove(), 300);
    }, 2000);
  }
}
```

### Fallback Menu Styles

```css
.share-menu {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.share-menu-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
}

.share-menu-content {
  position: relative;
  width: 100%;
  max-width: 400px;
  background: white;
  border-radius: 16px 16px 0 0;
  padding: 16px;
  padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px));
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from { transform: translateY(100%); }
  to { transform: translateY(0); }
}

.share-menu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  font-size: 18px;
  font-weight: 600;
}

.share-menu-close {
  background: none;
  border: none;
  font-size: 20px;
  padding: 8px;
  cursor: pointer;
}

.share-menu-options {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.share-option {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: none;
  border: none;
  cursor: pointer;
}

.share-icon {
  font-size: 32px;
}

.share-name {
  font-size: 12px;
  color: #666;
}

.share-toast {
  position: fixed;
  bottom: 100px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 24px;
  background: #333;
  color: white;
  border-radius: 24px;
  opacity: 0;
  transition: opacity 0.3s;
}

.share-toast.visible {
  opacity: 1;
}
```

### Unified Share Handler

```javascript
class UnifiedShareHandler {
  constructor() {
    this.nativeShare = new ShareManager();
    this.fallbackShare = new FallbackShare();
  }
  
  async share(data, container = document.body) {
    // Try native share first
    if (this.nativeShare.isSupported()) {
      const result = await this.nativeShare.share(data);
      if (result.success || result.reason === 'cancelled') {
        return result;
      }
      // Fall through to fallback if native failed
    }
    
    // Show fallback menu
    this.fallbackShare.showMenu(data, container);
    return { success: true, method: 'fallback' };
  }
}
```

---

## Share UI Components

### Share Button

```javascript
class ShareButton {
  constructor(options = {}) {
    this.shareHandler = options.shareHandler || new UnifiedShareHandler();
    this.getData = options.getData || (() => ({}));
    this.onShare = options.onShare || (() => {});
    
    this.element = this.createElement();
    this.attachEvents();
  }
  
  createElement() {
    const button = document.createElement('button');
    button.className = 'share-button';
    button.innerHTML = `
      <svg viewBox="0 0 24 24" width="20" height="20">
        <path fill="currentColor" d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z"/>
      </svg>
      <span>Share</span>
    `;
    return button;
  }
  
  attachEvents() {
    this.element.addEventListener('click', async () => {
      const data = this.getData();
      const result = await this.shareHandler.share(data);
      this.onShare(result);
    });
  }
  
  mount(container) {
    container.appendChild(this.element);
  }
}
```

### Message Share Action

```javascript
function addShareToMessage(messageElement, messageData) {
  const shareBtn = document.createElement('button');
  shareBtn.className = 'message-action share-action';
  shareBtn.innerHTML = '↗';
  shareBtn.title = 'Share message';
  
  shareBtn.addEventListener('click', async () => {
    const shareHandler = new UnifiedShareHandler();
    await shareHandler.share({
      text: messageData.content,
      title: 'Shared from Chat'
    });
  });
  
  messageElement.querySelector('.message-actions')?.appendChild(shareBtn);
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Check `navigator.share` before showing share UI | Assume share is available |
| Use `canShare()` before attempting file shares | Try sharing unsupported file types |
| Handle `AbortError` gracefully (user cancelled) | Treat cancellation as an error |
| Provide fallback for unsupported browsers | Break on browsers without share |
| Trigger share from user gesture | Call share on page load |
| Test on multiple platforms | Only test on one browser |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Calling `share()` without user gesture | Attach to click/tap handler |
| Not handling `AbortError` | Check `error.name === 'AbortError'` |
| Sharing empty data | Require at least one of text/url/files |
| Ignoring `canShare()` for files | Always verify before file sharing |
| No fallback for desktop browsers | Implement fallback share menu |
| Assuming file types are supported | Test each MIME type with `canShare()` |

---

## Hands-on Exercise

### Your Task

Build a complete sharing system for a chat interface:

1. **Share button** on each message
2. **Native share** when available
3. **Fallback menu** with copy, email, and social options
4. **File sharing** for image messages (when supported)

### Requirements

1. Detect Web Share API support
2. Show appropriate UI based on capabilities
3. Handle all error cases gracefully
4. Provide at least 3 fallback options

<details>
<summary>💡 Hints (click to expand)</summary>

- Test `'share' in navigator` for basic support
- Use `canShare()` to verify file sharing
- `AbortError` means user cancelled—not an error
- Fallback menu should slide up from bottom

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Chat Share Demo</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: -apple-system, sans-serif;
      background: #f5f5f5;
      padding: 20px;
    }
    
    .message {
      background: white;
      padding: 12px 16px;
      border-radius: 12px;
      margin-bottom: 12px;
      position: relative;
    }
    
    .message-content {
      margin-bottom: 8px;
    }
    
    .message-actions {
      display: flex;
      gap: 12px;
    }
    
    .action-btn {
      background: none;
      border: none;
      color: #007AFF;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    /* Fallback menu styles */
    .share-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.4);
      z-index: 100;
      display: flex;
      align-items: flex-end;
    }
    
    .share-sheet {
      width: 100%;
      background: white;
      border-radius: 16px 16px 0 0;
      padding: 20px;
      padding-bottom: calc(20px + env(safe-area-inset-bottom));
    }
    
    .share-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 20px;
      font-weight: 600;
    }
    
    .share-options {
      display: flex;
      gap: 24px;
      overflow-x: auto;
    }
    
    .share-option {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      background: none;
      border: none;
      cursor: pointer;
    }
    
    .share-icon {
      width: 50px;
      height: 50px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
    }
    
    .share-option.copy .share-icon { background: #f0f0f0; }
    .share-option.email .share-icon { background: #e8f4fd; }
    .share-option.twitter .share-icon { background: #e8f6fd; }
    .share-option.whatsapp .share-icon { background: #e8fde8; }
    
    .share-label { font-size: 12px; color: #666; }
    
    .toast {
      position: fixed;
      bottom: 100px;
      left: 50%;
      transform: translateX(-50%);
      background: #333;
      color: white;
      padding: 10px 20px;
      border-radius: 20px;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <div id="messages">
    <div class="message" data-content="Hello! This is a test message for sharing.">
      <div class="message-content">Hello! This is a test message for sharing.</div>
      <div class="message-actions">
        <button class="action-btn share-btn">↗ Share</button>
      </div>
    </div>
    <div class="message" data-content="Check out this cool AI feature we built!">
      <div class="message-content">Check out this cool AI feature we built!</div>
      <div class="message-actions">
        <button class="action-btn share-btn">↗ Share</button>
      </div>
    </div>
  </div>

  <script>
    class ChatShare {
      constructor() {
        this.hasNativeShare = 'share' in navigator;
        this.attachEvents();
      }
      
      attachEvents() {
        document.querySelectorAll('.share-btn').forEach(btn => {
          btn.addEventListener('click', (e) => {
            const message = e.target.closest('.message');
            const content = message.dataset.content;
            this.share({ text: content, title: 'Chat Message' });
          });
        });
      }
      
      async share(data) {
        if (this.hasNativeShare) {
          try {
            await navigator.share(data);
            return;
          } catch (err) {
            if (err.name === 'AbortError') return;
          }
        }
        this.showFallback(data);
      }
      
      showFallback(data) {
        const overlay = document.createElement('div');
        overlay.className = 'share-overlay';
        overlay.innerHTML = `
          <div class="share-sheet">
            <div class="share-header">
              <span>Share</span>
              <button class="close-btn">✕</button>
            </div>
            <div class="share-options">
              <button class="share-option copy">
                <div class="share-icon">📋</div>
                <span class="share-label">Copy</span>
              </button>
              <button class="share-option email">
                <div class="share-icon">✉️</div>
                <span class="share-label">Email</span>
              </button>
              <button class="share-option twitter">
                <div class="share-icon">🐦</div>
                <span class="share-label">Twitter</span>
              </button>
              <button class="share-option whatsapp">
                <div class="share-icon">💬</div>
                <span class="share-label">WhatsApp</span>
              </button>
            </div>
          </div>
        `;
        
        const close = () => overlay.remove();
        overlay.querySelector('.close-btn').onclick = close;
        overlay.onclick = (e) => { if (e.target === overlay) close(); };
        
        overlay.querySelector('.copy').onclick = async () => {
          await navigator.clipboard.writeText(data.text);
          close();
          this.toast('Copied!');
        };
        
        overlay.querySelector('.email').onclick = () => {
          const body = encodeURIComponent(data.text);
          window.location.href = `mailto:?body=${body}`;
          close();
        };
        
        overlay.querySelector('.twitter').onclick = () => {
          const text = encodeURIComponent(data.text);
          window.open(`https://twitter.com/intent/tweet?text=${text}`);
          close();
        };
        
        overlay.querySelector('.whatsapp').onclick = () => {
          const text = encodeURIComponent(data.text);
          window.open(`https://wa.me/?text=${text}`);
          close();
        };
        
        document.body.appendChild(overlay);
      }
      
      toast(msg) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => toast.remove(), 300);
        }, 2000);
      }
    }
    
    new ChatShare();
  </script>
</body>
</html>
```

</details>

---

## Summary

✅ The Web Share API enables native sharing from web applications  
✅ Requires HTTPS and user gesture activation  
✅ Use `canShare()` to verify file sharing capabilities  
✅ Handle `AbortError` separately—it means user cancelled  
✅ Provide fallback sharing for unsupported browsers  
✅ Test on multiple platforms—behavior varies significantly  

**Next:** [Mobile-Specific Input](./06-mobile-specific-input.md)

---

<!-- 
Sources Consulted:
- MDN Web Share API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Share_API
- MDN navigator.share(): https://developer.mozilla.org/en-US/docs/Web/API/Navigator/share
- MDN navigator.canShare(): https://developer.mozilla.org/en-US/docs/Web/API/Navigator/canShare
- Can I Use Web Share: https://caniuse.com/web-share
-->
