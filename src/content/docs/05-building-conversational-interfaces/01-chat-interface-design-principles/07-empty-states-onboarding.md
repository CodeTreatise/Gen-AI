---
title: "Empty States & Onboarding"
---

# Empty States & Onboarding

## Introduction

The first thing users see in a new chat interface is... nothing. An empty conversation. This blank slate is your opportunity to make a great first impression, guide users toward their first interaction, and showcase what your AI can do.

In this lesson, we'll explore patterns for designing engaging empty states and effective onboarding experiences.

### What We'll Cover

- First-time user welcome experience design
- Suggested prompt templates and categories
- Feature discovery through progressive disclosure
- Tutorial and walkthrough patterns
- Conversation starters and examples

### Prerequisites

- HTML/CSS fundamentals ([Unit 1](../../../01-web-development-fundamentals/02-css-fundamentals/00-css-fundamentals.md))
- Basic JavaScript event handling
- Understanding of UX principles

---

## First-Time User Experience

### The Welcome Screen

Create a warm, informative welcome that reduces uncertainty:

```html
<div class="empty-state">
  <div class="welcome-content">
    <div class="ai-avatar">
      <img src="/ai-logo.svg" alt="AI Assistant">
    </div>
    <h1>Welcome to AI Assistant</h1>
    <p class="welcome-description">
      I can help you with coding, writing, analysis, and more. 
      Ask me anything or try one of the suggestions below.
    </p>
  </div>
  
  <div class="suggestion-grid">
    <!-- Suggested prompts here -->
  </div>
</div>
```

```css
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  padding: 2rem;
  text-align: center;
}

.welcome-content {
  max-width: 32rem;
  margin-bottom: 2rem;
}

.ai-avatar {
  width: 4rem;
  height: 4rem;
  margin: 0 auto 1.5rem;
  border-radius: 50%;
  background: linear-gradient(135deg, #8b5cf6, #3b82f6);
  display: flex;
  align-items: center;
  justify-content: center;
}

.ai-avatar img {
  width: 2.5rem;
  height: 2.5rem;
}

.empty-state h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: #1f2937;
}

.welcome-description {
  color: #6b7280;
  line-height: 1.6;
}
```

### Capability Highlights

Show what the AI can do:

```html
<div class="capabilities">
  <h2 class="visually-hidden">What I can help with</h2>
  <div class="capability-grid">
    <div class="capability-card">
      <span class="capability-icon">💻</span>
      <h3>Write Code</h3>
      <p>Generate, debug, and explain code in any language</p>
    </div>
    <div class="capability-card">
      <span class="capability-icon">✍️</span>
      <h3>Draft Content</h3>
      <p>Write emails, articles, and creative pieces</p>
    </div>
    <div class="capability-card">
      <span class="capability-icon">📊</span>
      <h3>Analyze Data</h3>
      <p>Interpret data and create visualizations</p>
    </div>
    <div class="capability-card">
      <span class="capability-icon">🧠</span>
      <h3>Brainstorm</h3>
      <p>Generate ideas and solve problems creatively</p>
    </div>
  </div>
</div>
```

```css
.capability-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  max-width: 48rem;
  margin: 0 auto;
}

.capability-card {
  padding: 1.25rem;
  background: #f9fafb;
  border-radius: 1rem;
  text-align: left;
}

.capability-icon {
  font-size: 1.5rem;
  display: block;
  margin-bottom: 0.75rem;
}

.capability-card h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.capability-card p {
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
}
```

---

## Suggested Prompts

### Prompt Categories

Organize suggestions by use case:

```html
<div class="prompt-suggestions">
  <h2>Try asking about...</h2>
  
  <div class="prompt-categories">
    <button class="category-tab active" data-category="all">All</button>
    <button class="category-tab" data-category="code">Coding</button>
    <button class="category-tab" data-category="write">Writing</button>
    <button class="category-tab" data-category="learn">Learning</button>
  </div>
  
  <div class="prompt-grid">
    <button class="prompt-card" data-category="code">
      <span class="prompt-icon">🐛</span>
      <span class="prompt-text">Debug this error message</span>
    </button>
    <button class="prompt-card" data-category="code">
      <span class="prompt-icon">📝</span>
      <span class="prompt-text">Write a function that...</span>
    </button>
    <button class="prompt-card" data-category="write">
      <span class="prompt-icon">✉️</span>
      <span class="prompt-text">Draft a professional email</span>
    </button>
    <button class="prompt-card" data-category="learn">
      <span class="prompt-icon">📚</span>
      <span class="prompt-text">Explain how async/await works</span>
    </button>
  </div>
</div>
```

```css
.prompt-suggestions {
  width: 100%;
  max-width: 48rem;
}

.prompt-suggestions h2 {
  font-size: 0.875rem;
  font-weight: 500;
  color: #6b7280;
  margin-bottom: 1rem;
}

.prompt-categories {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  overflow-x: auto;
  padding-bottom: 0.5rem;
}

.category-tab {
  padding: 0.5rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 2rem;
  background: white;
  font-size: 0.875rem;
  white-space: nowrap;
  cursor: pointer;
  transition: all 0.2s ease;
}

.category-tab:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.category-tab.active {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.prompt-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
}

.prompt-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
}

.prompt-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.prompt-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.prompt-text {
  font-size: 0.875rem;
  color: #374151;
}
```

### Interactive Prompt Templates

Make prompts actionable:

```javascript
function setupPromptCards() {
  const promptCards = document.querySelectorAll('.prompt-card');
  const messageInput = document.querySelector('#message-input');
  
  promptCards.forEach(card => {
    card.addEventListener('click', () => {
      const promptText = card.querySelector('.prompt-text').textContent;
      messageInput.value = promptText;
      messageInput.focus();
      
      // Position cursor at end or at placeholder
      if (promptText.includes('...')) {
        const cursorPos = promptText.indexOf('...');
        messageInput.setSelectionRange(cursorPos, cursorPos + 3);
      }
    });
  });
}

function setupCategoryTabs() {
  const tabs = document.querySelectorAll('.category-tab');
  const cards = document.querySelectorAll('.prompt-card');
  
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const category = tab.dataset.category;
      
      // Update active tab
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      // Filter cards
      cards.forEach(card => {
        if (category === 'all' || card.dataset.category === category) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });
}
```

---

## Feature Discovery

### Progressive Disclosure

Reveal features as users explore:

```html
<div class="feature-hints">
  <div class="hint-card" data-hint="keyboard-shortcuts">
    <div class="hint-icon">⌨️</div>
    <div class="hint-content">
      <strong>Keyboard shortcuts</strong>
      <p>Press <kbd>Ctrl</kbd>+<kbd>/</kbd> to see all shortcuts</p>
    </div>
    <button class="hint-dismiss" aria-label="Dismiss">×</button>
  </div>
</div>
```

```css
.feature-hints {
  position: fixed;
  bottom: 5rem;
  right: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  z-index: 100;
}

.hint-card {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: #1f2937;
  color: white;
  border-radius: 0.75rem;
  max-width: 300px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(1rem);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.hint-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.hint-content {
  flex: 1;
}

.hint-content strong {
  display: block;
  margin-bottom: 0.25rem;
}

.hint-content p {
  font-size: 0.875rem;
  color: #9ca3af;
  margin: 0;
}

.hint-content kbd {
  display: inline-block;
  padding: 0.125rem 0.375rem;
  background: #374151;
  border-radius: 0.25rem;
  font-family: inherit;
  font-size: 0.75rem;
}

.hint-dismiss {
  background: none;
  border: none;
  color: #9ca3af;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.hint-dismiss:hover {
  color: white;
}
```

### Contextual Tooltips

Show tips when relevant:

```javascript
const hints = [
  {
    id: 'keyboard-shortcuts',
    trigger: 'first-message',
    content: 'Use Shift+Enter for new lines'
  },
  {
    id: 'copy-code',
    trigger: 'first-code-block',
    content: 'Click the copy button to copy code'
  },
  {
    id: 'regenerate',
    trigger: 'first-response',
    content: 'Not satisfied? Click regenerate for a new response'
  }
];

function showHint(hintId) {
  const hint = hints.find(h => h.id === hintId);
  if (!hint || localStorage.getItem(`hint-${hintId}-dismissed`)) return;
  
  const hintElement = createHintElement(hint);
  document.querySelector('.feature-hints').appendChild(hintElement);
  
  // Auto-dismiss after 10 seconds
  setTimeout(() => {
    hintElement.remove();
  }, 10000);
}

function dismissHint(hintId) {
  localStorage.setItem(`hint-${hintId}-dismissed`, 'true');
  document.querySelector(`[data-hint="${hintId}"]`)?.remove();
}
```

---

## Tutorial Patterns

### Step-by-Step Walkthrough

```html
<div class="onboarding-overlay" id="onboarding">
  <div class="onboarding-modal">
    <div class="onboarding-progress">
      <span class="step active"></span>
      <span class="step"></span>
      <span class="step"></span>
    </div>
    
    <div class="onboarding-step" data-step="1">
      <img src="/onboarding-1.svg" alt="Type a message">
      <h2>Start a Conversation</h2>
      <p>Type your question or request in the input box below. I can help with coding, writing, research, and more.</p>
    </div>
    
    <div class="onboarding-step hidden" data-step="2">
      <img src="/onboarding-2.svg" alt="Get responses">
      <h2>Get Instant Answers</h2>
      <p>I'll respond in real-time. You can copy code, regenerate responses, or continue the conversation.</p>
    </div>
    
    <div class="onboarding-step hidden" data-step="3">
      <img src="/onboarding-3.svg" alt="Advanced features">
      <h2>Explore More</h2>
      <p>Upload files, use voice input, or switch between models. There's a lot to discover!</p>
    </div>
    
    <div class="onboarding-actions">
      <button class="skip-btn" onclick="finishOnboarding()">Skip</button>
      <button class="next-btn" onclick="nextStep()">Next</button>
    </div>
  </div>
</div>
```

```css
.onboarding-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.onboarding-modal {
  background: white;
  border-radius: 1rem;
  padding: 2rem;
  max-width: 400px;
  text-align: center;
}

.onboarding-progress {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 2rem;
}

.onboarding-progress .step {
  width: 2rem;
  height: 0.25rem;
  background: #e5e7eb;
  border-radius: 0.125rem;
  transition: background 0.3s ease;
}

.onboarding-progress .step.active {
  background: #3b82f6;
}

.onboarding-step img {
  width: 200px;
  height: 150px;
  object-fit: contain;
  margin-bottom: 1.5rem;
}

.onboarding-step h2 {
  font-size: 1.25rem;
  margin-bottom: 0.75rem;
}

.onboarding-step p {
  color: #6b7280;
  line-height: 1.6;
}

.onboarding-step.hidden {
  display: none;
}

.onboarding-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 2rem;
}

.skip-btn {
  padding: 0.75rem 1.5rem;
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
}

.next-btn {
  padding: 0.75rem 2rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
}
```

```javascript
let currentStep = 1;
const totalSteps = 3;

function nextStep() {
  if (currentStep >= totalSteps) {
    finishOnboarding();
    return;
  }
  
  // Hide current step
  document.querySelector(`[data-step="${currentStep}"]`).classList.add('hidden');
  
  // Show next step
  currentStep++;
  document.querySelector(`[data-step="${currentStep}"]`).classList.remove('hidden');
  
  // Update progress
  document.querySelectorAll('.onboarding-progress .step').forEach((step, index) => {
    step.classList.toggle('active', index < currentStep);
  });
  
  // Update button text
  if (currentStep === totalSteps) {
    document.querySelector('.next-btn').textContent = 'Get Started';
  }
}

function finishOnboarding() {
  localStorage.setItem('onboarding-completed', 'true');
  document.getElementById('onboarding').remove();
}

// Show onboarding only for new users
if (!localStorage.getItem('onboarding-completed')) {
  document.getElementById('onboarding').style.display = 'flex';
} else {
  document.getElementById('onboarding').remove();
}
```

---

## Conversation Starters

### Dynamic Suggestions

Rotate suggestions based on context:

```javascript
const conversationStarters = {
  morning: [
    "Help me plan my day",
    "Summarize the latest news",
    "What should I have for breakfast?"
  ],
  afternoon: [
    "I need help with a work task",
    "Review this code for me",
    "Help me write an email"
  ],
  evening: [
    "Recommend a movie to watch",
    "Help me unwind with some trivia",
    "What's a quick recipe for dinner?"
  ],
  coding: [
    "Debug this error message",
    "Explain this code snippet",
    "Write a function that..."
  ],
  general: [
    "Tell me an interesting fact",
    "Help me brainstorm ideas",
    "Explain a complex topic simply"
  ]
};

function getContextualStarters() {
  const hour = new Date().getHours();
  let timeContext = 'afternoon';
  
  if (hour < 12) timeContext = 'morning';
  else if (hour >= 18) timeContext = 'evening';
  
  // Mix time-based and general suggestions
  return [
    ...conversationStarters[timeContext].slice(0, 2),
    ...conversationStarters.general.slice(0, 2)
  ];
}

function renderStarters() {
  const starters = getContextualStarters();
  const container = document.querySelector('.starter-buttons');
  
  container.innerHTML = starters.map(starter => `
    <button class="starter-btn">${starter}</button>
  `).join('');
}
```

### Quick Action Buttons

```html
<div class="quick-actions">
  <button class="quick-action" data-action="upload">
    <span class="action-icon">📎</span>
    <span>Upload a file</span>
  </button>
  <button class="quick-action" data-action="voice">
    <span class="action-icon">🎤</span>
    <span>Start voice chat</span>
  </button>
  <button class="quick-action" data-action="image">
    <span class="action-icon">🖼️</span>
    <span>Analyze an image</span>
  </button>
</div>
```

```css
.quick-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 1.5rem;
}

.quick-action {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 2rem;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.quick-action:hover {
  border-color: #3b82f6;
  background: #f0f9ff;
}

.action-icon {
  font-size: 1rem;
}
```

---

## Complete Empty State Component

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chat - Get Started</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f9fafb;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    
    .chat-app {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 48rem;
      margin: 0 auto;
      width: 100%;
    }
    
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      text-align: center;
    }
    
    .welcome-section {
      margin-bottom: 2rem;
    }
    
    .ai-avatar {
      width: 4rem;
      height: 4rem;
      margin: 0 auto 1.5rem;
      border-radius: 50%;
      background: linear-gradient(135deg, #8b5cf6, #3b82f6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2rem;
    }
    
    .welcome-section h1 {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #1f2937;
    }
    
    .welcome-section p {
      color: #6b7280;
      max-width: 24rem;
    }
    
    .suggestion-section h2 {
      font-size: 0.875rem;
      font-weight: 500;
      color: #6b7280;
      margin-bottom: 1rem;
    }
    
    .suggestion-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.75rem;
      width: 100%;
      max-width: 32rem;
    }
    
    .suggestion-card {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 1rem;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.75rem;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .suggestion-card:hover {
      border-color: #3b82f6;
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
    }
    
    .suggestion-icon {
      font-size: 1.25rem;
      flex-shrink: 0;
    }
    
    .suggestion-text {
      font-size: 0.875rem;
      color: #374151;
    }
    
    .input-area {
      padding: 1rem;
      background: white;
      border-top: 1px solid #e5e7eb;
    }
    
    .input-container {
      display: flex;
      gap: 0.5rem;
      max-width: 48rem;
      margin: 0 auto;
    }
    
    #message-input {
      flex: 1;
      padding: 0.875rem 1.25rem;
      border: 1px solid #e5e7eb;
      border-radius: 1.5rem;
      font-size: 1rem;
      resize: none;
      outline: none;
    }
    
    #message-input:focus {
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .send-button {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: #3b82f6;
      color: white;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    @media (max-width: 640px) {
      .suggestion-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="chat-app">
    <div class="empty-state" id="empty-state">
      <div class="welcome-section">
        <div class="ai-avatar">✨</div>
        <h1>How can I help you today?</h1>
        <p>Ask me anything—I can help with coding, writing, analysis, brainstorming, and more.</p>
      </div>
      
      <div class="suggestion-section">
        <h2>Try one of these</h2>
        <div class="suggestion-grid">
          <button class="suggestion-card">
            <span class="suggestion-icon">💻</span>
            <span class="suggestion-text">Explain async/await in JavaScript</span>
          </button>
          <button class="suggestion-card">
            <span class="suggestion-icon">✍️</span>
            <span class="suggestion-text">Help me write a cover letter</span>
          </button>
          <button class="suggestion-card">
            <span class="suggestion-icon">🐛</span>
            <span class="suggestion-text">Debug this error message...</span>
          </button>
          <button class="suggestion-card">
            <span class="suggestion-icon">💡</span>
            <span class="suggestion-text">Brainstorm project ideas for...</span>
          </button>
        </div>
      </div>
    </div>
    
    <div class="message-list" id="messages" style="display: none;"></div>
    
    <div class="input-area">
      <div class="input-container">
        <textarea 
          id="message-input" 
          rows="1" 
          placeholder="Message AI Assistant..."
          aria-label="Type your message"
        ></textarea>
        <button class="send-button" aria-label="Send message">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
  
  <script>
    const input = document.querySelector('#message-input');
    const sendButton = document.querySelector('.send-button');
    const emptyState = document.querySelector('#empty-state');
    const messageList = document.querySelector('#messages');
    const suggestionCards = document.querySelectorAll('.suggestion-card');
    
    // Handle suggestion clicks
    suggestionCards.forEach(card => {
      card.addEventListener('click', () => {
        const text = card.querySelector('.suggestion-text').textContent;
        input.value = text;
        input.focus();
        
        // Position cursor at "..." if present
        if (text.includes('...')) {
          const pos = text.indexOf('...');
          input.setSelectionRange(pos, pos + 3);
        }
      });
    });
    
    // Send message
    function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      // Hide empty state, show messages
      emptyState.style.display = 'none';
      messageList.style.display = 'block';
      
      // Add user message
      const userMsg = document.createElement('div');
      userMsg.className = 'message user';
      userMsg.textContent = text;
      messageList.appendChild(userMsg);
      
      input.value = '';
      
      // Simulate AI response
      setTimeout(() => {
        const aiMsg = document.createElement('div');
        aiMsg.className = 'message ai';
        aiMsg.textContent = 'This is where the AI response would appear...';
        messageList.appendChild(aiMsg);
      }, 500);
    }
    
    sendButton.addEventListener('click', sendMessage);
    
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show capability highlights for new users | Leave empty state completely blank |
| Provide clickable prompt suggestions | Make users think of prompts from scratch |
| Use progressive disclosure for features | Overwhelm with all features at once |
| Remember dismissed hints | Show same hints repeatedly |
| Offer contextual suggestions | Show irrelevant generic prompts |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Empty state feels cold/impersonal | Add personality with avatar and warm copy |
| Too many suggestions overwhelm | Limit to 4-6 curated suggestions |
| Suggestions don't match user intent | Categorize and personalize suggestions |
| Onboarding feels mandatory | Always offer skip option |
| Hints interrupt workflow | Show hints in non-blocking UI |

---

## Summary

✅ **Welcome screens** set the tone and reduce first-use anxiety  
✅ **Suggested prompts** lower the barrier to first interaction  
✅ **Progressive disclosure** reveals features at the right moment  
✅ **Onboarding tutorials** guide users through key capabilities  
✅ **Contextual starters** provide relevant, timely suggestions

---

## Further Reading

- [Empty States Design Patterns](https://www.nngroup.com/articles/empty-state/)
- [Onboarding UX Best Practices](https://www.nngroup.com/articles/user-onboarding/)
- [Google Conversation Design: Greetings](https://developers.google.com/assistant/conversation-design/greetings)

---

**Previous:** [Mobile-Responsive Chat Design](./06-mobile-responsive-chat-design.md)  
**Back to:** [Chat Interface Design Principles](./00-chat-interface-design-principles.md)

<!-- 
Sources Consulted:
- Nielsen Norman Group Empty States: https://www.nngroup.com/articles/empty-state/
- Google Conversation Design: https://developers.google.com/assistant/conversation-design/
- ChatGPT/Claude UI patterns (observation)
-->
