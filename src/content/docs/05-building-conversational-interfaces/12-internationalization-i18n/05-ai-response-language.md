---
title: "AI Response Language Handling"
---

# AI Response Language Handling

## Introduction

When a French user asks a question in French, they expect the AI to respond in French—not English. When an Arabic user switches mid-conversation to English for technical terms, the AI should adapt appropriately. AI response language handling goes beyond simple translation; it requires detecting language preferences, maintaining consistency, and handling the inherent unpredictability of LLM outputs.

This lesson covers how to ensure AI responses match user expectations, how to handle language switching, and how to build robust systems for multilingual AI conversations.

### What We'll Cover

- Storing and applying user language preferences
- Prompting LLMs to respond in specific languages
- Detecting AI response language
- Handling language mismatches gracefully
- Mixed-language conversations
- Language fallback strategies

### Prerequisites

- Understanding of LLM API basics (prompts, completions)
- Previous lessons on translation keys and locale detection
- JavaScript async/await for API calls

---

## User Language Preferences

### Storage Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| User profile | Persistent, explicit | Requires account |
| localStorage | Persistent, no account | Per-device only |
| URL parameter | Shareable, bookmarkable | Lost on navigation |
| Session cookie | Per-session persistence | Clears on close |
| Detect each message | Dynamic | Inconsistent, slow |

### Complete Preference Management

```javascript
class LanguagePreferenceManager {
  constructor(options = {}) {
    this.supportedLanguages = options.supportedLanguages || ['en', 'es', 'fr', 'de', 'ar', 'zh', 'ja'];
    this.defaultLanguage = options.defaultLanguage || 'en';
    this.storageKey = options.storageKey || 'chat_language';
    this.apiEndpoint = options.apiEndpoint || null;  // For server sync
  }
  
  // Get current language preference
  async get(userId = null) {
    // Priority 1: User profile (if logged in)
    if (userId && this.apiEndpoint) {
      const serverPref = await this.fetchFromServer(userId);
      if (serverPref) return serverPref;
    }
    
    // Priority 2: Local storage
    const stored = localStorage.getItem(this.storageKey);
    if (stored && this.supportedLanguages.includes(stored)) {
      return stored;
    }
    
    // Priority 3: Browser language
    const browserLang = this.detectBrowserLanguage();
    if (browserLang) return browserLang;
    
    // Fallback
    return this.defaultLanguage;
  }
  
  // Set language preference
  async set(language, userId = null) {
    if (!this.supportedLanguages.includes(language)) {
      console.warn(`Unsupported language: ${language}`);
      return false;
    }
    
    // Save locally
    localStorage.setItem(this.storageKey, language);
    
    // Sync to server if user is logged in
    if (userId && this.apiEndpoint) {
      await this.syncToServer(userId, language);
    }
    
    // Dispatch event for UI updates
    window.dispatchEvent(new CustomEvent('languageChanged', { 
      detail: { language } 
    }));
    
    return true;
  }
  
  detectBrowserLanguage() {
    const languages = navigator.languages || [navigator.language];
    
    for (const lang of languages) {
      // Try exact match first
      if (this.supportedLanguages.includes(lang)) {
        return lang;
      }
      
      // Try base language (e.g., "en" from "en-US")
      const base = lang.split('-')[0];
      if (this.supportedLanguages.includes(base)) {
        return base;
      }
    }
    
    return null;
  }
  
  async fetchFromServer(userId) {
    try {
      const response = await fetch(`${this.apiEndpoint}/users/${userId}/preferences`);
      if (response.ok) {
        const data = await response.json();
        return data.language;
      }
    } catch (error) {
      console.error('Failed to fetch language preference:', error);
    }
    return null;
  }
  
  async syncToServer(userId, language) {
    try {
      await fetch(`${this.apiEndpoint}/users/${userId}/preferences`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language })
      });
    } catch (error) {
      console.error('Failed to sync language preference:', error);
    }
  }
  
  // Get language info for display
  getLanguageInfo(code) {
    const displayNames = new Intl.DisplayNames([code], { type: 'language' });
    return {
      code: code,
      nativeName: displayNames.of(code),
      englishName: new Intl.DisplayNames(['en'], { type: 'language' }).of(code)
    };
  }
  
  // Get all supported languages with display names
  getSupportedLanguages(displayLocale = 'en') {
    const displayNames = new Intl.DisplayNames([displayLocale], { type: 'language' });
    
    return this.supportedLanguages.map(code => ({
      code: code,
      name: displayNames.of(code),
      nativeName: new Intl.DisplayNames([code], { type: 'language' }).of(code)
    }));
  }
}

// Usage
const langManager = new LanguagePreferenceManager({
  supportedLanguages: ['en', 'es', 'fr', 'de', 'ar', 'zh-Hans', 'ja'],
  defaultLanguage: 'en'
});

const currentLang = await langManager.get();
console.log(`Current language: ${currentLang}`);

// Display language picker
const languages = langManager.getSupportedLanguages('en');
// [
//   { code: 'en', name: 'English', nativeName: 'English' },
//   { code: 'es', name: 'Spanish', nativeName: 'Español' },
//   { code: 'ar', name: 'Arabic', nativeName: 'العربية' },
//   ...
// ]
```

---

## Prompting LLMs for Specific Languages

The most reliable way to get AI responses in a specific language is through system prompts.

### System Prompt Approaches

**Approach 1: Direct Language Instruction**
```javascript
function buildSystemPrompt(language) {
  return `You are a helpful assistant. Always respond in ${language}. 
Do not switch languages unless explicitly asked. 
If the user writes in a different language, still respond in ${language}.`;
}

// Usage
const systemPrompt = buildSystemPrompt('Spanish');
```

**Approach 2: Locale-Specific System Prompts**
```javascript
const systemPrompts = {
  en: `You are a helpful assistant. Respond in English.`,
  
  es: `Eres un asistente útil. Responde siempre en español.
Usa un tono amigable y profesional.`,
  
  ar: `أنت مساعد مفيد. الرجاء الرد دائماً باللغة العربية.
استخدم لغة واضحة ومهذبة.`,
  
  ja: `あなたは親切なアシスタントです。必ず日本語で回答してください。
丁寧な敬語を使用してください。`,
  
  zh: `你是一个有帮助的助手。请始终用中文回复。
使用清晰礼貌的语言。`
};

function getSystemPrompt(languageCode) {
  return systemPrompts[languageCode] || systemPrompts.en;
}
```

**Approach 3: Dynamic Prompt with Locale Data**
```javascript
function buildLocalizedSystemPrompt(languageCode, options = {}) {
  const languageName = new Intl.DisplayNames([languageCode], { 
    type: 'language' 
  }).of(languageCode);
  
  let prompt = `You are a helpful AI assistant.\n\n`;
  
  prompt += `IMPORTANT: Respond ONLY in ${languageName}.\n`;
  prompt += `- Do not translate your responses to other languages.\n`;
  prompt += `- If the user writes in a different language, still respond in ${languageName}.\n`;
  prompt += `- For technical terms with no good translation, you may keep them in the original language and explain.\n`;
  
  if (options.formalityLevel === 'formal') {
    prompt += `- Use formal/polite language register.\n`;
  }
  
  if (options.context) {
    prompt += `\nContext: ${options.context}\n`;
  }
  
  return prompt;
}

// Usage
const systemPrompt = buildLocalizedSystemPrompt('ja', { 
  formalityLevel: 'formal' 
});
```

### Complete API Integration

```javascript
class MultilingualChatClient {
  constructor(config) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'gpt-4';
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
    this.languagePrefs = new LanguagePreferenceManager(config.languageOptions);
  }
  
  async sendMessage(userMessage, conversationHistory = []) {
    const language = await this.languagePrefs.get();
    const systemPrompt = this.buildSystemPrompt(language);
    
    const messages = [
      { role: 'system', content: systemPrompt },
      ...conversationHistory,
      { role: 'user', content: userMessage }
    ];
    
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: this.model,
        messages: messages,
        temperature: 0.7
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    const assistantMessage = data.choices[0].message.content;
    
    // Verify response language
    const responseLanguage = await this.detectLanguage(assistantMessage);
    
    return {
      content: assistantMessage,
      expectedLanguage: language,
      detectedLanguage: responseLanguage,
      languageMismatch: responseLanguage !== language
    };
  }
  
  buildSystemPrompt(language) {
    const langName = new Intl.DisplayNames(['en'], { type: 'language' }).of(language);
    
    return `You are a helpful AI assistant.

LANGUAGE REQUIREMENT: Respond in ${langName} (${language}).
- Always respond in ${langName} regardless of the user's language.
- Keep technical terms in their original form when no good translation exists.
- Maintain consistent language throughout the conversation.`;
  }
  
  async detectLanguage(text) {
    // Simple detection based on character ranges
    // For production, use a proper language detection library or API
    const sample = text.substring(0, 200);
    
    // Check for script-specific characters
    if (/[\u0600-\u06FF]/.test(sample)) return 'ar';
    if (/[\u3040-\u309F\u30A0-\u30FF]/.test(sample)) return 'ja';
    if (/[\u4E00-\u9FFF]/.test(sample)) return 'zh';
    if (/[\uAC00-\uD7AF]/.test(sample)) return 'ko';
    if (/[\u0400-\u04FF]/.test(sample)) return 'ru';
    if (/[\u0590-\u05FF]/.test(sample)) return 'he';
    if (/[\u0E00-\u0E7F]/.test(sample)) return 'th';
    
    // For Latin scripts, would need more sophisticated detection
    return 'en'; // Default assumption
  }
}
```

---

## Language Detection for AI Responses

### Why Detect Response Language?

LLMs don't always follow language instructions perfectly. Detection allows you to:
- Verify the response matches expectations
- Trigger re-generation if wrong language
- Log mismatches for quality monitoring
- Apply fallback strategies

### Detection Methods

**Method 1: Character Script Detection (Fast, Limited)**
```javascript
function detectScriptLanguage(text) {
  const sample = text.substring(0, 500);
  
  const scripts = [
    { pattern: /[\u0600-\u06FF]/, lang: 'ar' },  // Arabic
    { pattern: /[\u0590-\u05FF]/, lang: 'he' },  // Hebrew
    { pattern: /[\u0900-\u097F]/, lang: 'hi' },  // Devanagari (Hindi)
    { pattern: /[\u3040-\u309F]/, lang: 'ja' },  // Hiragana
    { pattern: /[\u30A0-\u30FF]/, lang: 'ja' },  // Katakana
    { pattern: /[\u4E00-\u9FFF]/, lang: 'zh' },  // CJK (Chinese)
    { pattern: /[\uAC00-\uD7AF]/, lang: 'ko' },  // Korean
    { pattern: /[\u0400-\u04FF]/, lang: 'ru' },  // Cyrillic
    { pattern: /[\u0E00-\u0E7F]/, lang: 'th' },  // Thai
  ];
  
  for (const { pattern, lang } of scripts) {
    if (pattern.test(sample)) {
      return { language: lang, confidence: 0.9, method: 'script' };
    }
  }
  
  // Default to Latin script languages
  return { language: 'unknown', confidence: 0, method: 'script' };
}
```

**Method 2: Using External API (Accurate)**
```javascript
async function detectLanguageViaAPI(text, apiKey) {
  // Example using Google Cloud Translation API
  const response = await fetch(
    `https://translation.googleapis.com/language/translate/v2/detect?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: text.substring(0, 500) })
    }
  );
  
  const data = await response.json();
  const detection = data.data.detections[0][0];
  
  return {
    language: detection.language,
    confidence: detection.confidence,
    method: 'api'
  };
}
```

**Method 3: Lightweight Library (franc)**
```javascript
// Using franc library (npm install franc)
import { franc } from 'franc';

function detectLanguageLocal(text) {
  const code = franc(text.substring(0, 500));
  
  // franc returns ISO 639-3 codes, convert to ISO 639-1
  const codeMap = {
    'eng': 'en',
    'spa': 'es',
    'fra': 'fr',
    'deu': 'de',
    'ara': 'ar',
    'zho': 'zh',
    'jpn': 'ja',
    'kor': 'ko',
    'rus': 'ru'
    // Add more as needed
  };
  
  return {
    language: codeMap[code] || code,
    confidence: code !== 'und' ? 0.7 : 0,
    method: 'franc'
  };
}
```

### Comprehensive Detection Service

```javascript
class LanguageDetectionService {
  constructor(config = {}) {
    this.fallbackOrder = config.fallbackOrder || ['script', 'patterns', 'default'];
    this.defaultLanguage = config.defaultLanguage || 'en';
  }
  
  detect(text) {
    if (!text || text.trim().length === 0) {
      return { language: this.defaultLanguage, confidence: 0, method: 'empty' };
    }
    
    const sample = text.substring(0, 500);
    
    // Try each method in order
    for (const method of this.fallbackOrder) {
      const result = this.detectByMethod(sample, method);
      if (result.confidence > 0.5) {
        return result;
      }
    }
    
    return { language: this.defaultLanguage, confidence: 0.1, method: 'default' };
  }
  
  detectByMethod(text, method) {
    switch (method) {
      case 'script':
        return this.detectByScript(text);
      case 'patterns':
        return this.detectByPatterns(text);
      default:
        return { language: this.defaultLanguage, confidence: 0, method: 'default' };
    }
  }
  
  detectByScript(text) {
    const scripts = [
      { pattern: /[\u0600-\u06FF]/, lang: 'ar', name: 'Arabic' },
      { pattern: /[\u0590-\u05FF]/, lang: 'he', name: 'Hebrew' },
      { pattern: /[\u3040-\u309F\u30A0-\u30FF]/, lang: 'ja', name: 'Japanese' },
      { pattern: /[\u4E00-\u9FFF]/, lang: 'zh', name: 'Chinese' },
      { pattern: /[\uAC00-\uD7AF]/, lang: 'ko', name: 'Korean' },
      { pattern: /[\u0400-\u04FF]/, lang: 'ru', name: 'Cyrillic' },
      { pattern: /[\u0E00-\u0E7F]/, lang: 'th', name: 'Thai' },
      { pattern: /[\u0900-\u097F]/, lang: 'hi', name: 'Hindi' }
    ];
    
    // Count characters for each script
    let maxCount = 0;
    let detectedLang = null;
    
    for (const { pattern, lang } of scripts) {
      const matches = text.match(new RegExp(pattern.source, 'g'));
      const count = matches ? matches.length : 0;
      if (count > maxCount) {
        maxCount = count;
        detectedLang = lang;
      }
    }
    
    if (detectedLang && maxCount > 5) {
      return { language: detectedLang, confidence: 0.9, method: 'script' };
    }
    
    return { language: 'unknown', confidence: 0, method: 'script' };
  }
  
  detectByPatterns(text) {
    // Common word patterns for Latin-script languages
    const patterns = [
      { 
        lang: 'es',
        patterns: [/\b(el|la|los|las|un|una|es|son|está|están|que|de|en|con|para)\b/gi],
        weight: 0.1
      },
      {
        lang: 'fr',
        patterns: [/\b(le|la|les|un|une|est|sont|je|tu|nous|vous|ils|que|de|et|en)\b/gi],
        weight: 0.1
      },
      {
        lang: 'de',
        patterns: [/\b(der|die|das|ein|eine|ist|sind|ich|du|wir|sie|und|oder|mit|für)\b/gi],
        weight: 0.1
      },
      {
        lang: 'en',
        patterns: [/\b(the|a|an|is|are|was|were|have|has|will|would|can|could|this|that)\b/gi],
        weight: 0.1
      }
    ];
    
    let bestMatch = { lang: 'unknown', score: 0 };
    
    for (const { lang, patterns: langPatterns, weight } of patterns) {
      let score = 0;
      for (const pattern of langPatterns) {
        const matches = text.match(pattern);
        score += (matches ? matches.length : 0) * weight;
      }
      
      if (score > bestMatch.score) {
        bestMatch = { lang, score };
      }
    }
    
    if (bestMatch.score > 2) {
      return { 
        language: bestMatch.lang, 
        confidence: Math.min(0.8, bestMatch.score / 10),
        method: 'patterns'
      };
    }
    
    return { language: 'unknown', confidence: 0, method: 'patterns' };
  }
}
```

---

## Handling Language Mismatches

When the AI responds in the wrong language:

### Retry Strategy

```javascript
class LanguageEnforcedChatClient {
  constructor(config) {
    this.baseClient = new MultilingualChatClient(config);
    this.detector = new LanguageDetectionService();
    this.maxRetries = config.maxRetries || 2;
  }
  
  async sendMessage(userMessage, options = {}) {
    const expectedLanguage = options.language || await this.baseClient.languagePrefs.get();
    let lastResponse = null;
    
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const response = await this.baseClient.sendMessage(userMessage, options.history);
      const detection = this.detector.detect(response.content);
      
      lastResponse = {
        ...response,
        detectedLanguage: detection.language,
        detectionConfidence: detection.confidence,
        attempt: attempt + 1
      };
      
      // Check if language matches
      if (this.languagesMatch(expectedLanguage, detection.language)) {
        return lastResponse;
      }
      
      // Language mismatch - enhance prompt for retry
      if (attempt < this.maxRetries) {
        console.warn(`Language mismatch (attempt ${attempt + 1}): expected ${expectedLanguage}, got ${detection.language}`);
        
        // Add stronger language instruction
        options.history = [
          ...options.history || [],
          { role: 'user', content: userMessage },
          { role: 'assistant', content: response.content },
          { 
            role: 'user', 
            content: `Please respond in ${this.getLanguageName(expectedLanguage)} only. Translate your previous response.`
          }
        ];
      }
    }
    
    // Return last response with mismatch flag
    lastResponse.languageMismatch = true;
    return lastResponse;
  }
  
  languagesMatch(expected, detected) {
    if (expected === detected) return true;
    
    // Handle variants (zh === zh-Hans)
    const expectedBase = expected.split('-')[0];
    const detectedBase = detected.split('-')[0];
    
    return expectedBase === detectedBase;
  }
  
  getLanguageName(code) {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(code);
  }
}
```

### Fallback Display Strategy

```javascript
function renderMessageWithLanguageFallback(response, i18n) {
  const { content, expectedLanguage, detectedLanguage, languageMismatch } = response;
  
  if (!languageMismatch) {
    return `<div class="message">${escapeHtml(content)}</div>`;
  }
  
  // Show message with language indicator
  const expectedName = new Intl.DisplayNames([i18n.locale], { type: 'language' })
    .of(expectedLanguage);
  const detectedName = new Intl.DisplayNames([i18n.locale], { type: 'language' })
    .of(detectedLanguage);
  
  return `
    <div class="message message--language-mismatch">
      <div class="message-content">${escapeHtml(content)}</div>
      <div class="message-language-notice">
        <span class="notice-icon">⚠️</span>
        ${i18n.t('chat.language_mismatch', { 
          expected: expectedName, 
          detected: detectedName 
        })}
        <button onclick="requestTranslation()" class="translate-btn">
          ${i18n.t('chat.translate')}
        </button>
      </div>
    </div>
  `;
}
```

---

## Mixed-Language Conversations

Real conversations often include multiple languages.

### Scenarios

| Scenario | Example | Handling |
|----------|---------|----------|
| Code-switching | User mixes Spanish and English | Maintain response in preferred language |
| Technical terms | "How do I use `useState` in React?" | Keep technical terms, explain in preferred language |
| Quoted content | "What does 'Carpe diem' mean?" | Quote original, explain in preferred language |
| Language request | "Translate this to French" | Respond in French as requested |
| Explicit switch | "Let's continue in German" | Switch response language |

### Detection and Handling

```javascript
class MixedLanguageHandler {
  constructor(config) {
    this.preferredLanguage = config.preferredLanguage || 'en';
    this.detector = new LanguageDetectionService();
  }
  
  analyzeUserMessage(message) {
    // Check for explicit language switch requests
    const switchRequest = this.detectLanguageSwitchRequest(message);
    if (switchRequest) {
      return {
        type: 'language_switch',
        targetLanguage: switchRequest,
        respondIn: switchRequest
      };
    }
    
    // Check for translation requests
    const translationRequest = this.detectTranslationRequest(message);
    if (translationRequest) {
      return {
        type: 'translation',
        targetLanguage: translationRequest.targetLang,
        respondIn: translationRequest.targetLang
      };
    }
    
    // Detect primary message language
    const detection = this.detector.detect(message);
    
    return {
      type: 'standard',
      userLanguage: detection.language,
      respondIn: this.preferredLanguage
    };
  }
  
  detectLanguageSwitchRequest(message) {
    const patterns = [
      /(?:let's|can we|please)\s+(?:continue|speak|talk|chat)\s+in\s+(\w+)/i,
      /(?:switch|change)\s+(?:to|language to)\s+(\w+)/i,
      /respond\s+in\s+(\w+)/i,
      /use\s+(\w+)\s+(?:language|from now)/i
    ];
    
    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match) {
        return this.normalizeLanguageName(match[1]);
      }
    }
    
    return null;
  }
  
  detectTranslationRequest(message) {
    const patterns = [
      /translate\s+(?:this|it)?\s*(?:to|into)\s+(\w+)/i,
      /(?:in|to)\s+(\w+)\s*(?:please|:)/i,
      /how\s+(?:do you|to)\s+say\s+.+\s+in\s+(\w+)/i
    ];
    
    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match) {
        const targetLang = this.normalizeLanguageName(match[1]);
        if (targetLang) {
          return { targetLang };
        }
      }
    }
    
    return null;
  }
  
  normalizeLanguageName(name) {
    const nameToCode = {
      'english': 'en',
      'spanish': 'es',
      'french': 'fr',
      'german': 'de',
      'arabic': 'ar',
      'chinese': 'zh',
      'japanese': 'ja',
      'korean': 'ko',
      'russian': 'ru',
      'portuguese': 'pt',
      'italian': 'it'
      // Add more as needed
    };
    
    const normalized = name.toLowerCase().trim();
    return nameToCode[normalized] || null;
  }
  
  buildContextualPrompt(analysis) {
    const langName = new Intl.DisplayNames(['en'], { type: 'language' })
      .of(analysis.respondIn);
    
    switch (analysis.type) {
      case 'language_switch':
        return `The user has requested to switch to ${langName}. 
From now on, respond only in ${langName}.`;
      
      case 'translation':
        return `Translate the content to ${langName} and respond in ${langName}.`;
      
      default:
        return `Respond in ${langName}. 
If the user writes in a different language, still respond in ${langName}.
Technical terms may be kept in their original language.`;
    }
  }
}
```

---

## Complete Multilingual Chat Service

```javascript
class MultilingualChatService {
  constructor(config) {
    this.apiClient = config.apiClient;
    this.languagePrefs = new LanguagePreferenceManager(config.languageOptions);
    this.detector = new LanguageDetectionService();
    this.mixedLangHandler = new MixedLanguageHandler({
      preferredLanguage: config.defaultLanguage
    });
    
    this.conversationLanguage = null;
    this.history = [];
    this.maxRetries = config.maxRetries || 1;
  }
  
  async initialize() {
    this.conversationLanguage = await this.languagePrefs.get();
    this.mixedLangHandler.preferredLanguage = this.conversationLanguage;
  }
  
  async sendMessage(userMessage) {
    // Analyze the user's message
    const analysis = this.mixedLangHandler.analyzeUserMessage(userMessage);
    
    // Handle language switch
    if (analysis.type === 'language_switch') {
      this.conversationLanguage = analysis.targetLanguage;
      await this.languagePrefs.set(analysis.targetLanguage);
    }
    
    const responseLanguage = analysis.respondIn;
    
    // Build system prompt with language instruction
    const systemPrompt = this.buildSystemPrompt(responseLanguage, analysis);
    
    // Add to history
    this.history.push({ role: 'user', content: userMessage });
    
    // Send to API with retries
    let response = await this.sendWithRetry(systemPrompt, responseLanguage);
    
    // Add response to history
    this.history.push({ role: 'assistant', content: response.content });
    
    return {
      content: response.content,
      language: responseLanguage,
      languageVerified: response.languageVerified,
      analysis: analysis
    };
  }
  
  async sendWithRetry(systemPrompt, expectedLanguage) {
    let lastContent = null;
    
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const messages = [
        { role: 'system', content: systemPrompt },
        ...this.history
      ];
      
      const response = await this.apiClient.complete({ messages });
      lastContent = response.content;
      
      // Verify language
      const detection = this.detector.detect(lastContent);
      const matches = this.languagesMatch(expectedLanguage, detection.language);
      
      if (matches || detection.confidence < 0.5) {
        return { content: lastContent, languageVerified: matches };
      }
      
      // Retry with stronger instruction
      if (attempt < this.maxRetries) {
        this.history.push(
          { role: 'assistant', content: lastContent },
          { 
            role: 'user', 
            content: `Please respond in ${this.getLanguageName(expectedLanguage)} only.`
          }
        );
      }
    }
    
    return { content: lastContent, languageVerified: false };
  }
  
  buildSystemPrompt(language, analysis) {
    const langName = this.getLanguageName(language);
    
    let prompt = `You are a helpful AI assistant.

LANGUAGE: Always respond in ${langName} (${language}).
`;
    
    if (analysis.type === 'translation') {
      prompt += `\nThe user is requesting a translation. Provide the translation in ${langName}.\n`;
    } else if (analysis.type === 'language_switch') {
      prompt += `\nThe user has just switched to ${langName}. Acknowledge and continue in ${langName}.\n`;
    }
    
    prompt += `
Guidelines:
- Keep technical terms (code, API names) in their original form
- Explain technical concepts in ${langName}
- If unsure about a translation, provide the original term with explanation
- Be consistent with the language throughout your response`;
    
    return prompt;
  }
  
  languagesMatch(expected, detected) {
    if (expected === detected) return true;
    return expected.split('-')[0] === detected.split('-')[0];
  }
  
  getLanguageName(code) {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(code);
  }
  
  async setLanguage(language) {
    this.conversationLanguage = language;
    this.mixedLangHandler.preferredLanguage = language;
    await this.languagePrefs.set(language);
  }
  
  getConversationLanguage() {
    return this.conversationLanguage;
  }
  
  clearHistory() {
    this.history = [];
  }
}

// Usage
const chat = new MultilingualChatService({
  apiClient: openaiClient,
  languageOptions: {
    supportedLanguages: ['en', 'es', 'fr', 'de', 'ar', 'zh', 'ja'],
    defaultLanguage: 'en'
  },
  maxRetries: 1
});

await chat.initialize();

// Regular message
const response = await chat.sendMessage("What is machine learning?");
console.log(response.content);  // Response in user's preferred language

// Language switch
const switched = await chat.sendMessage("Let's continue in Spanish");
console.log(switched.language);  // "es"

// Subsequent messages in Spanish
const spanishResponse = await chat.sendMessage("¿Qué es el aprendizaje automático?");
console.log(spanishResponse.language);  // "es"
```

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Trusting LLM to follow language instructions | Verify response language and retry if needed |
| Hardcoding language in prompts | Make language configurable per-user |
| Translating technical terms | Keep code/API names, explain in target language |
| Ignoring language detection confidence | Use confidence thresholds for decisions |
| Resetting language on each request | Persist language preference in session/storage |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Store user language preference | Consistent experience across sessions |
| Include language in system prompt | Most reliable way to control LLM output language |
| Verify response language | LLMs don't always follow instructions |
| Handle mixed-language gracefully | Real conversations are multilingual |
| Provide language switcher | Let users change preference easily |
| Log language mismatches | Monitor and improve over time |

---

## Hands-on Exercise

### Your Task

Build a language preference component for a chat interface.

### Requirements

1. Detect user's browser language on first visit
2. Allow user to select from supported languages
3. Persist selection in localStorage
4. Display languages in their native names
5. Emit an event when language changes

### Expected Result

```javascript
const langSelector = new LanguageSelector({
  supported: ['en', 'es', 'fr', 'de', 'ar', 'zh'],
  default: 'en'
});

// Auto-detects and sets initial language
await langSelector.init();

// Gets available languages with native names
langSelector.getOptions();
// [
//   { code: 'en', name: 'English' },
//   { code: 'es', name: 'Español' },
//   { code: 'ar', name: 'العربية' },
//   ...
// ]

// Change language
langSelector.setLanguage('es');
// Emits 'language-changed' event
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `navigator.languages` for browser language detection
- Use `Intl.DisplayNames` with the language code itself to get native name
- Use `CustomEvent` for the change notification
- Check if selected language is in supported list

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
class LanguageSelector extends EventTarget {
  constructor(config) {
    super();
    this.supported = config.supported || ['en'];
    this.defaultLang = config.default || 'en';
    this.storageKey = 'user_language';
    this.current = null;
  }
  
  async init() {
    // Try localStorage
    const stored = localStorage.getItem(this.storageKey);
    if (stored && this.supported.includes(stored)) {
      this.current = stored;
      return this.current;
    }
    
    // Try browser language
    const browserLangs = navigator.languages || [navigator.language];
    for (const lang of browserLangs) {
      const base = lang.split('-')[0];
      if (this.supported.includes(base)) {
        this.current = base;
        break;
      }
    }
    
    // Fallback
    if (!this.current) {
      this.current = this.defaultLang;
    }
    
    localStorage.setItem(this.storageKey, this.current);
    return this.current;
  }
  
  getOptions() {
    return this.supported.map(code => ({
      code,
      name: new Intl.DisplayNames([code], { type: 'language' }).of(code)
    }));
  }
  
  getCurrentLanguage() {
    return this.current;
  }
  
  setLanguage(code) {
    if (!this.supported.includes(code)) {
      console.warn(`Unsupported language: ${code}`);
      return false;
    }
    
    const previous = this.current;
    this.current = code;
    localStorage.setItem(this.storageKey, code);
    
    this.dispatchEvent(new CustomEvent('language-changed', {
      detail: { previous, current: code }
    }));
    
    return true;
  }
}

// Usage
const selector = new LanguageSelector({
  supported: ['en', 'es', 'fr', 'de', 'ar', 'zh'],
  default: 'en'
});

selector.addEventListener('language-changed', (e) => {
  console.log(`Language changed from ${e.detail.previous} to ${e.detail.current}`);
});

await selector.init();
console.log(selector.getOptions());
selector.setLanguage('es');
```

</details>

---

## Summary

✅ Store user language preferences in user profile, localStorage, or session

✅ Include explicit language instructions in system prompts for LLMs

✅ Verify AI response language and retry if there's a mismatch

✅ Handle language switch requests and translation requests explicitly

✅ Keep technical terms in original form while explaining in target language

**Next:** [Translation Integration Patterns](./06-translation-integration.md)

---

## Further Reading

- [OpenAI: Multilingual Capabilities](https://platform.openai.com/docs/guides/text-generation) - Working with multiple languages
- [MDN: Intl.DisplayNames](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DisplayNames) - Language name display
- [Unicode CLDR Language Data](https://cldr.unicode.org/) - Locale data source

<!--
Sources Consulted:
- OpenAI API Documentation: https://platform.openai.com/docs/
- MDN Intl.DisplayNames: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DisplayNames
- Google Cloud Translation API: https://cloud.google.com/translate/docs
-->
