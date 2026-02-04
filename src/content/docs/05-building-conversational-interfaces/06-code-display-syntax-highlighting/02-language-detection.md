---
title: "Language Detection"
---

# Language Detection

## Introduction

When users paste code or AI generates examples, the language tag isn't always explicit. Auto-detection enables highlighting even for untagged blocks, while proper fallbacks ensure graceful degradation when detection fails.

In this lesson, we'll implement intelligent language detection for code blocks.

### What We'll Cover

- Explicit language tag handling
- Auto-detection libraries and techniques
- Heuristic-based detection
- Fallbacks for unknown languages
- Performance considerations

### Prerequisites

- [Detecting Code Blocks](./01-detecting-code-blocks.md)
- Regular expressions
- Common programming language syntax

---

## Explicit Language Tags

### Priority System

```javascript
function detectLanguage(code, options = {}) {
  const { explicitLang, filename, contentType } = options;
  
  // 1. Explicit tag takes priority
  if (explicitLang && explicitLang !== 'text') {
    return normalizeLanguage(explicitLang);
  }
  
  // 2. Filename extension
  if (filename) {
    const fromFilename = languageFromFilename(filename);
    if (fromFilename) return fromFilename;
  }
  
  // 3. Content-Type header (for API responses)
  if (contentType) {
    const fromContentType = languageFromContentType(contentType);
    if (fromContentType) return fromContentType;
  }
  
  // 4. Auto-detect from content
  return autoDetectLanguage(code);
}
```

### Filename to Language Map

```javascript
const EXTENSION_MAP = {
  // JavaScript ecosystem
  '.js': 'javascript',
  '.mjs': 'javascript',
  '.cjs': 'javascript',
  '.jsx': 'jsx',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  
  // Python
  '.py': 'python',
  '.pyw': 'python',
  '.pyi': 'python',
  
  // Web
  '.html': 'html',
  '.htm': 'html',
  '.css': 'css',
  '.scss': 'scss',
  '.sass': 'sass',
  '.less': 'less',
  
  // Data formats
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.xml': 'xml',
  '.toml': 'toml',
  
  // Shell
  '.sh': 'bash',
  '.bash': 'bash',
  '.zsh': 'zsh',
  '.fish': 'fish',
  '.ps1': 'powershell',
  
  // Systems
  '.c': 'c',
  '.h': 'c',
  '.cpp': 'cpp',
  '.cc': 'cpp',
  '.hpp': 'cpp',
  '.rs': 'rust',
  '.go': 'go',
  '.java': 'java',
  '.kt': 'kotlin',
  '.swift': 'swift',
  
  // Other
  '.rb': 'ruby',
  '.php': 'php',
  '.sql': 'sql',
  '.md': 'markdown',
  '.r': 'r',
  '.R': 'r'
};

function languageFromFilename(filename) {
  const ext = '.' + filename.split('.').pop().toLowerCase();
  return EXTENSION_MAP[ext] || null;
}
```

---

## Auto-Detection Libraries

### highlight.js Auto-Detection

```javascript
import hljs from 'highlight.js';

function autoDetectWithHljs(code) {
  const result = hljs.highlightAuto(code);
  
  return {
    language: result.language || 'text',
    confidence: result.relevance,
    secondBest: result.secondBest?.language
  };
}
```

**Pros:**
- Built into highlight.js
- Good accuracy for common languages
- Returns confidence score

**Cons:**
- Larger bundle if loading all languages
- Can be slow for large code

### Subset Detection

Limit detection to likely languages for performance:

```javascript
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import python from 'highlight.js/lib/languages/python';
import typescript from 'highlight.js/lib/languages/typescript';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import css from 'highlight.js/lib/languages/css';
import html from 'highlight.js/lib/languages/xml';

// Register only common languages
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('json', json);
hljs.registerLanguage('css', css);
hljs.registerLanguage('html', html);

function autoDetectSubset(code) {
  const result = hljs.highlightAuto(code, [
    'javascript', 'python', 'typescript', 
    'bash', 'json', 'css', 'html'
  ]);
  
  return result.language || 'text';
}
```

---

## Heuristic-Based Detection

### Pattern Matching

```javascript
const LANGUAGE_PATTERNS = [
  // JavaScript/TypeScript
  {
    language: 'javascript',
    patterns: [
      /\bconst\s+\w+\s*=/,
      /\blet\s+\w+\s*=/,
      /\bfunction\s+\w+\s*\(/,
      /=>\s*{/,
      /\bconsole\.(log|error|warn)\(/,
      /\brequire\s*\(/,
      /\bimport\s+.*\s+from\s+['"]/
    ],
    weight: 1
  },
  {
    language: 'typescript',
    patterns: [
      /:\s*(string|number|boolean|any)\b/,
      /interface\s+\w+\s*{/,
      /type\s+\w+\s*=/,
      /<\w+>/,  // Generic type
      /as\s+(string|number|boolean|const)\b/
    ],
    weight: 1.5  // Higher weight if TypeScript-specific
  },
  
  // Python
  {
    language: 'python',
    patterns: [
      /\bdef\s+\w+\s*\(/,
      /\bclass\s+\w+.*:/,
      /\bimport\s+\w+/,
      /\bfrom\s+\w+\s+import/,
      /\bif\s+.*:\s*$/m,
      /\bprint\s*\(/,
      /\bself\./
    ],
    weight: 1
  },
  
  // HTML
  {
    language: 'html',
    patterns: [
      /<!DOCTYPE\s+html/i,
      /<html/i,
      /<head>/i,
      /<body>/i,
      /<div[>\s]/i,
      /<\/\w+>/
    ],
    weight: 1
  },
  
  // CSS
  {
    language: 'css',
    patterns: [
      /{\s*[\w-]+\s*:\s*[^}]+}/,
      /\.([\w-]+)\s*{/,
      /#[\w-]+\s*{/,
      /@media\s/,
      /@import\s/
    ],
    weight: 1
  },
  
  // JSON
  {
    language: 'json',
    patterns: [
      /^\s*{[\s\S]*}$/,
      /^\s*\[[\s\S]*\]$/,
      /"[\w]+"\s*:\s*["\d\[\{]/
    ],
    weight: 0.8  // Lower because many things look like JSON
  },
  
  // Bash/Shell
  {
    language: 'bash',
    patterns: [
      /^#!/,
      /\$\{?\w+\}?/,
      /\becho\s/,
      /\bsudo\s/,
      /\bcd\s/,
      /\|\s*\w+/
    ],
    weight: 1
  },
  
  // SQL
  {
    language: 'sql',
    patterns: [
      /\bSELECT\b.*\bFROM\b/i,
      /\bINSERT\s+INTO\b/i,
      /\bUPDATE\b.*\bSET\b/i,
      /\bCREATE\s+TABLE\b/i,
      /\bWHERE\b/i
    ],
    weight: 1.2
  }
];

function heuristicDetect(code) {
  const scores = {};
  
  for (const { language, patterns, weight } of LANGUAGE_PATTERNS) {
    let matchCount = 0;
    
    for (const pattern of patterns) {
      if (pattern.test(code)) {
        matchCount++;
      }
    }
    
    if (matchCount > 0) {
      scores[language] = (matchCount / patterns.length) * weight;
    }
  }
  
  // Find highest scoring language
  const sorted = Object.entries(scores)
    .sort(([, a], [, b]) => b - a);
  
  if (sorted.length === 0) {
    return { language: 'text', confidence: 0 };
  }
  
  return {
    language: sorted[0][0],
    confidence: sorted[0][1],
    alternatives: sorted.slice(1, 3).map(([lang]) => lang)
  };
}
```

### Special Case Detection

```javascript
function detectSpecialCases(code) {
  const trimmed = code.trim();
  
  // JSON (try parsing)
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
      (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      JSON.parse(trimmed);
      return 'json';
    } catch {
      // Not valid JSON
    }
  }
  
  // Shebang
  if (trimmed.startsWith('#!')) {
    if (trimmed.includes('python')) return 'python';
    if (trimmed.includes('node')) return 'javascript';
    if (trimmed.includes('bash') || trimmed.includes('sh')) return 'bash';
    if (trimmed.includes('ruby')) return 'ruby';
    if (trimmed.includes('perl')) return 'perl';
  }
  
  // Doctype
  if (/^<!DOCTYPE\s+html/i.test(trimmed)) {
    return 'html';
  }
  
  // XML declaration
  if (trimmed.startsWith('<?xml')) {
    return 'xml';
  }
  
  return null;
}
```

---

## Combined Detection Strategy

```javascript
class LanguageDetector {
  constructor(options = {}) {
    this.useHljs = options.useHljs !== false;
    this.hljsInstance = options.hljs || null;
    this.minConfidence = options.minConfidence || 0.3;
  }
  
  detect(code, hints = {}) {
    // 1. Explicit language
    if (hints.language && hints.language !== 'text') {
      return {
        language: normalizeLanguage(hints.language),
        source: 'explicit',
        confidence: 1
      };
    }
    
    // 2. Filename
    if (hints.filename) {
      const lang = languageFromFilename(hints.filename);
      if (lang) {
        return {
          language: lang,
          source: 'filename',
          confidence: 0.95
        };
      }
    }
    
    // 3. Special cases (shebang, doctype, etc.)
    const special = detectSpecialCases(code);
    if (special) {
      return {
        language: special,
        source: 'special',
        confidence: 0.9
      };
    }
    
    // 4. Heuristic patterns
    const heuristic = heuristicDetect(code);
    if (heuristic.confidence >= this.minConfidence) {
      return {
        language: heuristic.language,
        source: 'heuristic',
        confidence: heuristic.confidence,
        alternatives: heuristic.alternatives
      };
    }
    
    // 5. highlight.js auto-detection
    if (this.useHljs && this.hljsInstance) {
      const hlResult = this.hljsInstance.highlightAuto(code);
      if (hlResult.relevance > 5) {  // Minimum relevance threshold
        return {
          language: hlResult.language,
          source: 'hljs',
          confidence: Math.min(hlResult.relevance / 20, 1)
        };
      }
    }
    
    // 6. Fallback
    return {
      language: 'text',
      source: 'fallback',
      confidence: 0
    };
  }
}
```

---

## Fallbacks for Unknown Languages

### Plain Text Fallback

```jsx
function CodeBlock({ code, language, detectionResult }) {
  const isUnknown = language === 'text' || 
                    detectionResult?.confidence < 0.3;
  
  return (
    <div className={`code-block ${isUnknown ? 'plain' : ''}`}>
      {isUnknown && (
        <div className="language-badge unknown">
          Plain text
        </div>
      )}
      
      <pre className={isUnknown ? 'plain-text' : ''}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
```

### User Override

```jsx
function CodeBlockWithOverride({ code, detectedLanguage }) {
  const [language, setLanguage] = useState(detectedLanguage);
  const [showPicker, setShowPicker] = useState(false);
  
  return (
    <div className="code-block">
      <div className="code-header">
        <button 
          onClick={() => setShowPicker(!showPicker)}
          className="language-badge"
        >
          {language}
          <span className="edit-icon">✎</span>
        </button>
        
        {showPicker && (
          <LanguagePicker
            current={language}
            onSelect={(lang) => {
              setLanguage(lang);
              setShowPicker(false);
            }}
          />
        )}
      </div>
      
      <SyntaxHighlighter language={language}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

function LanguagePicker({ current, onSelect }) {
  const popular = [
    'javascript', 'typescript', 'python', 'java',
    'go', 'rust', 'bash', 'json', 'html', 'css'
  ];
  
  return (
    <div className="language-picker">
      {popular.map(lang => (
        <button
          key={lang}
          onClick={() => onSelect(lang)}
          className={lang === current ? 'selected' : ''}
        >
          {lang}
        </button>
      ))}
    </div>
  );
}
```

---

## Performance Optimization

### Lazy Detection

```javascript
function useLazyLanguageDetection(code) {
  const [language, setLanguage] = useState('text');
  const [isDetecting, setIsDetecting] = useState(true);
  
  useEffect(() => {
    // Quick heuristic first
    const quick = heuristicDetect(code);
    if (quick.confidence > 0.5) {
      setLanguage(quick.language);
      setIsDetecting(false);
      return;
    }
    
    // Defer heavy detection
    const timer = setTimeout(() => {
      const detector = new LanguageDetector({ useHljs: true });
      const result = detector.detect(code);
      setLanguage(result.language);
      setIsDetecting(false);
    }, 50);
    
    return () => clearTimeout(timer);
  }, [code]);
  
  return { language, isDetecting };
}
```

### Caching

```javascript
const detectionCache = new Map();

function cachedDetect(code) {
  // Use code hash as cache key
  const hash = simpleHash(code.slice(0, 500));  // First 500 chars
  
  if (detectionCache.has(hash)) {
    return detectionCache.get(hash);
  }
  
  const result = detectLanguage(code);
  detectionCache.set(hash, result);
  
  // Limit cache size
  if (detectionCache.size > 100) {
    const firstKey = detectionCache.keys().next().value;
    detectionCache.delete(firstKey);
  }
  
  return result;
}

function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash = hash & hash;  // Convert to 32-bit int
  }
  return hash.toString(36);
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Prioritize explicit language | Override user's choice |
| Use multiple detection strategies | Rely on single method |
| Set confidence thresholds | Trust low-confidence results |
| Cache detection results | Re-detect on every render |
| Provide manual override | Force incorrect detection |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| TypeScript detected as JavaScript | Check for type annotations |
| JSON confused with JavaScript | Try `JSON.parse()` first |
| HTML inside JSX | Check for JSX patterns |
| Shell aliases mismatch | Normalize bash/sh/zsh |
| Large code causes lag | Limit detection to first 500 chars |

---

## Hands-on Exercise

### Your Task

Build a language detector that:
1. Uses explicit language tag when available
2. Falls back to filename extension
3. Applies heuristic patterns
4. Returns confidence score

### Requirements

1. Map common aliases (js → javascript)
2. Handle shebang lines
3. Detect at least 5 languages
4. Return alternatives for close matches

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with high-confidence indicators
- Use `test()` for pattern matching
- Weight patterns by specificity
- Normalize language names consistently

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `LanguageDetector` class in the Combined Detection Strategy section above.

</details>

---

## Summary

✅ **Explicit tags** take priority over detection  
✅ **Filename extensions** are reliable indicators  
✅ **Heuristic patterns** catch common languages  
✅ **Auto-detection libraries** provide fallback  
✅ **Confidence scores** indicate reliability  
✅ **Caching** improves performance

---

## Further Reading

- [highlight.js Language Detection](https://highlightjs.org/usage/#auto-detection)
- [Linguist Language Detection](https://github.com/github/linguist)
- [CodeMirror Language Data](https://codemirror.net/docs/ref/#language-data)

---

**Previous:** [Detecting Code Blocks](./01-detecting-code-blocks.md)  
**Next:** [Syntax Highlighting Implementation](./03-syntax-highlighting.md)

<!-- 
Sources Consulted:
- highlight.js: https://highlightjs.org/
- GitHub Linguist: https://github.com/github/linguist
- Prism.js languages: https://prismjs.com/#supported-languages
-->
