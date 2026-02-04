---
title: "Character & Token Counters"
---

# Character & Token Counters

## Introduction

Users need to know how much content they can send and how it affects API costs. Character counters are straightforward, but token counters—the actual unit LLMs use—require estimation. Showing both helps users understand their input's impact.

In this lesson, we'll implement counters that display characters, estimate tokens, and warn users as they approach limits.

### What We'll Cover

- Character counting and display
- Token estimation strategies
- Visual counter components
- Limit warnings at thresholds
- Cost preview (optional)
- Model-specific token limits

### Prerequisites

- [Input Validation](./05-input-validation.md)
- Understanding of LLM tokenization
- Basic math for estimation

---

## Character Counting

```tsx
interface CharacterCounterProps {
  value: string;
  maxLength: number;
  warningThreshold?: number;
}

function CharacterCounter({ 
  value, 
  maxLength, 
  warningThreshold = 0.9 
}: CharacterCounterProps) {
  const count = value.length;
  const percentage = count / maxLength;
  
  const getColorClass = () => {
    if (count > maxLength) return 'text-red-500 font-semibold';
    if (percentage >= warningThreshold) return 'text-yellow-600';
    return 'text-gray-400';
  };
  
  return (
    <span 
      className={`text-xs tabular-nums ${getColorClass()}`}
      aria-label={`${count} of ${maxLength} characters`}
    >
      {count.toLocaleString()} / {maxLength.toLocaleString()}
    </span>
  );
}
```

---

## Token Estimation

### Simple Estimation (chars ÷ 4)

For English text, a rough estimate is **1 token ≈ 4 characters**:

```tsx
function estimateTokens(text: string): number {
  // Simple estimation: ~4 characters per token for English
  return Math.ceil(text.length / 4);
}
```

### Better Estimation with Word Counting

```tsx
function estimateTokensAdvanced(text: string): number {
  if (!text.trim()) return 0;
  
  // Count different elements
  const words = text.trim().split(/\s+/).length;
  const punctuation = (text.match(/[.,!?;:'"()\[\]{}]/g) || []).length;
  const numbers = (text.match(/\d+/g) || []).length;
  const whitespace = (text.match(/\s/g) || []).length;
  
  // Rough estimation formula
  // - Words are typically 1-2 tokens
  // - Punctuation is usually 1 token each
  // - Numbers vary based on length
  const wordTokens = words * 1.3; // Average 1.3 tokens per word
  const punctTokens = punctuation * 0.5; // Punctuation often merges
  const numberTokens = numbers * 1.5; // Numbers can be multiple tokens
  
  return Math.ceil(wordTokens + punctTokens + numberTokens);
}
```

### Using a Tokenizer Library

For accurate counting, use a tokenizer like `gpt-tokenizer`:

```tsx
// npm install gpt-tokenizer
import { encode } from 'gpt-tokenizer';

function countTokensAccurate(text: string): number {
  const tokens = encode(text);
  return tokens.length;
}

// With caching for performance
function useTokenCount(text: string) {
  const [count, setCount] = useState(0);
  
  // Debounce for performance
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        const tokens = encode(text);
        setCount(tokens.length);
      } catch {
        // Fallback to estimation
        setCount(Math.ceil(text.length / 4));
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }, [text]);
  
  return count;
}
```

---

## Token Counter Component

```tsx
interface TokenCounterProps {
  value: string;
  maxTokens: number;
  estimationMethod?: 'simple' | 'advanced' | 'accurate';
}

function TokenCounter({ 
  value, 
  maxTokens,
  estimationMethod = 'simple'
}: TokenCounterProps) {
  const [tokens, setTokens] = useState(0);
  
  // Calculate tokens based on method
  useEffect(() => {
    let count: number;
    
    switch (estimationMethod) {
      case 'accurate':
        // Would use gpt-tokenizer here
        count = Math.ceil(value.length / 3.5); // Closer estimate
        break;
      case 'advanced':
        count = estimateTokensAdvanced(value);
        break;
      case 'simple':
      default:
        count = Math.ceil(value.length / 4);
    }
    
    setTokens(count);
  }, [value, estimationMethod]);
  
  const percentage = tokens / maxTokens;
  const isNearLimit = percentage >= 0.9;
  const isOverLimit = tokens > maxTokens;
  
  return (
    <div className="flex items-center gap-1 text-xs">
      <span className={`
        tabular-nums
        ${isOverLimit ? 'text-red-500 font-semibold' : 
          isNearLimit ? 'text-yellow-600' : 'text-gray-400'}
      `}>
        ~{tokens.toLocaleString()} tokens
      </span>
      
      {isOverLimit && (
        <span className="text-red-500">
          (exceeds {maxTokens.toLocaleString()} limit)
        </span>
      )}
    </div>
  );
}
```

---

## Combined Counter Component

```tsx
interface InputCountersProps {
  value: string;
  maxChars?: number;
  maxTokens?: number;
  showTokens?: boolean;
  showCost?: boolean;
  pricePerMillionTokens?: number;
}

function InputCounters({
  value,
  maxChars = 4000,
  maxTokens = 4096,
  showTokens = true,
  showCost = false,
  pricePerMillionTokens = 3.0 // Example: GPT-4 input price
}: InputCountersProps) {
  const charCount = value.length;
  const tokenEstimate = Math.ceil(charCount / 4);
  
  const charPercentage = charCount / maxChars;
  const tokenPercentage = tokenEstimate / maxTokens;
  
  // Determine overall status
  const isOverLimit = charCount > maxChars || tokenEstimate > maxTokens;
  const isNearLimit = charPercentage >= 0.9 || tokenPercentage >= 0.9;
  
  const statusColor = isOverLimit 
    ? 'text-red-500' 
    : isNearLimit 
    ? 'text-yellow-600' 
    : 'text-gray-400';
  
  // Cost calculation
  const cost = showCost 
    ? (tokenEstimate / 1_000_000) * pricePerMillionTokens 
    : 0;
  
  return (
    <div className={`flex items-center gap-3 text-xs ${statusColor}`}>
      {/* Character count */}
      <span className="tabular-nums">
        {charCount.toLocaleString()} / {maxChars.toLocaleString()} chars
      </span>
      
      {/* Token count */}
      {showTokens && (
        <>
          <span className="text-gray-300">|</span>
          <span className="tabular-nums">
            ~{tokenEstimate.toLocaleString()} tokens
          </span>
        </>
      )}
      
      {/* Cost estimate */}
      {showCost && cost > 0 && (
        <>
          <span className="text-gray-300">|</span>
          <span className="tabular-nums">
            ~${cost.toFixed(4)}
          </span>
        </>
      )}
    </div>
  );
}
```

---

## Progress Bar Counter

```tsx
interface ProgressCounterProps {
  value: string;
  maxChars: number;
}

function ProgressCounter({ value, maxChars }: ProgressCounterProps) {
  const percentage = Math.min((value.length / maxChars) * 100, 100);
  
  const getBarColor = () => {
    if (value.length > maxChars) return 'bg-red-500';
    if (percentage >= 90) return 'bg-yellow-500';
    if (percentage >= 75) return 'bg-blue-500';
    return 'bg-green-500';
  };
  
  return (
    <div className="space-y-1">
      {/* Progress bar */}
      <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-200 ${getBarColor()}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      {/* Text counter */}
      <div className="flex justify-between text-xs text-gray-500">
        <span>{value.length.toLocaleString()} characters</span>
        <span>{maxChars.toLocaleString()} max</span>
      </div>
    </div>
  );
}
```

---

## Model-Specific Limits

```tsx
interface ModelLimits {
  name: string;
  maxInputTokens: number;
  maxOutputTokens: number;
  pricePerMillionInput: number;
  pricePerMillionOutput: number;
}

const MODEL_LIMITS: Record<string, ModelLimits> = {
  'gpt-4o': {
    name: 'GPT-4o',
    maxInputTokens: 128000,
    maxOutputTokens: 16384,
    pricePerMillionInput: 2.5,
    pricePerMillionOutput: 10.0
  },
  'gpt-4o-mini': {
    name: 'GPT-4o Mini',
    maxInputTokens: 128000,
    maxOutputTokens: 16384,
    pricePerMillionInput: 0.15,
    pricePerMillionOutput: 0.6
  },
  'claude-3-5-sonnet': {
    name: 'Claude 3.5 Sonnet',
    maxInputTokens: 200000,
    maxOutputTokens: 8192,
    pricePerMillionInput: 3.0,
    pricePerMillionOutput: 15.0
  }
};

function useModelLimits(modelId: string) {
  return MODEL_LIMITS[modelId] || MODEL_LIMITS['gpt-4o-mini'];
}
```

### Model-Aware Counter

```tsx
function ModelAwareCounter({ 
  value, 
  modelId 
}: { 
  value: string; 
  modelId: string;
}) {
  const limits = useModelLimits(modelId);
  const tokenEstimate = Math.ceil(value.length / 4);
  
  const percentage = (tokenEstimate / limits.maxInputTokens) * 100;
  const cost = (tokenEstimate / 1_000_000) * limits.pricePerMillionInput;
  
  return (
    <div className="text-xs text-gray-500">
      <span className="font-medium">{limits.name}</span>
      <span className="mx-2">•</span>
      <span className="tabular-nums">
        ~{tokenEstimate.toLocaleString()} / {limits.maxInputTokens.toLocaleString()} tokens
        ({percentage.toFixed(1)}%)
      </span>
      <span className="mx-2">•</span>
      <span className="tabular-nums">
        ~${cost.toFixed(4)}
      </span>
    </div>
  );
}
```

---

## Complete Input with Counters

```tsx
interface ChatInputWithCountersProps {
  onSubmit: (message: string) => void;
  modelId?: string;
  showCost?: boolean;
}

function ChatInputWithCounters({
  onSubmit,
  modelId = 'gpt-4o-mini',
  showCost = false
}: ChatInputWithCountersProps) {
  const [input, setInput] = useState('');
  const limits = useModelLimits(modelId);
  
  const charCount = input.length;
  const tokenEstimate = Math.ceil(charCount / 4);
  
  const isOverLimit = tokenEstimate > limits.maxInputTokens;
  const isNearLimit = tokenEstimate > limits.maxInputTokens * 0.9;
  
  const handleSubmit = () => {
    if (input.trim() && !isOverLimit) {
      onSubmit(input.trim());
      setInput('');
    }
  };
  
  return (
    <div className="space-y-2">
      <div className="relative">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          placeholder="Type your message..."
          rows={3}
          className={`
            w-full p-4 rounded-xl border-2
            resize-none
            ${isOverLimit 
              ? 'border-red-300 focus:border-red-500' 
              : 'border-gray-200 focus:border-blue-500'
            }
            focus:outline-none focus:ring-2
            ${isOverLimit ? 'focus:ring-red-200' : 'focus:ring-blue-200'}
          `}
        />
      </div>
      
      {/* Counters row */}
      <div className="flex items-center justify-between">
        <InputCounters
          value={input}
          maxChars={limits.maxInputTokens * 4} // Approx char limit
          maxTokens={limits.maxInputTokens}
          showTokens={true}
          showCost={showCost}
          pricePerMillionTokens={limits.pricePerMillionInput}
        />
        
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || isOverLimit}
          className="
            px-4 py-2 rounded-lg
            bg-blue-500 text-white text-sm font-medium
            disabled:bg-gray-200 disabled:text-gray-400
            hover:bg-blue-600
            transition-colors
          "
        >
          Send
        </button>
      </div>
      
      {/* Warning message */}
      {isOverLimit && (
        <p className="text-sm text-red-500">
          Message exceeds {limits.maxInputTokens.toLocaleString()} token limit for {limits.name}
        </p>
      )}
      
      {!isOverLimit && isNearLimit && (
        <p className="text-sm text-yellow-600">
          Approaching token limit ({((tokenEstimate / limits.maxInputTokens) * 100).toFixed(0)}%)
        </p>
      )}
    </div>
  );
}
```

---

## Token Estimation Accuracy

| Method | Accuracy | Performance | Use Case |
|--------|----------|-------------|----------|
| `chars / 4` | ~70% | Instant | Quick estimates |
| Word-based | ~80% | Fast | Better UX feedback |
| gpt-tokenizer | ~99% | Slower | Precise limits |

```tsx
// Choose based on needs
function getTokenizer(precision: 'fast' | 'accurate') {
  if (precision === 'accurate') {
    // Lazy load for performance
    return import('gpt-tokenizer').then(m => m.encode);
  }
  return (text: string) => Math.ceil(text.length / 4);
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show `~` for estimates | Claim exact token counts |
| Use `tabular-nums` for alignment | Let numbers jump around |
| Warn at 90% threshold | Wait until 100% |
| Debounce accurate tokenization | Tokenize on every keystroke |
| Show model-specific limits | Use generic limits |
| Make cost optional | Show cost by default (may concern users) |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Blocking on estimated limits | Use as warning, allow slight overflow |
| No visual change near limit | Color changes at threshold |
| Slow UI with large inputs | Debounce token calculation |
| Confusing chars with tokens | Label clearly |
| Stale model limits | Keep limits updated |

---

## Hands-on Exercise

### Your Task

Build a counter component that:
1. Shows character count with limit
2. Shows estimated token count
3. Changes color at 90% and 100%
4. Displays a progress bar
5. Optionally shows cost estimate

### Requirements

1. Support multiple models (GPT-4o, GPT-4o-mini)
2. Use `~` prefix for token estimates
3. Warning state at 90%
4. Error state over 100%
5. Accessible `aria-label`

<details>
<summary>💡 Hints (click to expand)</summary>

- Create a `MODEL_LIMITS` object with pricing
- Use `chars / 4` for quick estimation
- Add `tabular-nums` class for number alignment
- Calculate percentage for progress bar width
- Format cost with `toFixed(4)` for precision

</details>

---

## Summary

✅ **Character counting** is straightforward  
✅ **Token estimation** uses ~4 chars/token rule  
✅ **Model-specific limits** vary significantly  
✅ **Warning at 90%** gives users time to edit  
✅ **Cost preview** helps with budget awareness  
✅ **Debounce** expensive calculations

---

## Further Reading

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [gpt-tokenizer npm](https://www.npmjs.com/package/gpt-tokenizer)
- [Anthropic Pricing](https://www.anthropic.com/pricing)

---

**Previous:** [Input Validation](./05-input-validation.md)  
**Next:** [File Attachment UI](./07-file-attachment-ui.md)

<!-- 
Sources Consulted:
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- OpenAI Pricing: https://openai.com/pricing
- Anthropic Pricing: https://anthropic.com/pricing
-->
