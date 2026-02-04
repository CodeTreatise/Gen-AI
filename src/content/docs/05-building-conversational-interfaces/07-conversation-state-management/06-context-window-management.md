---
title: "Context Window Management"
---

# Context Window Management

## Introduction

Large language models have finite context windows—the maximum tokens they can process in a single request. As conversations grow, you must manage context to stay within limits while preserving important information. OpenAI's compaction endpoint and modern compression strategies help maintain long conversations efficiently.

In this lesson, we'll implement context window management for production chat applications.

### What We'll Cover

- Understanding token limits
- The `/responses/compact` endpoint
- Encrypted compaction items
- Client-side compression strategies
- ZDR-compatible compression

### Prerequisites

- [Automatic Context Chaining](./05-automatic-context-chaining.md)
- Understanding of tokenization
- API rate limits and quotas

---

## Context Window Basics

### Model Token Limits

| Model | Context Window | Output Limit |
|-------|---------------|--------------|
| GPT-4o | 128K tokens | 16K tokens |
| GPT-4o-mini | 128K tokens | 16K tokens |
| GPT-4 Turbo | 128K tokens | 4K tokens |
| Claude 3.5 Sonnet | 200K tokens | 8K tokens |
| o1 | 200K tokens | 100K tokens |

### Token Estimation

```typescript
// Rough estimation (actual tokenization varies)
function estimateTokens(text: string): number {
  // Average: ~4 characters per token for English
  return Math.ceil(text.length / 4);
}

// More accurate with tiktoken
import { encoding_for_model } from 'tiktoken';

function countTokens(text: string, model = 'gpt-4o'): number {
  const encoder = encoding_for_model(model);
  const tokens = encoder.encode(text);
  encoder.free();  // Clean up
  return tokens.length;
}

// Count tokens in message array
function countConversationTokens(messages: Message[]): number {
  let total = 0;
  
  for (const msg of messages) {
    // Add tokens for role and formatting
    total += 4;  // <role>: ...
    total += countTokens(msg.content);
  }
  
  total += 2;  // Conversation overhead
  return total;
}
```

---

## Token Budget Planning

### Budget Allocation

```typescript
interface TokenBudget {
  contextLimit: number;      // Model's max context
  reserveForOutput: number;  // Space for response
  reserveForSystem: number;  // System prompt
  availableForHistory: number;
}

function calculateBudget(
  model: string,
  systemPrompt: string
): TokenBudget {
  const limits: Record<string, { context: number; output: number }> = {
    'gpt-4o': { context: 128000, output: 16000 },
    'gpt-4o-mini': { context: 128000, output: 16000 },
    'claude-3-5-sonnet': { context: 200000, output: 8000 }
  };
  
  const { context, output } = limits[model] || limits['gpt-4o'];
  const systemTokens = countTokens(systemPrompt) + 10;
  
  return {
    contextLimit: context,
    reserveForOutput: output,
    reserveForSystem: systemTokens,
    availableForHistory: context - output - systemTokens - 100  // Safety margin
  };
}
```

### Monitoring Usage

```typescript
class ContextMonitor {
  private budget: TokenBudget;
  private currentUsage: number = 0;
  
  constructor(model: string, systemPrompt: string) {
    this.budget = calculateBudget(model, systemPrompt);
  }
  
  addMessage(content: string): { added: boolean; remaining: number } {
    const tokens = countTokens(content);
    const newUsage = this.currentUsage + tokens;
    
    if (newUsage > this.budget.availableForHistory) {
      return {
        added: false,
        remaining: this.budget.availableForHistory - this.currentUsage
      };
    }
    
    this.currentUsage = newUsage;
    return {
      added: true,
      remaining: this.budget.availableForHistory - this.currentUsage
    };
  }
  
  getStatus(): {
    used: number;
    available: number;
    percentage: number;
    warning: boolean;
  } {
    const percentage = (this.currentUsage / this.budget.availableForHistory) * 100;
    
    return {
      used: this.currentUsage,
      available: this.budget.availableForHistory - this.currentUsage,
      percentage,
      warning: percentage > 80
    };
  }
}
```

---

## OpenAI Compaction Endpoint

### How Compaction Works

The `/responses/compact` endpoint compresses conversation history into an encrypted, opaque item that preserves context while reducing token usage.

```typescript
async function compactConversation(responseId: string) {
  const compaction = await openai.responses.compact(responseId);
  
  return {
    compactedItem: compaction.item,  // Encrypted representation
    originalTokens: compaction.original_tokens,
    compactedTokens: compaction.compacted_tokens,
    compressionRatio: compaction.original_tokens / compaction.compacted_tokens
  };
}
```

### Using Compacted Context

```typescript
async function chatWithCompactedContext(
  message: string,
  compactedItem: string
) {
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: message,
    context: [
      { type: 'compacted', item: compactedItem }
    ]
  });
  
  return response.output_text;
}
```

### Automatic Compaction Strategy

```typescript
class AutoCompactingChat {
  private responseHistory: string[] = [];
  private compactedContext: string | null = null;
  private compactionThreshold = 50000;  // Tokens
  
  async send(message: string): Promise<string> {
    const response = await openai.responses.create({
      model: 'gpt-4o',
      input: message,
      previous_response_id: this.getLastResponseId(),
      context: this.compactedContext 
        ? [{ type: 'compacted', item: this.compactedContext }]
        : undefined
    });
    
    this.responseHistory.push(response.id);
    
    // Check if compaction needed
    if (response.usage.total_tokens > this.compactionThreshold) {
      await this.compact();
    }
    
    return response.output_text;
  }
  
  private async compact(): Promise<void> {
    const lastResponseId = this.getLastResponseId();
    if (!lastResponseId) return;
    
    const compaction = await openai.responses.compact(lastResponseId);
    
    this.compactedContext = compaction.item;
    this.responseHistory = [];  // Clear history, context is now compacted
    
    console.log(
      `Compacted ${compaction.original_tokens} → ${compaction.compacted_tokens} tokens ` +
      `(${Math.round((1 - compaction.compacted_tokens / compaction.original_tokens) * 100)}% reduction)`
    );
  }
  
  private getLastResponseId(): string | undefined {
    return this.responseHistory[this.responseHistory.length - 1];
  }
}
```

---

## Client-Side Compression

### Sliding Window

```typescript
class SlidingWindowContext {
  private messages: Message[] = [];
  private maxTokens: number;
  private systemPrompt: string;
  
  constructor(maxTokens: number, systemPrompt: string) {
    this.maxTokens = maxTokens;
    this.systemPrompt = systemPrompt;
  }
  
  addMessage(message: Message): void {
    this.messages.push(message);
    this.trim();
  }
  
  private trim(): void {
    let totalTokens = countTokens(this.systemPrompt);
    
    // Keep most recent messages that fit
    const keep: Message[] = [];
    
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const msgTokens = countTokens(this.messages[i].content);
      
      if (totalTokens + msgTokens <= this.maxTokens) {
        keep.unshift(this.messages[i]);
        totalTokens += msgTokens;
      } else {
        break;  // No more room
      }
    }
    
    this.messages = keep;
  }
  
  getMessages(): Message[] {
    return this.messages;
  }
  
  getContextForAPI() {
    return [
      { role: 'system', content: this.systemPrompt },
      ...this.messages
    ];
  }
}
```

### Summarization-Based Compression

```typescript
class SummarizingContext {
  private recentMessages: Message[] = [];
  private summarizedHistory: string = '';
  private maxRecentMessages = 10;
  private maxSummaryTokens = 2000;
  
  async addMessage(message: Message): Promise<void> {
    this.recentMessages.push(message);
    
    if (this.recentMessages.length > this.maxRecentMessages) {
      await this.summarizeOldMessages();
    }
  }
  
  private async summarizeOldMessages(): Promise<void> {
    // Take oldest half of messages
    const toSummarize = this.recentMessages.splice(
      0, 
      Math.floor(this.recentMessages.length / 2)
    );
    
    // Summarize with AI
    const summaryPrompt = `
Summarize the following conversation exchange concisely, 
preserving key facts, decisions, and context:

${toSummarize.map(m => `${m.role}: ${m.content}`).join('\n\n')}

Previous context summary: ${this.summarizedHistory || 'None'}
`;
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',  // Use cheaper model for summaries
      messages: [{ role: 'user', content: summaryPrompt }],
      max_tokens: this.maxSummaryTokens
    });
    
    this.summarizedHistory = response.choices[0].message.content || '';
  }
  
  getContextForAPI() {
    const context = [];
    
    if (this.summarizedHistory) {
      context.push({
        role: 'system',
        content: `Previous conversation summary:\n${this.summarizedHistory}`
      });
    }
    
    context.push(...this.recentMessages);
    
    return context;
  }
}
```

---

## ZDR-Compatible Compression

Zero Data Retention (ZDR) mode requires special handling since OpenAI doesn't store data.

### ZDR Considerations

```typescript
interface ZDRConfig {
  enabled: boolean;
  clientSideCompression: 'sliding-window' | 'summarization';
  maxClientTokens: number;
}

class ZDRCompatibleChat {
  private config: ZDRConfig;
  private context: SlidingWindowContext | SummarizingContext;
  
  constructor(config: ZDRConfig) {
    this.config = config;
    
    // Can't use server-side compaction with ZDR
    if (config.clientSideCompression === 'sliding-window') {
      this.context = new SlidingWindowContext(config.maxClientTokens, '');
    } else {
      this.context = new SummarizingContext();
    }
  }
  
  async send(message: string, systemPrompt: string): Promise<string> {
    await this.context.addMessage({ 
      role: 'user', 
      content: message,
      id: generateId(),
      createdAt: new Date(),
      status: 'complete'
    });
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: systemPrompt },
        ...this.context.getContextForAPI()
      ]
    });
    
    const assistantContent = response.choices[0].message.content || '';
    
    await this.context.addMessage({
      role: 'assistant',
      content: assistantContent,
      id: generateId(),
      createdAt: new Date(),
      status: 'complete'
    });
    
    return assistantContent;
  }
}
```

---

## UI Feedback

### Context Usage Indicator

```tsx
function ContextUsageBar({ monitor }: { monitor: ContextMonitor }) {
  const status = monitor.getStatus();
  
  return (
    <div className="context-usage">
      <div 
        className={`usage-bar ${status.warning ? 'warning' : ''}`}
        style={{ width: `${Math.min(status.percentage, 100)}%` }}
      />
      <span className="usage-label">
        {status.percentage.toFixed(0)}% context used
        {status.warning && ' - Consider starting new chat'}
      </span>
    </div>
  );
}
```

### Warning Dialog

```tsx
function ContextWarningDialog({ 
  onNewChat, 
  onContinue 
}: { 
  onNewChat: () => void;
  onContinue: () => void;
}) {
  return (
    <dialog open className="context-warning">
      <h3>Conversation Getting Long</h3>
      <p>
        This conversation is approaching the context limit. 
        The AI may lose track of earlier messages.
      </p>
      <div className="actions">
        <button onClick={onNewChat}>Start New Chat</button>
        <button onClick={onContinue}>Continue Anyway</button>
      </div>
    </dialog>
  );
}
```

---

## Advanced Strategies

### Priority-Based Retention

```typescript
interface PrioritizedMessage extends Message {
  priority: 'high' | 'medium' | 'low';
  canSummarize: boolean;
}

class PriorityContext {
  private messages: PrioritizedMessage[] = [];
  
  addMessage(message: Message, priority: 'high' | 'medium' | 'low' = 'medium'): void {
    this.messages.push({
      ...message,
      priority,
      canSummarize: priority !== 'high'
    });
  }
  
  async trim(targetTokens: number): Promise<void> {
    // Sort by priority (keep high priority)
    const sorted = [...this.messages].sort((a, b) => {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
    
    let currentTokens = 0;
    const keep: PrioritizedMessage[] = [];
    const summarize: PrioritizedMessage[] = [];
    
    for (const msg of sorted) {
      const tokens = countTokens(msg.content);
      
      if (currentTokens + tokens <= targetTokens) {
        keep.push(msg);
        currentTokens += tokens;
      } else if (msg.canSummarize) {
        summarize.push(msg);
      }
      // High priority that doesn't fit is still dropped (edge case)
    }
    
    // Summarize dropped messages
    if (summarize.length > 0) {
      const summary = await this.summarize(summarize);
      keep.unshift({
        id: 'summary',
        role: 'system',
        content: `Previous discussion summary: ${summary}`,
        createdAt: new Date(),
        status: 'complete',
        priority: 'medium',
        canSummarize: true
      });
    }
    
    // Restore chronological order
    this.messages = keep.sort(
      (a, b) => a.createdAt.getTime() - b.createdAt.getTime()
    );
  }
  
  private async summarize(messages: PrioritizedMessage[]): Promise<string> {
    // ... summarization logic
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Monitor token usage | Wait for API errors |
| Reserve output tokens | Use full context for history |
| Compact proactively | Wait until limit exceeded |
| Show usage to users | Hide context state |
| Use appropriate model for summaries | Use expensive model for compression |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Hitting context limit | Implement sliding window |
| Lost important context | Use priority-based retention |
| Expensive summarization | Use smaller model |
| No user warning | Add usage indicator |
| ZDR breaks compaction | Use client-side compression |

---

## Hands-on Exercise

### Your Task

Build a context manager that:
1. Tracks token usage in real-time
2. Shows usage percentage to user
3. Automatically summarizes when at 70%
4. Warns user at 90%

### Requirements

1. Implement token counting
2. Create usage monitoring hook
3. Add summarization trigger
4. Build warning UI component

<details>
<summary>💡 Hints (click to expand)</summary>

- Use rough estimation (length/4) for speed
- Track tokens after each message
- Use useEffect for threshold checks
- Make warning dismissable

</details>

---

## Summary

✅ **Token limits** vary by model  
✅ **Budget planning** reserves space  
✅ **Compaction endpoint** compresses server-side  
✅ **Sliding window** keeps recent messages  
✅ **Summarization** preserves context  
✅ **ZDR** requires client-side compression

---

## Further Reading

- [OpenAI Context Windows](https://platform.openai.com/docs/models)
- [tiktoken Tokenizer](https://github.com/openai/tiktoken)
- [Context Window Optimization](https://platform.openai.com/docs/guides/optimizing-context)

---

**Previous:** [Automatic Context Chaining](./05-automatic-context-chaining.md)  
**Next:** [State Updates During Streaming](./07-streaming-state-updates.md)

<!-- 
Sources Consulted:
- OpenAI Models: https://platform.openai.com/docs/models
- tiktoken: https://github.com/openai/tiktoken
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
-->
