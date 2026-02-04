---
title: "Agents SDK Voice Integration"
---

# Agents SDK Voice Integration

## Introduction

While raw WebRTC provides low-level control, the OpenAI Agents SDK offers a higher-level abstraction that handles connection management, tool execution, guardrails, and conversation history automatically. The SDK's `RealtimeAgent` and `RealtimeSession` classes simplify building production-ready voice agents.

This lesson covers the Agents SDK's voice capabilities: creating agents, managing sessions, voice selection, tools, guardrails, and conversation history.

### What We'll Cover

- RealtimeAgent configuration
- RealtimeSession lifecycle
- Voice selection and audio options
- Tools and function calling
- Output guardrails
- Conversation history management
- Handoffs between agents

### Prerequisites

- [WebRTC Implementation](./02-webrtc-implementation.md)
- Node.js and npm basics
- TypeScript fundamentals

---

## RealtimeAgent Configuration

### Creating a Basic Agent

```typescript
import { RealtimeAgent } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
  instructions: `You are a helpful voice assistant.
    Keep responses concise since this is a voice conversation.
    Confirm understanding before taking actions.`,
});
```

### Agent with Tools

```typescript
import { RealtimeAgent, tool } from '@openai/agents/realtime';
import { z } from 'zod';

// Define a tool
const getWeather = tool({
  name: 'get_weather',
  description: 'Get current weather for a city',
  parameters: z.object({
    city: z.string().describe('City name'),
    unit: z.enum(['celsius', 'fahrenheit']).optional(),
  }),
  async execute({ city, unit = 'celsius' }) {
    // API call here
    return `The weather in ${city} is 22°${unit === 'celsius' ? 'C' : 'F'} and sunny.`;
  },
});

const weatherAgent = new RealtimeAgent({
  name: 'Weather Assistant',
  instructions: 'You help users check the weather. Be brief and clear.',
  tools: [getWeather],
});
```

### Agent Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `name` | string | Agent identifier |
| `instructions` | string | System prompt for the agent |
| `tools` | Tool[] | Function tools the agent can call |
| `handoffs` | RealtimeAgent[] | Agents to delegate to |
| `handoffDescription` | string | Description for when to handoff to this agent |

---

## RealtimeSession Lifecycle

### Creating a Session

```typescript
import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant.',
});

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
  config: {
    inputAudioFormat: 'pcm16',
    outputAudioFormat: 'pcm16',
    inputAudioTranscription: {
      model: 'gpt-4o-mini-transcribe',
    },
  },
});
```

### Session Configuration Options

```typescript
interface RealtimeSessionConfig {
  model?: string;                    // Model to use
  config?: {
    inputAudioFormat?: 'pcm16' | 'g711-ulaw' | 'g711-alaw';
    outputAudioFormat?: 'pcm16' | 'g711-ulaw' | 'g711-alaw';
    inputAudioTranscription?: {
      model: string;                 // Transcription model
    };
    turnDetection?: {
      type: 'semantic_vad' | 'server_vad';
      eagerness?: 'low' | 'medium' | 'high';
      createResponse?: boolean;
      interruptResponse?: boolean;
    };
  };
  outputGuardrails?: RealtimeOutputGuardrail[];
  historyStoreAudio?: boolean;       // Store audio in history
  tracingDisabled?: boolean;         // Disable tracing
}
```

### Connecting the Session

```typescript
// Browser: Auto-configures WebRTC with microphone
await session.connect({
  apiKey: ephemeralKey,  // Client ephemeral key
});

// Server (Node.js): Uses WebSocket
await session.connect({
  apiKey: process.env.OPENAI_API_KEY,
});
```

### Session Events

```typescript
// Connection events
session.on('connected', () => {
  console.log('Session connected');
});

session.on('disconnected', () => {
  console.log('Session disconnected');
});

session.on('error', (error) => {
  console.error('Session error:', error);
});

// Conversation events
session.on('history_updated', (history) => {
  console.log('History updated:', history.length, 'items');
});

session.on('history_added', (item) => {
  console.log('New item:', item.type, item.role);
});

// Audio events (WebSocket only)
session.on('audio', (event) => {
  // event.data is PCM16 audio chunk
  playAudio(event.data);
});

session.on('audio_interrupted', () => {
  // Stop local audio playback
  stopAudioPlayback();
});

// Tool events
session.on('tool_approval_requested', (context, agent, request) => {
  // Show approval UI
  const approved = await showApprovalDialog(request);
  if (approved) {
    session.approve(request.approvalItem);
  } else {
    session.reject(request.rawItem);
  }
});

// Guardrail events
session.on('guardrail_tripped', (context, guardrail, details) => {
  console.warn('Guardrail tripped:', guardrail.name, details);
});
```

---

## Voice Selection

### Available Voices

OpenAI's Realtime API supports multiple voices:

| Voice | Characteristics |
|-------|-----------------|
| `alloy` | Neutral, balanced |
| `ash` | Warm, conversational |
| `ballad` | Soft, gentle |
| `coral` | Clear, professional |
| `sage` | Calm, thoughtful |
| `verse` | Dynamic, expressive |
| `marin` | Friendly, approachable |

### Setting Voice in Session

```typescript
const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
  config: {
    voice: 'coral',
  },
});
```

### Voice Selection UI

```tsx
interface VoiceSelectorProps {
  selectedVoice: string;
  onVoiceChange: (voice: string) => void;
  disabled?: boolean;
}

const voices = [
  { id: 'alloy', name: 'Alloy', description: 'Neutral, balanced' },
  { id: 'ash', name: 'Ash', description: 'Warm, conversational' },
  { id: 'coral', name: 'Coral', description: 'Clear, professional' },
  { id: 'sage', name: 'Sage', description: 'Calm, thoughtful' },
  { id: 'verse', name: 'Verse', description: 'Dynamic, expressive' },
  { id: 'marin', name: 'Marin', description: 'Friendly, approachable' },
];

export function VoiceSelector({ 
  selectedVoice, 
  onVoiceChange,
  disabled 
}: VoiceSelectorProps) {
  return (
    <div className="voice-selector">
      <label>Assistant Voice</label>
      <div className="voice-grid">
        {voices.map((voice) => (
          <button
            key={voice.id}
            onClick={() => onVoiceChange(voice.id)}
            disabled={disabled}
            className={`voice-option ${selectedVoice === voice.id ? 'selected' : ''}`}
          >
            <span className="voice-name">{voice.name}</span>
            <span className="voice-description">{voice.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
```

```css
.voice-selector {
  margin-bottom: 20px;
}

.voice-selector label {
  display: block;
  font-weight: 500;
  margin-bottom: 12px;
  color: #334155;
}

.voice-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
}

.voice-option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.voice-option:hover:not(:disabled) {
  border-color: #3b82f6;
}

.voice-option.selected {
  border-color: #3b82f6;
  background: #eff6ff;
}

.voice-option:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.voice-name {
  font-weight: 500;
  color: #1e293b;
}

.voice-description {
  font-size: 0.75rem;
  color: #64748b;
  margin-top: 2px;
}
```

---

## Tools and Function Calling

### Defining Tools

```typescript
import { tool, RealtimeContextData } from '@openai/agents/realtime';
import { z } from 'zod';

// Simple tool
const getCurrentTime = tool({
  name: 'get_current_time',
  description: 'Get the current time',
  parameters: z.object({
    timezone: z.string().optional(),
  }),
  async execute({ timezone }) {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
      timeZone: timezone || 'UTC' 
    });
  },
});

// Tool with context access
const summarizeConversation = tool<
  z.ZodObject<{ topic: z.ZodString }>,
  RealtimeContextData
>({
  name: 'summarize_conversation',
  description: 'Summarize the current conversation',
  parameters: z.object({
    topic: z.string().describe('Topic to focus on'),
  }),
  async execute({ topic }, details) {
    const history = details?.context?.history ?? [];
    
    // Process history to create summary
    const messages = history
      .filter(item => item.type === 'message')
      .map(item => `${item.role}: ${item.transcript || '[audio]'}`)
      .join('\n');
    
    return `Conversation about ${topic}:\n${messages}`;
  },
});

// Tool requiring approval
const placeOrder = tool({
  name: 'place_order',
  description: 'Place an order for the customer',
  parameters: z.object({
    items: z.array(z.string()),
    total: z.number(),
  }),
  needsApproval: true,  // Requires user approval
  async execute({ items, total }) {
    // Process order
    return `Order placed: ${items.join(', ')} for $${total}`;
  },
});
```

### Tool Approval UI

```tsx
interface ToolApprovalDialogProps {
  request: {
    toolName: string;
    arguments: Record<string, unknown>;
  };
  onApprove: () => void;
  onReject: () => void;
}

export function ToolApprovalDialog({ 
  request, 
  onApprove, 
  onReject 
}: ToolApprovalDialogProps) {
  return (
    <div className="tool-approval-dialog">
      <div className="dialog-content">
        <h3>Action Requested</h3>
        <p>The assistant wants to perform:</p>
        
        <div className="tool-details">
          <div className="tool-name">{request.toolName}</div>
          <pre className="tool-args">
            {JSON.stringify(request.arguments, null, 2)}
          </pre>
        </div>
        
        <div className="dialog-actions">
          <button onClick={onReject} className="reject-button">
            Deny
          </button>
          <button onClick={onApprove} className="approve-button">
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

## Output Guardrails

### Creating Guardrails

Guardrails monitor the agent's output and can interrupt if violations are detected:

```typescript
import { 
  RealtimeOutputGuardrail, 
  RealtimeAgent, 
  RealtimeSession 
} from '@openai/agents/realtime';

const noProfanityGuardrail: RealtimeOutputGuardrail = {
  name: 'no_profanity',
  async execute({ agentOutput }) {
    const profanityList = ['badword1', 'badword2'];
    const hasProfanity = profanityList.some(word => 
      agentOutput.toLowerCase().includes(word)
    );
    
    return {
      tripwireTriggered: hasProfanity,
      outputInfo: { hasProfanity },
    };
  },
};

const stayOnTopicGuardrail: RealtimeOutputGuardrail = {
  name: 'stay_on_topic',
  async execute({ agentOutput }) {
    const offTopicPhrases = [
      'I cannot help with that',
      'outside my expertise',
    ];
    
    const isOffTopic = offTopicPhrases.some(phrase =>
      agentOutput.toLowerCase().includes(phrase.toLowerCase())
    );
    
    return {
      tripwireTriggered: isOffTopic,
      outputInfo: { isOffTopic },
    };
  },
};

// Apply guardrails to session
const session = new RealtimeSession(agent, {
  outputGuardrails: [noProfanityGuardrail, stayOnTopicGuardrail],
  outputGuardrailSettings: {
    debounceTextLength: 100,  // Check every 100 characters
  },
});

// Handle guardrail trips
session.on('guardrail_tripped', (context, guardrail, details) => {
  console.warn(`Guardrail "${guardrail.name}" tripped:`, details);
  // Audio is automatically cut off
});
```

---

## Conversation History Management

### Accessing History

```typescript
// Get current history snapshot
const history = session.history;

// Listen for updates
session.on('history_updated', (history) => {
  renderConversationHistory(history);
});

session.on('history_added', (item) => {
  appendToConversation(item);
});
```

### History Item Types

```typescript
interface RealtimeItem {
  id: string;
  type: 'message' | 'function_call' | 'function_call_output';
  role?: 'user' | 'assistant' | 'system';
  content?: Array<{
    type: 'input_text' | 'input_audio' | 'text' | 'audio';
    text?: string;
    transcript?: string;
  }>;
  transcript?: string;  // Transcription of audio
  status?: 'in_progress' | 'completed' | 'incomplete';
}
```

### Rendering Conversation History

```tsx
interface ConversationHistoryProps {
  history: RealtimeItem[];
}

export function ConversationHistory({ history }: ConversationHistoryProps) {
  return (
    <div className="conversation-history">
      {history.map((item) => (
        <HistoryItem key={item.id} item={item} />
      ))}
    </div>
  );
}

function HistoryItem({ item }: { item: RealtimeItem }) {
  if (item.type === 'message') {
    const isUser = item.role === 'user';
    
    return (
      <div className={`history-message ${isUser ? 'user' : 'assistant'}`}>
        <div className="message-role">
          {isUser ? 'You' : 'Assistant'}
        </div>
        <div className="message-content">
          {item.transcript || '[Audio message]'}
        </div>
        {item.status === 'in_progress' && (
          <span className="in-progress-indicator">Speaking...</span>
        )}
      </div>
    );
  }
  
  if (item.type === 'function_call') {
    return (
      <div className="history-function-call">
        <span className="function-icon">🔧</span>
        <span className="function-name">{item.name}</span>
      </div>
    );
  }
  
  return null;
}
```

### Modifying History

```typescript
// Remove all assistant messages
session.updateHistory((currentHistory) => {
  return currentHistory.filter(
    item => !(item.type === 'message' && item.role === 'assistant')
  );
});

// Clear entire history
session.updateHistory([]);

// Add system context
session.updateHistory((currentHistory) => [
  {
    id: 'system-context',
    type: 'message',
    role: 'system',
    content: [{ type: 'input_text', text: 'User prefers brief responses.' }],
  },
  ...currentHistory,
]);
```

---

## Agent Handoffs

### Creating Handoff Agents

```typescript
const mathTutor = new RealtimeAgent({
  name: 'Math Tutor',
  handoffDescription: 'Specialist for math questions and calculations',
  instructions: `You are a math tutor. 
    Explain concepts step by step.
    Use simple examples.`,
  tools: [calculator],
});

const writingHelper = new RealtimeAgent({
  name: 'Writing Helper',
  handoffDescription: 'Specialist for writing and grammar questions',
  instructions: `You help with writing tasks.
    Suggest improvements and explain grammar rules.`,
});

const mainAgent = new RealtimeAgent({
  name: 'Assistant',
  instructions: `You are a helpful assistant.
    When users have math questions, hand off to the Math Tutor.
    When users need writing help, hand off to the Writing Helper.`,
  handoffs: [mathTutor, writingHelper],
});
```

> **Note:** During handoffs, the session updates with the new agent's configuration but retains conversation history. Voice and model cannot change during handoffs.

---

## Complete Voice Agent Example

```tsx
import { useState, useEffect, useCallback } from 'react';
import { 
  RealtimeAgent, 
  RealtimeSession,
  tool,
  RealtimeItem 
} from '@openai/agents/realtime';
import { z } from 'zod';

export function VoiceAgentChat() {
  const [session, setSession] = useState<RealtimeSession | null>(null);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected'>('idle');
  const [history, setHistory] = useState<RealtimeItem[]>([]);
  const [voice, setVoice] = useState('coral');
  
  const connect = useCallback(async () => {
    setStatus('connecting');
    
    // Get ephemeral token
    const tokenRes = await fetch('/api/realtime/token');
    const { value: apiKey } = await tokenRes.json();
    
    // Create agent with tools
    const agent = new RealtimeAgent({
      name: 'Voice Assistant',
      instructions: 'You are a helpful voice assistant. Be concise.',
      tools: [
        tool({
          name: 'get_time',
          description: 'Get current time',
          parameters: z.object({}),
          execute: async () => new Date().toLocaleTimeString(),
        }),
      ],
    });
    
    // Create session
    const newSession = new RealtimeSession(agent, {
      model: 'gpt-realtime',
      config: { voice },
    });
    
    // Event handlers
    newSession.on('history_updated', setHistory);
    newSession.on('error', (err) => console.error(err));
    
    // Connect
    await newSession.connect({ apiKey });
    
    setSession(newSession);
    setStatus('connected');
  }, [voice]);
  
  const disconnect = useCallback(() => {
    session?.disconnect();
    setSession(null);
    setStatus('idle');
    setHistory([]);
  }, [session]);
  
  return (
    <div className="voice-agent">
      <div className="controls">
        <VoiceSelector
          selectedVoice={voice}
          onVoiceChange={setVoice}
          disabled={status !== 'idle'}
        />
        
        {status === 'idle' && (
          <button onClick={connect} className="connect-button">
            Start Voice Chat
          </button>
        )}
        
        {status === 'connecting' && (
          <div className="connecting">Connecting...</div>
        )}
        
        {status === 'connected' && (
          <button onClick={disconnect} className="disconnect-button">
            End Chat
          </button>
        )}
      </div>
      
      {status === 'connected' && (
        <ConversationHistory history={history} />
      )}
    </div>
  );
}
```

---

## Summary

✅ RealtimeAgent defines voice assistant behavior with instructions and tools

✅ RealtimeSession manages the connection lifecycle and events

✅ Voice selection customizes the assistant's speaking voice

✅ Tools enable function calling with optional approval requirements

✅ Guardrails monitor output and can interrupt problematic responses

✅ Conversation history is automatically managed and can be modified

**Previous:** [WebRTC Implementation](./02-webrtc-implementation.md) | **Next:** [Voice-Specific UX](./04-voice-specific-ux.md)

---

## Further Reading

- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-js/) — Full SDK documentation
- [Voice Agents Guide](https://openai.github.io/openai-agents-js/guides/voice-agents/build) — Building voice agents
- [Realtime Transport](https://openai.github.io/openai-agents-js/guides/voice-agents/transport) — Transport mechanisms

---

<!-- 
Sources Consulted:
- OpenAI Agents SDK: https://openai.github.io/openai-agents-js/
- Voice Agents Build Guide: https://openai.github.io/openai-agents-js/guides/voice-agents/build
- Voice Agents Quickstart: https://openai.github.io/openai-agents-js/guides/voice-agents/quickstart
-->
