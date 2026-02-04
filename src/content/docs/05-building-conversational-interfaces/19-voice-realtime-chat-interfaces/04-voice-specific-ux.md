---
title: "Voice-Specific UX"
---

# Voice-Specific UX

## Introduction

Voice interfaces require UX patterns distinct from text chat. Users need to understand when to speak, see confirmation that they're being heard, and have ways to interrupt or correct the AI. This lesson covers the UX patterns that make voice chat feel natural: push-to-talk, voice activity detection, interruption handling, real-time transcripts, and voice command hints.

### What We'll Cover

- Push-to-talk vs voice activity detection
- Configuring VAD sensitivity
- Interruption handling and UI
- Real-time transcript display
- Voice command hints and onboarding

### Prerequisites

- [Agents SDK Voice Integration](./03-agents-sdk-voice.md)
- Understanding of voice chat states
- React hooks basics

---

## Push-to-Talk vs Voice Activity Detection

### Comparison

| Mode | How It Works | Best For |
|------|--------------|----------|
| **Push-to-Talk (PTT)** | User holds button while speaking | Noisy environments, precise control |
| **Voice Activity Detection (VAD)** | Auto-detects when user speaks | Hands-free, natural conversation |

### Push-to-Talk Implementation

```tsx
interface PushToTalkButtonProps {
  onStartSpeaking: () => void;
  onStopSpeaking: () => void;
  isActive: boolean;
  disabled?: boolean;
}

export function PushToTalkButton({
  onStartSpeaking,
  onStopSpeaking,
  isActive,
  disabled,
}: PushToTalkButtonProps) {
  const [isPressed, setIsPressed] = useState(false);
  
  function handlePointerDown(e: React.PointerEvent) {
    e.preventDefault();
    if (disabled) return;
    
    setIsPressed(true);
    onStartSpeaking();
  }
  
  function handlePointerUp() {
    if (isPressed) {
      setIsPressed(false);
      onStopSpeaking();
    }
  }
  
  // Handle touch/mouse leaving the button
  function handlePointerLeave() {
    if (isPressed) {
      setIsPressed(false);
      onStopSpeaking();
    }
  }
  
  // Keyboard support
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.code === 'Space' && !e.repeat && !disabled) {
        e.preventDefault();
        setIsPressed(true);
        onStartSpeaking();
      }
    }
    
    function handleKeyUp(e: KeyboardEvent) {
      if (e.code === 'Space') {
        setIsPressed(false);
        onStopSpeaking();
      }
    }
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [disabled, onStartSpeaking, onStopSpeaking]);
  
  return (
    <button
      className={`ptt-button ${isPressed ? 'pressed' : ''} ${isActive ? 'active' : ''}`}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onPointerCancel={handlePointerUp}
      disabled={disabled}
      aria-label={isPressed ? 'Speaking...' : 'Hold to speak'}
    >
      <div className="ptt-icon">
        {isPressed ? '🎙️' : '🎤'}
      </div>
      <span className="ptt-label">
        {isPressed ? 'Release to send' : 'Hold to speak'}
      </span>
      <span className="ptt-hint">or hold Spacebar</span>
    </button>
  );
}
```

```css
.ptt-button {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 120px;
  height: 120px;
  border-radius: 50%;
  border: 3px solid #e2e8f0;
  background: white;
  cursor: pointer;
  transition: all 0.15s ease;
  user-select: none;
  touch-action: none;  /* Prevent scroll on touch */
}

.ptt-button:hover:not(:disabled) {
  border-color: #3b82f6;
  background: #eff6ff;
}

.ptt-button.pressed {
  border-color: #22c55e;
  background: #f0fdf4;
  transform: scale(0.95);
}

.ptt-button.active {
  border-color: #22c55e;
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2);
}

.ptt-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.ptt-icon {
  font-size: 2rem;
  margin-bottom: 4px;
}

.ptt-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: #334155;
}

.ptt-hint {
  font-size: 0.625rem;
  color: #94a3b8;
  margin-top: 2px;
}
```

---

## Configuring Voice Activity Detection

### VAD Options

```typescript
const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
  config: {
    turnDetection: {
      type: 'semantic_vad',      // or 'server_vad'
      eagerness: 'medium',       // 'low', 'medium', 'high'
      createResponse: true,      // Auto-trigger response after speech
      interruptResponse: true,   // Allow user to interrupt AI
    },
  },
});
```

### VAD Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `semantic_vad` | Uses AI to understand when user finishes speaking | Natural conversation, handles pauses |
| `server_vad` | Simple silence detection | Lower latency, simpler logic |

### Eagerness Settings

| Setting | Behavior |
|---------|----------|
| `low` | Waits longer for user to finish, fewer false triggers |
| `medium` | Balanced responsiveness |
| `high` | Responds quickly, may interrupt mid-thought |

### VAD Settings UI

```tsx
interface VADSettingsProps {
  settings: {
    type: 'semantic_vad' | 'server_vad';
    eagerness: 'low' | 'medium' | 'high';
    interruptResponse: boolean;
  };
  onSettingsChange: (settings: VADSettingsProps['settings']) => void;
  disabled?: boolean;
}

export function VADSettings({ 
  settings, 
  onSettingsChange, 
  disabled 
}: VADSettingsProps) {
  return (
    <div className="vad-settings">
      <h4>Voice Detection Settings</h4>
      
      <div className="setting-group">
        <label>Detection Mode</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="vadType"
              value="semantic_vad"
              checked={settings.type === 'semantic_vad'}
              onChange={() => onSettingsChange({ 
                ...settings, 
                type: 'semantic_vad' 
              })}
              disabled={disabled}
            />
            <span>Semantic (AI-powered)</span>
          </label>
          <label>
            <input
              type="radio"
              name="vadType"
              value="server_vad"
              checked={settings.type === 'server_vad'}
              onChange={() => onSettingsChange({ 
                ...settings, 
                type: 'server_vad' 
              })}
              disabled={disabled}
            />
            <span>Simple (silence-based)</span>
          </label>
        </div>
      </div>
      
      <div className="setting-group">
        <label>Responsiveness</label>
        <div className="eagerness-slider">
          {(['low', 'medium', 'high'] as const).map((level) => (
            <button
              key={level}
              className={settings.eagerness === level ? 'active' : ''}
              onClick={() => onSettingsChange({ 
                ...settings, 
                eagerness: level 
              })}
              disabled={disabled}
            >
              {level.charAt(0).toUpperCase() + level.slice(1)}
            </button>
          ))}
        </div>
        <span className="setting-description">
          {settings.eagerness === 'low' && 'Waits longer before responding'}
          {settings.eagerness === 'medium' && 'Balanced responsiveness'}
          {settings.eagerness === 'high' && 'Responds quickly'}
        </span>
      </div>
      
      <div className="setting-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={settings.interruptResponse}
            onChange={(e) => onSettingsChange({
              ...settings,
              interruptResponse: e.target.checked,
            })}
            disabled={disabled}
          />
          <span>Allow interrupting the assistant</span>
        </label>
      </div>
    </div>
  );
}
```

---

## Interruption Handling

### Handling Interruptions in Code

```typescript
// Listen for audio interruption events
session.on('audio_interrupted', () => {
  // Stop local audio playback (WebSocket mode)
  audioPlayer.stop();
  
  // Update UI state
  setIsAISpeaking(false);
  setStatus('listening');
});

// Manual interruption (e.g., stop button)
function handleStopButton() {
  session.interrupt();
  // The audio_interrupted event will fire
}
```

### Interruption UI

```tsx
interface InterruptButtonProps {
  isAISpeaking: boolean;
  onInterrupt: () => void;
}

export function InterruptButton({ 
  isAISpeaking, 
  onInterrupt 
}: InterruptButtonProps) {
  if (!isAISpeaking) return null;
  
  return (
    <button
      className="interrupt-button"
      onClick={onInterrupt}
      aria-label="Stop assistant from speaking"
    >
      <span className="interrupt-icon">⏹️</span>
      <span className="interrupt-label">Stop</span>
    </button>
  );
}
```

```css
.interrupt-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  color: #dc2626;
  cursor: pointer;
  transition: all 0.2s;
  animation: fadeIn 0.2s ease;
}

.interrupt-button:hover {
  background: #fee2e2;
  border-color: #f87171;
}

.interrupt-icon {
  font-size: 1rem;
}

.interrupt-label {
  font-size: 0.875rem;
  font-weight: 500;
}

@keyframes fadeIn {
  from { opacity: 0; transform: scale(0.9); }
  to { opacity: 1; transform: scale(1); }
}
```

### Visual Interruption Feedback

```tsx
interface SpeakingIndicatorProps {
  speaker: 'user' | 'assistant' | null;
  canInterrupt: boolean;
  onInterrupt: () => void;
}

export function SpeakingIndicator({ 
  speaker, 
  canInterrupt,
  onInterrupt 
}: SpeakingIndicatorProps) {
  if (!speaker) return null;
  
  return (
    <div className={`speaking-indicator ${speaker}`}>
      <div className="speaking-avatar">
        {speaker === 'user' ? '👤' : '🤖'}
      </div>
      
      <div className="speaking-info">
        <span className="speaking-label">
          {speaker === 'user' ? 'You are speaking' : 'Assistant speaking'}
        </span>
        <div className="speaking-waves">
          <span />
          <span />
          <span />
        </div>
      </div>
      
      {speaker === 'assistant' && canInterrupt && (
        <button 
          className="interrupt-inline"
          onClick={onInterrupt}
        >
          Tap to interrupt
        </button>
      )}
    </div>
  );
}
```

---

## Real-Time Transcript Display

### Streaming Transcript Component

```tsx
interface LiveTranscriptProps {
  userTranscript: string;
  assistantTranscript: string;
  isUserSpeaking: boolean;
  isAssistantSpeaking: boolean;
}

export function LiveTranscript({
  userTranscript,
  assistantTranscript,
  isUserSpeaking,
  isAssistantSpeaking,
}: LiveTranscriptProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [userTranscript, assistantTranscript]);
  
  return (
    <div className="live-transcript" ref={containerRef}>
      {isUserSpeaking && userTranscript && (
        <div className="transcript-bubble user">
          <span className="transcript-role">You</span>
          <p className="transcript-text">{userTranscript}</p>
          <span className="typing-indicator">•••</span>
        </div>
      )}
      
      {isAssistantSpeaking && assistantTranscript && (
        <div className="transcript-bubble assistant">
          <span className="transcript-role">Assistant</span>
          <p className="transcript-text">{assistantTranscript}</p>
          <span className="speaking-wave" />
        </div>
      )}
    </div>
  );
}
```

```css
.live-transcript {
  max-height: 200px;
  overflow-y: auto;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.transcript-bubble {
  padding: 12px 16px;
  border-radius: 12px;
  margin-bottom: 12px;
  animation: slideUp 0.2s ease;
}

.transcript-bubble.user {
  background: #dbeafe;
  margin-left: 20%;
}

.transcript-bubble.assistant {
  background: white;
  border: 1px solid #e2e8f0;
  margin-right: 20%;
}

.transcript-role {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}

.transcript-bubble.user .transcript-role {
  color: #2563eb;
}

.transcript-bubble.assistant .transcript-role {
  color: #64748b;
}

.transcript-text {
  margin: 0;
  font-size: 0.9375rem;
  line-height: 1.5;
  color: #1e293b;
}

.typing-indicator {
  display: inline-block;
  margin-left: 4px;
  animation: blink 1s infinite;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}
```

### Tracking Transcripts from Events

```typescript
function useTranscripts(session: RealtimeSession | null) {
  const [userTranscript, setUserTranscript] = useState('');
  const [assistantTranscript, setAssistantTranscript] = useState('');
  
  useEffect(() => {
    if (!session) return;
    
    // User's speech transcript
    session.transport.on('*', (event: any) => {
      if (event.type === 'conversation.item.input_audio_transcription.completed') {
        setUserTranscript(event.transcript);
      }
      
      if (event.type === 'conversation.item.input_audio_transcription.delta') {
        setUserTranscript(prev => prev + event.delta);
      }
      
      // Assistant's response transcript
      if (event.type === 'response.audio_transcript.delta') {
        setAssistantTranscript(prev => prev + event.delta);
      }
      
      if (event.type === 'response.audio_transcript.done') {
        setAssistantTranscript(event.transcript);
      }
      
      // Clear on new turn
      if (event.type === 'input_audio_buffer.speech_started') {
        setUserTranscript('');
      }
      
      if (event.type === 'response.created') {
        setAssistantTranscript('');
      }
    });
  }, [session]);
  
  return { userTranscript, assistantTranscript };
}
```

---

## Voice Command Hints

### Onboarding Component

```tsx
interface VoiceOnboardingProps {
  onDismiss: () => void;
}

export function VoiceOnboarding({ onDismiss }: VoiceOnboardingProps) {
  return (
    <div className="voice-onboarding">
      <h3>🎤 Voice Chat Tips</h3>
      
      <div className="onboarding-tips">
        <div className="tip">
          <span className="tip-icon">💬</span>
          <div className="tip-content">
            <strong>Just start talking</strong>
            <p>The assistant will automatically detect when you speak.</p>
          </div>
        </div>
        
        <div className="tip">
          <span className="tip-icon">✋</span>
          <div className="tip-content">
            <strong>Interrupt anytime</strong>
            <p>Start speaking to stop the assistant mid-sentence.</p>
          </div>
        </div>
        
        <div className="tip">
          <span className="tip-icon">⏸️</span>
          <div className="tip-content">
            <strong>Pause to finish</strong>
            <p>The assistant responds when you stop speaking.</p>
          </div>
        </div>
      </div>
      
      <button onClick={onDismiss} className="onboarding-dismiss">
        Got it, let's chat!
      </button>
    </div>
  );
}
```

### Command Suggestions

```tsx
interface CommandHintsProps {
  suggestions: string[];
  onSelect: (command: string) => void;
}

export function CommandHints({ suggestions, onSelect }: CommandHintsProps) {
  return (
    <div className="command-hints">
      <span className="hints-label">Try saying:</span>
      <div className="hints-list">
        {suggestions.map((suggestion, i) => (
          <button
            key={i}
            className="hint-chip"
            onClick={() => onSelect(suggestion)}
          >
            "{suggestion}"
          </button>
        ))}
      </div>
    </div>
  );
}
```

```css
.command-hints {
  padding: 12px 16px;
  background: #fefce8;
  border-radius: 12px;
  border: 1px solid #fef08a;
}

.hints-label {
  display: block;
  font-size: 0.75rem;
  font-weight: 500;
  color: #a16207;
  margin-bottom: 8px;
}

.hints-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hint-chip {
  padding: 6px 12px;
  background: white;
  border: 1px solid #fde047;
  border-radius: 16px;
  font-size: 0.875rem;
  color: #854d0e;
  cursor: pointer;
  transition: all 0.2s;
}

.hint-chip:hover {
  background: #fef9c3;
  border-color: #facc15;
}
```

### Context-Aware Suggestions

```typescript
function useContextualSuggestions(
  conversationContext: string,
  agentCapabilities: string[]
): string[] {
  const baseSuggestions = [
    'What can you help me with?',
    'Tell me about yourself',
  ];
  
  const capabilitySuggestions: Record<string, string[]> = {
    weather: ['What\'s the weather like?', 'Will it rain tomorrow?'],
    calendar: ['What\'s on my schedule?', 'Add a meeting for tomorrow'],
    search: ['Search for...', 'Find information about...'],
    math: ['Calculate...', 'What\'s 15% of 200?'],
  };
  
  const suggestions = [...baseSuggestions];
  
  agentCapabilities.forEach(cap => {
    if (capabilitySuggestions[cap]) {
      suggestions.push(...capabilitySuggestions[cap]);
    }
  });
  
  return suggestions.slice(0, 4);  // Limit to 4 suggestions
}
```

---

## Complete Voice UX Component

```tsx
interface VoiceUXProps {
  session: RealtimeSession | null;
  status: 'idle' | 'connecting' | 'connected';
}

export function VoiceUX({ session, status }: VoiceUXProps) {
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [speaker, setSpeaker] = useState<'user' | 'assistant' | null>(null);
  const { userTranscript, assistantTranscript } = useTranscripts(session);
  
  useEffect(() => {
    if (!session) return;
    
    session.transport.on('*', (event: any) => {
      if (event.type === 'input_audio_buffer.speech_started') {
        setSpeaker('user');
      }
      if (event.type === 'input_audio_buffer.speech_stopped') {
        setSpeaker(null);
      }
      if (event.type === 'response.audio.delta') {
        setSpeaker('assistant');
      }
      if (event.type === 'response.done') {
        setSpeaker(null);
      }
    });
  }, [session]);
  
  function handleInterrupt() {
    session?.interrupt();
    setSpeaker(null);
  }
  
  if (status !== 'connected') {
    return null;
  }
  
  return (
    <div className="voice-ux">
      {showOnboarding && (
        <VoiceOnboarding onDismiss={() => setShowOnboarding(false)} />
      )}
      
      <SpeakingIndicator
        speaker={speaker}
        canInterrupt={true}
        onInterrupt={handleInterrupt}
      />
      
      <LiveTranscript
        userTranscript={userTranscript}
        assistantTranscript={assistantTranscript}
        isUserSpeaking={speaker === 'user'}
        isAssistantSpeaking={speaker === 'assistant'}
      />
      
      {!speaker && (
        <CommandHints
          suggestions={[
            'Tell me a joke',
            'What time is it?',
            'Help me brainstorm',
          ]}
          onSelect={(text) => session?.sendMessage(text)}
        />
      )}
    </div>
  );
}
```

---

## Summary

✅ Push-to-talk provides precise control in noisy environments

✅ VAD with semantic detection enables natural hands-free conversation

✅ Eagerness settings balance responsiveness vs false triggers

✅ Interruption handling lets users naturally take over the conversation

✅ Real-time transcripts provide visual feedback for audio

✅ Command hints help users discover voice capabilities

**Previous:** [Agents SDK Voice Integration](./03-agents-sdk-voice.md) | **Next:** [Turn-Taking & Audio Settings](./05-turn-taking-audio-settings.md)

---

## Further Reading

- [Realtime VAD Documentation](https://platform.openai.com/docs/guides/realtime-vad) — VAD settings
- [Voice Agents Build Guide](https://openai.github.io/openai-agents-js/guides/voice-agents/build#interruptions) — Interruptions
- [Conversational Design](https://designguidelines.withgoogle.com/conversation/) — Google guidelines

---

<!-- 
Sources Consulted:
- OpenAI Agents SDK Build Guide: https://openai.github.io/openai-agents-js/guides/voice-agents/build
- OpenAI Realtime API: https://platform.openai.com/docs/guides/realtime-webrtc
- Google Conversation Design: https://designguidelines.withgoogle.com/conversation/
-->
