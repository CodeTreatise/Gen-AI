---
title: "Turn-Taking & Audio Settings"
---

# Turn-Taking & Audio Settings

## Introduction

Effective voice chat requires clear turn-taking cues and user control over audio devices. Users need to know who's speaking, see speaking duration, and have access to input/output device selection and volume controls. This lesson covers the UI patterns for visualizing conversation flow and managing audio settings.

### What We'll Cover

- Who's speaking indicators
- Speaking duration display
- Conversation flow visualization
- Input/output device selection
- Volume controls and monitoring
- Echo cancellation and noise suppression settings

### Prerequisites

- [Voice-Specific UX](./04-voice-specific-ux.md)
- Understanding of Web Audio API basics
- React state management

---

## Who's Speaking Indicator

### Turn Indicator Component

```tsx
interface TurnIndicatorProps {
  currentSpeaker: 'user' | 'assistant' | null;
  userName?: string;
  assistantName?: string;
}

export function TurnIndicator({ 
  currentSpeaker, 
  userName = 'You',
  assistantName = 'Assistant'
}: TurnIndicatorProps) {
  return (
    <div className="turn-indicator">
      <div className={`turn-participant user ${currentSpeaker === 'user' ? 'speaking' : ''}`}>
        <div className="participant-avatar">👤</div>
        <span className="participant-name">{userName}</span>
        {currentSpeaker === 'user' && (
          <div className="speaking-waves">
            <span />
            <span />
            <span />
          </div>
        )}
      </div>
      
      <div className="turn-divider">
        <div className={`turn-arrow ${currentSpeaker || 'idle'}`}>
          {currentSpeaker === 'user' && '→'}
          {currentSpeaker === 'assistant' && '←'}
          {!currentSpeaker && '↔'}
        </div>
      </div>
      
      <div className={`turn-participant assistant ${currentSpeaker === 'assistant' ? 'speaking' : ''}`}>
        <div className="participant-avatar">🤖</div>
        <span className="participant-name">{assistantName}</span>
        {currentSpeaker === 'assistant' && (
          <div className="speaking-waves">
            <span />
            <span />
            <span />
          </div>
        )}
      </div>
    </div>
  );
}
```

```css
.turn-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 16px;
}

.turn-participant {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-radius: 12px;
  background: white;
  border: 2px solid transparent;
  transition: all 0.3s ease;
  min-width: 100px;
}

.turn-participant.speaking {
  border-color: #22c55e;
  background: #f0fdf4;
}

.turn-participant.user.speaking {
  border-color: #3b82f6;
  background: #eff6ff;
}

.participant-avatar {
  font-size: 2rem;
}

.participant-name {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
}

.turn-divider {
  display: flex;
  align-items: center;
}

.turn-arrow {
  font-size: 1.5rem;
  color: #94a3b8;
  transition: all 0.3s ease;
}

.turn-arrow.user {
  color: #3b82f6;
  animation: pulse-right 1s ease-in-out infinite;
}

.turn-arrow.assistant {
  color: #22c55e;
  animation: pulse-left 1s ease-in-out infinite;
}

.speaking-waves {
  display: flex;
  gap: 3px;
  height: 16px;
  align-items: center;
}

.speaking-waves span {
  width: 3px;
  height: 8px;
  background: currentColor;
  border-radius: 2px;
  animation: wave 0.5s ease-in-out infinite;
}

.turn-participant.user .speaking-waves {
  color: #3b82f6;
}

.turn-participant.assistant .speaking-waves {
  color: #22c55e;
}

.speaking-waves span:nth-child(1) { animation-delay: 0s; }
.speaking-waves span:nth-child(2) { animation-delay: 0.15s; }
.speaking-waves span:nth-child(3) { animation-delay: 0.3s; }

@keyframes wave {
  0%, 100% { height: 4px; }
  50% { height: 16px; }
}

@keyframes pulse-right {
  0%, 100% { transform: translateX(0); opacity: 1; }
  50% { transform: translateX(4px); opacity: 0.7; }
}

@keyframes pulse-left {
  0%, 100% { transform: translateX(0); opacity: 1; }
  50% { transform: translateX(-4px); opacity: 0.7; }
}
```

---

## Speaking Duration Display

### Duration Timer Hook

```typescript
function useSpeakingDuration(isSpeaking: boolean) {
  const [duration, setDuration] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  
  useEffect(() => {
    if (isSpeaking) {
      startTimeRef.current = Date.now();
      
      const interval = setInterval(() => {
        if (startTimeRef.current) {
          setDuration(Date.now() - startTimeRef.current);
        }
      }, 100);
      
      return () => clearInterval(interval);
    } else {
      startTimeRef.current = null;
      // Keep last duration visible briefly
      const timeout = setTimeout(() => setDuration(0), 2000);
      return () => clearTimeout(timeout);
    }
  }, [isSpeaking]);
  
  return duration;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  
  if (minutes > 0) {
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
  return `${seconds}s`;
}
```

### Duration Display Component

```tsx
interface SpeakingDurationProps {
  speaker: 'user' | 'assistant';
  isSpeaking: boolean;
}

export function SpeakingDuration({ speaker, isSpeaking }: SpeakingDurationProps) {
  const duration = useSpeakingDuration(isSpeaking);
  
  if (duration === 0) return null;
  
  return (
    <div className={`speaking-duration ${speaker} ${isSpeaking ? 'active' : 'fading'}`}>
      <span className="duration-label">
        {speaker === 'user' ? 'You' : 'Assistant'}
      </span>
      <span className="duration-time">{formatDuration(duration)}</span>
    </div>
  );
}
```

```css
.speaking-duration {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.875rem;
  transition: opacity 0.3s ease;
}

.speaking-duration.user {
  background: #eff6ff;
  color: #2563eb;
}

.speaking-duration.assistant {
  background: #f0fdf4;
  color: #16a34a;
}

.speaking-duration.fading {
  opacity: 0.5;
}

.duration-label {
  font-weight: 500;
}

.duration-time {
  font-family: 'JetBrains Mono', monospace;
}
```

---

## Conversation Flow Visualization

### Turn History Timeline

```tsx
interface Turn {
  id: string;
  speaker: 'user' | 'assistant';
  startTime: number;
  endTime?: number;
  transcript?: string;
}

interface ConversationTimelineProps {
  turns: Turn[];
  currentTime: number;
}

export function ConversationTimeline({ 
  turns, 
  currentTime 
}: ConversationTimelineProps) {
  const totalDuration = currentTime - (turns[0]?.startTime || currentTime);
  
  return (
    <div className="conversation-timeline">
      <div className="timeline-track">
        {turns.map((turn) => {
          const startPercent = ((turn.startTime - turns[0].startTime) / totalDuration) * 100;
          const duration = (turn.endTime || currentTime) - turn.startTime;
          const widthPercent = (duration / totalDuration) * 100;
          
          return (
            <div
              key={turn.id}
              className={`timeline-segment ${turn.speaker}`}
              style={{
                left: `${startPercent}%`,
                width: `${Math.max(widthPercent, 1)}%`,
              }}
              title={turn.transcript || `${turn.speaker} speaking`}
            />
          );
        })}
      </div>
      
      <div className="timeline-labels">
        <span>Start</span>
        <span>{formatDuration(totalDuration)}</span>
      </div>
    </div>
  );
}
```

```css
.conversation-timeline {
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.timeline-track {
  position: relative;
  height: 24px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.timeline-segment {
  position: absolute;
  height: 100%;
  border-radius: 2px;
  transition: width 0.1s linear;
}

.timeline-segment.user {
  background: #3b82f6;
}

.timeline-segment.assistant {
  background: #22c55e;
}

.timeline-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 0.75rem;
  color: #64748b;
}
```

---

## Device Selection

### Audio Device Hook

```typescript
interface AudioDevice {
  deviceId: string;
  label: string;
  kind: 'audioinput' | 'audiooutput';
}

function useAudioDevices() {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedInput, setSelectedInput] = useState<string>('');
  const [selectedOutput, setSelectedOutput] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  
  async function loadDevices() {
    try {
      // Need permission first to get device labels
      await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioDevices = allDevices
        .filter(d => d.kind === 'audioinput' || d.kind === 'audiooutput')
        .map(d => ({
          deviceId: d.deviceId,
          label: d.label || `${d.kind === 'audioinput' ? 'Microphone' : 'Speaker'} ${d.deviceId.slice(0, 8)}`,
          kind: d.kind as 'audioinput' | 'audiooutput',
        }));
      
      setDevices(audioDevices);
      
      // Set defaults
      const defaultInput = audioDevices.find(d => d.kind === 'audioinput');
      const defaultOutput = audioDevices.find(d => d.kind === 'audiooutput');
      
      if (defaultInput && !selectedInput) {
        setSelectedInput(defaultInput.deviceId);
      }
      if (defaultOutput && !selectedOutput) {
        setSelectedOutput(defaultOutput.deviceId);
      }
    } catch (err) {
      setError('Failed to enumerate audio devices');
    }
  }
  
  useEffect(() => {
    loadDevices();
    
    // Listen for device changes
    navigator.mediaDevices.addEventListener('devicechange', loadDevices);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', loadDevices);
    };
  }, []);
  
  const inputDevices = devices.filter(d => d.kind === 'audioinput');
  const outputDevices = devices.filter(d => d.kind === 'audiooutput');
  
  return {
    inputDevices,
    outputDevices,
    selectedInput,
    selectedOutput,
    setSelectedInput,
    setSelectedOutput,
    error,
    refresh: loadDevices,
  };
}
```

### Device Selection UI

```tsx
interface DeviceSelectorProps {
  inputDevices: AudioDevice[];
  outputDevices: AudioDevice[];
  selectedInput: string;
  selectedOutput: string;
  onInputChange: (deviceId: string) => void;
  onOutputChange: (deviceId: string) => void;
  disabled?: boolean;
}

export function DeviceSelector({
  inputDevices,
  outputDevices,
  selectedInput,
  selectedOutput,
  onInputChange,
  onOutputChange,
  disabled,
}: DeviceSelectorProps) {
  return (
    <div className="device-selector">
      <div className="device-group">
        <label>
          <span className="device-icon">🎤</span>
          Microphone
        </label>
        <select
          value={selectedInput}
          onChange={(e) => onInputChange(e.target.value)}
          disabled={disabled}
        >
          {inputDevices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label}
            </option>
          ))}
        </select>
      </div>
      
      <div className="device-group">
        <label>
          <span className="device-icon">🔊</span>
          Speaker
        </label>
        <select
          value={selectedOutput}
          onChange={(e) => onOutputChange(e.target.value)}
          disabled={disabled}
        >
          {outputDevices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
```

```css
.device-selector {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.device-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.device-group label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
}

.device-icon {
  font-size: 1rem;
}

.device-group select {
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.875rem;
  background: white;
  cursor: pointer;
}

.device-group select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.device-group select:disabled {
  background: #f1f5f9;
  cursor: not-allowed;
}
```

### Applying Device Selection

```typescript
async function applyInputDevice(
  deviceId: string,
  pc: RTCPeerConnection
): Promise<MediaStream> {
  // Get new stream with selected device
  const newStream = await navigator.mediaDevices.getUserMedia({
    audio: { deviceId: { exact: deviceId } },
  });
  
  // Replace existing track
  const sender = pc.getSenders().find(s => s.track?.kind === 'audio');
  if (sender) {
    await sender.replaceTrack(newStream.getTracks()[0]);
  }
  
  return newStream;
}

function applyOutputDevice(
  deviceId: string,
  audioElement: HTMLAudioElement
) {
  // setSinkId is not available on all browsers
  if ('setSinkId' in audioElement) {
    (audioElement as any).setSinkId(deviceId);
  }
}
```

---

## Volume Controls

### Volume Slider Component

```tsx
interface VolumeControlProps {
  volume: number;  // 0-1
  onVolumeChange: (volume: number) => void;
  isMuted: boolean;
  onMuteToggle: () => void;
  label: string;
}

export function VolumeControl({
  volume,
  onVolumeChange,
  isMuted,
  onMuteToggle,
  label,
}: VolumeControlProps) {
  const volumePercent = Math.round(volume * 100);
  
  return (
    <div className="volume-control">
      <button
        className={`mute-button ${isMuted ? 'muted' : ''}`}
        onClick={onMuteToggle}
        aria-label={isMuted ? 'Unmute' : 'Mute'}
      >
        {isMuted ? '🔇' : volume > 0.5 ? '🔊' : volume > 0 ? '🔉' : '🔈'}
      </button>
      
      <div className="volume-slider-container">
        <label>{label}</label>
        <input
          type="range"
          min="0"
          max="100"
          value={isMuted ? 0 : volumePercent}
          onChange={(e) => onVolumeChange(parseInt(e.target.value) / 100)}
          className="volume-slider"
          aria-label={`${label} volume`}
        />
        <span className="volume-value">{isMuted ? 0 : volumePercent}%</span>
      </div>
    </div>
  );
}
```

```css
.volume-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mute-button {
  width: 40px;
  height: 40px;
  border: none;
  background: #f1f5f9;
  border-radius: 8px;
  font-size: 1.25rem;
  cursor: pointer;
  transition: all 0.2s;
}

.mute-button:hover {
  background: #e2e8f0;
}

.mute-button.muted {
  background: #fef2f2;
}

.volume-slider-container {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 12px;
}

.volume-slider-container label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
  min-width: 80px;
}

.volume-slider {
  flex: 1;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: #e2e8f0;
  border-radius: 3px;
  cursor: pointer;
}

.volume-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  background: #3b82f6;
  border-radius: 50%;
  cursor: pointer;
  transition: transform 0.1s;
}

.volume-slider::-webkit-slider-thumb:hover {
  transform: scale(1.2);
}

.volume-value {
  font-size: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  color: #64748b;
  min-width: 40px;
  text-align: right;
}
```

### Microphone Gain Control

```typescript
function useMicrophoneGain(stream: MediaStream | null) {
  const [gain, setGain] = useState(1);
  const gainNodeRef = useRef<GainNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  
  useEffect(() => {
    if (!stream) return;
    
    const audioContext = new AudioContext();
    audioContextRef.current = audioContext;
    
    const source = audioContext.createMediaStreamSource(stream);
    const gainNode = audioContext.createGain();
    gainNodeRef.current = gainNode;
    
    source.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    return () => {
      audioContext.close();
    };
  }, [stream]);
  
  function updateGain(newGain: number) {
    setGain(newGain);
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = newGain;
    }
  }
  
  return { gain, setGain: updateGain };
}
```

---

## Audio Processing Settings

### Echo Cancellation & Noise Suppression

```tsx
interface AudioProcessingSettingsProps {
  settings: {
    echoCancellation: boolean;
    noiseSuppression: boolean;
    autoGainControl: boolean;
  };
  onSettingsChange: (settings: AudioProcessingSettingsProps['settings']) => void;
  disabled?: boolean;
}

export function AudioProcessingSettings({
  settings,
  onSettingsChange,
  disabled,
}: AudioProcessingSettingsProps) {
  return (
    <div className="audio-processing-settings">
      <h4>Audio Processing</h4>
      
      <label className="setting-toggle">
        <input
          type="checkbox"
          checked={settings.echoCancellation}
          onChange={(e) => onSettingsChange({
            ...settings,
            echoCancellation: e.target.checked,
          })}
          disabled={disabled}
        />
        <span className="toggle-slider" />
        <span className="toggle-label">
          Echo Cancellation
          <span className="toggle-description">
            Reduces echo from speakers
          </span>
        </span>
      </label>
      
      <label className="setting-toggle">
        <input
          type="checkbox"
          checked={settings.noiseSuppression}
          onChange={(e) => onSettingsChange({
            ...settings,
            noiseSuppression: e.target.checked,
          })}
          disabled={disabled}
        />
        <span className="toggle-slider" />
        <span className="toggle-label">
          Noise Suppression
          <span className="toggle-description">
            Filters background noise
          </span>
        </span>
      </label>
      
      <label className="setting-toggle">
        <input
          type="checkbox"
          checked={settings.autoGainControl}
          onChange={(e) => onSettingsChange({
            ...settings,
            autoGainControl: e.target.checked,
          })}
          disabled={disabled}
        />
        <span className="toggle-slider" />
        <span className="toggle-label">
          Auto Gain Control
          <span className="toggle-description">
            Normalizes volume levels
          </span>
        </span>
      </label>
    </div>
  );
}
```

```css
.audio-processing-settings {
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.audio-processing-settings h4 {
  margin: 0 0 16px;
  font-size: 0.875rem;
  font-weight: 600;
  color: #334155;
}

.setting-toggle {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #e2e8f0;
  cursor: pointer;
}

.setting-toggle:last-child {
  border-bottom: none;
}

.setting-toggle input {
  display: none;
}

.toggle-slider {
  width: 44px;
  height: 24px;
  background: #e2e8f0;
  border-radius: 12px;
  position: relative;
  transition: background 0.2s;
  flex-shrink: 0;
}

.toggle-slider::after {
  content: '';
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background: white;
  border-radius: 50%;
  transition: transform 0.2s;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.setting-toggle input:checked + .toggle-slider {
  background: #22c55e;
}

.setting-toggle input:checked + .toggle-slider::after {
  transform: translateX(20px);
}

.toggle-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 0.875rem;
  font-weight: 500;
  color: #334155;
}

.toggle-description {
  font-size: 0.75rem;
  font-weight: 400;
  color: #64748b;
}
```

### Applying Audio Constraints

```typescript
async function applyAudioConstraints(
  settings: {
    echoCancellation: boolean;
    noiseSuppression: boolean;
    autoGainControl: boolean;
  },
  deviceId?: string
): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      echoCancellation: settings.echoCancellation,
      noiseSuppression: settings.noiseSuppression,
      autoGainControl: settings.autoGainControl,
    },
  });
}
```

---

## Complete Audio Settings Panel

```tsx
export function AudioSettingsPanel() {
  const {
    inputDevices,
    outputDevices,
    selectedInput,
    selectedOutput,
    setSelectedInput,
    setSelectedOutput,
  } = useAudioDevices();
  
  const [outputVolume, setOutputVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  
  const [processingSettings, setProcessingSettings] = useState({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  });
  
  return (
    <div className="audio-settings-panel">
      <h3>🎧 Audio Settings</h3>
      
      <section>
        <h4>Devices</h4>
        <DeviceSelector
          inputDevices={inputDevices}
          outputDevices={outputDevices}
          selectedInput={selectedInput}
          selectedOutput={selectedOutput}
          onInputChange={setSelectedInput}
          onOutputChange={setSelectedOutput}
        />
      </section>
      
      <section>
        <h4>Volume</h4>
        <VolumeControl
          volume={outputVolume}
          onVolumeChange={setOutputVolume}
          isMuted={isMuted}
          onMuteToggle={() => setIsMuted(!isMuted)}
          label="Output"
        />
      </section>
      
      <section>
        <AudioProcessingSettings
          settings={processingSettings}
          onSettingsChange={setProcessingSettings}
        />
      </section>
    </div>
  );
}
```

---

## Summary

✅ Turn indicators show who's currently speaking with visual feedback

✅ Duration displays help users track speaking time

✅ Device selection uses `enumerateDevices()` and `setSinkId()`

✅ Volume controls manage output level with mute toggle

✅ Audio processing settings configure echo cancellation and noise suppression

✅ All settings should be changeable before and during a session

**Previous:** [Voice-Specific UX](./04-voice-specific-ux.md) | **Back to Overview:** [Voice & Realtime Overview](./00-voice-realtime-overview.md)

---

## Further Reading

- [MDN MediaDevices.enumerateDevices()](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/enumerateDevices) — Device enumeration
- [MDN getUserMedia() constraints](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) — Audio constraints
- [Web Audio API GainNode](https://developer.mozilla.org/en-US/docs/Web/API/GainNode) — Volume control

---

<!-- 
Sources Consulted:
- MDN MediaDevices API: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices
- MDN Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- OpenAI Realtime Console: https://github.com/openai/openai-realtime-console/
-->
