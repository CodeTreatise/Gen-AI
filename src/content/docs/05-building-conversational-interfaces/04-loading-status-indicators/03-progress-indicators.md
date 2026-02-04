---
title: "Progress Indicators"
---

# Progress Indicators

## Introduction

When AI operations take more than a few seconds—file processing, complex analysis, multi-step generation—users need more than a spinner. Progress indicators show how far along an operation is and how much longer it might take.

In this lesson, we'll build progress indicators from simple bars to detailed step-based workflows.

### What We'll Cover

- Percentage progress bars
- Step-based progress
- Time estimates
- Indeterminate spinners
- Combined progress patterns

### Prerequisites

- [Typing Indicators](./01-typing-indicators.md)
- CSS transitions and animations
- React state management

---

## Percentage Progress Bars

### Basic Progress Bar

```jsx
function ProgressBar({ value, max = 100 }) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div 
      className="progress-bar-container"
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label="Operation progress"
    >
      <div 
        className="progress-bar-fill"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
```

```css
.progress-bar-container {
  width: 100%;
  height: 8px;
  background: var(--progress-bg, #e0e0e0);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: var(--progress-color, #007bff);
  border-radius: 4px;
  transition: width 0.3s ease-out;
}
```

### Animated Fill

```css
.progress-bar-fill {
  background: linear-gradient(
    90deg,
    var(--progress-color, #007bff),
    var(--progress-highlight, #4da3ff),
    var(--progress-color, #007bff)
  );
  background-size: 200% 100%;
  animation: progress-shimmer 1.5s linear infinite;
}

@keyframes progress-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Progress with Label

```jsx
function LabeledProgressBar({ value, max = 100, label }) {
  const percentage = Math.round((value / max) * 100);
  
  return (
    <div className="labeled-progress">
      <div className="progress-header">
        <span className="progress-label">{label}</span>
        <span className="progress-percentage">{percentage}%</span>
      </div>
      <ProgressBar value={value} max={max} />
    </div>
  );
}
```

```css
.labeled-progress {
  width: 100%;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 14px;
}

.progress-label {
  color: var(--text-primary);
}

.progress-percentage {
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}
```

### Segmented Progress

```jsx
function SegmentedProgress({ segments, currentSegment }) {
  return (
    <div className="segmented-progress" role="progressbar">
      {segments.map((segment, i) => (
        <div
          key={i}
          className={`segment ${i < currentSegment ? 'complete' : ''} ${i === currentSegment ? 'active' : ''}`}
          title={segment.label}
        />
      ))}
    </div>
  );
}
```

```css
.segmented-progress {
  display: flex;
  gap: 4px;
  width: 100%;
}

.segment {
  flex: 1;
  height: 6px;
  background: var(--segment-bg, #e0e0e0);
  border-radius: 3px;
  transition: background 0.3s ease;
}

.segment.complete {
  background: var(--progress-color, #007bff);
}

.segment.active {
  background: var(--progress-color, #007bff);
  animation: segment-pulse 1s ease-in-out infinite;
}

@keyframes segment-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
```

---

## Step-Based Progress

### Vertical Stepper

```jsx
function StepProgress({ steps, currentStep }) {
  return (
    <div className="step-progress" role="list" aria-label="Progress steps">
      {steps.map((step, i) => (
        <div 
          key={i}
          className={`step ${i < currentStep ? 'complete' : ''} ${i === currentStep ? 'active' : ''}`}
          role="listitem"
        >
          <div className="step-indicator">
            {i < currentStep ? (
              <span className="step-check">✓</span>
            ) : (
              <span className="step-number">{i + 1}</span>
            )}
          </div>
          <div className="step-content">
            <span className="step-label">{step.label}</span>
            {step.description && (
              <span className="step-description">{step.description}</span>
            )}
          </div>
          {i < steps.length - 1 && <div className="step-connector" />}
        </div>
      ))}
    </div>
  );
}
```

```css
.step-progress {
  display: flex;
  flex-direction: column;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  position: relative;
  padding-bottom: 24px;
}

.step:last-child {
  padding-bottom: 0;
}

.step-indicator {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--step-bg, #e0e0e0);
  color: var(--text-secondary);
  font-weight: 600;
  flex-shrink: 0;
  z-index: 1;
}

.step.complete .step-indicator {
  background: var(--success-color, #28a745);
  color: white;
}

.step.active .step-indicator {
  background: var(--progress-color, #007bff);
  color: white;
  animation: step-pulse 1.5s ease-in-out infinite;
}

.step-connector {
  position: absolute;
  left: 15px;
  top: 32px;
  bottom: 0;
  width: 2px;
  background: var(--step-bg, #e0e0e0);
}

.step.complete .step-connector {
  background: var(--success-color, #28a745);
}

.step-content {
  display: flex;
  flex-direction: column;
  padding-top: 4px;
}

.step-label {
  font-weight: 500;
  color: var(--text-primary);
}

.step-description {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 2px;
}
```

### Horizontal Stepper

```jsx
function HorizontalStepper({ steps, currentStep }) {
  return (
    <div className="horizontal-stepper">
      {steps.map((step, i) => (
        <React.Fragment key={i}>
          <div className={`h-step ${i <= currentStep ? 'active' : ''}`}>
            <div className="h-step-indicator">
              {i < currentStep ? '✓' : i + 1}
            </div>
            <span className="h-step-label">{step.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div className={`h-step-line ${i < currentStep ? 'complete' : ''}`} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
```

```css
.horizontal-stepper {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.h-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.h-step-indicator {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--step-bg, #e0e0e0);
  font-size: 12px;
  font-weight: 600;
}

.h-step.active .h-step-indicator {
  background: var(--progress-color, #007bff);
  color: white;
}

.h-step-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.h-step.active .h-step-label {
  color: var(--text-primary);
}

.h-step-line {
  flex: 1;
  height: 2px;
  background: var(--step-bg, #e0e0e0);
  margin: 0 8px;
}

.h-step-line.complete {
  background: var(--progress-color, #007bff);
}
```

---

## Time Estimates

### Countdown Timer

```jsx
function TimeEstimate({ estimatedSeconds }) {
  const [remaining, setRemaining] = useState(estimatedSeconds);
  
  useEffect(() => {
    if (remaining <= 0) return;
    
    const timer = setInterval(() => {
      setRemaining(prev => Math.max(0, prev - 1));
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);
  
  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };
  
  return (
    <span className="time-estimate">
      ~{formatTime(remaining)} remaining
    </span>
  );
}
```

### Dynamic Time Estimate

```jsx
function DynamicTimeEstimate({ progress, startTime }) {
  const [estimate, setEstimate] = useState(null);
  
  useEffect(() => {
    if (progress <= 0) return;
    
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = progress / elapsed;
    const remaining = (100 - progress) / rate;
    
    setEstimate(Math.round(remaining));
  }, [progress, startTime]);
  
  if (!estimate || estimate < 1) return null;
  
  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds} seconds`;
    if (seconds < 3600) return `${Math.ceil(seconds / 60)} minutes`;
    return `${Math.ceil(seconds / 3600)} hours`;
  };
  
  return (
    <span className="time-estimate">
      About {formatTime(estimate)} remaining
    </span>
  );
}
```

### Progress with Time

```jsx
function ProgressWithTime({ value, startTime, label }) {
  return (
    <div className="progress-with-time">
      <div className="progress-header">
        <span className="progress-label">{label}</span>
        <DynamicTimeEstimate progress={value} startTime={startTime} />
      </div>
      <ProgressBar value={value} />
      <div className="progress-footer">
        <span className="progress-percentage">{Math.round(value)}% complete</span>
      </div>
    </div>
  );
}
```

---

## Indeterminate Spinners

### CSS Spinner

```jsx
function Spinner({ size = 24, color }) {
  return (
    <div 
      className="spinner"
      style={{ 
        width: size, 
        height: size,
        borderColor: color 
      }}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
}
```

```css
.spinner {
  border: 3px solid var(--spinner-track, #e0e0e0);
  border-top-color: var(--spinner-color, #007bff);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Dots Spinner

```jsx
function DotsSpinner() {
  return (
    <div className="dots-spinner" role="status">
      <span></span>
      <span></span>
      <span></span>
    </div>
  );
}
```

```css
.dots-spinner {
  display: flex;
  gap: 4px;
}

.dots-spinner span {
  width: 8px;
  height: 8px;
  background: var(--spinner-color, #007bff);
  border-radius: 50%;
  animation: dots-bounce 1.4s ease-in-out infinite;
}

.dots-spinner span:nth-child(1) { animation-delay: 0s; }
.dots-spinner span:nth-child(2) { animation-delay: 0.2s; }
.dots-spinner span:nth-child(3) { animation-delay: 0.4s; }

@keyframes dots-bounce {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}
```

### Indeterminate Progress Bar

```jsx
function IndeterminateProgress() {
  return (
    <div className="indeterminate-bar" role="progressbar" aria-label="Loading">
      <div className="indeterminate-fill" />
    </div>
  );
}
```

```css
.indeterminate-bar {
  width: 100%;
  height: 4px;
  background: var(--progress-bg, #e0e0e0);
  border-radius: 2px;
  overflow: hidden;
}

.indeterminate-fill {
  width: 40%;
  height: 100%;
  background: var(--progress-color, #007bff);
  border-radius: 2px;
  animation: indeterminate 1.5s ease-in-out infinite;
}

@keyframes indeterminate {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(350%); }
}
```

---

## Combined Progress Patterns

### Operation Progress Card

```jsx
function OperationProgress({ 
  title,
  status,  // 'pending' | 'processing' | 'complete' | 'error'
  progress,
  steps,
  currentStep,
  onCancel
}) {
  return (
    <div className={`operation-card ${status}`}>
      <div className="operation-header">
        <h3 className="operation-title">{title}</h3>
        {status === 'processing' && onCancel && (
          <button onClick={onCancel} className="cancel-btn">Cancel</button>
        )}
      </div>
      
      {progress !== undefined ? (
        <ProgressBar value={progress} />
      ) : (
        <IndeterminateProgress />
      )}
      
      {steps && (
        <div className="operation-steps">
          <HorizontalStepper steps={steps} currentStep={currentStep} />
        </div>
      )}
      
      <div className="operation-footer">
        {status === 'processing' && progress !== undefined && (
          <DynamicTimeEstimate progress={progress} startTime={Date.now()} />
        )}
        {status === 'complete' && (
          <span className="status-complete">✓ Complete</span>
        )}
        {status === 'error' && (
          <span className="status-error">✗ Failed</span>
        )}
      </div>
    </div>
  );
}
```

### Chat with Progress

```jsx
function ChatWithProgress({ messages, status, operationProgress }) {
  return (
    <div className="chat">
      <div className="messages">
        {messages.map(msg => (
          <Message key={msg.id} {...msg} />
        ))}
        
        {operationProgress && (
          <div className="message system">
            <OperationProgress {...operationProgress} />
          </div>
        )}
      </div>
    </div>
  );
}
```

### Multi-Stage Processing

```jsx
function MultiStageProgress({ stages }) {
  const completedStages = stages.filter(s => s.status === 'complete').length;
  const currentStage = stages.find(s => s.status === 'processing');
  
  return (
    <div className="multi-stage-progress">
      <div className="overall-progress">
        <ProgressBar 
          value={completedStages} 
          max={stages.length} 
        />
        <span>{completedStages} of {stages.length} complete</span>
      </div>
      
      <div className="stages">
        {stages.map((stage, i) => (
          <div key={i} className={`stage ${stage.status}`}>
            <div className="stage-header">
              <span className="stage-icon">
                {stage.status === 'complete' ? '✓' : 
                 stage.status === 'processing' ? <Spinner size={16} /> : 
                 '○'}
              </span>
              <span className="stage-name">{stage.name}</span>
            </div>
            
            {stage.status === 'processing' && stage.progress !== undefined && (
              <ProgressBar value={stage.progress} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Accessibility

### ARIA Attributes

```jsx
function AccessibleProgress({ value, max, label }) {
  return (
    <div
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label={label}
      aria-live="polite"
    >
      <div 
        className="progress-fill"
        style={{ width: `${(value / max) * 100}%` }}
      />
      <span className="sr-only">
        {Math.round((value / max) * 100)}% complete
      </span>
    </div>
  );
}
```

### Screen Reader Announcements

```jsx
function ProgressWithAnnouncements({ value, stages }) {
  const prevValueRef = useRef(value);
  const [announcement, setAnnouncement] = useState('');
  
  useEffect(() => {
    // Announce at key milestones
    if (value >= 25 && prevValueRef.current < 25) {
      setAnnouncement('25% complete');
    } else if (value >= 50 && prevValueRef.current < 50) {
      setAnnouncement('Halfway there');
    } else if (value >= 75 && prevValueRef.current < 75) {
      setAnnouncement('75% complete');
    } else if (value >= 100 && prevValueRef.current < 100) {
      setAnnouncement('Complete');
    }
    
    prevValueRef.current = value;
  }, [value]);
  
  return (
    <div>
      <ProgressBar value={value} />
      <div aria-live="polite" className="sr-only">
        {announcement}
      </div>
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Show accurate progress when possible | Fake progress percentages |
| Use indeterminate for unknown duration | Show stuck progress bar |
| Provide time estimates when possible | Leave users guessing |
| Allow cancellation for long operations | Force users to wait |
| Announce milestones to screen readers | Leave progress silent |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Progress jumps backward | Only allow forward movement |
| Estimate wildly inaccurate | Base on actual elapsed time |
| Progress stuck at 99% | Reserve final 5% for cleanup |
| No feedback for stuck operation | Add timeout with retry option |
| Progress too fast then stalls | Smooth the progress curve |

---

## Hands-on Exercise

### Your Task

Create a multi-step operation progress component that:
1. Shows overall percentage
2. Displays individual steps
3. Estimates remaining time
4. Allows cancellation

### Requirements

1. Step indicators (complete, active, pending)
2. Progress bar with percentage
3. Dynamic time estimate
4. Cancel button

<details>
<summary>💡 Hints (click to expand)</summary>

- Track `startTime` with `useState` on mount
- Calculate rate from progress/elapsed
- Map steps to determine current step
- Disable cancel when complete

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the `OperationProgress` component in the Combined Progress Patterns section above.

</details>

---

## Summary

✅ **Progress bars** show completion percentage  
✅ **Step indicators** show discrete stages  
✅ **Time estimates** set expectations  
✅ **Indeterminate spinners** for unknown duration  
✅ **Combined patterns** for complex operations  
✅ **ARIA attributes** ensure accessibility

---

## Further Reading

- [ARIA Progressbar](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/progressbar_role)
- [Progress Element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/progress)
- [UX of Progress Indicators](https://www.nngroup.com/articles/progress-indicators/)

---

**Previous:** [Skeleton Loading States](./02-skeleton-loading-states.md)  
**Next:** [Status Messages](./04-status-messages.md)

<!-- 
Sources Consulted:
- MDN ARIA progressbar: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/progressbar_role
- MDN progress element: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/progress
- Nielsen Norman Group: https://www.nngroup.com/articles/progress-indicators/
-->
