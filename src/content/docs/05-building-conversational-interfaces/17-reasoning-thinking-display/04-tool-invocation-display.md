---
title: "Tool Invocation Display"
---

# Tool Invocation Display

## Introduction

When AI models call tools (functions), your interface needs to communicate this clearly to users. Tool invocations go through multiple states: streaming input, waiting for execution, showing results, or displaying errors. Rendering each state appropriately creates a transparent, trustworthy experience.

This lesson covers rendering tool invocation parts in AI SDK, handling the various tool states, displaying arguments, and styling success/error outcomes.

### What We'll Cover

- Tool part types: `tool-invocation` and `tool-result`
- Tool invocation states: `input-streaming`, `input-available`, `output-available`, `output-error`
- Rendering tool name and arguments
- Loading spinners during execution
- Success and error state styling
- Approval workflows for sensitive tools

### Prerequisites

- [Streaming Reasoning Display](./03-streaming-reasoning-display.md)
- AI SDK tool usage basics
- Understanding of function calling concepts

---

## Understanding Tool Parts

### Tool Part Types in AI SDK

AI SDK uses typed tool parts with specific names:

```typescript
// Tool part naming convention
type ToolPartType = 
  | `tool-${toolName}`      // Typed tool part
  | 'dynamic-tool';         // Dynamic/unknown tool

// Example: weather tool creates 'tool-getWeather' parts
```

### Tool States Flow

```mermaid
stateDiagram-v2
    [*] --> input-streaming: Tool called
    input-streaming --> input-available: Arguments complete
    input-available --> approval-requested: Needs approval
    input-available --> output-available: Execution success
    input-available --> output-error: Execution failed
    approval-requested --> output-available: Approved
    approval-requested --> output-error: Denied
    output-available --> [*]
    output-error --> [*]
```

| State | Description | UI Treatment |
|-------|-------------|--------------|
| `input-streaming` | Arguments being generated | Show partial args, loading |
| `input-available` | Arguments complete, executing | Show args, spinner |
| `approval-requested` | Awaiting user approval | Show args, approval buttons |
| `output-available` | Execution complete | Show result |
| `output-error` | Execution failed | Show error message |

---

## Basic Tool Rendering

### Detecting Tool Parts

```tsx
import type { UIMessage } from 'ai';

function renderMessageParts(message: UIMessage) {
  return message.parts.map((part, index) => {
    // Text parts
    if (part.type === 'text') {
      return <TextPart key={index} text={part.text} />;
    }
    
    // Reasoning parts
    if (part.type === 'reasoning') {
      return <ReasoningPart key={index} text={part.text} />;
    }
    
    // Typed tool parts (tool-getWeather, tool-searchDatabase, etc.)
    if (part.type.startsWith('tool-')) {
      return <ToolPart key={index} part={part} />;
    }
    
    // Dynamic tools (runtime-defined)
    if (part.type === 'dynamic-tool') {
      return <DynamicToolPart key={index} part={part} />;
    }
    
    return null;
  });
}
```

### Generic Tool Component

```tsx
interface ToolPartProps {
  part: {
    type: string;
    toolCallId: string;
    toolName: string;
    state: 'input-streaming' | 'input-available' | 'approval-requested' | 'output-available' | 'output-error';
    input?: unknown;
    output?: unknown;
    errorText?: string;
  };
}

export function ToolPart({ part }: ToolPartProps) {
  const { toolName, state, input, output, errorText, toolCallId } = part;
  
  return (
    <div className={`tool-invocation ${state}`}>
      <ToolHeader 
        toolName={toolName}
        state={state}
      />
      
      <ToolBody
        state={state}
        input={input}
        output={output}
        errorText={errorText}
      />
    </div>
  );
}
```

---

## Tool Header with Status

### Header Component

```tsx
interface ToolHeaderProps {
  toolName: string;
  state: string;
}

export function ToolHeader({ toolName, state }: ToolHeaderProps) {
  const displayName = formatToolName(toolName);
  const icon = getToolIcon(toolName);
  const statusInfo = getStatusInfo(state);
  
  return (
    <div className="tool-header">
      <div className="tool-info">
        <span className="tool-icon">{icon}</span>
        <span className="tool-name">{displayName}</span>
      </div>
      
      <div className={`tool-status ${state}`}>
        {statusInfo.icon}
        <span>{statusInfo.label}</span>
      </div>
    </div>
  );
}

function formatToolName(name: string): string {
  // Convert camelCase to Title Case
  return name
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .trim();
}

function getToolIcon(toolName: string): string {
  const icons: Record<string, string> = {
    getWeather: '🌤️',
    searchDatabase: '🔍',
    sendEmail: '📧',
    calculatePrice: '💰',
    getLocation: '📍',
    readFile: '📄',
    writeCode: '💻',
  };
  return icons[toolName] || '🔧';
}

function getStatusInfo(state: string) {
  const statusMap: Record<string, { icon: React.ReactNode; label: string }> = {
    'input-streaming': { icon: <Spinner size="sm" />, label: 'Preparing...' },
    'input-available': { icon: <Spinner size="sm" />, label: 'Running...' },
    'approval-requested': { icon: '⏳', label: 'Awaiting approval' },
    'output-available': { icon: '✓', label: 'Complete' },
    'output-error': { icon: '✕', label: 'Failed' },
  };
  return statusMap[state] || { icon: '?', label: 'Unknown' };
}
```

### Header Styles

```css
.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
  border-radius: 8px 8px 0 0;
}

.tool-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-icon {
  font-size: 1.25rem;
}

.tool-name {
  font-weight: 600;
  color: #334155;
}

.tool-status {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.tool-status.input-streaming,
.tool-status.input-available {
  background: #fef3c7;
  color: #92400e;
}

.tool-status.approval-requested {
  background: #dbeafe;
  color: #1e40af;
}

.tool-status.output-available {
  background: #dcfce7;
  color: #166534;
}

.tool-status.output-error {
  background: #fee2e2;
  color: #991b1b;
}
```

---

## Tool Body Content

### Body Component

```tsx
interface ToolBodyProps {
  state: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
}

export function ToolBody({ state, input, output, errorText }: ToolBodyProps) {
  return (
    <div className="tool-body">
      {/* Input/Arguments section */}
      {input && (
        <CollapsibleArguments 
          args={input} 
          isStreaming={state === 'input-streaming'}
        />
      )}
      
      {/* Output section */}
      {state === 'output-available' && output && (
        <ToolOutput output={output} />
      )}
      
      {/* Error section */}
      {state === 'output-error' && (
        <ToolError errorText={errorText} />
      )}
      
      {/* Loading state */}
      {state === 'input-available' && (
        <ExecutingIndicator />
      )}
    </div>
  );
}
```

### Collapsible Arguments

```tsx
import { useState } from 'react';

interface CollapsibleArgumentsProps {
  args: unknown;
  isStreaming?: boolean;
}

export function CollapsibleArguments({ args, isStreaming }: CollapsibleArgumentsProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const formattedArgs = JSON.stringify(args, null, 2);
  const isLong = formattedArgs.length > 100;
  
  if (!isLong) {
    return (
      <div className="tool-args compact">
        <span className="args-label">Arguments:</span>
        <code>{formattedArgs}</code>
        {isStreaming && <BlinkingCursor />}
      </div>
    );
  }
  
  return (
    <div className="tool-args expandable">
      <button 
        className="args-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span>{isExpanded ? '▼' : '▶'}</span>
        <span>Arguments</span>
        {isStreaming && <span className="streaming-badge">Streaming...</span>}
      </button>
      
      {isExpanded && (
        <pre className="args-content">
          {formattedArgs}
          {isStreaming && <BlinkingCursor />}
        </pre>
      )}
    </div>
  );
}
```

```css
.tool-args {
  padding: 12px 16px;
}

.tool-args.compact {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.args-label {
  font-size: 0.75rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tool-args code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  background: #f1f5f9;
  padding: 4px 8px;
  border-radius: 4px;
}

.args-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 0.8rem;
  color: #64748b;
  padding: 4px 0;
}

.args-toggle:hover {
  color: #334155;
}

.streaming-badge {
  font-size: 0.7rem;
  background: #fef3c7;
  color: #92400e;
  padding: 2px 6px;
  border-radius: 4px;
  animation: pulse 1.5s infinite;
}

.args-content {
  margin-top: 8px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 0.8rem;
  overflow-x: auto;
  max-height: 200px;
  overflow-y: auto;
}
```

---

## Output and Error Display

### Tool Output Component

```tsx
interface ToolOutputProps {
  output: unknown;
}

export function ToolOutput({ output }: ToolOutputProps) {
  // Handle different output types
  if (typeof output === 'string') {
    return (
      <div className="tool-output text">
        <span className="output-label">Result:</span>
        <p>{output}</p>
      </div>
    );
  }
  
  if (typeof output === 'object') {
    return (
      <div className="tool-output json">
        <span className="output-label">Result:</span>
        <pre>{JSON.stringify(output, null, 2)}</pre>
      </div>
    );
  }
  
  return (
    <div className="tool-output primitive">
      <span className="output-label">Result:</span>
      <code>{String(output)}</code>
    </div>
  );
}
```

### Tool Error Component

```tsx
interface ToolErrorProps {
  errorText?: string;
}

export function ToolError({ errorText }: ToolErrorProps) {
  return (
    <div className="tool-error">
      <div className="error-icon">⚠️</div>
      <div className="error-content">
        <span className="error-title">Tool execution failed</span>
        {errorText && (
          <p className="error-message">{errorText}</p>
        )}
      </div>
    </div>
  );
}
```

```css
.tool-output {
  padding: 12px 16px;
  border-top: 1px solid #e2e8f0;
}

.output-label {
  display: block;
  font-size: 0.75rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.tool-output.text p {
  margin: 0;
  color: #166534;
}

.tool-output.json pre {
  margin: 0;
  font-size: 0.8rem;
  background: #f0fdf4;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}

.tool-error {
  display: flex;
  gap: 12px;
  padding: 12px 16px;
  background: #fef2f2;
  border-top: 1px solid #fecaca;
}

.error-icon {
  font-size: 1.25rem;
}

.error-title {
  font-weight: 600;
  color: #991b1b;
}

.error-message {
  margin: 4px 0 0;
  font-size: 0.875rem;
  color: #b91c1c;
}
```

---

## Loading and Execution States

### Executing Indicator

```tsx
export function ExecutingIndicator() {
  return (
    <div className="executing-indicator">
      <div className="spinner" />
      <span>Executing tool...</span>
    </div>
  );
}
```

```css
.executing-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px;
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  border-top: 1px solid #fde68a;
}

.executing-indicator .spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #fcd34d;
  border-top-color: #f59e0b;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.executing-indicator span {
  color: #92400e;
  font-size: 0.875rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

---

## Tool Approval Workflows

### Approval UI

For tools that require user confirmation:

```tsx
import { useChat } from '@ai-sdk/react';

export function ToolWithApproval() {
  const { messages, addToolApprovalResponse } = useChat();
  
  return (
    <div>
      {messages.map(message => (
        <div key={message.id}>
          {message.parts.map((part) => {
            // Check for tool parts needing approval
            if (part.type.startsWith('tool-') && part.state === 'approval-requested') {
              return (
                <ApprovalCard
                  key={part.toolCallId}
                  toolName={part.toolName}
                  input={part.input}
                  approvalId={part.approval.id}
                  onApprove={() => 
                    addToolApprovalResponse({ id: part.approval.id, approved: true })
                  }
                  onDeny={() =>
                    addToolApprovalResponse({ id: part.approval.id, approved: false })
                  }
                />
              );
            }
            // ... other part rendering
          })}
        </div>
      ))}
    </div>
  );
}
```

### Approval Card Component

```tsx
interface ApprovalCardProps {
  toolName: string;
  input: unknown;
  approvalId: string;
  onApprove: () => void;
  onDeny: () => void;
}

export function ApprovalCard({
  toolName,
  input,
  onApprove,
  onDeny,
}: ApprovalCardProps) {
  const displayName = formatToolName(toolName);
  
  return (
    <div className="approval-card">
      <div className="approval-header">
        <span className="approval-icon">🔐</span>
        <span className="approval-title">Approval Required</span>
      </div>
      
      <div className="approval-content">
        <p className="approval-message">
          The AI wants to execute <strong>{displayName}</strong>
        </p>
        
        <details className="approval-args">
          <summary>View arguments</summary>
          <pre>{JSON.stringify(input, null, 2)}</pre>
        </details>
      </div>
      
      <div className="approval-actions">
        <button 
          className="btn-deny"
          onClick={onDeny}
        >
          Deny
        </button>
        <button 
          className="btn-approve"
          onClick={onApprove}
        >
          Approve
        </button>
      </div>
    </div>
  );
}
```

```css
.approval-card {
  border: 2px solid #3b82f6;
  border-radius: 12px;
  overflow: hidden;
  margin: 12px 0;
}

.approval-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #eff6ff;
  border-bottom: 1px solid #bfdbfe;
}

.approval-icon {
  font-size: 1.25rem;
}

.approval-title {
  font-weight: 600;
  color: #1e40af;
}

.approval-content {
  padding: 16px;
}

.approval-message {
  margin: 0 0 12px;
  color: #334155;
}

.approval-args summary {
  cursor: pointer;
  color: #64748b;
  font-size: 0.875rem;
}

.approval-args pre {
  margin-top: 8px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 0.8rem;
}

.approval-actions {
  display: flex;
  gap: 12px;
  padding: 12px 16px;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
}

.btn-deny {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  color: #64748b;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-deny:hover {
  background: #fee2e2;
  border-color: #fecaca;
  color: #991b1b;
}

.btn-approve {
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 8px;
  background: #3b82f6;
  color: white;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-approve:hover {
  background: #2563eb;
}
```

---

## Complete Tool Invocation Example

### Full Implementation

```tsx
import type { UIMessage } from 'ai';
import { useState } from 'react';

interface ToolInvocationCardProps {
  part: {
    type: string;
    toolCallId: string;
    toolName: string;
    state: string;
    input?: unknown;
    output?: unknown;
    errorText?: string;
    approval?: { id: string };
  };
  onApprove?: (id: string) => void;
  onDeny?: (id: string) => void;
}

export function ToolInvocationCard({
  part,
  onApprove,
  onDeny,
}: ToolInvocationCardProps) {
  const [showArgs, setShowArgs] = useState(false);
  const { toolName, state, input, output, errorText, approval } = part;
  
  const icon = getToolIcon(toolName);
  const displayName = formatToolName(toolName);
  
  return (
    <div className={`tool-card state-${state}`}>
      {/* Header */}
      <div className="tool-header">
        <div className="tool-identity">
          <span className="tool-icon">{icon}</span>
          <span className="tool-name">{displayName}</span>
        </div>
        <StatusBadge state={state} />
      </div>
      
      {/* Arguments */}
      {input && (
        <div className="tool-arguments">
          <button
            className="args-toggle"
            onClick={() => setShowArgs(!showArgs)}
          >
            {showArgs ? '▼' : '▶'} Arguments
          </button>
          {showArgs && (
            <pre className="args-json">
              {JSON.stringify(input, null, 2)}
            </pre>
          )}
        </div>
      )}
      
      {/* State-specific content */}
      {state === 'input-streaming' && (
        <div className="tool-loading">
          <Spinner /> Preparing arguments...
        </div>
      )}
      
      {state === 'input-available' && (
        <div className="tool-loading">
          <Spinner /> Executing...
        </div>
      )}
      
      {state === 'approval-requested' && approval && (
        <div className="tool-approval">
          <p>This tool requires your approval to run.</p>
          <div className="approval-buttons">
            <button onClick={() => onDeny?.(approval.id)}>
              Deny
            </button>
            <button onClick={() => onApprove?.(approval.id)}>
              Approve
            </button>
          </div>
        </div>
      )}
      
      {state === 'output-available' && (
        <div className="tool-output">
          <span className="output-label">Result</span>
          <div className="output-content">
            {typeof output === 'string' ? (
              <p>{output}</p>
            ) : (
              <pre>{JSON.stringify(output, null, 2)}</pre>
            )}
          </div>
        </div>
      )}
      
      {state === 'output-error' && (
        <div className="tool-error">
          <span className="error-icon">⚠️</span>
          <span>{errorText || 'An error occurred'}</span>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ state }: { state: string }) {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    'input-streaming': { bg: '#fef3c7', text: '#92400e', label: 'Preparing' },
    'input-available': { bg: '#fef3c7', text: '#92400e', label: 'Running' },
    'approval-requested': { bg: '#dbeafe', text: '#1e40af', label: 'Approval needed' },
    'output-available': { bg: '#dcfce7', text: '#166534', label: 'Complete' },
    'output-error': { bg: '#fee2e2', text: '#991b1b', label: 'Failed' },
  };
  
  const { bg, text, label } = config[state] || { bg: '#f1f5f9', text: '#64748b', label: 'Unknown' };
  
  return (
    <span 
      className="status-badge"
      style={{ background: bg, color: text }}
    >
      {label}
    </span>
  );
}

function Spinner() {
  return <span className="spinner" />;
}
```

---

## Summary

✅ Tool parts have typed names like `tool-getWeather` for type safety

✅ Tools go through states: `input-streaming` → `input-available` → `output-available/output-error`

✅ Use collapsible JSON display for tool arguments

✅ Show clear loading indicators during tool execution

✅ Style success and error states distinctly (green for success, red for errors)

✅ Implement approval workflows for sensitive tools with `addToolApprovalResponse`

**Next:** [Cost & Token Visualization](./05-cost-token-visualization.md)

---

## Further Reading

- [AI SDK Chatbot Tool Usage](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) — Complete tool documentation
- [Tool Execution Approval](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling#tool-execution-approval) — Approval patterns
- [Dynamic Tools](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage#dynamic-tools) — Runtime-defined tools

---

<!-- 
Sources Consulted:
- AI SDK Chatbot Tool Usage: https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage
- AI SDK Tool Calling: https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling
-->
