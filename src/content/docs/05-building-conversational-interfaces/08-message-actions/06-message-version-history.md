---
title: "Message Version History"
---

# Message Version History

## Introduction

Message version history enables users to navigate between different responses and edits within a conversation. This creates an explorable conversation tree where users can compare alternatives, restore previous versions, and understand how the conversation evolved.

In this lesson, we'll build version tracking systems, navigation UIs, and diff comparison tools.

### What We'll Cover

- Version data structures
- Version storage strategies
- Navigation UI components
- Side-by-side comparison
- Diff visualization
- Version restoration

### Prerequisites

- [Edit and Resend Messages](./05-edit-resend-messages.md)
- [Regenerate Response](./03-regenerate-response.md)
- Map/Object data structures

---

## Version Data Structures

```typescript
interface MessageVersion {
  id: string;
  content: string;
  createdAt: Date;
  source: 'original' | 'regenerate' | 'edit';
  metadata?: {
    modelId?: string;
    temperature?: number;
    tokenCount?: number;
  };
}

interface VersionedMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  versions: MessageVersion[];
  currentVersionIndex: number;
}
```

---

## Version Storage Strategies

### Strategy 1: In-Message Array

```tsx
// Store versions directly in message object
interface MessageWithVersions extends Message {
  versions: MessageVersion[];
  currentVersionIndex: number;
}

function addVersion(
  message: MessageWithVersions,
  newContent: string,
  source: 'regenerate' | 'edit'
): MessageWithVersions {
  const newVersion: MessageVersion = {
    id: `v_${Date.now()}`,
    content: newContent,
    createdAt: new Date(),
    source
  };
  
  return {
    ...message,
    content: newContent,
    versions: [...message.versions, newVersion],
    currentVersionIndex: message.versions.length
  };
}
```

### Strategy 2: External Version Map

```tsx
// Keep versions in separate Map keyed by message ID
type VersionMap = Map<string, MessageVersion[]>;

function useVersionHistory() {
  const [versionMap, setVersionMap] = useState<VersionMap>(new Map());
  
  const addVersion = (messageId: string, content: string, source: 'regenerate' | 'edit') => {
    setVersionMap(prev => {
      const existing = prev.get(messageId) || [];
      const newVersion: MessageVersion = {
        id: `v_${Date.now()}`,
        content,
        createdAt: new Date(),
        source
      };
      
      return new Map(prev).set(messageId, [...existing, newVersion]);
    });
  };
  
  const getVersions = (messageId: string) => {
    return versionMap.get(messageId) || [];
  };
  
  return { versionMap, addVersion, getVersions };
}
```

### Strategy 3: useReducer for Complex State

```tsx
interface VersionState {
  versions: Map<string, MessageVersion[]>;
  currentIndices: Map<string, number>;
}

type VersionAction = 
  | { type: 'ADD_VERSION'; messageId: string; version: MessageVersion }
  | { type: 'SET_CURRENT'; messageId: string; index: number }
  | { type: 'CLEAR_VERSIONS'; messageId: string };

function versionReducer(state: VersionState, action: VersionAction): VersionState {
  switch (action.type) {
    case 'ADD_VERSION': {
      const existing = state.versions.get(action.messageId) || [];
      const updated = new Map(state.versions);
      updated.set(action.messageId, [...existing, action.version]);
      
      const indices = new Map(state.currentIndices);
      indices.set(action.messageId, existing.length);
      
      return { versions: updated, currentIndices: indices };
    }
    
    case 'SET_CURRENT': {
      const indices = new Map(state.currentIndices);
      indices.set(action.messageId, action.index);
      return { ...state, currentIndices: indices };
    }
    
    case 'CLEAR_VERSIONS': {
      const versions = new Map(state.versions);
      versions.delete(action.messageId);
      const indices = new Map(state.currentIndices);
      indices.delete(action.messageId);
      return { versions, currentIndices: indices };
    }
    
    default:
      return state;
  }
}
```

---

## Complete Version Hook

```tsx
function useMessageVersions() {
  const [state, dispatch] = useReducer(versionReducer, {
    versions: new Map(),
    currentIndices: new Map()
  });
  
  const addVersion = useCallback((
    messageId: string,
    content: string,
    source: 'original' | 'regenerate' | 'edit',
    metadata?: MessageVersion['metadata']
  ) => {
    dispatch({
      type: 'ADD_VERSION',
      messageId,
      version: {
        id: `v_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        content,
        createdAt: new Date(),
        source,
        metadata
      }
    });
  }, []);
  
  const setCurrentVersion = useCallback((messageId: string, index: number) => {
    const versions = state.versions.get(messageId);
    if (versions && index >= 0 && index < versions.length) {
      dispatch({ type: 'SET_CURRENT', messageId, index });
    }
  }, [state.versions]);
  
  const getVersions = useCallback((messageId: string): MessageVersion[] => {
    return state.versions.get(messageId) || [];
  }, [state.versions]);
  
  const getCurrentVersion = useCallback((messageId: string): MessageVersion | null => {
    const versions = state.versions.get(messageId);
    const index = state.currentIndices.get(messageId) ?? 0;
    return versions?.[index] ?? null;
  }, [state.versions, state.currentIndices]);
  
  const getCurrentIndex = useCallback((messageId: string): number => {
    return state.currentIndices.get(messageId) ?? 0;
  }, [state.currentIndices]);
  
  return {
    addVersion,
    setCurrentVersion,
    getVersions,
    getCurrentVersion,
    getCurrentIndex
  };
}
```

---

## Version Navigation UI

```tsx
interface VersionNavigatorProps {
  messageId: string;
  versions: MessageVersion[];
  currentIndex: number;
  onNavigate: (index: number) => void;
}

function VersionNavigator({ 
  messageId, 
  versions, 
  currentIndex, 
  onNavigate 
}: VersionNavigatorProps) {
  if (versions.length <= 1) return null;
  
  const canGoPrev = currentIndex > 0;
  const canGoNext = currentIndex < versions.length - 1;
  
  return (
    <div className="flex items-center gap-2 text-sm text-gray-500">
      <button
        onClick={() => onNavigate(currentIndex - 1)}
        disabled={!canGoPrev}
        className="p-1 hover:bg-gray-100 rounded disabled:opacity-30"
        aria-label="Previous version"
      >
        <ChevronLeftIcon className="w-4 h-4" />
      </button>
      
      <span className="min-w-[60px] text-center">
        {currentIndex + 1} / {versions.length}
      </span>
      
      <button
        onClick={() => onNavigate(currentIndex + 1)}
        disabled={!canGoNext}
        className="p-1 hover:bg-gray-100 rounded disabled:opacity-30"
        aria-label="Next version"
      >
        <ChevronRightIcon className="w-4 h-4" />
      </button>
    </div>
  );
}
```

---

## Enhanced Version Selector

```tsx
function VersionSelector({
  versions,
  currentIndex,
  onSelect
}: {
  versions: MessageVersion[];
  currentIndex: number;
  onSelect: (index: number) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  // Close on click outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  if (versions.length <= 1) return null;
  
  const currentVersion = versions[currentIndex];
  
  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'regenerate': return '🔄';
      case 'edit': return '✏️';
      default: return '📝';
    }
  };
  
  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded"
      >
        <span>{getSourceIcon(currentVersion.source)}</span>
        <span>Version {currentIndex + 1}/{versions.length}</span>
        <ChevronDownIcon className="w-3 h-3" />
      </button>
      
      {isOpen && (
        <div className="absolute top-full mt-1 right-0 bg-white border rounded-lg shadow-lg z-20 min-w-[200px]">
          <div className="p-2 border-b">
            <h4 className="text-xs font-medium text-gray-500">Response Versions</h4>
          </div>
          
          <div className="max-h-64 overflow-y-auto">
            {versions.map((version, index) => (
              <button
                key={version.id}
                onClick={() => {
                  onSelect(index);
                  setIsOpen(false);
                }}
                className={`
                  w-full px-3 py-2 text-left text-sm flex items-center gap-2
                  ${index === currentIndex ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}
                `}
              >
                <span>{getSourceIcon(version.source)}</span>
                <div className="flex-1 min-w-0">
                  <p className="truncate">{version.content.slice(0, 40)}...</p>
                  <p className="text-xs text-gray-400">
                    {version.createdAt.toLocaleTimeString()}
                  </p>
                </div>
                {index === currentIndex && (
                  <CheckIcon className="w-4 h-4 text-blue-500" />
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Side-by-Side Comparison

```tsx
interface VersionComparisonProps {
  versions: MessageVersion[];
  leftIndex: number;
  rightIndex: number;
  onLeftChange: (index: number) => void;
  onRightChange: (index: number) => void;
  onClose: () => void;
}

function VersionComparison({
  versions,
  leftIndex,
  rightIndex,
  onLeftChange,
  onRightChange,
  onClose
}: VersionComparisonProps) {
  const leftVersion = versions[leftIndex];
  const rightVersion = versions[rightIndex];
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-[90vw] max-w-4xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Compare Versions</h2>
          <button 
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <XIcon className="w-5 h-5" />
          </button>
        </div>
        
        {/* Comparison panels */}
        <div className="flex-1 overflow-hidden grid grid-cols-2 divide-x">
          {/* Left panel */}
          <div className="flex flex-col">
            <div className="p-3 border-b bg-gray-50 flex items-center justify-between">
              <select
                value={leftIndex}
                onChange={e => onLeftChange(Number(e.target.value))}
                className="text-sm border rounded px-2 py-1"
              >
                {versions.map((v, i) => (
                  <option key={v.id} value={i}>
                    Version {i + 1} ({v.source})
                  </option>
                ))}
              </select>
              <span className="text-xs text-gray-500">
                {leftVersion.createdAt.toLocaleString()}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <p className="whitespace-pre-wrap">{leftVersion.content}</p>
            </div>
          </div>
          
          {/* Right panel */}
          <div className="flex flex-col">
            <div className="p-3 border-b bg-gray-50 flex items-center justify-between">
              <select
                value={rightIndex}
                onChange={e => onRightChange(Number(e.target.value))}
                className="text-sm border rounded px-2 py-1"
              >
                {versions.map((v, i) => (
                  <option key={v.id} value={i}>
                    Version {i + 1} ({v.source})
                  </option>
                ))}
              </select>
              <span className="text-xs text-gray-500">
                {rightVersion.createdAt.toLocaleString()}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <p className="whitespace-pre-wrap">{rightVersion.content}</p>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t bg-gray-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

## Inline Diff View

```tsx
// Simple word-level diff for text comparison
function computeWordDiff(oldText: string, newText: string): DiffResult[] {
  const oldWords = oldText.split(/\s+/);
  const newWords = newText.split(/\s+/);
  
  // Simple LCS-based diff
  const result: DiffResult[] = [];
  let oldIdx = 0, newIdx = 0;
  
  while (oldIdx < oldWords.length || newIdx < newWords.length) {
    if (oldIdx >= oldWords.length) {
      result.push({ type: 'add', text: newWords[newIdx++] });
    } else if (newIdx >= newWords.length) {
      result.push({ type: 'remove', text: oldWords[oldIdx++] });
    } else if (oldWords[oldIdx] === newWords[newIdx]) {
      result.push({ type: 'same', text: oldWords[oldIdx] });
      oldIdx++; newIdx++;
    } else {
      // Simple heuristic: look ahead to find match
      const lookAhead = newWords.slice(newIdx, newIdx + 5).indexOf(oldWords[oldIdx]);
      if (lookAhead > 0) {
        // Add new words until we match
        for (let i = 0; i < lookAhead; i++) {
          result.push({ type: 'add', text: newWords[newIdx++] });
        }
      } else {
        result.push({ type: 'remove', text: oldWords[oldIdx++] });
        result.push({ type: 'add', text: newWords[newIdx++] });
      }
    }
  }
  
  return result;
}

interface DiffResult {
  type: 'add' | 'remove' | 'same';
  text: string;
}

function DiffView({ oldText, newText }: { oldText: string; newText: string }) {
  const diff = useMemo(() => computeWordDiff(oldText, newText), [oldText, newText]);
  
  return (
    <div className="font-mono text-sm p-4 bg-gray-50 rounded">
      {diff.map((part, i) => (
        <span
          key={i}
          className={
            part.type === 'add' 
              ? 'bg-green-200 text-green-800' 
              : part.type === 'remove'
              ? 'bg-red-200 text-red-800 line-through'
              : ''
          }
        >
          {part.text}{' '}
        </span>
      ))}
    </div>
  );
}
```

---

## Version Timeline

```tsx
function VersionTimeline({ 
  versions, 
  currentIndex,
  onSelect 
}: { 
  versions: MessageVersion[];
  currentIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="p-4 border rounded-lg">
      <h4 className="text-sm font-medium mb-3">Version Timeline</h4>
      
      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-2 top-2 bottom-2 w-0.5 bg-gray-200" />
        
        {/* Version points */}
        <div className="space-y-3">
          {versions.map((version, index) => (
            <button
              key={version.id}
              onClick={() => onSelect(index)}
              className={`
                relative flex items-start gap-3 w-full text-left p-2 rounded
                ${index === currentIndex ? 'bg-blue-50' : 'hover:bg-gray-50'}
              `}
            >
              {/* Dot */}
              <div className={`
                relative z-10 w-4 h-4 rounded-full border-2
                ${index === currentIndex 
                  ? 'bg-blue-500 border-blue-500' 
                  : 'bg-white border-gray-300'
                }
              `} />
              
              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium">
                    {version.source === 'original' ? 'Original' :
                     version.source === 'regenerate' ? 'Regenerated' : 'Edited'}
                  </span>
                  <span className="text-xs text-gray-400">
                    {version.createdAt.toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-gray-600 truncate">
                  {version.content.slice(0, 60)}...
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

## Integrated Message with Versions

```tsx
function VersionedAssistantMessage({ 
  message,
  versions,
  currentIndex,
  onVersionChange,
  onCompare
}: {
  message: Message;
  versions: MessageVersion[];
  currentIndex: number;
  onVersionChange: (index: number) => void;
  onCompare: () => void;
}) {
  const currentVersion = versions[currentIndex] || { content: message.content };
  const hasMultipleVersions = versions.length > 1;
  
  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
        🤖
      </div>
      
      <div className="flex-1">
        {/* Version controls */}
        {hasMultipleVersions && (
          <div className="flex items-center gap-2 mb-2">
            <VersionNavigator
              messageId={message.id}
              versions={versions}
              currentIndex={currentIndex}
              onNavigate={onVersionChange}
            />
            
            <button
              onClick={onCompare}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Compare versions
            </button>
          </div>
        )}
        
        {/* Content */}
        <div className="p-4 bg-gray-100 rounded-lg">
          <p className="whitespace-pre-wrap">{currentVersion.content}</p>
        </div>
        
        {/* Metadata */}
        {currentVersion.metadata && (
          <div className="mt-1 text-xs text-gray-400 flex gap-3">
            {currentVersion.metadata.modelId && (
              <span>{currentVersion.metadata.modelId}</span>
            )}
            {currentVersion.metadata.tokenCount && (
              <span>{currentVersion.metadata.tokenCount} tokens</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Store version metadata | Only store content |
| Limit stored versions | Keep unlimited history |
| Show clear version indicators | Hide version info |
| Provide comparison tools | Force sequential viewing |
| Add timestamps | Omit creation time |
| Enable keyboard navigation | Mouse-only navigation |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Version index out of bounds | Validate before accessing |
| Missing original version | Always store original on first load |
| Confusing version numbers | Use 1-based indexing in UI |
| No way to restore old version | Add "Use this version" button |
| Diff is slow for long content | Limit diff to first N words |

---

## Hands-on Exercise

### Your Task

Build a message version system with:
1. Store versions on regenerate
2. Navigation arrows (prev/next)
3. Version dropdown selector
4. Side-by-side comparison modal
5. Version timeline view

### Requirements

1. Use useReducer for version state
2. Show version source (original/regenerate/edit)
3. Include timestamps
4. Keyboard navigation (arrow keys)

<details>
<summary>💡 Hints (click to expand)</summary>

- Start with the version reducer
- Use Map for O(1) lookups by messageId
- Memoize diff computation
- Add keyboard event listener for arrows

</details>

---

## Summary

✅ **Version storage** via Map or in-message arrays  
✅ **Navigation UI** with prev/next and dropdown  
✅ **Comparison views** for side-by-side analysis  
✅ **Diff visualization** highlights changes  
✅ **Timeline view** shows version history  
✅ **Metadata** provides context (model, tokens, time)

---

## Further Reading

- [diff-match-patch Library](https://github.com/google/diff-match-patch)
- [React useReducer](https://react.dev/reference/react/useReducer)
- [Git-style Diff Visualization](https://developer.github.com/v3/pulls/#list-pull-requests-files)

---

**Previous:** [Edit and Resend Messages](./05-edit-resend-messages.md)  
**Next:** [Delete Messages](./07-delete-messages.md)

<!-- 
Sources Consulted:
- React useReducer: https://react.dev/reference/react/useReducer
- diff-match-patch: https://github.com/google/diff-match-patch
- AI SDK useChat: https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat
-->
