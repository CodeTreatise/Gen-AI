---
title: "Artifact Display Patterns"
---

# Artifact Display Patterns

## Introduction

Artifacts are rich, self-contained outputs that deserve special UI treatment—code files, documents, diagrams, or any content that benefits from expanded viewing, versioning, and export options. Claude's artifact system popularized this pattern, and it's become a standard for AI interfaces.

This lesson covers how to build artifact containers that enhance user experience when displaying complex generated content.

### What We'll Cover

- Collapsible artifact containers
- Full-screen artifact expansion
- Artifact type icons and headers
- Version history tracking
- Download and export options
- Side-by-side comparison
- Artifact editing and regeneration

### Prerequisites

- [Streaming React Components](./01-streaming-react-components.md)
- Basic React state management
- CSS layout (flexbox, grid)

---

## Artifact Container Component

### Basic Structure

```tsx
interface Artifact {
  id: string;
  type: 'code' | 'document' | 'image' | 'chart' | 'table' | 'diagram';
  title: string;
  content: string | React.ReactNode;
  language?: string;  // For code artifacts
  createdAt: Date;
  version: number;
}

interface ArtifactContainerProps {
  artifact: Artifact;
  versions?: Artifact[];
  onRegenerate?: () => void;
  onEdit?: (content: string) => void;
}
```

### Container Implementation

```tsx
import { useState } from 'react';
import { 
  ChevronDown, 
  ChevronUp, 
  Maximize2, 
  Download, 
  Copy, 
  RotateCcw,
  History 
} from 'lucide-react';

export function ArtifactContainer({ 
  artifact, 
  versions = [],
  onRegenerate,
  onEdit 
}: ArtifactContainerProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showVersions, setShowVersions] = useState(false);

  const Icon = getArtifactIcon(artifact.type);

  return (
    <div className={`artifact-container ${isFullscreen ? 'fullscreen' : ''}`}>
      {/* Header */}
      <div className="artifact-header">
        <div className="artifact-title">
          <Icon className="artifact-icon" />
          <span>{artifact.title}</span>
          <span className="artifact-version">v{artifact.version}</span>
        </div>

        <div className="artifact-actions">
          {versions.length > 1 && (
            <button 
              onClick={() => setShowVersions(!showVersions)}
              aria-label="Show versions"
            >
              <History size={16} />
            </button>
          )}
          <button onClick={() => copyToClipboard(artifact)} aria-label="Copy">
            <Copy size={16} />
          </button>
          <button onClick={() => downloadArtifact(artifact)} aria-label="Download">
            <Download size={16} />
          </button>
          <button 
            onClick={() => setIsFullscreen(!isFullscreen)}
            aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            <Maximize2 size={16} />
          </button>
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {/* Version History Dropdown */}
      {showVersions && (
        <VersionHistory 
          versions={versions} 
          currentId={artifact.id}
          onSelect={(v) => {/* handle version select */}}
        />
      )}

      {/* Content */}
      {isExpanded && (
        <div className="artifact-content">
          <ArtifactContent artifact={artifact} />
        </div>
      )}

      {/* Footer with actions */}
      {isExpanded && (
        <div className="artifact-footer">
          {onRegenerate && (
            <button onClick={onRegenerate} className="artifact-action-btn">
              <RotateCcw size={14} />
              Regenerate
            </button>
          )}
          {onEdit && artifact.type === 'code' && (
            <button 
              onClick={() => {/* open editor */}} 
              className="artifact-action-btn"
            >
              Edit
            </button>
          )}
        </div>
      )}

      {/* Fullscreen Overlay */}
      {isFullscreen && (
        <div className="fullscreen-overlay" onClick={() => setIsFullscreen(false)} />
      )}
    </div>
  );
}
```

### Styling

```css
.artifact-container {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: white;
  margin: 16px 0;
  overflow: hidden;
  transition: all 0.2s ease;
}

.artifact-container:hover {
  border-color: #cbd5e1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.artifact-container.fullscreen {
  position: fixed;
  top: 16px;
  left: 16px;
  right: 16px;
  bottom: 16px;
  z-index: 1000;
  border-radius: 16px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
}

.fullscreen-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 999;
}

.artifact-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.artifact-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: #334155;
}

.artifact-icon {
  width: 18px;
  height: 18px;
  color: #64748b;
}

.artifact-version {
  font-size: 0.75rem;
  color: #94a3b8;
  background: #e2e8f0;
  padding: 2px 6px;
  border-radius: 4px;
}

.artifact-actions {
  display: flex;
  gap: 4px;
}

.artifact-actions button {
  padding: 6px;
  border: none;
  background: transparent;
  border-radius: 6px;
  cursor: pointer;
  color: #64748b;
  transition: all 0.15s;
}

.artifact-actions button:hover {
  background: #e2e8f0;
  color: #334155;
}

.artifact-content {
  padding: 16px;
  max-height: 400px;
  overflow: auto;
}

.artifact-container.fullscreen .artifact-content {
  max-height: calc(100vh - 150px);
}

.artifact-footer {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
}

.artifact-action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  font-size: 0.875rem;
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 6px;
  cursor: pointer;
  color: #475569;
  transition: all 0.15s;
}

.artifact-action-btn:hover {
  background: #f1f5f9;
  border-color: #cbd5e1;
}
```

---

## Artifact Type Icons

```tsx
import { 
  Code, 
  FileText, 
  Image, 
  BarChart2, 
  Table, 
  GitBranch 
} from 'lucide-react';

export function getArtifactIcon(type: Artifact['type']) {
  const icons = {
    code: Code,
    document: FileText,
    image: Image,
    chart: BarChart2,
    table: Table,
    diagram: GitBranch,
  };
  
  return icons[type] || FileText;
}

export function getArtifactColor(type: Artifact['type']) {
  const colors = {
    code: '#3b82f6',      // blue
    document: '#8b5cf6',  // purple
    image: '#10b981',     // green
    chart: '#f59e0b',     // amber
    table: '#06b6d4',     // cyan
    diagram: '#ec4899',   // pink
  };
  
  return colors[type] || '#64748b';
}
```

### Type-Specific Headers

```tsx
function ArtifactHeader({ artifact }: { artifact: Artifact }) {
  const Icon = getArtifactIcon(artifact.type);
  const color = getArtifactColor(artifact.type);

  return (
    <div className="artifact-header" style={{ borderLeftColor: color }}>
      <div className="artifact-title">
        <div 
          className="artifact-icon-wrapper" 
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon style={{ color }} size={16} />
        </div>
        <span>{artifact.title}</span>
        {artifact.language && (
          <span className="artifact-language">{artifact.language}</span>
        )}
      </div>
      {/* ... actions */}
    </div>
  );
}
```

```css
.artifact-header {
  border-left: 3px solid transparent;
}

.artifact-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
}

.artifact-language {
  font-size: 0.75rem;
  color: #64748b;
  background: #f1f5f9;
  padding: 2px 8px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
}
```

---

## Version History

### Version History Component

```tsx
interface VersionHistoryProps {
  versions: Artifact[];
  currentId: string;
  onSelect: (artifact: Artifact) => void;
  onCompare?: (v1: Artifact, v2: Artifact) => void;
}

export function VersionHistory({ 
  versions, 
  currentId, 
  onSelect,
  onCompare 
}: VersionHistoryProps) {
  const [compareMode, setCompareMode] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);

  const sortedVersions = [...versions].sort((a, b) => b.version - a.version);

  const handleVersionClick = (artifact: Artifact) => {
    if (compareMode) {
      if (selected.includes(artifact.id)) {
        setSelected(selected.filter(id => id !== artifact.id));
      } else if (selected.length < 2) {
        const newSelected = [...selected, artifact.id];
        setSelected(newSelected);
        
        if (newSelected.length === 2 && onCompare) {
          const [v1, v2] = newSelected.map(
            id => versions.find(v => v.id === id)!
          );
          onCompare(v1, v2);
        }
      }
    } else {
      onSelect(artifact);
    }
  };

  return (
    <div className="version-history">
      <div className="version-header">
        <span>Version History</span>
        {onCompare && (
          <button 
            onClick={() => {
              setCompareMode(!compareMode);
              setSelected([]);
            }}
            className={compareMode ? 'active' : ''}
          >
            Compare
          </button>
        )}
      </div>

      <div className="version-list">
        {sortedVersions.map((version) => (
          <div
            key={version.id}
            className={`version-item ${
              version.id === currentId ? 'current' : ''
            } ${selected.includes(version.id) ? 'selected' : ''}`}
            onClick={() => handleVersionClick(version)}
          >
            <div className="version-info">
              <span className="version-number">v{version.version}</span>
              <span className="version-date">
                {formatRelativeTime(version.createdAt)}
              </span>
            </div>
            {version.id === currentId && (
              <span className="current-badge">Current</span>
            )}
            {compareMode && selected.includes(version.id) && (
              <span className="compare-badge">
                {selected.indexOf(version.id) + 1}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
}
```

```css
.version-history {
  background: white;
  border-bottom: 1px solid #e2e8f0;
  padding: 12px 16px;
}

.version-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-size: 0.875rem;
  font-weight: 500;
  color: #475569;
}

.version-header button {
  font-size: 0.75rem;
  padding: 4px 8px;
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 4px;
  cursor: pointer;
}

.version-header button.active {
  background: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

.version-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.version-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}

.version-item:hover {
  background: #f1f5f9;
}

.version-item.current {
  background: #eff6ff;
}

.version-item.selected {
  background: #dbeafe;
  border: 1px solid #3b82f6;
}

.version-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.version-number {
  font-weight: 500;
  color: #334155;
}

.version-date {
  font-size: 0.75rem;
  color: #94a3b8;
}

.current-badge,
.compare-badge {
  font-size: 0.625rem;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}

.current-badge {
  background: #dbeafe;
  color: #2563eb;
}

.compare-badge {
  background: #3b82f6;
  color: white;
}
```

---

## Download and Export

```tsx
async function downloadArtifact(artifact: Artifact) {
  const { content, type, title, language } = artifact;
  
  let blob: Blob;
  let filename: string;
  
  switch (type) {
    case 'code':
      const ext = getExtensionForLanguage(language || 'txt');
      blob = new Blob([content as string], { type: 'text/plain' });
      filename = `${sanitizeFilename(title)}.${ext}`;
      break;
      
    case 'document':
      blob = new Blob([content as string], { type: 'text/markdown' });
      filename = `${sanitizeFilename(title)}.md`;
      break;
      
    case 'image':
      // Assume base64 or URL
      const response = await fetch(content as string);
      blob = await response.blob();
      filename = `${sanitizeFilename(title)}.png`;
      break;
      
    default:
      blob = new Blob([JSON.stringify(content, null, 2)], { 
        type: 'application/json' 
      });
      filename = `${sanitizeFilename(title)}.json`;
  }
  
  // Trigger download
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function copyToClipboard(artifact: Artifact) {
  const text = typeof artifact.content === 'string' 
    ? artifact.content 
    : JSON.stringify(artifact.content, null, 2);
    
  navigator.clipboard.writeText(text);
}

function getExtensionForLanguage(language: string): string {
  const extensions: Record<string, string> = {
    javascript: 'js',
    typescript: 'ts',
    python: 'py',
    html: 'html',
    css: 'css',
    json: 'json',
    markdown: 'md',
    sql: 'sql',
    bash: 'sh',
    yaml: 'yml',
  };
  return extensions[language.toLowerCase()] || 'txt';
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
}
```

---

## Side-by-Side Comparison

```tsx
interface ComparisonViewProps {
  left: Artifact;
  right: Artifact;
  onClose: () => void;
}

export function ComparisonView({ left, right, onClose }: ComparisonViewProps) {
  return (
    <div className="comparison-overlay">
      <div className="comparison-container">
        <div className="comparison-header">
          <h3>Comparing Versions</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="comparison-content">
          <div className="comparison-pane">
            <div className="pane-header">
              <span>v{left.version}</span>
              <span className="pane-date">
                {left.createdAt.toLocaleString()}
              </span>
            </div>
            <div className="pane-content">
              <ArtifactContent artifact={left} />
            </div>
          </div>

          <div className="comparison-divider" />

          <div className="comparison-pane">
            <div className="pane-header">
              <span>v{right.version}</span>
              <span className="pane-date">
                {right.createdAt.toLocaleString()}
              </span>
            </div>
            <div className="pane-content">
              <ArtifactContent artifact={right} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

```css
.comparison-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.comparison-container {
  width: 90vw;
  max-width: 1400px;
  height: 80vh;
  background: white;
  border-radius: 16px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.comparison-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.comparison-header h3 {
  margin: 0;
  font-size: 1.125rem;
}

.comparison-content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.comparison-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.comparison-divider {
  width: 1px;
  background: #e2e8f0;
}

.pane-header {
  display: flex;
  justify-content: space-between;
  padding: 12px 16px;
  background: #f1f5f9;
  font-weight: 500;
}

.pane-date {
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 400;
}

.pane-content {
  flex: 1;
  overflow: auto;
  padding: 16px;
}
```

---

## Artifact Content Renderer

```tsx
function ArtifactContent({ artifact }: { artifact: Artifact }) {
  switch (artifact.type) {
    case 'code':
      return (
        <CodeBlock 
          code={artifact.content as string} 
          language={artifact.language || 'text'} 
        />
      );

    case 'document':
      return (
        <div className="prose">
          <ReactMarkdown>{artifact.content as string}</ReactMarkdown>
        </div>
      );

    case 'image':
      return (
        <img 
          src={artifact.content as string} 
          alt={artifact.title}
          className="artifact-image"
        />
      );

    case 'chart':
      return <ChartRenderer data={artifact.content} />;

    case 'table':
      return <TableRenderer data={artifact.content} />;

    case 'diagram':
      return <DiagramRenderer content={artifact.content as string} />;

    default:
      return <pre>{JSON.stringify(artifact.content, null, 2)}</pre>;
  }
}
```

---

## Summary

✅ Artifact containers provide enhanced viewing for rich AI outputs

✅ Collapsible headers keep chat interfaces clean

✅ Fullscreen mode enables detailed examination

✅ Type-specific icons help users identify content quickly

✅ Version history enables tracking and comparison

✅ Download/export options make artifacts actionable

**Next:** [Server-Side Generation](./03-server-side-generation.md)

---

## Further Reading

- [Claude Artifacts](https://support.anthropic.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) — Original artifact pattern
- [React Portal](https://react.dev/reference/react-dom/createPortal) — For fullscreen overlays
- [Diff Libraries](https://github.com/kpdecker/jsdiff) — For version comparison

---

<!-- 
Sources Consulted:
- Claude Artifacts UI patterns
- AI SDK Generative UI: https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces
- shadcn/ui component patterns
-->
