---
title: "AI-Generated File Display"
---

# AI-Generated File Display

## Introduction

Modern AI assistants don't just generate text—they create code files, images, documents, and structured data. Displaying these generated artifacts requires specialized UI components that handle previews, downloads, and inline editing.

In this lesson, we'll build comprehensive file display components for AI-generated content.

### What We'll Cover

- Code file display with syntax highlighting
- Image generation previews
- Document and data file handling
- Artifact containers and actions
- Live preview and editing
- Download and export functionality

### Prerequisites

- [Message Container Structure](./01-message-container-structure.md)
- [AI Response Styling](./03-ai-response-styling.md)
- Basic file handling concepts

---

## Understanding AI Artifacts

### Types of Generated Content

| Type | Examples | Display Method |
|------|----------|----------------|
| **Code Files** | `.js`, `.py`, `.html`, `.css` | Syntax-highlighted editor |
| **Images** | DALL-E, Midjourney outputs | Gallery with preview |
| **Documents** | Markdown, PDF, DOCX | Rendered preview |
| **Data** | JSON, CSV, XML | Formatted viewer |
| **Diagrams** | Mermaid, PlantUML | Rendered SVG |
| **Audio** | TTS outputs | Audio player |

### Artifact Data Structure

```typescript
interface Artifact {
  id: string;
  type: 'code' | 'image' | 'document' | 'data' | 'diagram' | 'audio';
  filename: string;
  language?: string;        // For code files
  content: string;          // Raw content or base64
  mimeType: string;
  size: number;             // Bytes
  createdAt: string;
  metadata?: {
    prompt?: string;        // Generation prompt
    model?: string;
    dimensions?: { width: number; height: number };
    lineCount?: number;
  };
}
```

---

## Code File Display

### Basic Code Block with File Header

```css
.code-artifact {
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
  margin: 0.75rem 0;
  background: #1e1e1e;
}

.code-artifact-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
}

.code-artifact-file-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.code-artifact-icon {
  width: 1rem;
  height: 1rem;
  color: #858585;
}

.code-artifact-filename {
  font-family: 'Fira Code', monospace;
  font-size: 0.8125rem;
  color: #cccccc;
}

.code-artifact-language {
  padding: 0.125rem 0.5rem;
  background: #404040;
  border-radius: 0.25rem;
  font-size: 0.6875rem;
  color: #9d9d9d;
  text-transform: uppercase;
}

.code-artifact-actions {
  display: flex;
  gap: 0.25rem;
}

.code-artifact-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  background: none;
  border: none;
  border-radius: 0.25rem;
  color: #858585;
  cursor: pointer;
}

.code-artifact-btn:hover {
  background: #404040;
  color: #cccccc;
}

.code-artifact-btn.copied {
  color: #4ade80;
}

.code-artifact-content {
  max-height: 24rem;
  overflow: auto;
}

.code-artifact-content pre {
  margin: 0;
  padding: 1rem;
  font-size: 0.8125rem;
  line-height: 1.6;
}

.code-artifact-content code {
  font-family: 'Fira Code', 'Consolas', monospace;
  color: #d4d4d4;
}

/* Line numbers */
.code-artifact-content .line-number {
  display: inline-block;
  width: 3rem;
  margin-right: 1rem;
  color: #6e6e6e;
  text-align: right;
  user-select: none;
}
```

```html
<div class="code-artifact" data-artifact-id="artifact-123">
  <div class="code-artifact-header">
    <div class="code-artifact-file-info">
      <svg class="code-artifact-icon"><!-- file icon --></svg>
      <span class="code-artifact-filename">auth-handler.js</span>
      <span class="code-artifact-language">JavaScript</span>
    </div>
    <div class="code-artifact-actions">
      <button class="code-artifact-btn" aria-label="Copy code" title="Copy">
        <svg><!-- copy icon --></svg>
      </button>
      <button class="code-artifact-btn" aria-label="Download file" title="Download">
        <svg><!-- download icon --></svg>
      </button>
      <button class="code-artifact-btn" aria-label="Open in editor" title="Edit">
        <svg><!-- edit icon --></svg>
      </button>
    </div>
  </div>
  <div class="code-artifact-content">
    <pre><code class="language-javascript">async function handleAuth(req, res) {
  const { token } = req.headers;
  
  if (!token) {
    return res.status(401).json({ error: 'Missing token' });
  }
  
  // Verify token...
}</code></pre>
  </div>
</div>
```

### Multi-File Artifacts

```css
.multi-file-artifact {
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
}

.multi-file-tabs {
  display: flex;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
  overflow-x: auto;
}

.multi-file-tab {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: #9d9d9d;
  font-family: 'Fira Code', monospace;
  font-size: 0.8125rem;
  cursor: pointer;
  white-space: nowrap;
}

.multi-file-tab:hover {
  background: rgba(255, 255, 255, 0.05);
  color: #cccccc;
}

.multi-file-tab.active {
  background: #1e1e1e;
  border-bottom-color: #3b82f6;
  color: #ffffff;
}

.multi-file-tab-icon {
  width: 1rem;
  height: 1rem;
}

.multi-file-content {
  background: #1e1e1e;
}
```

```jsx
function MultiFileArtifact({ files, activeFile, onFileSelect }) {
  const currentFile = files.find(f => f.id === activeFile);
  
  return (
    <div className="multi-file-artifact">
      <div className="multi-file-tabs" role="tablist">
        {files.map(file => (
          <button
            key={file.id}
            role="tab"
            className={`multi-file-tab ${file.id === activeFile ? 'active' : ''}`}
            aria-selected={file.id === activeFile}
            onClick={() => onFileSelect(file.id)}
          >
            <FileIcon filename={file.filename} />
            <span>{file.filename}</span>
          </button>
        ))}
      </div>
      <div className="multi-file-content" role="tabpanel">
        <CodeArtifact file={currentFile} />
      </div>
    </div>
  );
}
```

---

## Image Generation Display

### Image Artifact Container

```css
.image-artifact {
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
  margin: 0.75rem 0;
  background: white;
}

.image-artifact-preview {
  position: relative;
  aspect-ratio: 1;
  max-height: 32rem;
  overflow: hidden;
  background: #f3f4f6;
}

.image-artifact-preview img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.image-artifact-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  background: #f3f4f6;
}

.image-artifact-spinner {
  width: 2.5rem;
  height: 2.5rem;
  border: 3px solid #e5e7eb;
  border-top-color: #8b5cf6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.image-artifact-progress {
  font-size: 0.875rem;
  color: #6b7280;
}

.image-artifact-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  border-top: 1px solid #e5e7eb;
}

.image-artifact-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.image-artifact-dimensions {
  font-size: 0.75rem;
  color: #6b7280;
}

.image-artifact-prompt {
  font-size: 0.875rem;
  color: #374151;
  max-width: 20rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.image-artifact-actions {
  display: flex;
  gap: 0.5rem;
}

.image-action-btn {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.5rem 0.75rem;
  background: none;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 0.8125rem;
  color: #374151;
  cursor: pointer;
}

.image-action-btn:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

.image-action-btn.primary {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.image-action-btn.primary:hover {
  background: #2563eb;
}
```

```html
<div class="image-artifact" data-artifact-id="img-456">
  <div class="image-artifact-preview">
    <img 
      src="/generated/img-456.png" 
      alt="AI-generated image: A futuristic cityscape at sunset"
      loading="lazy"
    />
  </div>
  <div class="image-artifact-footer">
    <div class="image-artifact-info">
      <span class="image-artifact-prompt" title="A futuristic cityscape at sunset with flying cars">
        A futuristic cityscape at sunset with flying cars
      </span>
      <span class="image-artifact-dimensions">1024 × 1024 · DALL-E 3</span>
    </div>
    <div class="image-artifact-actions">
      <button class="image-action-btn" aria-label="Regenerate image">
        <svg><!-- refresh icon --></svg>
        Regenerate
      </button>
      <button class="image-action-btn primary" aria-label="Download image">
        <svg><!-- download icon --></svg>
        Download
      </button>
    </div>
  </div>
</div>
```

### Image Gallery for Multiple Generations

```css
.image-gallery-artifact {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  padding: 0.5rem;
  background: #f3f4f6;
  border-radius: 0.75rem;
}

.image-gallery-item {
  position: relative;
  aspect-ratio: 1;
  border-radius: 0.5rem;
  overflow: hidden;
  cursor: pointer;
}

.image-gallery-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.2s ease;
}

.image-gallery-item:hover img {
  transform: scale(1.05);
}

.image-gallery-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.image-gallery-item:hover .image-gallery-overlay {
  opacity: 1;
}

.image-gallery-select {
  width: 2.5rem;
  height: 2.5rem;
  background: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-gallery-item.selected {
  outline: 3px solid #3b82f6;
  outline-offset: 2px;
}
```

---

## Document and Data Display

### Markdown Document Preview

```css
.document-artifact {
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  overflow: hidden;
  margin: 0.75rem 0;
}

.document-artifact-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 1rem;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.document-artifact-file {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.document-artifact-icon {
  width: 1.25rem;
  height: 1.25rem;
  color: #6b7280;
}

.document-artifact-name {
  font-weight: 500;
  color: #374151;
}

.document-artifact-meta {
  font-size: 0.75rem;
  color: #9ca3af;
}

.document-artifact-content {
  max-height: 20rem;
  overflow: auto;
  padding: 1rem 1.25rem;
}

.document-artifact-content.rendered {
  /* Rendered markdown styles */
}

.document-artifact-content.raw {
  font-family: 'Fira Code', monospace;
  font-size: 0.875rem;
  white-space: pre-wrap;
  background: #f9fafb;
}

.document-artifact-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: #f9fafb;
  border-top: 1px solid #e5e7eb;
}

.document-view-toggle {
  display: flex;
  background: #e5e7eb;
  border-radius: 0.25rem;
  padding: 0.125rem;
}

.document-view-toggle button {
  padding: 0.375rem 0.75rem;
  background: none;
  border: none;
  border-radius: 0.125rem;
  font-size: 0.75rem;
  color: #6b7280;
  cursor: pointer;
}

.document-view-toggle button.active {
  background: white;
  color: #374151;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
```

### JSON/Data Viewer

```css
.data-artifact {
  font-family: 'Fira Code', monospace;
  font-size: 0.8125rem;
}

.json-viewer {
  padding: 1rem;
  background: #1e1e1e;
  color: #d4d4d4;
  border-radius: 0.5rem;
  overflow: auto;
  max-height: 24rem;
}

.json-key {
  color: #9cdcfe;
}

.json-string {
  color: #ce9178;
}

.json-number {
  color: #b5cea8;
}

.json-boolean {
  color: #569cd6;
}

.json-null {
  color: #569cd6;
}

.json-bracket {
  color: #ffd700;
}

.json-collapsible {
  cursor: pointer;
}

.json-collapsible::before {
  content: '▼';
  display: inline-block;
  margin-right: 0.25rem;
  font-size: 0.625rem;
  transition: transform 0.2s ease;
}

.json-collapsible.collapsed::before {
  transform: rotate(-90deg);
}

.json-collapsible.collapsed + .json-content {
  display: none;
}

.json-collapsed-preview {
  color: #6a9955;
  font-style: italic;
}
```

```jsx
function JsonViewer({ data, defaultExpanded = 2 }) {
  return (
    <div className="json-viewer">
      <JsonNode 
        data={data} 
        depth={0}
        defaultExpanded={defaultExpanded}
      />
    </div>
  );
}

function JsonNode({ data, depth, defaultExpanded }) {
  const [expanded, setExpanded] = useState(depth < defaultExpanded);
  
  if (data === null) {
    return <span className="json-null">null</span>;
  }
  
  if (typeof data === 'string') {
    return <span className="json-string">"{data}"</span>;
  }
  
  if (typeof data === 'number') {
    return <span className="json-number">{data}</span>;
  }
  
  if (typeof data === 'boolean') {
    return <span className="json-boolean">{String(data)}</span>;
  }
  
  if (Array.isArray(data)) {
    return (
      <span>
        <span 
          className={`json-collapsible ${expanded ? '' : 'collapsed'}`}
          onClick={() => setExpanded(!expanded)}
        >
          <span className="json-bracket">[</span>
        </span>
        {expanded ? (
          <div className="json-content" style={{ marginLeft: '1rem' }}>
            {data.map((item, i) => (
              <div key={i}>
                <JsonNode data={item} depth={depth + 1} defaultExpanded={defaultExpanded} />
                {i < data.length - 1 && ','}
              </div>
            ))}
          </div>
        ) : (
          <span className="json-collapsed-preview">
            {data.length} items
          </span>
        )}
        <span className="json-bracket">]</span>
      </span>
    );
  }
  
  // Object
  const entries = Object.entries(data);
  return (
    <span>
      <span 
        className={`json-collapsible ${expanded ? '' : 'collapsed'}`}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="json-bracket">{'{'}</span>
      </span>
      {expanded ? (
        <div className="json-content" style={{ marginLeft: '1rem' }}>
          {entries.map(([key, value], i) => (
            <div key={key}>
              <span className="json-key">"{key}"</span>: 
              <JsonNode data={value} depth={depth + 1} defaultExpanded={defaultExpanded} />
              {i < entries.length - 1 && ','}
            </div>
          ))}
        </div>
      ) : (
        <span className="json-collapsed-preview">
          {entries.length} keys
        </span>
      )}
      <span className="json-bracket">{'}'}</span>
    </span>
  );
}
```

---

## Artifact Container Component

### Unified Artifact Wrapper

```css
.artifact-container {
  position: relative;
  margin: 1rem 0;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
  background: white;
}

.artifact-container.generating {
  border-color: #a5b4fc;
  animation: artifact-pulse 2s ease-in-out infinite;
}

@keyframes artifact-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.2); }
  50% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
}

.artifact-badge {
  position: absolute;
  top: 0.5rem;
  left: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.25rem 0.625rem;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  border-radius: 0.375rem;
  font-size: 0.75rem;
  color: white;
  z-index: 10;
}

.artifact-badge-icon {
  width: 0.875rem;
  height: 0.875rem;
}

.artifact-expand-btn {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  border: none;
  border-radius: 0.375rem;
  color: white;
  cursor: pointer;
  z-index: 10;
}

.artifact-expand-btn:hover {
  background: rgba(0, 0, 0, 0.85);
}
```

```jsx
function ArtifactContainer({ artifact, isGenerating = false }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const renderArtifact = () => {
    switch (artifact.type) {
      case 'code':
        return <CodeArtifact file={artifact} />;
      case 'image':
        return <ImageArtifact image={artifact} />;
      case 'document':
        return <DocumentArtifact document={artifact} />;
      case 'data':
        return <DataArtifact data={artifact} />;
      default:
        return <GenericArtifact artifact={artifact} />;
    }
  };
  
  return (
    <div className={`artifact-container ${isGenerating ? 'generating' : ''}`}>
      <div className="artifact-badge">
        <ArtifactIcon type={artifact.type} />
        <span>{artifact.type}</span>
      </div>
      
      <button 
        className="artifact-expand-btn"
        onClick={() => setIsExpanded(true)}
        aria-label="Expand artifact"
      >
        <ExpandIcon />
      </button>
      
      {renderArtifact()}
      
      {isExpanded && (
        <ArtifactModal 
          artifact={artifact} 
          onClose={() => setIsExpanded(false)} 
        />
      )}
    </div>
  );
}
```

---

## Live Preview and Editing

### Code with Live Preview

```css
.live-preview-artifact {
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 20rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
}

.live-preview-editor {
  display: flex;
  flex-direction: column;
  border-right: 1px solid #e5e7eb;
}

.live-preview-editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.live-preview-editor-content {
  flex: 1;
  overflow: auto;
}

.live-preview-panel {
  display: flex;
  flex-direction: column;
}

.live-preview-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.live-preview-frame {
  flex: 1;
  background: white;
}

.live-preview-frame iframe {
  width: 100%;
  height: 100%;
  border: none;
}

.live-preview-error {
  padding: 1rem;
  background: #fef2f2;
  color: #991b1b;
  font-size: 0.875rem;
}
```

```jsx
function LivePreviewArtifact({ code, language }) {
  const [editedCode, setEditedCode] = useState(code);
  const [previewError, setPreviewError] = useState(null);
  const iframeRef = useRef(null);
  
  const updatePreview = useMemo(() => {
    if (language !== 'html') return null;
    
    return debounce((code) => {
      try {
        const iframe = iframeRef.current;
        const doc = iframe.contentDocument;
        doc.open();
        doc.write(code);
        doc.close();
        setPreviewError(null);
      } catch (error) {
        setPreviewError(error.message);
      }
    }, 500);
  }, [language]);
  
  useEffect(() => {
    if (updatePreview) {
      updatePreview(editedCode);
    }
  }, [editedCode, updatePreview]);
  
  return (
    <div className="live-preview-artifact">
      <div className="live-preview-editor">
        <div className="live-preview-editor-header">
          <span>Code</span>
          <button onClick={() => setEditedCode(code)}>Reset</button>
        </div>
        <div className="live-preview-editor-content">
          <CodeEditor 
            value={editedCode}
            language={language}
            onChange={setEditedCode}
          />
        </div>
      </div>
      
      <div className="live-preview-panel">
        <div className="live-preview-panel-header">
          <span>Preview</span>
          <button onClick={() => updatePreview(editedCode)}>
            Refresh
          </button>
        </div>
        {previewError ? (
          <div className="live-preview-error">{previewError}</div>
        ) : (
          <div className="live-preview-frame">
            <iframe ref={iframeRef} title="Live Preview" sandbox="allow-scripts" />
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## Download and Export

### Download Functionality

```javascript
function downloadArtifact(artifact) {
  let blob;
  let filename = artifact.filename;
  
  switch (artifact.type) {
    case 'code':
    case 'document':
      blob = new Blob([artifact.content], { type: 'text/plain' });
      break;
      
    case 'data':
      if (artifact.mimeType === 'application/json') {
        blob = new Blob(
          [JSON.stringify(JSON.parse(artifact.content), null, 2)],
          { type: 'application/json' }
        );
      } else {
        blob = new Blob([artifact.content], { type: artifact.mimeType });
      }
      break;
      
    case 'image':
      // For base64 images
      const base64Data = artifact.content.replace(/^data:image\/\w+;base64,/, '');
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      blob = new Blob([byteArray], { type: artifact.mimeType });
      break;
      
    default:
      blob = new Blob([artifact.content], { type: 'application/octet-stream' });
  }
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
```

### Export Options Menu

```jsx
function ArtifactExportMenu({ artifact, onExport }) {
  const [isOpen, setIsOpen] = useState(false);
  
  const exportOptions = getExportOptions(artifact.type);
  
  return (
    <div className="export-menu-container">
      <button 
        className="export-menu-trigger"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <DownloadIcon />
        Export
        <ChevronIcon />
      </button>
      
      {isOpen && (
        <div className="export-menu-dropdown">
          {exportOptions.map(option => (
            <button
              key={option.format}
              className="export-menu-item"
              onClick={() => {
                onExport(artifact, option.format);
                setIsOpen(false);
              }}
            >
              <option.icon />
              <span>{option.label}</span>
              <span className="export-menu-ext">.{option.extension}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function getExportOptions(type) {
  const options = {
    code: [
      { format: 'original', label: 'Original', extension: 'js', icon: CodeIcon },
      { format: 'text', label: 'Plain Text', extension: 'txt', icon: TextIcon },
    ],
    image: [
      { format: 'png', label: 'PNG Image', extension: 'png', icon: ImageIcon },
      { format: 'jpg', label: 'JPEG Image', extension: 'jpg', icon: ImageIcon },
      { format: 'webp', label: 'WebP Image', extension: 'webp', icon: ImageIcon },
    ],
    data: [
      { format: 'json', label: 'JSON', extension: 'json', icon: DataIcon },
      { format: 'csv', label: 'CSV', extension: 'csv', icon: TableIcon },
    ],
  };
  
  return options[type] || [
    { format: 'original', label: 'Download', extension: '', icon: FileIcon }
  ];
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use syntax highlighting for code | Display raw code without formatting |
| Show file metadata (size, language) | Hide useful context information |
| Provide copy/download buttons | Force users to select and copy |
| Add loading states during generation | Leave blank while generating |
| Support inline editing when useful | Require external editor for all edits |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No preview for generated images | Always show image inline |
| Large files crash the browser | Limit preview size, virtualize large content |
| Copy button doesn't work on mobile | Use `navigator.clipboard` with fallback |
| Can't download files on iOS Safari | Use proper MIME types and filenames |
| Code preview has no scrolling | Add `overflow: auto` with `max-height` |

---

## Hands-on Exercise

### Your Task

Build an artifact display system that handles:
1. Code files with syntax highlighting
2. Generated images with download
3. JSON data with collapsible tree view
4. Unified artifact container component

### Requirements

1. File header with name, type, and actions
2. Copy and download functionality
3. Expandable/collapsible view
4. Loading state for generation in progress

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `Blob` and `URL.createObjectURL` for downloads
- Add `data-artifact-type` for styling variations
- Implement clipboard API with fallback
- Use `<details>` for collapsible JSON nodes

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

See the complete implementations throughout this lesson, including `CodeArtifact`, `ImageArtifact`, `JsonViewer`, and `ArtifactContainer` components.

</details>

---

## Summary

✅ **Code artifacts** need syntax highlighting and line numbers  
✅ **Image artifacts** require preview, regenerate, and download  
✅ **Data artifacts** benefit from collapsible tree views  
✅ **Unified containers** handle all artifact types consistently  
✅ **Download functionality** uses Blob URLs properly  
✅ **Live preview** enables real-time editing when appropriate

---

## Further Reading

- [Prism.js Syntax Highlighting](https://prismjs.com/)
- [File API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [Clipboard API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API)

---

**Previous:** [Message Grouping Strategies](./06-message-grouping-strategies.md)  
**Next:** [Input Area Components](../03-input-area-components/00-input-area-components.md)

<!-- 
Sources Consulted:
- MDN File API: https://developer.mozilla.org/en-US/docs/Web/API/File_API
- MDN Clipboard API: https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API
- Prism.js: https://prismjs.com/
-->
