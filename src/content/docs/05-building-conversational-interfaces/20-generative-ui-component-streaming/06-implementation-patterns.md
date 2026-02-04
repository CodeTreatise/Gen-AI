---
title: "Implementation Patterns"
---

# Implementation Patterns

## Introduction

This lesson covers essential patterns for production generative UI: tool definitions with Zod schemas, component registries for dynamic lookup, fallback rendering for unknown tools, error handling strategies, and hydration considerations.

### What We'll Cover

- Tool definitions with Zod schemas
- Component registry architecture
- Fallback rendering for unknown tools
- Error handling strategies
- Dynamic tools and MCP integration
- Hydration and serialization

### Prerequisites

- [Generative UI Use Cases](./05-generative-ui-use-cases.md)
- Zod schema validation
- React Suspense basics

---

## Tool Definitions with Zod

Zod schemas define what the AI can generate:

### Schema Design Patterns

```typescript
// ai/tools.ts
import { tool } from 'ai';
import { z } from 'zod';

// Base schemas for reuse
const colorSchema = z.enum(['primary', 'secondary', 'success', 'warning', 'error']);

const sizeSchema = z.enum(['sm', 'md', 'lg']);

const iconSchema = z.enum([
  'info', 'warning', 'error', 'success', 
  'user', 'settings', 'home', 'search'
]);

// Component-specific schemas
const buttonSchema = z.object({
  label: z.string().max(50),
  variant: colorSchema.default('primary'),
  size: sizeSchema.default('md'),
  icon: iconSchema.optional(),
  action: z.string().describe('Action identifier for click handler'),
});

const alertSchema = z.object({
  title: z.string(),
  message: z.string(),
  severity: z.enum(['info', 'warning', 'error', 'success']),
  dismissible: z.boolean().default(true),
});

// Define tools
export const tools = {
  showButton: tool({
    description: 'Display a clickable button for user action',
    inputSchema: buttonSchema,
    // Client-side tool - no execute function
  }),

  showAlert: tool({
    description: 'Show an alert message to the user',
    inputSchema: alertSchema,
    execute: async (params) => {
      // Could log alerts, send notifications, etc.
      return { shown: true, ...params };
    },
  }),
};

// Export types for components
export type ButtonToolInput = z.infer<typeof buttonSchema>;
export type AlertToolInput = z.infer<typeof alertSchema>;
```

### Complex Nested Schemas

```typescript
// Hierarchical data structures
const addressSchema = z.object({
  street: z.string(),
  city: z.string(),
  state: z.string(),
  zip: z.string(),
  country: z.string().default('US'),
});

const contactSchema = z.object({
  id: z.string(),
  name: z.string(),
  email: z.string().email(),
  phone: z.string().optional(),
  address: addressSchema.optional(),
  tags: z.array(z.string()).default([]),
});

export const tools = {
  showContactCard: tool({
    description: 'Display a contact card',
    inputSchema: z.object({
      contact: contactSchema,
      actions: z.array(z.enum(['call', 'email', 'edit', 'delete'])).default(['email']),
    }),
    execute: async ({ contact }) => {
      return await getContactDetails(contact.id);
    },
  }),

  showContactList: tool({
    description: 'Display a list of contacts',
    inputSchema: z.object({
      contacts: z.array(contactSchema),
      groupBy: z.enum(['name', 'company', 'tag']).optional(),
      limit: z.number().max(50).default(10),
    }),
  }),
};
```

### Discriminated Unions

```typescript
// Different component types with shared base
const baseCardSchema = z.object({
  id: z.string(),
  title: z.string(),
});

const articleCardSchema = baseCardSchema.extend({
  type: z.literal('article'),
  content: z.string(),
  author: z.string(),
  publishedAt: z.string(),
});

const videoCardSchema = baseCardSchema.extend({
  type: z.literal('video'),
  url: z.string().url(),
  duration: z.number(),
  thumbnail: z.string().url(),
});

const imageCardSchema = baseCardSchema.extend({
  type: z.literal('image'),
  src: z.string().url(),
  alt: z.string(),
  width: z.number().optional(),
  height: z.number().optional(),
});

const cardSchema = z.discriminatedUnion('type', [
  articleCardSchema,
  videoCardSchema,
  imageCardSchema,
]);

export const tools = {
  showCard: tool({
    description: 'Display a content card (article, video, or image)',
    inputSchema: cardSchema,
  }),
};
```

---

## Component Registry Pattern

Map tool names to components dynamically:

### Registry Architecture

```typescript
// components/registry.ts
import { ComponentType } from 'react';

// Define the registry interface
export interface ToolComponent<T = any> {
  component: ComponentType<T>;
  skeleton?: ComponentType;
  errorFallback?: ComponentType<{ error: Error }>;
}

type ToolRegistry = Map<string, ToolComponent>;

// Create the registry
const registry: ToolRegistry = new Map();

// Registration function
export function registerTool<T>(
  name: string,
  config: ToolComponent<T>
): void {
  registry.set(name, config);
}

// Lookup function
export function getToolComponent(name: string): ToolComponent | undefined {
  return registry.get(name);
}

// Check if tool is registered
export function hasToolComponent(name: string): boolean {
  return registry.has(name);
}

// Get all registered tool names
export function getRegisteredTools(): string[] {
  return Array.from(registry.keys());
}
```

### Component Registration

```typescript
// components/register-tools.ts
import { registerTool } from './registry';
import { WeatherCard, WeatherSkeleton } from './weather-card';
import { ProductCard, ProductSkeleton } from './product-card';
import { ChartDisplay, ChartSkeleton } from './chart-display';
import { BookingWidget, BookingSkeleton } from './booking-widget';
import { DataTable, TableSkeleton } from './data-table';
import { DynamicForm, FormSkeleton } from './dynamic-form';

// Register all tool components
export function registerAllTools(): void {
  registerTool('getWeather', {
    component: WeatherCard,
    skeleton: WeatherSkeleton,
  });

  registerTool('showProduct', {
    component: ProductCard,
    skeleton: ProductSkeleton,
  });

  registerTool('showChart', {
    component: ChartDisplay,
    skeleton: ChartSkeleton,
  });

  registerTool('showBookingWidget', {
    component: BookingWidget,
    skeleton: BookingSkeleton,
  });

  registerTool('showDataTable', {
    component: DataTable,
    skeleton: TableSkeleton,
  });

  registerTool('showForm', {
    component: DynamicForm,
    skeleton: FormSkeleton,
  });
}

// Call on app initialization
registerAllTools();
```

### Using the Registry

```tsx
// components/tool-renderer.tsx
import { getToolComponent, hasToolComponent } from './registry';
import { UnknownToolFallback } from './unknown-tool-fallback';

interface ToolRendererProps {
  toolName: string;
  input: Record<string, any>;
  output?: Record<string, any>;
  state: 'streaming' | 'available' | 'error';
  error?: Error;
  onAction?: (action: string, data: any) => void;
}

export function ToolRenderer({
  toolName,
  input,
  output,
  state,
  error,
  onAction,
}: ToolRendererProps) {
  // Check if tool is registered
  if (!hasToolComponent(toolName)) {
    return <UnknownToolFallback toolName={toolName} input={input} />;
  }

  const toolConfig = getToolComponent(toolName)!;
  const { component: Component, skeleton: Skeleton, errorFallback: ErrorFallback } = toolConfig;

  // Handle streaming state
  if (state === 'streaming' && Skeleton) {
    return <Skeleton />;
  }

  // Handle error state
  if (state === 'error') {
    if (ErrorFallback && error) {
      return <ErrorFallback error={error} />;
    }
    return <div className="tool-error">Failed to load {toolName}</div>;
  }

  // Render the component with merged props
  const props = { ...input, ...output, onAction };
  return <Component {...props} />;
}
```

---

## Fallback Rendering

Handle unknown or unregistered tools gracefully:

### Unknown Tool Fallback

```tsx
// components/unknown-tool-fallback.tsx
import { useState } from 'react';

interface UnknownToolFallbackProps {
  toolName: string;
  input: Record<string, any>;
  output?: Record<string, any>;
}

export function UnknownToolFallback({
  toolName,
  input,
  output,
}: UnknownToolFallbackProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="unknown-tool">
      <div className="unknown-tool-header">
        <span className="icon">🔧</span>
        <span className="name">{toolName}</span>
        <button
          className="toggle"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Hide' : 'Show'} details
        </button>
      </div>

      {expanded && (
        <div className="unknown-tool-details">
          <div className="section">
            <h4>Input</h4>
            <pre>{JSON.stringify(input, null, 2)}</pre>
          </div>
          {output && (
            <div className="section">
              <h4>Output</h4>
              <pre>{JSON.stringify(output, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

```css
.unknown-tool {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
}

.unknown-tool-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.unknown-tool-header .icon {
  font-size: 1.25rem;
}

.unknown-tool-header .name {
  font-family: monospace;
  font-weight: 500;
}

.unknown-tool-header .toggle {
  margin-left: auto;
  background: none;
  border: none;
  color: #d97706;
  cursor: pointer;
  font-size: 0.75rem;
}

.unknown-tool-details {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #fcd34d;
}

.unknown-tool-details h4 {
  margin: 0 0 8px;
  font-size: 0.75rem;
  color: #92400e;
}

.unknown-tool-details pre {
  background: white;
  padding: 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  overflow-x: auto;
  margin: 0 0 12px;
}
```

### Graceful Degradation

```tsx
// components/tool-part.tsx
import { ToolRenderer } from './tool-renderer';
import { UnknownToolFallback } from './unknown-tool-fallback';
import { ToolErrorBoundary } from './tool-error-boundary';

interface ToolPartProps {
  toolName: string;
  input: Record<string, any>;
  output?: Record<string, any>;
  state: string;
  onAction?: (action: string, data: any) => void;
}

export function ToolPart({ toolName, input, output, state, onAction }: ToolPartProps) {
  return (
    <ToolErrorBoundary
      fallback={(error) => (
        <div className="tool-crash">
          <p>Component crashed: {toolName}</p>
          <details>
            <summary>Error details</summary>
            <pre>{error.message}</pre>
          </details>
        </div>
      )}
    >
      <ToolRenderer
        toolName={toolName}
        input={input}
        output={output}
        state={state as any}
        onAction={onAction}
      />
    </ToolErrorBoundary>
  );
}
```

---

## Error Handling

Comprehensive error handling for generative UI:

### Error Boundary for Tools

```tsx
// components/tool-error-boundary.tsx
import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback: (error: Error) => ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ToolErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    error: null,
  };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('Tool component error:', error, info);
    // Could send to error tracking service
  }

  render() {
    if (this.state.hasError && this.state.error) {
      return this.props.fallback(this.state.error);
    }
    return this.props.children;
  }
}
```

### Stream Error Handling

```typescript
// app/api/chat/route.ts
import { streamText, toUIMessageStreamResponse } from 'ai';
import { openai } from '@ai-sdk/openai';
import { tools } from '@/ai/tools';

export async function POST(request: Request) {
  const { messages } = await request.json();

  const result = streamText({
    model: openai('gpt-4o'),
    messages,
    tools,
    maxSteps: 5,
    onError: (error) => {
      console.error('Stream error:', error);
      // Could trigger alerting here
    },
    onToolCall: ({ toolCall }) => {
      console.log('Tool called:', toolCall.toolName);
    },
  });

  return result.toUIMessageStreamResponse({
    onError: (error) => {
      // This error is sent to the client
      console.error('Response error:', error);
      return {
        type: 'error',
        message: 'An error occurred processing your request.',
      };
    },
  });
}
```

### Client-Side Error Handling

```tsx
// components/chat.tsx
import { useChat } from '@ai-sdk/react';

export function Chat() {
  const { messages, error, isLoading, reload } = useChat({
    onError: (error) => {
      console.error('Chat error:', error);
      // Show toast notification
    },
    onToolCall: async ({ toolCall }) => {
      try {
        // Handle client-side tool
        return await executeClientTool(toolCall);
      } catch (error) {
        console.error('Client tool error:', error);
        return { error: 'Tool execution failed' };
      }
    },
  });

  if (error) {
    return (
      <div className="chat-error">
        <p>Something went wrong</p>
        <button onClick={() => reload()}>Try again</button>
      </div>
    );
  }

  // ... render messages
}
```

---

## Dynamic Tools (MCP Integration)

Support tools discovered at runtime:

### Dynamic Tool Registration

```typescript
// lib/dynamic-tools.ts
import { tool } from 'ai';
import { z } from 'zod';

interface MCPTool {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
}

// Convert MCP tool to AI SDK tool
function convertMCPTool(mcpTool: MCPTool) {
  // Convert JSON Schema to Zod (simplified)
  const zodSchema = jsonSchemaToZod(mcpTool.inputSchema);

  return tool({
    description: mcpTool.description,
    inputSchema: zodSchema,
    execute: async (params) => {
      // Call MCP server to execute
      return await executeMCPTool(mcpTool.name, params);
    },
  });
}

// Fetch and register tools from MCP server
export async function loadMCPTools(
  serverUrl: string
): Promise<Record<string, ReturnType<typeof tool>>> {
  const response = await fetch(`${serverUrl}/tools`);
  const mcpTools: MCPTool[] = await response.json();

  const tools: Record<string, ReturnType<typeof tool>> = {};

  for (const mcpTool of mcpTools) {
    tools[mcpTool.name] = convertMCPTool(mcpTool);
  }

  return tools;
}
```

### Runtime Component Loading

```tsx
// components/dynamic-tool-renderer.tsx
import { lazy, Suspense, useState, useEffect, ComponentType } from 'react';

interface DynamicToolRendererProps {
  toolName: string;
  input: Record<string, any>;
  output?: Record<string, any>;
  componentUrl?: string; // URL to load component from
}

// Cache for dynamically loaded components
const componentCache = new Map<string, ComponentType<any>>();

export function DynamicToolRenderer({
  toolName,
  input,
  output,
  componentUrl,
}: DynamicToolRendererProps) {
  const [Component, setComponent] = useState<ComponentType<any> | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);

  useEffect(() => {
    // Check cache first
    if (componentCache.has(toolName)) {
      setComponent(() => componentCache.get(toolName)!);
      return;
    }

    // Try to load dynamically
    if (componentUrl) {
      loadRemoteComponent(componentUrl)
        .then((LoadedComponent) => {
          componentCache.set(toolName, LoadedComponent);
          setComponent(() => LoadedComponent);
        })
        .catch(setLoadError);
    }
  }, [toolName, componentUrl]);

  if (loadError) {
    return <div>Failed to load component: {loadError.message}</div>;
  }

  if (!Component) {
    // Render JSON fallback
    return (
      <div className="json-fallback">
        <pre>{JSON.stringify({ input, output }, null, 2)}</pre>
      </div>
    );
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Component {...input} {...output} />
    </Suspense>
  );
}

// Load a remote component (e.g., from CDN or MCP server)
async function loadRemoteComponent(url: string): Promise<ComponentType<any>> {
  const module = await import(/* webpackIgnore: true */ url);
  return module.default;
}
```

---

## Hydration and Serialization

Handle server-to-client transitions:

### Serializable Props Only

```typescript
// ai/tools.ts
// ✅ Good: All serializable
export const tools = {
  showChart: tool({
    inputSchema: z.object({
      type: z.string(),
      data: z.array(z.object({
        label: z.string(),
        value: z.number(),
      })),
    }),
  }),
};

// ❌ Bad: Functions, Dates as objects, etc.
const badTool = tool({
  inputSchema: z.object({
    onClick: z.function(), // Can't serialize!
    date: z.date(), // Becomes string
  }),
});
```

### Handling Non-Serializable Data

```tsx
// components/chart-display.tsx
interface ChartDisplayProps {
  type: string;
  data: { label: string; value: number }[];
  // Callbacks are passed via context, not props
}

// Use context for callbacks
import { useGeneratedUIContext } from './generated-ui-context';

export function ChartDisplay({ type, data }: ChartDisplayProps) {
  const { onAction } = useGeneratedUIContext();

  const handleClick = (item: { label: string; value: number }) => {
    // Call through context instead of prop
    onAction?.('chartClick', item);
  };

  return (
    <div onClick={() => handleClick(data[0])}>
      {/* Chart content */}
    </div>
  );
}
```

### Context Provider Pattern

```tsx
// components/generated-ui-context.tsx
import { createContext, useContext, ReactNode } from 'react';

interface GeneratedUIContextValue {
  onAction?: (action: string, data: any) => void;
  addMessage?: (content: string) => void;
  theme?: 'light' | 'dark';
}

const GeneratedUIContext = createContext<GeneratedUIContextValue>({});

export function GeneratedUIProvider({
  children,
  onAction,
  addMessage,
  theme = 'light',
}: GeneratedUIContextValue & { children: ReactNode }) {
  return (
    <GeneratedUIContext.Provider value={{ onAction, addMessage, theme }}>
      {children}
    </GeneratedUIContext.Provider>
  );
}

export function useGeneratedUIContext() {
  return useContext(GeneratedUIContext);
}
```

### Using the Context

```tsx
// components/chat.tsx
import { GeneratedUIProvider } from './generated-ui-context';
import { MessageList } from './message-list';

export function Chat() {
  const { messages, append } = useChat();

  const handleAction = (action: string, data: any) => {
    // Handle actions from generated components
    console.log('Action:', action, data);

    // Could send a new message
    if (action === 'askMore') {
      append({ role: 'user', content: data.query });
    }
  };

  return (
    <GeneratedUIProvider
      onAction={handleAction}
      addMessage={(content) => append({ role: 'user', content })}
    >
      <MessageList messages={messages} />
    </GeneratedUIProvider>
  );
}
```

---

## Hands-on Exercise

### Your Task

Build a component registry with fallback rendering:

### Requirements

1. Create a registry with at least 3 tool components
2. Implement the `ToolRenderer` that uses the registry
3. Create an `UnknownToolFallback` for unregistered tools
4. Wrap everything in an error boundary
5. Test with both registered and unregistered tool names

<details>
<summary>💡 Hints</summary>

- Use a `Map` for O(1) lookup performance
- Export type-safe registration functions
- Include skeleton components for loading states
- Test error boundary by throwing in a component

</details>

<details>
<summary>✅ Solution</summary>

```typescript
// registry.ts
type ToolConfig = {
  component: React.ComponentType<any>;
  skeleton?: React.ComponentType;
};

const registry = new Map<string, ToolConfig>();

export function register(name: string, config: ToolConfig) {
  registry.set(name, config);
}

export function get(name: string) {
  return registry.get(name);
}

export function has(name: string) {
  return registry.has(name);
}
```

```tsx
// ToolRenderer.tsx
export function ToolRenderer({ name, props, state }: Props) {
  if (!has(name)) {
    return <UnknownToolFallback name={name} props={props} />;
  }

  const { component: Component, skeleton: Skeleton } = get(name)!;

  if (state === 'streaming' && Skeleton) {
    return <Skeleton />;
  }

  return (
    <ToolErrorBoundary fallback={(e) => <ErrorDisplay error={e} />}>
      <Component {...props} />
    </ToolErrorBoundary>
  );
}
```

</details>

---

## Summary

✅ Zod schemas define tool inputs with validation and types

✅ Component registries provide dynamic tool-to-component mapping

✅ Fallback rendering handles unknown tools gracefully

✅ Error boundaries prevent component crashes from breaking the UI

✅ Dynamic tools enable MCP and runtime tool discovery

✅ Context providers pass callbacks without serialization issues

**Next:** [Unit 6: Prompt Engineering](../../06-prompt-engineering/00-overview.md)

---

## Further Reading

- [Zod Documentation](https://zod.dev/) — Schema validation library
- [AI SDK Tools Reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool) — Tool function API
- [React Error Boundaries](https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary) — Error handling patterns
- [Model Context Protocol](https://modelcontextprotocol.io/) — Dynamic tool discovery

---

<!-- 
Sources Consulted:
- AI SDK Tools: https://ai-sdk.dev/docs/reference/ai-sdk-core/tool
- AI SDK Generative UI: https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces
- Zod documentation: https://zod.dev/
- React Error Boundaries: https://react.dev/reference/react/Component
-->
