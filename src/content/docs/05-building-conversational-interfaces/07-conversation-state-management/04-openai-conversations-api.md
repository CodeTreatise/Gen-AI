---
title: "OpenAI Conversations API"
---

# OpenAI Conversations API

## Introduction

The OpenAI Conversations API (introduced in 2024) provides server-side conversation management. Instead of manually tracking message arrays, OpenAI stores and manages conversation state, enabling persistent chats, cross-device access, and simplified multi-turn interactions.

In this lesson, we'll implement the Conversations API for stateful chat applications.

### What We'll Cover

- Conversations API fundamentals
- Creating and managing conversations
- Durable conversation IDs
- Cross-session state persistence
- Migration from message arrays

### Prerequisites

- [Conversation History Storage](./03-conversation-history-storage.md)
- OpenAI API basics
- Understanding of multi-turn conversations

---

## Why Server-Side Conversations?

### Traditional Approach

```typescript
// Client must manage message history
const messages = [];

async function chat(userMessage: string) {
  messages.push({ role: 'user', content: userMessage });
  
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages  // Send entire history every time
  });
  
  const assistantMessage = response.choices[0].message;
  messages.push(assistantMessage);
  
  return assistantMessage.content;
}

// Problems:
// - Client stores all history
// - Sent over network each request
// - Lost on page refresh (unless persisted)
// - Token usage grows linearly
```

### Conversations API Approach

```typescript
// Server manages conversation state
let conversationId: string | null = null;

async function chat(userMessage: string) {
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: userMessage,
    conversation: conversationId || undefined,
    store: true
  });
  
  // Store conversation ID for future messages
  conversationId = response.conversation_id;
  
  return response.output_text;
}

// Benefits:
// - No client-side history management
// - Only send new message
// - Persists across sessions
// - OpenAI handles context
```

---

## Creating Conversations

### Basic Conversation Flow

```typescript
import OpenAI from 'openai';

const openai = new OpenAI();

// First message creates conversation
async function startConversation(systemPrompt: string, firstMessage: string) {
  const response = await openai.responses.create({
    model: 'gpt-4o',
    instructions: systemPrompt,
    input: firstMessage,
    store: true  // Enable conversation storage
  });
  
  return {
    conversationId: response.conversation_id,
    responseId: response.id,
    content: response.output_text
  };
}

// Subsequent messages use conversation ID
async function continueConversation(conversationId: string, message: string) {
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: message,
    conversation: conversationId,
    store: true
  });
  
  return {
    responseId: response.id,
    content: response.output_text
  };
}
```

### Complete Chat Session

```typescript
async function chatSession() {
  // Start new conversation
  const { conversationId, content: greeting } = await startConversation(
    'You are a helpful coding assistant.',
    'Hi! I need help with JavaScript.'
  );
  
  console.log('Assistant:', greeting);
  
  // Continue conversation
  const { content: response1 } = await continueConversation(
    conversationId,
    'How do I make an async function?'
  );
  
  console.log('Assistant:', response1);
  
  // Further continuation
  const { content: response2 } = await continueConversation(
    conversationId,
    'Can you show an example with error handling?'
  );
  
  console.log('Assistant:', response2);
  
  // Later session (same conversation)
  const { content: laterResponse } = await continueConversation(
    conversationId,
    'Going back to our earlier discussion about async...'
  );
  
  console.log('Assistant:', laterResponse);  // Has full context
}
```

---

## Conversation Management

### Listing Conversations

```typescript
async function listConversations(limit = 20) {
  const response = await openai.conversations.list({
    limit,
    order: 'desc'  // Most recent first
  });
  
  return response.data.map(conv => ({
    id: conv.id,
    createdAt: new Date(conv.created_at * 1000),
    metadata: conv.metadata
  }));
}
```

### Retrieving Conversation History

```typescript
async function getConversationMessages(conversationId: string) {
  const conversation = await openai.conversations.retrieve(conversationId);
  
  // Get all responses in this conversation
  const responses = await openai.responses.list({
    conversation: conversationId
  });
  
  return responses.data.map(response => ({
    id: response.id,
    input: response.input,
    output: response.output_text,
    createdAt: new Date(response.created_at * 1000)
  }));
}
```

### Deleting Conversations

```typescript
async function deleteConversation(conversationId: string) {
  await openai.conversations.del(conversationId);
}

// Batch delete old conversations
async function cleanupOldConversations(daysOld: number) {
  const cutoff = Date.now() - (daysOld * 24 * 60 * 60 * 1000);
  const conversations = await openai.conversations.list({ limit: 100 });
  
  for (const conv of conversations.data) {
    if (conv.created_at * 1000 < cutoff) {
      await openai.conversations.del(conv.id);
    }
  }
}
```

---

## Conversation Metadata

### Adding Metadata

```typescript
async function createConversationWithMetadata(
  userId: string,
  title: string
) {
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: 'Hello',
    store: true,
    metadata: {
      user_id: userId,
      title: title,
      created_by: 'web_app',
      version: '1.0'
    }
  });
  
  return response.conversation_id;
}
```

### Filtering by Metadata

```typescript
async function getUserConversations(userId: string) {
  const allConversations = await openai.conversations.list({ limit: 100 });
  
  return allConversations.data.filter(
    conv => conv.metadata?.user_id === userId
  );
}
```

### Updating Conversation Title

```typescript
async function updateConversationTitle(
  conversationId: string, 
  newTitle: string
) {
  await openai.conversations.update(conversationId, {
    metadata: {
      title: newTitle,
      updated_at: new Date().toISOString()
    }
  });
}
```

---

## React Integration

### Conversation Hook

```typescript
import { useState, useCallback } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

function useConversation(apiEndpoint: string) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const sendMessage = useCallback(async (content: string) => {
    setIsLoading(true);
    
    // Add user message optimistically
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      content
    };
    setMessages(prev => [...prev, userMessage]);
    
    try {
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          conversationId
        })
      });
      
      const data = await response.json();
      
      // Store conversation ID
      if (!conversationId) {
        setConversationId(data.conversationId);
      }
      
      // Add assistant message
      const assistantMessage: Message = {
        id: data.responseId,
        role: 'assistant',
        content: data.content
      };
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      // Handle error
      setMessages(prev => prev.slice(0, -1));  // Remove optimistic message
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [conversationId, apiEndpoint]);
  
  const loadConversation = useCallback(async (id: string) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiEndpoint}/${id}`);
      const data = await response.json();
      
      setConversationId(id);
      setMessages(data.messages);
    } finally {
      setIsLoading(false);
    }
  }, [apiEndpoint]);
  
  const newConversation = useCallback(() => {
    setConversationId(null);
    setMessages([]);
  }, []);
  
  return {
    conversationId,
    messages,
    isLoading,
    sendMessage,
    loadConversation,
    newConversation
  };
}
```

### API Route (Next.js)

```typescript
// app/api/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI();

export async function POST(request: NextRequest) {
  const { message, conversationId } = await request.json();
  
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: message,
    conversation: conversationId || undefined,
    store: true
  });
  
  return NextResponse.json({
    conversationId: response.conversation_id,
    responseId: response.id,
    content: response.output_text
  });
}
```

---

## Cross-Device State

### Saving Conversation Reference

```typescript
// Save conversation ID to user profile
async function linkConversationToUser(
  userId: string, 
  conversationId: string,
  db: Database
) {
  await db.userConversations.create({
    data: {
      userId,
      openaiConversationId: conversationId,
      createdAt: new Date()
    }
  });
}

// Get user's conversations for any device
async function getUserConversationIds(userId: string, db: Database) {
  const records = await db.userConversations.findMany({
    where: { userId },
    orderBy: { createdAt: 'desc' }
  });
  
  return records.map(r => r.openaiConversationId);
}
```

### Resuming on Different Device

```typescript
async function resumeConversation(
  conversationId: string,
  newMessage: string
) {
  // OpenAI has full history - just continue
  const response = await openai.responses.create({
    model: 'gpt-4o',
    input: newMessage,
    conversation: conversationId,
    store: true
  });
  
  // Works seamlessly - all context preserved
  return response.output_text;
}
```

---

## Migration from Messages Array

### Gradual Migration

```typescript
interface LegacyConversation {
  id: string;
  messages: Array<{ role: string; content: string }>;
}

async function migrateToConversationsAPI(
  legacy: LegacyConversation
): Promise<string> {
  let conversationId: string | null = null;
  
  for (const message of legacy.messages) {
    if (message.role === 'user') {
      const response = await openai.responses.create({
        model: 'gpt-4o',
        input: message.content,
        conversation: conversationId || undefined,
        store: true,
        // Skip actual completion, just store
        max_tokens: 1
      });
      
      if (!conversationId) {
        conversationId = response.conversation_id;
      }
    }
  }
  
  return conversationId!;
}
```

### Hybrid Approach During Migration

```typescript
class ConversationManager {
  private useNewAPI: boolean;
  
  constructor(useNewAPI = true) {
    this.useNewAPI = useNewAPI;
  }
  
  async chat(message: string, context: ConversationContext) {
    if (this.useNewAPI && context.openaiConversationId) {
      return this.chatWithConversationsAPI(message, context.openaiConversationId);
    } else {
      return this.chatWithMessagesArray(message, context.messages);
    }
  }
  
  private async chatWithConversationsAPI(message: string, convId: string) {
    const response = await openai.responses.create({
      model: 'gpt-4o',
      input: message,
      conversation: convId,
      store: true
    });
    
    return {
      content: response.output_text,
      conversationId: response.conversation_id
    };
  }
  
  private async chatWithMessagesArray(
    message: string, 
    messages: Message[]
  ) {
    const allMessages = [
      ...messages,
      { role: 'user' as const, content: message }
    ];
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: allMessages
    });
    
    return {
      content: response.choices[0].message.content,
      newMessages: allMessages.concat([response.choices[0].message])
    };
  }
}
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Store conversation IDs in your DB | Rely only on OpenAI storage |
| Add metadata for filtering | Store sensitive data in metadata |
| Handle conversation not found | Assume conversations exist forever |
| Use `store: true` for persistence | Forget to enable storage |
| Clean up old conversations | Let conversations accumulate |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Lost conversation ID | Persist to database |
| No error handling | Check if conversation exists |
| Mixing approaches | Pick one and stick with it |
| Metadata too large | Keep metadata minimal |
| No user association | Link conversation to user ID |

---

## Hands-on Exercise

### Your Task

Build a chat application using the Conversations API that:
1. Creates new conversations
2. Continues existing conversations
3. Lists user's conversation history
4. Supports conversation deletion

### Requirements

1. Create API routes for CRUD operations
2. Build React hook for state management
3. Store conversation IDs in your database
4. Handle errors gracefully

<details>
<summary>💡 Hints (click to expand)</summary>

- Use metadata to link conversations to users
- Persist conversation ID immediately on creation
- Handle 404 for deleted conversations
- Add loading states for all operations

</details>

---

## Summary

✅ **Conversations API** simplifies state management  
✅ **Durable IDs** enable cross-session persistence  
✅ **Metadata** links conversations to your data  
✅ **No message arrays** to manage  
✅ **Cross-device** works automatically  
✅ **Migration** can be gradual

---

## Further Reading

- [OpenAI Conversations API](https://platform.openai.com/docs/api-reference/conversations)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Building Stateful AI Applications](https://platform.openai.com/docs/guides/stateful-conversations)

---

**Previous:** [Conversation History Storage](./03-conversation-history-storage.md)  
**Next:** [Automatic Context Chaining](./05-automatic-context-chaining.md)

<!-- 
Sources Consulted:
- OpenAI Conversations API: https://platform.openai.com/docs/api-reference/conversations
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Node.js SDK: https://github.com/openai/openai-node
-->
