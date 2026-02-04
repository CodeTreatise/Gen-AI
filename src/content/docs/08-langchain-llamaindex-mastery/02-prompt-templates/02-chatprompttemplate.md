---
title: "ChatPromptTemplate"
---

# ChatPromptTemplate

## Introduction

`ChatPromptTemplate` is the primary template class for working with modern chat models like GPT-4, Claude, and Gemini. Unlike `PromptTemplate` which produces a single string, `ChatPromptTemplate` produces a **list of messages** with distinct roles—enabling rich, multi-turn conversations.

Chat models expect messages with roles (system, human, ai) rather than raw text, making `ChatPromptTemplate` essential for production applications.

### What We'll Cover

- Creating templates with `from_messages()`
- Message types: SystemMessage, HumanMessage, AIMessage
- The tuple syntax shorthand
- MessagesPlaceholder for dynamic messages
- Multi-turn conversation templates
- Message prompt template classes

### Prerequisites

- PromptTemplate basics (Lesson 8.2.1)
- Understanding of chat model APIs

---

## Creating ChatPromptTemplates

### Using `from_messages()`

The recommended way to create a `ChatPromptTemplate`:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who speaks {language}."),
    ("human", "{question}")
])

# Check input variables
print(template.input_variables)  # ['language', 'question']

# Format to get messages
messages = template.format_messages(
    language="French",
    question="What is the capital of France?"
)

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

**Output:**
```
system: You are a helpful assistant who speaks French.
human: What is the capital of France?
```

### Using the Constructor

Direct instantiation with a list of messages:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ("system", "You are an expert {role}."),
    ("human", "Please help me with: {task}")
])

# Same behavior as from_messages()
print(template.input_variables)
```

---

## Message Types

### The Tuple Shorthand

The most common way to specify messages:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant."),
    ("human", "Write a function to {task}"),
    ("ai", "Here's a solution:"),
    ("human", "Can you explain line {line_number}?")
])
```

| Role String | Message Type | Description |
|-------------|--------------|-------------|
| `"system"` | SystemMessage | Instructions for the model |
| `"human"` / `"user"` | HumanMessage | User input |
| `"ai"` / `"assistant"` | AIMessage | Model responses |
| `"placeholder"` | MessagesPlaceholder | Dynamic message insertion |

### Using Message Classes

For more control, use message classes directly:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is a decorator?"),
    AIMessage(content="A decorator is a function that modifies another function."),
    ("human", "{followup}")  # Can mix approaches
])

messages = template.format_messages(followup="Give me an example")
print(len(messages))  # 4
```

### Message Prompt Template Classes

For templated message content:

```python
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

system_template = SystemMessagePromptTemplate.from_template(
    "You are an expert in {domain}. Respond in {language}."
)

human_template = HumanMessagePromptTemplate.from_template(
    "Question: {question}"
)

template = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

print(template.input_variables)  # ['domain', 'language', 'question']
```

---

## MessagesPlaceholder

`MessagesPlaceholder` enables dynamic message injection—essential for conversation history.

### Basic Usage

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# Format with message history
from langchain_core.messages import HumanMessage, AIMessage

messages = template.format_messages(
    history=[
        HumanMessage(content="Hi! I'm Alice."),
        AIMessage(content="Hello Alice! How can I help you?")
    ],
    input="What's my name?"
)

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

**Output:**
```
system: You are a helpful assistant.
human: Hi! I'm Alice.
ai: Hello Alice! How can I help you?
human: What's my name?
```

### Optional Placeholders

Make history optional with `optional=True`:

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history", optional=True),
    ("human", "{input}")
])

# Works without history
messages = template.format_messages(input="Hello!")
print(len(messages))  # 2 (system + human)

# Also works with history
messages_with_history = template.format_messages(
    input="Hello!",
    history=[HumanMessage(content="Previous message")]
)
print(len(messages_with_history))  # 3
```

### Tuple Shorthand for Placeholders

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("placeholder", "{history}"),  # Shorthand syntax
    ("human", "{input}")
])
```

### Limiting Message Count

Use `n_messages` to limit history:

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Only keep last 5 messages from history
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history", n_messages=5),
    ("human", "{input}")
])
```

---

## Multi-Turn Conversation Templates

### Conversation with Context

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent for {company}.
    
Current customer context:
- Name: {customer_name}
- Account type: {account_type}
- Issue category: {category}

Be helpful and professional."""),
    MessagesPlaceholder("conversation_history", optional=True),
    ("human", "{current_message}")
])

print(template.input_variables)
# ['company', 'customer_name', 'account_type', 'category', 'current_message']
# Note: conversation_history is optional so not in input_variables
```

### RAG Pattern with Chat History

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use the following context to answer questions.

Context:
{context}

If you don't know the answer, say "I don't know."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}")
])
```

---

## Formatting and Invocation

### `format_messages()` vs `invoke()`

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{question}")
])

# format_messages() returns list of BaseMessage
messages = template.format_messages(question="Hello")
print(type(messages))  # list
print(type(messages[0]))  # SystemMessage

# invoke() returns ChatPromptValue (Runnable interface)
prompt_value = template.invoke({"question": "Hello"})
print(type(prompt_value))  # ChatPromptValue
print(prompt_value.to_messages())  # list of messages
print(prompt_value.to_string())  # string representation
```

### Single-Variable Shorthand

If your template has only one variable:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("human", "{text}")
])

# Can pass the value directly instead of dict
prompt_value = template.invoke("Hello world")
# Equivalent to: template.invoke({"text": "Hello world"})
```

---

## Chaining with Models

### Basic Chain

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "Explain {concept} in Python with an example.")
])

model = init_chat_model("gpt-4o")
parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"concept": "list comprehensions"})
print(result)
```

### With Conversation History

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

model = init_chat_model("gpt-4o")
chain = template | model

# First turn
response1 = chain.invoke({
    "history": [],
    "input": "My name is Alice"
})
print(response1.content)

# Second turn with history
response2 = chain.invoke({
    "history": [
        HumanMessage(content="My name is Alice"),
        AIMessage(content="Hello Alice! Nice to meet you.")
    ],
    "input": "What's my name?"
})
print(response2.content)
```

---

## Template Operations

### Adding Messages with `+`

```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

base = ChatPromptTemplate.from_messages([
    ("system", "You are helpful.")
])

# Add more messages
extended = base + HumanMessagePromptTemplate.from_template("{question}")

print(len(extended.messages))  # 2
```

### Appending Messages

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful.")
])

# Append modifies in place
template.append(("human", "{question}"))
template.append(("ai", "Let me help you with that."))

print(len(template.messages))  # 3
```

### Extending with Multiple Messages

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful.")
])

template.extend([
    ("human", "Question 1"),
    ("ai", "Answer 1"),
    ("human", "{followup}")
])

print(len(template.messages))  # 4
```

---

## Partial Templates

Pre-fill some variables:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for {company}. Speak in {language}."),
    ("human", "{question}")
])

# Create partial with company pre-filled
acme_template = template.partial(company="Acme Corp")

print(acme_template.input_variables)  # ['language', 'question']

messages = acme_template.format_messages(
    language="English",
    question="What products do you sell?"
)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Always include a system message | Sets model behavior and context |
| Use `optional=True` for history | Works with both new and continuing conversations |
| Limit history with `n_messages` | Control token usage in long conversations |
| Use tuple syntax for simplicity | Cleaner code, same functionality |
| Validate `input_variables` | Catch missing variables early |
| Use partials for constants | Company name, language, persona |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting system message | Always set context/behavior |
| History as required variable | Use `optional=True` for flexibility |
| Mixing message types inconsistently | Pick one style (tuples or classes) |
| Not handling empty history | Test with and without history |
| Overly long system prompts | Move context to retrieval |

---

## Hands-on Exercise

### Your Task

Create a customer support chatbot template that:
1. Has a detailed system message with company info
2. Includes optional conversation history
3. Handles current user input
4. Works with or without history

### Requirements

1. Include company name, support hours, and policies in system message
2. Use `MessagesPlaceholder` with `optional=True`
3. Test the template with and without history
4. Chain with a model and get a response

### Expected Result

```
# Without history
System: You are a support agent for TechCorp...
Human: What are your hours?

# With history  
System: You are a support agent for TechCorp...
Human: I have a billing question
AI: I'd be happy to help with billing...
Human: Can I get a refund?
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use triple-quoted string for the system message
- `MessagesPlaceholder("history", optional=True)` for flexibility
- Test `format_messages()` with both empty and populated history
- Partial variables can pre-fill company info

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

# Create the template
template = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent for {company}.

Support Hours: Monday-Friday, 9 AM - 5 PM EST
Refund Policy: Full refunds within 30 days of purchase

Guidelines:
- Be helpful and professional
- If you can't help, escalate to a human agent
- Never share customer personal information"""),
    MessagesPlaceholder("history", optional=True),
    ("human", "{input}")
])

# Create partial with company pre-filled
support_template = template.partial(company="TechCorp")

print(f"Required variables: {support_template.input_variables}")

# Test without history
messages_no_history = support_template.format_messages(
    input="What are your support hours?"
)
print(f"\nWithout history ({len(messages_no_history)} messages):")
for msg in messages_no_history:
    print(f"  {msg.type}: {msg.content[:50]}...")

# Test with history
messages_with_history = support_template.format_messages(
    history=[
        HumanMessage(content="I have a billing question"),
        AIMessage(content="I'd be happy to help with billing. What's your question?")
    ],
    input="Can I get a refund for my subscription?"
)
print(f"\nWith history ({len(messages_with_history)} messages):")
for msg in messages_with_history:
    print(f"  {msg.type}: {msg.content[:50]}...")

# Chain with model (optional - requires API key)
# model = init_chat_model("gpt-4o")
# chain = support_template | model
# response = chain.invoke({"input": "What's your refund policy?"})
# print(response.content)
```

</details>

### Bonus Challenges

- [ ] Add a `MessagesPlaceholder` for tool calls/results
- [ ] Create templates for different support categories (billing, technical, general)
- [ ] Implement message trimming to stay within token limits

---

## Summary

✅ `ChatPromptTemplate` creates message lists for chat models  
✅ Use tuple syntax `("role", "content")` for simple messages  
✅ **MessagesPlaceholder** enables dynamic history injection  
✅ Set `optional=True` for flexible conversation handling  
✅ Use `n_messages` to limit history and control tokens  
✅ Templates chain seamlessly with `|` operator  

**Next:** [Template Composition](./03-template-composition.md) — Combine and extend templates for complex prompts

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [PromptTemplate Basics](./01-prompttemplate-basics.md) | [Prompt Templates](./00-prompt-templates.md) | [Template Composition](./03-template-composition.md) |

<!-- 
Sources Consulted:
- LangChain GitHub prompts/chat.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain_core/prompts/chat.py
- LangChain GitHub tests/unit_tests/prompts/test_chat.py: https://github.com/langchain-ai/langchain/blob/main/libs/core/tests/unit_tests/prompts/test_chat.py
-->
