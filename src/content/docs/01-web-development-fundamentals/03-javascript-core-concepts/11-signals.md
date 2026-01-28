---
title: "Signals (TC39 Proposal)"
---

# Signals (TC39 Proposal)

## Introduction

Signals are a proposed JavaScript feature for reactive state management—a standardized way to create values that automatically update dependent computations when they change. While frameworks like React (with hooks), Vue (with refs), and Solid.js already have reactive primitives, Signals aim to provide a **native, framework-agnostic** solution built into JavaScript itself.

For AI applications, Signals could simplify state management in chat interfaces, real-time data updates, and reactive UI patterns without framework dependencies. This lesson covers the TC39 proposal's current design and demonstrates the concepts using polyfill implementations.

### What We'll Cover
- What Signals are and why they matter
- The TC39 Signals proposal overview
- Core concepts: State, Computed, Effect
- Practical patterns for reactive UIs
- Current status and alternatives

### Prerequisites
- JavaScript functions and closures
- Understanding of state management concepts
- Basic knowledge of reactive programming (helpful but not required)

---

## What Are Signals?

A **Signal** is a reactive value that tracks its dependencies and notifies dependents when it changes:

```javascript
// Conceptual example (not native yet)
const count = Signal.State(0);
const doubled = Signal.Computed(() => count.get() * 2);

console.log(doubled.get());  // 0
count.set(5);
console.log(doubled.get());  // 10 (automatically updated)
```

### Key Characteristics

1. **Fine-grained reactivity**: Only affected computations re-run
2. **Automatic dependency tracking**: No manual subscription management
3. **Synchronous by default**: Updates happen immediately
4. **Framework-agnostic**: Works in vanilla JS, any framework, or web components

---

## The TC39 Proposal

As of 2025, Signals are a **Stage 1 proposal** at TC39 (the JavaScript standards committee). The proposal defines three core primitives:

### 1. Signal.State

Mutable reactive value:

```javascript
const state = new Signal.State(initialValue);
state.get();        // Read current value
state.set(newValue); // Update value
```

### 2. Signal.Computed

Derived value that auto-updates when dependencies change:

```javascript
const computed = new Signal.Computed(() => {
  return state1.get() + state2.get();
});

computed.get();  // Automatically recalculates if dependencies changed
```

### 3. Signal.effect (or Signal.subtle.effect)

Side effect that runs when dependencies change:

```javascript
const effect = new Signal.subtle.effect(() => {
  console.log("Count is:", count.get());
});
// Runs immediately and whenever count changes
```

---

## Conceptual Examples

Since Signals aren't native yet, let's use conceptual examples to understand the patterns:

### Counter Example

```javascript
// State
const count = new Signal.State(0);
const doubleCount = new Signal.Computed(() => count.get() * 2);

// Effect (runs on change)
new Signal.subtle.effect(() => {
  document.getElementById('count').textContent = count.get();
  document.getElementById('double').textContent = doubleCount.get();
});

// Update state
document.getElementById('increment').addEventListener('click', () => {
  count.set(count.get() + 1);
});
```

**HTML:**
```html
<div>
  <p>Count: <span id="count">0</span></p>
  <p>Double: <span id="double">0</span></p>
  <button id="increment">Increment</button>
</div>
```

When you click "Increment", the effect automatically updates both spans.

### Chat Message Counter

```javascript
const messages = new Signal.State([]);
const messageCount = new Signal.Computed(() => messages.get().length);
const userMessageCount = new Signal.Computed(() => {
  return messages.get().filter(m => m.role === 'user').length;
});

// Auto-update UI
new Signal.subtle.effect(() => {
  document.getElementById('total').textContent = messageCount.get();
  document.getElementById('user').textContent = userMessageCount.get();
});

// Add message
function addMessage(role, content) {
  const currentMessages = messages.get();
  messages.set([...currentMessages, { role, content, timestamp: Date.now() }]);
}

addMessage('user', 'Hello');
// UI automatically updates!
```

---

## Polyfill Implementation

While we wait for native Signals, here's a simplified polyfill to demonstrate the concept:

```javascript
class SignalState {
  #value;
  #subscribers = new Set();
  
  constructor(initialValue) {
    this.#value = initialValue;
  }
  
  get() {
    // Track this signal as a dependency
    if (SignalState.currentEffect) {
      this.#subscribers.add(SignalState.currentEffect);
    }
    return this.#value;
  }
  
  set(newValue) {
    if (this.#value !== newValue) {
      this.#value = newValue;
      // Notify all subscribers
      this.#subscribers.forEach(effect => effect());
    }
  }
}

class SignalComputed {
  #fn;
  #value;
  #dirty = true;
  
  constructor(fn) {
    this.#fn = fn;
  }
  
  get() {
    if (this.#dirty) {
      const prevEffect = SignalState.currentEffect;
      SignalState.currentEffect = () => this.#dirty = true;
      this.#value = this.#fn();
      SignalState.currentEffect = prevEffect;
      this.#dirty = false;
    }
    return this.#value;
  }
}

class SignalEffect {
  constructor(fn) {
    const execute = () => {
      const prevEffect = SignalState.currentEffect;
      SignalState.currentEffect = execute;
      fn();
      SignalState.currentEffect = prevEffect;
    };
    execute();
  }
}

SignalState.currentEffect = null;

// Export as Signal namespace
const Signal = {
  State: SignalState,
  Computed: SignalComputed,
  subtle: {
    effect: SignalEffect
  }
};
```

### Using the Polyfill

```javascript
// Create reactive state
const count = new Signal.State(0);
const doubled = new Signal.Computed(() => count.get() * 2);
const quadrupled = new Signal.Computed(() => doubled.get() * 2);

// Create effect
new Signal.subtle.effect(() => {
  console.log(`Count: ${count.get()}, Doubled: ${doubled.get()}, Quadrupled: ${quadrupled.get()}`);
});
// Output: Count: 0, Doubled: 0, Quadrupled: 0

count.set(5);
// Output: Count: 5, Doubled: 10, Quadrupled: 20

count.set(10);
// Output: Count: 10, Doubled: 20, Quadrupled: 40
```

---

## Practical AI Application Example

Chat interface with reactive state:

```javascript
// State
const messages = new Signal.State([]);
const input = new Signal.State('');
const isLoading = new Signal.State(false);

// Computed
const messageCount = new Signal.Computed(() => messages.get().length);
const canSend = new Signal.Computed(() => {
  return !isLoading.get() && input.get().trim() !== '';
});

// Effects for UI updates
new Signal.subtle.effect(() => {
  const container = document.getElementById('messages');
  container.innerHTML = messages.get()
    .map(m => `<div class="${m.role}">${m.content}</div>`)
    .join('');
});

new Signal.subtle.effect(() => {
  document.getElementById('count').textContent = messageCount.get();
});

new Signal.subtle.effect(() => {
  document.getElementById('send').disabled = !canSend.get();
});

new Signal.subtle.effect(() => {
  document.getElementById('loading').style.display = 
    isLoading.get() ? 'block' : 'none';
});

// Actions
async function sendMessage() {
  const content = input.get().trim();
  if (!content) return;
  
  // Add user message
  messages.set([...messages.get(), { role: 'user', content }]);
  input.set('');
  isLoading.set(true);
  
  // Simulate AI response
  await new Promise(resolve => setTimeout(resolve, 1000));
  messages.set([...messages.get(), { 
    role: 'assistant', 
    content: `Echo: ${content}` 
  }]);
  isLoading.set(false);
}
```

---

## Comparison with Framework Solutions

| Feature | TC39 Signals | React useState | Vue ref | Solid.js createSignal |
|---------|--------------|----------------|---------|------------------------|
| Native | Yes (future) | No (React only) | No (Vue only) | No (Solid only) |
| Fine-grained | Yes | No (re-renders component) | Yes | Yes |
| Auto-tracking | Yes | Manual deps | Yes | Yes |
| Synchronous | Yes | Async batching | Yes | Yes |
| Framework-agnostic | Yes | No | No | No |

---

## Current Status and Alternatives

### Proposal Status (2025)

- **Stage**: 1 (Proposal)
- **Repository**: [tc39/proposal-signals](https://github.com/tc39/proposal-signals)
- **Timeline**: Potentially years before Stage 4 and native implementation

### Alternatives Today

Until native Signals arrive, use:

1. **@preact/signals** - Polyfill-like library, framework-agnostic
2. **Solid.js Signals** - Framework with Signals as core primitive
3. **Vue 3 Composition API** - `ref()` and `computed()` are Signal-like
4. **MobX** - Mature observable state library
5. **RxJS** - Reactive programming with Observables

Example with @preact/signals:

```javascript
import { signal, computed, effect } from "@preact/signals-core";

const count = signal(0);
const doubled = computed(() => count.value * 2);

effect(() => {
  console.log(`Count: ${count.value}, Doubled: ${doubled.value}`);
});

count.value = 5;  // Auto-updates
```

---

## Best Practices (When Signals Become Native)

| Practice | Why It Matters |
|----------|----------------|
| Use `State` for primitive values | Simplest reactive primitive |
| Use `Computed` for derived values | Automatic memoization |
| Use effects sparingly | Only for side effects (DOM, logging, etc.) |
| Avoid side effects in computed | Keep computations pure |
| Batch updates when possible | Reduce unnecessary recalculations |
| Use framework wrappers if available | Better integration with existing code |

---

## Common Pitfalls (Hypothetical)

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Side effects in computed functions | Use effects for side effects |
| Forgetting to call `.get()` | Signals aren't values—must call `.get()` |
| Over-using effects | Prefer computed for derived state |
| Mutating state directly | Always use `.set()` |
| Creating too many fine-grained signals | Balance granularity with complexity |
| Not understanding synchronous updates | Effects run immediately, unlike React's batching |

---

## Hands-on Exercise

### Your Task
Using the polyfill provided earlier, create a reactive todo list with Signals. Implement add, toggle, and filter functionality with automatic UI updates.

### Requirements
1. Use `Signal.State` for todos array and filter
2. Use `Signal.Computed` for filtered todos and stats
3. Use `Signal.subtle.effect` for DOM updates
4. Implement: add todo, toggle complete, filter (all/active/completed)

### Expected Behavior
```javascript
// Add todo
addTodo("Learn Signals");
// UI automatically updates

// Toggle todo
toggleTodo(todoId);
// Completed count automatically updates

// Change filter
setFilter("completed");
// Filtered list automatically updates
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Store todos as array of objects: `{ id, text, completed }`
- Use `.get()` to read signal values
- Use `.set()` with new arrays/objects (immutable updates)
- Create computed signals for filtered lists and counts
- Effects should only update DOM, not modify state
</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```javascript
// Copy the Signal polyfill from earlier here
// ...

// State
const todos = new Signal.State([]);
const filter = new Signal.State('all');  // 'all' | 'active' | 'completed'

// Computed
const filteredTodos = new Signal.Computed(() => {
  const currentTodos = todos.get();
  const currentFilter = filter.get();
  
  if (currentFilter === 'active') {
    return currentTodos.filter(t => !t.completed);
  } else if (currentFilter === 'completed') {
    return currentTodos.filter(t => t.completed);
  }
  return currentTodos;
});

const stats = new Signal.Computed(() => {
  const currentTodos = todos.get();
  return {
    total: currentTodos.length,
    active: currentTodos.filter(t => !t.completed).length,
    completed: currentTodos.filter(t => t.completed).length
  };
});

// Effects (UI updates)
new Signal.subtle.effect(() => {
  const list = document.getElementById('todo-list');
  list.innerHTML = filteredTodos.get()
    .map(todo => `
      <div>
        <input type="checkbox" 
               ${todo.completed ? 'checked' : ''} 
               onchange="toggleTodo(${todo.id})">
        <span style="text-decoration: ${todo.completed ? 'line-through' : 'none'}">
          ${todo.text}
        </span>
      </div>
    `)
    .join('');
});

new Signal.subtle.effect(() => {
  const currentStats = stats.get();
  document.getElementById('stats').textContent = 
    `Total: ${currentStats.total}, Active: ${currentStats.active}, Completed: ${currentStats.completed}`;
});

// Actions
let nextId = 1;

function addTodo(text) {
  const currentTodos = todos.get();
  todos.set([...currentTodos, { id: nextId++, text, completed: false }]);
}

function toggleTodo(id) {
  const currentTodos = todos.get();
  todos.set(currentTodos.map(todo => 
    todo.id === id ? { ...todo, completed: !todo.completed } : todo
  ));
}

function setFilter(newFilter) {
  filter.set(newFilter);
}

// HTML setup
document.body.innerHTML = `
  <div>
    <input id="input" placeholder="Add todo">
    <button onclick="addTodo(document.getElementById('input').value); document.getElementById('input').value = ''">Add</button>
    <div>
      <button onclick="setFilter('all')">All</button>
      <button onclick="setFilter('active')">Active</button>
      <button onclick="setFilter('completed')">Completed</button>
    </div>
    <div id="todo-list"></div>
    <div id="stats"></div>
  </div>
`;

// Test
addTodo("Learn Signals");
addTodo("Build reactive UI");
addTodo("Master JavaScript");
```
</details>

### Bonus Challenges
- [ ] Add "Clear completed" button
- [ ] Implement edit functionality
- [ ] Add persistence with LocalStorage (as effect)
- [ ] Create computed for "all completed" checkbox state

---

## Summary

✅ Signals are a TC39 proposal for native reactive state in JavaScript
✅ Three primitives: State (mutable), Computed (derived), Effect (side effects)
✅ Fine-grained reactivity—only affected computations re-run
✅ Framework-agnostic—works in vanilla JS, any framework, or web components
✅ Use alternatives like @preact/signals today while waiting for native support

[Previous: Web Components](./10-web-components.md) | [Back to Overview](./00-javascript-core-concepts.md)

---

<!--
Sources Consulted:
- TC39 Signals Proposal: https://github.com/tc39/proposal-signals
- Signals Overview (JavaScript.info): https://javascript.info/
- Preact Signals Documentation: https://preactjs.com/guide/v10/signals/
- Solid.js Reactivity: https://www.solidjs.com/tutorial/introduction_signals
-->
