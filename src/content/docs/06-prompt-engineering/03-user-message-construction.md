---
title: "User Message Construction"
---

# User Message Construction

- Clear task description
  - Action verbs
  - Specific goals
  - Success criteria
  - Scope boundaries
- Providing context
  - Background information
  - Relevant prior conversation
  - Domain context
  - User intent clarification
- Input data formatting
  - Delimiters for data sections
  - Labels and markers
  - Data format specification
  - Handling special characters
- Specifying desired output
  - Output format description
  - Example outputs
  - Required fields
  - Optional elements
- Handling multi-step tasks
  - Step numbering
  - Dependencies between steps
  - Intermediate output handling
  - Final output specification
- **Grounding & RAG Prompting**
  - Adding retrieved context to prompts
  - Clear separation of context and instructions
  - "Answer based only on the provided context"
  - Source citation instructions
  - Handling when context doesn't contain answer
  - Grounding prompt template for Gemini 3:
    ```
    You are a strictly grounded assistant limited to the information
    provided in the User Context. Rely only on facts directly mentioned.
    If the answer is not in the context, state that.
    ```
