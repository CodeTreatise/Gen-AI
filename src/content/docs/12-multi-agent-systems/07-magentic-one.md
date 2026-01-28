---
title: "12.7 Magentic-One"
---

# 12.7 Magentic-One

## What is Magentic-One
- Microsoft's benchmark-leading multi-agent system
- Achieved SOTA on GAIA, AssistantBench, WebArena
- Generalist architecture for web/file tasks
- Part of AutoGen ecosystem
- Open-source reference implementation

## Architecture Overview
- Orchestrator as central coordinator
- 4 specialized worker agents
- Ledger-based task tracking
- Progress monitoring
- Dynamic replanning

## The Orchestrator Agent
- Task decomposition
- Agent assignment
- Progress tracking via ledger
- Stall detection and recovery
- Final answer synthesis

## WebSurfer Agent
- Chromium-based browser control
- Navigation and interaction
- Form filling
- Content extraction
- JavaScript execution

## FileSurfer Agent
- File system navigation
- Document reading (PDF, DOCX, etc.)
- Content extraction
- File search
- Directory traversal

## Coder Agent
- Code writing and execution
- Python environment access
- Package installation
- Error handling
- Result interpretation

## ComputerTerminal Agent
- Shell command execution
- System operations
- Environment management
- Output processing
- Error recovery

## MagenticOneGroupChat
- AutoGen integration
- Configuration options
- Custom agent addition
- Termination conditions
- Streaming results

## Use Cases
- Web research tasks
- Document analysis
- Data extraction
- Complex automation
- Multi-step workflows

