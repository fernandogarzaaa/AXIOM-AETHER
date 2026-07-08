# Set up Axiom with Cline (VS Code)

This guide gets Axiom running as an MCP server connected to Cline (the VS Code
extension formerly known as Claude Dev / Roo Code). Cline uses stdio MCP
transport -- no HTTP, no tunnel, no network exposure.

## Prerequisites

1. Axiom binary at: d:\AXIOM-AETHER\axiom_engine_rs\target\release\axiom_engine.exe
2. Trained checkpoint at: d:\AXIOM-AETHER\checkpoints\axiom_production_bpe.bin
3. Cline extension installed in VS Code (saoudrizwan.claude-dev)

## Configuration

The Cline MCP settings file is at:
%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json

It should contain the Axiom stdio server registration with all 16 tools,
environment variables for BPE model, memory, vibe priming, and alwaysAllow
for all tools to reduce friction.

## Available Tools (16 total)

### Context and Compression
- axiom_compress_path: Absorb a directory through TTT, return a context fingerprint
- axiom_expand: Retrieve a symbol body dropped from a digest
- axiom_evaluate_drift: Score code against fast-weights for architectural drift

### Memory (Persistent)
- axiom_remember: Store a decision, fix, convention, or snippet
- axiom_recall: Search long-term memory for relevant content
- axiom_forget: Tombstone a memory by id

### Grounding and Verification
- axiom_verify: Check response claims against supplied evidence
- axiom_validate_epistemic: Semantic LLM judge for soft hallucination detection

### Self-Healing
- axiom_immunity: Report learned self-healing experience for a command

### Session Awareness
- axiom_status: Report token budget, compression ratio, expansion-miss count

### Task Board (Inter-Agent)
- axiom_post_task: Post a task to a named channel
- axiom_claim_task: Claim the next available task from a channel
- axiom_task_result: Report the result of a claimed task
- axiom_list_tasks: List tasks in a channel, optionally filtered by status
- axiom_channels: List all task-board channels with at least one task

### ChatGPT Connector Aliases
- search: Search Axiom memory, return ranked results
- fetch: Fetch full text of a memory record by id

## Verify

After reloading VS Code with Cline:
1. Open the Cline panel in VS Code
2. Start a new chat
3. Ask: Use the Axiom axiom_status tool to check session awareness
4. Cline should call axiom_status and return the session state

## Proxy vs MCP

Axiom has two independent runtime paths:
1. HTTP Proxy (--mode server, port 3000) -- compression proxy for API traffic
2. MCP Server (--mode mcp, stdio) -- tool provider for Cline

Both share the same engine internals but run as separate processes.