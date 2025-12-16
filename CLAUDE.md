# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-utils` is a unified Python interface for interacting with multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini). It provides a consistent API for chat completions, embeddings, web search, and structured outputs across all providers.

## Environment Setup

The project requires a `.env` file with API credentials:

```bash
ANTHROPIC_API_KEY=...
OPENAI_KEY=...
GEMINI_TYPE=...
GEMINI_PROJECT_ID=...
# (additional Gemini service account fields)
```

## Development Commands

### Environment
- Activate virtual environment: `source .venv/bin/activate` (or `.venv/Scripts/activate` on Windows)
- Install dependencies: `pip install -e .`

### Testing
- Run Jupyter notebooks: `jupyter notebook` or `jupyter lab`
- Main demo notebook: `demo.ipynb`

## Architecture

### Core Abstraction Pattern

The codebase uses an abstract base class pattern with provider-specific implementations:

1. **`LLMInterface`** (llm_utils/llm_interface.py) - Abstract base class defining:
   - `_embedding_call()` - Abstract method for single embedding request
   - `_concurrent_embedding_call()` - Concrete method handling batching and parallelization
   - `_chat_completion_call()` - Abstract method for single chat completion
   - `_chat_completion()` - Concrete method handling concurrent chat requests
   - `ask()` - Public API for chat completions

2. **Provider Implementations**:
   - `OpenAIInterface` (llm_utils/openai_interface.py) - Uses OpenAI's Responses API with native web search
   - `ClaudeInterface` (llm_utils/claude_interface.py) - Uses Anthropic SDK with tools for structured output and web search
   - `GeminiInterface` (llm_utils/gemini_interface.py) - Uses Google GenAI SDK with two-step approach for web search + structured output

3. **`LLMWrapper`** (llm_utils/wrapper.py) - Unified entry point that routes to correct provider based on model name:
   - Detects "gpt" → OpenAI
   - Detects "gemini" → Gemini
   - Detects "claude" → Claude

### Key Design Details

**Embedding Architecture:**
- Batch processing with configurable batch sizes (provider-specific limits)
- Concurrent processing using ThreadPoolExecutor (splits batches into CPU-count minibatches)
- Optional ChromaDB integration for direct database writes
- Exponential backoff retry logic

**Chat Completion Architecture:**
- Accepts lists of messages for concurrent processing
- Each provider implements structured output differently:
  - OpenAI: Native JSON schema support via `text.format`
  - Claude: Converts schema to tool with forced tool_choice
  - Gemini: Uses `response_schema` with two-pass approach for web search
- Web search handling varies by provider (native tools vs multi-step)

**Response Format Normalization:**
- All providers accept OpenAI-style `response_format` dicts
- Schema cleaning removes unsupported fields (e.g., `additionalProperties` for Claude)
- Claude special case: appends system message instruction when web search + structured output both enabled

### Important Implementation Notes

1. **Claude doesn't support embeddings** - `_embedding_call()` returns None
2. **Gemini web search + structured output** requires two API calls (search first, then format)
3. **Batch size limits**: OpenAI (2048/request), ChromaDB (5461 uploads)
4. **All chat methods** return `dict[id, response]` for easy result mapping
5. **Error handling**: 5 retry attempts with exponential backoff, returns `{"error": "..."}` on failure

## Common Patterns

### Adding a new provider
1. Subclass `LLMInterface`
2. Implement `_embedding_call()` and `_chat_completion_call()`
3. Add routing logic to `LLMWrapper.ask()` and `LLMWrapper.embed()`
4. Handle provider-specific response format conversion

### Modifying chat completion behavior
- Provider-specific logic goes in `_chat_completion_call()`
- Shared concurrency/batching logic stays in base class `_chat_completion()`
- Always maintain consistent return format: `dict[str, any]` mapping IDs to responses
