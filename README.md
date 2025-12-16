# llm-utils

A unified Python interface for interacting with multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini). Provides a consistent API for chat completions, embeddings, web search, and structured outputs across all providers.

## Features

- **Unified API** - Single interface for OpenAI, Claude, and Gemini
- **Structured Outputs** - JSON schema support across all providers
- **Web Search** - Native web search integration for real-time information
- **Concurrent Processing** - Batch processing with configurable parallelization
- **Embeddings** - Text embeddings from OpenAI and Gemini (with optional ChromaDB integration)
- **Robust Error Handling** - Provider-specific error handling with automatic retries

## Installation

```bash
pip install llm-utils
```

Or install from source:

```bash
git clone https://github.com/yourusername/llm-utils.git
cd llm-utils
pip install -e .
```

## Environment Setup

Create a `.env` file in your project root with your API keys:

```bash
# Required for OpenAI
OPENAI_KEY=your-openai-key

# Required for Anthropic Claude
ANTHROPIC_API_KEY=your-anthropic-key

# Required for Google Gemini
GOOGLE_API_KEY=your-google-api-key
```

## Quick Start

```python
from llm_utils.wrapper import LLMWrapper

llm = LLMWrapper()

# Simple chat completion
response = llm.ask(
    system_message="You are a helpful assistant.",
    ids=["0"],
    user_messages=["What is the capital of France?"],
    response_formats=[None],  # No structured output
    model="gpt-4o-mini"
)

print(response["0"])
```

## Usage

### Chat Completions with Structured Output

All providers support structured JSON outputs using a consistent schema format:

```python
from llm_utils.wrapper import LLMWrapper
import numpy as np

llm = LLMWrapper()

# Define your response schema
response_format = {
    "name": "city_info",
    "type": "json_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "string"}
        },
        "required": ["city", "country", "population"]
    }
}

# Works with any provider - just change the model name
response = llm.ask(
    system_message="You are a helpful assistant.",
    ids=["0"],
    user_messages=["Tell me about Tokyo."],
    response_formats=[response_format],
    model="gpt-4o-mini"  # or "claude-3-5-haiku-latest" or "gemini-2.0-flash"
)

print(response["0"])
# {'city': 'Tokyo', 'country': 'Japan', 'population': '13.96 million'}
```

### Concurrent Processing

Process multiple messages in parallel:

```python
messages = [
    "What is 1+1?",
    "What is 2+2?",
    "What is 3+3?"
]

response_format = {
    "name": "math_answer",
    "type": "json_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    }
}

response = llm.ask(
    system_message="Answer with just the number.",
    ids=["q1", "q2", "q3"],
    user_messages=messages,
    response_formats=[response_format] * 3,
    model="gpt-4o-mini",
    n_workers=3  # Process 3 requests in parallel
)

# {'q1': {'answer': '2'}, 'q2': {'answer': '4'}, 'q3': {'answer': '6'}}
```

### Web Search

Enable real-time web search for up-to-date information:

```python
response = llm.ask(
    system_message="You are a helpful assistant.",
    ids=["0"],
    user_messages=["What is the current price of Tesla stock?"],
    response_formats=[{
        "name": "stock_info",
        "type": "json_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "price": {"type": "string"},
                "exchange": {"type": "string"}
            },
            "required": ["ticker", "price", "exchange"]
        }
    }],
    model="gpt-4o-mini",
    web_search=True  # Enable web search
)
```

### Text Embeddings

Generate embeddings with OpenAI or Gemini:

```python
# OpenAI embeddings
texts = ["cat", "kitty", "dog", "potato"]
ids = [str(i) for i in range(len(texts))]

_, embeddings = llm.embed(
    ids=ids,
    texts=texts,
    size=1536,  # Embedding dimension
    db=None,    # Optional ChromaDB instance
    name="my_embeddings",
    model="text-embedding-3-small"
)

# Gemini embeddings
_, embeddings = llm.embed(
    ids=ids,
    texts=texts,
    size=768,
    db=None,
    name="my_embeddings",
    model="gemini-embedding-001"
)
```

## Supported Models

### Chat Completions

| Provider | Model Examples |
|----------|---------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| Claude | `claude-sonnet-4-5`, `claude-3-5-haiku-latest` |
| Gemini | `gemini-2.0-flash`, `gemini-2.5-flash` |

The wrapper automatically routes to the correct provider based on the model name:
- Models containing `"gpt"` → OpenAI
- Models containing `"claude"` → Anthropic
- Models containing `"gemini"` → Google

### Embeddings

| Provider | Model | Dimensions |
|----------|-------|------------|
| OpenAI | `text-embedding-3-small` | 512-1536 |
| OpenAI | `text-embedding-3-large` | 256-3072 |
| Gemini | `gemini-embedding-001` | 768 |

Note: Claude does not support embeddings.

## API Reference

### `LLMWrapper.ask()`

```python
llm.ask(
    system_message: str,           # System prompt
    ids: list[str],                # Unique IDs for each message
    user_messages: list[str],      # User messages to process
    response_formats: list[dict],  # JSON schemas for structured output
    model: str,                    # Model name (determines provider)
    web_search: bool = False,      # Enable web search
    max_tool_calls: int = 3,       # Max tool calls (for web search)
    tool_choice: str = None,       # Tool selection preference
    reasoning: str = None,         # Reasoning mode (provider-specific)
    verbosity: str = "medium",     # Response verbosity
    max_tokens: int = None,        # Max output tokens
    n_workers: int = 1             # Parallel workers
) -> dict[str, any]
```

### `LLMWrapper.embed()`

```python
llm.embed(
    ids: list[str],           # Unique IDs for each text
    texts: list[str],         # Texts to embed
    size: int,                # Embedding dimension
    db: ChromaDB = None,      # Optional ChromaDB for storage
    name: str,                # Embedding collection name
    verbose: bool = True,     # Show progress bar
    model: str = None         # Embedding model name
) -> tuple[list[str], list[list[float]]]
```

## Error Handling

All interfaces include robust error handling with automatic retries:

- **Authentication errors** - Fail immediately with clear error message
- **Rate limits** - Automatic retry with exponential backoff
- **Connection errors** - Retry with backoff
- **Invalid requests** - Fail immediately with details

Error responses include both `error` (message) and `error_type` (category):

```python
response = llm.ask(...)
if "error" in response["0"]:
    print(f"Error type: {response['0']['error_type']}")
    print(f"Message: {response['0']['error']}")
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## License

MIT
