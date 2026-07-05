# TeaLow

**TeaLow** is a unified AI SDK for Python that lets you talk to Google
Gemini, OpenAI, Anthropic Claude, and DeepSeek through **one consistent
API**. Switch providers by changing a model name — no other code
changes required.

```python
from TeaLow import TeaLow

ai = TeaLow(
    model="gemini-2.5-flash",
    api="YOUR_API_KEY"
)

response = ai.send("Hello!")
print(response)
```

The exact same code works for OpenAI, Anthropic, or DeepSeek — just
swap the `model` and `api` values:

```python
ai = TeaLow(model="gpt-4o-mini", api="YOUR_OPENAI_KEY")
ai = TeaLow(model="claude-sonnet-4-6", api="YOUR_ANTHROPIC_KEY")
ai = TeaLow(model="deepseek-chat", api="YOUR_DEEPSEEK_KEY")
```

## Features

- 🔀 **Automatic provider routing** — the provider is detected from the model name.
- 🧵 **Conversation history** — multi-turn context managed for you.
- 🌊 **Streaming** — token-by-token streaming for every provider.
- ⚡ **Sync & async** — `TeaLow` for synchronous code, `AsyncTeaLow` for `asyncio`.
- 🖼️ **Vision & file uploads** — attach images/files to your messages.
- 🎨 **Image generation** — generate images via OpenAI (DALL-E) or Google (Imagen).
- 🧾 **JSON mode** — force and validate structured JSON responses.
- 🔁 **Automatic retries** — exponential backoff with jitter on transient errors.
- 🚦 **Rate-limit detection** — dedicated `RateLimitError` with `retry_after`.
- 🛡️ **Robust error handling** — a clear exception hierarchy for every failure mode.
- 🧩 **Fully typed** — complete type hints and a `py.typed` marker.
- 🪝 **Custom headers & proxy support** — for corporate networks and gateways.
- 📝 **Logging** — opt-in structured logging via `configure_logging()`.
- 🔑 **Environment variable support** — omit `api=` and TeaLow reads
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`.

## Installation

```bash
pip install TeaLow
```

Or, from source:

```bash
git clone https://github.com/tealow/tealow.git
cd tealow
pip install -e ".[dev]"
```

## Quick Start

### Basic usage

```python
from TeaLow import TeaLow

ai = TeaLow(model="gpt-4o-mini", api="YOUR_OPENAI_KEY")
response = ai.send("What is the capital of France?")
print(response.text)      # "The capital of France is Paris."
print(response.model)     # "gpt-4o-mini"
print(response.provider)  # "openai"
print(response.usage)     # Usage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
```

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from TeaLow import TeaLow

ai = TeaLow(model="gpt-4o-mini")  # api key resolved from OPENAI_API_KEY
print(ai.send("Hello!"))
```

### Conversation history

```python
ai = TeaLow(model="claude-sonnet-4-6", api="YOUR_KEY", system_prompt="You are a helpful pirate.")

ai.send("What's your name?")
ai.send("What did I just ask you?")   # remembers the prior turn

for message in ai.history:
    print(message["role"], "->", message["content"])
```

### Streaming

```python
for chunk in ai.stream("Write a haiku about the ocean."):
    print(chunk.delta, end="", flush=True)
```

### Async usage

```python
import asyncio
from TeaLow import AsyncTeaLow

async def main():
    ai = AsyncTeaLow(model="gemini-2.5-flash", api="YOUR_KEY")
    response = await ai.send("Hello from asyncio!")
    print(response)

    async for chunk in ai.stream("Tell me a joke."):
        print(chunk.delta, end="")

    await ai.close()

asyncio.run(main())
```

### Vision (image input)

```python
ai = TeaLow(model="gpt-4o-mini", api="YOUR_KEY")
image = TeaLow.load_image(path="photo.jpg")
response = ai.send("What is in this image?", images=[image])
print(response)
```

### JSON mode

```python
response = ai.send(
    "Return a JSON object with keys 'name' and 'age' for a fictional person.",
    json_mode=True,
)
import json
data = json.loads(response.text)
```

### Image generation

```python
ai = TeaLow(model="dall-e-3", api="YOUR_OPENAI_KEY")
urls = ai.generate_image("A watercolor painting of a teapot on a low table")
print(urls[0])
```

### Custom headers, proxy, and timeouts

```python
ai = TeaLow(
    model="gpt-4o-mini",
    api="YOUR_KEY",
    timeout=30.0,
    max_retries=5,
    proxies={"https": "http://proxy.internal:8080"},
    headers={"X-Org-Id": "acme-corp"},
)
```

### Logging

```python
from TeaLow import configure_logging
import logging

configure_logging(level=logging.DEBUG)
```

## Supported Models

TeaLow detects the provider automatically from the model name:

| Provider  | Example models                              | Detected by substring        |
|-----------|----------------------------------------------|-------------------------------|
| Gemini    | `gemini-2.5-flash`, `gemini-2.5-pro`          | `gemini`                      |
| OpenAI    | `gpt-4o`, `gpt-4o-mini`, `o3`, `dall-e-3`      | `gpt-`, `o1`/`o3`/`o4`, `dall-e` |
| Anthropic | `claude-sonnet-4-6`, `claude-opus-4-8`         | `claude`                      |
| DeepSeek  | `deepseek-chat`, `deepseek-reasoner`           | `deepseek`                    |

## Error Handling

All exceptions inherit from `TeaLowError`:

```python
from TeaLow import TeaLow, RateLimitError, InvalidAPIKeyError, TeaLowError

ai = TeaLow(model="gpt-4o-mini", api="YOUR_KEY")

try:
    response = ai.send("Hello!")
except RateLimitError as e:
    print("Rate limited, retry after:", e.retry_after)
except InvalidAPIKeyError:
    print("Your API key was rejected.")
except TeaLowError as e:
    print("Something else went wrong:", e)
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md), including:

- [Architecture overview](docs/architecture.md)
- [Provider reference](docs/providers.md)
- [API reference](docs/api_reference.md)

## Examples

See the [`examples/`](examples/) folder for runnable scripts covering
basic usage, streaming, async, vision, JSON mode, and image generation.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

TeaLow is released under the [MIT License](LICENSE).
