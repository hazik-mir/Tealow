# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added

- Initial public release of TeaLow.
- Unified `TeaLow` synchronous client with automatic provider detection.
- `AsyncTeaLow` asynchronous client built on `httpx`.
- Provider adapters for Google Gemini, OpenAI, Anthropic Claude, and DeepSeek.
- Conversation history management via `Conversation`.
- Streaming support (`stream()` / `astream()`) for all providers.
- Vision (image input) support for Gemini, OpenAI, and Anthropic.
- File upload helpers (`TeaLow.load_image`, `TeaLow.load_file`).
- Image generation support for OpenAI (DALL-E) and Gemini (Imagen).
- JSON response mode with response validation.
- Automatic retries with exponential backoff and jitter.
- Rate-limit detection via a dedicated `RateLimitError`.
- Custom headers, proxy configuration, and TLS verification toggles.
- Environment-variable based API key resolution.
- API key format validation.
- Structured logging via `configure_logging()`.
- Comprehensive exception hierarchy rooted at `TeaLowError`.
- Full type hints and `py.typed` marker for downstream type checking.
- Unit test suite covering routing, retries, providers, and models.
- Examples covering basic usage, streaming, async, vision, and JSON mode.
- Complete documentation set (README, architecture, provider reference).
- MIT License, CONTRIBUTING guide, GitHub Actions CI workflow,
  pre-commit configuration, Ruff/Black/Mypy tool configuration.

[1.0.0]: https://github.com/tealow/tealow/releases/tag/v1.0.0
