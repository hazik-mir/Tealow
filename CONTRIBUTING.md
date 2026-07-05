# Contributing to TeaLow

Thank you for considering a contribution to TeaLow! This document
explains how to set up your environment, the standards we hold code
to, and how to submit changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install development dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

## Development Workflow

- **Formatting**: Code is formatted with [Black](https://black.readthedocs.io/).
  Run `black .` before committing, or let pre-commit do it for you.
- **Linting**: We use [Ruff](https://docs.astral.sh/ruff/) for linting and
  import sorting. Run `ruff check .`.
- **Type checking**: All code must pass `mypy TeaLow` in strict mode.
- **Tests**: Add or update tests under `tests/` for any behavioural
  change. Run the full suite with:

  ```bash
  pytest
  ```

- **Docstrings**: Every public module, class, and function must have a
  docstring following the Google style used throughout the codebase.

## Commit Messages

Please write clear, descriptive commit messages. We loosely follow
[Conventional Commits](https://www.conventionalcommits.org/), e.g.:

```
feat(providers): add support for DeepSeek reasoning models
fix(streaming): handle empty SSE keep-alive lines
docs(readme): clarify async usage example
```

## Adding a New Provider

1. Create a new module under `TeaLow/providers/`, subclassing
   `TeaLow.providers.base.BaseProvider`.
2. Implement `build_headers`, `send`, `stream`, `asend`, and `astream`.
3. Register the provider in `TeaLow/constants.py`
   (`MODEL_PREFIX_ROUTES`, `DEFAULT_BASE_URLS`, `ENV_VAR_NAMES`) and in
   `TeaLow/providers/__init__.py`.
4. Add unit tests under `tests/` mocking the new provider's HTTP API.
5. Document the provider in `docs/providers.md` and the README table.

## Pull Requests

- Keep pull requests focused on a single change where possible.
- Ensure `pytest`, `ruff check .`, `black --check .`, and `mypy TeaLow`
  all pass before requesting review.
- Update `CHANGELOG.md` under an "Unreleased" section describing your change.
- Link any relevant issues in the PR description.

## Reporting Issues

When filing a bug report, please include:

- The TeaLow version (`python -c "import TeaLow; print(TeaLow.__version__)"`)
- The provider and model you were using
- A minimal reproducible example
- The full traceback, with any API keys redacted

## Code of Conduct

Be respectful and constructive. We want TeaLow to be a welcoming
project for contributors of all experience levels.
