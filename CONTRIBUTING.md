# Contributing to scicomap

Thank you for contributing.

## Development setup

```shell
uv sync --extra lint --extra test --extra docs
```

## Common checks

```shell
uv run python -m pytest
uv run python -m flake8 src
uv run python -m black --check src
uv run sphinx-build -n -b html docs/source docs/build/html
uv run python scripts/build_llm_assets.py
```

## Pull requests

- Use conventional commits.
- Keep each PR focused on one outcome.
- Update docs for user-facing changes.
- Include validation commands in the PR description.
