# Contributing to scicomap

Thank you for contributing.

## Development setup

```shell
uv sync --extra lint --extra test --extra docs
```

## Common checks

```shell
uv run python -m pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run sphinx-build -n -b html docs/source docs/build/html
uv run python scripts/build_llm_assets.py
uv run python -m pytest tests/docs/test_build_llm_assets.py
```

## Docstring style

- Use NumPy-style docstrings for new and modified public APIs.
- Keep docstrings synchronized with parameters/defaults/return values.
- Legacy docstrings are normalized incrementally; convert touched legacy docstrings when practical.

Ruff enforces a phased subset of docstring style checks while this migration is in progress.

## LLM docs maintenance

If you update the docs theme or Sphinx structure, validate parser assumptions:

- Ensure parser tests pass.
- Rebuild HTML docs and regenerate LLM assets.
- Spot-check that markdown mirrors keep one H1 and no sidebar content.

## Pull requests

- Use conventional commits.
- Keep each PR focused on one outcome.
- Update docs for user-facing changes.
- Include validation commands in the PR description.
