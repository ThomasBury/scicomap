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

## Release workflow

Use SemVer: patch for fixes, minor for additive features, major for breaking
changes.

1. Bump `src/scicomap/__init__.py` version.
2. Run release validation with `just release-check`.
3. Optionally smoke-test a pre-release on TestPyPI with
   `just smoke-testpypi <version>`.
4. Tag and push (`just tag <version>` then `just push-tag <version>`).
5. The `Publish to PyPI` workflow builds and publishes on tag pushes using
   Trusted Publishing.

### Trusted Publishing setup

Before first automated release, register this repository as a trusted publisher
for project `scicomap` on PyPI and configure GitHub environment `pypi`.

Install `just` locally to use these commands, and keep using `uv` as the
isolated runtime backend.
