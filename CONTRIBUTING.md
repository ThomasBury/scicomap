# Contributing to scicomap

Thank you for contributing.

## Development setup

```shell
just sync  # uv sync --locked --extra lint --extra test
```

## Common checks

```shell
just check       # ordinary pytest, Ruff on src/tests/scripts, ty on src/scicomap
just docs        # install docs extras, build HTML and LLM assets
just check-docs  # strict generated examples and WASM bootstrap tests
uv run --locked python -m pytest tests/docs/test_build_llm_assets.py
```

All recipes use the same project `.venv`. Routine synchronization is locked;
run `uv lock` only when changing dependencies and review the lockfile diff.
`just check` needs only the lint/test extras. Tests marked `docs` run through
`just check-docs`, which installs the docs extra. Install the Pandoc binary
separately for notebook documentation. Use `just sync-docs` before running
documentation commands directly with `uv run --locked`.

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
2. Run release validation with `just release-check`, including installed-wheel checks.
   Serve `docs/build/html` over HTTP and verify the candidate wheel in the browser tutorial.
3. Publish a release candidate to TestPyPI by tagging `*rc*` (for example,
   `2.0.0rc1`) and run `just smoke-testpypi <version>`.
4. Tag and push (`just tag <version>` then `just push-tag <version>`).
5. The `Publish to PyPI` workflow builds and publishes on tag pushes using
   Trusted Publishing.

### Trusted Publishing setup

Before first automated release, register this repository as a trusted publisher
for project `scicomap` on PyPI/TestPyPI and configure GitHub environments
`pypi` and `testpypi`.

Install `just` locally to use these commands, and keep using `uv` as the
isolated runtime backend.
