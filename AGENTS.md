# AGENTS.md
Operating guide for agentic coding tools in this repository.

## Project Overview
- Language: Python (`>=3.10`)
- Package root: `src/scicomap`
- Tests: `tests/`
- Docs source: `docs/source`
- Main tooling: `uv`
- Packaging config: `pyproject.toml`

## Instruction Priority
Apply guidance in this order:
1. Direct user instruction
2. This `AGENTS.md`
3. Existing repository patterns

## Cursor/Copilot Rule Files
Checked and currently absent:
- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`
If these files are added later, treat them as high-priority repo guidance.

## Setup Commands
Run from repository root.
```bash
uv lock
uv sync --extra lint --extra test --extra docs
```
Notes:
- Notebook docs via `nbsphinx` require `pandoc` installed.
- Use `uv run ...` for all project commands.

## Build, Lint, and Test Commands
### Standard local checks
```bash
uv run python -m pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

### Single-test workflows (important)
```bash
# single test file
uv run python -m pytest tests/docs/test_build_llm_assets.py

# single test function
uv run python -m pytest tests/docs/test_build_llm_assets.py::test_to_markdown_does_not_duplicate_h1

# run tests by keyword expression
uv run python -m pytest -k "sidebar_skip"
```

### Docs and LLM asset pipeline
```bash
uv run sphinx-build -n -b html docs/source docs/build/html
uv run python scripts/build_llm_assets.py
uv run python -m pytest tests/docs/test_build_llm_assets.py
```

### CI-style docs asset validation
```bash
uv run python -c "from pathlib import Path; root=Path('docs/build/html'); llms=root/'llms.txt'; md=list((root/'llm').rglob('*.md')); assert llms.exists(), 'llms.txt missing'; assert md, 'No markdown mirrors generated'"
```

### Optional Makefile path
```bash
make -C docs html
```

## Codebase Landmarks
- `src/scicomap/scicomap.py`: public classes and plotting entry points
- `src/scicomap/cmath.py`: color math and colormap transforms
- `src/scicomap/cblind.py`: color-vision deficiency transforms
- `src/scicomap/utils.py`: plot/data helper functions
- `src/scicomap/datasets.py`: packaged data loaders
- `scripts/build_llm_assets.py`: docs HTML to markdown mirror + `llms.txt`
- `tests/docs/test_build_llm_assets.py`: parser regression tests

## Code Style Guidelines
Follow existing style in touched files. For new code, use these defaults.

### Imports
- Prefer absolute package imports (e.g., `from scicomap.cmath import ...`).
- Group imports by standard library, third-party, local package.
- Avoid wildcard imports in new code.
- Keep import lists tidy and remove unused names.

### Formatting
- Use `ruff format` and keep `ruff check` clean.
- Use 4-space indentation.
- Prefer readable line breaks over dense one-liners.
- Keep function bodies focused; extract helpers when logic grows.

### Types
- Add type hints to new/modified public APIs.
- Keep annotations consistent inside a file.
- Use precise return types, including tuple shapes where useful.
- Prefer explicit optionality (`X | None` or `Optional[X]`) consistently.

### Naming
- Classes: `CamelCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Internal helpers: leading underscore allowed and common here.
- Preserve established domain names (`cmap`, `Jpapbp`, `uniformized`).

### Docstrings
- Use NumPy-style docstrings for public APIs.
- Keep parameter and return docs synchronized with implementation.
- Include short examples only when non-obvious behavior needs clarification.

### Error Handling
- Raise specific exceptions (`ValueError`, `TypeError`, etc.).
- Include actionable error messages describing expected inputs.
- Do not silently swallow failures.
- Use `warnings.warn(...)` only for intentional fallback behavior.

### Plotting/Numerical Behavior
- Preserve existing defaults unless change is intentional.
- Return figure objects for plotting APIs where practical.
- Avoid hidden global-state changes.
- For randomness in examples/tests, set deterministic seeds.

### Paths and Files
- Prefer `pathlib.Path` in scripts and tests.
- Keep packaged resource loading compatible with existing patterns.
- Avoid absolute machine-specific paths.

## Test Selection Guidance
Run the smallest check that proves your change:
1. `scripts/build_llm_assets.py` changes:
   - `uv run python -m pytest tests/docs/test_build_llm_assets.py`
2. Runtime package changes in `src/scicomap`:
   - `uv run python -m pytest`
3. Formatting/import-only changes:
   - `uv run ruff check src tests`
   - `uv run ruff format --check src tests`
4. Docs build changes:
   - `uv run sphinx-build -n -b html docs/source docs/build/html`
   - `uv run python scripts/build_llm_assets.py`

## PR and Commit Expectations
- Keep changes scoped to one outcome.
- Update docs when user-facing behavior changes.
- Report exact validation commands run.
- Follow conventional commit style for every commit.

### Conventional Commits (required)
- Format: `<type>(<optional-scope>): <imperative summary>`
- Keep summary concise and action-oriented (for example: `fix(parser): handle nested skip depth`).
- Common types in this repo: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`.
- Use `!` or a `BREAKING CHANGE:` footer for breaking API/behavior changes.
- Add a body when useful to explain why the change was needed.

### Conventional Comments (required for review feedback)
- Structure comments as `<label> [non-blocking]: <message>`.
- Use labels from Conventional Comments where possible: `issue`, `suggestion`, `question`, `praise`, `nitpick`, `thought`, `chore`.
- Mark non-critical suggestions as `non-blocking`.
- Keep comments specific, actionable, and tied to observed code behavior.
- Avoid vague style-only feedback unless it violates documented repository conventions.

## Recommended Agent Workflow
1. Read relevant module(s) and nearby tests first.
2. Make minimal, targeted changes.
3. Run narrow validation first (often a single test).
4. Run broader checks when risk or blast radius increases.
5. Return concise results with commands and outcomes.
