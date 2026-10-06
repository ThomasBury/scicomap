# AGENTS.md

## Scope and invariants
- Python >=3.10; package `src/scicomap`, tests `tests`, docs `docs/source`.
- Implement only the next unfinished `MILESTONES.md` item; verify and stop.
  Reviews do not authorize edits. Preserve unrelated work.
- Trace callers before editing; prefer existing code, stdlib, and installed
  dependencies. No speculative abstractions or compatibility shims for v2.
- Preserve sample count, endpoints/centers, alpha, finite color values, input
  validation, packaged resources, and caller random/plotting state.
- CAM02-UCS J' is lightness; chroma correction is distinct. Numerical hue
  angles are radians unless documented otherwise. Diagnostics are heuristics;
  CVD simulations do not certify accessibility.
- Public APIs use precise types, explicit exports, NumPy-style docstrings,
  and Figure/Colormap returns; plotting must not implicitly display.
- Mark deliberate limits with `ponytail:` and the condition for replacing them.

## Commands and checks
- Use `just` and `uv` in the single project `.venv`; no ad-hoc environments.
- `just sync`: locked lint/test extras. `just check`: ordinary pytest, Ruff
  on `src tests scripts`, and required ty on `src/scicomap`.
- `just sync-docs`: add docs extras; install the Pandoc binary separately.
- `just docs`: Sphinx and LLM mirrors. `just check-docs`: strict generated
  examples and WASM bootstrap tests. `just marimo`: export the browser app.
- `just validate-doc-artifacts`: verify exported assets. `just build`: packages.
- Routine sync/run commands use `--locked`. Run `uv lock` only when changing
  dependencies, and review `uv.lock`.
- Narrow check: `uv run --locked python -m pytest tests/core/test_cmath.py`.
  Parser edits: `tests/docs/test_build_llm_assets.py`; runtime edits: `just check`.
  Docs edits: `just docs check-docs`; release preparation: `just release-check`.
- Consult Context7 before framework/tool-specific configuration or debugging.
- Marimo: read UI values in later cells; WASM must install before importing
  through explicit cell dependencies. Verify browser helper packaging/paths.

## Delivery and releases
- Keep each change scoped to one outcome; update affected docs. Report exact
  checks and limitations. Use Conventional Commits and Conventional Comments.
- Use `gh` for GitHub operations. No destructive Git actions without approval.
- M1-M4 target main; M5-M9 target `feat/v2`. Version 2.0.0 belongs to M9.
- Publishing and pushing release tags require explicit authorization; verify
  the publish workflow with `gh run` before pushing a stable release tag.
