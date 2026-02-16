# scicomap 1.1.0

## Highlights

- Added a full human-friendly CLI workflow with explicit aliases and machine
  output support.
- Added CLI profiles (`quick-look`, `publication`, `presentation`,
  `cvd-safe`, `agent`) and profile-aware defaults in report/wizard flows.
- Added `wizard`, `doctor`, and `report` commands for guided diagnostics and
  artifact bundles.
- Improved docs information architecture, quick-start paths, and visual gallery.
- Added interactive Marimo tutorials (local full app + browser WASM lite app).

## User-facing improvements

- New and expanded CLI commands for listing, checking, comparing, fixing,
  color-vision diagnostics, image apply workflows, and one-command reports.
- Better documentation discoverability with task-based navigation and direct
  link to the browser tutorial.
- Python and CLI examples are now aligned and easier to follow.

## Fixes

- Fixed single-axis compare handling in CLI report/apply flow.
- Increased default figure height for CVD views to improve label readability.
- Hardened docs/pages workflows for Marimo deployment and artifact validation.
- Added explicit `.nojekyll` handling in Pages pipeline.

## Developer experience

- Migrated lint and formatting checks to Ruff.
- Added phased NumPy docstring enforcement policy.
- Added docs/CI checks for generated LLM assets.
- Added automated PyPI publishing workflow using Trusted Publishing on version
  tags (`Publish to PyPI`).
