# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project follows Semantic
Versioning.

## [2.0.0] - Unreleased

### Added

- One public `diagnose_cmap` function shared by Python, CLI, and tutorials.
- Reusable RGBA table exports with source colors, ordered correction parameters,
  family, name, and package version; CLI commands load exported tables.
- Reports with separate original and transformed diagnostics, labeled artifacts,
  and the CVD simulation conditions actually used.
- Generated public API signatures and parameter documentation, matching Python
  and CLI onboarding, and a concise v1-to-v2 migration guide.
- Browser tutorial installs the candidate wheel served with the documentation.
  CI checks Python 3.10 and 3.14; release checks include installed-wheel smoke tests.

### Changed

- Constructors resolve and validate maps immediately. Family conveniences share
  `SciCoMap` operations; the multi-sequential default is now `bukavu`.
- Transformation functions and methods return Colormaps directly. Methods replace
  object state. `lightness_rounding` names the lower-bound rounding operation.
- Plotting returns Figures without implicit display. CVD methods use the current
  map. Standalone plotting retains explicit correction options.
- CLI inspection preserves original maps; correction, simulation, and application
  require explicit stage choices. Only wizard prompts; JSON mode stays headless.
- Every CLI command uses the same JSON envelope, absolute artifact paths, and
  exit codes (0 success, 2 invalid input, 1 operational failure).
- Required Typer is now >=0.26.0. Development uses one locked uv environment;
  ordinary checks require pytest, Ruff, and ty, with docs checks separate.
- Diagnostics and simulations describe heuristics and conditions, without
  accessibility guarantees.

### Removed

- Redundant discovery methods, accidental dependency exports, Python `lift` and
  caller-controlled `uniformized` arguments/state, and tuple transformation returns.
- Duplicate `cmap` CLI aliases, documentation-build commands, profiles, inferred
  workflow goals, and alternate output-format options. No forwarding aliases remain.
- Unused dependencies and private CVD converters.

See [the migration guide](docs/source/migrating-v2.rst) for before/after examples.
Publishing 2.0.0 remains a separate release action.

## [1.1.1] - Unreleased

### Fixed

- Preserve user files during CLI environment probes, image alpha across apply,
  wizard, and report, and caller random state in examples.
- Preserve transformed sample counts, centers, endpoints, and alpha; validate
  color tables and repair reversed catalog entries and plotting/data helpers.
- Share family-aware diagnostics and honor workflow stages and selected maps;
  validate inputs before creating artifacts and report JSON failures consistently.
- Preserve syntax-highlighted code whitespace and table rows in Markdown mirrors.
- Package the documentation generator so `docs-llm` and `docs llm-assets` work
  from installed wheels, with the repository script using the same implementation.
- Bound Typer below 0.26, whose vendored Click breaks the v1 custom command group.
- Correct numerical example inputs and results, chroma/lightness descriptions,
  hue units, family-list output, contributor checks, and documentation version
  metadata. Diagnostic statuses and CVD simulations do not certify accessibility.

- Raise explicit `TypeError` for invalid inputs in `cmath.get_ctab` and
  `datasets.load_pic`.
- Harden `cmath.max_chroma` scalar/array behavior with consistent broadcasting
  and scalar return handling.

## [1.1.0] - 2026-02-18

### Added

- Human-friendly CLI workflows with explicit aliases and machine-readable output.
- CLI profiles (`quick-look`, `publication`, `presentation`, `cvd-safe`,
  `agent`) with profile-aware defaults.
- Guided CLI commands (`wizard`, `doctor`, `report`) for diagnostics and
  reproducible artifact bundles.
- Interactive Marimo tutorials (local full app and browser WASM lite app).
- Trusted Publishing workflows for PyPI and TestPyPI (rc tags).

### Changed

- Documentation information architecture improved with task-based navigation,
  gallery pages, and direct tutorial links.
- Lint and format checks migrated to Ruff.

### Fixed

- Single-axis colormap comparison handling in report/apply workflows.
- Color-vision diagnostics default figure height for better label readability.
- Pages deployment hardening for Marimo artifacts and `.nojekyll` handling.

## [1.0.1] - 2024-05-15

### Added

- Initial stable package release on PyPI.

[2.0.0]: https://github.com/ThomasBury/scicomap/compare/1.1.0...HEAD
[1.1.1]: https://github.com/ThomasBury/scicomap/compare/1.1.0...11d9da5
[1.1.0]: https://github.com/ThomasBury/scicomap/releases/tag/1.1.0
[1.0.1]: https://github.com/ThomasBury/scicomap/releases/tag/1.0.1
