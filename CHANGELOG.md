# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog and this project follows Semantic
Versioning.

## [1.1.0] - Unreleased

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

[1.1.0]: https://github.com/ThomasBury/scicomap/compare/1.0.1...HEAD
[1.0.1]: https://github.com/ThomasBury/scicomap/releases/tag/1.0.1
