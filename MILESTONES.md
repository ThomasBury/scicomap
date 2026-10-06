# Milestones

Modernize Scicomap in two stages: repair the existing version, then build
**2.0.0** on a separate branch.

Python and CLI users have equal priority. Complete one independently reviewable
milestone at a time. Each milestone must pass its relevant checks before starting
the next.

## Branch and release strategy

- **M1–M4:** work on bug-fix branches based on current `main` (`fix/v1-bugs`
  for M1, `fix/v1-m2` for M2). After merging and deleting a branch, start a
  fresh branch from updated `main`. Preserve existing public names, signatures,
  commands, and output shapes. Correct behavior that contradicts the documented
  operation.
- Merge the completed bug fixes into `main` and prepare the next patch release.
- Create **`feat/v2` from the stabilized `main`** before starting M5.
- **M5–M9:** implement breaking changes for **2.0.0**. No compatibility shims,
  deprecated forwarding aliases, or support for legacy behavior.
- This document does not authorize implementing additional milestones, publishing
  packages, or pushing release tags.

## Stage 1 — Fix existing bugs

### [x] M1 — File safety and image handling

**Outcome:** commands preserve user files, image transparency, and caller state.

- Replace `doctor`’s fixed probe filename with a uniquely created temporary file.
- Share image reading, scalar conversion, and remapping across `apply`, wizard,
  and report.
- Preserve input alpha and handle malformed images with actionable errors.
- Replace global random seeding in examples with local deterministic generators.

**Acceptance:** regression checks prove that existing files survive `doctor`,
transparent pixels remain transparent, all image workflows agree, and examples
leave the caller’s random state unchanged.

**Completed:** 72 tests passed; Ruff lint and formatting checks passed; the
documentation build and LLM asset generation passed.

### [x] M2 — Numerical correctness, catalog entries, and plotting

**Outcome:** transformations preserve their input contracts and ordinary plotting
calls work.

- Preserve sample count, endpoints, center handling, and alpha when transforming
  odd-length diverging maps.
- Correct `hawaii_r` and `nuuk_r`; check every catalog reversal pair.
- Convert numerical color tables to floating point and validate supported shapes
  and values.
- Support documented color-name lists through Matplotlib’s existing conversion
  functions.
- Treat `lift=0` as no lift; document the existing rounding behavior and hue angles
  in radians.
- Return an honest `uniformized` flag when a map cannot be uniformized.
- Repair family discovery, single-palette plotting, pyramid loading, and incorrect
  coordinate unpacking in examples.

**Acceptance:** focused numerical regressions pass; every catalog map is checked
for finite output and preserved sample count; reversal pairs reverse correctly;
the previously failing plotting calls succeed.

**Completed:** `just check` passed (146 tests, Ruff lint and formatting).
All 181 catalog maps produced finite transformed output with their sample
count and alpha preserved; every catalog reversal pair passed. Numerical
regressions cover odd/even centers, asymmetric endpoints, integer and invalid
tables, color-name lists, zero lift, unknown-map flags, and hue broadcasting.
Plotting regressions cover family discovery, single palettes, pyramid data,
periodic coordinates, and rendered examples. The Sphinx build and LLM asset
generation passed.

**V1 constraint:** the legacy `ScicoMultiSequential()` default (`chroma`) is
not in its family catalog. Its signature remains unchanged; example checks
use the supported `cmap="bukavu"`. Resolve this constructor inconsistency in
M5, where breaking signatures are allowed.

### [x] M3 — Trustworthy diagnostics and CLI workflows

**Outcome:** requested stages run correctly and every artifact uses the intended
map.

- Assess lightness according to family: sequential progression, diverging
  branches, circular behavior, and unordered categorical colors.
- Share diagnostics between the CLI and tutorials.
- Prepare original and transformed maps once; use the selected object
  consistently for assessment, CVD simulation, and image application.
- Make wizard and report honor stage flags and apply goals.
- Collect interactive inputs before validating required values.
- Validate counts and paths before creating artifacts; make existing JSON modes
  handle validation and expected runtime failures consistently.
- Document that diagnostic statuses are heuristics and CVD simulations do not
  certify accessibility.

**Acceptance:** regressions cover each family, disabled stages, interactive apply,
agent apply, original versus transformed artifacts, invalid counts, and parseable
JSON failures. Existing public interfaces remain available.

**Completed:** `just check` passed (207 tests, Ruff lint and formatting).
Family-aware diagnostics are shared by the CLI and both tutorials. Workflow
regressions cover disabled stages, interactive and agent apply, one-time map
correction, original versus corrected assessments/simulations/applied images,
invalid counts and paths, finite lift values, parseable JSON failures, and
preserved usage/help guidance for text errors.
The v1 `cvd-safe` profile's documented CVD enforcement remains unchanged.
`just docs`, strict Sphinx validation, LLM asset generation, full local
tutorial execution, and `just marimo validate-doc-artifacts` passed.
The additional checks used these exact commands:

```bash
UV_PROJECT_ENVIRONMENT=.venv.just uv run sphinx-build -n -W -b html docs/source docs/build/html
UV_PROJECT_ENVIRONMENT=.venv.just uv run marimo export html docs/marimo/tutorial_app.py -o /tmp/scicomap-m3-tutorial.html
```

The browser export includes the shared
diagnostic module; a bootstrap regression verifies dependency installation,
worker-relative URL resolution, and importing that module. Marimo retains its
existing formatting warnings. Live browser verification remains part of M9.

### [x] M4 — Documentation integrity and v1 package readiness

**Outcome:** documented examples and installed commands work outside the checkout.

- Preserve code whitespace and table structure in Markdown mirrors.
- Test syntax-highlighted HTML fixtures and examples from actual generated
  documentation.
- Package the existing documentation generator so advertised v1 documentation
  commands work from a wheel; keep the repository script as a thin entry point.
- Correct inaccurate units, transformation descriptions, example outputs, and
  documentation version metadata.
- Record the fixes in patch-release notes.

**Acceptance:** tests and current lint checks pass; strict documentation builds
pass; generated examples remain executable; wheel smoke checks verify data
resources and advertised commands.

**Completed:** `just check` passed (217 tests, Ruff lint and formatting).
Parser regressions cover highlighted tokens, indentation, blank lines, inline
code, code blocks nested in lists, table rows/cells, and installed docs command aliases. A fresh strict
Sphinx build and all 34 generated Python examples passed, including exact
notebook cell comparisons after Sphinx trims trailing line whitespace.
Numerical, dataset, and family-class docstring examples passed.
The documentation version comes from the package version; units, chroma
correction descriptions, sample inputs/results, family output, and contributor
commands are corrected. Patch fixes are recorded in `CHANGELOG.md`.
`just build` passed for the sdist and wheel, including Twine validation.
An isolated wheel installation verified all five packaged data resources and
13 v1 commands outside the checkout. Typer is bounded below 0.26 because that
release replaced the Click group internals used by the v1 JSON error handling;
revisit the bound during M6. Additional validation used:

```bash
UV_PROJECT_ENVIRONMENT=.venv.just uv run sphinx-build -n -W -b html docs/source docs/build/html
UV_PROJECT_ENVIRONMENT=.venv.just uv run python scripts/build_llm_assets.py
uv run --isolated --no-project --with ./dist/scicomap-1.1.1-py3-none-any.whl python scripts/smoke_wheel.py
UV_PROJECT_ENVIRONMENT=.venv.just uv run ruff check scripts/build_llm_assets.py scripts/smoke_wheel.py
UV_PROJECT_ENVIRONMENT=.venv.just uv run ruff format --check scripts/build_llm_assets.py scripts/smoke_wheel.py
uv lock --check
```

**Branch transition:** merge the completed fixes into `main`, then create
`feat/v2`. Breaking changes begin only after this checkpoint.

## Stage 2 — Build v2 without backward compatibility

### [ ] M5 — Consistent Python API

**Outcome:** users can discover, inspect, transform, and plot maps through a
coherent interface.

- Retain `SciCoMap` and useful family conveniences; consolidate their duplicated
  implementation and documentation.
- Resolve and validate colormaps at construction so object state has consistent
  types.
- Define intended public exports explicitly; remove accidental dependency exports
  and redundant discovery APIs.
- Expose one structured diagnostic function for Python, CLI, and tutorials.
- Make plotting APIs return `Figure` objects without implicit display.
- Make transformation APIs return the resulting Matplotlib colormap consistently;
  remove caller-managed `uniformized` controls.
- Give lightness parameters precise names and documented meanings.

**Acceptance:** public API tests cover discovery, supported inputs, return types,
transformations, and absence of hidden plotting side effects. Removed interfaces
receive no forwarding aliases.

### [ ] M6 — Smaller CLI for people and agents

**Outcome:** one command surface provides predictable human and machine behavior.

- Keep Typer and one canonical command hierarchy; remove duplicate `cmap` aliases
  and product-facing documentation-build commands.
- Remove profile-based precedence and enforcement; use explicit options and a
  guided wizard.
- Make inspection the default. Transformation happens only when explicitly
  requested.
- Keep prompting confined to wizard; machine mode never prompts or opens windows.
- Standardize JSON output, structured family lists, artifact paths, and exit codes
  across commands.
- Implement commands using shared package functions rather than duplicated
  workflow logic.

**Acceptance:** human and JSON modes exercise the same operations; failure output
is parseable; rendering in machine mode requires an output destination; help and
tests contain no removed commands or profiles.

### [ ] M7 — Reusable corrections and useful reports

**Outcome:** users can reuse a correction and judge what changed.

- Export corrected color tables with the parameters and package version needed to
  reproduce them.
- Support loading exported tables into Matplotlib through the existing colormap
  APIs.
- Report original and transformed diagnostics separately.
- Clearly identify which map each simulation and applied image uses.
- Replace accessibility guarantees with descriptions of the simulations
  performed.
- Explain data range and midpoint selection using Matplotlib’s existing
  normalization objects.

**Acceptance:** exported maps reload and reproduce their sampled colors; reports
agree with Python results; examples apply the exported map to real scalar data.

### [ ] M8 — Modern tooling and short AGENTS.md

**Outcome:** development checks are reproducible and contributor guidance matches
reality.

- Use one uv project environment and locked routine synchronization.
- Keep existing dependency extras; separate ordinary checks from documentation
  setup.
- Add ty to the lint extra and make it a required check for `src/scicomap` after
  resolving the baseline.
- Handle dynamic provider attributes narrowly; remove broad Ruff unused-code
  exemptions.
- Include maintained scripts in lint and formatting checks.
- Remove unused dependencies and private code after checking all references.
- Replace AGENTS.md with fewer than 45 lines covering scope, scientific
  invariants, commands, tests, and release boundaries.
- Keep NumPy, SciPy, Matplotlib, Colorspacious, necessary palette providers, Typer,
  and Rich. Add no Pydantic dependency for current array or CLI validation.

**Acceptance:** `just check` runs pytest, Ruff, and ty successfully in a freshly
synchronized environment. Documented commands match the justfile and routine
checks do not update the lockfile.

### [ ] M9 — Documentation, browser tutorial, and 2.0 release readiness

**Outcome:** the new version is understandable and verified as an installed
product.

- Rewrite onboarding around matching Python and CLI workflows.
- Generate public API signatures and parameter documentation with existing Sphinx
  tooling.
- Remove stale aliases, profiles, compatibility guidance, and duplicated inherited
  documentation.
- Make both tutorials use shared diagnostics and consistent map selection.
- Verify browser WASM against the candidate v2 package rather than a floating v1
  installation.
- Document breaking changes and provide concise before/after usage examples.
- Set the release version to 2.0.0 and prepare release notes.

**Acceptance:** full checks, strict docs builds, executable examples, browser
verification, installed-wheel smoke checks, and `just release-check` pass. CI
covers the declared minimum Python version and a recent supported version.
Publishing remains a separate authorized action.
