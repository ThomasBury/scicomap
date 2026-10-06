<img src="pics/logo.png" alt="Scicomap logo" width="200"/>

[![Docs](https://img.shields.io/website?url=https%3A%2F%2Fthomasbury.github.io%2Fscicomap%2F&label=docs)](https://thomasbury.github.io/scicomap/)
[![Docs Quality](https://github.com/ThomasBury/scicomap/actions/workflows/docs.yml/badge.svg)](https://github.com/ThomasBury/scicomap/actions/workflows/docs.yml)
[![Marimo](https://img.shields.io/badge/marimo-live-1f7a8c.svg)](https://thomasbury.github.io/scicomap/marimo/index.html)
[![PyPI version](https://img.shields.io/pypi/v/scicomap.svg)](https://pypi.org/project/scicomap/)
[![Python](https://img.shields.io/pypi/pyversions/scicomap.svg)](https://pypi.org/project/scicomap/)
[![GitHub stars](https://img.shields.io/github/stars/ThomasBury/scicomap)](https://github.com/ThomasBury/scicomap/stargazers)

[buy me caffeine](https://ko-fi.com/V7V72SOHX)

# Scientific color maps

Scicomap helps you choose, assess, and improve scientific colormaps so your
figures remain readable and faithful to the underlying data.

- [Documentation](https://thomasbury.github.io/scicomap/)
- [Marimo demo](https://thomasbury.github.io/scicomap/marimo/index.html)

## Install

Python 3.10 or newer. v1 users should read the
[migration guide](https://thomasbury.github.io/scicomap/migrating-v2.html).

```shell
uv add 'scicomap>=2,<3'
```

## Inspect the original map

Python and CLI use the same diagnostics and selected map:

```python
import scicomap as sc

chart = sc.ScicoSequential("hawaii")
print(sc.diagnose_cmap(chart.cmap, chart.ctype)["status"])
chart.assess_cmap(figsize=(14, 6)).savefig("hawaii-original.png")
```

```shell
scicomap check hawaii --type sequential
scicomap preview hawaii --type sequential --out hawaii-original.png
```

## Correct and reuse

Correction is explicit. Save the exact corrected colors for later use:

```python
corrected = chart.unif_sym_cmap(lightness_rounding=0, bitonic=False)
chart.assess_cmap(figsize=(14, 6)).savefig("hawaii-corrected.png")
chart.export_cmap("hawaii.json")
```

```shell
scicomap fix hawaii --lightness-rounding 0 --no-bitonic --out hawaii-corrected.png --export hawaii.json
scicomap apply hawaii.json --image input.png --out mapped.png --json
```

Python plots return Figures; call `plt.show()` to display them. Only `wizard`
prompts. Every CLI command accepts `--json`, which never prompts or opens a
window; rendering requires `--out`. JSON responses use `ok`, `command`,
`inputs`, `data`, `warnings`, and `errors`. Artifacts include their kind,
absolute path, and selected map. Exit codes: 0 success, 2 invalid input,
1 operational failure.

Diagnostics are family-specific heuristics. CVD previews simulate selected
color-vision conditions and do not certify accessibility. Review the actual
figure, labels, contrast, and alternate encodings.

Scalar data, normalization, and reports are in the
[user guide](https://thomasbury.github.io/scicomap/user-guide.html).

- [Getting Started](https://thomasbury.github.io/scicomap/getting-started.html)
- [User Guide](https://thomasbury.github.io/scicomap/user-guide.html)
- [API Reference](https://thomasbury.github.io/scicomap/api-reference.html)
- [CLI Reference](https://thomasbury.github.io/scicomap/cli-reference.html)
- [FAQ](https://thomasbury.github.io/scicomap/faq.html) and [Troubleshooting](https://thomasbury.github.io/scicomap/troubleshooting.html)
- [LLM Access](https://thomasbury.github.io/scicomap/llm-access.html)

## Development

Use `just` recipes with `uv`'s project `.venv`. Routine commands use the
committed lockfile without updating it. Run `uv lock` when changing dependencies.

```shell
just sync          # lint and test extras only
just check         # ordinary tests, Ruff on src/tests/scripts, and ty on src/scicomap
just sync-docs     # add documentation tools; Pandoc binary required separately
just docs          # build web docs and LLM assets
just check-docs    # strict generated examples and browser bootstrap tests
```

Run a focused test with `uv run --locked python -m pytest tests/core/test_cmath.py`.
Tests marked `docs` need the docs extra and run separately through `just check-docs`.

Contribution guidelines are in `CONTRIBUTING.md`.
Release notes are in `CHANGELOG.md` and GitHub releases.

## Background

Scicomap uses CAM02-UCS lightness J', chroma C', and hue to assess and transform
colormaps. It builds on [ehtplot](https://github.com/liamedeiros/ehtplot) and
palettes from cmcrameri, cmasher, palettable, colorcet, and cmocean.
See the [introduction](https://thomasbury.github.io/scicomap/Introduction.html)
for color-space concepts and the [gallery](https://thomasbury.github.io/scicomap/gallery.html)
for the six map families. The longer write-up is the
[Towards Data Science post](https://towardsdatascience.com/your-colour-map-is-bad-heres-how-to-fix-it-lessons-learnt-from-the-event-horizon-telescope-b82523f09469).



## Star history

<a href="https://www.star-history.com/?repos=ThomasBury%2Fscicomap&type=date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ThomasBury/scicomap&type=date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ThomasBury/scicomap&type=date" />
   <img alt="Star history of ThomasBury/scicomap" src="https://api.star-history.com/chart?repos=ThomasBury/scicomap&type=date" />
 </picture>
</a>
