import marimo

app = marimo.App(width="full")


@app.cell
async def _():
    import micropip
    import marimo as _mo

    wheel_url = str(
        _mo.notebook_location() / "public" / "scicomap-2.0.0-py3-none-any.whl"
    )
    await micropip.install(wheel_url)
    wasm_deps_ready = True
    return (wasm_deps_ready,)


@app.cell
def _(wasm_deps_ready):
    _ = wasm_deps_ready
    COLORMAP_FAMILIES = (
        "sequential",
        "diverging",
        "multi-sequential",
        "circular",
        "miscellaneous",
        "qualitative",
    )

    def build_cmap_options(ctype):
        cmap_names = sorted(get_cmap_dict()[ctype])
        default_cmap = cmap_names[0]
        if ctype == "sequential" and "thermal" in cmap_names:
            default_cmap = "thermal"
        return cmap_names, default_cmap

    import marimo as mo

    from scicomap import (
        SciCoMap,
        get_cmap_dict,
        diagnose_cmap,
        plot_colorblind_vision,
        __version__ as scicomap_version,
    )

    if scicomap_version != "2.0.0":
        raise RuntimeError(
            "The browser tutorial requires the bundled v2 candidate."
        )

    return (
        scicomap_version,
        COLORMAP_FAMILIES,
        SciCoMap,
        build_cmap_options,
        diagnose_cmap,
        mo,
        plot_colorblind_vision,
    )


@app.cell
def _(scicomap_version, mo):
    mo.md(
        f"""
# scicomap {scicomap_version} interactive tutorial

Explore colormaps, diagnose artifacts, simulate color-vision deficiencies, and map the
same decisions to CLI commands.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Reading the color-space diagnostics (quick intuition)

> These notes are intentionally simplified.
> They are **not** a full color-science treatment; they are here to help you interpret the charts.

### How to Encode Information Correctly

| Attribute | Role in Encoding | Rule of Thumb |
| --- | --- | --- |
| **Lightness (`J'`)** | **Ordered Values** | Sequential maps should progress smoothly; diverging maps use two branches around a reference. |
| **Hue (`h'`)** | **Categories & Cycles** | Use with labels or alternate encodings; numerical hue angles are in radians. |
| **Chroma (`C'`)** | **Distance from Gray** | Chroma is distinct from lightness; abrupt changes can introduce false boundaries. |


### The "Scicomap" Uniformization Process

Correction linearizes recognized lightness patterns and adjusts chroma.
Review the result on your actual data: these operations do not certify accessibility.

1. **Linearize Lightness:** Make recognized sequential or diverging lightness branches linear.
2. **Round the Floor:** `lightness_rounding` rounds the lower lightness bound up to a multiple of that step. `None` and `0` leave it unchanged.
3. **Smooth the Chroma:** We symmetrize the `C'` curve to remove "kinks" or sharp edges.
4. **Remove Artifacts:** Reassess the transformed map for abrupt transitions and remaining artifacts.
        """
    )
    return


@app.cell
def _(COLORMAP_FAMILIES, mo):
    ctype = mo.ui.dropdown(
        options=list(COLORMAP_FAMILIES),
        value="sequential",
        label="Colormap family",
    )
    return (ctype,)


@app.cell
def _(build_cmap_options, ctype, mo):
    cmap_names, default_cmap = build_cmap_options(ctype.value)

    cmap = mo.ui.dropdown(
        options=cmap_names,
        value=default_cmap,
        label="Colormap",
    )
    return (cmap,)


@app.cell
def _(mo):
    fix = mo.ui.checkbox(value=False, label="Apply fix")
    bitonic = mo.ui.checkbox(value=True, label="Bitonic")
    diffuse = mo.ui.checkbox(value=True, label="Diffuse")
    lightness_rounding = mo.ui.slider(
        start=0, stop=40, value=10, step=1, label="Lightness rounding step"
    )
    n_colors = mo.ui.slider(
        start=16,
        stop=256,
        value=128,
        step=16,
        label="CVD color bins",
    )
    return bitonic, diffuse, fix, lightness_rounding, n_colors


@app.cell
def _(bitonic, cmap, ctype, diffuse, fix, lightness_rounding, mo, n_colors):
    controls = mo.vstack(
        [
            mo.md("## Controls"),
            ctype,
            cmap,
            fix,
            bitonic,
            diffuse,
            lightness_rounding,
            n_colors,
        ],
        gap=0.5,
    )
    return (controls,)


@app.cell
def _(SciCoMap, bitonic, cmap, ctype, diffuse, fix, lightness_rounding):
    chart = SciCoMap(ctype=ctype.value, cmap=cmap.value)
    if fix.value:
        chart.unif_sym_cmap(
            lightness_rounding=float(lightness_rounding.value),
            bitonic=bitonic.value,
            diffuse=diffuse.value,
        )
    selected_map = chart.get_mpl_color_map()
    return chart, selected_map


@app.cell
def _(diagnose_cmap, ctype, selected_map):
    diagnostics = diagnose_cmap(selected_map, ctype.value)
    return (diagnostics,)


@app.cell
def _(diagnostics, mo):
    reason_lines = "\n".join(f"- {msg}" for msg in diagnostics["reasons"])
    if not reason_lines:
        reason_lines = "- No obvious issues detected."

    diag_md = mo.md(
        f"""
## Diagnostics for the selected map

Statuses are lightness heuristics. CVD simulations do not certify accessibility.

- **Status:** `{diagnostics["status"]}`
- **Class:** `{diagnostics["classification"]}`
- **Monotonic lightness:** `{diagnostics["monotonic_lightness"]}`
- **Extrema count:** `{diagnostics["extrema_count"]}`

**Reasons**
{reason_lines}
        """
    )
    return (diag_md,)


@app.cell
def _(chart):
    fig_preview = chart.assess_cmap(figsize=(14, 5.5))
    return (fig_preview,)


@app.cell
def _(ctype, n_colors, plot_colorblind_vision, selected_map):
    fig_cvd = plot_colorblind_vision(
        ctype=ctype.value,
        cmap_list=[selected_map],
        n_colors=int(n_colors.value),
        facecolor="white",
        uniformize=False,
        symmetrize=False,
    )
    return (fig_cvd,)


@app.cell
def _(bitonic, cmap, ctype, diffuse, fix, lightness_rounding, mo):
    cmd_report = (
        f"scicomap report --cmap {cmap.value} --type {ctype.value} "
        f"{'--fix' if fix.value else '--no-fix'} --cvd "
        f"--lightness-rounding {float(lightness_rounding.value):.0f} "
        f"{'--bitonic' if bitonic.value else '--no-bitonic'} "
        f"{'--diffuse' if diffuse.value else '--no-diffuse'} "
        "--out tutorial-report"
    )
    cli_md = mo.md(
        f"""
## Matching map and stages in the CLI

The report uses the same correction and stage choices. Its CVD simulation
uses 256 color bins.

```bash
{cmd_report}
```
        """
    )
    return (cli_md,)


@app.cell
def _(cli_md, fig_cvd, mo):
    secondary_tabs = mo.ui.tabs(
        {
            "Color-vision deficiency": fig_cvd,
            "Equivalent CLI": cli_md,
        }
    )
    return (secondary_tabs,)


@app.cell
def _(controls, diag_md, fig_preview, mo, secondary_tabs):
    mo.vstack(
        [
            diag_md,
            mo.md("## Preview"),
            fig_preview,
            mo.md("## Explore more"),
            mo.hstack([controls, secondary_tabs], gap=1.0, align="start"),
        ],
        gap=0.75,
    )
    return


if __name__ == "__main__":
    app.run()
