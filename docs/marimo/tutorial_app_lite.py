import marimo

app = marimo.App(width="full")


@app.cell
async def _():
    import micropip
    import marimo as _mo
    from pyodide.http import pyfetch

    await micropip.install("scicomap>=1.1.0")
    response = await pyfetch(
        str(_mo.notebook_location() / "public" / "_diagnostics.py")
    )
    if not response.ok:
        raise OSError("Cannot load the shared diagnostic module.")
    with open("_scicomap_diagnostics.py", "w", encoding="utf-8") as module:
        module.write(await response.string())
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

    def build_cmap_options(sci_co_map_cls, ctype):
        cmap_names = sorted(sci_co_map_cls(ctype=ctype).get_color_map_names())
        default_cmap = cmap_names[0]
        if ctype == "sequential" and "thermal" in cmap_names:
            default_cmap = "thermal"
        return cmap_names, default_cmap

    import marimo as mo

    from _scicomap_diagnostics import _diagnose_cmap as diagnose_cmap
    from scicomap.scicomap import SciCoMap
    from scicomap.scicomap import plot_colorblind_vision

    return (
        COLORMAP_FAMILIES,
        SciCoMap,
        build_cmap_options,
        diagnose_cmap,
        mo,
        plot_colorblind_vision,
    )


@app.cell
def _(mo):
    mo.md(
        """
# scicomap interactive tutorial

Perceptual uniformity is the idea that Euclidean distance between colors in color space should match human color perception distance judgements.

**Data should speak for itself, not for the color map.**
Using the wrong gradient can lead to "optical illusions" where your data looks broken or banded when it is actually smooth.

| Problem | Consequence | Example |
| --- | --- | --- |
| **Uneven Gradients** | Creates "false boundaries" (artifacts). | The infamous **`jet`** map. |
| **Non-Linearity** | Distorts the perceived magnitude of data. | A 10% change looks like 50% in certain zones. |
| **Color-Vision Deficiency (CVD)** | Excludes **8% of the male population**. | Red-Green maps that look identical to a color-blind user. |
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
| **Lightness (`J'`)** | **The Scalar Value** | Must vary **linearly** with the data. If the data goes up, the brightness must follow smoothly. |
| **Hue (`h'`)** | **Appeal & Clarity** | Ideal for making a map attractive. It can encode an extra variable if it changes at a constant rate. |
| **Chroma (`C'`)** | **Aesthetics Only** | **Do not use for data.** Humans struggle to distinguish subtle saturation changes accurately. |


### The "Scicomap" Uniformization Process

To "fix" a problematic color map, we follow a rigorous scientific recipe:

1. **Linearize Lightness:** We force `J'` into a straight line so that the visual weight matches the data points.
2. **Round the Floor:** `lift` rounds the lower lightness bound up to a multiple of that step. `None` and `0` leave it unchanged.
3. **Smooth the Chroma:** We symmetrize the `C'` curve to remove "kinks" or sharp edges.
4. **Remove Artifacts:** We avoid abrupt changes in the chroma trajectory to prevent the eye from seeing "steps" that don't exist in the data.
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
def _(SciCoMap, build_cmap_options, ctype, mo):
    cmap_names, default_cmap = build_cmap_options(SciCoMap, ctype.value)
    cmap = mo.ui.dropdown(
        options=cmap_names, value=default_cmap, label="Colormap"
    )
    return (cmap,)


@app.cell
def _(mo):
    n_colors = mo.ui.slider(
        16, 256, value=128, step=16, label="CVD color bins"
    )
    return (n_colors,)


@app.cell
def _(cmap, ctype, mo, n_colors):
    controls = mo.hstack([ctype, cmap, n_colors], gap=1.0, align="center")
    controls
    return


@app.cell
def _(SciCoMap, cmap, ctype):
    chart = SciCoMap(ctype=ctype.value, cmap=cmap.value)
    selected_map = chart.get_mpl_color_map()
    return chart, selected_map


@app.cell
def _(diagnose_cmap, ctype, selected_map):
    diagnostics = diagnose_cmap(selected_map, ctype.value)
    return (diagnostics,)


@app.cell
def _(cmap, ctype, diagnostics, mo):
    diag_md = mo.md(
        f"""
## Diagnostics for the selected map

Statuses are lightness heuristics. CVD simulations do not certify accessibility.

- **Status:** `{diagnostics["status"]}`
- **Class:** `{diagnostics["classification"]}`
- **Monotonic lightness:** `{diagnostics["monotonic_lightness"]}`
- **Extrema count:** `{diagnostics["extrema_count"]}`

```bash
scicomap check {cmap.value} --type {ctype.value}
```
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
def _(cmap, ctype, mo, n_colors):
    cli_md = mo.md(
        f"""
## Equivalent CLI

```bash
scicomap check {cmap.value} --type {ctype.value}
scicomap cvd {cmap.value} --type {ctype.value} --n-colors {int(n_colors.value)}
```
        """
    )
    return (cli_md,)


@app.cell
def _(cli_md, diag_md, fig_cvd, fig_preview, mo):
    tabs = mo.ui.tabs(
        {
            "Diagnostics": diag_md,
            "Color-vision deficiency": fig_cvd,
            "Equivalent CLI": cli_md,
        }
    )
    mo.vstack(
        [mo.md("## Preview"), fig_preview, mo.md("## Explore more"), tabs],
        gap=0.75,
    )
    return


if __name__ == "__main__":
    app.run()
