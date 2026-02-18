import marimo

app = marimo.App(width="full")


@app.cell
async def _():
    import micropip

    await micropip.install("scicomap>=1.1.0")
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

    def compute_diagnostics(
        jpapbp, classify_fn, extrema_fn, include_reasons=False
    ):
        j_values = jpapbp[:, 0]
        is_monotonic = bool(
            (j_values[1:] >= j_values[:-1]).all()
            or (j_values[1:] <= j_values[:-1]).all()
        )
        cmap_class = classify_fn(jpapbp)
        n_extrema = int(len(extrema_fn(j_values)))

        status = "good"
        reasons = []
        if not is_monotonic:
            status = "fix-recommended" if include_reasons else "caution"
            reasons.append("Lightness is not monotonic.")
        elif cmap_class in {"asym_div", "unknown"}:
            status = "caution"
            reasons.append(f"Classification is '{cmap_class}'.")

        if n_extrema > 2:
            if status == "good":
                status = "caution"
            reasons.append("Lightness has multiple extrema.")

        diagnostics = {
            "status": status,
            "classification": cmap_class,
            "monotonic_lightness": is_monotonic,
            "extrema_count": n_extrema,
        }
        if include_reasons:
            diagnostics["reasons"] = reasons
        return diagnostics

    import marimo as mo

    from scicomap.cmath import classify
    from scicomap.cmath import extrema
    from scicomap.cmath import get_ctab
    from scicomap.cmath import transform
    from scicomap.scicomap import SciCoMap
    from scicomap.scicomap import plot_colorblind_vision

    return (
        COLORMAP_FAMILIES,
        SciCoMap,
        build_cmap_options,
        classify,
        compute_diagnostics,
        extrema,
        get_ctab,
        mo,
        plot_colorblind_vision,
        transform,
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
2. **Lift the Floor:** we increase the minimum lightness to prevent data from disappearing into "pure black" shadows.
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
def _(
    SciCoMap,
    classify,
    cmap,
    compute_diagnostics,
    ctype,
    extrema,
    get_ctab,
    transform,
):
    cmap_obj = SciCoMap.get_color_map_dic()[ctype.value][cmap.value]
    ctab = get_ctab(cmap_obj)
    jpapbp = transform(ctab)
    diagnostics = compute_diagnostics(jpapbp, classify, extrema)
    return (diagnostics,)


@app.cell
def _(cmap, ctype, diagnostics, mo):
    diag_md = mo.md(
        f"""
## Diagnostics

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
def _(SciCoMap, cmap, ctype):
    fig_preview = SciCoMap(ctype=ctype.value, cmap=cmap.value).assess_cmap(
        figsize=(14, 5.5)
    )
    return (fig_preview,)


@app.cell
def _(cmap, ctype, n_colors, plot_colorblind_vision):
    fig_cvd = plot_colorblind_vision(
        ctype=ctype.value,
        cmap_list=[cmap.value],
        n_colors=int(n_colors.value),
        facecolor="white",
        uniformize=False,
        symmetrize=False,
    )
    return (fig_cvd,)


@app.cell
def _(cmap, ctype, mo):
    cli_md = mo.md(
        f"""
## Equivalent CLI

```bash
scicomap check {cmap.value} --type {ctype.value}
scicomap cvd {cmap.value} --type {ctype.value} --n-colors 128
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
