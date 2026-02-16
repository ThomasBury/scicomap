import marimo

app = marimo.App()


@app.cell
async def _():
    import micropip

    await micropip.install("scicomap>=1.1.0")
    return


@app.cell
def _():
    import marimo as mo

    from scicomap.cmath import classify
    from scicomap.cmath import extrema
    from scicomap.cmath import get_ctab
    from scicomap.cmath import transform
    from scicomap.scicomap import SciCoMap
    from scicomap.scicomap import plot_colorblind_vision

    return (
        SciCoMap,
        classify,
        extrema,
        get_ctab,
        mo,
        plot_colorblind_vision,
        transform,
    )


@app.cell
def _(SciCoMap, mo):
    mo.md(
        """
        # scicomap interactive tutorial (WASM lite)

        This browser-friendly version focuses on diagnostics and accessibility.
        It installs `scicomap` in-browser with `micropip`, so first load may take
        a few extra seconds.
        For full workflows, run the local app with `marimo run docs/marimo/tutorial_app.py`.
        """
    )
    ctype = mo.ui.dropdown(
        options=[
            "sequential",
            "diverging",
            "multi-sequential",
            "circular",
            "miscellaneous",
            "qualitative",
        ],
        value="sequential",
        label="Colormap family",
    )
    cmap_names = sorted(SciCoMap(ctype=ctype.value).get_color_map_names())
    default_cmap = cmap_names[0]
    if ctype.value == "sequential" and "thermal" in cmap_names:
        default_cmap = "thermal"
    cmap = mo.ui.dropdown(
        options=cmap_names, value=default_cmap, label="Colormap"
    )
    n_colors = mo.ui.slider(
        16, 256, value=128, step=16, label="CVD color bins"
    )
    controls = mo.hstack([ctype, cmap, n_colors], gap=1.0, align="center")
    controls
    return cmap, ctype, n_colors


@app.cell
def _(SciCoMap, classify, cmap, ctype, extrema, get_ctab, transform):
    cmap_obj = SciCoMap.get_color_map_dic()[ctype.value][cmap.value]
    ctab = get_ctab(cmap_obj)
    jpapbp = transform(ctab)
    j_values = jpapbp[:, 0]
    is_monotonic = bool(
        (j_values[1:] >= j_values[:-1]).all()
        or (j_values[1:] <= j_values[:-1]).all()
    )
    cmap_class = classify(jpapbp)
    n_extrema = int(len(extrema(j_values)))
    status = "good"
    if (not is_monotonic) or cmap_class in {"asym_div", "unknown"}:
        status = "caution"
    diagnostics = {
        "status": status,
        "classification": cmap_class,
        "monotonic_lightness": is_monotonic,
        "extrema_count": n_extrema,
    }
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
