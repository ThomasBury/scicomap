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

    def compute_diagnostics(jpapbp, classify_fn, extrema_fn, include_reasons=False):
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
        # scicomap interactive tutorial (WASM lite)

        This browser-friendly version focuses on diagnostics and accessibility.
        It installs `scicomap` in-browser with `micropip`, so first load may take
        a few extra seconds.
        For full workflows, run the local app with `marimo run docs/marimo/tutorial_app.py`.
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

        ### Core coordinates (CAM02-UCS)

        | Symbol | Intuition | Why it matters |
        |---|---|---|
        | `J'` | Lightness (dark -> bright) | For scalar data, smooth and mostly monotonic `J'` usually gives more faithful gradients. |
        | `a'` | Green <-> Red opponent axis | Together with `b'`, it defines the chromatic direction of colors. |
        | `b'` | Blue <-> Yellow opponent axis | Together with `a'`, it defines hue/chroma behavior. |

        ### Derived cylindrical view

        - `C'` (chroma): colorfulness/saturation, approximately the radius in the `(a', b')` plane.
        - `h'` (hue angle): the color direction around that plane.

        ### How to read the assessment outputs

        - If **`J'` is monotonic** (or close), the map usually encodes value changes more reliably.
        - Many **`J'` extrema or kinks** can create visual artifacts (false boundaries or bands).
        - Strong asymmetry or abrupt trajectory changes in `a'`, `b'`, `C'`, or `h'` can reduce interpretability.
        - CVD previews help check whether structure remains visible under color-vision deficiencies.
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
