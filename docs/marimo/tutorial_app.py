import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from scicomap.cmath import classify
    from scicomap.cmath import extrema
    from scicomap.cmath import get_ctab
    from scicomap.cmath import transform
    from scicomap.scicomap import SciCoMap
    from scicomap.scicomap import compare_cmap
    from scicomap.scicomap import plot_colorblind_vision

    return (
        SciCoMap,
        classify,
        compare_cmap,
        extrema,
        get_ctab,
        mo,
        np,
        plot_colorblind_vision,
        plt,
        transform,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # scicomap interactive tutorial

        Explore colormaps, diagnose artifacts, test accessibility, and map the
        same decisions to CLI commands.
        """
    )
    return


@app.cell
def _(mo):
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
    return (ctype,)


@app.cell
def _(SciCoMap, ctype, mo):
    cmap_names = sorted(SciCoMap(ctype=ctype.value).get_color_map_names())
    default_cmap = cmap_names[0]
    if ctype.value == "sequential" and "thermal" in cmap_names:
        default_cmap = "thermal"

    cmap = mo.ui.dropdown(
        options=cmap_names,
        value=default_cmap,
        label="Colormap",
    )
    return (cmap,)


@app.cell
def _(mo):
    profile = mo.ui.dropdown(
        options=["quick-look", "publication", "presentation", "cvd-safe"],
        value="publication",
        label="Profile",
    )
    fix = mo.ui.checkbox(value=True, label="Apply fix")
    bitonic = mo.ui.checkbox(value=True, label="Bitonic")
    diffuse = mo.ui.checkbox(value=True, label="Diffuse")
    lift = mo.ui.slider(start=0, stop=40, value=10, step=1, label="Lift")
    n_colors = mo.ui.slider(
        start=16,
        stop=256,
        value=128,
        step=16,
        label="CVD color bins",
    )
    sample_image = mo.ui.dropdown(
        options=[
            "scan",
            "topography",
            "fn_roots",
            "phase",
            "grmhd",
            "vortex",
            "tng",
        ],
        value="scan",
        label="Sample image",
    )
    return bitonic, diffuse, fix, lift, n_colors, profile, sample_image


@app.cell
def _(bitonic, ctype, diffuse, fix, lift, mo, n_colors, profile, sample_image):
    controls = mo.vstack(
        [
            mo.md("## Controls"),
            ctype,
            profile,
            fix,
            bitonic,
            diffuse,
            lift,
            n_colors,
            sample_image,
        ],
        gap=0.5,
    )
    return (controls,)


@app.cell
def _(SciCoMap, classify, cmap, ctype, extrema, get_ctab, np, transform):
    cmap_obj = SciCoMap.get_color_map_dic()[ctype.value][cmap.value]
    ctab = get_ctab(cmap_obj)
    jpapbp = transform(ctab)
    j_values = jpapbp[:, 0]
    j_diff = np.diff(j_values)
    is_monotonic = bool(np.all(j_diff >= 0) or np.all(j_diff <= 0))
    cmap_class = classify(jpapbp)
    n_extrema = int(len(extrema(j_values)))

    status = "good"
    reasons = []
    if not is_monotonic:
        status = "fix-recommended"
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
        "reasons": reasons,
    }
    return (diagnostics,)


@app.cell
def _(diagnostics, mo):
    reason_lines = "\n".join(f"- {msg}" for msg in diagnostics["reasons"])
    if not reason_lines:
        reason_lines = "- No obvious issues detected."

    diag_md = mo.md(
        f"""
        ## Diagnostics

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
def _(SciCoMap, bitonic, cmap, ctype, diffuse, fix, lift):
    chart = SciCoMap(ctype=ctype.value, cmap=cmap.value)
    if fix.value:
        chart.unif_sym_cmap(
            lift=float(lift.value),
            bitonic=bitonic.value,
            diffuse=diffuse.value,
        )
    fig_preview = chart.assess_cmap(figsize=(14, 5.5))
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
def _(bitonic, cmap, compare_cmap, ctype, diffuse, fix, lift, sample_image):
    fig_apply = compare_cmap(
        image=sample_image.value,
        ctype=ctype.value,
        cm_list=[cmap.value],
        ncols=1,
        title=False,
        uniformize=fix.value,
        symmetrize=fix.value,
        unif_kwargs={"lift": float(lift.value)},
        sym_kwargs={"bitonic": bitonic.value, "diffuse": diffuse.value},
        facecolor="white",
        figsize=(6.5, 4.5),
    )
    return (fig_apply,)


@app.cell
def _(bitonic, cmap, ctype, diffuse, fix, lift, mo, sample_image):
    cmd_check = f"scicomap check {cmap.value} --type {ctype.value}"
    cmd_fix = (
        "scicomap fix "
        f"{cmap.value} --type {ctype.value} --lift {float(lift.value):.0f} "
        f"{'--bitonic' if bitonic.value else '--no-bitonic'} "
        f"{'--diffuse' if diffuse.value else '--no-diffuse'}"
    )
    cmd_cvd = f"scicomap cvd {cmap.value} --type {ctype.value} --n-colors 256"
    cmd_apply = (
        f"scicomap compare {cmap.value} viridis --type {ctype.value} "
        f"--image {sample_image.value}"
    )

    if not fix.value:
        cmd_fix = f"# fix disabled\n{cmd_fix}"

    cli_md = mo.md(
        f"""
        ## Equivalent CLI

        ```bash
        {cmd_check}
        {cmd_fix}
        {cmd_cvd}
        {cmd_apply}
        ```
        """
    )
    return (cli_md,)


@app.cell
def _(cli_md, fig_apply, fig_cvd, mo):
    secondary_tabs = mo.ui.tabs(
        {
            "Color-vision deficiency": fig_cvd,
            "Apply to sample image": fig_apply,
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
