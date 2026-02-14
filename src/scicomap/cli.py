"""Command-line interface for scicomap."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from scicomap.cmath import classify, extrema, get_ctab, transform
from scicomap.scicomap import SciCoMap, compare_cmap, plot_colorblind_vision

DEFAULT_TYPE = "sequential"
DEFAULT_CMAP = "thermal"
BUILTIN_IMAGES = {
    "scan",
    "topography",
    "fn_roots",
    "phase",
    "grmhd",
    "vortex",
    "tng",
}

app = typer.Typer(help="Scientific colormap tools for humans and agents.")
cmap_app = typer.Typer(help="Explicit colormap command aliases.")
docs_app = typer.Typer(help="Explicit docs command aliases.")
console = Console()

app.add_typer(cmap_app, name="cmap")
app.add_typer(docs_app, name="docs")


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if payload.get("errors"):
        for err in payload["errors"]:
            console.print(f"[red]error:[/red] {err}")
        return

    data = payload.get("data", {})
    title = payload.get("command", "scicomap")
    console.print(f"[bold]{title}[/bold]")

    if isinstance(data, dict) and data:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Field")
        table.add_column("Value")
        for key, value in data.items():
            table.add_row(str(key), str(value))
        console.print(table)

    for warning in payload.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _fail(command: str, message: str, as_json: bool, code: int = 2) -> None:
    payload = {
        "ok": False,
        "command": command,
        "inputs": {},
        "data": {},
        "warnings": [],
        "errors": [message],
    }
    _emit(payload, as_json=as_json)
    raise typer.Exit(code=code)


def _resolve_cmap(cmap: str, ctype: str | None) -> tuple[str, Any]:
    cmap_dict = SciCoMap.get_color_map_dic()
    if ctype is not None:
        if ctype not in cmap_dict:
            raise ValueError(f"Unknown ctype '{ctype}'.")
        if cmap not in cmap_dict[ctype]:
            raise ValueError(f"Unknown cmap '{cmap}' for type '{ctype}'.")
        return ctype, cmap_dict[ctype][cmap]

    matches: list[tuple[str, Any]] = []
    for family, cmap_items in cmap_dict.items():
        if cmap in cmap_items:
            matches.append((family, cmap_items[cmap]))

    if not matches:
        raise ValueError(f"Unknown cmap '{cmap}'.")
    if len(matches) > 1:
        families = ", ".join(family for family, _ in matches)
        raise ValueError(
            "Ambiguous cmap "
            f"'{cmap}' found in multiple types: {families}. Use --type."
        )
    return matches[0]


def _save_figure(fig: Any, out: Path | None) -> str:
    if out is None:
        plt.show()
        return "displayed"
    out_path = out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


@app.command(name="list")
def list_command(
    family: str | None = typer.Argument(
        None, help="Optional colormap family."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List colormap families or names."""
    cmap_dict = SciCoMap.get_color_map_dic()

    if family is None:
        counts = {key: len(value) for key, value in cmap_dict.items()}
        payload = {
            "ok": True,
            "command": "scicomap list",
            "inputs": {"family": None},
            "data": {
                "families": ", ".join(cmap_dict.keys()),
                "counts": counts,
            },
            "warnings": [],
            "errors": [],
        }
        _emit(payload, as_json=as_json)
        return

    if family not in cmap_dict:
        _fail("scicomap list", f"Unknown family '{family}'.", as_json)

    items = sorted(cmap_dict[family].keys())
    payload = {
        "ok": True,
        "command": "scicomap list",
        "inputs": {"family": family},
        "data": {"family": family, "count": len(items), "items": items},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def check(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Diagnose one colormap and print key metrics."""
    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap check", str(exc), as_json)

    ctab = get_ctab(cmap_obj)
    jpapbp = transform(ctab)
    j_values = jpapbp[:, 0]
    j_diff = np.diff(j_values)
    is_monotonic = bool(np.all(j_diff >= 0) or np.all(j_diff <= 0))
    cmap_class = classify(jpapbp)
    n_extrema = int(len(extrema(j_values)))
    recommendation = "good to use"
    if not is_monotonic or cmap_class in {"asym_div", "unknown"}:
        recommendation = "consider fix"

    payload = {
        "ok": True,
        "command": "scicomap check",
        "inputs": {"cmap": cmap, "type": resolved_type},
        "data": {
            "classification": cmap_class,
            "monotonic_lightness": is_monotonic,
            "extrema_count": n_extrema,
            "recommendation": recommendation,
        },
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def preview(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Render a diagnostic figure for one colormap."""
    try:
        resolved_type, _ = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap preview", str(exc), as_json)

    chart = SciCoMap(ctype=resolved_type, cmap=cmap)
    fig = chart.assess_cmap()
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap preview",
        "inputs": {"cmap": cmap, "type": resolved_type},
        "data": {"artifact": artifact},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def compare(
    cmaps: list[str] = typer.Argument(..., help="Colormap names to compare."),
    ctype: str = typer.Option(DEFAULT_TYPE, "--type", help="Colormap family."),
    image: str = typer.Option(
        "scan", "--image", help="Builtin key or image path."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    ncols: int = typer.Option(3, "--ncols", help="Number of subplot columns."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Compare multiple colormaps on one image."""
    if len(cmaps) < 2:
        _fail(
            "scicomap compare", "Provide at least two colormap names.", as_json
        )

    try:
        for name in cmaps:
            _resolve_cmap(name, ctype)
    except ValueError as exc:
        _fail("scicomap compare", str(exc), as_json)

    if image not in BUILTIN_IMAGES:
        img_path = Path(image)
        if not img_path.exists():
            _fail(
                "scicomap compare",
                f"Image path does not exist: {img_path}",
                as_json,
            )

    fig = compare_cmap(image=image, ctype=ctype, cm_list=cmaps, ncols=ncols)
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap compare",
        "inputs": {"cmaps": cmaps, "type": ctype, "image": image},
        "data": {"artifact": artifact},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def fix(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    lift: float | None = typer.Option(
        None, "--lift", min=0.0, max=100.0, help="Lift value."
    ),
    bitonic: bool = typer.Option(
        True, "--bitonic/--no-bitonic", help="Bitonic symmetrization."
    ),
    diffuse: bool = typer.Option(
        True, "--diffuse/--no-diffuse", help="Diffuse symmetrization."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Apply uniformize+symmetrize and preview result."""
    try:
        resolved_type, _ = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap fix", str(exc), as_json)

    chart = SciCoMap(ctype=resolved_type, cmap=cmap)
    chart.unif_sym_cmap(lift=lift, bitonic=bitonic, diffuse=diffuse)
    fig = chart.assess_cmap()
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap fix",
        "inputs": {
            "cmap": cmap,
            "type": resolved_type,
            "lift": lift,
            "bitonic": bitonic,
            "diffuse": diffuse,
        },
        "data": {"artifact": artifact},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command(name="cvd")
def cvd_command(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    n_colors: int = typer.Option(
        256, "--n-colors", help="Number of color bins."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Render color-vision-deficiency simulation for one colormap."""
    try:
        resolved_type, _ = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap cvd", str(exc), as_json)

    fig = plot_colorblind_vision(
        ctype=resolved_type, cmap_list=[cmap], n_colors=n_colors
    )
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap cvd",
        "inputs": {"cmap": cmap, "type": resolved_type, "n_colors": n_colors},
        "data": {"artifact": artifact},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def apply(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    image: Path = typer.Option(
        ..., "--image", exists=True, readable=True, help="User image path."
    ),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image conversion mode: luminance, first-channel, or gray-only.",
    ),
    out: Path = typer.Option(..., "--out", help="Output image path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Apply a colormap to a user-provided image."""
    valid_modes = {"luminance", "first-channel", "gray-only"}
    if mode not in valid_modes:
        _fail("scicomap apply", f"Invalid mode '{mode}'.", as_json)

    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap apply", str(exc), as_json)

    arr = plt.imread(image)
    if arr.ndim == 2:
        scalar = arr.astype(float)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(float)
        if mode == "gray-only":
            _fail(
                "scicomap apply",
                "gray-only mode requires a grayscale image.",
                as_json,
            )
        if mode == "first-channel":
            scalar = rgb[..., 0]
        else:
            scalar = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
    else:
        _fail("scicomap apply", "Unsupported image format.", as_json)

    mapped = cmap_obj(_normalize(scalar))
    out_path = out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, mapped)

    payload = {
        "ok": True,
        "command": "scicomap apply",
        "inputs": {
            "cmap": cmap,
            "type": resolved_type,
            "image": str(image.resolve()),
            "mode": mode,
        },
        "data": {"artifact": str(out_path)},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command(name="docs-llm")
def docs_llm(
    html_dir: Path = typer.Option(
        Path("docs/build/html"), "--html-dir", help="Built docs directory."
    ),
    base_url: str = typer.Option(
        "https://thomasbury.github.io/scicomap",
        "--base-url",
        help="Canonical docs base URL.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Generate markdown mirrors and llms.txt from Sphinx HTML."""
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "build_llm_assets.py"
    )
    if not script_path.exists():
        _fail("scicomap docs-llm", f"Missing script: {script_path}", as_json)

    spec = importlib.util.spec_from_file_location(
        "build_llm_assets", script_path
    )
    if spec is None or spec.loader is None:
        _fail(
            "scicomap docs-llm", "Unable to load build_llm_assets.py", as_json
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    html_root = html_dir.resolve()
    markdown_dir = html_root / "llm"
    docs = module.build_markdown_mirror(
        html_dir=html_root, markdown_dir=markdown_dir
    )
    if not docs:
        _fail(
            "scicomap docs-llm",
            "No HTML pages were converted.",
            as_json,
            code=1,
        )
    module.write_llms_txt(html_dir=html_root, base_url=base_url, docs=docs)

    payload = {
        "ok": True,
        "command": "scicomap docs-llm",
        "inputs": {"html_dir": str(html_root), "base_url": base_url},
        "data": {
            "generated_pages": len(docs),
            "markdown_dir": str(markdown_dir),
            "llms_txt": str((html_root / "llms.txt").resolve()),
        },
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def version(
    as_json: bool = typer.Option(False, "--json", help="Output JSON.")
) -> None:
    """Print installed scicomap version."""
    from scicomap import __version__

    payload = {
        "ok": True,
        "command": "scicomap version",
        "inputs": {},
        "data": {"version": __version__},
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@cmap_app.command(name="families")
def cmap_families_alias(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap list`."""
    list_command(family=None, as_json=as_json)


@cmap_app.command(name="list")
def cmap_list_alias(
    ctype: str = typer.Option(DEFAULT_TYPE, "--type", help="Colormap family."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap list <family>`."""
    list_command(family=ctype, as_json=as_json)


@cmap_app.command(name="assess")
def cmap_assess_alias(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap preview`."""
    preview(cmap=cmap, ctype=ctype, out=out, as_json=as_json)


@cmap_app.command(name="compare")
def cmap_compare_alias(
    cmaps: list[str] = typer.Option(..., "--cmaps", help="Colormap names."),
    ctype: str = typer.Option(DEFAULT_TYPE, "--type", help="Colormap family."),
    image: str = typer.Option(
        "scan", "--image", help="Builtin key or image path."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    ncols: int = typer.Option(3, "--ncols", help="Number of subplot columns."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap compare`."""
    compare(
        cmaps=cmaps,
        ctype=ctype,
        image=image,
        out=out,
        ncols=ncols,
        as_json=as_json,
    )


@cmap_app.command(name="fix")
def cmap_fix_alias(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    lift: float | None = typer.Option(
        None, "--lift", min=0.0, max=100.0, help="Lift value."
    ),
    bitonic: bool = typer.Option(
        True, "--bitonic/--no-bitonic", help="Bitonic symmetrization."
    ),
    diffuse: bool = typer.Option(
        True, "--diffuse/--no-diffuse", help="Diffuse symmetrization."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap fix`."""
    fix(
        cmap=cmap,
        ctype=ctype,
        lift=lift,
        bitonic=bitonic,
        diffuse=diffuse,
        out=out,
        as_json=as_json,
    )


@cmap_app.command(name="colorblind")
def cmap_colorblind_alias(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output file path."
    ),
    n_colors: int = typer.Option(
        256, "--n-colors", help="Number of color bins."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap cvd`."""
    cvd_command(
        cmap=cmap,
        ctype=ctype,
        out=out,
        n_colors=n_colors,
        as_json=as_json,
    )


@cmap_app.command(name="apply")
def cmap_apply_alias(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    image: Path = typer.Option(
        ..., "--image", exists=True, readable=True, help="User image path."
    ),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image conversion mode: luminance, first-channel, or gray-only.",
    ),
    out: Path = typer.Option(..., "--out", help="Output image path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap apply`."""
    apply(
        cmap=cmap,
        image=image,
        ctype=ctype,
        mode=mode,
        out=out,
        as_json=as_json,
    )


@docs_app.command(name="llm-assets")
def docs_llm_assets_alias(
    html_dir: Path = typer.Option(
        Path("docs/build/html"), "--html-dir", help="Built docs directory."
    ),
    base_url: str = typer.Option(
        "https://thomasbury.github.io/scicomap",
        "--base-url",
        help="Canonical docs base URL.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Alias for `scicomap docs-llm`."""
    docs_llm(html_dir=html_dir, base_url=base_url, as_json=as_json)


def main() -> None:
    """Entrypoint for python -m usage."""
    app()


if __name__ == "__main__":
    main()
