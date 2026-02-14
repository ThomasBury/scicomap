"""Command-line interface for scicomap."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from importlib.util import find_spec
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
VALID_GOALS = {"diagnose", "improve", "apply"}
VALID_MODES = {"luminance", "first-channel", "gray-only"}
VALID_FORMATS = {"text", "json"}

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


def _diagnose_cmap(cmap_obj: Any) -> dict[str, Any]:
    ctab = get_ctab(cmap_obj)
    jpapbp = transform(ctab)
    j_values = jpapbp[:, 0]
    j_diff = np.diff(j_values)
    is_monotonic = bool(np.all(j_diff >= 0) or np.all(j_diff <= 0))
    cmap_class = classify(jpapbp)
    n_extrema = int(len(extrema(j_values)))

    reasons: list[str] = []
    status = "good"
    if not is_monotonic:
        status = "fix-recommended"
        reasons.append("lightness is not monotonic")
    elif cmap_class in {"asym_div", "unknown"}:
        status = "caution"
        reasons.append(f"classification is '{cmap_class}'")

    if n_extrema > 2:
        if status == "good":
            status = "caution"
        reasons.append("lightness has many extrema")

    recommendation = "good to use"
    if status in {"caution", "fix-recommended"}:
        recommendation = "consider fix"

    return {
        "classification": cmap_class,
        "monotonic_lightness": is_monotonic,
        "extrema_count": n_extrema,
        "status": status,
        "reasons": reasons,
        "recommendation": recommendation,
    }


def _report_output_dir(out: Path | None) -> Path:
    if out is not None:
        report_dir = out.resolve()
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        report_dir = (Path.cwd() / f"scicomap-report-{stamp}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _write_summary_txt(report_dir: Path, payload: dict[str, Any]) -> Path:
    data = payload.get("data", {})
    diagnostics = data.get("diagnostics", {})
    artifacts = data.get("artifacts", [])

    lines = [
        "scicomap report",
        f"status: {data.get('status', 'unknown')}",
        f"goal: {data.get('goal', 'unknown')}",
        f"cmap: {data.get('cmap', 'unknown')}",
        f"type: {data.get('type', 'unknown')}",
        "",
        "diagnostics:",
        f"- classification: {diagnostics.get('classification', 'unknown')}",
        "- monotonic_lightness: "
        f"{diagnostics.get('monotonic_lightness', 'unknown')}",
        f"- extrema_count: {diagnostics.get('extrema_count', 'unknown')}",
    ]

    reasons = diagnostics.get("reasons", [])
    if reasons:
        lines.append("- reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")

    lines.extend(["", "artifacts:"])
    for artifact in artifacts:
        lines.append(f"- {artifact['kind']}: {artifact['path']}")

    if payload.get("warnings"):
        lines.append("")
        lines.append("warnings:")
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")

    if data.get("next_step"):
        lines.append("")
        lines.append(f"next_step: {data['next_step']}")

    summary_path = report_dir / "summary.txt"
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


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
    """Diagnose one colormap and print key metrics.

    Examples
    --------
    scicomap check hawaii
    scicomap check hawaii --type sequential --json
    """
    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap check", str(exc), as_json)

    diagnostics = _diagnose_cmap(cmap_obj)

    payload = {
        "ok": True,
        "command": "scicomap check",
        "inputs": {"cmap": cmap, "type": resolved_type},
        "data": diagnostics,
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
    if mode not in VALID_MODES:
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
def doctor(
    out_dir: Path = typer.Option(
        Path("."), "--out-dir", help="Directory for generated artifacts."
    ),
    image: Path | None = typer.Option(
        None, "--image", help="Optional image path to validate."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Validate local environment and common CLI prerequisites.

    Examples
    --------
    scicomap doctor
    scicomap doctor --out-dir outputs --json
    """
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for module_name in ["typer", "rich", "colorspacious", "matplotlib"]:
        installed = find_spec(module_name) is not None
        checks.append(
            {
                "name": f"dependency:{module_name}",
                "ok": installed,
            }
        )
        if not installed:
            errors.append(f"Missing dependency: {module_name}")

    backend = plt.get_backend()
    checks.append({"name": "matplotlib_backend", "ok": bool(backend)})
    if "agg" in backend.lower():
        warnings.append(
            "Matplotlib backend is non-interactive (Agg). "
            "Use --out for image commands."
        )

    out_ok = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".scicomap_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        out_ok = False
        errors.append(f"Output directory is not writable: {out_dir}")
    checks.append({"name": "output_directory", "ok": out_ok})

    if image is not None:
        image_ok = image.exists() and image.is_file()
        checks.append({"name": "image_path", "ok": image_ok})
        if not image_ok:
            errors.append(f"Image path is invalid: {image}")

    payload = {
        "ok": not errors,
        "command": "scicomap doctor",
        "inputs": {
            "out_dir": str(out_dir.resolve()),
            "image": None if image is None else str(image.resolve()),
        },
        "data": {
            "checks": checks,
            "backend": backend,
            "status": "healthy" if not errors else "action-required",
        },
        "warnings": warnings,
        "errors": errors,
    }
    _emit(payload, as_json=as_json)
    if errors:
        raise typer.Exit(code=1)


@app.command()
def wizard(
    goal: str | None = typer.Option(
        None,
        "--goal",
        help="Workflow goal: diagnose, improve, apply.",
    ),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    cmap: str | None = typer.Option(None, "--cmap", help="Colormap name."),
    image: Path | None = typer.Option(
        None, "--image", help="Image path for apply."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output path."
    ),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image mode for apply: luminance, first-channel, gray-only.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Prompt for missing values.",
    ),
) -> None:
    """Run a guided workflow for diagnose, improve, or apply.

    Examples
    --------
    scicomap wizard
    scicomap wizard --goal diagnose --cmap thermal --type sequential \
        --no-interactive --json
    """
    selected_goal = goal
    selected_type = ctype
    selected_cmap = cmap
    selected_image = image

    if interactive:
        if selected_goal is None:
            selected_goal = typer.prompt(
                "Goal (diagnose/improve/apply)", default="diagnose"
            )
        if selected_type is None:
            selected_type = typer.prompt("Colormap type", default=DEFAULT_TYPE)
        if selected_cmap is None:
            selected_cmap = typer.prompt("Colormap name", default=DEFAULT_CMAP)
        if selected_goal == "apply" and selected_image is None:
            selected_image = Path(typer.prompt("Image path"))
        if out is None and typer.confirm(
            "Save output to file?", default=False
        ):
            out = Path(typer.prompt("Output path"))

    if selected_goal is None:
        _fail(
            "scicomap wizard",
            "Missing --goal in non-interactive mode.",
            as_json,
        )
    if selected_goal not in VALID_GOALS:
        _fail("scicomap wizard", f"Invalid goal '{selected_goal}'.", as_json)
    if selected_cmap is None:
        _fail("scicomap wizard", "Missing --cmap value.", as_json)
    if mode not in VALID_MODES:
        _fail("scicomap wizard", f"Invalid mode '{mode}'.", as_json)

    try:
        resolved_type, cmap_obj = _resolve_cmap(selected_cmap, selected_type)
    except ValueError as exc:
        _fail("scicomap wizard", str(exc), as_json)

    result: dict[str, Any] = {
        "goal": selected_goal,
        "cmap": selected_cmap,
        "type": resolved_type,
    }
    warnings: list[str] = []

    if selected_goal == "diagnose":
        result["diagnostics"] = _diagnose_cmap(cmap_obj)
        result["next_step"] = "run 'scicomap fix <cmap>' if status is caution"
    elif selected_goal == "improve":
        chart = SciCoMap(ctype=resolved_type, cmap=selected_cmap)
        chart.unif_sym_cmap(lift=None, bitonic=True, diffuse=True)
        figure = chart.assess_cmap()
        result["artifact"] = _save_figure(figure, out)
        result["next_step"] = "use the fixed colormap in your plot pipeline"
    else:
        if selected_image is None:
            _fail("scicomap wizard", "Apply goal requires --image.", as_json)
        if not selected_image.exists():
            _fail(
                "scicomap wizard",
                f"Image path does not exist: {selected_image}",
                as_json,
            )
        if out is None:
            warnings.append(
                "No --out provided; writing 'scicomap-applied.png' in cwd."
            )
            out = Path("scicomap-applied.png")
        arr = plt.imread(selected_image)
        if arr.ndim == 2:
            scalar = arr.astype(float)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(float)
            if mode == "gray-only":
                _fail(
                    "scicomap wizard",
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
            _fail("scicomap wizard", "Unsupported image format.", as_json)

        mapped = cmap_obj(_normalize(scalar))
        out_path = out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, mapped)
        result["artifact"] = str(out_path)
        result["next_step"] = (
            "preview the generated image and compare with original"
        )

    payload = {
        "ok": True,
        "command": "scicomap wizard",
        "inputs": {
            "goal": selected_goal,
            "type": resolved_type,
            "cmap": selected_cmap,
            "image": (
                None
                if selected_image is None
                else str(selected_image.resolve())
            ),
            "out": None if out is None else str(out.resolve()),
            "mode": mode,
            "interactive": interactive,
        },
        "data": result,
        "warnings": warnings,
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def report(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    image: str | None = typer.Option(
        None, "--image", help="Input image path or builtin key."
    ),
    goal: str | None = typer.Option(
        None,
        "--goal",
        help="Workflow goal: diagnose, improve, apply.",
    ),
    fix: bool | None = typer.Option(
        None,
        "--fix/--no-fix",
        help="Apply colormap fix stage.",
    ),
    cvd: bool | None = typer.Option(
        None,
        "--cvd/--no-cvd",
        help="Generate colorblind simulation artifact.",
    ),
    apply_output: bool | None = typer.Option(
        None,
        "--apply/--no-apply",
        help="Generate colormap-applied image.",
    ),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image mode for apply: luminance, first-channel, gray-only.",
    ),
    lift: float | None = typer.Option(
        None,
        "--lift",
        min=0.0,
        max=100.0,
        help="Lift value for fix stage.",
    ),
    bitonic: bool = typer.Option(
        True,
        "--bitonic/--no-bitonic",
        help="Bitonic symmetrization for fix stage.",
    ),
    diffuse: bool = typer.Option(
        True,
        "--diffuse/--no-diffuse",
        help="Diffuse symmetrization for fix stage.",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output report directory."
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
    ),
) -> None:
    """Generate a full report bundle for one colormap workflow.

    Examples
    --------
    scicomap report --cmap hawaii --type sequential --out reports/hawaii
    scicomap report --cmap thermal --image input.png --apply --format json
    """
    if output_format not in VALID_FORMATS:
        _fail("scicomap report", f"Invalid format '{output_format}'.", False)
    as_json = output_format == "json"

    resolved_goal = goal
    if resolved_goal is None:
        resolved_goal = "apply" if image is not None else "diagnose"
    if resolved_goal not in VALID_GOALS:
        _fail("scicomap report", f"Invalid goal '{resolved_goal}'.", as_json)
    if mode not in VALID_MODES:
        _fail("scicomap report", f"Invalid mode '{mode}'.", as_json)

    run_fix = fix if fix is not None else (resolved_goal == "improve")
    run_cvd = (
        cvd if cvd is not None else (resolved_goal in {"diagnose", "improve"})
    )
    run_apply = (
        apply_output
        if apply_output is not None
        else (resolved_goal == "apply")
    )

    if run_apply and image is None:
        _fail(
            "scicomap report",
            "Apply stage requires --image.",
            as_json,
        )

    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap report", str(exc), as_json)

    diagnostics = _diagnose_cmap(cmap_obj)
    report_dir = _report_output_dir(out)
    artifacts: list[dict[str, str]] = []
    warnings: list[str] = []

    if resolved_goal in {"diagnose", "improve"}:
        chart = SciCoMap(ctype=resolved_type, cmap=cmap)
        assess_path = report_dir / "assess.png"
        artifact = _save_figure(chart.assess_cmap(), assess_path)
        artifacts.append(
            {"kind": "assessment", "path": artifact, "format": "png"}
        )

    if run_fix:
        fixed_chart = SciCoMap(ctype=resolved_type, cmap=cmap)
        fixed_chart.unif_sym_cmap(lift=lift, bitonic=bitonic, diffuse=diffuse)
        fixed_path = report_dir / "fixed-assess.png"
        artifact = _save_figure(fixed_chart.assess_cmap(), fixed_path)
        artifacts.append(
            {"kind": "fixed_assessment", "path": artifact, "format": "png"}
        )

    if run_cvd:
        cvd_path = report_dir / "cvd.png"
        cvd_fig = plot_colorblind_vision(
            ctype=resolved_type,
            cmap_list=[cmap],
            n_colors=256,
        )
        artifact = _save_figure(cvd_fig, cvd_path)
        artifacts.append(
            {"kind": "colorblind", "path": artifact, "format": "png"}
        )

    if run_apply:
        applied_path = report_dir / "applied.png"
        if image in BUILTIN_IMAGES:
            apply_fig = compare_cmap(
                image=image,
                ctype=resolved_type,
                cm_list=[cmap],
                ncols=1,
                uniformize=False,
                title=False,
                symmetrize=False,
                facecolor="white",
            )
            artifact = _save_figure(apply_fig, applied_path)
            warnings.append(
                "Builtin image apply uses rendered figure output "
                "rather than raw remap."
            )
        else:
            if image is None:
                _fail("scicomap report", "Missing --image value.", as_json)
            image_path = Path(image)
            if not image_path.exists():
                _fail(
                    "scicomap report",
                    f"Image path does not exist: {image_path}",
                    as_json,
                )
            arr = plt.imread(image_path)
            if arr.ndim == 2:
                scalar = arr.astype(float)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                rgb = arr[..., :3].astype(float)
                if mode == "gray-only":
                    _fail(
                        "scicomap report",
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
                _fail("scicomap report", "Unsupported image format.", as_json)

            mapped = cmap_obj(_normalize(scalar))
            plt.imsave(applied_path, mapped)
            artifact = str(applied_path.resolve())
        artifacts.append(
            {"kind": "applied", "path": artifact, "format": "png"}
        )

    next_step = "run 'scicomap report --fix' to improve the colormap"
    if diagnostics["status"] == "good":
        next_step = "use this colormap in your plotting pipeline"

    payload = {
        "ok": True,
        "command": "scicomap report",
        "inputs": {
            "cmap": cmap,
            "type": resolved_type,
            "image": image,
            "goal": resolved_goal,
            "fix": run_fix,
            "cvd": run_cvd,
            "apply": run_apply,
            "mode": mode,
            "lift": lift,
            "bitonic": bitonic,
            "diffuse": diffuse,
            "out": str(report_dir),
            "format": output_format,
        },
        "data": {
            "status": diagnostics["status"],
            "goal": resolved_goal,
            "cmap": cmap,
            "type": resolved_type,
            "diagnostics": diagnostics,
            "actions": {
                "fix_applied": run_fix,
                "cvd_generated": run_cvd,
                "image_applied": run_apply,
            },
            "artifacts": artifacts,
            "next_step": next_step,
            "report_dir": str(report_dir),
        },
        "warnings": warnings,
        "errors": [],
    }

    report_json = report_dir / "report.json"
    report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = _write_summary_txt(report_dir, payload)
    payload["data"]["report_json"] = str(report_json.resolve())
    payload["data"]["summary_txt"] = str(summary_path.resolve())

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
