"""Command-line interface for scicomap."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
from typer._click.exceptions import ClickException
from matplotlib.colors import Colormap
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from scicomap._diagnostics import diagnose_cmap
from scicomap.scicomap import (
    SciCoMap,
    compare_cmap,
    get_cmap_dict,
    plot_colorblind_vision,
)

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
VALID_MODES = {"luminance", "first-channel", "gray-only"}
_CVD_DESCRIPTION = (
    "Colorspacious sRGB1+CVD simulations: deuteranomaly at severity 50 and "
    "100, protanomaly at severity 50, and tritanomaly at severity 100. "
    "RGB output is clipped to [0, 1]. These simulations do not certify "
    "accessibility or guarantee distinguishable colors."
)


class _WorkflowGroup(TyperGroup):
    """Keep validation and expected runtime failures in the requested format."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        tokens = args[: args.index("--")] if "--" in args else args
        ctx.meta["json_output"] = "--json" in tokens
        ctx.meta["command"] = "scicomap " + " ".join(args[:1])
        try:
            return super().parse_args(ctx, args)
        except ClickException as exc:
            if not ctx.meta["json_output"]:
                raise
            _fail(
                ctx.meta["command"],
                exc.format_message(),
                ctx.meta["json_output"],
                exc.exit_code,
            )

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except typer.Exit:
            raise
        except ClickException as exc:
            if not ctx.meta["json_output"]:
                raise
            _fail(
                ctx.meta["command"],
                exc.format_message(),
                ctx.meta["json_output"],
                exc.exit_code,
            )
        except (ValueError, SyntaxError) as exc:
            _fail(ctx.meta["command"], str(exc), ctx.meta["json_output"], 2)
        except (OSError, RuntimeError) as exc:
            _fail(ctx.meta["command"], str(exc), ctx.meta["json_output"], 1)


app = typer.Typer(
    cls=_WorkflowGroup,
    help="Scientific colormap tools for humans and agents.",
)
console = Console()


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
    if Path(cmap).suffix == ".json":
        path = Path(cmap)
        if not path.is_file():
            raise ValueError(f"Exported colormap file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or not {"family", "name", "rgba"} <= payload.keys()
        ):
            raise ValueError(
                "Exported colormap JSON requires family, name, and rgba fields."
            )
        if (
            not isinstance(payload["name"], str)
            or not isinstance(payload["family"], str)
            or not isinstance(payload["rgba"], list)
        ):
            raise ValueError(
                "Exported colormap name and family must be strings and rgba must be a color table."
            )
        try:
            chart = SciCoMap(
                ctype=payload["family"] if ctype is None else ctype,
                cmap=payload["rgba"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid exported colormap: {exc}") from exc
        chart.cmap.name = payload["name"]
        return chart.ctype, chart.cmap
    cmap_dict = get_cmap_dict()
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
    try:
        if out is None:
            plt.show()
            return "displayed"
        _validate_output(out)
        out_path = out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return str(out_path)


def _validate_output(out: Path | None, *, directory: bool = False) -> None:
    if out is None:
        return
    path = out.resolve()
    if path.exists() and (path.is_dir() != directory):
        kind = "directory" if directory else "file"
        raise ValueError(f"Output must be a {kind}: {path}")
    for parent in path.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Output parent is not a directory: {parent}")


def _validate_render_output(
    out: Path | None, as_json: bool, *, directory: bool = False
) -> None:
    if as_json and out is None:
        raise ValueError("Rendering with --json requires --out.")
    _validate_output(out, directory=directory)
    if as_json:
        plt.switch_backend("Agg")


def _prepare_maps(
    ctype: str, cmap_obj: Colormap, config: dict[str, Any]
) -> tuple[SciCoMap, SciCoMap]:
    original = SciCoMap(ctype=ctype, cmap=cmap_obj)
    selected = original
    if config["fix"]:
        selected = SciCoMap(ctype=ctype, cmap=cmap_obj)
        selected.unif_sym_cmap(
            lightness_rounding=config["lightness_rounding"],
            bitonic=config["bitonic"],
            diffuse=config["diffuse"],
        )
    return original, selected


def _normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _remap_image(image: Path, cmap_obj: Colormap, mode: str) -> np.ndarray:
    """Read and remap a grayscale, RGB, or RGBA image, preserving its alpha."""
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'.")
    try:
        arr = plt.imread(image)
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(
            f"Cannot read image '{image}': {exc}. "
            "Provide a readable PNG, JPEG, or another supported image."
        ) from exc

    if arr.size == 0 or not (
        arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in {3, 4})
    ):
        raise ValueError(
            "Unsupported image shape; expected grayscale (H, W), "
            "RGB (H, W, 3), or RGBA (H, W, 4)."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "Image pixels must be finite; remove NaN or infinity."
        )

    pixels = arr.astype(float)
    if arr.ndim == 2:
        scalar = pixels
    else:
        if mode == "gray-only":
            raise ValueError("gray-only mode requires a grayscale image.")
        if np.issubdtype(arr.dtype, np.integer):
            pixels /= np.iinfo(arr.dtype).max
        if np.any((pixels < 0) | (pixels > 1)):
            raise ValueError(
                "RGB and alpha values must be in the range [0, 1]."
            )
        if mode == "first-channel":
            scalar = pixels[..., 0]
        else:
            scalar = (
                0.2126 * pixels[..., 0]
                + 0.7152 * pixels[..., 1]
                + 0.0722 * pixels[..., 2]
            )

    mapped = cmap_obj(_normalize(scalar))
    if arr.ndim == 3 and arr.shape[2] == 4:
        mapped[..., 3] = pixels[..., 3]
    return mapped


def _report_output_dir(out: Path | None) -> Path:
    if out is not None:
        report_dir = out.resolve()
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        report_dir = (Path.cwd() / f"scicomap-report-{stamp}").resolve()
    _validate_output(report_dir, directory=True)
    return report_dir


def _write_summary_txt(report_dir: Path, payload: dict[str, Any]) -> Path:
    data = payload.get("data", {})
    artifacts = data.get("artifacts", [])

    lines = [
        "scicomap report",
        f"status: {data.get('status', 'unknown')}",
        f"fix applied: {data['actions']['fix_applied']}",
        f"cmap: {data.get('cmap', 'unknown')}",
        f"type: {data.get('type', 'unknown')}",
        f"map used: {data.get('map_used', 'unknown')}",
        "Statuses are heuristics; CVD simulations do not certify accessibility.",
    ]

    for label in ("original", "transformed"):
        diagnostics = data[f"{label}_diagnostics"]
        lines.extend(["", f"{label} diagnostics:"])
        if diagnostics is None:
            lines.append("- not requested")
            continue
        for key in (
            "status",
            "classification",
            "monotonic_lightness",
            "extrema_count",
        ):
            lines.append(f"- {key}: {diagnostics[key]}")
        for reason in diagnostics["reasons"]:
            lines.append(f"- reason: {reason}")

    if data["cvd_simulation"] is not None:
        lines.extend(
            ["", f"CVD map: {data['cvd_simulation']['map']}", _CVD_DESCRIPTION]
        )

    lines.extend(["", "artifacts:"])
    for artifact in artifacts:
        lines.append(
            f"- {artifact['kind']} ({artifact['map']}): {artifact['path']}"
        )

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
    cmap_dict = get_cmap_dict()

    if family is None:
        counts = {key: len(value) for key, value in cmap_dict.items()}
        payload = {
            "ok": True,
            "command": "scicomap list",
            "inputs": {"family": None},
            "data": {
                "families": list(cmap_dict),
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

    diagnostics = diagnose_cmap(cmap_obj, resolved_type)

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
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap preview", str(exc), as_json)

    _validate_render_output(out, as_json)
    chart = SciCoMap(ctype=resolved_type, cmap=cmap_obj)
    fig = chart.assess_cmap()
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap preview",
        "inputs": {"cmap": cmap, "type": resolved_type},
        "data": {
            "artifacts": [
                {"kind": "assessment", "path": artifact, "map": "original"}
            ]
        },
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
    if ncols < 1:
        _fail("scicomap compare", "--ncols must be at least 1.", as_json)
    _validate_render_output(out, as_json)
    if len(cmaps) < 2:
        _fail(
            "scicomap compare", "Provide at least two colormap names.", as_json
        )

    try:
        resolved_maps = [_resolve_cmap(name, ctype)[1] for name in cmaps]
    except ValueError as exc:
        _fail("scicomap compare", str(exc), as_json)

    if image not in BUILTIN_IMAGES:
        img_path = Path(image)
        if not img_path.is_file():
            _fail(
                "scicomap compare",
                f"Image path does not exist: {img_path}",
                as_json,
            )

    fig = compare_cmap(
        image=image,
        ctype=ctype,
        cm_list=resolved_maps,
        ncols=ncols,
        uniformize=False,
        symmetrize=False,
    )
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap compare",
        "inputs": {"cmaps": cmaps, "type": ctype, "image": image},
        "data": {
            "artifacts": [
                {"kind": "comparison", "path": artifact, "map": "original"}
            ]
        },
        "warnings": [],
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def fix(
    cmap: str = typer.Argument(DEFAULT_CMAP, help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    lightness_rounding: float | None = typer.Option(
        None,
        "--lightness-rounding",
        min=0.0,
        help="Lower lightness rounding step in CAM02-UCS J' units.",
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
    export: Path | None = typer.Option(
        None,
        "--export",
        help="Save corrected RGBA colors and parameters as JSON.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Apply uniformize+symmetrize and preview result."""
    if lightness_rounding is not None and not np.isfinite(lightness_rounding):
        _fail("scicomap fix", "--lightness-rounding must be finite.", as_json)
    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap fix", str(exc), as_json)

    renders = out is not None or export is None
    if renders:
        _validate_render_output(out, as_json)
    _validate_output(export)
    if (
        export is not None
        and out is not None
        and export.resolve() == out.resolve()
    ):
        raise ValueError("--export and --out must be different files.")
    chart = SciCoMap(ctype=resolved_type, cmap=cmap_obj)
    chart.unif_sym_cmap(
        lightness_rounding=lightness_rounding, bitonic=bitonic, diffuse=diffuse
    )
    artifacts = []
    if renders:
        artifact = _save_figure(chart.assess_cmap(), out)
        artifacts.append(
            {"kind": "assessment", "path": artifact, "map": "transformed"}
        )
    if export is not None:
        artifacts.append(
            {
                "kind": "color_table",
                "path": str(chart.export_cmap(export)),
                "map": "transformed",
            }
        )
    payload = {
        "ok": True,
        "command": "scicomap fix",
        "inputs": {
            "cmap": cmap,
            "type": resolved_type,
            "lightness_rounding": lightness_rounding,
            "bitonic": bitonic,
            "diffuse": diffuse,
            "export": str(export.resolve()) if export is not None else None,
        },
        "data": {
            "artifacts": artifacts,
        },
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
    if n_colors < 1:
        _fail("scicomap cvd", "--n-colors must be at least 1.", as_json)
    _validate_render_output(out, as_json)
    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap cvd", str(exc), as_json)

    fig = plot_colorblind_vision(
        ctype=resolved_type,
        cmap_list=[cmap_obj],
        n_colors=n_colors,
        uniformize=False,
        symmetrize=False,
    )
    artifact = _save_figure(fig, out)
    payload = {
        "ok": True,
        "command": "scicomap cvd",
        "inputs": {"cmap": cmap, "type": resolved_type, "n_colors": n_colors},
        "data": {
            "cvd_simulation": {
                "map": "original",
                "n_colors": n_colors,
                "description": _CVD_DESCRIPTION,
            },
            "artifacts": [
                {"kind": "colorblind", "path": artifact, "map": "original"}
            ],
        },
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
    _validate_output(out)
    if mode not in VALID_MODES:
        _fail("scicomap apply", f"Invalid mode '{mode}'.", as_json)

    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap apply", str(exc), as_json)

    try:
        mapped = _remap_image(image, cmap_obj, mode)
    except ValueError as exc:
        _fail("scicomap apply", str(exc), as_json)
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
        "data": {
            "artifacts": [
                {"kind": "applied", "path": str(out_path), "map": "original"}
            ]
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
        with NamedTemporaryFile(
            dir=out_dir,
            prefix=".scicomap_write_test-",
            mode="w",
            encoding="utf-8",
        ) as probe:
            probe.write("ok")
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


def _run_workflow(
    *,
    command: str,
    cmap: str,
    ctype: str | None,
    image: str | None,
    out: Path | None,
    mode: str,
    fix: bool,
    cvd: bool,
    apply_output: bool,
    lightness_rounding: float | None,
    bitonic: bool,
    diffuse: bool,
    as_json: bool,
    bundle: bool,
    export: Path | None = None,
) -> dict[str, Any]:
    """Prepare a map once and run only the explicitly requested stages."""
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'.")
    if lightness_rounding is not None and not np.isfinite(lightness_rounding):
        raise ValueError("--lightness-rounding must be finite.")
    if apply_output and image is None:
        raise ValueError("Apply stage requires --image.")
    if apply_output and out is None and not bundle:
        raise ValueError("Apply stage requires --out.")
    if export is not None and not fix:
        raise ValueError("--export requires --fix in wizard.")
    renders = (
        bundle
        or (fix and export is None)
        or cvd
        or apply_output
        or out is not None
    )
    if renders:
        _validate_render_output(out, as_json, directory=bundle)
    resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    config = {
        "fix": fix,
        "cvd": cvd,
        "apply": apply_output,
        "lightness_rounding": lightness_rounding,
        "bitonic": bitonic,
        "diffuse": diffuse,
    }
    original, selected = _prepare_maps(resolved_type, cmap_obj, config)
    selected_map = selected.cmap
    mapped = None
    if apply_output and image not in BUILTIN_IMAGES:
        mapped = _remap_image(Path(image), selected_map, mode)
    report_dir = _report_output_dir(out) if bundle else None
    if bundle and fix:
        export = report_dir / "corrected-cmap.json"
    _validate_output(export)
    # Validate every destination before writing the first artifact.
    outputs: list[tuple[str, Path | None, str]] = []
    map_used = "transformed" if fix else "original"
    if bundle:
        outputs.append(("assessment", report_dir / "assess.png", "original"))
        if fix:
            outputs.append(
                ("fixed_assessment", report_dir / "fixed-assess.png", map_used)
            )
    elif (fix and renders) or (out is not None and not apply_output):
        assess_out = out
        if apply_output and out is not None:
            assess_out = out.with_name(out.stem + "-assess.png")
        outputs.append(("assessment", assess_out, map_used))
    if cvd:
        cvd_out = (
            report_dir / "cvd.png"
            if bundle
            else (
                out.with_name(out.stem + "-cvd.png")
                if out is not None
                else None
            )
        )
        outputs.append(("colorblind", cvd_out, map_used))
    if apply_output:
        outputs.append(
            (
                "applied",
                report_dir / "applied.png" if bundle else out,
                map_used,
            )
        )
    for _, path, _ in outputs:
        _validate_output(path)
        if (
            export is not None
            and path is not None
            and export.resolve() == path.resolve()
        ):
            raise ValueError("--export must differ from image artifact paths.")
    if bundle:
        for filename in ("report.json", "summary.txt"):
            _validate_output(report_dir / filename)
        report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    warnings = []
    if export is not None:
        artifacts.append(
            {
                "kind": "color_table",
                "path": str(selected.export_cmap(export)),
                "map": map_used,
            }
        )
    for kind, path, used in outputs:
        if kind in {"assessment", "fixed_assessment"}:
            chart = original if used == "original" else selected
            artifact = _save_figure(chart.assess_cmap(), path)
        elif kind == "colorblind":
            figure = plot_colorblind_vision(
                ctype=resolved_type,
                cmap_list=[selected_map],
                n_colors=256,
                uniformize=False,
                symmetrize=False,
            )
            artifact = _save_figure(figure, path)
        elif image in BUILTIN_IMAGES:
            figure = compare_cmap(
                image=image,
                ctype=resolved_type,
                cm_list=[selected_map],
                ncols=1,
                uniformize=False,
                symmetrize=False,
                title=False,
                facecolor="white",
            )
            artifact = _save_figure(figure, path)
            warnings.append(
                "Builtin image apply uses rendered figure output rather than raw remap."
            )
        else:
            path = path.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(path, mapped)
            artifact = str(path)
        artifacts.append({"kind": kind, "path": artifact, "map": used})
    diagnostics = diagnose_cmap(selected_map, resolved_type)
    payload = {
        "ok": True,
        "command": command,
        "inputs": {
            "cmap": cmap,
            "type": resolved_type,
            "image": (
                image
                if image is None or image in BUILTIN_IMAGES
                else str(Path(image).resolve())
            ),
            "out": str(report_dir)
            if bundle
            else (str(out.resolve()) if out else None),
            "mode": mode,
            "export": str(export.resolve()) if export is not None else None,
            **config,
        },
        "data": {
            "cmap": cmap,
            "type": resolved_type,
            "status": diagnostics["status"],
            "diagnostics": diagnostics,
            "original_diagnostics": diagnose_cmap(cmap_obj, resolved_type),
            "transformed_diagnostics": diagnostics if fix else None,
            "cvd_simulation": {
                "map": map_used,
                "n_colors": 256,
                "description": _CVD_DESCRIPTION,
            }
            if cvd
            else None,
            "map_used": map_used,
            "actions": {
                "fix_applied": fix,
                "cvd_generated": cvd,
                "image_applied": apply_output,
            },
            "artifacts": artifacts,
            "next_step": "inspect the selected map with your data",
        },
        "warnings": warnings,
        "errors": [],
    }
    if bundle:
        report_json = report_dir / "report.json"
        summary_path = report_dir / "summary.txt"
        payload["data"].update(
            report_dir=str(report_dir),
            report_json=str(report_json),
            summary_txt=str(summary_path),
        )
        artifacts.extend(
            [
                {"kind": "report", "path": str(report_json), "map": map_used},
                {
                    "kind": "summary",
                    "path": str(summary_path),
                    "map": map_used,
                },
            ]
        )
        report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_summary_txt(report_dir, payload)
    return payload


@app.command()
def wizard(
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    cmap: str | None = typer.Option(None, "--cmap", help="Colormap name."),
    image: str | None = typer.Option(
        None, "--image", help="Image path or builtin key for apply."
    ),
    out: Path | None = typer.Option(None, "--out", help="Output image path."),
    export: Path | None = typer.Option(
        None,
        "--export",
        help="Save corrected RGBA colors and parameters as JSON; requires --fix.",
    ),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image conversion: luminance, first-channel, gray-only.",
    ),
    fix: bool | None = typer.Option(
        None, "--fix/--no-fix", help="Correct the colormap."
    ),
    cvd: bool | None = typer.Option(
        None, "--cvd/--no-cvd", help="Generate a CVD simulation."
    ),
    apply_output: bool | None = typer.Option(
        None, "--apply/--no-apply", help="Apply the map to an image."
    ),
    lightness_rounding: float | None = typer.Option(
        None,
        "--lightness-rounding",
        min=0.0,
        help="Lower lightness rounding step in CAM02-UCS J' units.",
    ),
    bitonic: bool = typer.Option(
        True, "--bitonic/--no-bitonic", help="Bitonic symmetrization."
    ),
    diffuse: bool = typer.Option(
        True, "--diffuse/--no-diffuse", help="Diffuse symmetrization."
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Prompt for missing choices in text mode.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output JSON without prompting or displaying figures.",
    ),
) -> None:
    """Inspect a map, with optional guided correction, simulation, and apply."""
    if interactive and not as_json:
        if ctype is None:
            ctype = typer.prompt("Colormap type", default=DEFAULT_TYPE)
        if cmap is None:
            cmap = typer.prompt("Colormap name", default=DEFAULT_CMAP)
        if fix is None:
            fix = typer.confirm("Correct the colormap?", default=False)
        if cvd is None:
            cvd = typer.confirm("Generate a CVD simulation?", default=False)
        if apply_output is None:
            apply_output = typer.confirm("Apply to an image?", default=False)
        if apply_output and image is None:
            image = typer.prompt("Image path or builtin key")
        if out is None:
            if apply_output or typer.confirm(
                "Save an assessment?", default=False
            ):
                out = Path(typer.prompt("Output path"))
    payload = _run_workflow(
        command="scicomap wizard",
        cmap=cmap or DEFAULT_CMAP,
        ctype=ctype,
        image=image,
        out=out,
        mode=mode,
        fix=bool(fix),
        cvd=bool(cvd),
        apply_output=bool(apply_output),
        lightness_rounding=lightness_rounding,
        bitonic=bitonic,
        diffuse=diffuse,
        as_json=as_json,
        bundle=False,
        export=export,
    )
    _emit(payload, as_json)


@app.command()
def report(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    image: str | None = typer.Option(
        None, "--image", help="Image path or builtin key for apply."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output report directory; required with --json."
    ),
    mode: str = typer.Option(
        "luminance",
        "--mode",
        help="Image conversion: luminance, first-channel, gray-only.",
    ),
    fix: bool = typer.Option(
        False, "--fix/--no-fix", help="Correct the colormap."
    ),
    cvd: bool = typer.Option(
        False, "--cvd/--no-cvd", help="Generate a CVD simulation."
    ),
    apply_output: bool = typer.Option(
        False, "--apply/--no-apply", help="Apply the map to an image."
    ),
    lightness_rounding: float | None = typer.Option(
        None,
        "--lightness-rounding",
        min=0.0,
        help="Lower lightness rounding step in CAM02-UCS J' units.",
    ),
    bitonic: bool = typer.Option(
        True, "--bitonic/--no-bitonic", help="Bitonic symmetrization."
    ),
    diffuse: bool = typer.Option(
        True, "--diffuse/--no-diffuse", help="Diffuse symmetrization."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Write an inspection report; enable correction, CVD, and apply explicitly."""
    payload = _run_workflow(
        command="scicomap report",
        cmap=cmap,
        ctype=ctype,
        image=image,
        out=out,
        mode=mode,
        fix=fix,
        cvd=cvd,
        apply_output=apply_output,
        lightness_rounding=lightness_rounding,
        bitonic=bitonic,
        diffuse=diffuse,
        as_json=as_json,
        bundle=True,
    )
    _emit(payload, as_json)


@app.command()
def version(
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
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


def main() -> None:
    """Entrypoint for python -m usage."""
    app()


if __name__ == "__main__":
    main()
