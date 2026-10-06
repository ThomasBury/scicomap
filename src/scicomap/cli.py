"""Command-line interface for scicomap."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
from click import ClickException
from matplotlib.colors import Colormap
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from scicomap._diagnostics import _diagnose_cmap
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
VALID_PROFILES = {
    "quick-look",
    "publication",
    "presentation",
    "cvd-safe",
    "agent",
}
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "quick-look": {
        "goal": "diagnose",
        "fix": False,
        "cvd": False,
        "apply": None,
        "format": "text",
        "lift": None,
        "bitonic": True,
        "diffuse": True,
        "interactive": True,
    },
    "publication": {
        "goal": "improve",
        "fix": True,
        "cvd": True,
        "apply": None,
        "format": "text",
        "lift": None,
        "bitonic": True,
        "diffuse": True,
        "interactive": True,
    },
    "presentation": {
        "goal": "improve",
        "fix": True,
        "cvd": True,
        "apply": None,
        "format": "text",
        "lift": 10.0,
        "bitonic": True,
        "diffuse": True,
        "interactive": True,
    },
    "cvd-safe": {
        "goal": "diagnose",
        "fix": True,
        "cvd": True,
        "apply": None,
        "format": "json",
        "lift": None,
        "bitonic": True,
        "diffuse": True,
        "interactive": True,
    },
    "agent": {
        "goal": None,
        "fix": False,
        "cvd": False,
        "apply": None,
        "format": "json",
        "lift": None,
        "bitonic": True,
        "diffuse": True,
        "interactive": False,
    },
}


class _WorkflowGroup(TyperGroup):
    """Keep validation and expected runtime failures in the requested format."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        options = {}
        tokens = args[: args.index("--")] if "--" in args else args
        for index, token in enumerate(tokens):
            key, separator, value = token.partition("=")
            if key in {"--format", "--profile"}:
                options[key] = (
                    value
                    if separator
                    else tokens[index + 1]
                    if index + 1 < len(tokens)
                    else None
                )
        ctx.meta["json_output"] = (
            "--json" in tokens
            or options.get("--profile") == "agent"
            or options.get("--format") == "json"
            or (
                options.get("--format") is None
                and options.get("--profile") == "cvd-safe"
                and tokens[:1] == ["wizard"]
            )
        )
        ctx.meta["command"] = "scicomap " + " ".join(
            args[:2] if args and args[0] in {"cmap", "docs"} else args[:1]
        )
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
        except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
            _fail(ctx.meta["command"], str(exc), ctx.meta["json_output"])


app = typer.Typer(
    cls=_WorkflowGroup,
    help="Scientific colormap tools for humans and agents.",
)
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
    try:
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


def _prepare_maps(
    ctype: str, cmap_obj: Colormap, config: dict[str, Any]
) -> tuple[SciCoMap, SciCoMap]:
    original = SciCoMap(ctype=ctype, cmap=cmap_obj)
    selected = original
    if config["fix"]:
        selected = SciCoMap(ctype=ctype, cmap=cmap_obj)
        selected.unif_sym_cmap(
            lift=config["lift"],
            bitonic=config["bitonic"],
            diffuse=config["diffuse"],
        )
    return original, selected


def _validate_apply(config: dict[str, Any], image: Any) -> None:
    if config["apply"] and image is None:
        raise ValueError("Apply stage requires --image.")


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
    diagnostics = data.get("diagnostics", {})
    artifacts = data.get("artifacts", [])

    lines = [
        "scicomap report",
        f"status: {data.get('status', 'unknown')}",
        f"goal: {data.get('goal', 'unknown')}",
        f"cmap: {data.get('cmap', 'unknown')}",
        f"type: {data.get('type', 'unknown')}",
        f"map used: {data.get('map_used', 'unknown')}",
        "Statuses are heuristics; CVD simulations do not certify accessibility.",
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


def _resolve_profile_config(
    *,
    profile: str | None,
    goal: str | None,
    has_image: bool,
    fix: bool | None,
    cvd: bool | None,
    apply_output: bool | None,
    output_format: str | None,
    lift: float | None,
    bitonic: bool | None,
    diffuse: bool | None,
    interactive: bool | None,
) -> tuple[dict[str, Any], list[str]]:
    resolved_profile = profile or "publication"
    if resolved_profile not in VALID_PROFILES:
        raise ValueError(f"Invalid profile '{resolved_profile}'.")

    warnings: list[str] = []
    defaults = PROFILE_DEFAULTS[resolved_profile]
    cfg = dict(defaults)

    if goal is not None:
        cfg["goal"] = goal

    if cfg["goal"] is None:
        cfg["goal"] = "apply" if has_image else "diagnose"

    if apply_output is not None:
        cfg["apply"] = apply_output
    elif defaults["apply"] is None:
        cfg["apply"] = cfg["goal"] == "apply"

    if fix is not None:
        cfg["fix"] = fix
    elif defaults["fix"] is None:
        cfg["fix"] = cfg["goal"] == "improve"

    if cvd is not None:
        cfg["cvd"] = cvd
    elif defaults["cvd"] is None:
        cfg["cvd"] = cfg["goal"] in {"diagnose", "improve"}

    if output_format is not None:
        cfg["format"] = output_format
    elif defaults["format"] is None:
        cfg["format"] = "json" if resolved_profile == "agent" else "text"

    if lift is not None:
        if not np.isfinite(lift):
            raise ValueError("--lift must be finite.")
        cfg["lift"] = lift
    if bitonic is not None:
        cfg["bitonic"] = bitonic
    if diffuse is not None:
        cfg["diffuse"] = diffuse
    if interactive is not None:
        cfg["interactive"] = interactive

    if resolved_profile == "cvd-safe" and cfg["cvd"] is False:
        cfg["cvd"] = True
        warnings.append("cvd-safe profile forces CVD analysis on.")

    if resolved_profile == "agent":
        if cfg["format"] != "json":
            cfg["format"] = "json"
            warnings.append("agent profile forces --format json.")
        if cfg["interactive"] is True:
            warnings.append("agent profile runs in non-interactive mode.")
        cfg["interactive"] = False

    if cfg["goal"] not in VALID_GOALS:
        raise ValueError(f"Invalid goal '{cfg['goal']}'.")
    if cfg["format"] not in VALID_FORMATS:
        raise ValueError(f"Invalid format '{cfg['format']}'.")
    cfg["profile"] = resolved_profile
    return cfg, warnings


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

    diagnostics = _diagnose_cmap(cmap_obj, resolved_type)

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
    if ncols < 1:
        _fail("scicomap compare", "--ncols must be at least 1.", as_json)
    _validate_output(out)
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
        if not img_path.is_file():
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
    if lift is not None and not np.isfinite(lift):
        _fail("scicomap fix", "--lift must be finite.", as_json)
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
    if n_colors < 1:
        _fail("scicomap cvd", "--n-colors must be at least 1.", as_json)
    _validate_output(out)
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


@app.command()
def wizard(
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Workflow profile: quick-look, publication, "
            "presentation, cvd-safe, agent."
        ),
    ),
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
    output_format: str | None = typer.Option(
        None,
        "--format",
        help="Output format: text or json.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON."),
    fix: bool | None = typer.Option(
        None,
        "--fix/--no-fix",
        help="Apply colormap fix in improve workflows.",
    ),
    cvd: bool | None = typer.Option(
        None,
        "--cvd/--no-cvd",
        help="Enable colorblind checks in diagnostics workflows.",
    ),
    apply_output: bool | None = typer.Option(
        None,
        "--apply/--no-apply",
        help="Enable image apply stage.",
    ),
    lift: float | None = typer.Option(
        None,
        "--lift",
        min=0.0,
        max=100.0,
        help="Lift value for improve stage.",
    ),
    bitonic: bool | None = typer.Option(
        None,
        "--bitonic/--no-bitonic",
        help="Bitonic symmetrization for improve stage.",
    ),
    diffuse: bool | None = typer.Option(
        None,
        "--diffuse/--no-diffuse",
        help="Diffuse symmetrization for improve stage.",
    ),
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
    explicit_format = output_format
    if as_json:
        explicit_format = "json"

    try:
        effective, profile_warnings = _resolve_profile_config(
            profile=profile,
            goal=goal,
            has_image=image is not None,
            fix=fix,
            cvd=cvd,
            apply_output=apply_output,
            output_format=explicit_format,
            lift=lift,
            bitonic=bitonic,
            diffuse=diffuse,
            interactive=interactive,
        )
    except ValueError as exc:
        _fail(
            "scicomap wizard",
            str(exc),
            as_json
            or explicit_format == "json"
            or profile == "agent"
            or (profile == "cvd-safe" and explicit_format is None),
        )

    as_json = effective["format"] == "json"
    selected_goal = effective["goal"]
    selected_type = ctype
    selected_cmap = cmap
    selected_image = image
    selected_interactive = effective["interactive"]

    if selected_interactive:
        if selected_type is None:
            selected_type = typer.prompt("Colormap type", default=DEFAULT_TYPE)
        if selected_cmap is None:
            selected_cmap = typer.prompt("Colormap name", default=DEFAULT_CMAP)
        if effective["apply"] and selected_image is None:
            selected_image = Path(typer.prompt("Image path"))
        if out is None and typer.confirm(
            "Save output to file?", default=False
        ):
            out = Path(typer.prompt("Output path"))

    if selected_cmap is None:
        _fail("scicomap wizard", "Missing --cmap value.", as_json)
    if mode not in VALID_MODES:
        _fail("scicomap wizard", f"Invalid mode '{mode}'.", as_json)

    _validate_apply(effective, selected_image)
    _validate_output(out)

    try:
        resolved_type, cmap_obj = _resolve_cmap(selected_cmap, selected_type)
    except ValueError as exc:
        _fail("scicomap wizard", str(exc), as_json)

    result: dict[str, Any] = {
        "goal": selected_goal,
        "cmap": selected_cmap,
        "type": resolved_type,
    }
    warnings = list(profile_warnings)
    _, chart = _prepare_maps(resolved_type, cmap_obj, effective)
    selected_map = chart.get_mpl_color_map()
    result["diagnostics"] = _diagnose_cmap(selected_map, resolved_type)
    result["original_diagnostics"] = _diagnose_cmap(cmap_obj, resolved_type)
    result["map_used"] = "transformed" if effective["fix"] else "original"
    artifacts = []
    mapped = None
    if effective["apply"]:
        mapped = _remap_image(selected_image, selected_map, mode)
        if out is None:
            warnings.append(
                "No --out provided; writing 'scicomap-applied.png' in cwd."
            )
            out = Path("scicomap-applied.png")
        _validate_output(out)

    assess_out = out
    if effective["apply"]:
        assess_out = out.with_name(out.stem + "-assess.png")
    cvd_out = None if out is None else out.with_name(out.stem + "-cvd.png")
    if selected_goal == "improve" or effective["fix"]:
        _validate_output(assess_out)
    if effective["cvd"]:
        _validate_output(cvd_out)

    if selected_goal == "improve" or effective["fix"]:
        result["artifact"] = _save_figure(chart.assess_cmap(), assess_out)
        artifacts.append({"kind": "assessment", "path": result["artifact"]})
    if effective["cvd"]:
        figure = plot_colorblind_vision(
            ctype=resolved_type,
            cmap_list=[selected_map],
            n_colors=256,
            uniformize=False,
            symmetrize=False,
        )
        artifact = _save_figure(figure, cvd_out)
        artifacts.append({"kind": "colorblind", "path": artifact})
    if effective["apply"]:
        out_path = out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, mapped)
        result["artifact"] = str(out_path)
        artifacts.append({"kind": "applied", "path": str(out_path)})
    result["artifacts"] = artifacts
    for artifact in artifacts:
        artifact["map"] = result["map_used"]
    result["actions"] = {
        "fix_applied": effective["fix"],
        "cvd_generated": effective["cvd"],
        "image_applied": effective["apply"],
    }
    result["next_step"] = "inspect the selected map with your data"

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
            "interactive": selected_interactive,
            "profile": effective["profile"],
            "format": effective["format"],
        },
        "data": {**result, "effective_config": effective},
        "warnings": warnings,
        "errors": [],
    }
    _emit(payload, as_json=as_json)


@app.command()
def report(
    cmap: str = typer.Option(DEFAULT_CMAP, "--cmap", help="Colormap name."),
    ctype: str | None = typer.Option(None, "--type", help="Colormap family."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Workflow profile: quick-look, publication, "
            "presentation, cvd-safe, agent."
        ),
    ),
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
    bitonic: bool | None = typer.Option(
        None,
        "--bitonic/--no-bitonic",
        help="Bitonic symmetrization for fix stage.",
    ),
    diffuse: bool | None = typer.Option(
        None,
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
    try:
        effective, profile_warnings = _resolve_profile_config(
            profile=profile,
            goal=goal,
            has_image=image is not None,
            fix=fix,
            cvd=cvd,
            apply_output=apply_output,
            output_format=output_format,
            lift=lift,
            bitonic=bitonic,
            diffuse=diffuse,
            interactive=None,
        )
    except ValueError as exc:
        _fail(
            "scicomap report",
            str(exc),
            output_format == "json" or profile == "agent",
        )

    as_json = effective["format"] == "json"
    resolved_goal = effective["goal"]
    if mode not in VALID_MODES:
        _fail("scicomap report", f"Invalid mode '{mode}'.", as_json)

    run_fix = effective["fix"]
    run_cvd = effective["cvd"]
    run_apply = effective["apply"]
    _validate_apply(effective, image)
    _validate_output(out, directory=True)

    try:
        resolved_type, cmap_obj = _resolve_cmap(cmap, ctype)
    except ValueError as exc:
        _fail("scicomap report", str(exc), as_json)

    original_chart, selected_chart = _prepare_maps(
        resolved_type, cmap_obj, effective
    )
    selected_map = selected_chart.get_mpl_color_map()
    diagnostics = _diagnose_cmap(selected_map, resolved_type)
    original_diagnostics = _diagnose_cmap(cmap_obj, resolved_type)
    mapped = None
    if run_apply and image not in BUILTIN_IMAGES:
        mapped = _remap_image(Path(image), selected_map, mode)
    report_dir = _report_output_dir(out)
    filenames = ["report.json", "summary.txt"]
    if resolved_goal in {"diagnose", "improve"}:
        filenames.append("assess.png")
    if run_fix:
        filenames.append("fixed-assess.png")
    if run_cvd:
        filenames.append("cvd.png")
    if run_apply:
        filenames.append("applied.png")
    for filename in filenames:
        _validate_output(report_dir / filename)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    warnings = list(profile_warnings)

    if resolved_goal in {"diagnose", "improve"}:
        assess_path = report_dir / "assess.png"
        artifact = _save_figure(original_chart.assess_cmap(), assess_path)
        artifacts.append(
            {"kind": "assessment", "path": artifact, "format": "png"}
        )

    if run_fix:
        fixed_path = report_dir / "fixed-assess.png"
        artifact = _save_figure(selected_chart.assess_cmap(), fixed_path)
        artifacts.append(
            {"kind": "fixed_assessment", "path": artifact, "format": "png"}
        )

    if run_cvd:
        cvd_path = report_dir / "cvd.png"
        cvd_fig = plot_colorblind_vision(
            ctype=resolved_type,
            cmap_list=[selected_map],
            n_colors=256,
            uniformize=False,
            symmetrize=False,
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
                cm_list=[selected_map],
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
            plt.imsave(applied_path, mapped)
            artifact = str(applied_path.resolve())
        artifacts.append(
            {"kind": "applied", "path": artifact, "format": "png"}
        )

    for artifact in artifacts:
        artifact["map"] = (
            "original"
            if artifact["kind"] == "assessment" or not run_fix
            else "transformed"
        )

    next_step = "inspect this colormap with your data"
    if diagnostics["status"] == "fix-recommended":
        next_step = "run 'scicomap report --fix' to inspect a correction"

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
            "lift": effective["lift"],
            "bitonic": effective["bitonic"],
            "diffuse": effective["diffuse"],
            "out": str(report_dir),
            "format": effective["format"],
            "profile": effective["profile"],
        },
        "data": {
            "status": diagnostics["status"],
            "goal": resolved_goal,
            "cmap": cmap,
            "type": resolved_type,
            "profile": effective["profile"],
            "diagnostics": diagnostics,
            "original_diagnostics": original_diagnostics,
            "map_used": "transformed" if run_fix else "original",
            "actions": {
                "fix_applied": run_fix,
                "cvd_generated": run_cvd,
                "image_applied": run_apply,
            },
            "artifacts": artifacts,
            "next_step": next_step,
            "report_dir": str(report_dir),
            "effective_config": effective,
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
