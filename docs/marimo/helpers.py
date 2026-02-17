"""Shared helpers for Marimo tutorial notebooks.

This module centralizes small pieces of logic used by both the full local
Marimo tutorial and the browser-oriented WASM lite tutorial.
"""

from __future__ import annotations

from typing import Any

COLORMAP_FAMILIES: tuple[str, ...] = (
    "sequential",
    "diverging",
    "multi-sequential",
    "circular",
    "miscellaneous",
    "qualitative",
)


def build_cmap_options(sci_co_map_cls: Any, ctype: str) -> tuple[list[str], str]:
    """Build colormap options and a recommended default for a family.

    Parameters
    ----------
    sci_co_map_cls : Any
        The ``SciCoMap`` class from ``scicomap.scicomap``.
    ctype : str
        Colormap family name (for example ``"sequential"``).

    Returns
    -------
    tuple[list[str], str]
        A pair containing the sorted colormap names and the default selection.
        For sequential maps, ``"thermal"`` is preferred when available.
    """

    cmap_names = sorted(sci_co_map_cls(ctype=ctype).get_color_map_names())
    default_cmap = cmap_names[0]
    if ctype == "sequential" and "thermal" in cmap_names:
        default_cmap = "thermal"
    return cmap_names, default_cmap


def compute_diagnostics(
    jpapbp: Any,
    classify_fn: Any,
    extrema_fn: Any,
    *,
    include_reasons: bool = False,
) -> dict[str, Any]:
    """Compute lightness and classification diagnostics for a colormap.

    Parameters
    ----------
    jpapbp : Any
        CAM02-UCS coordinates returned by ``scicomap.cmath.transform``.
    classify_fn : Any
        Function used to classify the colormap type.
    extrema_fn : Any
        Function used to compute extrema over the lightness trajectory.
    include_reasons : bool, default=False
        If ``True``, include a human-readable ``reasons`` list in the result.

    Returns
    -------
    dict[str, Any]
        Diagnostic dictionary with keys ``status``, ``classification``,
        ``monotonic_lightness``, and ``extrema_count``. When
        ``include_reasons=True``, includes an additional ``reasons`` key.
    """

    j_values = jpapbp[:, 0]
    is_monotonic = bool(
        (j_values[1:] >= j_values[:-1]).all() or (j_values[1:] <= j_values[:-1]).all()
    )
    cmap_class = classify_fn(jpapbp)
    n_extrema = int(len(extrema_fn(j_values)))

    status = "good"
    reasons: list[str] = []
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

    diagnostics: dict[str, Any] = {
        "status": status,
        "classification": cmap_class,
        "monotonic_lightness": is_monotonic,
        "extrema_count": n_extrema,
    }
    if include_reasons:
        diagnostics["reasons"] = reasons
    return diagnostics
