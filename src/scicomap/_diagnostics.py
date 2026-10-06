"""Family-aware diagnostics shared by Python, commands, and tutorials."""

from typing import Any

import numpy as np
from matplotlib.colors import Colormap

from scicomap.cmath import classify, get_ctab, transform


def diagnose_cmap(
    cmap_obj: Colormap, ctype: str = "sequential"
) -> dict[str, Any]:
    """Assess sampled lightness according to the intended colormap family.

    Parameters
    ----------
    cmap_obj : matplotlib.colors.Colormap
        Colormap to inspect; samples must be finite RGB(A) values in [0, 1].
    ctype : str, optional
        Intended family: sequential, diverging, multi-sequential, circular,
        qualitative, or miscellaneous.

    Returns
    -------
    dict[str, Any]
        JSON-compatible classification, lightness monotonicity and turn count,
        family, branch progression or circular seam assessment, status, reasons,
        recommendation, and a heuristic flag. Inapplicable branch/seam fields
        are None. Status is good, caution, or fix-recommended.

    Raises
    ------
    ValueError
        If the family or sampled colors are invalid.

    Notes
    -----
    These sampled heuristics and CVD simulations do not certify accessibility.
    """
    colors = transform(get_ctab(cmap_obj))[:, :3]
    lightness = colors[:, 0]
    tolerance = 1e-6
    differences = np.diff(lightness)
    monotonic = bool(
        np.all(differences >= -tolerance) or np.all(differences <= tolerance)
    )
    signs = np.sign(differences[np.abs(differences) > tolerance])
    turns = int(np.count_nonzero(np.diff(signs)))
    reasons = []
    status = "good"
    branches_monotonic = None
    seam_closed = None

    if ctype == "sequential":
        if not monotonic:
            status = "fix-recommended"
            reasons.append("Sequential lightness reverses direction.")
        elif np.ptp(lightness) <= tolerance:
            status = "caution"
            reasons.append("Sequential lightness has no progression.")
    elif ctype in {"diverging", "multi-sequential"}:
        middle = len(lightness) // 2
        left = lightness[: (len(lightness) + 1) // 2]
        right = lightness[middle:]
        if ctype == "multi-sequential":
            left = lightness[: (len(lightness) + 1) // 2]
            right = lightness[(len(lightness) + 1) // 2 :]
        branches_monotonic = all(
            len(branch) >= 2
            and np.ptp(branch) > tolerance
            and (
                np.all(np.diff(branch) >= -tolerance)
                or np.all(np.diff(branch) <= tolerance)
            )
            for branch in (left, right)
        )
        if ctype == "diverging":
            branches_monotonic = branches_monotonic and bool(
                (left[-1] - left[0]) * (right[-1] - right[0]) < 0
            )
        if not branches_monotonic:
            status = "fix-recommended"
            reasons.append(
                "Lightness should progress along each branch"
                + (
                    " toward a central extremum."
                    if ctype == "diverging"
                    else "."
                )
            )
    elif ctype == "circular":
        steps = np.linalg.norm(np.diff(colors, axis=0), axis=1)
        typical_step = float(np.median(steps)) if steps.size else 0.0
        # ponytail: sampled seam and turning-point heuristics; use a measured
        # perceptual task if a circular map needs application-specific limits.
        seam_closed = bool(
            np.linalg.norm(colors[-1] - colors[0])
            <= max(2 * typical_step, tolerance)
        )
        loop_diff = np.diff(np.r_[lightness, lightness[0]])
        loop_signs = np.sign(loop_diff[np.abs(loop_diff) > tolerance])
        loop_turns = int(
            np.count_nonzero(loop_signs != np.roll(loop_signs, 1))
        )
        if not seam_closed:
            status = "caution"
            reasons.append("Circular endpoints have a perceptual seam.")
        if loop_turns > 2:
            status = "caution"
            reasons.append("Circular lightness has multiple oscillations.")
    elif ctype in {"qualitative", "miscellaneous"}:
        status = "caution"
        reasons.append(
            "Unordered colors have no required lightness progression; "
            "inspect their distinctions for the intended data."
        )
    else:
        raise ValueError(f"Unknown colormap family '{ctype}'.")

    return {
        "classification": classify(colors),
        "monotonic_lightness": monotonic,
        "extrema_count": turns,
        "family": ctype,
        "branches_monotonic": branches_monotonic,
        "seam_closed": seam_closed,
        "status": status,
        "reasons": reasons,
        "recommendation": (
            "consider fix"
            if status == "fix-recommended"
            else "inspect in context"
            if status == "caution"
            else "no lightness issue detected"
        ),
        "heuristic": True,
    }
