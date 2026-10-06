"""Scientific colormap discovery, diagnostics, transformations, and plots."""

from scicomap import cblind, cmath, datasets
from scicomap._diagnostics import diagnose_cmap
from scicomap.cmath import symmetrize_cmap, unif_sym_cmap, uniformize_cmap
from scicomap.scicomap import (
    SciCoMap,
    ScicoCircular,
    ScicoDiverging,
    ScicoMiscellaneous,
    ScicoMultiSequential,
    ScicoQualitative,
    ScicoSequential,
    compare_cmap,
    get_cmap_dict,
    jch_plot,
    plot_colorblind_vision,
    plot_colormap,
)

__version__ = "1.1.1"
__author__ = "Thomas Bury"
__all__ = [
    "SciCoMap",
    "ScicoSequential",
    "ScicoMultiSequential",
    "ScicoDiverging",
    "ScicoCircular",
    "ScicoMiscellaneous",
    "ScicoQualitative",
    "get_cmap_dict",
    "diagnose_cmap",
    "uniformize_cmap",
    "symmetrize_cmap",
    "unif_sym_cmap",
    "plot_colormap",
    "plot_colorblind_vision",
    "compare_cmap",
    "jch_plot",
    "datasets",
    "cmath",
    "cblind",
]
