"""Scientific colormap catalog, family conveniences, and plotting functions."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
import numpy as np
from scicomap.datasets import load_hill_topography, load_scan_image, load_pic
from typing import List, Tuple, Union, Callable, Optional, Dict, Any

# Scientific Colours
import colorcet as cc
import cmasher as cmr
from cmcrameri import cm as scico
import cmocean
from palettable.cubehelix import perceptual_rainbow_16, classic_16
from palettable.cartocolors.qualitative import (
    Bold_10,
    Pastel_10,
    Prism_10,
    Vivid_10,
)
from palettable.colorbrewer.qualitative import Set1_9

# internal import
from scicomap.cmath import (
    get_ctab,
    uniformize_cmap,
    symmetrize_cmap,
    unif_sym_cmap,
    _ax_cylinder_JCh,
    _ax_scatter_Jpapbp,
)
from scicomap.cblind import _get_color_weak_cmap, colorblind_vision
from scicomap.utils import (
    _pyramid,
    _pyramid_zombie,
    _fn_with_roots,
    _periodic_fn,
    _plot_examples,
    _plot_examples_qual,
    _complex_phase,
)


__all__ = [
    "SciCoMap",
    "ScicoSequential",
    "ScicoMultiSequential",
    "ScicoDiverging",
    "ScicoCircular",
    "ScicoMiscellaneous",
    "ScicoQualitative",
    "get_cmap_dict",
    "plot_colormap",
    "plot_colorblind_vision",
    "compare_cmap",
    "jch_plot",
]


class SciCoMap:
    """Resolve, inspect, transform, and plot a scientific colormap.

    Parameters
    ----------
    ctype : str, optional
        Catalog family: sequential, multi-sequential, diverging, circular,
        miscellaneous, or qualitative.
    cmap : str, matplotlib.colors.Colormap, list, or None, optional
        A name in the selected family, a Matplotlib colormap, or a nonempty
        list of color names or RGB(A) rows in [0, 1]. None selects the family
        default. The resolved object is available as ``cmap`` immediately.

    Raises
    ------
    TypeError
        If the family or colormap input has an unsupported type.
    ValueError
        If the family, name, or sampled color values are invalid.

    Notes
    -----
    Transformations replace ``cmap`` and return that same Matplotlib object.
    Plotting returns a Figure; call ``plt.show()`` explicitly to display it.
    Discover families and names with ``get_cmap_dict()``.

    Examples
    --------
    >>> sc_map = SciCoMap(ctype="sequential", cmap="thermal")
    >>> corrected = sc_map.unif_sym_cmap(lightness_rounding=0)
    >>> corrected is sc_map.cmap
    True
    """

    def __init__(
        self,
        ctype: str = "sequential",
        cmap: str | Colormap | list | None = None,
    ) -> None:
        catalog = get_cmap_dict()
        if not isinstance(ctype, str):
            raise TypeError("ctype must be a catalog family name.")
        if ctype not in catalog:
            raise ValueError(
                f"Unknown colormap family {ctype!r}; choose from {list(catalog)}."
            )
        if cmap is None:
            cmap = {
                "sequential": "thermal",
                "multi-sequential": "bukavu",
                "diverging": "wildfire",
                "circular": "colorwheel",
                "miscellaneous": "turbo",
                "qualitative": "glasbey_dark",
            }[ctype]
        if isinstance(cmap, str):
            if cmap not in catalog[ctype]:
                raise ValueError(
                    f"Unknown colormap {cmap!r} for family {ctype!r}; choose from {list(catalog[ctype])}."
                )
            name = cmap
            cmap = catalog[ctype][cmap]
        elif isinstance(cmap, list):
            cmap = ListedColormap(get_ctab(cmap), name="custom")
            name = cmap.name
        elif isinstance(cmap, Colormap):
            name = cmap.name
        else:
            raise TypeError(
                "cmap must be a catalog name, Matplotlib Colormap, or color list."
            )
        get_ctab(cmap)
        self.ctype = ctype
        self.cmap = cmap
        self.cname = name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ctype={self.ctype!r}, cmap={self.cname!r})"
        )

    def get_mpl_color_map(self) -> Colormap:
        """Return the resolved Matplotlib colormap without changing state.

        Returns
        -------
        matplotlib.colors.Colormap
            The current object stored in ``cmap``.
        """
        return self.cmap

    def uniformize_cmap(
        self, lightness_rounding: float | None = None
    ) -> Colormap:
        """Linearize CAM02-UCS lightness J' and replace the current map.

        Parameters
        ----------
        lightness_rounding : float or None, optional
            Finite nonnegative step in CAM02-UCS J' units. Round the lower
            lightness bound to ceil(J' / step) * step. None or 0 leaves it
            unchanged; this is rounding, not an additive increase.

        Returns
        -------
        matplotlib.colors.Colormap
            The resulting map, also stored in ``cmap``. Unrecognized
            lightness patterns warn and retain the sampled colors.
        """
        self.cmap = uniformize_cmap(
            self.cmap, name=self.cname, lightness_rounding=lightness_rounding
        )
        return self.cmap

    def symmetrize_cmap(
        self, bitonic: bool = True, diffuse: bool = True
    ) -> Colormap:
        """Symmetrize CAM02-UCS chroma C' and replace the current map.

        Parameters
        ----------
        bitonic : bool, optional
            Require a central chroma extremum.
        diffuse : bool, optional
            Smooth the chroma curve.

        Returns
        -------
        matplotlib.colors.Colormap
            The resulting map, also stored in ``cmap``.
        """
        self.cmap = symmetrize_cmap(
            self.cmap, name=self.cname, bitonic=bitonic, diffuse=diffuse
        )
        return self.cmap

    def unif_sym_cmap(
        self,
        lightness_rounding: float | None = None,
        bitonic: bool = True,
        diffuse: bool = True,
    ) -> Colormap:
        """Linearize lightness, then symmetrize chroma, replacing the map.

        Parameters
        ----------
        lightness_rounding : float or None, optional
            Finite nonnegative step in CAM02-UCS J' units. Round the lower
            bound to ceil(J' / step) * step; None or 0 leaves it unchanged.
        bitonic : bool, optional
            Require a central chroma extremum.
        diffuse : bool, optional
            Smooth the chroma curve.

        Returns
        -------
        matplotlib.colors.Colormap
            The resulting map, also stored in ``cmap``.
        """
        self.cmap = unif_sym_cmap(
            self.cmap,
            name=self.cname,
            lightness_rounding=lightness_rounding,
            bitonic=bitonic,
            diffuse=diffuse,
        )
        return self.cmap

    def assess_cmap(self, figsize: tuple[float, float] = (18, 8)) -> Figure:
        """Return a Figure showing lightness, chroma, hue, and CVD simulations.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure width and height in inches.

        Returns
        -------
        matplotlib.figure.Figure
            Assessment plots without implicit display.
        """
        return jch_plot(self.cmap, figsize=figsize)

    def illustrate_palettes(
        self,
        figsize: tuple[float, float] = (12, 10),
        n_colors: int = 256,
        facecolor: str = "black",
    ) -> Figure:
        """Return a Figure of the selected family's catalog palettes.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure width and height in inches.
        n_colors : int, optional
            Number of sampled colors.
        facecolor : str, optional
            Matplotlib background color.

        Returns
        -------
        matplotlib.figure.Figure
            Palettes without implicit display or transformation.
        """
        return plot_colormap(
            self.ctype, "all", figsize, n_colors, facecolor, uniformize=False
        )

    def colorblind(
        self,
        figsize: tuple[float, float] = (12, 5),
        n_colors: int = 256,
        facecolor: str = "black",
    ) -> Figure:
        """Return a Figure of CVD simulations of the current map.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure width and height in inches.
        n_colors : int, optional
            Number of sampled colors.
        facecolor : str, optional
            Matplotlib background color.

        Returns
        -------
        matplotlib.figure.Figure
            Simulations without implicit display or transformation. These
            do not certify accessibility.
        """
        return colorblind_vision(
            [self.cmap],
            figsize=figsize,
            n_colors=n_colors,
            facecolor=facecolor,
        )

    def draw_example(
        self,
        facecolor: str = "black",
        figsize: tuple[float, float] = (20, 20),
        cblind: bool = True,
    ) -> Figure:
        """Return a Figure applying the current map to family-specific data.

        Parameters
        ----------
        facecolor : str, optional
            Matplotlib background color.
        figsize : tuple of float, optional
            Figure width and height in inches.
        cblind : bool, optional
            Include color-vision deficiency simulations.

        Returns
        -------
        matplotlib.figure.Figure
            Examples without implicit display or transformation. Random
            examples use local seeded generators.
        """
        if self.ctype == "qualitative":
            return self._draw_qualitative_example(facecolor, figsize, cblind)
        if self.ctype == "circular":
            images = [
                load_hill_topography(),
                load_scan_image(),
                "electric",
                "complex",
            ]
            arr_3d = None
        else:
            if self.ctype == "sequential":
                pyramid = _pyramid()
            else:
                pyramid = _pyramid_zombie(
                    stacked=self.ctype == "multi-sequential"
                )
            periodic = _periodic_fn()
            arr_3d = [pyramid, periodic]
            if self.ctype in {"sequential", "multi-sequential"}:
                images = [
                    load_hill_topography(),
                    load_scan_image(),
                    pyramid[2],
                    "3D",
                    periodic[2],
                    "3D",
                ]
            else:
                images = [
                    _fn_with_roots(),
                    pyramid[2],
                    "3D",
                    periodic[2],
                    "3D",
                ]
        return _plot_examples(
            color_map=self.cmap,
            images=images,
            arr_3d=arr_3d,
            figsize=figsize,
            facecolor=facecolor,
            cname=self.cname,
            cblind=cblind,
            norm=self.ctype in {"diverging", "miscellaneous"},
        )

    def _draw_qualitative_example(
        self, facecolor: str, figsize: tuple[float, float], cblind: bool
    ) -> Figure:
        # data from United Nations World Population Prospects (Revision 2019)
        # https://population.un.org/wpp/, license: CC BY 3.0 IGO
        year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
        population_by_continent = {
            "africa": [228, 284, 365, 477, 631, 814, 1044, 1275],
            "americas": [340, 425, 519, 619, 727, 840, 943, 1006],
            "asia": [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
            "europe": [220, 253, 276, 295, 310, 303, 294, 293],
            "oceania": [12, 15, 19, 22, 26, 31, 36, 39],
        }
        x = np.linspace(0, 10)
        rng = np.random.default_rng(19680801)
        noisy_trends = np.array(
            [
                np.sin(x) + x + rng.standard_normal(50),
                np.sin(x) + 0.5 * x + rng.standard_normal(50),
                np.sin(x) + 2 * x + rng.standard_normal(50),
                np.sin(x) - 0.5 * x + rng.standard_normal(50),
                np.sin(x) - 2 * x + rng.standard_normal(50),
                np.sin(x) + rng.standard_normal(50),
            ]
        )
        noisy_trends = noisy_trends.T

        dict_arr = [population_by_continent, "scatter", noisy_trends]

        return _plot_examples_qual(
            color_map=self.cmap,
            dict_arr=dict_arr,
            figsize=figsize,
            facecolor=facecolor,
            cname=self.cname,
            year=year,
            cblind=cblind,
        )


class ScicoSequential(SciCoMap):
    """Select the sequential family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'thermal'.
    """

    def __init__(self, cmap: str | Colormap | list = "thermal") -> None:
        super().__init__(ctype="sequential", cmap=cmap)


class ScicoMultiSequential(SciCoMap):
    """Select the multi-sequential family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'bukavu'.
    """

    def __init__(self, cmap: str | Colormap | list = "bukavu") -> None:
        super().__init__(ctype="multi-sequential", cmap=cmap)


class ScicoDiverging(SciCoMap):
    """Select the diverging family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'wildfire'.
    """

    def __init__(self, cmap: str | Colormap | list = "wildfire") -> None:
        super().__init__(ctype="diverging", cmap=cmap)


class ScicoCircular(SciCoMap):
    """Select the circular family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'colorwheel'.
    """

    def __init__(self, cmap: str | Colormap | list = "colorwheel") -> None:
        super().__init__(ctype="circular", cmap=cmap)


class ScicoMiscellaneous(SciCoMap):
    """Select the miscellaneous family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'turbo'.
    """

    def __init__(self, cmap: str | Colormap | list = "turbo") -> None:
        super().__init__(ctype="miscellaneous", cmap=cmap)


class ScicoQualitative(SciCoMap):
    """Select the qualitative family; inherit SciCoMap's operations.

    Parameters
    ----------
    cmap : str, matplotlib.colors.Colormap, or list, optional
        A family catalog name, colormap, or color list. Default: 'glasbey_dark'.
    """

    def __init__(self, cmap: str | Colormap | list = "glasbey_dark") -> None:
        super().__init__(ctype="qualitative", cmap=cmap)


def get_cmap_dict() -> dict[str, dict[str, Colormap]]:
    """
    Get a dictionary of color maps organized by categories.

    Returns
    -------
    dict
        A nested dictionary containing various color maps categorized as 'diverging', 'sequential',
        'multi-sequential', 'circular', 'miscellaneous', and 'qualitative'. Each category
        contains a dictionary of color maps with their associated names.
    """
    cmap_dict = {
        "diverging": {
            "berlin": scico.berlin,
            "bjy": cc.cm.bjy,
            "bky": cc.cm.bky,
            "BrBG": plt.get_cmap("BrBG"),
            "broc": scico.broc,
            "bwr": plt.get_cmap("bwr"),
            "coolwarm": plt.get_cmap("coolwarm"),
            "curl": cmocean.cm.curl,
            "delta": cmocean.cm.delta,
            "fusion": cmr.fusion,
            "fusion_r": cmr.fusion_r,
            "guppy": cmr.guppy,
            "guppy_r": cmr.guppy_r,
            "iceburn": cmr.iceburn,
            "iceburn_r": cmr.iceburn_r,
            "lisbon": scico.lisbon,
            "PRGn": plt.get_cmap("PRGn"),
            "PiYG": plt.get_cmap("PiYG"),
            "pride": cmr.pride,
            "pride_r": cmr.pride_r,
            "PuOr": plt.get_cmap("PuOr"),
            "RdBu": plt.get_cmap("RdBu"),
            "RdGy": plt.get_cmap("RdGy"),
            "RdYlBu": plt.get_cmap("RdYlBu"),
            "RdYlGn": plt.get_cmap("RdYlGn"),
            "redshift": cmr.redshift,
            "redshift_r": cmr.redshift_r,
            "roma": scico.roma,
            "seasons_r": cmr.seasons_r,
            "seismic": plt.get_cmap("seismic"),
            "spectral": plt.get_cmap("Spectral"),
            "turbo": plt.get_cmap("turbo"),
            "vanimo": scico.vanimo,
            "vik": scico.vik,
            "viola": cmr.viola,
            "viola_r": cmr.viola_r,
            "waterlily": cmr.waterlily,
            "waterlily_r": cmr.waterlily_r,
            "watermelon": cmr.watermelon,
            "watermelon_r": cmr.watermelon_r,
            "wildfire": cmr.wildfire,
            "wildfire_r": cmr.wildfire_r,
        },
        "sequential": {
            "afmhot": plt.get_cmap("afmhot"),
            "amber": cmr.amber,
            "amber_r": cmr.amber_r,
            "amp": cmocean.cm.amp,
            "apple": cmr.apple,
            "apple_r": cmr.apple_r,
            "autumn": plt.get_cmap("autumn"),
            "batlow": scico.batlow,
            "bilbao": scico.bilbao,
            "bilbao_r": scico.bilbao_r,
            "binary": plt.get_cmap("binary"),
            "Blues": plt.get_cmap("Blues"),
            "bone": plt.get_cmap("bone"),
            "BuGn": plt.get_cmap("BuGn"),
            "BuPu": plt.get_cmap("BuPu"),
            "chroma": cmr.chroma,
            "chroma_r": cmr.chroma_r,
            "cividis": plt.get_cmap("cividis"),
            "cool": plt.get_cmap("cool"),
            "copper": plt.get_cmap("copper"),
            "cosmic": cmr.cosmic,
            "cosmic_r": cmr.cosmic_r,
            "deep": cmocean.cm.deep,
            "dense": cmocean.cm.dense,
            "dusk": cmr.dusk,
            "dusk_r": cmr.dusk_r,
            "eclipse": cmr.eclipse,
            "eclipse_r": cmr.eclipse_r,
            "ember": cmr.ember,
            "ember_r": cmr.ember_r,
            "fall": cmr.fall,
            "fall_r": cmr.fall_r,
            "gem": cmr.gem,
            "gem_r": cmr.gem_r,
            "gist_gray": plt.get_cmap("gist_gray"),
            "gist_heat": plt.get_cmap("gist_heat"),
            "gist_yarg": plt.get_cmap("gist_yarg"),
            "GnBu": plt.get_cmap("GnBu"),
            "Greens": plt.get_cmap("Greens"),
            "gray": plt.get_cmap("gray"),
            "Greys": plt.get_cmap("Greys"),
            "haline": cmocean.cm.haline,
            "hawaii": scico.hawaii,
            "hawaii_r": scico.hawaii_r,
            "heat": cmr.torch,
            "heat_r": cmr.torch_r,
            "hot": plt.get_cmap("hot"),
            "ice": cmocean.cm.ice,
            "inferno": plt.get_cmap("inferno"),
            "imola": scico.imola,
            "imola_r": scico.imola_r,
            "lapaz": scico.lapaz,
            "lapaz_r": scico.lapaz_r,
            "magma": plt.get_cmap("magma"),
            "matter": cmocean.cm.matter,
            "neon": cmr.neon,
            "neon_r": cmr.neon_r,
            "neutral": cmr.neutral,
            "neutral_r": cmr.neutral_r,
            "nuuk": scico.nuuk,
            "nuuk_r": scico.nuuk_r,
            "ocean": cmr.ocean,
            "ocean_r": cmr.ocean_r,
            "OrRd": plt.get_cmap("OrRd"),
            "Oranges": plt.get_cmap("Oranges"),
            "pink": plt.get_cmap("pink"),
            "plasma": plt.get_cmap("plasma"),
            "PuBu": plt.get_cmap("PuBu"),
            "PuBuGn": plt.get_cmap("PuBuGn"),
            "PuRd": plt.get_cmap("PuRd"),
            "Purples": plt.get_cmap("Purples"),
            "rain": cmocean.cm.rain,
            "rainbow": perceptual_rainbow_16.mpl_colormap,
            "rainbow-sc": scico.batlow,
            "rainbow-sc_r": scico.batlow_r,
            "rainforest": cmr.rainforest,
            "rainforest_r": cmr.rainforest_r,
            "RdPu": plt.get_cmap("RdPu"),
            "Reds": plt.get_cmap("Reds"),
            "savanna": cmr.savanna,
            "savanna_r": cmr.savanna_r,
            "sepia": cmr.sepia,
            "sepia_r": cmr.sepia_r,
            "speed": cmocean.cm.speed,
            "solar": cmocean.cm.solar,
            "spring": plt.get_cmap("spring"),
            "summer": plt.get_cmap("summer"),
            "tempo": cmocean.cm.tempo,
            "thermal": cmocean.cm.thermal,
            "thermal_r": cmocean.cm.thermal_r,
            "thermal-2": cc.cm.bmy,
            "tokyo": scico.tokyo,
            "tokyo_r": scico.tokyo_r,
            "tropical": cmr.tropical,
            "tropical_r": cmr.tropical_r,
            "turbid": cmocean.cm.turbid,
            "turku": scico.turku,
            "turku_r": scico.turku_r,
            "viridis": plt.get_cmap("viridis"),
            "winter": plt.get_cmap("winter"),
            "Wistia": plt.get_cmap("Wistia"),
            "YlGn": plt.get_cmap("YlGn"),
            "YlGnBu": plt.get_cmap("YlGnBu"),
            "YlOrBr": plt.get_cmap("YlOrBr"),
            "YlOrRd": plt.get_cmap("YlOrRd"),
        },
        "multi-sequential": {
            "bukavu": scico.bukavu,
            "fes": scico.fes,
            "infinity": cmr.infinity,
            "infinity_s": cmr.infinity_s,
            "oleron": scico.oleron,
            "topo": cmocean.cm.topo,
        },
        "circular": {
            "bamo": scico.bamO,
            "broco": scico.brocO,
            "cet_c1": cc.cm.CET_C1,
            "colorwheel": cc.cm.colorwheel,
            "corko": scico.corkO,
            "phase": cmocean.cm.phase,
            "rainbow-iso": cc.cm.CET_I1,
            "romao": scico.romaO,
            "seasons": cmr.seasons,
            "seasons_s": cmr.seasons_s,
            "twilight": plt.get_cmap("twilight"),
            "twilight_s": plt.get_cmap("twilight_shifted"),
        },
        "miscellaneous": {
            "oxy": cmocean.cm.oxy,
            "rainbow-kov": cc.cm.rainbow,
            "turbo": plt.get_cmap("turbo"),
        },
        "qualitative": {
            "538": ListedColormap(
                [
                    [0, 143 / 255, 213 / 255],
                    [252 / 255, 79 / 255, 48 / 255],
                    [229 / 255, 174 / 255, 56 / 255],
                    [109 / 255, 144 / 255, 79 / 255],
                    [139 / 255, 139 / 255, 139 / 255],
                    [129 / 255, 15 / 255, 124 / 255],
                ],
                name="538",
            ),
            "bold": ListedColormap(Bold_10.mpl_colors, name="bold"),
            "brewer": ListedColormap(Set1_9.mpl_colors, name="brewer"),
            "colorblind": ListedColormap(
                [
                    [0.1, 0.1, 0.1],
                    [230 / 255, 159 / 255, 0],
                    [86 / 255, 180 / 255, 233 / 255],
                    [0, 158 / 255, 115 / 255],
                    [213 / 255, 94 / 255, 0],
                    [0, 114 / 255, 178 / 255],
                ],
                name="colorblind",
            ),
            "glasbey": cc.cm.glasbey,
            "glasbey_bw": cc.cm.glasbey_bw,
            "glasbey_category10": cc.cm.glasbey_category10,
            "glasbey_dark": cc.cm.glasbey_dark,
            "glasbey_hv": cc.cm.glasbey_hv,
            "glasbey_light": cc.cm.glasbey_light,
            "pastel": ListedColormap(Pastel_10.mpl_colors, name="pastel"),
            "prism": ListedColormap(Prism_10.mpl_colors, name="prism"),
            "vivid": ListedColormap(Vivid_10.mpl_colors, name="vivid"),
        },
    }
    return cmap_dict


def plot_colormap(
    ctype: str,
    cmap_list: Union[str, List[Union[str, Colormap]]] = "all",
    figsize: Optional[Tuple[float, float]] = None,
    n_colors: int = 10,
    facecolor: str = "black",
    uniformize: bool = True,
    symmetrize: bool = False,
    unif_kwargs: Optional[Dict[str, Any]] = None,
    sym_kwargs: Optional[Dict[str, Any]] = None,
) -> Figure:
    """Return a Figure of continuous gradients or qualitative color bars.

    Parameters
    ----------
    ctype : str
        Catalog family used to resolve names.
    cmap_list : list of str or Colormap, or str, optional
        Maps to plot, or "all" for every map in the family.
    figsize : tuple of float or None, optional
        Figure width and height in inches; None selects a size from map count.
    n_colors : int, optional
        Number of sampled colors in continuous gradients. Qualitative bars
        use up to ten colors from each map.
    facecolor : str, optional
        Matplotlib background color.
    uniformize : bool, optional
        Linearize lightness before plotting.
    symmetrize : bool, optional
        Symmetrize chroma before plotting.
    unif_kwargs : dict or None, optional
        Arguments to uniformize_cmap, including lightness_rounding.
    sym_kwargs : dict or None, optional
        Arguments to symmetrize_cmap.

    Returns
    -------
    matplotlib.figure.Figure
        Palette plots without implicit display.
    """
    if sym_kwargs is None:
        sym_kwargs = {}
    if unif_kwargs is None:
        unif_kwargs = {}

    gradient = np.linspace(0, 1, n_colors)
    gradient = np.vstack((gradient, gradient))

    if cmap_list == "all":
        cmap_list = list(get_cmap_dict()[ctype])

    nrows = len(cmap_list)

    if figsize is None:
        figsize = (10, 0.25 * nrows)

    fontcolor = "white" if facecolor == "black" else "black"
    font = {"color": fontcolor, "size": 16}
    fig, axes = plt.subplots(nrows=nrows, figsize=figsize, facecolor=facecolor)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title("Colormaps", fontdict=font)

    for ax, name in zip(axes, cmap_list):
        cmap = SciCoMap(ctype=ctype, cmap=name)

        if uniformize:
            cmap.uniformize_cmap(**unif_kwargs)
        if symmetrize:
            cmap.symmetrize_cmap(**sym_kwargs)

        cmap = cmap.get_mpl_color_map()

        if ctype == "qualitative":
            col_map = cmap(range(10)) if cmap.N > 10 else cmap(range(cmap.N))
            x = np.linspace(0, 1, len(col_map))
            ax.bar(
                x,
                np.ones_like(x),
                color=col_map,
                width=1 / max(len(col_map) - 1, 1),
            )
        else:
            ax.imshow(gradient, aspect="auto", cmap=cmap)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.0

        font = {"color": fontcolor, "size": 12}
        fig.text(x_text, y_text, name, va="center", ha="right", fontdict=font)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()
    return fig


def plot_colorblind_vision(
    ctype: str = "sequential",
    cmap_list: str | list[str | Colormap] = "all",
    figsize: tuple[float, float] | None = None,
    n_colors: int = 10,
    facecolor: str = "black",
    uniformize: bool = True,
    symmetrize: bool = False,
    unif_kwargs: dict[str, Any] | None = None,
    sym_kwargs: dict[str, Any] | None = None,
) -> Figure:
    """Return a Figure comparing simulated color-vision deficiencies.

    Parameters
    ----------
    ctype : str, optional
        Catalog family used to resolve names.
    cmap_list : list of str or Colormap, or str, optional
        Maps to compare, or "all" for every map in the family.
    figsize : tuple of float or None, optional
        Figure width and height in inches; None selects a size from map count.
    n_colors : int, optional
        Number of sampled colors.
    facecolor : str, optional
        Matplotlib background color.
    uniformize : bool, optional
        Linearize lightness before plotting.
    symmetrize : bool, optional
        Symmetrize chroma before plotting.
    unif_kwargs : dict or None, optional
        Arguments to uniformize_cmap, including lightness_rounding.
    sym_kwargs : dict or None, optional
        Arguments to symmetrize_cmap.

    Returns
    -------
    matplotlib.figure.Figure
        Simulations without implicit display. These do not certify accessibility.
    """
    if sym_kwargs is None:
        sym_kwargs = {}
    if unif_kwargs is None:
        unif_kwargs = {}
    cm_list = []

    if cmap_list == "all":
        cmap_list = list(get_cmap_dict()[ctype])

    for name in cmap_list:
        cmap = SciCoMap(ctype=ctype, cmap=name)
        if uniformize:
            cmap.uniformize_cmap(**unif_kwargs)
        if symmetrize:
            cmap.symmetrize_cmap(**sym_kwargs)
        cmap = cmap.get_mpl_color_map()
        cm_list.append(cmap)

    return colorblind_vision(
        cmap=cm_list, figsize=figsize, n_colors=n_colors, facecolor=facecolor
    )


def compare_cmap(
    image: Optional[str] = "scan",
    ctype: str = "sequential",
    cm_list: list[str | Colormap] | None = None,
    ncols: int = 3,
    uniformize: bool = True,
    title: bool = True,
    symmetrize: bool = False,
    unif_kwargs: Optional[Dict[str, Any]] = None,
    sym_kwargs: Optional[Dict[str, Any]] = None,
    facecolor: str = "black",
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Return a Figure applying maps to the same scalar image.

    Parameters
    ----------
    image : str or None, optional
        A JPG/PNG path or bundled example: scan, topography, fn_roots, phase,
        grmhd, vortex, tng, or pyramid. None or unknown names select pyramid.
    ctype : str, optional
        Catalog family used to resolve names.
    cm_list : list of str or Colormap, or None, optional
        Maps to compare; None selects every map in the family.
    ncols : int, optional
        Number of subplot columns.
    uniformize : bool, optional
        Linearize lightness before plotting.
    title : bool, optional
        Show each map's name above its subplot.
    symmetrize : bool, optional
        Symmetrize chroma before plotting.
    unif_kwargs : dict or None, optional
        Arguments to uniformize_cmap, including lightness_rounding.
    sym_kwargs : dict or None, optional
        Arguments to symmetrize_cmap.
    facecolor : str, optional
        Matplotlib background color.
    figsize : tuple of float or None, optional
        Figure width and height in inches; None selects a size from map count.

    Returns
    -------
    matplotlib.figure.Figure
        Comparison plots without implicit display.
    """
    if unif_kwargs is None:
        unif_kwargs = {}
    if sym_kwargs is None:
        sym_kwargs = {}

    if image is not None and not isinstance(image, str):
        raise TypeError(
            "image should be a string or a path to an existing file"
        )

    if (image is not None) and (image.endswith(("jpg", "jpeg", "png"))):
        img = mpimg.imread(image)
        lum_img = img[:, :, 0]
    elif image == "pyramid":
        lum_img = _pyramid()[2]
    elif image == "topography":
        lum_img = load_hill_topography()
    elif image == "fn_roots":
        lum_img = _fn_with_roots()
    elif image == "scan":
        lum_img = load_scan_image()
    elif image == "phase":
        lum_img = _complex_phase()
    elif image == "grmhd":
        lum_img = load_pic(name=image)
    elif image == "vortex":
        lum_img = load_pic(name=image)
    elif image == "tng":
        lum_img = load_pic(name=image)
    else:
        lum_img = _pyramid()[2]

    if cm_list is None:
        cm_list = list(get_cmap_dict()[ctype])

    nrows = int(np.ceil(len(cm_list) / ncols))
    # delete non-used axes
    n_charts = len(cm_list)
    n_subplots = nrows * ncols

    if figsize is None:
        figsize = (2 * ncols, 2.5 * nrows)
    f, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # subplot_kw={"aspect": 1},
        facecolor=facecolor,
        # gridspec_kw={"hspace": 0.0, "wspace": 0.5},
    )
    # Normalize axes to 1D so single-subplot layouts are handled uniformly.
    axs = np.atleast_1d(axs).ravel()
    fontcolor = "white" if facecolor == "black" else "black"

    # loop over the columns to illustrate
    for i, color_map in enumerate(cm_list):
        # select the axis where the map will go
        ax = axs[i]

        chartcm = SciCoMap(ctype=ctype, cmap=color_map)

        if uniformize:
            chartcm.uniformize_cmap(**unif_kwargs)
        if symmetrize:
            chartcm.symmetrize_cmap(**sym_kwargs)

        ax.imshow(lum_img, cmap=chartcm.get_mpl_color_map())
        if title:
            ax.set_title(color_map, fontsize=16, color=fontcolor)
        # Remove axis clutter
        ax.set_axis_off()

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    # plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)
    h_space = 0.25 if title else 0.0
    f.subplots_adjust(wspace=0.0, hspace=h_space)

    return f


def jch_plot(
    cmap: Union[str, Colormap], figsize: Tuple[float, float] = (12, 10)
) -> Figure:
    """Plot CAM02-UCS lightness, chroma, and hue for a colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        A Matplotlib colormap name or object.
    figsize : tuple of float, optional
        Figure width and height in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Assessment plots for normal vision and simulated color-vision
        deficiencies. Hue is displayed in degrees.
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    f = plt.figure(figsize=figsize)
    c_maps, _ = _get_color_weak_cmap(cmap, n_images=2)
    color_map, deuter50_cm, prot50_cm, deuter100_cm, trit100_cm = c_maps

    title_str = cmap if isinstance(cmap, str) else cmap.name
    f.suptitle(title_str, fontsize=24)

    ax0 = f.add_subplot(2, 4, 1)
    ax0 = _ax_cylinder_JCh(ax0, cmap, title="Normal (90-95%% of pop)")

    ax2 = f.add_subplot(2, 4, 3)
    ax2 = _ax_cylinder_JCh(ax2, deuter50_cm, title="Deuter-50%, RG-weak")

    ax4 = f.add_subplot(2, 4, 7)
    ax4 = _ax_cylinder_JCh(ax4, deuter100_cm, title="Deuter-100%, RG-blind")

    ax6 = f.add_subplot(2, 4, 5)
    ax6 = _ax_cylinder_JCh(ax6, trit100_cm, title="Trit-100%, BY deficient")

    ax3d = f.add_subplot(2, 4, 2, projection="3d", elev=25, azim=-75)
    ax3d = _ax_scatter_Jpapbp(ax3d, cmap, title="Normal (90-95%% of pop)")

    ax3d2 = f.add_subplot(2, 4, 4, projection="3d", elev=25, azim=-75)
    ax3d2 = _ax_scatter_Jpapbp(ax3d2, deuter50_cm, title="Deuter-50%, RG-weak")

    ax3d3 = f.add_subplot(2, 4, 8, projection="3d", elev=25, azim=-75)
    ax3d3 = _ax_scatter_Jpapbp(
        ax3d3, deuter100_cm, title="Deuter-100%, RG-blind"
    )

    ax3d4 = f.add_subplot(2, 4, 6, projection="3d", elev=25, azim=-75)
    ax3d4 = _ax_scatter_Jpapbp(
        ax3d4, trit100_cm, title="Trit-100%, BY deficient"
    )

    f.tight_layout()

    return f
