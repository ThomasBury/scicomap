"""
Module collecting function for evaluating and transforming cmap to render
color weak or blind vision
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Callable, Optional
from colorspacious import cspace_converter
from matplotlib.colors import Colormap, ListedColormap
from scicomap.cmath import get_ctab

_deuter50_space = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 50}
_deuter50_to_sRGB1 = cspace_converter(_deuter50_space, "sRGB1")

_deuter100_space = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}
_deuter100_to_sRGB1 = cspace_converter(_deuter100_space, "sRGB1")

_prot50_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 50}
_prot50_to_sRGB1 = cspace_converter(_prot50_space, "sRGB1")

_prot100_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100}
_prot100_to_sRGB1 = cspace_converter(_prot100_space, "sRGB1")

_trit50_space = {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 50}
_trit50_to_sRGB1 = cspace_converter(_trit50_space, "sRGB1")

_trit100_space = {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 100}
_trit100_to_sRGB1 = cspace_converter(_trit100_space, "sRGB1")


def colorblind_transform(RGBA: np.ndarray, colorblind_space: callable) -> np.ndarray:
    """
    Apply a colorblind transformation to an RGBA image.

    Parameters
    ----------
    RGBA : np.ndarray
        An input RGBA image represented as a NumPy array with shape (..., 4).
    colorblind_space : callable
        A callable representing the colorblind transformation function.
        This function should take an RGB color (as a NumPy array with shape (..., 3))
        as input and return the transformed RGB color as output.

    Returns
    -------
    np.ndarray
        The colorblind-transformed RGBA image, represented as a NumPy array with the same shape
        as the input RGBA, where the RGB channels are transformed and the alpha channel is preserved.

    Notes
    -----
    This function applies a colorblind transformation to an RGBA image by first clipping the input
    colors, applying the colorblind transformation using the provided `colorblind_space` function,
    and then concatenating the transformed RGB channels with the original alpha channel.

    The `colorblind_space` function should accept an RGB color (as a NumPy array with shape (..., 3))
    and return the transformed RGB color.

    Example
    -------
    >>> import numpy as np
    >>> def simulate_colorblindness(rgb):
    ...     # Simulate colorblindness by desaturating the colors
    ...     return np.mean(rgb, axis=-1, keepdims=True)
    >>> rgba_image = np.random.rand(10, 10, 4)  # Example RGBA image
    >>> transformed_image = colorblind_transform(rgba_image, simulate_colorblindness)
    """
    # clipping, alpha handling
    RGB = RGBA[..., :3]
    RGB = np.clip(colorblind_space(RGB), 0, 1)
    return np.concatenate((RGB, RGBA[..., 3:]), axis=-1)


def _get_color_weak_cmap(
    color_map: Union[str, Colormap], n_images: int
) -> Tuple[List[Union[str, Colormap]], List[str]]:
    """
    Generate color maps for different color vision deficiencies.

    Parameters
    ----------
    color_map : str or mcolors.Colormap
        The base color map to be used.
    n_images : int
        The number of color maps to generate, including the base color map.

    Returns
    -------
    Tuple[List[Union[str, mcolors.Colormap]], List[str]]
        A tuple containing two lists:
        1. List of color maps, including the base color map and color maps transformed for color vision deficiencies.
        2. List of subtitles describing each color map in the order they appear in the color maps list.

    Notes
    -----
    This function generates color maps for different color vision deficiencies, including deuteranopia and protanopia,
    by transforming the base color map using colorblind transformations. It also generates color maps for tritanopia
    and a normal color vision map. The `n_images` parameter specifies the total number of color maps to generate.

    Example
    -------
    >>> base_cmap = 'viridis'
    >>> n_color_maps = 5
    >>> c_maps, sub_title = _get_color_weak_cmap(base_cmap, n_color_maps)
    >>> print(c_maps)
    >>> print(sub_title)
    """
    _deuter50_transform = lambda x: colorblind_transform(x, _deuter50_to_sRGB1)
    _deuter100_transform = lambda x: colorblind_transform(x, _deuter100_to_sRGB1)
    _prot50_transform = lambda x: colorblind_transform(x, _prot50_to_sRGB1)
    _trit100_transform = lambda x: colorblind_transform(x, _trit100_to_sRGB1)

    deuter50_cm = _colorblind_cmap(color_map, _deuter50_transform)
    deuter100_cm = _colorblind_cmap(color_map, _deuter100_transform)
    prot50_cm = _colorblind_cmap(color_map, _prot50_transform)
    trit100_cm = _colorblind_cmap(color_map, _trit100_transform)
    c_maps = [color_map, deuter50_cm, prot50_cm, deuter100_cm, trit100_cm]
    n_blank = n_images - 1
    sub_title = (
        ["Normal\n(~95%% of pop)"]
        + n_blank * [""]
        + ["Deuter-50%\n(RG-weak, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Prot-50%\n(RG-weak, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Deuter-100%\n(RG-blind, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Trit-100%%\n(BY deficient, very rare)"]
        + n_blank * [""]
    )

    return c_maps, sub_title


def _get_color_weak_ctab(
    color_map: Union[str, Callable], n_blank: int
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate color tables (ctabs) for different color vision deficiencies.

    Parameters
    ----------
    color_map : str or Callable
        The base color map or a callable function to generate it.
    n_blank : int
        The number of blank entries to insert between each pair of generated color tables.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        A tuple containing two lists:
        1. List of color tables, including the base color table and tables transformed for color vision deficiencies.
        2. List of subtitles describing each color table in the order they appear in the color tables list.

    Notes
    -----
    This function generates color tables (ctabs) for different color vision deficiencies, including deuteranopia and protanopia,
    by transforming the base color table using colorblind transformations. It also generates color tables for tritanopia
    and a normal color vision table. The `n_blank` parameter specifies the number of blank entries between each pair
    of generated color tables.

    Example
    -------
    >>> base_color_map = 'viridis'
    >>> n_blank_entries = 3
    >>> c_tabs, sub_title = _get_color_weak_ctab(base_color_map, n_blank_entries)
    >>> print(c_tabs)
    >>> print(sub_title)
    """
    _deuter100_transform = lambda x: colorblind_transform(x, _deuter100_to_sRGB1)
    ctab = get_ctab(color_map)  # get the colormap as a color table in sRGB
    ctab_deuter100 = _deuter100_transform(ctab)

    _deuter50_transform = lambda x: colorblind_transform(x, _deuter50_to_sRGB1)
    ctab_deuter50 = _deuter50_transform(ctab)

    _prot50_transform = lambda x: colorblind_transform(x, _prot50_to_sRGB1)
    ctab_prot50 = _prot50_transform(ctab)

    _trit100_transform = lambda x: colorblind_transform(x, _trit100_to_sRGB1)
    # _trit100_transform = lambda x: colorblind_transform(x, _trit100_to_sRGB1)
    ctab_trit100 = _trit100_transform(ctab)

    c_tabs = [ctab, ctab_deuter50, ctab_prot50, ctab_deuter100, ctab_trit100]
    sub_title = (
        ["Normal\n(~95%% of pop)"]
        + n_blank * [""]
        + ["Deuter-50%\n(RG-weak, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Prot-50%\n(RG-weak, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Deuter-100%\n(RG-blind, D/P: 5%% of male)"]
        + n_blank * [""]
        + ["Trit-100%\n(BY deficient, very rare)"]
        + n_blank * [""]
    )

    return c_tabs, sub_title


def _colorblind_cmap(
    cmap: Union[str, Callable], c_space_transf: Callable
) -> ListedColormap:
    """
    Create a colorblind-friendly colormap by applying a color space transformation to a base colormap.

    Parameters
    ----------
    cmap : str or Callable
        The base colormap or a callable function to generate it.
    c_space_transf : Callable
        A function for transforming the color space to create a colorblind-friendly colormap.

    Returns
    -------
    ListedColormap
        A colorblind-friendly colormap created by applying the specified color space transformation.

    Notes
    -----
    This function takes a base colormap and applies a color space transformation using the provided `c_space_transf`
    function to create a colorblind-friendly colormap. It returns the resulting colormap as a `ListedColormap` object.

    Example
    -------
    >>> base_colormap = 'viridis'
    >>> def deuteranopia_transform(x):
    ...     # Your color space transformation function for deuteranopia
    ...     pass
    >>> colorblind_cm = _colorblind_cmap(base_colormap, deuteranopia_transform)
    >>> print(colorblind_cm)
    """
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    ctab_cb = c_space_transf(ctab)
    return ListedColormap(np.clip(ctab_cb, 0, 1))


def colorblind_vision(
    cmap: Union[str, List[str], List[Colormap]],
    figsize: Optional[Tuple[int, int]] = None,
    n_colors: int = 10,
    facecolor: str = "black",
) -> plt.Figure:
    """
    Generate a visualization of colorblind-friendly colormaps.

    Parameters
    ----------
    cmap : str, list of str, or list of plt.Colormap
        The base colormap(s) or list of colormaps to visualize.
    figsize : tuple of int, optional
        The size of the generated figure (width, height).
    n_colors : int, optional (default=10)
        The number of colors to include in the colorblind visualization.
    facecolor : {'black', 'white'}, optional (default='black')
        The facecolor for the generated figure.

    Returns
    -------
    plt.Figure
        A Matplotlib figure showing colorblind-friendly versions of the specified colormap(s).

    Notes
    -----
    This function generates a figure displaying colorblind-friendly versions of the input colormap(s).
    It creates subplots for each colormap and visualizes the colormap with a gradient of colors.

    Example
    -------
    >>> base_colormap = 'viridis'
    >>> colorblind_fig = colorblind_vision(base_colormap, figsize=(8, 6))
    >>> plt.show()
    """

    gradient = np.linspace(0, 1, n_colors)
    gradient = np.vstack((gradient, gradient))

    cmap_dic = {}
    visible_spectrum_cmap = plt.get_cmap("gist_rainbow")
    spectral_list, _ = _get_color_weak_cmap(color_map=visible_spectrum_cmap, n_images=0)
    cmap_dic["visible spectrum"] = spectral_list

    if isinstance(cmap, list):
        for cm in cmap:
            cmap_list, _ = _get_color_weak_cmap(color_map=cm, n_images=0)
            cmap_dic[cm.name] = cmap_list
    else:
        cmap_list, _ = _get_color_weak_cmap(color_map=cmap, n_images=0)
        cmap_dic[cmap.name] = cmap_list

    sub_titles = (
        ["Normal\n~95%% of pop"]
        + ["Deuter-50%\nRG-weak, D/P: 5%% of male"]
        + ["Prot-50%\nRG-weak, D/P: 5%% of male"]
        + ["Deuter-100%\nRG-blind, D/P: 5%% of male"]
        + ["Trit-100%\nBY deficient, very rare"]
    )

    nrows = len(spectral_list)
    ncols = len(cmap_dic)

    if figsize is None:
        figsize = (10, 0.25 * nrows)

    fontcolor = "white" if facecolor == "black" else "black"
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, facecolor=facecolor
    )
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    cmap_list = list(cmap_dic.values())
    cmap_name = list(cmap_dic.keys())

    for i, j in itertools.product(range(nrows), range(ncols)):
        ax = axes[i, j]
        cmap = cmap_list[j][i]
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        if i == 0:
            font = {"color": fontcolor, "size": 24}
            ax.set_title(cmap_name[j], fontdict=font)
        if j == 0:
            font = {"color": fontcolor, "size": 14}
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.0
            fig.text(
                x_text, y_text, sub_titles[i], va="center", ha="right", fontdict=font
            )

    for ax in axes:
        ax[0].set_axis_off()
        ax[1].set_axis_off()
    return fig
