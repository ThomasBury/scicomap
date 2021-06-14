"""
Module collecting function for evaluating and transforming cmap to render
color weak or blind vision
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_converter
from matplotlib.colors import ListedColormap
from scicomap.cmath import get_ctab

_deuter50_space = {"name": "sRGB1+CVD",
                   "cvd_type": "deuteranomaly",
                   "severity": 50}
_deuter50_to_sRGB1 = cspace_converter(_deuter50_space, "sRGB1")

_deuter100_space = {"name": "sRGB1+CVD",
                    "cvd_type": "deuteranomaly",
                    "severity": 100}
_deuter100_to_sRGB1 = cspace_converter(_deuter100_space, "sRGB1")

_prot50_space = {"name": "sRGB1+CVD",
                 "cvd_type": "protanomaly",
                 "severity": 50}
_prot50_to_sRGB1 = cspace_converter(_prot50_space, "sRGB1")

_prot100_space = {"name": "sRGB1+CVD",
                  "cvd_type": "protanomaly",
                  "severity": 100}
_prot100_to_sRGB1 = cspace_converter(_prot100_space, "sRGB1")

_trit50_space = {"name": "sRGB1+CVD",
                 "cvd_type": "tritanomaly",
                 "severity": 50}
_trit50_to_sRGB1 = cspace_converter(_trit50_space, "sRGB1")

_trit100_space = {"name": "sRGB1+CVD",
                  "cvd_type": "tritanomaly",
                  "severity": 100}
_trit100_to_sRGB1 = cspace_converter(_trit100_space, "sRGB1")


def colorblind_transform(RGBA, colorblind_space):
    # clipping, alpha handling
    RGB = RGBA[..., :3]
    RGB = np.clip(colorblind_space(RGB), 0, 1)
    return np.concatenate((RGB, RGBA[..., 3:]), axis=-1)


def _get_color_weak_cmap(color_map, n_images):
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
    sub_title = ["Normal (~95% of pop)"] + n_blank * [""] + \
                ["Deuter-50% (RG-weak, D/P: 5% of male)"] + n_blank * [""] + \
                ["Prot-50% (RG-weak, D/P: 5% of male)"] + n_blank * [""] + \
                ["Deuter-100% (RG-blind, D/P: g5% of male)"] + n_blank * [""] + \
                ["Trit-100% (BY deficient, very rare)"] + n_blank * [""]

    return c_maps, sub_title


def _get_color_weak_ctab(color_map, n_blank):
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
    sub_title = ["Normal (~95% of pop)"] + n_blank * [""] + \
                ["Deuter-50% (RG-weak, D/P: 5% of male)"] + n_blank * [""] + \
                ["Prot-50% (RG-weak, D/P: 5% of male)"] + n_blank * [""] + \
                ["Deuter-100% (RG-blind, D/P: 5% of male)"] + n_blank * [""] + \
                ["Trit-100% (BY deficient, very rare)"] + n_blank * [""]

    return c_tabs, sub_title


def _colorblind_cmap(cmap, c_space_transf):
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    ctab_cb = c_space_transf(ctab)
    return ListedColormap(np.clip(ctab_cb, 0, 1))


def colorblind_vision(cmap, figsize=None, n_colors=10, facecolor="black"):
    """
    Plot the colormap in different color-weak/blind configuration

    :param cmap: Colormap or ListedColormap object
        the color map you want to draw
    :param figsize: 2-uple of int
        the figure size
    :param n_colors: int, default=10
        the number of colors to plot (e.g. 10 for qualitative and 256 for continuous)
    :return:
    """
    gradient = np.linspace(0, 1, n_colors)
    gradient = np.vstack((gradient, gradient))

    cmap_dic = {}
    visible_spectrum_cmap = plt.get_cmap('gist_rainbow')
    spectral_list, _ = _get_color_weak_cmap(color_map=visible_spectrum_cmap, n_images=0)
    cmap_dic['visible spectrum'] = spectral_list

    if isinstance(cmap, list):
        for cm in cmap:
            cmap_list, _ = _get_color_weak_cmap(color_map=cm, n_images=0)
            cmap_dic[cm.name] = cmap_list
    else:
        cmap_list, _ = _get_color_weak_cmap(color_map=cmap, n_images=0)
        cmap_dic[cmap.name] = cmap_list

    sub_titles = ["Normal\n~95$%$ of pop"] + \
                 ["Deuter-50$%$\nRG-weak, D/P: 5$%$ of male"] + \
                 ["Prot-50$%$\nRG-weak, D/P: 5$%$ of male"] + \
                 ["Deuter-100$%$\nRG-blind, D/P: 5$%$ of male"] + \
                 ["Trit-100$%$\nBY deficient, very rare"]

    nrows = len(spectral_list)
    ncols = len(cmap_dic)

    if figsize is None:
        figsize = (10, 0.25 * nrows)

    fontcolor = "white" if facecolor == "black" else "black"
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=facecolor)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    cmap_list = list(cmap_dic.values())
    cmap_name = list(cmap_dic.keys())

    for i, j in itertools.product(range(nrows), range(ncols)):
        ax = axes[i, j]
        cmap = cmap_list[j][i]
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        if i == 0:
            font = {'color': fontcolor, 'size': 24}
            ax.set_title(cmap_name[j], fontdict=font)
        if j == 0:
            font = {'color': fontcolor, 'size': 14}
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.0
            fig.text(x_text, y_text, sub_titles[i], va="center", ha="right", fontdict=font)

    for ax in axes:
        ax[0].set_axis_off()
        ax[1].set_axis_off()
    return fig
