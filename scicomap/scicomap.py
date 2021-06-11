"""
Main module for scientific colormaps. It uses the CAM02-UCS color space
Its three coordinates are usually denoted by J', a', and b' and its cylindrical coordinates are J', C', and h'.
This package is built on matplotlib, colorspacious, viscm and EHTplot packages for
the color mathematics and transformations and uses colormaps coming from different packages aiming
to provide scientific colormaps but requiring some adjustments.

 - Provide sequential, multi-sequential, diverging, circular and qualitative (discrete) cmaps
 - Uniformize: linearize the CAM02-UCS lightness J'
 - Symmetrize: make the CAM02-UCS chroma C' symmetrical, bitonic or not, smooth or not
 - Get the matplotlib cmap object, before and after the adjustments
 - Charts to assess the quality of the colormaps (JCh plot)
 - Charts to assess the readability by colour weak/deficient/blind people
 - Charts for illustrating all the available colormaps

The module structure is the following:
---------------------------------------
- ``SciCoMap`` Parent class for all the colormap types
- ``ScicoSequential`` Child class for sequential colormaps
- ``ScicoMultiSequential`` Child class for multi-sequential colormaps
- ``ScicoDiverging`` Child class for diverging colormaps
- ``ScicoCircular`` Child class for circular colormaps (circular diverging and circular flat aka phase)
- ``ScicoMiscellaneous`` Child class for continuous colormaps which are none of the above
- ``ScicoQualitative`` Child class for discrete colormaps
- ``plot_colormap`` function for illustrating all the color maps of a given type
- ``plot_colorblind_vision`` function for comparing the rendering with different color deficiencies
- ``compare_cmap`` function for comparing the rendering when using an image.

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import numpy as np
from scicomap.datasets import load_hill_topography, load_scan_image, load_pic

# Scientific Colours
import colorcet as cc
import cmasher as cmr
from cmcrameri import cm as scico
import cmocean
from palettable.cubehelix import perceptual_rainbow_16, classic_16
from palettable.cartocolors.qualitative import Bold_10, Pastel_10, Prism_10, Vivid_10
from palettable.colorbrewer.qualitative import Set1_9

# internal import
from scicomap.cmath import uniformize_cmap, symmetrize_cmap, unif_sym_cmap, \
    _ax_cylinder_JCh, _ax_scatter_Jpapbp
from scicomap.cblind import _get_color_weak_cmap, colorblind_vision
from scicomap.utils import _pyramid, _pyramid_zombie, _fn_with_roots, \
    _periodic_fn, _plot_examples, _plot_examples_qual, _complex_phase


class SciCoMap:
    """
    Get a matplotlib compatible color map from different packages.
    Mainly scientific color maps. Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param ctype: str
        color map type, either 'sequential', 'multi-sequential', 'diverging',
         'circular', 'miscellaneous'  or 'qualitative'
    :param cmap: str or cmap object
        the name of the color map you want to use or the matplotlib cmap object


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map
    uniformized: Boolean
        if the cmap has been uniformized or not

    Methods:
    --------
    get_ctype()
        get the colormap type

    get_mpl_color_map()
        get the matplotlib color map (cmap object)

    uniformize_cmap(lift=None)
        uniformize the colormap (linearize the brightness J')
        You can use `lift` to make it brighter (increase the
        minimum brightness of the colormap)

    symmetrize_cmap(bitonic=True, diffuse=True)
        symmetrise the hue (h'), using bi-tonic or not
        and smoothing or not the hue curve

    unif_sym_cmap(lift=None, bitonic=True, diffuse=True)
        uniformize and symmetrize at once (first uniformize and then symmetrize)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    assess_cmap(figsize=(18, 8))
        plot the Jch tensor to visualize the brightness (J'), the hue (h')
        and the chroma (c')

    illustrate_palettes(figsize=(12, 10), n_colors=256):
        plot the gradient or the discrete palettes of all the colormaps of the given type

    colorblind(figsize=(12, 5), n_colors=256, facecolor="black")
        plot the gradient or barchart of the colormap for different kinds of color deficiencies

    Example:
    --------
    ccmap = SciCoMap(ctype='sequential', cname='thermal')
    mpl_map = ccmap.get_mpl_color_map()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/

    etc.
    """

    def __init__(self, ctype="sequential", cmap="thermal"):

        self.ctype = ctype
        self.color_map_dic = get_cmap_dict()

        self.cmap = cmap
        self.cname = 'cmap'
        self.uniformized = False

    def __repr__(self):

        s = "SciCoMap(ctype={ctype}, \n" \
            "     cmap={cmap})".format(ctype=self.ctype, cmap=self.cname)

        return s

    @classmethod
    def get_ctype(cls):
        """ return the colormap type """
        return get_cmap_dict().keys()

    def get_mpl_color_map(self):
        """
        Get the matplotlib colormap object

        :return:
         color_map: mpl colormap
        """
        if isinstance(self.cmap, str):
            if self.cmap not in self.color_map_dic[self.ctype].keys():
                raise ValueError(
                    "Current builtin cmap are: {}".format(
                        self.color_map_dic[self.ctype].keys()
                    )
                )
            self.cname = self.cmap
            self.cmap = self.color_map_dic[self.ctype][self.cmap]
        else:
            self.cname = self.cmap.name

        return self.cmap

    def uniformize_cmap(self, lift=None):
        """
        Uniformize the colormap, meaning linearizing the brightness (J')
        in the CAM02-UCS color space.

        :param lift: None or int in [0, 100]
            lift or not the darkest part of the cmap

        :return:
        """
        self.get_mpl_color_map()
        self.cmap, self.uniformized = uniformize_cmap(cmap=self.cmap,
                                                      name=self.cname,
                                                      lift=lift,
                                                      uniformized=self.uniformized)

    def symmetrize_cmap(self, bitonic=True, diffuse=True):
        """
        Symmetrize the colormap (the hue) in the CAM02-UCS color space.
        It can be symmetrized in a bitonic way or not (if bitonic, the hue
        curve will be symmetric with an extremum at its centre)

        The hue curve can be smoothed (diffuse) or not (edges might occur)

        :param bitonic: Boolean, default=True
            Bitonic symmetrization or not (extremum located at the centre of the hue curve)
        :param diffuse: Boolean, default=True
            Smooth hue curve or not (if not, edges might occur)
        :return:
        """
        self.get_mpl_color_map()
        self.cmap = symmetrize_cmap(cmap=self.cmap,
                                    name=self.cname,
                                    bitonic=bitonic,
                                    diffuse=diffuse)

    def unif_sym_cmap(self, lift=None, bitonic=True, diffuse=True):
        """
        First, uniformize the colormap, meaning linearizing the brightness (J')
        in the CAM02-UCS color space.

        Second, symmetrize the colormap (the hue) in the CAM02-UCS color space.
        It can be symmetrized in a bitonic way or not (if bitonic, the hue
        curve will be symmetric with an extremum at its centre)

        The hue curve can be smoothed (diffuse) or not (edges might occur)

        :param lift: None or int in [0, 100]
            lift or not the darkest part of the cmap
        :param bitonic: Boolean, default=True
            Bitonic symmetrization or not (extremum located at the centre of the hue curve)
        :param diffuse: Boolean, default=True
            Smooth hue curve or not (if not, edges might occur)
        :return:
        """
        self.get_mpl_color_map()
        self.cmap, self.uniformized = unif_sym_cmap(cmap=self.cmap,
                                                    name=self.cname,
                                                    lift=lift,
                                                    uniformized=self.uniformized,
                                                    bitonic=bitonic,
                                                    diffuse=diffuse)

    def get_color_map_names(self):
        """
        Get the names of the implemented colormaps for the chosen ctype

        :return: list
            list of colormap names
        """
        return self.color_map_dic[self.ctype].keys()

    @classmethod
    def get_color_map_dic(cls):
        """
        Get the mapping dict of all the available color maps

        :return: dict,
            the dictionary of all the color maps for all ctypes
        """
        return get_cmap_dict()

    def assess_cmap(self, figsize=(18, 8)):
        """
        Stolen from the ehtplot package

        Plot J', C', and h' of a colormap as function of the mapped value

        The CAM02-UCS lightness J' is linearized for
        generating perceptually uniform colormaps (working definition of Perceptually
        Uniform Sequential colormaps by matplotlib).

        Hue h' can encode an additional physical quantity in an image
        (when used in this way, the change of hue should be linearly
        proportional to the quantity)

        The other dimension chroma is less recognizable and should not be
        used to encode physical information. Since sRGB is only a subset
        of the Lab color space, there are human recognizable colors that
        are not displayable. In order to accurately represent the physical
        quantities

        :param figsize: 2-uple of int
            the figure size
        :return fig object
            the matplotlib figure object
        """
        color_map = self.get_mpl_color_map()
        return jch_plot(cmap=color_map, figsize=figsize)

    def illustrate_palettes(self, figsize=(12, 10), n_colors=256, facecolor="black"):
        """
        Draw the gradient or discrete color palettes for each colormaps of the chosen ctype

        :param figsize: 2-uple of int
            the figure size
        :param n_colors: int, default=10
            the number of colors to plot (e.g. 10 for qualitative and 256 for continuous)
        :param facecolor: str
            the chart face color. It should be a string of builtin matplotlib colors or a string
            corresponding to a hex color.

        :return:
        """
        cmap_list = self.get_color_map_names()
        ctype = self.ctype
        plot_colormap(ctype, cmap_list, figsize, n_colors, facecolor)
        plt.show()

    def colorblind(self, figsize=(12, 5), n_colors=256, facecolor="black"):
        """
        Draw the gradient or discrete color palettes for different kind of color vision deficiencies
        (provide the color vision deficiencies rendering of the visible spectrum for comparison).

        :param figsize: 2-uple of int
            the figure size
        :param n_colors: int, default=10
            the number of colors to plot (e.g. 10 for qualitative and 256 for continuous)
        :param facecolor: str
            the chart face color. It should be a string of builtin matplotlib colors or a string
            corresponding to a hex color.

        :return:
        """
        plot_colorblind_vision(ctype=self.ctype,
                               cmap_list=[self.get_mpl_color_map()],
                               figsize=figsize,
                               n_colors=n_colors,
                               facecolor=facecolor)


class ScicoSequential(SciCoMap):
    """
    Get a matplotlib compatible sequential color map from different packages
    providing scientific color maps. The color map can be uniformized and symmetrized if needed.
    The perception can be assessed as well as the color vision deficient rendering.

    This is useful for continuous values, for which there is no "centre" or mid value.

    Some are suited to a dark background and other for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoSequential(cname='chroma')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="chroma"):
        super().__init__(cmap=cmap, ctype="sequential")

    def __repr__(self):

        s = "ScicoSequential(cmap={cmap})".format(cmap=self.cname)

        return s


    def draw_example(self, facecolor="black", figsize=(20, 20), cblind=True):
        color_map = self.get_mpl_color_map()
        elevation = load_hill_topography()
        scan_im = load_scan_image()
        xpyr, ypyr, zpyr = _pyramid()
        per_x, per_z, per_z = _periodic_fn()
        images = [elevation, scan_im, zpyr, "3D", per_z, "3D"]

        fig = _plot_examples(color_map=color_map,
                             images=images,
                             arr_3d=[(xpyr, ypyr, zpyr), (per_x, per_z, per_z)],
                             figsize=figsize,
                             facecolor=facecolor,
                             cname=self.cname,
                             cblind=cblind)

        return fig


class ScicoMultiSequential(SciCoMap):
    """
    Get a matplotlib compatible sequential color map from different packages.
    This is useful for continuous values, for which there is no "centre" or mid value
    (if there is one, use ScicoDiverging)
    Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoSequential(cname='chroma')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="chroma"):
        super().__init__(cmap=cmap, ctype="multi-sequential")

    def __repr__(self):

        s = "ScicoMultiSequential(cmap={cmap})".format(cmap=self.cname)

        return s

    def draw_example(self, facecolor="black", figsize=(20, 20), cblind=True):
        color_map = self.get_mpl_color_map()
        elevation = load_hill_topography()
        scan_im = load_scan_image()
        xpyr, ypyr, zpyr = _pyramid_zombie(stacked=True)
        per_x, per_z, per_z = _periodic_fn()
        images = [elevation, scan_im, zpyr, "3D", per_z, "3D"]

        fig = _plot_examples(color_map=color_map,
                             images=images,
                             arr_3d=[(xpyr, ypyr, zpyr), (per_x, per_z, per_z)],
                             figsize=figsize,
                             facecolor=facecolor,
                             cname=self.cname,
                             cblind=cblind)

        return fig


class ScicoDiverging(SciCoMap):
    """
    Get a matplotlib compatible diverging color map from different packages.
    This is useful for continuous values, for which there is a "centre" or mid value
    (if there isn't, use ScicoSequential)
    Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use

    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoDiverging(cname='redshift')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="wildfire"):
        super().__init__(cmap=cmap, ctype="diverging")

    def __repr__(self):

        s = "ScicoDiverging(cmap={cmap})".format(cmap=self.cname)

        return s

    def draw_example(self, facecolor="black", figsize=(20, 20)):
        color_map = self.get_mpl_color_map()
        # Create diverging image data
        image_div = _fn_with_roots()
        xpyr, ypyr, zpyr = _pyramid_zombie(stacked=False)
        per_x, per_z, per_z = _periodic_fn()

        images = [image_div, zpyr, '3D', per_z, "3D"]

        fig = _plot_examples(color_map=color_map,
                             images=images,
                             arr_3d=[(xpyr, ypyr, zpyr), (per_x, per_z, per_z)],
                             figsize=figsize,
                             facecolor=facecolor,
                             cname=self.cname)

        return fig


class ScicoCircular(SciCoMap):
    """
    Get a matplotlib compatible circular color map from different packages.
    This is useful for anglular values (circular "flat" as the phase cmap),
    for which there is no "centre" or mid value
    (if there is one, use ScicoDiverging)
    Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoSequential(cname='chroma')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="colorwheel"):
        super().__init__(cmap=cmap, ctype="circular")

    def __repr__(self):

        s = "ScicoCircular(cmap={cmap})".format(cmap=self.cname)

        return s

    def draw_example(self, facecolor="black", figsize=(20, 20), cblind=True):
        color_map = self.get_mpl_color_map()
        elevation = load_hill_topography()
        scan_im = load_scan_image()
        per_x, per_z, per_z = _periodic_fn()
        images = [elevation, scan_im, "electric", "complex"]

        fig = _plot_examples(color_map=color_map,
                             images=images,
                             arr_3d=None,
                             figsize=figsize,
                             facecolor=facecolor,
                             cname=self.cname,
                             cblind=cblind)

        return fig


class ScicoMiscellaneous(SciCoMap):
    """
    Get a matplotlib compatible sequential color map from different packages.
    This is useful for continuous values, for which there is no "center" or mid value
    (if there is one, use ScicoDiverging)
    Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoSequential(cname='chroma')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="turbo"):
        super().__init__(cmap=cmap, ctype="miscellaneous")

    def __repr__(self):

        s = "ScicoMiscellaneous(cmap={cmap})".format(cmap=self.cname)

        return s

    def draw_example(self, facecolor="black", figsize=(20, 20)):
        color_map = self.get_mpl_color_map()
        # Create diverging image data
        image_div = _fn_with_roots()
        xpyr, ypyr, zpyr = _pyramid_zombie(stacked=False)
        per_x, per_z, per_z = _periodic_fn()

        images = [image_div, zpyr, '3D', per_z, "3D"]

        fig = _plot_examples(color_map=color_map,
                             images=images,
                             arr_3d=[(xpyr, ypyr, zpyr), (per_x, per_z, per_z)],
                             figsize=figsize,
                             facecolor=facecolor,
                             cname=self.cname)

        return fig


class ScicoQualitative(SciCoMap):
    """
    Get a matplotlib compatible qualitative list of colors from different packages.
    This is useful for discrete values or categorical variables.
    Some are suited to a dark background and others for light backgrounds.

    Params:
    -------
    :param cmap: str or matplotlib cmap
        the name of the color map you want to use


    Attributes:
    -----------
    color_map_dic: dict
        the mapping dictionary for some colormaps
    ctype: str,
        color map type, either 'sequential', 'diverging' or 'qualitative'
    cname: str,
        the name of the color map you want to use
    cmap: matplotlib cmap or list of hex/rgb
        the color map

    Methods:
    --------
    get_mpl_color_map()
        get the matplotlib color map (or list of hex/rgb colors for qualitative)

    get_color_map_names()
        get the name of all the available color maps

    get_color_map_dic()
        get the color maps dictionary

    illustrate_palettes():
        plot the gradient or the discrete palettes (all of them)

    draw_example(facecolor="black")
        draw two charts for illustrative purposes

    Example:
    --------
    sc_map = ScicoQualitative(cname='glasbey_dark')
    mpl_map = sc_map.get_mpl_color_map()
    sc_map.draw_example()



    References:
    -----------
    https://www.kennethmoreland.com/color-advice/
    https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
    http://www.fabiocrameri.ch/colourmaps.php
    https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/
    """

    def __init__(self, cmap="glasbey_dark"):
        super().__init__(cmap=cmap, ctype="qualitative")

    def __repr__(self):

        s = "ScicoQualitative(cmap={cmap})".format(cmap=self.cname)

        return s

    def draw_example(self, facecolor="black", figsize=(20, 20)):
        color_map = self.get_mpl_color_map()

        # data from United Nations World Population Prospects (Revision 2019)
        # https://population.un.org/wpp/, license: CC BY 3.0 IGO
        year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
        population_by_continent = {
            'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
            'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
            'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
            'europe': [220, 253, 276, 295, 310, 303, 294, 293],
            'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
        }
        x = np.linspace(0, 10)
        # Fixing random state for reproducibility
        np.random.seed(19680801)
        noisy_trends = np.array(
            [np.sin(x) + x + np.random.randn(50),
             np.sin(x) + 0.5 * x + np.random.randn(50),
             np.sin(x) + 2 * x + np.random.randn(50),
             np.sin(x) - 0.5 * x + np.random.randn(50),
             np.sin(x) - 2 * x + np.random.randn(50),
             np.sin(x) + np.random.randn(50)]
        )
        noisy_trends = noisy_trends.T

        dict_arr = [population_by_continent, "scatter", noisy_trends]

        return _plot_examples_qual(color_map=color_map,
                                   dict_arr=dict_arr,
                                   figsize=figsize,
                                   facecolor=facecolor,
                                   cname=self.cname,
                                   year=year)


def get_cmap_dict():
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
            "hawaii_r": scico.hawaii,
            "heat": cmr.heat,
            "heat_r": cmr.heat_r,
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
            "nuuk_r": scico.nuuk,
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
            "538": ListedColormap([[0, 143 / 255, 213 / 255],
                                   [252 / 255, 79 / 255, 48 / 255],
                                   [229 / 255, 174 / 255, 56 / 255],
                                   [109 / 255, 144 / 255, 79 / 255],
                                   [139 / 255, 139 / 255, 139 / 255],
                                   [129 / 255, 15 / 255, 124 / 255]], name="538"),
            "bold": ListedColormap(Bold_10.mpl_colors, name="bold"),
            "brewer": ListedColormap(Set1_9.mpl_colors, name="brewer"),
            "colorblind": ListedColormap(
                [[0.1, 0.1, 0.1],
                 [230 / 255, 159 / 255, 0],
                 [86 / 255, 180 / 255, 233 / 255],
                 [0, 158 / 255, 115 / 255],
                 [213 / 255, 94 / 255, 0],
                 [0, 114 / 255, 178 / 255]], name="colorblind"),
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


def get_available_ctype():
    """ return available the colormap type """
    return get_cmap_dict.keys()


def plot_colormap(ctype, cmap_list='all', figsize=None, n_colors=10, facecolor="black",
                  uniformize=True, symmetrize=False, unif_kwargs=None, sym_kwargs=None
                  ):
    """
    Plot the gradient of the corresponding color palette (or bar plot if qualitative)

    :param ctype: str, default="sequential"
        the color map type
    :param cmap_list: list of string or 'all', default='all
        list of color map names to draw
    :param figsize: 2-uple of int
        the figure size
    :param n_colors: int, default=10
        the number of colors to plot (e.g. 10 for qualitative and 256 for continuous)
    :param facecolor: str
        the chart face color. It should be a string of builtin matplotlib colors or a string
        corresponding to a hex color.
    :param uniformize: Boolean, default=True
        uniformize or not the cmap before plotting
    :param symmetrize: Boolean, default=False
        symmetrize or not the cmap before plotting
    :param unif_kwargs: dict or None
        the kwargs for the uniformize_cmap method
    :param sym_kwargs: dict or None
        the kwargs for the symmetrize_cmap method
    :return:
    """
    if sym_kwargs is None:
        sym_kwargs = {}
    if unif_kwargs is None:
        unif_kwargs = {}

    gradient = np.linspace(0, 1, n_colors)
    gradient = np.vstack((gradient, gradient))

    if cmap_list == 'all':
        cmap_list = list(SciCoMap(ctype=ctype).get_color_map_names())

    nrows = len(cmap_list)

    if figsize is None:
        figsize = (10, 0.25 * nrows)

    fontcolor = "white" if facecolor == "black" else "black"
    font = {'color': fontcolor, 'size': 16}
    fig, axes = plt.subplots(nrows=nrows, figsize=figsize, facecolor=facecolor)
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
            ax.bar(x, np.ones_like(x), color=col_map, width=1 / (len(col_map) - 1))
        else:
            ax.imshow(gradient, aspect="auto", cmap=cmap)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.0

        font = {'color': fontcolor, 'size': 12}
        fig.text(x_text, y_text, name, va="center", ha="right", fontdict=font)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()
    return fig


def plot_colorblind_vision(ctype='sequential', cmap_list='all', figsize=None,
                           n_colors=10, facecolor="black", uniformize=True,
                           symmetrize=False, unif_kwargs=None, sym_kwargs=None):
    """

    Render the color map (adjusted or not) in different color deficiencies vision


    :param ctype: str, default="sequential"
        the color map type
    :param cmap_list: list of string or 'all', default='all
        list of color map names to draw
    :param figsize: 2-uple of int
        the figure size
    :param n_colors: int, default=10
        the number of colors to plot (e.g. 10 for qualitative and 256 for continuous)
    :param facecolor: str
        the chart face color. It should be a string of builtin matplotlib colors or a string
        corresponding to a hex color.
    :param uniformize: Boolean, default=True
        uniformize or not the cmap before plotting
    :param symmetrize: Boolean, default=False
        symmetrize or not the cmap before plotting
    :param unif_kwargs: dict or None
        the kwargs for the uniformize_cmap method
    :param sym_kwargs: dict or None
        the kwargs for the symmetrize_cmap method
    :return:
    """
    if sym_kwargs is None:
        sym_kwargs = {}
    if unif_kwargs is None:
        unif_kwargs = {}
    cm_list = []

    if cmap_list == 'all':
        cmap_list = list(SciCoMap(ctype=ctype).get_color_map_names())

    for name in cmap_list:
        cmap = SciCoMap(ctype=ctype, cmap=name)
        if uniformize:
            cmap.uniformize_cmap(**unif_kwargs)
        if symmetrize:
            cmap.symmetrize_cmap(**sym_kwargs)
        cmap = cmap.get_mpl_color_map()
        cm_list.append(cmap)

    return colorblind_vision(cmap=cm_list, figsize=figsize, n_colors=n_colors, facecolor=facecolor)


def compare_cmap(image='scan', ctype="sequential", cm_list=None, ncols=3, uniformize=True,
                 symmetrize=False, unif_kwargs=None, sym_kwargs=None, facecolor="black"):
    """
    Utility function to visualize how the different color maps render the details and the information.
    You can pass the image of your choice, like a topographic profile for sequential and sea-earth level for
    diverging (negative and positive values) for instance.
    Some are suited to a dark background and others for light backgrounds.

    :param image: str or None
        the path to the jpg or png picture you want to use for the comparison or one
        of the builtin images as a pyramid image to visualize if there is any artifact
    :param ctype: str,
        either "sequential", "diverging", "qualitative"
    :param cm_list: list of str or None
        the list of cmaps you want to compare, if None all of them (for the chosen ctype) will be compared
    :param ncols: int
        the number of columns in the matplotlib subplot figure
    :param uniformize: Boolean, default=True
        uniformize or not the cmap before plotting
    :param symmetrize: Boolean, default=False
        symmetrize or not the cmap before plotting
    :param unif_kwargs: dict or None
        the kwargs for the uniformize_cmap method
    :param sym_kwargs: dict or None
        the kwargs for the symmetrize_cmap method
    :param facecolor: str
        the chart face color. It should be a string of builtin matplotlib colors or a string
        corresponding to a hex color.

    :return: f, matplotlib figure
    """

    if unif_kwargs is None:
        unif_kwargs = {}
    if sym_kwargs is None:
        sym_kwargs = {}

    if image is not None and not isinstance(image, str):
        raise TypeError("image should be a string or a path to an existing file")

    if (image is not None) and (image.endswith(("jpg", "jpeg", "png"))):
        img = mpimg.imread(image)
        lum_img = img[:, :, 0]
    elif image == "pyramid":
        lum_img = _pyramid()
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
        lum_img = _pyramid()

    if cm_list is None:
        cm_list = list(SciCoMap(ctype=ctype).get_color_map_names())

    nrows = int(np.ceil(len(cm_list) / ncols))
    # delete non-used axes
    n_charts = len(cm_list)
    n_subplots = nrows * ncols

    figsize = (2 * ncols, 2.5 * nrows)
    f, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # subplot_kw={"aspect": 1},
        facecolor=facecolor,
        # gridspec_kw={"hspace": 0.0, "wspace": 0.5},
    )
    # Make the axes accessible with single indexing
    axs = axs.flatten()
    fontcolor = "white" if facecolor == "black" else "black"

    # loop over the columns to illustrate
    for i, color_map in enumerate(cm_list):
        # select the axis where the map will go
        if n_charts > 1:
            ax = axs[i]
        else:
            ax = axs

        chartcm = SciCoMap(ctype=ctype, cmap=color_map)

        if uniformize:
            chartcm.uniformize_cmap(**unif_kwargs)
        if symmetrize:
            chartcm.symmetrize_cmap(**sym_kwargs)

        ax.imshow(lum_img, cmap=chartcm.get_mpl_color_map())
        ax.set_title(color_map, fontsize=16, color=fontcolor)
        # Remove axis clutter
        ax.set_axis_off()

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    # plt.tight_layout(pad=0, w_pad=0.5, h_pad=0)
    f.subplots_adjust(wspace=0, hspace=.25)
    return f


def jch_plot(cmap, figsize=(12, 10)):
    """
    Stolen from the ehtplot package

    Plot J', C', and h' of a colormap as function of the mapped value

    The CAM02-UCS lightness J' should serve us as a good approximation for
    generating perceptually uniform colormaps. In fact, linearity in J'
    is used as the working definition of Perceptually Uniform Sequential
    colormaps by matplotlib.

    Hue h' can encode an additional physical quantity in an image
    (when used in this way, the change of hue should be linearly
    proportional to the quantity)

    The other dimension chroma is less recognizable and should not be
    used to encode physical information. Since sRGB is only a subset
    of the Lab color space, there are human recognizable colors that
    are not displayable. In order to accurately represent the physical
    quantities.

    :param cmap: string or matplotlib.colors.Colormap): The colormap to
            be plotted.
    :param figsize: 2-uple of int
        the figure size
    """
    f = plt.figure(figsize=figsize)
    c_maps, _ = _get_color_weak_cmap(cmap, n_images=2)
    color_map, deuter50_cm, prot50_cm, deuter100_cm, trit100_cm = c_maps

    title_str = cmap if isinstance(cmap, str) else cmap.name
    f.suptitle(title_str, fontsize=24)

    ax0 = f.add_subplot(2, 4, 1)
    ax0 = _ax_cylinder_JCh(ax0, cmap, title="Normal (90-95% of pop)")

    ax2 = f.add_subplot(2, 4, 3)
    ax2 = _ax_cylinder_JCh(ax2, deuter50_cm, title="Deuter-50%, RG-weak")

    ax4 = f.add_subplot(2, 4, 7)
    ax4 = _ax_cylinder_JCh(ax4, deuter100_cm, title="Deuter-100%, RG-blind")

    ax6 = f.add_subplot(2, 4, 5)
    ax6 = _ax_cylinder_JCh(ax6, trit100_cm, title="Trit-100%, BY deficient")

    ax3d = f.add_subplot(2, 4, 2, projection="3d", elev=25, azim=-75)
    ax3d = _ax_scatter_Jpapbp(ax3d, cmap, title="Normal (90-95% of pop)")

    ax3d2 = f.add_subplot(2, 4, 4, projection="3d", elev=25, azim=-75)
    ax3d2 = _ax_scatter_Jpapbp(ax3d2, deuter50_cm, title="Deuter-50%, RG-weak")

    ax3d3 = f.add_subplot(2, 4, 8, projection="3d", elev=25, azim=-75)
    ax3d3 = _ax_scatter_Jpapbp(ax3d3, deuter100_cm, title="Deuter-100%, RG-blind")

    ax3d4 = f.add_subplot(2, 4, 6, projection="3d", elev=25, azim=-75)
    ax3d4 = _ax_scatter_Jpapbp(ax3d4, trit100_cm, title="Trit-100%, BY deficient")

    plt.tight_layout()

    return f
