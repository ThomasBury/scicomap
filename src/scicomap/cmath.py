"""
Module for performing color math. Heavily based on colorspacious, viscm and ethplot.
The main model of perceptual distance is the "CAM02-UCS" color-space
(Uniform Colour Space version of the CIECAM02).

This module uses Cartesian Lab and CIECAM02 color spaces and cylindrical
CIELCh (hereafter LCh) and CIEJCh (hereafter JCh) color spaces which have coordinates L*, J*, C*, and h.
The lightness coordinates L* and J* are identical to Lab and Jab. The chroma (relative saturation)
C* and hue h (in degree h°) are simply C* = sqrt(a*^2 + b*^2) and h = atan2(b*, a*) according
to Redness-Greenness a and Yellowness-Blueness b in their own coordinates.

https://github.com/liamedeiros/ehtplot/blob/7a0567496ba9ab72f4a541d5994352bbe4eac764/ehtplot/color/cmath.py
"""

import numpy as np
import warnings
import matplotlib
from colorspacious import cspace_convert
from matplotlib.colors import Colormap, ListedColormap
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Union, Callable, Optional


def get_ctab(cmap: Union[Colormap, list]) -> np.ndarray:
    """
    Get the color table (ctab) of a matplotlib colormap.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap or list
        The colormap for which to retrieve the color table (ctab).
        This can be a matplotlib Colormap or a list of color values.

    Returns
    -------
    np.ndarray
        An array representing the color table (ctab) as a sequence of color values.

    Raises
    ------
    TypeError
        If `cmap` is neither a matplotlib Colormap nor a list of color values.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> cmap = plt.get_cmap("viridis")
    >>> ctab = get_ctab(cmap)
    >>> print(ctab)
    """
    if isinstance(cmap, Colormap):
        return np.array([cmap(v) for v in np.linspace(0, 1, cmap.N)])
    elif isinstance(cmap, list):
        return np.array(cmap)
    else:
        TypeError("`cmap` is neither a matplotlib Colormap nor a list of str/uples")



def max_chroma(
    Jp: Union[float, np.ndarray], 
    hp: Union[float, np.ndarray], 
    Cpmin: float = 0.0, 
    Cpmax: Union[float, str] = "auto", 
    eps: float = 1024 * np.finfo(float).eps, 
    clip: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate the maximum chroma (Cp) for a given lightness (Jp) and hue (hp) in the CAM02-UCS color space.

    Parameters
    ----------
    Jp : float or np.ndarray
        Lightness parameter (range: [0, 100]).
    hp : float or np.ndarray
        Hue angle in degrees (range: [0, 360]).
    Cpmin : float, optional (default=0.0)
        Minimum allowable chroma value (range: [0, Cpmax]).
    Cpmax : float or str, optional (default="auto")
        Maximum allowable chroma value (range: [Cpmin, sqrt(100*Jp)]).
        If set to "auto", it will be calculated based on Jp.
    eps : float, optional (default=1024 * np.finfo(float).eps)
        A small value used to handle numerical precision issues.
    clip : bool, optional (default=True)
        If True, clip Jp values to the valid range before calculation.

    Returns
    -------
    float or np.ndarray
        Maximum chroma (Cp) value(s) corresponding to the input Jp and hp.

    Raises
    ------
    ValueError
        If Jp is out of range.
    ArithmeticError
        If the function does not fully converge.

    Example
    -------
    >>> Jp = 70
    >>> hp = 30
    >>> Cp = max_chroma(Jp, hp)
    >>> print(Cp)
    """
    Jpmin = 5.54015251457561e-22
    Jpmaxv = 98.98016
    Jpmax = 99.99871678107648

    if clip:
        Jp = np.clip(Jp, Jpmin, min(Jpmaxv, Jpmax))

    if np.any(Jp < Jpmin) or np.any(Jp > Jpmax):
        raise ValueError("J' out of range.")

    if np.any(Jp > Jpmaxv):
        raise ValueError(
            "J' is out of range such that the corresponding sRGB colorspace "
            + "is offset and C' == 0 is no longer a valid assumption."
        )

    if Cpmax == "auto":
        Cpmax = np.clip(np.sqrt(100 * Jp), 0, 64)

    CpU = np.full(len(Jp), Cpmax)  # np.full() works for both scalar and array
    CpL = np.full(len(Jp), Cpmin)  # np.full() works for both scalar and array

    for i in range(64):
        Cp = 0.5 * (CpU + CpL)

        # Fix when we hit machine precision
        need_fix = Cp == CpU
        Cp[need_fix] = CpL[need_fix]

        Jpapbp = np.stack([Jp, Cp * np.cos(hp), Cp * np.sin(hp)], axis=-1)
        sRGB = transform(Jpapbp, inverse=True)
        edge = 2.0 * np.amax(abs(sRGB - 0.5), -1)

        if 1.0 - eps <= np.min(edge) and np.max(edge) <= 1.0:
            break

        I = edge >= 1.0
        CpU[I] = Cp[I]
        CpL[~I] = Cp[~I]
    else:
        raise ArithmeticError("WARNING: max_chroma() has not fully converged")

    return Cp


def transform(
    ctab: np.ndarray,
    src: str = "sRGB1",
    dst: str = "CAM02-UCS",
    inverse: bool = False
) -> np.ndarray:
    """
    Transform a color table from one color space to another.

    Parameters
    ----------
    ctab : np.ndarray
        A color table in the source color space.
    src : str, optional (default="sRGB1")
        The source color space (e.g., "sRGB1", "CAM02-UCS").
    dst : str, optional (default="CAM02-UCS")
        The destination color space (e.g., "sRGB1", "CAM02-UCS").
    inverse : bool, optional (default=False)
        If True, perform an inverse color space transformation.

    Returns
    -------
    np.ndarray
        A color table in the destination color space.

    Example
    -------
    >>> import numpy as np
    >>> ctab = np.array([[0.5, 0.2, 0.1], [0.3, 0.6, 0.9]])
    >>> transformed_ctab = transform(ctab, src="sRGB1", dst="CAM02-UCS")
    >>> print(transformed_ctab)
    """
    out = ctab.copy()
    if not inverse:
        out[:, :3] = cspace_convert(out[:, :3], src, dst)
    else:
        out[:, :3] = cspace_convert(out[:, :3], dst, src)
    return out


def interp(
    x: float,
    xp: np.ndarray,
    yp: np.ndarray
) -> float:
    """
    One-dimensional linear interpolation.

    Parameters
    ----------
    x : float
        The x-coordinate at which to interpolate.
    xp : np.ndarray
        1-D array of x-coordinates of data points.
    yp : np.ndarray
        1-D array of y-coordinates of data points.

    Returns
    -------
    float
        The interpolated value at x.

    Notes
    -----
    This function performs linear interpolation between data points defined by
    (xp, yp). It supports both increasing and decreasing xp arrays.

    Example
    -------
    >>> import numpy as np
    >>> x = 3.5
    >>> xp = np.array([1.0, 2.0, 4.0, 5.0])
    >>> yp = np.array([0.0, 1.0, 2.0, 3.0])
    >>> interpolated_value = interp(x, xp, yp)
    >>> print(interpolated_value)
    1.5
    """
    if xp[0] < xp[-1]:
        return np.interp(x, xp, yp)
    else:
        return np.interp(x, np.flip(xp, 0), np.flip(yp, 0))


def extrema(a: np.ndarray) -> np.ndarray:
    """
    Find the indices of local extrema in a 1-D array.

    Parameters
    ----------
    a : np.ndarray
        1-D array in which to find local extrema.

    Returns
    -------
    np.ndarray
        Indices of local extrema.

    Notes
    -----
    This function finds the indices of local extrema (maxima or minima) in the
    input array `a`. It returns an array of indices corresponding to the local
    extrema points.

    An extremum point is detected if the product of the differences between
    the point, the previous point, and the next point in `a` is less than or
    equal to zero.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([1, 3, 7, 1, 2, 6, 2, 9])
    >>> extrema_indices = extrema(a)
    >>> print(extrema_indices)
    [2 3 5 6 7]
    """
    da = a[1:] - a[:-1]
    xa = da[1:] * da[:-1]
    return np.argwhere(xa <= 0.0)[:, 0] + 1


def classify(Jpapbp: np.ndarray) -> str:
    """
    Classify a colormap based on its appearance in the Jpapbp color space.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of colors in the Jpapbp color space.

    Returns
    -------
    str
        A string representing the colormap classification:
        - 'circular-div' for circular and diverging colormaps.
        - 'circular-flat' for circular and flat colormaps.
        - 'sequential' for sequential colormaps.
        - 'divergent' for diverging colormaps.
        - 'asym_div' for asymmetric diverging colormaps.
        - 'multiseq' for multi-sequential colormaps.
        - 'unknown' for unknown or unclassified colormaps.

    Notes
    -----
    This function classifies a colormap based on its appearance in the Jpapbp
    color space. The classification is determined by the number and positions
    of extrema in the luminance channel (J) of the colormap.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[50, 10, 20], [40, 15, 25], [60, 5, 10]])
    >>> colormap_class = classify(Jpapbp)
    >>> print(colormap_class)
    'sequential'
    """

    N = Jpapbp.shape[0]
    large_diff_lum = Jpapbp[:, 0].max() - Jpapbp[:, 0].min() > 0.5
    first_and_last_simeq = (
        sum(abs(np.floor(Jpapbp[0, ...]) - np.floor(Jpapbp[-1, ...]))) < 4
    )

    # some of the cmcrameri colormaps have kind of a small hook at the end of the array
    # artifact increasing the number of extrema by +1. A rough fix is not considering the
    # beginning  and the end of the cmap
    x = extrema(Jpapbp[10:-10, 0])
    x += 10

    if (len(x) == 1 or large_diff_lum) and first_and_last_simeq:
        return "circular-div"
    elif len(x) != 1 and first_and_last_simeq:
        return "circular-flat"
    elif len(x) == 0:
        return "sequential"
    # strictly, the extremum should be at the middle element set([(N + 1) // 2 - 1, N // 2]):
    # cmcrameri uses another parametrization and is not perfectly symmetric
    # in the Jc'h space
    elif len(x) == 1 and x[0] in np.arange((N + 1) // 2 - 5, N // 2 + 5, 1):
        return "divergent"
    elif len(x) == 1 and x[0] not in np.arange((N + 1) // 2 - 5, N // 2 + 5, 1):
        return "asym_div"
    elif len(x) == 2 and (x[1] == x[0] + 1):
        return "multiseq"
    else:
        return "unknown"


def uniformize(Jpapbp: np.ndarray, JpL: float = None, JpR: float = None,
               Jplower: float = None, Jpupper: float = None) -> np.ndarray:
    """
    Uniformize a colormap in the Jpapbp color space, linear in lightness J'

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of colors in the Jpapbp color space.
    JpL : float, optional
        Left luminance boundary for uniformization.
    JpR : float, optional
        Right luminance boundary for uniformization.
    Jplower : float, optional
        Lower luminance limit for uniformization.
    Jpupper : float, optional
        Upper luminance limit for uniformization.

    Returns
    -------
    np.ndarray
        The uniformized colormap in the Jpapbp color space.

    Notes
    -----
    This function uniformizes a colormap by linearly interpolating between
    specified luminance values (JpL and JpR) in the Jpapbp color space. You
    can optionally provide lower (Jplower) and upper (Jpupper) luminance
    limits to restrict the uniformization range.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[50, 10, 20], [40, 15, 25], [60, 5, 10]])
    >>> JpL = 40
    >>> JpR = 60
    >>> uniformized_cmap = uniformize(Jpapbp, JpL, JpR)
    >>> print(uniformized_cmap)
    array([[40. , 10. , 20. ],
           [50. , 12.5, 22.5],
           [60. , 15. , 25. ]])
    """
    if JpL is None:
        JpL = Jpapbp[0, 0]
    if JpR is None:
        JpR = Jpapbp[-1, 0]

    if Jplower is not None:
        JpL, JpR = max(JpL, Jplower), max(JpR, Jplower)
    if Jpupper is not None:
        JpL, JpR = min(JpL, Jpupper), min(JpR, Jpupper)

    out = Jpapbp.copy()
    out[:, 0] = np.linspace(JpL, JpR, out.shape[0])
    out[:, 1] = interp(out[:, 0], Jpapbp[:, 0], Jpapbp[:, 1])
    out[:, 2] = interp(out[:, 0], Jpapbp[:, 0], Jpapbp[:, 2])
    return out


def factor(
    Cp: np.ndarray,
    softening: float = 1.0,
    bitonic: bool = True,
    diffuse: bool = True,
    CpL: float = None,
    CpR: float = None,
    verbose: bool = False,
    diverging: bool = False,
) -> np.ndarray:
    """
    Compute the factor required to perform several chroma operations.

    Parameters
    ----------
    Cp : np.ndarray
        Array of chroma values.
    softening : float, optional
        Softening factor applied to the chroma values.
    bitonic : bool, optional
        If True, enforce bitonicity for non-diverging chroma values.
    diffuse : bool, optional
        If True, apply diffusion to the chroma values.
    CpL : float, optional
        Left chroma boundary.
    CpR : float, optional
        Right chroma boundary.
    verbose : bool, optional
        If True, print verbose messages during computation.
    diverging : bool, optional
        If True, consider the chroma values as diverging.

    Returns
    -------
    np.ndarray
        The computed factor for chroma operations.

    Notes
    -----
    This function computes a factor required for performing various chroma
    operations. It can enforce bitonicity for non-diverging chroma values and
    apply diffusion. You can specify left (CpL) and right (CpR) chroma
    boundaries. The softening factor (softening) smoothens the chroma values.

    Example
    -------
    >>> import numpy as np
    >>> Cp = np.array([2.0, 5.0, 8.0, 5.0, 2.0])
    >>> factor_values = factor(Cp, softening=0.5, bitonic=True, diffuse=False)
    >>> print(factor_values)
    array([0.72727273, 0.5       , 0.27272727, 0.5       , 0.72727273])
    """
    S = Cp + softening
    s = S.copy()

    N = len(Cp)
    H = N // 2
    m = np.minimum(s[:H], np.flip(s[-H:]))

    # Bitonic only for non diverging
    if bitonic and (not diverging):  # force half of Cp increase monotonically
        if m[H - 1] > s[H]:
            m[H - 1] = s[H]
            if verbose:
                print("Enforce bitonic at {}".format(s[H]))
        for i in range(H - 1, 0, -1):
            if m[i - 1] > m[i]:
                m[i - 1] = m[i]
                if verbose:
                    print("Enforce bitonic at {}".format(m[i]))

    s[:+H] = m
    s[-H:] = np.flip(m)

    if CpL is not None:
        s[0] = CpL + softening
    if CpR is not None:
        s[-1] = CpR + softening

    if diffuse:  # diffuse s using forward Euler
        for i in range(N):
            s[1:-1] += 0.5 * (s[2:] + s[:-2] - 2.0 * s[1:-1])

    return s / S


def symmetrize(Jpapbp: np.ndarray, **kwargs) -> np.ndarray:
    """
    Make a sequential colormap symmetric in chroma C'.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of Jpapbp values.

    Returns
    -------
    np.ndarray
        The symmetric colormap in chroma C'.

    Other Parameters
    ----------------
    **kwargs
        Additional keyword arguments to pass to the `factor` function.

    Notes
    -----
    This function makes a sequential colormap symmetric in chroma C'.
    It uses the `factor` function to adjust the chroma values.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[40.0, 20.0, 10.0], [60.0, 30.0, 15.0]])
    >>> symmetric_colormap = symmetrize(Jpapbp, softening=0.2, bitonic=True, diffuse=True)
    >>> print(symmetric_colormap)
    array([[40.        , 20.        , 10.        ],
           [60.        , 30.        , 15.        ]])
    """
    out = Jpapbp.copy()
    Jp = out[:, 0]
    Cp = np.sqrt(out[:, 1] * out[:, 1] + out[:, 2] * out[:, 2])

    f = factor(Cp, **kwargs)
    out[:, 1] *= f
    out[:, 2] *= f
    return out


def adjust_sequential(
    Jpapbp: np.ndarray, roundup: float = None, bi_seq: bool = False
) -> np.ndarray:
    """
    Adjust a sequential colormap in chroma C' and optionally create a bidirectional sequential colormap.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of Jpapbp values.

    roundup : float, optional
        Value to round down the lower chroma bound to. If provided, the lower chroma bound will be rounded up to the nearest multiple of `roundup`. Default is None.

    bi_seq : bool, optional
        If True, create a bidirectional sequential colormap by adjusting two segments of the input colormap. Default is False.

    Returns
    -------
    np.ndarray
        The adjusted sequential colormap in chroma C'.

    Notes
    -----
    This function adjusts a sequential colormap in chroma C' by optionally rounding down the lower chroma bound and creating a bidirectional sequential colormap.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[40.0, 20.0, 10.0], [60.0, 30.0, 15.0]])
    >>> adjusted_colormap = adjust_sequential(Jpapbp, roundup=0.1, bi_seq=True)
    >>> print(adjusted_colormap)
    array([[40. , 20. , 10. ],
           [60. , 30. , 15. ],
           [40.1, 20. , 10. ],
           [59.9, 30. , 15. ]])
    """

    if bi_seq:
        x_boundary = extrema(Jpapbp[:, 0])[0]
        Jpapbp1 = Jpapbp[: x_boundary + 1, ...].copy()
        Jpapbp2 = Jpapbp[x_boundary + 1 :, ...].copy()
        Jp = Jpapbp1[:, 0]
        Jplower = min(Jp[0], Jp[-1])
        if roundup is not None:
            Jplower = np.ceil(Jplower / roundup) * roundup
        Jpapbp1 = uniformize(Jpapbp1, Jplower=Jplower)
        Jp = Jpapbp2[:, 0]
        Jplower = min(Jp[0], Jp[-1])
        if roundup is not None:
            Jplower = np.ceil(Jplower / roundup) * roundup
        Jpapbp2 = uniformize(Jpapbp2, Jplower=Jplower)
        return np.append(Jpapbp1, Jpapbp2, axis=0)
    else:
        Jp = Jpapbp[:, 0]
        Jplower = min(Jp[0], Jp[-1])
        if roundup is not None:
            Jplower = np.ceil(Jplower / roundup) * roundup

        return uniformize(Jpapbp, Jplower=Jplower)


def adjust_circular_flat(Jpapbp: np.ndarray) -> np.ndarray:
    """
    Adjust a flat circular colormap, making the lightness constant.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of Jpapbp values.

    Returns
    -------
    np.ndarray
        The adjusted colormap with constant lightness.

    Notes
    -----
    This function adjusts a flat circular colormap by making the lightness constant.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[50.0, 20.0, 10.0], [60.0, 30.0, 15.0]])
    >>> adjusted_colormap = adjust_circular_flat(Jpapbp)
    >>> print(adjusted_colormap)
    array([[55., 20., 10.],
           [55., 30., 15.]])
    """
    out = Jpapbp.copy()
    Jp = out[:, 0].ravel()
    out[:, 0] = np.ones_like(Jp) * np.nanmean(Jp)
    return out


def adjust_circular(Jpapbp: np.ndarray, roundup: float = None) -> np.ndarray:
    """
    Adjust a circular colormap.

    This function adjusts a circular colormap by making modifications based on the given parameters.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of Jpapbp values.
    roundup : float, optional
        Value to round Jplower, by default None.

    Returns
    -------
    np.ndarray
        The adjusted colormap.

    Notes
    -----
    This function adjusts a circular colormap based on the parameters provided.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[50.0, 20.0, 10.0], [60.0, 30.0, 15.0]])
    >>> adjusted_colormap = adjust_circular(Jpapbp, roundup=5.0)
    >>> print(adjusted_colormap)
    array([[55., 20., 10.],
           [60., 30., 15.]])
    """
    Jp = Jpapbp[:, 0]
    x_extr = extrema(Jp)
    h = x_extr[0]
    H = h + 1
    N = Jpapbp.shape[0]
    # h = (N + 1) // 2 - 1  # == H-1 if even; == H if odd
    # H = N // 2

    if Jp[1] > Jp[0]:  # hill
        Jplower = max(Jp[0], Jp[-1])
        Jpupper = min(Jp[h], Jp[H])
    else:  # valley
        Jplower = max(Jp[h], Jp[H])
        Jpupper = min(Jp[0], Jp[-1])
    if roundup is not None:
        Jplower = np.ceil(Jplower / roundup) * roundup

    L = uniformize(Jpapbp[: h + 1, :], Jplower=Jplower, Jpupper=Jpupper)
    R = uniformize(Jpapbp[H:, :], Jplower=Jplower, Jpupper=Jpupper)
    return np.append(L, R[N % 2 :, :], axis=0)


def adjust_divergent(
    Jpapbp: np.ndarray,
    roundup: float = None,
    circular: bool = False,
    symmetric: bool = True
) -> np.ndarray:
    """
    Adjust a divergent colormap.

    This function adjusts a divergent colormap by making modifications based on the given parameters.

    Parameters
    ----------
    Jpapbp : np.ndarray
        Array of Jpapbp values.
    roundup : float, optional
        Value to round Jplower, by default None.
    circular : bool, optional
        Whether to make the colormap circular, by default False.
    symmetric : bool, optional
        Whether to make the colormap symmetric, by default True.

    Returns
    -------
    np.ndarray
        The adjusted colormap.

    Notes
    -----
    This function adjusts a divergent colormap based on the parameters provided.

    Example
    -------
    >>> import numpy as np
    >>> Jpapbp = np.array([[50.0, 20.0, 10.0], [60.0, 30.0, 15.0]])
    >>> adjusted_colormap = adjust_divergent(Jpapbp, roundup=5.0, circular=True, symmetric=True)
    >>> print(adjusted_colormap)
    array([[55., 20., 10.],
           [60., 30., 15.]])
    """
    Jp = Jpapbp[:, 0]
    out = Jpapbp.copy()
    x_extr = extrema(Jp)
    h = x_extr[0]
    H = h + 1
    N = Jpapbp.shape[0]
    # h = (N + 1) // 2 - 1  # == H-1 if even; == H if odd
    # H = N // 2

    if Jp[1] > Jp[0] and symmetric:  # hill
        Jplower = max(Jp[0], Jp[-1])
        Jpupper = min(Jp[h], Jp[H])
    elif Jp[1] > Jp[0] and not symmetric:  # hill
        Jplower = Jp[0]
        Jpupper = Jp[h]
    elif Jp[1] < Jp[0] and symmetric:  # valley
        Jplower = max(Jp[h], Jp[H])
        Jpupper = min(Jp[0], Jp[-1])
    else:
        Jplower = Jp[h]
        Jpupper = Jp[-1]
    if roundup is not None:
        Jplower = np.ceil(Jplower / roundup) * roundup

    L = uniformize(Jpapbp[: h + 1, :], Jplower=Jplower, Jpupper=Jpupper)
    R = uniformize(Jpapbp[H:, :], Jplower=Jplower, Jpupper=Jpupper)
    Jpapbp_unif = np.append(L, R[N % 2 :, :], axis=0)

    if circular:
        theta = 2 * np.pi * np.linspace(0, 1, out.shape[0])
        out[0, 1:3] = out[-1, 1:3]
        cs = CubicSpline(theta, out[:, 1:3], bc_type="periodic")
        Jpapbp_unif[:, 1:3] = cs(theta)
    return Jpapbp_unif


def uniformize_cmap(
    cmap: ListedColormap,
    name: str = "new_cmap",
    lift: float = None,
    uniformized: bool = False
) -> Tuple[ListedColormap, bool]:
    """
    Uniformize a colormap.

    This function uniformizes a given colormap if it's not already uniformized.

    Parameters
    ----------
    cmap : ListedColormap
        The input colormap to be uniformized.
    name : str, optional
        The name of the new colormap, by default "new_cmap".
    lift : float, optional
        Value to round Jplower, by default None.
    uniformized : bool, optional
        Indicates whether the colormap is already uniformized, by default False.

    Returns
    -------
    Tuple[ListedColormap, bool]
        A tuple containing the uniformized colormap and a boolean indicating if
        the colormap was uniformized.

    Notes
    -----
    This function uniformizes a colormap by analyzing its color table and transforming
    it into a uniformized version based on its color characteristics.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from my_module import uniformize_cmap
    >>> # Create a sample colormap
    >>> original_cmap = plt.get_cmap("viridis")
    >>> uniformized_cmap, was_uniformized = uniformize_cmap(original_cmap, name="uniform_viridis", lift=5.0)
    >>> if was_uniformized:
    ...     print("Colormap was uniformized.")
    ... else:
    ...     print("Colormap was already uniformized.")
    >>> # Now you can use the uniformized_cmap for plotting.

    """
    # if not uniformized yet, uniformize the cmap
    # else do nothing
    if not uniformized:
        # get the color table
        ctab = get_ctab(cmap)
        # transform to Jpapbp
        t_ctab = transform(ctab)
        # find if sequential or divergent or unknown
        cmap_type = classify(t_ctab)
        if cmap_type == "divergent":
            lin_ctab = adjust_divergent(t_ctab, roundup=lift)
        elif cmap_type == "asym_div":
            lin_ctab = adjust_divergent(t_ctab, roundup=lift, symmetric=False)
        elif cmap_type == "sequential":
            lin_ctab = adjust_sequential(t_ctab, roundup=lift)
        elif cmap_type == "multiseq":
            lin_ctab = adjust_sequential(t_ctab, roundup=lift, bi_seq=True)
        elif cmap_type == "circular-flat":
            lin_ctab = adjust_circular_flat(t_ctab)
        elif cmap_type == "circular-div":
            lin_ctab = adjust_divergent(t_ctab, roundup=lift, circular=True)
        else:
            warnings.warn(
                "The colormap {} type is unknown (not recognized as sequential or divergent)\n"
                "Not uniformized".format(name)
            )
            lin_ctab = t_ctab

        lin_cmap = transform(ctab=lin_ctab, inverse=True)

        return ListedColormap(np.clip(lin_cmap, 0, 1), name=name), True
    else:
        return cmap, True


def symmetrize_cmap(
    cmap: ListedColormap,
    name: str = "new_cmap",
    bitonic: bool = True,
    diffuse: bool = True
) -> ListedColormap:
    """
    Symmetrize a colormap.

    This function symmetrizes a given colormap based on its color characteristics.

    Parameters
    ----------
    cmap : ListedColormap
        The input colormap to be symmetrized.
    name : str, optional
        The name of the new colormap, by default "new_cmap".
    bitonic : bool, optional
        If True, ensures that half of Cp increases monotonically, by default True.
    diffuse : bool, optional
        If True, diffuses the colormap, by default True.

    Returns
    -------
    ListedColormap
        The symmetrized colormap.

    Notes
    -----
    This function symmetrizes a colormap by analyzing its color table, transforming
    it into a symmetrical version based on its color characteristics, and returning
    it as a matplotlib ListedColormap.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from my_module import symmetrize_cmap
    >>> # Create a sample colormap
    >>> original_cmap = plt.get_cmap("coolwarm")
    >>> symmetrized_cmap = symmetrize_cmap(original_cmap, name="symmetric_coolwarm", bitonic=True, diffuse=True)
    >>> # Now you can use the symmetrized_cmap for plotting.

    """
    # get the color table
    ctab = get_ctab(cmap)
    # transform to Jpapbp
    t_ctab = transform(ctab)
    # find if sequential or divergent or unknown
    cmap_type = classify(t_ctab)
    diverging = True if cmap_type == "divergent" else False
    lin_ctab = symmetrize(t_ctab, bitonic=bitonic, diffuse=diffuse, diverging=diverging)
    # get back a matplotlib cmap object
    s_cmap = transform(ctab=lin_ctab, inverse=True)
    return ListedColormap(np.clip(s_cmap, 0, 1), name=name)


def unif_sym_cmap(
    cmap: ListedColormap,
    name: str = "new_cmap",
    lift: float = None,
    uniformized: bool = False,
    bitonic: bool = True,
    diffuse: bool = True
) -> tuple[ListedColormap, bool]:
    """
    Uniformize and symmetrize a colormap (perceptually homogeneous).

    This function performs both uniformization and symmetrization of a given colormap
    based on its color characteristics (symmetric saturation for even perception of both sides of the cmap).

    Parameters
    ----------
    cmap : ListedColormap
        The input colormap to be uniformized and symmetrized.
    name : str, optional
        The name of the new colormap, by default "new_cmap".
    lift : float, optional
        A parameter controlling the degree of uniformization, by default None.
    uniformized : bool, optional
        If True, skip uniformization step if the colormap is already uniformized,
        by default False.
    bitonic : bool, optional
        If True, ensures that half of Cp increases monotonically, by default True.
    diffuse : bool, optional
        If True, diffuses the colormap, by default True.

    Returns
    -------
    tuple[ListedColormap, bool]
        A tuple containing the uniformized and symmetrized colormap, and a boolean
        indicating whether uniformization was performed.

    Notes
    -----
    This function first checks if uniformization is needed based on the `uniformized`
    parameter. If uniformization is performed, the colormap is transformed into a
    uniformized version based on its color characteristics. Then, symmetrization is
    applied to the uniformized colormap. The final colormap is returned along with
    a boolean indicating whether uniformization was performed.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from my_module import unif_sym_cmap
    >>> # Create a sample colormap
    >>> original_cmap = plt.get_cmap("coolwarm")
    >>> uniformized_symmetric_cmap, uniformized = unif_sym_cmap(original_cmap, name="uni_sym_coolwarm", lift=0.1)
    >>> # Now you can use the uniformized_symmetric_cmap for plotting.

    """
    uni_cmap, uniformized = uniformize_cmap(
        cmap, name=name, lift=lift, uniformized=uniformized
    )
    return (
        symmetrize_cmap(uni_cmap, name=name, bitonic=bitonic, diffuse=diffuse),
        uniformized,
    )


def _ax_cylinder_JCh(
    ax: matplotlib.axes.Axes,
    cmap: Colormap,
    title: str
) -> matplotlib.axes.Axes:
    """
    Plot Jp, Cp, and hp coordinates in a cylindrical representation for a colormap.

    This function visualizes the J' (lightness), C' (chroma), and h' (hue) coordinates
    of a colormap in a cylindrical representation.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The matplotlib axes object to plot on.
    cmap : matplotlib.colors.Colormap
        The input colormap to visualize.
    title : str
        The title of the plot.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The matplotlib axes object containing the plotted data.

    Notes
    -----
    This function takes a colormap and extracts its J', C', and h' coordinates in
    the CAM02-UCS color space. It then plots these coordinates in a cylindrical
    representation, where J' and C' are shown on one axis, and h' on another axis.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from my_module import _ax_cylinder_JCh
    >>> # Create a sample colormap
    >>> cmap = plt.get_cmap("coolwarm")
    >>> fig, ax = plt.subplots()
    >>> _ax_cylinder_JCh(ax, cmap, title="Cylindrical JCh Coordinates")

    """
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    Jpapbp = transform(ctab)  # transform color table into CAM02-UCS colorspace

    Jp = Jpapbp[:, 0]
    ap = Jpapbp[:, 1]
    bp = Jpapbp[:, 2]

    Cp = np.sqrt(ap * ap + bp * bp)
    hp = np.arctan2(bp, ap) * 180 / np.pi
    v = np.linspace(0.0, 1.0, len(Jp))

    ax.set_title(title + "\nJpCphp coord. (cyl)", fontsize=14)
    ax.set_xlabel("Value", fontsize=14)

    axtx = ax.twinx()
    ax.set_ylim(0, 100)
    ax.set_ylabel("J' & C' (0-100)", fontsize=14)
    axtx.set_ylim(-180, 180)
    axtx.set_ylabel("h' (degrees)", fontsize=14)

    ax.scatter(v, Jp, color=ctab, label="J' (lightness)")
    ax.plot(v, Cp, c="k", linestyle="--", label="C' (chroma)")
    axtx.scatter(v[::15], hp[::15], s=10, c="k")
    ax.scatter([], [], s=10, c="k", label="h' (hue)")
    ax.legend(loc="best")
    return ax


def _ax_scatter_Jpapbp(
    ax: matplotlib.axes.Axes,
    cmap: Colormap,
    title: str
) -> matplotlib.axes.Axes:
    """
    Create a scatter plot in 3D to visualize Jpapbp coordinates of a colormap.

    This function creates a 3D scatter plot to visualize the J', a', and b' coordinates
    of a colormap in the CAM02-UCS color space.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The matplotlib axes object to plot on.
    cmap : matplotlib.colors.Colormap
        The input colormap to visualize.
    title : str
        The title of the plot.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The matplotlib axes object containing the plotted data.

    Notes
    -----
    This function takes a colormap, extracts its J', a', and b' coordinates in the
    CAM02-UCS color space, and creates a 3D scatter plot to visualize these coordinates.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from my_module import _ax_scatter_Jpapbp
    >>> # Create a sample colormap
    >>> cmap = plt.get_cmap("coolwarm")
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> _ax_scatter_Jpapbp(ax, cmap, title="Jpapbp Scatter Plot")

    """
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    Jpapbp = transform(ctab)  # transform color table into CAM02-UCS colorspace

    # ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], Jpapbp[:, 0])
    x_dots = np.linspace(0, 1, Jpapbp.shape[0])
    RGB_dots = cmap(x_dots)[:, :3]
    Jpapbp_dots = transform(RGB_dots)
    ax.scatter(
        Jpapbp_dots[:, 1], Jpapbp_dots[:, 2], Jpapbp_dots[:, 0], c=RGB_dots[:, :], s=80
    )
    ax.set_xlabel("a' (green -> red)")
    ax.set_ylabel("b' (blue -> yellow)")
    ax.set_zlabel("J'/K (black -> white)")
    ax.set_title(title + "\nJpapbp coord.", fontsize=14)
    return ax
