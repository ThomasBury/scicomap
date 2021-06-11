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
from colorspacious import cspace_convert
from matplotlib.colors import Colormap, ListedColormap
from scipy.interpolate import CubicSpline


def get_ctab(cmap):
    """
    Get the tabular version of the colormap, with the same number of rows than colors
    :param cmap: matplotlib colormap
        the matplotlib colormap object to convert to a np.array
    :return: np.array
        the numpy array of the the colors
    """
    if isinstance(cmap, Colormap):
        return np.array([cmap(v) for v in np.linspace(0, 1, cmap.N)])
    elif isinstance(cmap, list):
        return np.array(cmap)
    else:
        TypeError("`cmap` is neither a matplotlib Colormap nor a list of str/uples")


def max_chroma(
        Jp, hp, Cpmin=0.0, Cpmax="auto", eps=1024 * np.finfo(np.float).eps, clip=True
):
    """
    Compute the maximum allowed chroma given lightness J' and hue h', from the CAM02-UCS
    color space.

    :param Jp: np.array
        The lightness (in the JCh space)
    :param hp: np.array
        The hue
    :param Cpmin: float, default=0.0
        The minimal chroma value
    :param Cpmax: float or 'auto'
        The maximal chroma value
    :param eps: float
        the precision
    :param clip: Boolean, default=True
        To clip or not the minimal and maximal values of the lightness
    :return:
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


def transform(ctab, src="sRGB1", dst="CAM02-UCS", inverse=False):
    """Transform a colortable between color spaces"""

    out = ctab.copy()
    if not inverse:
        out[:, :3] = cspace_convert(out[:, :3], src, dst)
    else:
        out[:, :3] = cspace_convert(out[:, :3], dst, src)
    return out


def interp(x, xp, yp):
    """Improve numpy's interp() function to allow decreasing `xp`"""
    if xp[0] < xp[-1]:
        return np.interp(x, xp, yp)
    else:
        return np.interp(x, np.flip(xp, 0), np.flip(yp, 0))


def extrema(a):
    """Find extrema in an array"""
    da = a[1:] - a[:-1]
    xa = da[1:] * da[:-1]
    return np.argwhere(xa <= 0.0)[:, 0] + 1


def classify(Jpapbp):
    """
    Classify a colormap as sequential, divergent, circular, multi-sequential
    """

    N = Jpapbp.shape[0]
    large_diff_lum = Jpapbp[:, 0].max() - Jpapbp[:, 0].min() > 0.5
    first_and_last_simeq = sum(abs(np.floor(Jpapbp[0, ...]) - np.floor(Jpapbp[-1, ...]))) < 4

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


def uniformize(Jpapbp, JpL=None, JpR=None, Jplower=None, Jpupper=None):
    """Make a colormap uniform in lightness J' (linearizing)"""
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
        Cp, softening=1.0, bitonic=True, diffuse=True, CpL=None, CpR=None, verbose=False, diverging=False
):
    """Compute the factor required to perform several chroma operations"""
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


def symmetrize(Jpapbp, **kwargs):
    """Make a sequential colormap symmetric in chroma C'"""
    out = Jpapbp.copy()
    Jp = out[:, 0]
    Cp = np.sqrt(out[:, 1] * out[:, 1] + out[:, 2] * out[:, 2])

    f = factor(Cp, **kwargs)
    out[:, 1] *= f
    out[:, 2] *= f
    return out


def adjust_sequential(Jpapbp, roundup=None, bi_seq=False):
    """API for uniformizing a sequential colormap"""

    if bi_seq:
        x_boundary = extrema(Jpapbp[:, 0])[0]
        Jpapbp1 = Jpapbp[:x_boundary + 1, ...].copy()
        Jpapbp2 = Jpapbp[x_boundary + 1:, ...].copy()
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


def adjust_circular_flat(Jpapbp):
    """Adjusting the flat circular colormap, making the lightness constant"""
    out = Jpapbp.copy()
    Jp = out[:, 0].ravel()
    out[:, 0] = np.ones_like(Jp) * np.nanmean(Jp)
    return out


def adjust_circular(Jpapbp, roundup=None):
    """API for uniformizing a circular colormap (non flat in lightness)"""
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
    return np.append(L, R[N % 2:, :], axis=0)


def adjust_divergent(Jpapbp, roundup=None, circular=False, symmetric=True):
    """API for uniformizing a divergent colormap"""
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
    Jpapbp_unif = np.append(L, R[N % 2:, :], axis=0)

    if circular:
        theta = 2 * np.pi * np.linspace(0, 1, out.shape[0])
        out[0, 1:3] = out[-1, 1:3]
        cs = CubicSpline(theta, out[:, 1:3], bc_type='periodic')
        Jpapbp_unif[:, 1:3] = cs(theta)
    return Jpapbp_unif


def uniformize_cmap(cmap, name="new_cmap", lift=None, uniformized=False):
    """
    Uniformize the color map (perceptually homogeneous)

    :param cmap: matplotlib cmap
        the cmap object to linearize
    :param name: str, default="new_cmap"
        the name of the new (created) cmap
    :param lift, None or int (0,100)
        whether or not increase the lightness (Jab parametrization)
    :param uniformized, Boolean
        if the cmap has been uniformized already, avoid redundant lift
    :return: cmap object
        a matplotlib colormap
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
            warnings.warn("The colormap {} type is unknown (not recognized as sequential or divergent)\n"
                          "Not uniformized".format(name))
            lin_ctab = t_ctab

        lin_cmap = transform(ctab=lin_ctab, inverse=True)

        return ListedColormap(np.clip(lin_cmap, 0, 1), name=name), True
    else:
        return cmap, True


def symmetrize_cmap(cmap, name="new_cmap", bitonic=True, diffuse=True):
    """
    Symmetrize the color map (symmetric saturation for even perception of both sides of the cmap)

    :param cmap: matplotlib cmap
        the cmap object to linearize
    :param name: str, default="new_cmap"
        the name of the new (created) cmap
    :param bitonic: Boolean, default=True
        if True, the chroma will be forced to increase monotonically,
        reach max or plateau and then decrease
        (bitonic, not suited for other than sequential cmaps).
    :param diffuse: Boolean, default=True
        diffuse colors, avoid sharp edges in Chroma (saturation)
    :return: cmap object
        a matplotlib colormap
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


def unif_sym_cmap(cmap, name="new_cmap", lift=None, uniformized=False, bitonic=True, diffuse=True):
    """
    Uniformize the color map (perceptually homogeneous) and Symmetrize the color map
    (symmetric saturation for even perception of both sides of the cmap)

    :param cmap: matplotlib cmap
        the cmap object to linearize
    :param name: str, default="new_cmap"
        the name of the new (created) cmap
    :param lift, None or int (0,100)
        whether or not increase the lightness (Jab parametrization)
    :param uniformized, Boolean
        if the cmap has been uniformized already, avoid redundant lift
    :param bitonic: Boolean, default=True
        if True, the chroma will be forced to increase monotonically,
        reach max or plateau and then decrease
        (bitonic, not suited for other than sequential cmaps).
    :param diffuse: Boolean, default=True
        diffuse colors, avoid sharp edges in Chroma (saturation)
    :return: cmap object
        a matplotlib colormap
    """
    uni_cmap, uniformized = uniformize_cmap(cmap, name=name, lift=lift, uniformized=uniformized)
    return symmetrize_cmap(uni_cmap, name=name, bitonic=bitonic, diffuse=diffuse), uniformized


def _ax_cylinder_JCh(ax, cmap, title):
    """Create the matplotlib ax for the JCh"""
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    Jpapbp = transform(ctab)  # transform color table into CAM02-UCS colorspace

    Jp = Jpapbp[:, 0]
    ap = Jpapbp[:, 1]
    bp = Jpapbp[:, 2]

    Cp = np.sqrt(ap * ap + bp * bp)
    hp = np.arctan2(bp, ap) * 180 / np.pi
    v = np.linspace(0.0, 1.0, len(Jp))

    ax.set_title(title + " - JpCphp coord. (cyl)", fontsize=14)
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


def _ax_scatter_Jpapbp(ax, cmap, title):
    """Create the matplotlib ax for the scatter plot in 3D"""
    ctab = get_ctab(cmap)  # get the colormap as a color table in sRGB
    Jpapbp = transform(ctab)  # transform color table into CAM02-UCS colorspace

    # ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], Jpapbp[:, 0])
    x_dots = np.linspace(0, 1, Jpapbp.shape[0])
    RGB_dots = cmap(x_dots)[:, :3]
    Jpapbp_dots = transform(RGB_dots)
    ax.scatter(Jpapbp_dots[:, 1],
               Jpapbp_dots[:, 2],
               Jpapbp_dots[:, 0],
               c=RGB_dots[:, :],
               s=80)
    ax.set_xlabel("a' (green -> red)")
    ax.set_ylabel("b' (blue -> yellow)")
    ax.set_zlabel("J'/K (black -> white)")
    ax.set_title(title + " - Jpapbp coord.", fontsize=14)
    return ax
