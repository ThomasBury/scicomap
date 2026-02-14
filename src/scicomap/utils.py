import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import dirname, join

# internal import
from scicomap.cblind import _get_color_weak_ctab, _get_color_weak_cmap


def _pyramid(n=513):
    """Create a pyramid function"""
    s = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(s, s)
    z = 1.0 - np.maximum(abs(x), abs(y))
    return x, y, z


def _pyramid_zombie(n=513, stacked=False):
    """Create a pyramid and its zombie copy (buried pyramid)"""
    # image plot
    s_y = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(s_y, s_y)
    z = 1.0 - np.maximum(abs(x), abs(y))
    z = np.hstack((z, z - 1)) if stacked else np.hstack((z, -z))
    # 3d plot
    s_x = np.linspace(-1.0, 3.0, 2 * n)
    s_y = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(s_x, s_y)
    return x, y, z


def _periodic_fn():
    """Create a periodic function with a step function"""
    dx = dy = 0.05
    y, x = np.mgrid[-5 : 5 + dy : dy, -5 : 10 + dx : dx]
    z = (
        np.sin(x) ** 10
        + np.cos(10 + y * x)
        + np.cos(x)
        + 0.2 * y
        + 0.1 * x
        + np.heaviside(x, 1) * np.heaviside(y, 1)
    )
    z = z - np.mean(z)
    return x, y, z


def _fn_with_roots(n=250):
    """Create a function with roots, for diverging colormaps"""
    x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-2, 2, n))
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2) - y**2)


def _complex_phase(n=100):
    """argument of a function in the complex plane, cyclic colormaps"""
    x, y = np.ogrid[-3 : 3 : n * 1j, -3 : 3 : n * 1j]
    z = x + 1j * y
    # w = (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)
    w = (z**2 - 1) * (z - 2 - 1j) ** 2
    return np.angle(w)


def _plot_complex_arg(ax, cmap, facecolor="black", title=False):
    """Create mlp ax of the complex argument"""
    arg_z = _complex_phase(n=100)
    title_color = "white" if facecolor == "black" else "black"
    ang = ax.imshow(np.degrees(arg_z), extent=[-2, 2, -2, 2], cmap=cmap)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    if title:
        ax.set_xlabel("Re", color=title_color, fontsize=12)
        ax.set_ylabel("Im", color=title_color, fontsize=12)
        ax.set_title(
            r"Argument of $(z^2 - 1)(z - 2 - j)^2$",
            color=title_color,
            fontsize=14,
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.yaxis.label.set_color(title_color)
    cax.tick_params(axis="y", colors=title_color)
    plt.colorbar(ang, cax=cax)
    return ax


def _E(q, r0, x, y):
    """Return the electric field vector E=(Ex,Ey) due to charge q at r0."""
    den = np.hypot(x - r0[0], y - r0[1]) ** 3
    return q * (x - r0[0]) / den, q * (y - r0[1]) / den


def _angle_E(n_neg_charges):
    """Create the angle of the electric field with the x-axis"""
    # Grid of x, y points
    nx, ny = 64, 64
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Create a multipole with nq charges of alternating sign, equally spaced
    # on the unit circle.
    nq = 2 ** int(n_neg_charges)
    charges = []
    for i in range(nq):
        q = i % 2 * 2 - 1
        charges.append(
            (q, (np.cos(2 * np.pi * i / nq), np.sin(2 * np.pi * i / nq)))
        )

    # Electric field vector, E=(Ex, Ey), as separate components
    Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
    for charge in charges:
        ex, ey = _E(*charge, x=X, y=Y)
        Ex += ex
        Ey += ey

    # angle wrt to 1_x
    cos_th = Ex / np.hypot(Ex, Ey)
    return np.degrees(np.arccos(cos_th)), charges, x, y, Ex, Ey


def _plot_e_field(ax, cmap, n_neg_charges=2, facecolor="black", title=False):
    """Create the mpl ax of the electric field"""
    title_color = "white" if facecolor == "black" else "black"

    th_deg, charges, x, y, Ex, Ey = _angle_E(n_neg_charges)

    # Plot the streamlines with an appropriate colormap and arrow style
    ax.streamplot(
        x,
        y,
        Ex,
        Ey,
        color="black",
        linewidth=0.5,
        density=0.5,
        arrowstyle="fancy",
        arrowsize=1,
    )

    ang = ax.imshow(th_deg, extent=[-2, 2, -2, 2], cmap=cmap)

    # Add filled circles for the charges themselves
    charge_colors = {True: "#f50000", False: "#0091ff"}
    charge_label = {True: "+", False: "-"}
    label_align = {True: "center", False: "center"}
    for q, pos in charges:
        ax.add_patch(Circle(pos, 0.175, color=charge_colors[q > 0]))
        ax.annotate(
            charge_label[q > 0],
            xy=pos,
            fontsize=30,
            ha="center",
            va=label_align[q > 0],
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    if title:
        ax.set_title(
            r"Angle of $\vec{E}$ with the x-axis",
            color=title_color,
            fontsize=14,
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.yaxis.label.set_color(title_color)
    cax.tick_params(axis="y", colors=title_color)
    plt.colorbar(ang, cax=cax)

    return ax


def _plot_examples(
    color_map,
    images,
    arr_3d,
    figsize,
    facecolor,
    cname,
    cblind=True,
    norm=False,
):
    """Create the figure based on the provided images, continuous colormaps"""
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    n_images = len(images)
    if cblind:
        c_maps, sub_title = _get_color_weak_cmap(color_map, n_images)
    else:
        c_maps = [color_map]
        sub_title = [""] * n_images

    title_color = "white" if facecolor == "black" else "black"

    axi = 1
    idx_3d = 0
    n_rows = len(c_maps)
    n_cols = len(images)

    for c_map, im in itertools.product(c_maps, images):
        if isinstance(im, str) and ("3D" in im):
            px, py, pz = arr_3d[idx_3d]
            ax3d = fig.add_subplot(
                n_rows,
                n_cols,
                axi,
                projection="3d",
                facecolor=facecolor,
                elev=10,
                azim=-45,
            )
            ax3d.plot_surface(
                px, py, pz, cmap=c_map, linewidth=0, antialiased=False
            )
            ax3d = plt.gca()
            ax3d.xaxis.set_ticklabels([])
            ax3d.yaxis.set_ticklabels([])
            ax3d.zaxis.set_ticklabels([])
            idx_3d = 0 if idx_3d == 1 else 1
        elif isinstance(im, str) and ("electric" in im):
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            ax = _plot_e_field(
                ax,
                c_map,
                n_neg_charges=2,
                facecolor=facecolor,
                title=axi <= n_cols,
            )
        elif isinstance(im, str) and ("complex" in im):
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            ax = _plot_complex_arg(ax, c_map, facecolor, title=axi <= n_cols)
        else:
            cmap_norm = mpl.colors.CenteredNorm() if norm else None
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            ax.imshow(im, cmap=c_map, aspect="auto", norm=cmap_norm)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(
                sub_title[axi - 1], color=title_color, fontsize=16, loc="left"
            )

        axi += 1
    fig.suptitle(cname, color=title_color, fontsize=24, y=0.95)
    return fig


def _plot_examples_qual(color_map, dict_arr, figsize, facecolor, cname, year):
    """Create the figure with examples for discrete colormaps"""
    fig = plt.figure(figsize=figsize, facecolor=facecolor)

    c_tabs, sub_title = _get_color_weak_ctab(color_map, len(dict_arr) - 1)

    title_color = "white" if facecolor == "black" else "black"

    axi = 1
    n_rows = len(c_tabs)
    n_cols = len(dict_arr)

    for c_map, d in itertools.product(c_tabs, dict_arr):
        if axi in range(1, n_rows * n_cols, len(dict_arr)):
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            n_colors = len(d.keys())
            ax.stackplot(
                year,
                d.values(),
                labels=d.keys(),
                colors=c_map[range(n_colors), ...],
            )
            ax.legend(loc="upper left")
            ax.set_facecolor(facecolor)
            ax.set_title(
                sub_title[axi - 1], color=title_color, fontsize=16, loc="left"
            )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        elif axi in range(2, n_rows * n_cols, n_cols):
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            N = 45
            x, y = np.random.rand(2, N)
            np.random.seed(19680801)
            s = np.random.randint(10, 220, size=N)
            c = np.random.randint(0, 5, size=N)
            scatter = ax.scatter(
                x, y, cmap=ListedColormap(np.clip(c_map, 0, 1)), s=s, c=c
            )
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(
                *scatter.legend_elements(), loc="lower left", title="Classes"
            )
            ax.add_artist(legend1)
            ax.set_facecolor(facecolor)
            # produce a legend with a cross section of sizes from the scatter
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
            legend2 = ax.legend(
                handles, labels, loc="upper right", title="Sizes"
            )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax = fig.add_subplot(n_rows, n_cols, axi, facecolor=facecolor)
            n_colors = d.shape[1]
            for col in range(n_colors):
                ax.plot(d[..., col], color=c_map[col, ...])
            ax.set_facecolor(facecolor)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        axi += 1
    fig.suptitle(cname, color=title_color, fontsize=24, y=0.95)
    return fig
