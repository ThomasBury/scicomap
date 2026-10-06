"""Catalog and ordinary plotting regressions for the v1 numerical fixes."""

import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt

from scicomap.cblind import _get_color_weak_cmap, colorblind_vision
from scicomap.cmath import get_ctab, unif_sym_cmap
from scicomap.scicomap import (
    SciCoMap,
    ScicoCircular,
    ScicoDiverging,
    ScicoMiscellaneous,
    ScicoMultiSequential,
    ScicoSequential,
    compare_cmap,
    get_cmap_dict,
    jch_plot,
    plot_colormap,
)
from scicomap.utils import _periodic_fn, _pyramid


def test_every_catalog_map_preserves_samples_and_reversal_pairs() -> None:
    for maps in get_cmap_dict().values():
        for name, cmap in maps.items():
            table = get_ctab(cmap)
            assert np.isfinite(table).all(), name
            if name.endswith("_r") and name[:-2] in maps:
                np.testing.assert_allclose(
                    table,
                    get_ctab(maps[name[:-2]])[::-1],
                    atol=1e-12,
                    err_msg=name,
                )
            # Unknown lightness patterns intentionally warn and skip uniformization.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The colormap .* type is unknown"
                )
                corrected = unif_sym_cmap(cmap)
            result = get_ctab(corrected)
            assert result.shape == table.shape, name
            assert np.isfinite(result).all(), name
            np.testing.assert_array_equal(
                result[:, 3], table[:, 3], err_msg=name
            )


def test_family_discovery_agrees_with_catalog() -> None:
    for family, maps in get_cmap_dict().items():
        assert SciCoMap(ctype=family).ctype == family
        assert maps


@pytest.mark.parametrize(
    "family, name", [("sequential", "viridis"), ("qualitative", "538")]
)
def test_single_palette_plot(family, name) -> None:
    fig = plot_colormap(family, [name], uniformize=False)
    try:
        assert len(fig.axes) == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("image", ["pyramid", None, "unknown"])
def test_compare_loads_pyramid_scalar_data(image) -> None:
    fig = compare_cmap(
        image=image, cm_list=["viridis"], ncols=1, uniformize=False
    )
    try:
        np.testing.assert_array_equal(
            fig.axes[0].images[0].get_array(), _pyramid()[2]
        )
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "family",
    [
        ScicoSequential,
        ScicoMultiSequential,
        ScicoDiverging,
        ScicoMiscellaneous,
    ],
)
def test_examples_pass_the_periodic_y_coordinate(family, monkeypatch) -> None:
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "figure"

    monkeypatch.setattr("scicomap.scicomap._plot_examples", capture)
    cmap = (
        family(cmap="bukavu") if family is ScicoMultiSequential else family()
    )
    assert cmap.draw_example() == "figure"
    for actual, expected in zip(captured["arr_3d"][1], _periodic_fn()):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "family",
    [
        ScicoSequential,
        ScicoMultiSequential,
        ScicoDiverging,
        ScicoCircular,
        ScicoMiscellaneous,
    ],
)
def test_continuous_examples_render(family) -> None:
    cmap = (
        family(cmap="bukavu") if family is ScicoMultiSequential else family()
    )
    fig = cmap.draw_example(figsize=(8, 6))
    try:
        fig.canvas.draw()
        assert fig.axes
    finally:
        plt.close(fig)


@pytest.mark.parametrize("cmap", ["viridis", ["viridis", "plasma"], []])
def test_cvd_plot_accepts_documented_names_and_hides_all_axes(cmap) -> None:
    fig = colorblind_vision(cmap, figsize=(8, 6))
    try:
        assert all(not ax.axison for ax in fig.axes)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_jch_plot_accepts_documented_matplotlib_name() -> None:
    fig = jch_plot("viridis", figsize=(8, 6))
    try:
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_cvd_palette_name_resolves_all_maps() -> None:
    from matplotlib.colors import Colormap

    maps, _ = _get_color_weak_cmap("viridis", n_images=1)
    assert len(maps) == 5
    assert all(isinstance(cmap, Colormap) for cmap in maps)
    np.testing.assert_array_equal(
        get_ctab(maps[0]), get_ctab(plt.get_cmap("viridis"))
    )
