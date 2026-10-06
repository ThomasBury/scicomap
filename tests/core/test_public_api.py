"""Public Python API contracts for M5."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure

import scicomap as sc
from scicomap import cli


FAMILIES = [
    (sc.ScicoSequential, "sequential", "thermal"),
    (sc.ScicoMultiSequential, "multi-sequential", "bukavu"),
    (sc.ScicoDiverging, "diverging", "wildfire"),
    (sc.ScicoCircular, "circular", "colorwheel"),
    (sc.ScicoMiscellaneous, "miscellaneous", "turbo"),
    (sc.ScicoQualitative, "qualitative", "glasbey_dark"),
]


def test_exports_and_single_discovery_api() -> None:
    namespace = {}
    exec("from scicomap import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(sc.__all__)
    assert all(hasattr(sc, name) for name in sc.__all__)
    for name in (
        "np",
        "plt",
        "cc",
        "Colormap",
        "get_available_ctype",
        "get_ctab",
    ):
        assert not hasattr(sc, name)
    for name in ("get_ctype", "get_color_map_dic", "get_color_map_names"):
        assert not hasattr(sc.SciCoMap, name)
    assert not hasattr(sc.scicomap, "get_available_ctype")
    catalog = sc.get_cmap_dict()
    assert set(catalog) == {family for _, family, _ in FAMILIES}
    assert all(
        maps and all(isinstance(cmap, Colormap) for cmap in maps.values())
        for maps in catalog.values()
    )
    assert cli.diagnose_cmap is sc.diagnose_cmap


@pytest.mark.parametrize("cls, family, default", FAMILIES)
def test_family_defaults_resolve_immediately(cls, family, default) -> None:
    chart = cls()
    assert chart.ctype == family
    assert chart.cname == default
    assert isinstance(chart.cmap, Colormap)
    assert chart.get_mpl_color_map() is chart.cmap
    assert sc.SciCoMap(ctype=family).cname == default
    assert cls.draw_example is sc.SciCoMap.draw_example
    assert repr(chart).startswith(cls.__name__)
    assert not hasattr(chart, "uniformized")


@pytest.mark.parametrize(
    "colors",
    [
        ["navy", "white"],
        [[0, 0, 0], [1, 1, 1]],
        [[0, 0, 0, 0.2], [1, 1, 1, 0.8]],
    ],
)
def test_color_lists_are_resolved_without_mutating_input(colors) -> None:
    chart = sc.SciCoMap(cmap=colors)
    assert isinstance(chart.cmap, ListedColormap)
    np.testing.assert_allclose(
        sc.cmath.get_ctab(chart.cmap)[:, :3], sc.cmath.get_ctab(colors)[:, :3]
    )
    assert chart.cmap.N == len(colors)
    supplied = plt.get_cmap("viridis")
    assert sc.SciCoMap(cmap=supplied).cmap is supplied


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"ctype": "unknown"}, ValueError),
        ({"ctype": []}, TypeError),
        ({"cmap": "unknown"}, ValueError),
        ({"ctype": "diverging", "cmap": "thermal"}, ValueError),
        ({"cmap": 42}, TypeError),
        ({"cmap": np.zeros((2, 3))}, TypeError),
        ({"cmap": []}, ValueError),
        ({"cmap": [[np.nan, 0, 0]]}, ValueError),
        ({"cmap": [[2, 0, 0]]}, ValueError),
        ({"cmap": ["invalid-color"]}, ValueError),
        ({"cmap": ListedColormap([[np.nan, 0, 0, 1]])}, ValueError),
    ],
)
def test_invalid_construction_fails_immediately(kwargs, error) -> None:
    with pytest.raises(error):
        sc.SciCoMap(**kwargs)


@pytest.mark.parametrize(
    "operation", ["uniformize_cmap", "symmetrize_cmap", "unif_sym_cmap"]
)
def test_transformations_return_colormaps_and_replace_object_state(
    operation,
) -> None:
    table = sc.cmath.get_ctab(plt.get_cmap("viridis"))
    table[:, 3] = np.linspace(0, 1, len(table))
    original = ListedColormap(table)
    chart = sc.SciCoMap(cmap=original)
    function = getattr(sc, operation)
    method = getattr(chart, operation)
    result = method()
    assert isinstance(result, Colormap)
    assert result is chart.cmap
    np.testing.assert_allclose(
        sc.cmath.get_ctab(result), sc.cmath.get_ctab(function(original))
    )
    np.testing.assert_array_equal(sc.cmath.get_ctab(original), table)
    np.testing.assert_array_equal(sc.cmath.get_ctab(result)[:, 3], table[:, 3])
    assert result.N == original.N
    assert "uniformized" not in inspect.signature(function).parameters
    assert "lift" not in inspect.signature(function).parameters
    assert "uniformized" not in inspect.signature(method).parameters
    assert "lift" not in inspect.signature(method).parameters


def test_repeated_transformation_honors_new_rounding_parameter() -> None:
    chart = sc.ScicoSequential("viridis")
    previous = chart.uniformize_cmap()
    expected = sc.uniformize_cmap(previous, lightness_rounding=10)
    result = chart.uniformize_cmap(lightness_rounding=10)
    np.testing.assert_allclose(
        sc.cmath.get_ctab(result), sc.cmath.get_ctab(expected)
    )
    assert not np.allclose(
        sc.cmath.get_ctab(previous), sc.cmath.get_ctab(result)
    )


@pytest.mark.parametrize(
    "step, error",
    [
        (-1, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        ("5", TypeError),
        ([], TypeError),
    ],
)
def test_invalid_rounding_fails_without_changing_state(step, error) -> None:
    chart = sc.ScicoSequential()
    original = chart.cmap
    with pytest.raises(error):
        chart.unif_sym_cmap(lightness_rounding=step)
    assert chart.cmap is original


@pytest.mark.parametrize(
    "plot",
    [
        lambda: sc.ScicoSequential().assess_cmap(figsize=(8, 6)),
        lambda: sc.ScicoSequential().colorblind(figsize=(8, 6)),
        lambda: sc.ScicoMultiSequential().illustrate_palettes(figsize=(8, 6)),
        lambda: sc.ScicoQualitative().draw_example(
            figsize=(8, 6), cblind=False
        ),
        lambda: sc.SciCoMap().draw_example(figsize=(8, 6), cblind=False),
        lambda: sc.plot_colormap("sequential", ["viridis"], uniformize=False),
        lambda: sc.plot_colorblind_vision(
            "sequential", ["viridis"], uniformize=False
        ),
        lambda: sc.compare_cmap(cm_list=["viridis"], uniformize=False),
        lambda: sc.jch_plot("viridis", figsize=(8, 6)),
        lambda: sc.cblind.colorblind_vision("viridis", figsize=(8, 6)),
    ],
)
def test_plotting_returns_figures_without_display_or_global_configuration_changes(
    plot, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Plotting must not implicitly display figures")

    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(Figure, "show", forbidden)
    settings = dict(plt.rcParams)
    try:
        figure = plot()
        assert isinstance(figure, Figure)
        assert figure.axes
        assert dict(plt.rcParams) == settings
    finally:
        plt.close("all")


def test_colorblind_plots_use_current_map_without_transforming(
    monkeypatch,
) -> None:
    chart = sc.ScicoSequential("viridis")
    current = chart.unif_sym_cmap()
    returned = Figure()

    def capture(cmap, **kwargs):
        assert cmap == [current]
        return returned

    monkeypatch.setattr(sc.scicomap, "colorblind_vision", capture)
    assert chart.colorblind() is returned


@pytest.mark.parametrize(
    "module", [sc.scicomap, sc.cmath, sc.cblind, sc.datasets]
)
def test_submodule_wildcard_exports_exclude_dependencies(module) -> None:
    namespace = {}
    exec(f"from {module.__name__} import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(module.__all__)
    assert not {
        "np",
        "plt",
        "matplotlib",
        "Colormap",
        "ListedColormap",
        "Real",
        "warnings",
    } & set(namespace)


@pytest.mark.parametrize("colors", [["red"], ["red", "blue"]])
def test_short_qualitative_palettes_render(colors) -> None:
    chart = sc.ScicoQualitative(cmap=colors)
    try:
        example = chart.draw_example(cblind=False, figsize=(6, 4))
        palette = sc.plot_colormap(
            "qualitative", [chart.cmap], uniformize=False
        )
        for figure in (example, palette):
            assert isinstance(figure, Figure)
            figure.canvas.draw()
        assert len(palette.axes[0].patches) == len(colors)
    finally:
        plt.close("all")
