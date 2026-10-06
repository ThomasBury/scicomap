import numpy as np
import pytest
from matplotlib.colors import ListedColormap, to_rgba_array

from scicomap.cmath import (
    adjust_circular,
    adjust_circular_flat,
    adjust_divergent,
    adjust_sequential,
    classify,
    get_ctab,
    max_chroma,
    symmetrize,
    transform,
    unif_sym_cmap,
    uniformize,
    uniformize_cmap,
)


def test_get_ctab_raises_type_error_for_invalid_input() -> None:
    with pytest.raises(TypeError, match="neither a matplotlib Colormap"):
        get_ctab(cmap=123)


def test_max_chroma_accepts_scalar_inputs() -> None:
    cp = max_chroma(Jp=50.0, hp=0.2)
    assert isinstance(cp, float)
    assert cp >= 0.0


def test_max_chroma_accepts_array_inputs() -> None:
    Jp = np.array([20.0, 40.0, 60.0])
    hp = np.array([0.1, 0.5, 1.0])
    cp = max_chroma(Jp=Jp, hp=hp)

    assert isinstance(cp, np.ndarray)
    assert cp.shape == Jp.shape
    assert np.all(cp >= 0.0)


def test_max_chroma_broadcasts_scalar_hue() -> None:
    Jp = np.array([20.0, 40.0, 60.0])
    cp = max_chroma(Jp=Jp, hp=0.2)
    assert cp.shape == Jp.shape


def test_max_chroma_broadcasts_multidimensional_inputs() -> None:
    Jp = np.array([[20.0], [40.0]])
    hp = np.array([0.1, 0.2, 0.3, 0.4])
    cp = max_chroma(Jp, hp)
    assert cp.shape == (2, 4)
    for row in range(2):
        for col in range(4):
            assert cp[row, col] == pytest.approx(
                max_chroma(Jp[row, 0], hp[col])
            )


def test_max_chroma_raises_value_error_for_out_of_range_without_clip() -> None:
    with pytest.raises(ValueError, match="J' out of range"):
        max_chroma(Jp=150.0, hp=0.2, clip=False)


def test_color_name_lists_use_matplotlib_conversion() -> None:
    colors = ["red", "#00ff0080", (0, 0, 1), "none"]
    np.testing.assert_array_equal(get_ctab(colors), to_rgba_array(colors))
    corrected = unif_sym_cmap(["red", "green", "blue"])
    assert corrected.N == 3
    assert np.isfinite(get_ctab(corrected)).all()


def test_integer_tables_are_converted_without_truncation_or_mutation() -> None:
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    before = rgb.copy()
    converted = transform(rgb)
    assert converted.dtype.kind == "f"
    np.testing.assert_allclose(converted, transform(rgb.astype(float)))
    np.testing.assert_allclose(
        transform(converted, inverse=True), rgb, atol=1e-12
    )
    np.testing.assert_array_equal(rgb, before)
    assert get_ctab(rgb.tolist()).shape == rgb.shape
    assert get_ctab(rgb.tolist()).dtype.kind == "f"

    jab = np.array([[21, 10, 4], [35, 20, 8], [50, 5, 2]])
    for adjust in (uniformize, symmetrize, adjust_circular_flat):
        result = adjust(jab)
        assert result.dtype.kind == "f"
        np.testing.assert_allclose(result, adjust(jab.astype(float)))
    assert uniformize(jab)[1, 0] == 35.5


@pytest.mark.parametrize(
    "table",
    [
        [],
        [1, 0, 0],
        [[0, 0]],
        [[0, 0, 0, 1, 1]],
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
        [[0, 0, 0, -0.1]],
        [[0, 0, 0, 1.1]],
    ],
)
def test_invalid_color_tables_are_rejected(table) -> None:
    with pytest.raises(ValueError):
        get_ctab(table)
    with pytest.raises(ValueError):
        transform(np.asarray(table))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rgb_input_must_be_in_unit_range(value) -> None:
    with pytest.raises(ValueError, match="range"):
        get_ctab([[value, 0, 0]])
    with pytest.raises(ValueError, match="range"):
        transform(np.array([[value, 0, 0]]))
    # Perceptual coordinates and inverse output are allowed outside [0, 1].
    assert np.isfinite(transform(np.array([[50, 10, 10]]), inverse=True)).all()


@pytest.mark.parametrize("n", [5, 6, 7, 32, 33, 256, 257])
@pytest.mark.parametrize("valley", [False, True])
@pytest.mark.parametrize("adjust", [adjust_divergent, adjust_circular])
def test_diverging_branches_preserve_samples_endpoints_center_and_alpha(
    n, valley, adjust
) -> None:
    left = (n + 1) // 2
    right = n // 2
    lightness = np.r_[
        np.linspace(20, 80, left),
        np.linspace(80, 20, right + n % 2)[n % 2 :],
    ]
    if valley:
        lightness = 100 - lightness
    jab = np.column_stack(
        (
            lightness,
            np.linspace(-10, 10, n),
            np.full(n, 5),
            np.linspace(0, 1, n),
        )
    )
    before = jab.copy()
    result = adjust(jab)
    assert result.shape == jab.shape
    np.testing.assert_array_equal(result[[0, -1]], jab[[0, -1]])
    np.testing.assert_array_equal(result[:, 3], jab[:, 3])
    if n % 2:
        np.testing.assert_array_equal(result[n // 2], jab[n // 2])
    np.testing.assert_array_equal(jab, before)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("valley", [False, True])
def test_asymmetric_diverging_map_preserves_both_endpoints(
    reverse, valley
) -> None:
    lightness = np.array([20, 50, 80, 70, 60, 50, 40])
    if valley:
        lightness = 100 - lightness
    if reverse:
        lightness = lightness[::-1]
    jab = np.column_stack((lightness, np.zeros((7, 2))))
    adjusted = adjust_divergent(jab, symmetric=False)
    assert adjusted.shape == jab.shape
    center = 4 if reverse else 2
    np.testing.assert_array_equal(
        adjusted[[0, center, -1]], jab[[0, center, -1]]
    )


@pytest.mark.parametrize("adjust", [adjust_divergent, adjust_circular])
def test_circular_adjustment_keeps_odd_count_and_closes_rgb_path(
    adjust,
) -> None:
    jab = np.array(
        [
            [20, 10, 5, 0],
            [50, 0, 10, 0.2],
            [80, -10, 5, 0.5],
            [50, 0, -10, 0.8],
            [20, 10, -5, 1],
        ]
    )
    kwargs = {"circular": True} if adjust is adjust_divergent else {}
    result = adjust(jab, **kwargs)
    assert result.shape == jab.shape
    np.testing.assert_array_equal(result[:, 3], jab[:, 3])
    if kwargs:
        np.testing.assert_allclose(result[0, :3], result[-1, :3])


@pytest.mark.parametrize(
    "adjust", [adjust_sequential, adjust_divergent, adjust_circular]
)
def test_zero_rounding_matches_no_rounding(adjust) -> None:
    lightness = (
        [13, 35, 57] if adjust is adjust_sequential else [13, 35, 57, 35, 13]
    )
    jab = np.column_stack((lightness, np.zeros((len(lightness), 2))))
    np.testing.assert_array_equal(
        adjust(jab, lightness_rounding=0), adjust(jab)
    )
    assert adjust(jab, lightness_rounding=10)[0, 0] == 20


def test_zero_rounding_and_alpha_are_preserved_through_combined_transform() -> (
    None
):
    table = to_rgba_array(["navy", "gray", "white", "gray", "maroon"])
    table[:, 3] = [0, 0.2, 0.5, 0.8, 1]
    cmap = ListedColormap(table)
    assert classify(transform(table)) == "divergent"
    for operation in (uniformize_cmap, unif_sym_cmap):
        corrected = operation(cmap, lightness_rounding=0)
        default = operation(cmap)
        assert corrected.N == cmap.N
        np.testing.assert_array_equal(get_ctab(corrected)[:, 3], table[:, 3])
        np.testing.assert_array_equal(get_ctab(corrected), get_ctab(default))


def test_unknown_map_warns_and_returns_a_colormap() -> None:
    colors = ["black", "white", "black", "white", "gray", "white", "red"]
    cmap = ListedColormap(colors)
    assert classify(transform(get_ctab(cmap))) == "unknown"
    with pytest.warns(UserWarning, match="Not uniformized"):
        result = uniformize_cmap(cmap)
    np.testing.assert_array_equal(get_ctab(result), get_ctab(cmap))
    with pytest.warns(UserWarning, match="Not uniformized"):
        result = unif_sym_cmap(cmap)
    assert isinstance(result, ListedColormap)


def test_single_color_transform_is_finite() -> None:
    result = unif_sym_cmap(ListedColormap(["red"]))
    assert result.N == 1
    assert np.isfinite(get_ctab(result)).all()


def test_short_monotonic_map_with_close_endpoints_is_sequential() -> None:
    jab = np.array([[20, 0, 0], [21, 0, 0], [22, 0, 0]])
    assert classify(jab) == "sequential"
    cmap = ListedColormap(transform(jab, inverse=True))
    corrected = uniformize_cmap(cmap)
    assert corrected.N == 3
