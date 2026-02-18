import numpy as np
import pytest

from scicomap.cmath import get_ctab, max_chroma


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


def test_max_chroma_raises_value_error_for_out_of_range_without_clip() -> None:
    with pytest.raises(ValueError, match="J' out of range"):
        max_chroma(Jp=150.0, hp=0.2, clip=False)
