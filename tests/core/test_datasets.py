import pytest

from scicomap.datasets import load_pic


def test_load_pic_raises_type_error_for_non_string_name() -> None:
    with pytest.raises(TypeError, match="name should be a string"):
        load_pic(name=123)


def test_load_pic_warns_and_defaults_for_unknown_name() -> None:
    with pytest.warns(UserWarning, match="Using a default image"):
        image = load_pic(name="unknown")

    assert image.ndim == 2
