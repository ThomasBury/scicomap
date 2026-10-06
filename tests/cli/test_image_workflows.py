"""Regression checks for shared image handling in v1 commands."""

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image
from typer.testing import CliRunner

from scicomap.cli import app
from scicomap.scicomap import SciCoMap


def _image_command(
    workflow: str, image: Path, out: Path, mode: str
) -> list[str]:
    options = ["--image", str(image), "--out", str(out), "--mode", mode]
    if workflow == "apply":
        return ["apply", "thermal", "--type", "sequential", *options, "--json"]
    if workflow == "cmap apply":
        return ["cmap", "apply", "--cmap", "thermal", *options, "--json"]
    options += ["--goal", "apply", "--cmap", "thermal", "--type", "sequential"]
    if workflow == "wizard":
        return [
            "wizard",
            *options,
            "--no-fix",
            "--no-cvd",
            "--no-interactive",
            "--json",
        ]
    return ["report", *options, "--no-fix", "--no-cvd", "--format", "json"]


@pytest.mark.parametrize(
    "workflow", ["apply", "cmap apply", "wizard", "report"]
)
@pytest.mark.parametrize(
    ("image_kind", "mode"),
    [
        ("rgba", "luminance"),
        ("rgba", "first-channel"),
        ("gray", "gray-only"),
        ("rgb", "luminance"),
        ("rgba-tiff", "luminance"),
    ],
)
def test_image_workflows_preserve_colors_alpha_and_input(
    tmp_path: Path, workflow: str, image_kind: str, mode: str
) -> None:
    rgb = np.array(
        [
            [[10, 200, 40], [40, 90, 150], [200, 20, 80]],
            [[60, 110, 30], [100, 60, 180], [250, 30, 200]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array([[0, 64, 128], [192, 255, 32]], dtype=np.uint8)
    source = rgb[..., 0] if image_kind == "gray" else rgb
    if image_kind.startswith("rgba"):
        source = np.dstack((rgb, alpha))
    suffix = ".tiff" if image_kind == "rgba-tiff" else ".png"
    image = tmp_path / f"input{suffix}"
    Image.fromarray(source).save(image)
    original_bytes = image.read_bytes()
    out = tmp_path / ("report" if workflow == "report" else "mapped.png")

    result = CliRunner().invoke(
        app, _image_command(workflow, image, out, mode)
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    artifact = out / "applied.png" if workflow == "report" else out
    mapped = plt.imread(artifact)

    scalar = rgb[..., 0].astype(float)
    if mode == "luminance":
        scalar = rgb.astype(float) @ np.array([0.2126, 0.7152, 0.0722])
    normalized = (scalar - scalar.min()) / (scalar.max() - scalar.min())
    cmap = SciCoMap(ctype="sequential", cmap="thermal").get_mpl_color_map()
    expected = cmap(normalized)
    np.testing.assert_allclose(
        mapped[..., :3], expected[..., :3], atol=1 / 255
    )
    expected_alpha = alpha / 255 if image_kind.startswith("rgba") else 1
    np.testing.assert_allclose(mapped[..., 3], expected_alpha, atol=1e-7)
    assert mapped.shape == (*rgb.shape[:2], 4)
    assert image.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "workflow", ["apply", "cmap apply", "wizard", "report"]
)
@pytest.mark.parametrize("malformed", [True, False])
def test_image_workflows_return_actionable_json_errors(
    tmp_path: Path, workflow: str, malformed: bool
) -> None:
    image = tmp_path / "input.png"
    if malformed:
        image.write_bytes(b"not an image")
    else:
        Image.fromarray(np.zeros((2, 3, 4), dtype=np.uint8)).save(image)
    out = tmp_path / ("report" if workflow == "report" else "mapped.png")
    mode = "luminance" if malformed else "gray-only"

    result = CliRunner().invoke(
        app, _image_command(workflow, image, out, mode)
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    message = (
        "Cannot read image" if malformed else "requires a grayscale image"
    )
    assert message in payload["errors"][0]
    artifact = out / "applied.png" if workflow == "report" else out
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("pixels", "message"),
    [
        (np.empty((0, 2)), "Unsupported image shape"),
        (np.zeros((2, 2, 2)), "Unsupported image shape"),
        (np.zeros((2, 2, 5)), "Unsupported image shape"),
        (np.array([[np.nan]]), "Image pixels must be finite"),
        (np.array([[np.inf]]), "Image pixels must be finite"),
        (np.full((2, 2, 3), 2.0), "must be in the range [0, 1]"),
    ],
)
def test_apply_rejects_invalid_decoded_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pixels: np.ndarray,
    message: str,
) -> None:
    image = tmp_path / "input.png"
    image.touch()
    monkeypatch.setattr(plt, "imread", lambda _path: pixels)
    out = tmp_path / "mapped.png"
    result = CliRunner().invoke(
        app, _image_command("apply", image, out, "luminance")
    )
    assert result.exit_code == 2
    assert message in json.loads(result.stdout)["errors"][0]
    assert not out.exists()
