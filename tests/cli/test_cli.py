"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from typer.testing import CliRunner

from scicomap.cli import app


def test_list_families_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "families" in payload["data"]


def test_check_thermal_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["check", "thermal", "--type", "sequential", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["classification"] in {
        "circular-div",
        "circular-flat",
        "sequential",
        "divergent",
        "asym_div",
        "multiseq",
        "unknown",
    }


def test_long_form_alias_list_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["cmap", "list", "--type", "sequential", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["family"] == "sequential"


def test_apply_grayscale_image(tmp_path: Path) -> None:
    runner = CliRunner()
    image_path = tmp_path / "gray.png"
    out_path = tmp_path / "mapped.png"
    arr = np.linspace(0, 1, 64).reshape(8, 8)
    plt.imsave(image_path, arr, cmap="gray")

    result = runner.invoke(
        app,
        [
            "apply",
            "thermal",
            "--type",
            "sequential",
            "--image",
            str(image_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert out_path.exists()
