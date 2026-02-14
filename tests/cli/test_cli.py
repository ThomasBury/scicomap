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
    assert payload["data"]["status"] in {
        "good",
        "caution",
        "fix-recommended",
    }
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


def test_doctor_json(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["doctor", "--out-dir", str(tmp_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["status"] == "healthy"


def test_wizard_noninteractive_diagnose_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "wizard",
            "--goal",
            "diagnose",
            "--type",
            "sequential",
            "--cmap",
            "thermal",
            "--no-interactive",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["goal"] == "diagnose"
    assert "diagnostics" in payload["data"]


def test_report_diagnose_writes_bundle(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "report-diagnose"
    result = runner.invoke(
        app,
        [
            "report",
            "--cmap",
            "thermal",
            "--type",
            "sequential",
            "--out",
            str(out_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert (out_dir / "report.json").exists()
    assert (out_dir / "summary.txt").exists()
    assert (out_dir / "assess.png").exists()


def test_report_apply_writes_image(tmp_path: Path) -> None:
    runner = CliRunner()
    image_path = tmp_path / "input.png"
    out_dir = tmp_path / "report-apply"
    arr = np.linspace(0, 1, 64).reshape(8, 8)
    plt.imsave(image_path, arr, cmap="gray")

    result = runner.invoke(
        app,
        [
            "report",
            "--cmap",
            "thermal",
            "--type",
            "sequential",
            "--goal",
            "apply",
            "--image",
            str(image_path),
            "--out",
            str(out_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert (out_dir / "applied.png").exists()


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
