"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
from matplotlib import pyplot as plt
from typer.testing import CliRunner

from scicomap.cli import app, _resolve_profile_config


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


def test_report_apply_builtin_image_writes_image(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "report-apply-builtin"

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
            "grmhd",
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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "profile": "quick-look",
                "goal": None,
                "has_image": False,
            },
            {
                "goal": "diagnose",
                "fix": False,
                "cvd": False,
                "apply": False,
                "format": "text",
            },
        ),
        (
            {
                "profile": "quick-look",
                "goal": None,
                "has_image": True,
                "apply_output": True,
            },
            {
                "goal": "diagnose",
                "fix": False,
                "cvd": False,
                "apply": True,
                "format": "text",
            },
        ),
        (
            {
                "profile": "publication",
                "goal": None,
                "has_image": False,
            },
            {
                "goal": "improve",
                "fix": True,
                "cvd": True,
                "apply": False,
                "format": "text",
            },
        ),
        (
            {
                "profile": "publication",
                "goal": None,
                "has_image": True,
                "apply_output": True,
            },
            {
                "goal": "improve",
                "fix": True,
                "cvd": True,
                "apply": True,
                "format": "text",
            },
        ),
        (
            {
                "profile": "presentation",
                "goal": None,
                "has_image": False,
            },
            {
                "goal": "improve",
                "fix": True,
                "cvd": True,
                "apply": False,
                "lift": 10.0,
                "format": "text",
            },
        ),
        (
            {
                "profile": "presentation",
                "goal": "diagnose",
                "has_image": False,
            },
            {
                "goal": "diagnose",
                "fix": True,
                "cvd": True,
                "apply": False,
                "format": "text",
            },
        ),
        (
            {
                "profile": "cvd-safe",
                "goal": None,
                "has_image": False,
            },
            {
                "goal": "diagnose",
                "fix": True,
                "cvd": True,
                "apply": False,
                "format": "json",
            },
        ),
        (
            {
                "profile": "cvd-safe",
                "goal": None,
                "has_image": False,
                "cvd": False,
            },
            {
                "goal": "diagnose",
                "fix": True,
                "cvd": True,
                "apply": False,
                "format": "json",
            },
        ),
        (
            {
                "profile": "agent",
                "goal": None,
                "has_image": False,
            },
            {
                "goal": "diagnose",
                "fix": False,
                "cvd": False,
                "apply": False,
                "format": "json",
                "interactive": False,
            },
        ),
        (
            {
                "profile": "agent",
                "goal": "apply",
                "has_image": True,
                "apply_output": True,
            },
            {
                "goal": "apply",
                "fix": False,
                "cvd": False,
                "apply": True,
                "format": "json",
                "interactive": False,
            },
        ),
        (
            {
                "profile": "agent",
                "goal": None,
                "has_image": False,
                "output_format": "text",
            },
            {
                "goal": "diagnose",
                "fix": False,
                "cvd": False,
                "apply": False,
                "format": "json",
            },
        ),
        (
            {
                "profile": "quick-look",
                "goal": "apply",
                "has_image": True,
                "apply_output": True,
            },
            {
                "goal": "apply",
                "fix": False,
                "cvd": False,
                "apply": True,
                "format": "text",
            },
        ),
    ],
)
def test_profile_resolution_matrix(
    kwargs: dict[str, object], expected: dict[str, object]
) -> None:
    config, _warnings = _resolve_profile_config(
        profile=kwargs.get("profile"),
        goal=kwargs.get("goal"),
        has_image=bool(kwargs.get("has_image", False)),
        fix=kwargs.get("fix"),
        cvd=kwargs.get("cvd"),
        apply_output=kwargs.get("apply_output"),
        output_format=kwargs.get("output_format"),
        lift=kwargs.get("lift"),
        bitonic=kwargs.get("bitonic"),
        diffuse=kwargs.get("diffuse"),
        interactive=kwargs.get("interactive"),
    )
    for key, value in expected.items():
        assert config[key] == value


def test_profile_resolution_apply_without_image_fails() -> None:
    with pytest.raises(ValueError, match="requires --image"):
        _resolve_profile_config(
            profile="quick-look",
            goal="apply",
            has_image=False,
            fix=None,
            cvd=None,
            apply_output=True,
            output_format=None,
            lift=None,
            bitonic=None,
            diffuse=None,
            interactive=None,
        )


def test_report_cvd_safe_enforces_cvd(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "cvd-safe"
    result = runner.invoke(
        app,
        [
            "report",
            "--profile",
            "cvd-safe",
            "--cmap",
            "thermal",
            "--type",
            "sequential",
            "--no-cvd",
            "--out",
            str(out_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["actions"]["cvd_generated"] is True
    assert payload["warnings"]


def test_wizard_agent_profile_forces_json_and_noninteractive() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "wizard",
            "--profile",
            "agent",
            "--goal",
            "diagnose",
            "--type",
            "sequential",
            "--cmap",
            "thermal",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["inputs"]["interactive"] is False
    assert payload["inputs"]["format"] == "json"


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
