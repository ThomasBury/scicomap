"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
from matplotlib import pyplot as plt
from typer.testing import CliRunner

from scicomap.cblind import colorblind_vision
from scicomap.cli import app


def test_list_families_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert isinstance(payload["data"]["families"], list)
    assert set(payload["data"]["families"]) == set(payload["data"]["counts"])


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


def test_list_names_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list", "sequential", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["family"] == "sequential"


def test_doctor_json(tmp_path: Path) -> None:
    runner = CliRunner()
    existing_file = tmp_path / ".scicomap_write_test"
    existing_file.write_bytes(b"user data")
    result = runner.invoke(
        app,
        ["doctor", "--out-dir", str(tmp_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["status"] == "healthy"
    assert existing_file.read_bytes() == b"user data"
    assert list(tmp_path.iterdir()) == [existing_file]


def test_wizard_noninteractive_diagnose_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "wizard",
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
    assert payload["data"]["map_used"] == "original"
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
            "--json",
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
            "--apply",
            "--image",
            str(image_path),
            "--out",
            str(out_dir),
            "--json",
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
            "--apply",
            "--image",
            "grmhd",
            "--out",
            str(out_dir),
            "--json",
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


def test_colorblind_vision_default_figsize_is_readable() -> None:
    fig = colorblind_vision(cmap=plt.get_cmap("viridis"), figsize=None)
    try:
        assert fig.get_size_inches()[1] >= 5.5
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "args",
    [
        ["preview"],
        ["compare", "hawaii", "viridis"],
        ["fix"],
        ["cvd"],
        ["report"],
        ["wizard", "--fix"],
        ["wizard", "--cvd"],
    ],
)
def test_machine_rendering_requires_destination(
    args, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Opened a window"))
    result = CliRunner().invoke(app, [*args, "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert "requires --out" in payload["errors"][0]
    assert list(tmp_path.iterdir()) == []


def test_machine_inspection_never_prompts_or_transforms(
    tmp_path, monkeypatch
) -> None:
    import typer
    from scicomap import SciCoMap

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection prompted, rendered, or transformed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(typer, "prompt", forbidden)
    monkeypatch.setattr(typer, "confirm", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(SciCoMap, "unif_sym_cmap", forbidden)
    monkeypatch.setattr(SciCoMap, "assess_cmap", forbidden)
    result = CliRunner().invoke(
        app, ["wizard", "--lightness-rounding", "10", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert data["map_used"] == "original"
    assert data["artifacts"] == []
    assert not any(data["actions"].values())
    assert list(tmp_path.iterdir()) == []


def test_canonical_command_help() -> None:
    from typer.main import get_command

    command = get_command(app)
    assert set(command.commands) == {
        "list",
        "check",
        "preview",
        "compare",
        "fix",
        "cvd",
        "apply",
        "doctor",
        "wizard",
        "report",
        "version",
    }
    runner = CliRunner()
    for name in [None, *command.commands]:
        result = runner.invoke(app, ([name] if name else []) + ["--help"])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "list",
        "check",
        "preview",
        "compare",
        "fix",
        "cvd",
        "apply",
        "doctor",
        "wizard",
        "report",
        "version",
    ],
)
def test_human_and_json_run_same_operations(
    command, tmp_path, monkeypatch
) -> None:
    import scicomap.cli as cli
    from scicomap import SciCoMap, get_cmap_dict

    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Opened a window"))
    monkeypatch.setattr(SciCoMap, "assess_cmap", lambda self: plt.figure())
    monkeypatch.setattr(
        cli, "plot_colorblind_vision", lambda **kwargs: plt.figure()
    )

    def compare(**kwargs):
        assert kwargs["uniformize"] is False
        assert kwargs["symmetrize"] is False
        return plt.figure()

    monkeypatch.setattr(cli, "compare_cmap", compare)
    image = tmp_path / "input.png"
    plt.imsave(image, np.arange(4).reshape(2, 2), cmap="gray")
    args = [command]
    if command == "compare":
        args += ["thermal", "viridis"]
    if command == "apply":
        args += ["--image", str(image)]
    if command in {"preview", "compare", "fix", "cvd", "apply", "wizard"}:
        args += ["--out", str(tmp_path / "out.png")]
    if command in {"wizard", "report"}:
        args += [
            "--cmap",
            "thermal",
            "--fix",
            "--cvd",
            "--apply",
            "--image",
            str(image),
        ]
    if command == "wizard":
        args += ["--no-interactive"]
    if command == "report":
        args += ["--out", str(tmp_path / "report")]
    if command == "doctor":
        args += ["--out-dir", str(tmp_path)]
    emitted = []
    original_emit = cli._emit

    def capture(payload, as_json):
        emitted.append(payload)
        original_emit(payload, as_json)

    monkeypatch.setattr(cli, "_emit", capture)
    runner = CliRunner()
    human = runner.invoke(app, args)
    machine = runner.invoke(app, [*args, "--json"])
    assert human.exit_code == machine.exit_code == 0, (
        human.output,
        machine.output,
    )
    assert emitted[0] == emitted[1] == json.loads(machine.stdout)
    assert set(emitted[1]) == {
        "ok",
        "command",
        "inputs",
        "data",
        "warnings",
        "errors",
    }
    for artifact in emitted[1]["data"].get("artifacts", []):
        assert set(artifact) == {"kind", "path", "map"}
        assert Path(artifact["path"]).is_absolute()
        assert Path(artifact["path"]).is_file()
    if command == "report":
        assert (
            json.loads((tmp_path / "report/report.json").read_text())
            == emitted[1]
        )


def test_json_rendering_uses_headless_backend(tmp_path, monkeypatch) -> None:
    from scicomap import SciCoMap

    original_backend = plt.get_backend()
    try:
        plt.switch_backend("svg")
        monkeypatch.setattr(SciCoMap, "assess_cmap", lambda self: plt.figure())
        result = CliRunner().invoke(
            app, ["preview", "--out", str(tmp_path / "preview.png"), "--json"]
        )
        assert result.exit_code == 0, result.output
        assert plt.get_backend().lower() == "agg"
        assert (tmp_path / "preview.png").is_file()
    finally:
        plt.switch_backend(original_backend)
