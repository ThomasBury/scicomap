"""M3 regressions for family diagnostics, stages, map selection, and failures."""

import json
from pathlib import Path

import numpy as np
import pytest
from click import unstyle
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from typer.testing import CliRunner

import scicomap.cli as cli
from scicomap._diagnostics import diagnose_cmap
from scicomap.cmath import transform
from scicomap.scicomap import SciCoMap


@pytest.mark.parametrize(
    ("family", "lightness", "status"),
    [
        ("sequential", [30, 40, 50, 60, 70], "good"),
        ("sequential", [70, 60, 50, 40, 30], "good"),
        ("sequential", [30, 50, 40, 60, 70], "fix-recommended"),
        ("sequential", [50] * 5, "caution"),
        ("diverging", [30, 50, 70, 50, 30], "good"),
        ("diverging", [70, 50, 30, 50, 70], "good"),
        ("diverging", [30, 50, 70, 69, 50, 30], "good"),
        ("diverging", [30, 70, 60, 50, 30], "fix-recommended"),
        ("diverging", [30, 40, 50, 60, 70], "fix-recommended"),
        ("multi-sequential", [30, 40, 50, 30, 40, 50], "good"),
        ("multi-sequential", [30, 50, 40, 30, 40, 50], "fix-recommended"),
        ("qualitative", [30, 70, 40, 60, 50], "caution"),
        ("miscellaneous", [30, 70, 40, 60, 50], "caution"),
    ],
)
def test_family_lightness_rules(family, lightness, status) -> None:
    perceptual = np.column_stack((lightness, np.zeros((len(lightness), 2))))
    cmap = ListedColormap(transform(perceptual, inverse=True))
    diagnostics = diagnose_cmap(cmap, family)
    assert diagnostics["status"] == status
    assert diagnostics["heuristic"] is True
    if family == "diverging" and status == "good":
        assert diagnostics["monotonic_lightness"] is False
        assert diagnostics["branches_monotonic"] is True


@pytest.mark.parametrize("oscillations", [0, 1, 3])
@pytest.mark.parametrize("closed", [False, True])
def test_circular_diagnostics(oscillations, closed) -> None:
    angle = np.linspace(0, 2 * np.pi if closed else np.pi, 64)
    colors = np.column_stack(
        (
            50 + 10 * np.sin(oscillations * angle),
            10 * np.cos(angle),
            10 * np.sin(angle),
        )
    )
    diagnostics = diagnose_cmap(
        ListedColormap(transform(colors, inverse=True)), "circular"
    )
    assert diagnostics["seam_closed"] is closed
    assert diagnostics["status"] == (
        "good" if closed and oscillations <= 1 else "caution"
    )


@pytest.mark.parametrize(
    ("workflow", "builtin"),
    [
        ("wizard", False),
        ("report", False),
        ("report", True),
    ],
)
@pytest.mark.parametrize("fix", [False, True])
def test_workflow_uses_one_selected_map(
    tmp_path, monkeypatch, workflow, fix, builtin
) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.array([[0, 50], [100, 255]], dtype=np.uint8)).save(
        image
    )
    expected = SciCoMap(cmap="thermal")
    if fix:
        expected.unif_sym_cmap(lightness_rounding=20)
    expected_map = expected.get_mpl_color_map()
    original_map = SciCoMap(cmap="thermal").get_mpl_color_map()
    assessed = []
    simulated = []
    applied = []
    fixes = []
    real_fix = SciCoMap.unif_sym_cmap

    def transform_once(self, **kwargs):
        fixes.append(self)
        return real_fix(self, **kwargs)

    def assess(self, **kwargs):
        assessed.append(self.get_mpl_color_map())
        return plt.figure()

    def simulate(**kwargs):
        assert kwargs["uniformize"] is False
        assert kwargs["symmetrize"] is False
        simulated.append(kwargs["cmap_list"][0])
        return plt.figure()

    def apply_builtin(**kwargs):
        assert kwargs["uniformize"] is False
        assert kwargs["symmetrize"] is False
        applied.append(kwargs["cm_list"][0])
        return plt.figure()

    monkeypatch.setattr(SciCoMap, "unif_sym_cmap", transform_once)
    monkeypatch.setattr(SciCoMap, "assess_cmap", assess)
    monkeypatch.setattr(cli, "plot_colorblind_vision", simulate)
    monkeypatch.setattr(cli, "compare_cmap", apply_builtin)
    out = tmp_path / ("report" if workflow == "report" else "applied.png")
    args = [
        workflow,
        "--cmap",
        "thermal",
        "--goal",
        "diagnose",
        "--image",
        "scan" if builtin else str(image),
        "--out",
        str(out),
        "--apply",
        "--cvd",
        "--fix" if fix else "--no-fix",
        "--lift",
        "20",
    ]
    args += (
        ["--format", "json"]
        if workflow == "report"
        else ["--json", "--no-interactive"]
    )
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert len(fixes) == int(fix)
    assert data["diagnostics"] == diagnose_cmap(expected_map, "sequential")
    assert data["original_diagnostics"] == diagnose_cmap(
        original_map, "sequential"
    )
    assert data["map_used"] == ("transformed" if fix else "original")
    samples = np.linspace(0, 1, 64)
    for used in simulated + applied:
        np.testing.assert_allclose(used(samples), expected_map(samples))
    if assessed:
        np.testing.assert_allclose(
            assessed[-1](samples), expected_map(samples)
        )
    if workflow == "report":
        np.testing.assert_allclose(assessed[0](samples), original_map(samples))
    if not builtin:
        artifact = out / "applied.png" if workflow == "report" else out
        np.testing.assert_allclose(
            plt.imread(artifact),
            cli._remap_image(image, expected_map, "luminance"),
            atol=1 / 255,
        )


@pytest.mark.parametrize("workflow", ["wizard", "report"])
@pytest.mark.parametrize("goal", ["diagnose", "improve", "apply"])
def test_disabled_stages_do_not_run(
    tmp_path, monkeypatch, workflow, goal
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled stage ran")

    monkeypatch.setattr(SciCoMap, "unif_sym_cmap", forbidden)
    monkeypatch.setattr(cli, "plot_colorblind_vision", forbidden)
    monkeypatch.setattr(cli, "_remap_image", forbidden)
    monkeypatch.setattr(SciCoMap, "assess_cmap", lambda self: plt.figure())
    out = tmp_path / ("report" if workflow == "report" else "assessment.png")
    args = [
        workflow,
        "--goal",
        goal,
        "--cmap",
        "thermal",
        "--out",
        str(out),
        "--no-fix",
        "--no-cvd",
        "--no-apply",
    ]
    args += (
        ["--format", "json"]
        if workflow == "report"
        else ["--json", "--no-interactive"]
    )
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert not any(data["actions"].values())
    assert all(item["kind"] == "assessment" for item in data["artifacts"])


def test_interactive_apply_collects_missing_image(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(image)
    out = tmp_path / "applied.png"
    result = CliRunner().invoke(
        cli.app,
        [
            "wizard",
            "--profile",
            "quick-look",
            "--goal",
            "apply",
            "--out",
            str(out),
        ],
        input=f"\n\n{image}\n",
    )
    assert result.exit_code == 0, result.output
    assert "Image path" in result.stdout
    assert out.is_file()


@pytest.mark.parametrize("workflow", ["wizard", "report"])
def test_agent_applies_image_by_default(tmp_path, workflow) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(image)
    out = tmp_path / ("report" if workflow == "report" else "applied.png")
    result = CliRunner().invoke(
        cli.app,
        [
            workflow,
            "--profile",
            "agent",
            "--cmap",
            "thermal",
            "--image",
            str(image),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert data["goal"] == "apply"
    assert data["actions"]["image_applied"] is True
    assert (out / "applied.png" if workflow == "report" else out).is_file()


@pytest.mark.parametrize(
    "args",
    [
        ["cvd", "--n-colors", "0", "--json"],
        ["cvd", "--n-colors", "-2", "--json"],
        ["cvd", "--n-colors", "bad", "--json"],
        ["compare", "thermal", "viridis", "--ncols", "0", "--json"],
        ["cmap", "colorblind", "--n-colors", "0", "--json"],
        [
            "apply",
            "--image",
            "/missing-image.png",
            "--out",
            "out.png",
            "--json",
        ],
        ["apply", "--json"],
        ["fix", "--lift", "-1", "--json"],
        ["fix", "--lift", "nan", "--json"],
        ["wizard", "--lift", "nan", "--format", "json"],
        ["report", "--lift", "nan", "--format", "json"],
        ["report", "--lift", "bad", "--format", "text", "--format=json"],
        ["wizard", "--goal", "bad", "--format", "json"],
        [
            "wizard",
            "--profile",
            "agent",
            "--goal",
            "apply",
            "--cmap",
            "thermal",
        ],
        ["report", "--profile", "bad", "--format=json"],
        ["report", "--profile", "agent", "--goal", "apply"],
        ["report", "--lift", "bad", "--format", "json"],
    ],
)
def test_json_validation_failures(args) -> None:
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["errors"][0]


@pytest.mark.parametrize(
    "args",
    [
        ["apply"],
        ["check", "--jsno"],
        ["--not-an-option"],
    ],
)
def test_text_usage_errors_keep_help_guidance(args) -> None:
    result = CliRunner().invoke(cli.app, args, env={"FORCE_COLOR": "1"})
    assert result.exit_code == 2
    output = unstyle(result.output)
    assert "Usage:" in output
    assert "--help" in output


@pytest.mark.parametrize("workflow", ["wizard", "report"])
def test_invalid_image_creates_no_workflow_artifacts(
    tmp_path, workflow
) -> None:
    image = tmp_path / "bad.png"
    image.write_bytes(b"not an image")
    out = tmp_path / ("report" if workflow == "report" else "applied.png")
    args = [
        workflow,
        "--cmap",
        "thermal",
        "--goal",
        "apply",
        "--image",
        str(image),
        "--out",
        str(out),
    ]
    args += (
        ["--format", "json"]
        if workflow == "report"
        else ["--json", "--no-interactive"]
    )
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert "Cannot read image" in json.loads(result.stdout)["errors"][0]
    assert list(tmp_path.iterdir()) == [image]


def test_json_runtime_failure(tmp_path, monkeypatch) -> None:
    def cannot_save(*args, **kwargs):
        raise OSError("Destination is not writable")

    monkeypatch.setattr(cli, "_save_figure", cannot_save)
    monkeypatch.setattr(SciCoMap, "assess_cmap", lambda self: None)
    result = CliRunner().invoke(
        cli.app,
        [
            "preview",
            "thermal",
            "--out",
            str(tmp_path / "out.png"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "not writable" in json.loads(result.stdout)["errors"][0]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("workflow", ["wizard", "report"])
def test_all_output_paths_checked_before_writing(tmp_path, workflow) -> None:
    if workflow == "report":
        out = tmp_path / "report"
        conflict = out / "cvd.png"
        args = ["--format", "json"]
    else:
        out = tmp_path / "out.png"
        conflict = tmp_path / "out-cvd.png"
        args = ["--json", "--no-interactive"]
    conflict.mkdir(parents=True)
    result = CliRunner().invoke(
        cli.app,
        [
            workflow,
            "--cmap",
            "thermal",
            "--out",
            str(out),
            *args,
        ],
    )
    assert result.exit_code == 2
    assert "Output must be a file" in json.loads(result.stdout)["errors"][0]
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_cvd_command_preserves_original_map(tmp_path, monkeypatch) -> None:
    original = SciCoMap(cmap="thermal").get_mpl_color_map()

    def simulate(**kwargs):
        assert kwargs["uniformize"] is False
        assert kwargs["symmetrize"] is False
        samples = np.linspace(0, 1, 64)
        np.testing.assert_allclose(
            kwargs["cmap_list"][0](samples), original(samples)
        )
        return plt.figure()

    monkeypatch.setattr(cli, "plot_colorblind_vision", simulate)
    result = CliRunner().invoke(
        cli.app,
        [
            "cvd",
            "thermal",
            "--out",
            str(tmp_path / "cvd.png"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
